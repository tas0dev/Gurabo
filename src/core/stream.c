#include "core.h"

static void remember(gurabo_stream *stream, gb_result result, gurabo_error_phase phase,
                     uint64_t submission)
{
    if (result.status == GURABO_OK) {
        return;
    }
    gurabo_context *context = stream->context;
    gurabo_error_info error = gb_make_error(result, context->backend->id, phase, submission);
    if (stream->error.status == GURABO_OK && phase != GURABO_ERROR_CALL) {
        stream->error = error;
    }
    gb_context_error(context, &error);
}

static void retire_job(gurabo_stream *stream, int safe)
{
    gb_job *job = stream->head;
    stream->head = job->next;
    if (!stream->head) {
        stream->tail = NULL;
    }
    if (safe) {
        gb_result result = stream->context->backend->marker_destroy(stream->native, job->marker);
        remember(stream, result, GURABO_ERROR_CLEANUP, 0);
    }
    for (uint32_t i = 0; i < job->count; ++i) {
        if (!safe) {
            job->buffers[i]->quarantined = 1;
        }
        gb_result result = gb_buffer_unref(job->buffers[i]);
        remember(stream, result, GURABO_ERROR_CLEANUP, 0);
    }
    free(job);
}

void gb_stream_reap(gurabo_stream *stream)
{
    const gb_backend *backend = stream->context->backend;
    while (stream->head && stream->error.status == GURABO_OK && !stream->context->lost) {
        gb_result result = backend->marker_query(stream->native, stream->head->marker);
        if (result.status == GURABO_NOT_READY) {
            break;
        }
        if (result.status != GURABO_OK) {
            remember(stream, result, GURABO_ERROR_EXECUTION, 0);
            break;
        }
        retire_job(stream, 1);
    }
}

gurabo_status gb_stream_ready(gurabo_stream *stream)
{
    gb_stream_reap(stream);
    gurabo_context *context = stream->context;
    if (context->lost) {
        return gb_fail(GURABO_DEVICE_LOST, context->backend->id);
    }
    return stream->error.status == GURABO_OK ? GURABO_OK : gb_report(&stream->error);
}

gurabo_status gb_job_begin(gurabo_stream *stream, gb_job **out)
{
    gurabo_context *context = stream->context;
    if (context->last_submission == UINT64_MAX) {
        return gb_fail(GURABO_OUT_OF_RANGE, context->backend->id);
    }
    if (!context->backend->marker_create) {
        *out = NULL;
        return GURABO_OK;
    }
    gb_job *job = calloc(1, sizeof(*job));
    if (!job) {
        return gb_fail(GURABO_OUT_OF_MEMORY, context->backend->id);
    }
    gb_result result = context->backend->marker_create(stream->native, &job->marker);
    if (result.status != GURABO_OK) {
        free(job);
        remember(stream, result, GURABO_ERROR_CALL, 0);
        return gb_call_result(result, context->backend->id);
    }
    *out = job;
    return GURABO_OK;
}

void gb_job_end(gurabo_stream *stream, gb_job *job, gurabo_buffer *const *buffers,
                uint32_t count, gb_result result)
{
    gurabo_context *context = stream->context;
    uint64_t id = ++context->last_submission;
    remember(stream, result, GURABO_ERROR_EXECUTION, id);
    if (!job) {
        return;
    }

    /* Native submission may have queued work even when it reports an error. */
    for (uint32_t i = 0; i < count; ++i) {
        ++buffers[i]->references;
        job->buffers[i] = buffers[i];
    }
    job->count = count;
    if (stream->tail) {
        stream->tail->next = job;
    } else {
        stream->head = job;
    }
    stream->tail = job;
    result = context->backend->marker_record(stream->native, job->marker);
    remember(stream, result, GURABO_ERROR_EXECUTION, 0);
}

static void retire_transfer(gurabo_stream *stream, int safe)
{
    gurabo_buffer *buffer = stream->faulted_transfer;
    if (buffer) {
        stream->faulted_transfer = NULL;
        buffer->transfer_in_doubt = 0;
        if (!safe) {
            buffer->quarantined = 1;
        }
        remember(stream, gb_buffer_unref(buffer), GURABO_ERROR_CLEANUP, 0);
    }
}

gurabo_status gb_stream_sync_locked(gurabo_stream *stream, int *completed)
{
    gb_result result = stream->context->backend->stream_sync(stream->native);
    *completed = result.status == GURABO_OK;
    remember(stream, result, GURABO_ERROR_EXECUTION, 0);
    if (*completed) {
        retire_transfer(stream, 1);
        while (stream->head) {
            retire_job(stream, 1);
        }
    }
    return stream->error.status == GURABO_OK ? GURABO_OK : gb_report(&stream->error);
}

gurabo_status gurabo_stream_create(gurabo_context *context, gurabo_stream **out)
{
    if (!context) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_backend id = context->backend->id;
    if (!out || *out) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gb_lock(&context->gate);
    if (context->lost) {
        gb_unlock(&context->gate);
        return gb_fail(GURABO_DEVICE_LOST, id);
    }
    gurabo_stream *stream = calloc(1, sizeof(*stream));
    if (!stream) {
        gb_unlock(&context->gate);
        return gb_fail(GURABO_OUT_OF_MEMORY, id);
    }
    stream->context = context;
    gb_result result = context->backend->stream_create(context->ordinal, &stream->native);
    if (result.status == GURABO_OK) {
        stream->next = context->streams;
        context->streams = stream;
        ++context->children;
        *out = stream;
    } else {
        remember(stream, result, GURABO_ERROR_CALL, 0);
        free(stream);
    }
    gb_unlock(&context->gate);
    return gb_call_result(result, id);
}

gurabo_status gurabo_stream_synchronize(gurabo_stream *stream)
{
    if (!stream) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gb_lock(&context->gate);
    int completed;
    gurabo_status status = gb_stream_sync_locked(stream, &completed);
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_stream_get_error(gurabo_stream *stream, gurabo_error_info *out)
{
    if (!stream) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gb_lock(&context->gate);
    gb_stream_reap(stream);
    gurabo_status status = gb_get_error(&stream->error, out);
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_stream_destroy(gurabo_stream **stream)
{
    if (!stream) {
        return gb_fail(GURABO_INVALID_ARGUMENT, 0);
    }
    gurabo_stream *value = *stream;
    if (!value) {
        return GURABO_OK;
    }
    gurabo_context *context = value->context;
    gb_lock(&context->gate);
    int completed;
    gb_stream_sync_locked(value, &completed);
    if (completed) {
        gb_result result = context->backend->stream_destroy(value->native);
        remember(value, result, GURABO_ERROR_CLEANUP, 0);
    } else {
        /* Without a completion proof, native handles and storage cannot be reused. */
        retire_transfer(value, 0);
        while (value->head) {
            retire_job(value, 0);
        }
    }
    gurabo_stream **link = &context->streams;
    while (*link != value) {
        link = &(*link)->next;
    }
    *link = value->next;
    --context->children;
    gurabo_status status = value->error.status == GURABO_OK ? GURABO_OK : gb_report(&value->error);
    free(value);
    *stream = NULL;
    gb_unlock(&context->gate);
    return status;
}
