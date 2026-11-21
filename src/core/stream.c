#include "core.h"

static void remember_error(gurabo_stream *stream, const gurabo_error_info *error,
                           uint64_t position)
{
    if (stream->error.status == GURABO_OK && error->phase != GURABO_ERROR_CALL) {
        stream->error = *error;
    }
    if (error->phase == GURABO_ERROR_EXECUTION && position &&
        (!stream->execution_error_at || position < stream->execution_error_at)) {
        stream->execution_error_at = position;
        stream->execution_error = *error;
    }
    gb_context_error(stream->context, error);
}

static void remember(gurabo_stream *stream, gb_result result, gurabo_error_phase phase,
                     uint64_t submission)
{
    if (result.status != GURABO_OK) {
        gurabo_error_info error = gb_make_error(result, stream->context->backend->id,
                                               phase, submission);
        uint64_t position = submission;
        if (!position && stream->head) {
            position = stream->head->id;
        }
        remember_error(stream, &error, position);
    }
}

static void event_observe_error(gurabo_event *event)
{
    gurabo_stream *source = event->source;
    if (!event->resolved && source && source->execution_error_at &&
        source->execution_error_at <= event->submission_id) {
        event->error = source->execution_error;
        event->resolved = 1;
    }
}

static gb_result event_unref(gurabo_event *event, int safe)
{
    if (!safe) {
        event->quarantined = 1;
    }
    gb_result result = gb_status(GURABO_OK);
    if (--event->references == 0) {
        gurabo_context *context = event->context;
        if (event->submission_id && context->backend->marker_destroy && !event->quarantined) {
            result = context->backend->marker_destroy(context->ordinal, event->marker);
        }
        free(event);
    }
    return result;
}

static void remember_dependency(gurabo_stream *stream, const gurabo_error_info *error,
                                uint64_t submission)
{
    if (error->status != GURABO_OK) {
        gurabo_error_info propagated = *error;
        propagated.submission_id = submission;
        remember_error(stream, &propagated, submission);
    }
}

static void retire_job(gurabo_stream *stream, int safe, int defer_buffer_free)
{
    gb_job *job = stream->head;
    stream->head = job->next;
    if (!stream->head) {
        stream->tail = NULL;
    }
    gurabo_context *context = stream->context;
    if (job->dependency) {
        gurabo_event *dependency = job->dependency;
        if (safe) {
            remember_dependency(stream, &dependency->error, job->id);
        }
        remember(stream, event_unref(dependency, safe), GURABO_ERROR_CLEANUP, 0);
    }
    if (job->recorded_event) {
        gurabo_event *event = job->recorded_event;
        event_observe_error(event);
        event->resolved = 1;
        event->source = NULL;
        remember(stream, event_unref(event, safe), GURABO_ERROR_CLEANUP, 0);
    } else if (safe) {
        gb_result result = context->backend->marker_destroy(context->ordinal, job->marker);
        remember(stream, result, GURABO_ERROR_CLEANUP, 0);
    }
    for (uint32_t i = 0; i < job->count; ++i) {
        if (!safe) {
            job->buffers[i]->quarantined = 1;
        }
        if (defer_buffer_free) {
            /* A native buffer free can synchronize unrelated device work. */
            gb_buffer_unref_deferred(job->buffers[i]);
        } else {
            gb_result result = gb_buffer_unref(job->buffers[i]);
            remember(stream, result, GURABO_ERROR_CLEANUP, 0);
        }
    }
    free(job);
}

static void reap_through(gurabo_stream *stream, uint64_t limit)
{
    gurabo_context *context = stream->context;
    while (stream->head && stream->head->id <= limit) {
        gurabo_stream *current = stream;
        /* Dependencies always refer to earlier submissions. Walk the chain
           iteratively so its length does not consume the C stack. */
        while (current->head->dependency) {
            gurabo_event *dependency = current->head->dependency;
            event_observe_error(dependency);
            if (dependency->resolved) {
                break;
            }
            current = dependency->source;
        }
        gb_job *job = current->head;
        if (!job->marker_recorded) {
            return;
        }
        gb_result result = context->backend->marker_query(context->ordinal, job->marker);
        if (result.status == GURABO_NOT_READY) {
            break;
        }
        if (result.status != GURABO_OK) {
            remember(current, result, GURABO_ERROR_EXECUTION, 0);
            if (current == stream) {
                break;
            }
            continue;
        }
        retire_job(current, 1, 1);
    }
}

void gb_stream_reap(gurabo_stream *stream)
{
    reap_through(stream, UINT64_MAX);
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
    gb_result result = context->backend->marker_create(context->ordinal, &job->marker);
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
    job->id = id;

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
    job->marker_recorded = result.status == GURABO_OK;
    remember(stream, result, GURABO_ERROR_EXECUTION, id);
}

/* The caller proved that the native Stream finished through limit. The
   dependency's result still has to be observed on its recording Stream. */
static void retire_completed_through(gurabo_stream *stream, uint64_t limit,
                                     int defer_buffer_free)
{
    while (stream->head && stream->head->id <= limit) {
        gb_job *job = stream->head;
        gurabo_event *dependency = job->dependency;
        if (dependency && !dependency->resolved) {
            reap_through(dependency->source, dependency->submission_id);
            event_observe_error(dependency);
            if (!dependency->resolved) {
                gb_result result = {GURABO_BACKEND_ERROR, GURABO_NATIVE_GURABO, 0,
                                    "stream completed before its event dependency"};
                remember(stream, result, GURABO_ERROR_EXECUTION, job->id);
            }
        }
        retire_job(stream, 1, defer_buffer_free);
    }
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
        retire_completed_through(stream, UINT64_MAX, 0);
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
            retire_job(value, 0, 0);
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

gurabo_status gurabo_event_create(gurabo_context *context, gurabo_event **out)
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
    gurabo_event *event = calloc(1, sizeof(*event));
    if (!event) {
        gb_unlock(&context->gate);
        return gb_fail(GURABO_OUT_OF_MEMORY, id);
    }
    event->context = context;
    event->references = 1;
    ++context->children;
    *out = event;
    gb_unlock(&context->gate);
    return GURABO_OK;
}

gurabo_status gurabo_event_record(gurabo_stream *stream, gurabo_event *event)
{
    if (!stream || !event) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (event->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    gb_lock(&context->gate);
    if (event->submission_id) {
        gb_unlock(&context->gate);
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gurabo_status status = gb_stream_ready(stream);
    gb_job *job = NULL;
    if (status == GURABO_OK) {
        status = gb_job_begin(stream, &job);
    }
    if (status == GURABO_OK) {
        event->submission_id = context->last_submission + 1;
        if (job) {
            event->source = stream;
            event->marker = job->marker;
            ++event->references;
            job->recorded_event = event;
        } else {
            event->resolved = 1;
        }
        gb_job_end(stream, job, NULL, 0, gb_status(GURABO_OK));
    }
    gb_unlock(&context->gate);
    return status;
}

static gurabo_status event_completion(gurabo_event *event, int wait)
{
    if (!event) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = event->context;
    gb_lock(&context->gate);
    if (!event->submission_id) {
        gb_unlock(&context->gate);
        return gb_fail(GURABO_INVALID_ARGUMENT, context->backend->id);
    }
    event_observe_error(event);
    if (!event->resolved && wait) {
        gurabo_stream *source = event->source;
        gb_result result = context->backend->marker_sync(context->ordinal, event->marker);
        if (result.status == GURABO_OK) {
            retire_completed_through(source, event->submission_id, 1);
        } else {
            remember(source, result, GURABO_ERROR_EXECUTION, 0);
            event_observe_error(event);
        }
    } else if (!event->resolved) {
        reap_through(event->source, event->submission_id);
        event_observe_error(event);
    }
    gurabo_status status = GURABO_NOT_READY;
    if (event->resolved) {
        status = event->error.status == GURABO_OK ? GURABO_OK : gb_report(&event->error);
    }
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_event_query(const gurabo_event *event)
{
    return event_completion((gurabo_event *)event, 0);
}

gurabo_status gurabo_event_synchronize(const gurabo_event *event)
{
    return event_completion((gurabo_event *)event, 1);
}

gurabo_status gurabo_stream_wait_event(gurabo_stream *stream, const gurabo_event *event)
{
    if (!stream || !event) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (event->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    gb_lock(&context->gate);
    if (!event->submission_id) {
        gb_unlock(&context->gate);
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gurabo_status status = gb_stream_ready(stream);
    gb_job *job = NULL;
    if (status == GURABO_OK) {
        status = gb_job_begin(stream, &job);
    }
    if (status == GURABO_OK) {
        gurabo_event *dependency = (gurabo_event *)event;
        event_observe_error(dependency);
        gb_result result = gb_status(GURABO_OK);
        if (job) {
            ++dependency->references;
            job->dependency = dependency;
            if (event->error.status == GURABO_OK) {
                result = context->backend->marker_wait(stream->native, event->marker);
            }
        }
        if (event->error.status != GURABO_OK) {
            remember_dependency(stream, &event->error, context->last_submission + 1);
        }
        gb_job_end(stream, job, NULL, 0, result);
    }
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_event_release(gurabo_event **event)
{
    if (!event) {
        return gb_fail(GURABO_INVALID_ARGUMENT, 0);
    }
    gurabo_event *value = *event;
    if (!value) {
        return GURABO_OK;
    }
    gurabo_context *context = value->context;
    gb_lock(&context->gate);
    --context->children;
    *event = NULL;
    gb_result result = event_unref(value, 1);
    gurabo_status status = GURABO_OK;
    if (result.status != GURABO_OK) {
        gurabo_error_info error = gb_make_error(result, context->backend->id,
                                                GURABO_ERROR_CLEANUP, 0);
        gb_context_error(context, &error);
        status = gb_report(&error);
    }
    gb_unlock(&context->gate);
    return status;
}
