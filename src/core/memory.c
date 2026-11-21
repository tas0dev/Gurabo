#include "core.h"

gurabo_status gurabo_buffer_alloc(gurabo_context *context, uint64_t bytes,
                                 uint64_t alignment, gurabo_buffer **out)
{
    if (!context) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    const gb_backend *backend = context->backend;
    if (!out || *out || !bytes || (alignment & (alignment - 1))) {
        return gb_fail(GURABO_INVALID_ARGUMENT, backend->id);
    }
    if (alignment > SIZE_MAX) {
        return gb_fail(GURABO_UNSUPPORTED, backend->id);
    }
    if (bytes > SIZE_MAX) {
        return gb_fail(GURABO_OUT_OF_RANGE, backend->id);
    }
    if (alignment < backend->alignment) {
        alignment = backend->alignment;
    }

    gb_lock(&context->gate);
    for (gurabo_stream *stream = context->streams; stream; stream = stream->next) {
        gb_stream_reap(stream);
    }
    if (context->lost || bytes > UINT64_MAX - context->allocated_bytes) {
        gurabo_status status = context->lost ? GURABO_DEVICE_LOST : GURABO_OUT_OF_RANGE;
        gb_unlock(&context->gate);
        return gb_fail(status, backend->id);
    }
    gurabo_buffer *buffer = calloc(1, sizeof(*buffer));
    if (!buffer) {
        gb_unlock(&context->gate);
        return gb_fail(GURABO_OUT_OF_MEMORY, backend->id);
    }
    gb_result result = backend->alloc(context->ordinal, (size_t)bytes, (size_t)alignment,
                                      &buffer->allocation);
    if (result.status == GURABO_OK) {
        buffer->context = context;
        buffer->references = 1;
        buffer->bytes = bytes;
        context->allocated_bytes += bytes;
        ++context->children;
        *out = buffer;
    } else {
        gurabo_error_info error = gb_make_error(result, backend->id, GURABO_ERROR_CALL, 0);
        gb_context_error(context, &error);
        free(buffer);
    }
    gb_unlock(&context->gate);
    return gb_call_result(result, backend->id);
}

gb_result gb_buffer_unref(gurabo_buffer *buffer)
{
    gb_result result = gb_status(GURABO_OK);
    if (--buffer->references == 0) {
        gurabo_context *context = buffer->context;
        if (!buffer->quarantined) {
            result = context->backend->free(context->ordinal, buffer->allocation);
            if (result.status == GURABO_OK) {
                context->allocated_bytes -= buffer->bytes;
            } else {
                gurabo_error_info error = gb_make_error(result, context->backend->id,
                                                        GURABO_ERROR_CLEANUP, 0);
                gb_context_error(context, &error);
            }
        }
        free(buffer);
    }
    return result;
}

gurabo_status gurabo_buffer_release(gurabo_buffer **buffer)
{
    if (!buffer) {
        return gb_fail(GURABO_INVALID_ARGUMENT, 0);
    }
    gurabo_buffer *value = *buffer;
    if (!value) {
        return GURABO_OK;
    }
    gurabo_context *context = value->context;
    gb_lock(&context->gate);
    --context->children;
    *buffer = NULL;
    gb_result result = gb_buffer_unref(value);
    gurabo_status status = GURABO_OK;
    if (result.status != GURABO_OK) {
        gurabo_error_info error = gb_make_error(result, context->backend->id,
                                                GURABO_ERROR_CLEANUP, 0);
        status = gb_report(&error);
    }
    gb_unlock(&context->gate);
    return status;
}

static gurabo_status transfer(gurabo_stream *stream, const gurabo_buffer *buffer,
                              uint64_t offset, void *host_dst, const void *host_src,
                              uint64_t bytes, int to_device)
{
    if (!stream || !buffer) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    const gb_backend *backend = context->backend;
    if (buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, backend->id);
    }
    if (!gb_range(buffer, offset, bytes)) {
        return gb_fail(GURABO_OUT_OF_RANGE, backend->id);
    }
    if (bytes && !(to_device ? host_src : host_dst)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, backend->id);
    }
    if (!bytes) {
        return GURABO_OK;
    }
    gb_lock(&context->gate);
    int completed;
    gurabo_status status = gb_stream_sync_locked(stream, &completed);
    if (status == GURABO_OK && context->lost) {
        status = gb_fail(GURABO_DEVICE_LOST, backend->id);
    }
    if (status == GURABO_OK && gb_buffer_unavailable(buffer)) {
        status = gb_fail(GURABO_BACKEND_ERROR, backend->id);
    }
    if (status == GURABO_OK) {
        void *device = gb_buffer_address(buffer, offset);
        int unsafe = 0;
        gb_result result = backend->transfer(stream->native, to_device ? device : host_dst,
                                             to_device ? host_src : device, (size_t)bytes,
                                             to_device, &unsafe);
        if (unsafe) {
            gurabo_buffer *retained = (gurabo_buffer *)buffer;
            retained->transfer_in_doubt = 1;
            ++retained->references;
            stream->faulted_transfer = retained;
        }
        if (result.status != GURABO_OK) {
            gurabo_error_phase phase = unsafe ? GURABO_ERROR_EXECUTION : GURABO_ERROR_CALL;
            gurabo_error_info error = gb_make_error(result, backend->id, phase, 0);
            if (unsafe) {
                stream->error = error;
            }
            gb_context_error(context, &error);
            status = gb_report(&error);
        }
    }
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_buffer_write(gurabo_stream *stream, gurabo_buffer *dst,
                                  uint64_t offset, const void *src, uint64_t bytes)
{
    return transfer(stream, dst, offset, NULL, src, bytes, 1);
}

gurabo_status gurabo_buffer_read(gurabo_stream *stream, void *dst,
                                 const gurabo_buffer *src, uint64_t offset, uint64_t bytes)
{
    return transfer(stream, src, offset, dst, NULL, bytes, 0);
}

gurabo_status gurabo_buffer_copy_async(gurabo_stream *stream, gurabo_buffer *dst,
                                       uint64_t dst_offset, const gurabo_buffer *src,
                                       uint64_t src_offset, uint64_t bytes)
{
    if (!stream || !dst || !src) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    const gb_backend *backend = context->backend;
    if (dst->context != context || src->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, backend->id);
    }
    if (!gb_range(dst, dst_offset, bytes) || !gb_range(src, src_offset, bytes)) {
        return gb_fail(GURABO_OUT_OF_RANGE, backend->id);
    }
    if (dst == src && gb_overlap(dst_offset, bytes, src_offset, bytes)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, backend->id);
    }
    if (!bytes) {
        return GURABO_OK;
    }
    gb_lock(&context->gate);
    gurabo_status status = gb_stream_ready(stream);
    if (status == GURABO_OK && (gb_buffer_unavailable(dst) || gb_buffer_unavailable(src))) {
        status = gb_fail(GURABO_BACKEND_ERROR, backend->id);
    }
    if (status == GURABO_OK) {
        gb_job *job;
        status = gb_job_begin(stream, &job);
        if (status == GURABO_OK) {
            gb_result result = backend->copy(stream->native, gb_buffer_address(dst, dst_offset),
                                             gb_buffer_address(src, src_offset), (size_t)bytes);
            gurabo_buffer *buffers[] = {dst, (gurabo_buffer *)src};
            gb_job_end(stream, job, buffers, 2, result);
        }
    }
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_context_memory_usage(gurabo_context *context, uint64_t *allocated_bytes)
{
    if (!context) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    if (!allocated_bytes) {
        return gb_fail(GURABO_INVALID_ARGUMENT, context->backend->id);
    }
    gb_lock(&context->gate);
    for (gurabo_stream *stream = context->streams; stream; stream = stream->next) {
        gb_stream_reap(stream);
    }
    *allocated_bytes = context->allocated_bytes;
    gb_unlock(&context->gate);
    return GURABO_OK;
}
