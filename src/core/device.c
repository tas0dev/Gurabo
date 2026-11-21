#include "core.h"

static const gb_backend *const backends[] = {
#ifdef GURABO_WITH_CPU
    &gb_cpu,
#endif
#ifdef GURABO_WITH_HIP
    &gb_hip,
#endif
};

#define GB_BACKEND_COUNT ((uint32_t)(sizeof(backends) / sizeof(backends[0])))

static const gb_backend *find_backend(gurabo_backend id)
{
    for (uint32_t i = 0; i < GB_BACKEND_COUNT; ++i) {
        if (backends[i]->id == id) {
            return backends[i];
        }
    }
    return NULL;
}

static gurabo_status missing_backend(gurabo_backend id)
{
    return gb_fail(id >= GURABO_CPU && id <= GURABO_CUDA
                   ? GURABO_BACKEND_UNAVAILABLE : GURABO_INVALID_ARGUMENT, id);
}

gurabo_status gurabo_backend_count(uint32_t *out)
{
    if (!out) {
        return gb_fail(GURABO_INVALID_ARGUMENT, 0);
    }
    *out = GB_BACKEND_COUNT;
    return GURABO_OK;
}

gurabo_status gurabo_backend_at(uint32_t index, gurabo_backend *out)
{
    if (!out) {
        return gb_fail(GURABO_INVALID_ARGUMENT, 0);
    }
    if (index >= GB_BACKEND_COUNT) {
        return gb_fail(GURABO_OUT_OF_RANGE, 0);
    }
    *out = backends[index]->id;
    return GURABO_OK;
}

gurabo_status gurabo_device_count(gurabo_backend id, uint32_t *out)
{
    if (!out) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    const gb_backend *backend = find_backend(id);
    if (!backend) {
        return missing_backend(id);
    }
    uint32_t count = 0;
    gb_result result = backend->device_count(&count);
    if (result.status == GURABO_OK) {
        *out = count;
    }
    return gb_call_result(result, id);
}

static gurabo_status check_device(gurabo_backend id, uint32_t ordinal)
{
    uint32_t count = 0;
    gurabo_status status = gurabo_device_count(id, &count);
    if (status == GURABO_OK && ordinal >= count) {
        return gb_fail(GURABO_OUT_OF_RANGE, id);
    }
    return status;
}

gurabo_status gurabo_device_get_info(gurabo_backend id, uint32_t ordinal,
                                    gurabo_device_info *out)
{
    if (!out || out->struct_size < sizeof(uint32_t)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gurabo_status status = check_device(id, ordinal);
    if (status != GURABO_OK) {
        return status;
    }
    gurabo_device_info info = {0};
    info.backend = id;
    info.ordinal = ordinal;
    gb_result result = find_backend(id)->device_info(ordinal, &info);
    if (result.status == GURABO_OK) {
        gb_write_struct(out, &info, out->struct_size, sizeof(info));
    }
    return gb_call_result(result, id);
}

gurabo_status gurabo_device_get_caps(gurabo_backend id, uint32_t ordinal,
                                    gurabo_device_caps *out)
{
    if (!out || out->struct_size < sizeof(uint32_t)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gurabo_status status = check_device(id, ordinal);
    if (status != GURABO_OK) {
        return status;
    }
    gurabo_device_caps caps = {0};
    caps.max_tensor_rank = GURABO_MAX_RANK;
    caps.storage_dtype_mask = UINT64_C(1) << GURABO_F32;
    caps.feature_mask = GURABO_FEATURE_F32_MATMUL;
    caps.min_buffer_alignment = find_backend(id)->alignment;
    gb_write_struct(out, &caps, out->struct_size, sizeof(caps));
    return GURABO_OK;
}

gurabo_status gurabo_context_create(gurabo_backend id, uint32_t ordinal,
                                   gurabo_context **out)
{
    if (!out || *out) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gurabo_status status = check_device(id, ordinal);
    if (status != GURABO_OK) {
        return status;
    }
    gurabo_context *context = calloc(1, sizeof(*context));
    if (!context || gb_mutex_init(&context->gate) != 0) {
        free(context);
        return gb_fail(GURABO_OUT_OF_MEMORY, id);
    }
    context->backend = find_backend(id);
    context->ordinal = ordinal;
    *out = context;
    return GURABO_OK;
}

gurabo_status gurabo_context_destroy(gurabo_context **context)
{
    if (!context) {
        return gb_fail(GURABO_INVALID_ARGUMENT, 0);
    }
    gurabo_context *value = *context;
    if (!value) {
        return GURABO_OK;
    }
    gb_lock(&value->gate);
    if (value->children) {
        gb_unlock(&value->gate);
        return gb_fail(GURABO_BUSY, value->backend->id);
    }
    gurabo_error_info error = value->error;
    gb_unlock(&value->gate);
    gb_mutex_destroy(&value->gate);
    free(value);
    *context = NULL;
    return error.status == GURABO_OK ? GURABO_OK : gb_report(&error);
}

void gb_context_error(gurabo_context *context, const gurabo_error_info *error)
{
    if (context->error.status == GURABO_OK &&
        (error->phase != GURABO_ERROR_CALL || error->status == GURABO_DEVICE_LOST)) {
        context->error = *error;
    }
    if (error->status == GURABO_DEVICE_LOST) {
        context->lost = 1;
    }
}

gurabo_status gurabo_context_get_error(gurabo_context *context, gurabo_error_info *out)
{
    if (!context) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gb_lock(&context->gate);
    for (gurabo_stream *stream = context->streams; stream; stream = stream->next) {
        gb_stream_reap(stream);
    }
    gurabo_status status = gb_get_error(&context->error, out);
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_context_synchronize(gurabo_context *context)
{
    if (!context) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gb_lock(&context->gate);
    for (gurabo_stream *stream = context->streams; stream; stream = stream->next) {
        int completed;
        gb_stream_sync_locked(stream, &completed);
    }
    gurabo_status status = context->error.status != GURABO_OK ? gb_report(&context->error)
                                                            : GURABO_OK;
    if (status == GURABO_OK && context->lost) {
        status = gb_fail(GURABO_DEVICE_LOST, context->backend->id);
    }
    gb_unlock(&context->gate);
    return status;
}
