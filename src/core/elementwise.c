#include "core.h"

_Static_assert(GB_POINTWISE_MAX_RANK == GURABO_MAX_RANK, "Pointwise rank must match Tensor rank");
_Static_assert(GB_POINTWISE_F32 == GURABO_F32 && GB_POINTWISE_F16 == GURABO_F16 &&
               GB_POINTWISE_BF16 == GURABO_BF16, "Pointwise dtypes must match public dtypes");
_Static_assert(GB_POINTWISE_ADD == GURABO_ADD && GB_POINTWISE_MUL == GURABO_MUL,
               "Pointwise operations must match public operations");

static int overlaps(const gurabo_tensor *out, const gurabo_tensor *input)
{
    return out->buffer == input->buffer &&
           gb_overlap(out->offset, out->span, input->offset, input->span);
}

static gurabo_status submit(gurabo_stream *stream, gurabo_tensor *out,
                            const gurabo_tensor *a, const gurabo_tensor *b, gb_pointwise *args)
{
    if (!out->span) {
        return GURABO_OK;
    }
    gurabo_context *context = stream->context;
    const gb_backend *backend = context->backend;
    if (!backend->pointwise) {
        return gb_fail(GURABO_UNSUPPORTED, backend->id);
    }
    args->out = gb_buffer_address(out->buffer, out->offset);
    args->a = gb_buffer_address(a->buffer, a->offset);
    args->b = b ? gb_buffer_address(b->buffer, b->offset) : NULL;
    args->input_dtype = a->desc.dtype;
    args->output_dtype = out->desc.dtype;
    args->count = out->span / (out->desc.dtype == GURABO_F32 ? sizeof(float) : sizeof(uint16_t));

    gb_lock(&context->gate);
    gurabo_status status = gb_stream_ready(stream);
    if (status == GURABO_OK && (gb_buffer_unavailable(out->buffer) ||
                               gb_buffer_unavailable(a->buffer) ||
                               (b && gb_buffer_unavailable(b->buffer)))) {
        status = gb_fail(GURABO_BACKEND_ERROR, backend->id);
    }
    if (status == GURABO_OK && backend->pointwise_prepare && !context->pointwise_state) {
        gb_result result = backend->pointwise_prepare(context->ordinal, &context->pointwise_state);
        if (result.status != GURABO_OK) {
            gurabo_error_info error = gb_make_error(result, backend->id, GURABO_ERROR_CALL, 0);
            gb_context_error(context, &error);
            status = gb_report(&error);
        }
    }
    if (status == GURABO_OK) {
        gb_job *job;
        status = gb_job_begin(stream, &job);
        if (status == GURABO_OK) {
            if (job) {
                job->uses_pointwise_module = context->pointwise_state != NULL;
            }
            gb_result result = backend->pointwise(stream->native, context->pointwise_state, args);
            gurabo_buffer *buffers[] = {out->buffer, a->buffer, b ? b->buffer : NULL};
            gb_job_end(stream, job, buffers, b ? 3 : 2, result);
        }
    }
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_tensor_cast_async(gurabo_stream *stream, gurabo_tensor *out,
                                       const gurabo_tensor *input)
{
    if (!stream || !out || !input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || input->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    if (out->desc.rank != input->desc.rank ||
        memcmp(out->desc.extent, input->desc.extent,
               out->desc.rank * sizeof(out->desc.extent[0])) != 0 || overlaps(out, input)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!out->contiguous || !input->contiguous ||
        (out->desc.dtype == GURABO_F32) == (input->desc.dtype == GURABO_F32)) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    gb_pointwise args = {.op = GB_POINTWISE_CAST};
    return submit(stream, out, input, NULL, &args);
}

static int make_broadcast(const gurabo_tensor_desc *out, const gurabo_tensor_desc *a,
                          const gurabo_tensor_desc *b, gb_pointwise *args)
{
    uint32_t rank = a->rank > b->rank ? a->rank : b->rank;
    if (out->rank != rank) {
        return 0;
    }
    for (uint32_t axis = 0; axis < rank; ++axis) {
        uint32_t from_end = rank - axis;
        int a_present = from_end <= a->rank;
        int b_present = from_end <= b->rank;
        uint64_t an = a_present ? a->extent[a->rank - from_end] : 1;
        uint64_t bn = b_present ? b->extent[b->rank - from_end] : 1;
        uint64_t n = an == 1 ? bn : an;
        if ((an != bn && an != 1 && bn != 1) || out->extent[axis] != n) {
            return 0;
        }
        args->extent[axis] = n;
        args->a_stride[axis] = a_present && an != 1 ? a->stride[a->rank - from_end] : 0;
        args->b_stride[axis] = b_present && bn != 1 ? b->stride[b->rank - from_end] : 0;
        if (an != bn) {
            args->broadcast_rank = rank;
        }
    }
    return 1;
}

gurabo_status gurabo_binary_async(gurabo_stream *stream, gurabo_binary_op op,
                                  gurabo_tensor *out, const gurabo_tensor *a,
                                  const gurabo_tensor *b)
{
    if (!stream || !out || !a || !b) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || a->buffer->context != context ||
        b->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    if (op != GURABO_ADD && op != GURABO_MUL) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (out->desc.dtype != a->desc.dtype || out->desc.dtype != b->desc.dtype ||
        !out->contiguous || !a->contiguous || !b->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    gb_pointwise args = {.op = op};
    if (!make_broadcast(&out->desc, &a->desc, &b->desc, &args) ||
        (overlaps(out, a) && (out->offset != a->offset || out->span != a->span)) ||
        (overlaps(out, b) && (out->offset != b->offset || out->span != b->span))) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    return submit(stream, out, a, b, &args);
}
