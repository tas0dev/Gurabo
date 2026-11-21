#include "core.h"
#include <math.h>

_Static_assert(GB_POINTWISE_MAX_RANK == GURABO_MAX_RANK, "Pointwise rank must match Tensor rank");
_Static_assert(GB_POINTWISE_F32 == GURABO_F32 && GB_POINTWISE_F16 == GURABO_F16 &&
               GB_POINTWISE_BF16 == GURABO_BF16, "Pointwise dtypes must match public dtypes");
_Static_assert(GB_POINTWISE_ADD == GURABO_ADD && GB_POINTWISE_MUL == GURABO_MUL,
               "Pointwise operations must match public operations");
_Static_assert(GB_ROW_SUM == GURABO_SUM && GB_ROW_MEAN == GURABO_MEAN,
               "Reduction operations must match public operations");

static int overlaps(const gurabo_tensor *out, const gurabo_tensor *input)
{
    return out->buffer == input->buffer &&
           gb_overlap(out->offset, out->span, input->offset, input->span);
}

static gurabo_status submit(gurabo_stream *stream, gurabo_tensor *out,
                            const gurabo_tensor *a, const gurabo_tensor *b, const gb_kernel *kernel)
{
    if (!out->span) {
        return GURABO_OK;
    }
    gurabo_context *context = stream->context;
    const gb_backend *backend = context->backend;
    if (!backend->kernel_launch) {
        return gb_fail(GURABO_UNSUPPORTED, backend->id);
    }

    gb_lock(&context->gate);
    gurabo_status status = gb_stream_ready(stream);
    if (status == GURABO_OK && (gb_buffer_unavailable(out->buffer) ||
                               gb_buffer_unavailable(a->buffer) ||
                               (b && gb_buffer_unavailable(b->buffer)))) {
        status = gb_fail(GURABO_BACKEND_ERROR, backend->id);
    }
    if (status == GURABO_OK && backend->kernel_prepare && !context->kernel_state) {
        gb_result result = backend->kernel_prepare(context->ordinal, &context->kernel_state);
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
                job->uses_kernel_module = context->kernel_state != NULL;
            }
            gb_result result = backend->kernel_launch(stream->native, context->kernel_state, kernel);
            gurabo_buffer *buffers[] = {out->buffer, a->buffer, b ? b->buffer : NULL};
            gb_job_end(stream, job, buffers, b ? 3 : 2, result);
        }
    }
    gb_unlock(&context->gate);
    return status;
}

static uint64_t element_count(const gurabo_tensor *tensor)
{
    return tensor->span / (tensor->desc.dtype == GURABO_F32 ? sizeof(float) : sizeof(uint16_t));
}

static gurabo_status submit_pointwise(gurabo_stream *stream, gurabo_tensor *out,
                                      const gurabo_tensor *a, const gurabo_tensor *b,
                                      gb_kernel *kernel)
{
    gb_pointwise *args = &kernel->args.pointwise;
    args->out = gb_buffer_address(out->buffer, out->offset);
    args->a = gb_buffer_address(a->buffer, a->offset);
    args->b = b ? gb_buffer_address(b->buffer, b->offset) : NULL;
    args->input_dtype = a->desc.dtype;
    args->output_dtype = out->desc.dtype;
    args->count = element_count(out);
    return submit(stream, out, a, b, kernel);
}

static int shape_equal(const gurabo_tensor_desc *a, const gurabo_tensor_desc *b)
{
    return a->rank == b->rank &&
           memcmp(a->extent, b->extent, a->rank * sizeof(a->extent[0])) == 0;
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
    if (!shape_equal(&out->desc, &input->desc) || overlaps(out, input)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!out->contiguous || !input->contiguous ||
        (out->desc.dtype == GURABO_F32) == (input->desc.dtype == GURABO_F32)) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise.op = GB_POINTWISE_CAST};
    return submit_pointwise(stream, out, input, NULL, &kernel);
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
    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise.op = op};
    if (!make_broadcast(&out->desc, &a->desc, &b->desc, &kernel.args.pointwise) ||
        (overlaps(out, a) && (out->offset != a->offset || out->span != a->span)) ||
        (overlaps(out, b) && (out->offset != b->offset || out->span != b->span))) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    return submit_pointwise(stream, out, a, b, &kernel);
}

gurabo_status gurabo_reduce_async(gurabo_stream *stream, gurabo_reduction op,
                                  gurabo_tensor *out, const gurabo_tensor *input,
                                  uint32_t axis_mask, uint32_t keep_dims)
{
    if (!stream || !out || !input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || input->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    uint32_t rank = input->desc.rank;
    if ((op != GURABO_SUM && op != GURABO_MEAN) || !rank || !axis_mask ||
        axis_mask >= (1u << rank) || keep_dims > 1) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    uint32_t axis = rank - 1;
    if (axis_mask != (1u << axis) || out->desc.dtype != GURABO_F32 ||
        !out->contiguous || !input->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    uint64_t columns = input->desc.extent[axis];
    if (out->desc.rank != axis + keep_dims ||
        memcmp(out->desc.extent, input->desc.extent, axis * sizeof(out->desc.extent[0])) != 0 ||
        (keep_dims && out->desc.extent[axis] != 1) ||
        (op == GURABO_MEAN && columns == 0) || overlaps(out, input)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_ROWWISE, .args.rowwise = {
        .out = gb_buffer_address(out->buffer, out->offset),
        .input = gb_buffer_address(input->buffer, input->offset),
        .rows = element_count(out),
        .columns = columns,
        .op = op,
        .input_dtype = input->desc.dtype
    }};
    return submit(stream, out, input, NULL, &kernel);
}

gurabo_status gurabo_rms_norm_async(gurabo_stream *stream, gurabo_tensor *out,
                                    const gurabo_tensor *input, const gurabo_tensor *scale,
                                    float epsilon)
{
    if (!stream || !out || !input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || input->buffer->context != context ||
        (scale && scale->buffer->context != context)) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    if (!input->desc.rank || !(epsilon > 0.0f) || !isfinite(epsilon) ||
        !shape_equal(&out->desc, &input->desc)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (out->desc.dtype != input->desc.dtype || !out->contiguous || !input->contiguous ||
        (scale && (!scale->contiguous ||
                   (scale->desc.dtype != GURABO_F32 && scale->desc.dtype != input->desc.dtype)))) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    uint64_t columns = input->desc.extent[input->desc.rank - 1];
    if (!columns || (overlaps(out, input) && out->offset != input->offset) ||
        (scale && (scale->desc.rank != 1 || scale->desc.extent[0] != columns ||
                   overlaps(out, scale)))) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_ROWWISE, .args.rowwise = {
        .out = gb_buffer_address(out->buffer, out->offset),
        .input = gb_buffer_address(input->buffer, input->offset),
        .scale = scale ? gb_buffer_address(scale->buffer, scale->offset) : NULL,
        .rows = element_count(out) / columns,
        .columns = columns,
        .op = GB_ROW_RMS_NORM,
        .input_dtype = input->desc.dtype,
        .scale_dtype = scale ? scale->desc.dtype : 0,
        .epsilon = epsilon
    }};
    return submit(stream, out, input, scale, &kernel);
}
