#include "core.h"
#include <math.h>

_Static_assert(GB_POINTWISE_MAX_RANK == GURABO_MAX_RANK, "Pointwise rank must match Tensor rank");
_Static_assert(GB_POINTWISE_F32 == GURABO_F32 && GB_POINTWISE_F16 == GURABO_F16 &&
               GB_POINTWISE_BF16 == GURABO_BF16, "Pointwise dtypes must match public dtypes");
_Static_assert(GB_POINTWISE_ADD == GURABO_ADD && GB_POINTWISE_MUL == GURABO_MUL &&
               GB_POINTWISE_SUB == GURABO_SUB && GB_POINTWISE_DIV == GURABO_DIV,
               "Pointwise operations must match public operations");
_Static_assert(GB_ROW_SUM == GURABO_SUM && GB_ROW_MEAN == GURABO_MEAN &&
               GB_ROW_MAX == GURABO_MAX,
               "Reduction operations must match public operations");

static int overlaps(const gurabo_tensor *out, const gurabo_tensor *input)
{
    return out->buffer == input->buffer &&
           gb_overlap(out->offset, out->span, input->offset, input->span);
}

static gurabo_status submit_kernel(gurabo_stream *stream, gurabo_tensor *out,
                                   const gurabo_tensor *a, const gurabo_tensor *b,
                                   const gurabo_tensor *c, const gb_kernel *kernel)
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
                               (a && gb_buffer_unavailable(a->buffer)) ||
                               (b && gb_buffer_unavailable(b->buffer)) ||
                               (c && gb_buffer_unavailable(c->buffer)))) {
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
            void *completion = NULL;
            gb_result result = backend->kernel_launch(stream->native, context->kernel_state,
                                                       kernel, &completion);
            if (job) {
                job->kernel_completion = completion;
            }
            gurabo_buffer *buffers[4] = {out->buffer};
            uint32_t count = 1;
            if (a) buffers[count++] = a->buffer;
            if (b) buffers[count++] = b->buffer;
            if (c) buffers[count++] = c->buffer;
            gb_job_end(stream, job, buffers, count, result);
        }
    }
    gb_unlock(&context->gate);
    return status;
}

static gurabo_status submit(gurabo_stream *stream, gurabo_tensor *out,
                            const gurabo_tensor *a, const gurabo_tensor *b, const gb_kernel *kernel)
{
    return submit_kernel(stream, out, a, b, NULL, kernel);
}

static uint64_t element_count(const gurabo_tensor *tensor)
{
    return tensor->span / gb_dtype_size(tensor->desc.dtype);
}

static gurabo_status submit_pointwise(gurabo_stream *stream, gurabo_tensor *out,
                                      const gurabo_tensor *a, const gurabo_tensor *b,
                                      gb_kernel *kernel)
{
    gb_pointwise *args = &kernel->args.pointwise;
    args->out = gb_buffer_address(out->buffer, out->offset);
    args->a = a ? gb_buffer_address(a->buffer, a->offset) : NULL;
    args->b = b ? gb_buffer_address(b->buffer, b->offset) : NULL;
    args->input_dtype = a ? a->desc.dtype : out->desc.dtype;
    args->output_dtype = out->desc.dtype;
    args->count = element_count(out);
    return submit(stream, out, a, b, kernel);
}

static int shape_equal(const gurabo_tensor_desc *a, const gurabo_tensor_desc *b)
{
    return a->rank == b->rank &&
           memcmp(a->extent, b->extent, a->rank * sizeof(a->extent[0])) == 0;
}

gurabo_status gurabo_tensor_copy_async(gurabo_stream *stream, gurabo_tensor *out,
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
    if (!shape_equal(&out->desc, &input->desc) || out->desc.dtype != input->desc.dtype) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!out->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    if (!out->span || (input->contiguous && out->buffer == input->buffer &&
                       out->offset == input->offset)) {
        return GURABO_OK;
    }
    if (overlaps(out, input)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (input->contiguous) {
        return gurabo_buffer_copy_async(stream, out->buffer, out->offset,
                                        input->buffer, input->offset, out->span);
    }

    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise = {
        .op = GB_POINTWISE_COPY_BITS,
        .element_bytes = (uint32_t)gb_dtype_size(input->desc.dtype),
        .broadcast_rank = input->desc.rank
    }};
    gb_pointwise *args = &kernel.args.pointwise;
    memcpy(args->extent, input->desc.extent, sizeof(args->extent));
    memcpy(args->a_stride, input->desc.stride, sizeof(args->a_stride));
    return submit_pointwise(stream, out, input, NULL, &kernel);
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
        !gb_floating_dtype(out->desc.dtype) || !gb_floating_dtype(input->desc.dtype) ||
        (out->desc.dtype == GURABO_F32) == (input->desc.dtype == GURABO_F32)) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise.op = GB_POINTWISE_CAST};
    return submit_pointwise(stream, out, input, NULL, &kernel);
}

static gurabo_status check_initializer(gurabo_stream *stream, gurabo_tensor *out)
{
    if (!stream || !out) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    if (out->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, context->backend->id);
    }
    if (!out->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, context->backend->id);
    }
    return GURABO_OK;
}

gurabo_status gurabo_tensor_fill_async(gurabo_stream *stream, gurabo_tensor *out,
                                       const void *value)
{
    gurabo_status status = check_initializer(stream, out);
    if (status != GURABO_OK) {
        return status;
    }
    if (!value) {
        return gb_fail(GURABO_INVALID_ARGUMENT, stream->context->backend->id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise = {
        .op = GB_POINTWISE_FILL, .element_bytes = (uint32_t)gb_dtype_size(out->desc.dtype)
    }};
    memcpy(kernel.args.pointwise.literal, value, kernel.args.pointwise.element_bytes);
    return submit_pointwise(stream, out, NULL, NULL, &kernel);
}

gurabo_status gurabo_tensor_zero_async(gurabo_stream *stream, gurabo_tensor *out)
{
    const uint64_t zero = 0;
    return gurabo_tensor_fill_async(stream, out, &zero);
}

gurabo_status gurabo_tensor_random_uniform_async(gurabo_stream *stream, gurabo_tensor *out,
                                                 uint64_t seed, uint64_t counter)
{
    gurabo_status status = check_initializer(stream, out);
    if (status != GURABO_OK) {
        return status;
    }
    gurabo_backend id = stream->context->backend->id;
    if (out->desc.dtype != GURABO_F32) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    uint64_t count = element_count(out);
    if (count && count - 1 > UINT64_MAX - counter) {
        return gb_fail(GURABO_OUT_OF_RANGE, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise = {
        .op = GB_POINTWISE_UNIFORM, .seed = seed, .counter = counter
    }};
    return submit_pointwise(stream, out, NULL, NULL, &kernel);
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
    if (op < GURABO_ADD || op > GURABO_DIV) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!gb_floating_dtype(out->desc.dtype) ||
        out->desc.dtype != a->desc.dtype || out->desc.dtype != b->desc.dtype ||
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
    if (op < GURABO_SUM || op > GURABO_MAX || !rank || !axis_mask ||
        axis_mask >= (1u << rank) || keep_dims > 1) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (out->desc.dtype != GURABO_F32 ||
        !gb_floating_dtype(input->desc.dtype) ||
        !out->contiguous || !input->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    uint32_t out_axis = 0;
    int reduced_empty = 0;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        uint64_t extent = input->desc.extent[axis];
        int reduced = (axis_mask & (1u << axis)) != 0;
        reduced_empty |= reduced && !extent;
        if (!reduced || keep_dims) {
            if (out_axis >= out->desc.rank || out->desc.extent[out_axis] != (reduced ? 1 : extent)) {
                return gb_fail(GURABO_INVALID_ARGUMENT, id);
            }
            ++out_axis;
        }
    }
    if (out_axis != out->desc.rank || (op != GURABO_SUM && reduced_empty) || overlaps(out, input)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    uint64_t rows = element_count(out);
    uint64_t columns = rows ? element_count(input) / rows : 0;
    uint32_t all_axes = (1u << rank) - 1;
    int trailing = (axis_mask | (axis_mask - 1)) == all_axes;
    gb_kernel kernel = {.kind = GB_KERNEL_ROWWISE, .args.rowwise = {
        .out = gb_buffer_address(out->buffer, out->offset),
        .input = gb_buffer_address(input->buffer, input->offset),
        .rows = rows,
        .columns = columns,
        .rank = rank,
        .reduced_axes = trailing ? 0 : axis_mask,
        .op = op,
        .input_dtype = input->desc.dtype
    }};
    memcpy(kernel.args.rowwise.extent, input->desc.extent, sizeof(input->desc.extent));
    memcpy(kernel.args.rowwise.stride, input->desc.stride, sizeof(input->desc.stride));
    return submit(stream, out, input, NULL, &kernel);
}

static int norm_parameter_supported(const gurabo_tensor *parameter, gurabo_dtype dtype)
{
    return !parameter || (parameter->contiguous &&
                          (parameter->desc.dtype == GURABO_F32 || parameter->desc.dtype == dtype));
}

static int norm_parameter_matches(const gurabo_tensor *parameter, const gurabo_tensor *out,
                                  uint64_t columns)
{
    return !parameter || (parameter->desc.rank == 1 && parameter->desc.extent[0] == columns &&
                          !overlaps(out, parameter));
}

static gurabo_status norm_async(gurabo_stream *stream, gurabo_tensor *out,
                                 const gurabo_tensor *input, const gurabo_tensor *scale,
                                 const gurabo_tensor *bias, float epsilon, uint32_t op)
{
    if (!stream || !out || !input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || input->buffer->context != context ||
        (scale && scale->buffer->context != context) || (bias && bias->buffer->context != context)) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    if (!input->desc.rank || !(epsilon > 0.0f) || !isfinite(epsilon) ||
        !shape_equal(&out->desc, &input->desc)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!gb_floating_dtype(input->desc.dtype) || out->desc.dtype != input->desc.dtype ||
        !out->contiguous || !input->contiguous ||
        !norm_parameter_supported(scale, input->desc.dtype) ||
        !norm_parameter_supported(bias, input->desc.dtype)) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    uint64_t columns = input->desc.extent[input->desc.rank - 1];
    if (!columns || (overlaps(out, input) && out->offset != input->offset) ||
        !norm_parameter_matches(scale, out, columns) || !norm_parameter_matches(bias, out, columns)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_ROWWISE, .args.rowwise = {
        .out = gb_buffer_address(out->buffer, out->offset),
        .input = gb_buffer_address(input->buffer, input->offset),
        .scale = scale ? gb_buffer_address(scale->buffer, scale->offset) : NULL,
        .bias = bias ? gb_buffer_address(bias->buffer, bias->offset) : NULL,
        .rows = element_count(out) / columns,
        .columns = columns,
        .op = op,
        .input_dtype = input->desc.dtype,
        .scale_dtype = scale ? scale->desc.dtype : 0,
        .bias_dtype = bias ? bias->desc.dtype : 0,
        .epsilon = epsilon
    }};
    return submit_kernel(stream, out, input, scale, bias, &kernel);
}

gurabo_status gurabo_rms_norm_async(gurabo_stream *stream, gurabo_tensor *out,
                                    const gurabo_tensor *input, const gurabo_tensor *scale,
                                    float epsilon)
{
    return norm_async(stream, out, input, scale, NULL, epsilon, GB_ROW_RMS_NORM);
}

gurabo_status gurabo_layer_norm_async(gurabo_stream *stream, gurabo_tensor *out,
                                      const gurabo_tensor *input, const gurabo_tensor *scale,
                                      const gurabo_tensor *bias, float epsilon)
{
    return norm_async(stream, out, input, scale, bias, epsilon, GB_ROW_LAYER_NORM);
}

gurabo_status gurabo_activation_async(gurabo_stream *stream, gurabo_activation op,
                                      gurabo_tensor *out, const gurabo_tensor *input)
{
    if (!stream || !out || !input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || input->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    if (op < GURABO_SILU || op > GURABO_GELU || !shape_equal(&out->desc, &input->desc) ||
        (overlaps(out, input) && out->offset != input->offset)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!gb_floating_dtype(input->desc.dtype) || out->desc.dtype != input->desc.dtype ||
        !out->contiguous || !input->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    const uint32_t kernels[] = {GB_POINTWISE_SILU, GB_POINTWISE_RELU, GB_POINTWISE_GELU};
    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise.op = kernels[op - GURABO_SILU]};
    return submit_pointwise(stream, out, input, NULL, &kernel);
}

static int make_softmax_mask(gb_rowwise *args, const gurabo_tensor_desc *input,
                             const gurabo_tensor_desc *mask)
{
    if (mask->rank > input->rank) {
        return 0;
    }
    uint32_t pad = input->rank - mask->rank;
    for (uint32_t axis = pad; axis < input->rank; ++axis) {
        uint64_t n = mask->extent[axis - pad];
        if (n != 1 && n != input->extent[axis]) {
            return 0;
        }
        args->stride[axis] = n == 1 ? 0 : mask->stride[axis - pad];
    }
    return 1;
}

gurabo_status gurabo_softmax_async(gurabo_stream *stream, gurabo_tensor *out,
                                   const gurabo_tensor *input, uint32_t axis,
                                   const gurabo_tensor *keep_mask)
{
    if (!stream || !out || !input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || input->buffer->context != context ||
        (keep_mask && keep_mask->buffer->context != context)) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    if (axis >= input->desc.rank || !shape_equal(&out->desc, &input->desc) ||
        (overlaps(out, input) && out->offset != input->offset) ||
        (keep_mask && overlaps(out, keep_mask))) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (axis + 1 != input->desc.rank || !gb_floating_dtype(input->desc.dtype) ||
        out->desc.dtype != input->desc.dtype || !out->contiguous || !input->contiguous ||
        (keep_mask && (keep_mask->desc.dtype != GURABO_U8 || !keep_mask->contiguous))) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    uint64_t columns = input->desc.extent[axis];
    gb_kernel kernel = {.kind = GB_KERNEL_ROWWISE, .args.rowwise = {
        .out = gb_buffer_address(out->buffer, out->offset),
        .input = gb_buffer_address(input->buffer, input->offset),
        .mask = keep_mask ? gb_buffer_address(keep_mask->buffer, keep_mask->offset) : NULL,
        .rank = input->desc.rank,
        .rows = columns ? element_count(out) / columns : 0,
        .columns = columns,
        .op = GB_ROW_SOFTMAX,
        .input_dtype = input->desc.dtype
    }};
    memcpy(kernel.args.rowwise.extent, input->desc.extent, sizeof(input->desc.extent));
    if (keep_mask && !make_softmax_mask(&kernel.args.rowwise, &input->desc, &keep_mask->desc)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    return submit(stream, out, input, keep_mask, &kernel);
}

gurabo_status gurabo_embedding_async(gurabo_stream *stream, gurabo_tensor *out,
                                     const gurabo_tensor *table, const gurabo_tensor *indices)
{
    if (!stream || !out || !table || !indices) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || table->buffer->context != context ||
        indices->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    uint32_t rank = indices->desc.rank;
    uint64_t vocabulary = table->desc.extent[0], width = table->desc.extent[1];
    if (table->desc.rank != 2 || !vocabulary || !width || out->desc.rank != rank + 1 ||
        out->desc.extent[rank] != width ||
        memcmp(out->desc.extent, indices->desc.extent, rank * sizeof(out->desc.extent[0])) != 0 ||
        overlaps(out, table) || overlaps(out, indices)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!gb_floating_dtype(table->desc.dtype) || out->desc.dtype != table->desc.dtype ||
        (indices->desc.dtype != GURABO_I32 && indices->desc.dtype != GURABO_I64) ||
        !out->contiguous || !table->contiguous || !indices->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_EMBEDDING, .args.embedding = {
        .out = gb_buffer_address(out->buffer, out->offset),
        .table = gb_buffer_address(table->buffer, table->offset),
        .indices = gb_buffer_address(indices->buffer, indices->offset),
        .index_count = element_count(indices),
        .vocabulary = vocabulary,
        .width = width,
        .element_bytes = (uint32_t)gb_dtype_size(table->desc.dtype),
        .index_bytes = (uint32_t)gb_dtype_size(indices->desc.dtype)
    }};
    return submit(stream, out, table, indices, &kernel);
}

gurabo_status gurabo_axpby_async(gurabo_stream *stream, gurabo_tensor *out,
                                 const gurabo_tensor *input, float alpha, float beta)
{
    if (!stream || !out || !input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    gurabo_backend id = context->backend->id;
    if (out->buffer->context != context || input->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, id);
    }
    if (!shape_equal(&out->desc, &input->desc) || !isfinite(alpha) || !isfinite(beta) ||
        (overlaps(out, input) && (out->offset != input->offset || input->desc.dtype != GURABO_F32))) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (out->desc.dtype != GURABO_F32 || !gb_floating_dtype(input->desc.dtype) ||
        !out->contiguous || !input->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    gb_kernel kernel = {.kind = GB_KERNEL_POINTWISE, .args.pointwise = {
        .op = GB_POINTWISE_AXPBY, .alpha = alpha, .beta = beta
    }};
    return submit_pointwise(stream, out, input, NULL, &kernel);
}
