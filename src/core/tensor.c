#include "core.h"

static gurabo_status tensor_layout(gurabo_tensor *tensor, const gurabo_tensor_desc *desc,
                                    size_t element_size)
{
    gurabo_tensor_desc *shape = &tensor->desc;
    shape->struct_size = GURABO_TENSOR_DESC_SIZE;
    shape->dtype = desc->dtype;
    shape->rank = desc->rank;
    shape->flags = desc->flags;
    int empty = 0;
    for (uint32_t axis = 0; axis < shape->rank; ++axis) {
        shape->extent[axis] = desc->extent[axis];
        empty |= desc->extent[axis] == 0;
    }

    uint64_t expected = 1;
    uint64_t last = 0;
    tensor->contiguous = 1;
    for (uint32_t i = shape->rank; i > 0; --i) {
        uint32_t axis = i - 1;
        uint64_t extent = shape->extent[axis];
        uint64_t stride = (shape->flags & GURABO_TENSOR_EXPLICIT_STRIDES)
                        ? desc->stride[axis] : expected;
        shape->stride[axis] = stride;
        if (!empty && extent > 1 && stride == 0) {
            return GURABO_INVALID_ARGUMENT;
        }
        if (empty) {
            if (extent && expected > UINT64_MAX / extent) {
                expected = 0;
            } else {
                expected *= extent;
            }
            continue;
        }
        if (extent > 1) {
            if (stride > (UINT64_MAX - last) / (extent - 1)) {
                return GURABO_OUT_OF_RANGE;
            }
            last += stride * (extent - 1);
            tensor->contiguous &= stride == expected;
        }
        if (expected > UINT64_MAX / extent) {
            return GURABO_OUT_OF_RANGE;
        }
        expected *= extent;
    }
    if (!empty) {
        if (last >= UINT64_MAX / element_size) {
            return GURABO_OUT_OF_RANGE;
        }
        tensor->span = (last + 1) * element_size;
    }
    return GURABO_OK;
}

gurabo_status gurabo_tensor_create(gurabo_buffer *buffer, uint64_t byte_offset,
                                   const gurabo_tensor_desc *desc, gurabo_tensor **out)
{
    if (!buffer) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = buffer->context;
    gurabo_backend id = context->backend->id;
    if (!out || *out || !desc || desc->struct_size < GURABO_TENSOR_DESC_SIZE ||
        desc->rank > GURABO_MAX_RANK ||
        (desc->flags & ~(uint32_t)GURABO_TENSOR_EXPLICIT_STRIDES) ||
        desc->dtype < GURABO_F32 || desc->dtype > GURABO_I64) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    size_t element_size = gb_dtype_size(desc->dtype);
    if (!element_size || buffer->kind != GB_BUFFER_DEVICE) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    if (byte_offset % element_size) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    gurabo_tensor tensor = {0};
    tensor.buffer = buffer;
    tensor.offset = byte_offset;
    gurabo_status status = tensor_layout(&tensor, desc, element_size);
    if (status != GURABO_OK) {
        return gb_fail(status, id);
    }
    if (!gb_range(buffer, byte_offset, tensor.span)) {
        return gb_fail(GURABO_OUT_OF_RANGE, id);
    }
    gurabo_tensor *value = malloc(sizeof(*value));
    if (!value) {
        return gb_fail(GURABO_OUT_OF_MEMORY, id);
    }
    *value = tensor;
    gb_lock(&context->gate);
    ++buffer->references;
    ++context->children;
    *out = value;
    gb_unlock(&context->gate);
    return GURABO_OK;
}

gurabo_status gurabo_tensor_release(gurabo_tensor **tensor)
{
    if (!tensor) {
        return gb_fail(GURABO_INVALID_ARGUMENT, 0);
    }
    gurabo_tensor *value = *tensor;
    if (!value) {
        return GURABO_OK;
    }
    gurabo_context *context = value->buffer->context;
    gb_lock(&context->gate);
    --context->children;
    *tensor = NULL;
    gb_result result = gb_buffer_unref(value->buffer);
    free(value);
    gurabo_status status = GURABO_OK;
    if (result.status != GURABO_OK) {
        gurabo_error_info error = gb_make_error(result, context->backend->id,
                                                GURABO_ERROR_CLEANUP, 0);
        status = gb_report(&error);
    }
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_tensor_get_desc(const gurabo_tensor *tensor, gurabo_tensor_desc *out)
{
    if (!tensor) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    if (!out || out->struct_size < sizeof(uint32_t)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, tensor->buffer->context->backend->id);
    }
    gb_write_struct(out, &tensor->desc, out->struct_size, sizeof(tensor->desc));
    return GURABO_OK;
}

gurabo_status gurabo_tensor_is_contiguous(const gurabo_tensor *tensor, uint32_t *out)
{
    if (!tensor) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    if (!out) {
        return gb_fail(GURABO_INVALID_ARGUMENT, tensor->buffer->context->backend->id);
    }
    *out = tensor->contiguous;
    return GURABO_OK;
}

gurabo_status gurabo_tensor_transpose(const gurabo_tensor *input, uint32_t axis_a,
                                       uint32_t axis_b, gurabo_tensor **out)
{
    if (!input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_backend id = input->buffer->context->backend->id;
    gurabo_tensor_desc desc = input->desc;
    if (axis_a >= desc.rank || axis_b >= desc.rank) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    desc.flags = GURABO_TENSOR_EXPLICIT_STRIDES;
    desc.extent[axis_a] = input->desc.extent[axis_b];
    desc.extent[axis_b] = input->desc.extent[axis_a];
    desc.stride[axis_a] = input->desc.stride[axis_b];
    desc.stride[axis_b] = input->desc.stride[axis_a];
    return gurabo_tensor_create(input->buffer, input->offset, &desc, out);
}

gurabo_status gurabo_tensor_slice(const gurabo_tensor *input, uint32_t axis, uint64_t start,
                                   uint64_t length, uint64_t step, gurabo_tensor **out)
{
    if (!input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_backend id = input->buffer->context->backend->id;
    gurabo_tensor_desc desc = input->desc;
    if (axis >= desc.rank || !step) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    uint64_t extent = desc.extent[axis];
    if (start > extent ||
        (length && (start == extent || length - 1 > (extent - 1 - start) / step))) {
        return gb_fail(GURABO_OUT_OF_RANGE, id);
    }
    uint64_t offset = input->offset;
    uint64_t stride = desc.stride[axis];
    if (input->span && length) {
        /* The selected first element already lies in the validated input span. */
        offset += start * stride * gb_dtype_size(desc.dtype);
        if (length > 1) {
            desc.stride[axis] = stride * step;
        }
    }
    desc.flags = GURABO_TENSOR_EXPLICIT_STRIDES;
    desc.extent[axis] = length;
    return gurabo_tensor_create(input->buffer, offset, &desc, out);
}

gurabo_status gurabo_tensor_reshape(const gurabo_tensor *input, uint32_t rank,
                                     const uint64_t *extent, gurabo_tensor **out)
{
    if (!input) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_backend id = input->buffer->context->backend->id;
    if (rank > GURABO_MAX_RANK || (rank && !extent)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    if (!input->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, id);
    }
    gurabo_tensor_desc desc = {.struct_size = GURABO_TENSOR_DESC_SIZE, .dtype = input->desc.dtype,
                               .rank = rank};
    for (uint32_t axis = 0; axis < rank; ++axis) {
        desc.extent[axis] = extent[axis];
    }
    size_t bytes = gb_dtype_size(desc.dtype);
    gurabo_tensor layout = {0};
    gurabo_status status = tensor_layout(&layout, &desc, bytes);
    if (status != GURABO_OK) {
        return gb_fail(status, id);
    }
    if (layout.span != input->span) {
        return gb_fail(GURABO_INVALID_ARGUMENT, id);
    }
    return gurabo_tensor_create(input->buffer, input->offset, &desc, out);
}

static int tensors_overlap(const gurabo_tensor *a, const gurabo_tensor *b)
{
    return a->buffer == b->buffer && gb_overlap(a->offset, a->span, b->offset, b->span);
}

typedef struct {
    uint64_t extent[GURABO_MAX_RANK - 2];
    uint64_t a_stride[GURABO_MAX_RANK - 2];
    uint64_t b_stride[GURABO_MAX_RANK - 2];
    uint64_t total, inner;
    uint32_t outer_rank;
} gemm_batches;

static gurabo_status gemm_batch_layout(gemm_batches *batch, gb_matmul *args,
                                       const gurabo_tensor_desc *out, const gurabo_tensor_desc *a,
                                       const gurabo_tensor_desc *b)
{
    uint32_t rank = out->rank - 2;
    uint32_t a_pad = out->rank - a->rank;
    uint32_t b_pad = out->rank - b->rank;
    batch->total = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        uint64_t an = i < a_pad ? 1 : a->extent[i - a_pad];
        uint64_t bn = i < b_pad ? 1 : b->extent[i - b_pad];
        uint64_t n = an == 1 ? bn : an;
        if (!an || !bn || (an != bn && an != 1 && bn != 1) || out->extent[i] != n) {
            return GURABO_INVALID_ARGUMENT;
        }
        if (n > INT32_MAX / batch->total) {
            return GURABO_UNSUPPORTED;
        }
        batch->extent[i] = n;
        batch->a_stride[i] = an == 1 ? 0 : a->stride[i - a_pad];
        batch->b_stride[i] = bn == 1 ? 0 : b->stride[i - b_pad];
        batch->total *= n;
    }

    /* Group the largest suffix with constant per-matrix strides. */
    uint64_t stride_a = 0, stride_b = 0;
    batch->inner = 1;
    batch->outer_rank = rank;
    while (batch->outer_rank) {
        uint32_t i = batch->outer_rank - 1;
        if (batch->extent[i] > 1) {
            uint64_t sa = batch->a_stride[i], sb = batch->b_stride[i];
            if (batch->inner == 1) {
                stride_a = sa;
                stride_b = sb;
            } else if (sa % batch->inner || sb % batch->inner ||
                       sa / batch->inner != stride_a || sb / batch->inner != stride_b) {
                break;
            }
            batch->inner *= batch->extent[i];
        }
        --batch->outer_rank;
    }
    args->batches = (int32_t)batch->inner;
    args->stride_a = (int64_t)stride_a;
    args->stride_b = (int64_t)stride_b;
    return GURABO_OK;
}

static gb_result dispatch_gemm(gurabo_stream *stream, const gb_matmul *args,
                               const gemm_batches *batch)
{
    const gb_backend *backend = stream->context->backend;
    if (!batch->outer_rank) {
        return backend->matmul(stream->native, args);
    }
    size_t input_bytes = gb_dtype_size(args->input_dtype);
    for (uint64_t group = 0; group < batch->total / batch->inner; ++group) {
        uint64_t index = group, a_offset = 0, b_offset = 0;
        for (uint32_t i = batch->outer_rank; i > 0; --i) {
            uint32_t axis = i - 1;
            uint64_t coordinate = index % batch->extent[axis];
            index /= batch->extent[axis];
            a_offset += coordinate * batch->a_stride[axis];
            b_offset += coordinate * batch->b_stride[axis];
        }
        gb_matmul matrix = *args;
        matrix.a = (const unsigned char *)args->a + a_offset * input_bytes;
        matrix.b = (const unsigned char *)args->b + b_offset * input_bytes;
        matrix.out = args->out + group * batch->inner * (uint64_t)args->stride_out;
        gb_result result = backend->matmul(stream->native, &matrix);
        if (result.status != GURABO_OK) {
            return result;
        }
    }
    return gb_status(GURABO_OK);
}

static gurabo_status submit_gemm(gurabo_stream *stream, gurabo_tensor *out,
                                 const gurabo_tensor *a, const gurabo_tensor *b,
                                 const gurabo_matmul_desc *desc, int allow_mixed)
{
    static const gurabo_matmul_desc defaults = {GURABO_MATMUL_DESC_SIZE, 0, 0, 1.0f, 0.0f};
    if (!stream || !out || !a || !b) {
        return gb_fail(GURABO_INVALID_HANDLE, 0);
    }
    gurabo_context *context = stream->context;
    const gb_backend *backend = context->backend;
    if (out->buffer->context != context || a->buffer->context != context ||
        b->buffer->context != context) {
        return gb_fail(GURABO_CONTEXT_MISMATCH, backend->id);
    }
    if (!desc) {
        desc = &defaults;
    }
    uint32_t a_rank = a->desc.rank;
    uint32_t b_rank = b->desc.rank;
    uint32_t out_rank = a_rank > b_rank ? a_rank : b_rank;
    if (desc->struct_size < GURABO_MATMUL_DESC_SIZE || desc->transpose_a > 1 ||
        desc->transpose_b > 1 || a_rank < 2 || b_rank < 2 ||
        out->desc.rank != out_rank) {
        return gb_fail(GURABO_INVALID_ARGUMENT, backend->id);
    }
    if (out->desc.dtype != GURABO_F32 || !gb_floating_dtype(a->desc.dtype) ||
        a->desc.dtype != b->desc.dtype ||
        (!allow_mixed && a->desc.dtype != GURABO_F32)) {
        return gb_fail(GURABO_UNSUPPORTED, backend->id);
    }

    const uint64_t *a_matrix = &a->desc.extent[a_rank - 2];
    const uint64_t *b_matrix = &b->desc.extent[b_rank - 2];
    const uint64_t *out_matrix = &out->desc.extent[out_rank - 2];
    uint64_t m = a_matrix[desc->transpose_a];
    uint64_t k = a_matrix[1 - desc->transpose_a];
    uint64_t n = b_matrix[1 - desc->transpose_b];
    if (!m || !n || !k || k != b_matrix[desc->transpose_b] ||
        m != out_matrix[0] || n != out_matrix[1] ||
        tensors_overlap(out, a) || tensors_overlap(out, b)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, backend->id);
    }
    if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX ||
        !out->contiguous || !a->contiguous || !b->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, backend->id);
    }

    gb_matmul args = {
        .m = (int32_t)m, .n = (int32_t)n, .k = (int32_t)k,
        .a_width = (int32_t)a_matrix[1], .b_width = (int32_t)b_matrix[1],
        .stride_out = (int64_t)(m * n),
        .transpose_a = desc->transpose_a, .transpose_b = desc->transpose_b,
        .alpha = desc->alpha, .beta = desc->beta,
        .input_dtype = a->desc.dtype,
        .a = gb_buffer_address(a->buffer, a->offset),
        .b = gb_buffer_address(b->buffer, b->offset),
        .out = gb_buffer_address(out->buffer, out->offset)
    };
    gemm_batches batch;
    gurabo_status status = gemm_batch_layout(&batch, &args, &out->desc, &a->desc, &b->desc);
    if (status != GURABO_OK) {
        return gb_fail(status, backend->id);
    }
    gb_lock(&context->gate);
    status = gb_stream_ready(stream);
    if (status == GURABO_OK && (gb_buffer_unavailable(out->buffer) ||
                               gb_buffer_unavailable(a->buffer) ||
                               gb_buffer_unavailable(b->buffer))) {
        status = gb_fail(GURABO_BACKEND_ERROR, backend->id);
    }
    if (status == GURABO_OK) {
        gb_job *job;
        status = gb_job_begin(stream, &job);
        if (status == GURABO_OK) {
            gb_result result = dispatch_gemm(stream, &args, &batch);
            gurabo_buffer *buffers[] = {out->buffer, a->buffer, b->buffer};
            gb_job_end(stream, job, buffers, 3, result);
        }
    }
    gb_unlock(&context->gate);
    return status;
}

gurabo_status gurabo_matmul_async(gurabo_stream *stream, gurabo_tensor *out,
                                  const gurabo_tensor *a, const gurabo_tensor *b,
                                  const gurabo_matmul_desc *desc)
{
    return submit_gemm(stream, out, a, b, desc, 0);
}

gurabo_status gurabo_gemm_async(gurabo_stream *stream, gurabo_tensor *out,
                                const gurabo_tensor *a, const gurabo_tensor *b,
                                const gurabo_matmul_desc *desc)
{
    return submit_gemm(stream, out, a, b, desc, 1);
}
