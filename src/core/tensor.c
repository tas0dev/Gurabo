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
        if (extent > 1 && stride == 0 && (shape->flags & GURABO_TENSOR_EXPLICIT_STRIDES)) {
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

static int tensors_overlap(const gurabo_tensor *a, const gurabo_tensor *b)
{
    return a->buffer == b->buffer && gb_overlap(a->offset, a->span, b->offset, b->span);
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
        desc->transpose_b > 1 || a_rank < 2 || b_rank < 2 || out_rank > 3 ||
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
    uint64_t a_batches = a_rank == 3 ? a->desc.extent[0] : 1;
    uint64_t b_batches = b_rank == 3 ? b->desc.extent[0] : 1;
    uint64_t batches = a_batches > b_batches ? a_batches : b_batches;
    uint64_t m = a_matrix[desc->transpose_a];
    uint64_t k = a_matrix[1 - desc->transpose_a];
    uint64_t n = b_matrix[1 - desc->transpose_b];
    if (!a_batches || !b_batches || (a_batches != b_batches && a_batches != 1 && b_batches != 1) ||
        (out_rank == 3 && out->desc.extent[0] != batches) ||
        !m || !n || !k || k != b_matrix[desc->transpose_b] ||
        m != out_matrix[0] || n != out_matrix[1] ||
        tensors_overlap(out, a) || tensors_overlap(out, b)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, backend->id);
    }
    if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX || batches > INT32_MAX ||
        !out->contiguous || !a->contiguous || !b->contiguous) {
        return gb_fail(GURABO_UNSUPPORTED, backend->id);
    }

    gb_matmul args = {
        (int32_t)m, (int32_t)n, (int32_t)k,
        (int32_t)a_matrix[1], (int32_t)b_matrix[1],
        (int32_t)batches,
        a_batches == 1 ? 0 : (int64_t)(m * k),
        b_batches == 1 ? 0 : (int64_t)(k * n),
        (int64_t)(m * n),
        desc->transpose_a, desc->transpose_b, desc->alpha, desc->beta,
        a->desc.dtype,
        gb_buffer_address(a->buffer, a->offset),
        gb_buffer_address(b->buffer, b->offset),
        gb_buffer_address(out->buffer, out->offset)
    };
    gb_lock(&context->gate);
    gurabo_status status = gb_stream_ready(stream);
    if (status == GURABO_OK && (gb_buffer_unavailable(out->buffer) ||
                               gb_buffer_unavailable(a->buffer) ||
                               gb_buffer_unavailable(b->buffer))) {
        status = gb_fail(GURABO_BACKEND_ERROR, backend->id);
    }
    if (status == GURABO_OK) {
        gb_job *job;
        status = gb_job_begin(stream, &job);
        if (status == GURABO_OK) {
            gb_result result = backend->matmul(stream->native, &args);
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
