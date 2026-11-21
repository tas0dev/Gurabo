#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"

static gb_result cpu_count(uint32_t *count)
{
    *count = 1;
    return gb_status(GURABO_OK);
}

static gb_result cpu_info(uint32_t ordinal, gurabo_device_info *info)
{
    (void)ordinal;
    memcpy(info->name, "Gurabo CPU", sizeof("Gurabo CPU"));
    memcpy(info->architecture, "host", sizeof("host"));
    return gb_status(GURABO_OK);
}

static gb_result cpu_alloc(uint32_t ordinal, gb_buffer_kind kind, size_t bytes, size_t alignment,
                           gb_allocation *allocation)
{
    (void)ordinal;
    (void)kind;
    size_t extra = alignment - 1;
    if (bytes > (size_t)PTRDIFF_MAX || extra > (size_t)PTRDIFF_MAX - bytes) {
        return gb_status(GURABO_UNSUPPORTED);
    }
    void *base = malloc(bytes + extra);
    if (!base) {
        return gb_status(GURABO_OUT_OF_MEMORY);
    }
    size_t offset = (size_t)(-(uintptr_t)base & (uintptr_t)extra);
    allocation->base = base;
    allocation->data = (unsigned char *)base + offset;
    return gb_status(GURABO_OK);
}

static gb_result cpu_free(uint32_t ordinal, gb_buffer_kind kind, gb_allocation allocation)
{
    (void)ordinal;
    (void)kind;
    free(allocation.base);
    return gb_status(GURABO_OK);
}

static gb_result cpu_stream_create(uint32_t ordinal, void **out)
{
    (void)ordinal;
    *out = NULL;
    return gb_status(GURABO_OK);
}

static gb_result cpu_stream_noop(void *stream)
{
    (void)stream;
    return gb_status(GURABO_OK);
}

static gb_result cpu_copy(void *stream, void *dst, const void *src, size_t bytes,
                          gb_copy_direction direction)
{
    (void)stream;
    (void)direction;
    memcpy(dst, src, bytes);
    return gb_status(GURABO_OK);
}

static gb_result cpu_transfer(void *stream, void *dst, const void *src, size_t bytes,
                              int to_device, int *unsafe)
{
    *unsafe = 0;
    return cpu_copy(stream, dst, src, bytes,
                    to_device ? GB_COPY_HOST_TO_DEVICE : GB_COPY_DEVICE_TO_HOST);
}

static float load_f32(const void *buffer, size_t index)
{
    float value;
    /* Buffer transfers may have initialized the data through a byte array. */
    memcpy(&value, (const unsigned char *)buffer + index * sizeof(value), sizeof(value));
    return value;
}

static float load_f16(const void *buffer, size_t index, gurabo_dtype dtype)
{
    uint16_t bits;
    memcpy(&bits, (const unsigned char *)buffer + index * sizeof(bits), sizeof(bits));
    return dtype == GURABO_F16 ? gurabo_f16_to_f32(bits) : gurabo_bf16_to_f32(bits);
}

static void cpu_matmul_matrix(const gb_matmul *args)
{
    size_t m = (size_t)args->m;
    size_t n = (size_t)args->n;
    size_t k = (size_t)args->k;
    size_t a_width = (size_t)args->a_width;
    size_t b_width = (size_t)args->b_width;
    size_t a_step = args->transpose_a ? a_width : 1;
    size_t b_step = args->transpose_b ? 1 : b_width;

    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            float sum = 0.0f;
            if (args->alpha != 0.0f) {
                size_t a_index = args->transpose_a ? row : row * a_width;
                size_t b_index = args->transpose_b ? column * b_width : column;
                if (args->input_dtype == GURABO_F32) {
                    for (size_t i = 0; i < k; ++i, a_index += a_step, b_index += b_step) {
                        sum += load_f32(args->a, a_index) * load_f32(args->b, b_index);
                    }
                } else {
                    for (size_t i = 0; i < k; ++i, a_index += a_step, b_index += b_step) {
                        sum += load_f16(args->a, a_index, args->input_dtype) *
                               load_f16(args->b, b_index, args->input_dtype);
                    }
                }
            }
            size_t index = row * n + column;
            float value = args->alpha * sum;
            if (args->beta != 0.0f) {
                value += args->beta * load_f32(args->out, index);
            }
            memcpy((unsigned char *)args->out + index * sizeof(value), &value, sizeof(value));
        }
    }
}

static gb_result cpu_matmul(void *stream, const gb_matmul *args)
{
    (void)stream;
    size_t input_size = args->input_dtype == GURABO_F32 ? sizeof(float) : sizeof(uint16_t);
    for (size_t batch = 0; batch < (size_t)args->batches; ++batch) {
        gb_matmul matrix = *args;
        matrix.a = (const unsigned char *)args->a + batch * (size_t)args->stride_a * input_size;
        matrix.b = (const unsigned char *)args->b + batch * (size_t)args->stride_b * input_size;
        matrix.out = args->out + batch * (size_t)args->stride_out;
        cpu_matmul_matrix(&matrix);
    }
    return gb_status(GURABO_OK);
}

static void cpu_softmax_row(const gb_rowwise *args, uint64_t row)
{
    uint64_t start = row * args->columns;
    uint64_t mask = gb_softmax_mask_base(args, row);
    float maximum = gb_float_from_bits(0xFF800000u);
    for (uint64_t col = 0; col < args->columns; ++col) {
        if (gb_softmax_keep(args, mask, col)) {
            maximum = gb_max_float(maximum, gb_load_float(args->input, start + col, args->input_dtype));
        }
    }
    float sum = 0.0f;
    for (uint64_t col = 0; col < args->columns; ++col) {
        if (gb_softmax_keep(args, mask, col)) {
            sum += gb_softmax_term(gb_load_float(args->input, start + col, args->input_dtype), maximum);
        }
    }
    for (uint64_t col = 0; col < args->columns; ++col) {
        float value = 0.0f;
        if (sum != 0.0f && gb_softmax_keep(args, mask, col)) {
            value = gb_softmax_term(gb_load_float(args->input, start + col, args->input_dtype),
                                    maximum) / sum;
        }
        gb_store_float(args->out, start + col, args->input_dtype, value);
    }
}

static void cpu_rowwise(const gb_rowwise *args)
{
    int rms = args->op == GB_ROW_RMS_NORM;
    int max = args->op == GB_ROW_MAX;
    for (uint64_t row = 0; row < args->rows; ++row) {
        if (args->op == GB_ROW_SOFTMAX) {
            cpu_softmax_row(args, row);
            continue;
        }
        uint64_t start = row * args->columns;
        float sum = max ? gb_float_from_bits(0xFF800000u) : 0.0f;
        for (uint64_t col = 0; col < args->columns; ++col) {
            float value = gb_load_float(args->input, start + col, args->input_dtype);
            sum = max ? gb_max_float(sum, value) : sum + (rms ? value * value : value);
        }
        if (rms) {
            float denominator = sqrtf(sum / (float)args->columns + args->epsilon);
            for (uint64_t col = 0; col < args->columns; ++col) {
                gb_rms_write(args, start, col, denominator);
            }
        } else {
            float value = args->op == GB_ROW_MEAN ? sum / (float)args->columns : sum;
            gb_store_float(args->out, row, GB_POINTWISE_F32, value);
        }
    }
}

static gb_result cpu_embedding(const gb_embedding *args)
{
    int invalid = 0;
    size_t row_bytes = (size_t)(args->width * args->element_bytes);
    unsigned char *out = args->out;
    const unsigned char *table = args->table;
    for (uint64_t i = 0; i < args->index_count; ++i) {
        int64_t index = gb_embedding_index(args, i);
        unsigned char *row = out + i * row_bytes;
        if (index < 0 || (uint64_t)index >= args->vocabulary) {
            memset(row, 0, row_bytes);
            invalid = 1;
        } else {
            memcpy(row, table + (uint64_t)index * row_bytes, row_bytes);
        }
    }
    return gb_status(invalid ? GURABO_OUT_OF_RANGE : GURABO_OK);
}

static gb_result cpu_kernel_launch(void *stream, void *state, const gb_kernel *kernel,
                                    void **completion)
{
    (void)stream;
    (void)state;
    (void)completion;
    if (kernel->kind == GB_KERNEL_EMBEDDING) {
        return cpu_embedding(&kernel->args.embedding);
    }
    if (kernel->kind == GB_KERNEL_ROWWISE) {
        cpu_rowwise(&kernel->args.rowwise);
    } else {
        const gb_pointwise *args = &kernel->args.pointwise;
        for (uint64_t i = 0; i < args->count; ++i) {
            gb_pointwise_apply(args, i);
        }
    }
    return gb_status(GURABO_OK);
}

const gb_backend gb_cpu = {
    .id = GURABO_CPU,
    .alignment = 64,
    .features = GURABO_FEATURE_F32_MATMUL | GURABO_FEATURE_F16_GEMM | GURABO_FEATURE_BF16_GEMM |
                GURABO_FEATURE_TENSOR_CAST | GURABO_FEATURE_BINARY_ADD_MUL |
                GURABO_FEATURE_ROW_REDUCTION | GURABO_FEATURE_RMS_NORM |
                GURABO_FEATURE_ACTIVATION | GURABO_FEATURE_SOFTMAX | GURABO_FEATURE_BINARY_SUB_DIV |
                GURABO_FEATURE_EMBEDDING | GURABO_FEATURE_BATCHED_GEMM,
    .device_count = cpu_count,
    .device_info = cpu_info,
    .alloc = cpu_alloc,
    .free = cpu_free,
    .stream_create = cpu_stream_create,
    .stream_destroy = cpu_stream_noop,
    .stream_sync = cpu_stream_noop,
    .transfer = cpu_transfer,
    .copy = cpu_copy,
    .matmul = cpu_matmul,
    .kernel_launch = cpu_kernel_launch
};
