#ifndef GURABO_POINTWISE_H
#define GURABO_POINTWISE_H

#define GB_POINTWISE_MAX_RANK 8

#define GB_POINTWISE_CAST 0u
#define GB_POINTWISE_ADD 1u
#define GB_POINTWISE_MUL 2u
#define GB_POINTWISE_SUB 3u
#define GB_POINTWISE_DIV 4u
#define GB_POINTWISE_SILU 5u
#define GB_POINTWISE_RELU 6u
#define GB_POINTWISE_GELU 7u
#define GB_POINTWISE_AXPBY 8u
#define GB_POINTWISE_F32 1u
#define GB_POINTWISE_F16 2u
#define GB_POINTWISE_BF16 3u
#define GB_ROW_SUM 1u
#define GB_ROW_MEAN 2u
#define GB_ROW_MAX 3u
#define GB_ROW_RMS_NORM 4u
#define GB_ROW_SOFTMAX 5u
#define GB_ROW_THREADS 256u

/* This layout is shared by the host and the compiled HIP kernel. */
typedef struct {
    void *out;
    const void *a;
    const void *b;
    uint64_t count;
    uint64_t extent[GB_POINTWISE_MAX_RANK];
    uint64_t a_stride[GB_POINTWISE_MAX_RANK];
    uint64_t b_stride[GB_POINTWISE_MAX_RANK];
    uint32_t broadcast_rank;
    uint32_t op;
    uint32_t input_dtype;
    uint32_t output_dtype;
    float alpha;
    float beta;
} gb_pointwise;

typedef struct {
    void *out;
    const void *input;
    const void *scale;
    const unsigned char *mask;
    uint64_t rows;
    uint64_t columns;
    uint64_t extent[GB_POINTWISE_MAX_RANK];
    uint64_t mask_stride[GB_POINTWISE_MAX_RANK];
    uint32_t rank;
    uint32_t op;
    uint32_t input_dtype;
    uint32_t scale_dtype;
    float epsilon;
} gb_rowwise;

typedef struct {
    void *out;
    const void *table;
    const void *indices;
    uint64_t index_count;
    uint64_t vocabulary;
    uint64_t width;
    uint32_t element_bytes;
    uint32_t index_bytes;
    unsigned *invalid_index;
} gb_embedding;

typedef struct {
    void *loss;
    void *gradient;
    const void *logits;
    const void *targets;
    unsigned *invalid_index;
    uint64_t rows;
    uint64_t columns;
    int64_t ignore_index;
    uint32_t logit_dtype;
    uint32_t index_bytes;
    float gradient_scale;
} gb_cross_entropy;

typedef enum {
    GB_KERNEL_POINTWISE, GB_KERNEL_ROWWISE, GB_KERNEL_EMBEDDING, GB_KERNEL_CROSS_ENTROPY
} gb_kernel_kind;

typedef struct {
    gb_kernel_kind kind;
    union {
        gb_pointwise pointwise;
        gb_rowwise rowwise;
        gb_embedding embedding;
        gb_cross_entropy cross_entropy;
    } args;
} gb_kernel;

#ifndef GB_KERNEL_INLINE
#define GB_KERNEL_INLINE static inline
#endif
#ifndef GB_KERNEL_MEMCPY
#include <string.h>
#define GB_KERNEL_MEMCPY memcpy
#endif
#ifndef GB_KERNEL_EXP
#include <math.h>
#define GB_KERNEL_EXP expf
#define GB_KERNEL_ERF erff
#define GB_KERNEL_LOG logf
#define GB_KERNEL_LOG1P log1pf
#endif

#include "float16_bits.h"

GB_KERNEL_INLINE int64_t gb_read_index(const void *base, uint64_t row, uint32_t index_bytes)
{
    const unsigned char *indices = (const unsigned char *)base;
    if (index_bytes == sizeof(int32_t)) {
        int32_t index;
        GB_KERNEL_MEMCPY(&index, indices + row * sizeof(index), sizeof(index));
        return index;
    }
    int64_t index;
    GB_KERNEL_MEMCPY(&index, indices + row * sizeof(index), sizeof(index));
    return index;
}

GB_KERNEL_INLINE void gb_pointwise_offsets(const gb_pointwise *args, uint64_t index,
                                            uint64_t *a, uint64_t *b)
{
    *a = index;
    *b = index;
    if (args->broadcast_rank) {
        *a = 0;
        *b = 0;
        for (uint32_t i = args->broadcast_rank; i > 0; --i) {
            uint32_t axis = i - 1;
            uint64_t coordinate = index % args->extent[axis];
            index /= args->extent[axis];
            *a += coordinate * args->a_stride[axis];
            *b += coordinate * args->b_stride[axis];
        }
    }
}

GB_KERNEL_INLINE uint32_t gb_pointwise_load(const void *base, uint64_t index, uint32_t dtype)
{
    const unsigned char *bytes = (const unsigned char *)base;
    if (dtype == GB_POINTWISE_F32) {
        uint32_t bits;
        GB_KERNEL_MEMCPY(&bits, bytes + index * sizeof(bits), sizeof(bits));
        return bits;
    }
    uint16_t bits;
    GB_KERNEL_MEMCPY(&bits, bytes + index * sizeof(bits), sizeof(bits));
    return dtype == GB_POINTWISE_F16 ? gb_f32_bits_from_f16(bits) : gb_f32_bits_from_bf16(bits);
}

GB_KERNEL_INLINE float gb_load_float(const void *base, uint64_t index, uint32_t dtype)
{
    uint32_t bits = gb_pointwise_load(base, index, dtype);
    float value;
    GB_KERNEL_MEMCPY(&value, &bits, sizeof(value));
    return value;
}

GB_KERNEL_INLINE void gb_store_float_bits(void *base, uint64_t index, uint32_t dtype,
                                          uint32_t bits)
{
    unsigned char *out = (unsigned char *)base;
    if (dtype == GB_POINTWISE_F32) {
        GB_KERNEL_MEMCPY(out + index * sizeof(bits), &bits, sizeof(bits));
    } else {
        uint16_t narrow = dtype == GB_POINTWISE_F16
                          ? gb_f16_from_f32_bits(bits) : gb_bf16_from_f32_bits(bits);
        GB_KERNEL_MEMCPY(out + index * sizeof(narrow), &narrow, sizeof(narrow));
    }
}

GB_KERNEL_INLINE void gb_store_float(void *base, uint64_t index, uint32_t dtype, float value)
{
    uint32_t bits;
    GB_KERNEL_MEMCPY(&bits, &value, sizeof(bits));
    gb_store_float_bits(base, index, dtype, bits);
}

GB_KERNEL_INLINE void gb_rms_write(const gb_rowwise *args, uint64_t row_start, uint64_t col,
                                   float denominator)
{
    float value = gb_load_float(args->input, row_start + col, args->input_dtype) / denominator;
    if (args->scale) {
        value *= gb_load_float(args->scale, col, args->scale_dtype);
    }
    gb_store_float(args->out, row_start + col, args->input_dtype, value);
}

GB_KERNEL_INLINE float gb_float_from_bits(uint32_t bits)
{
    float value;
    GB_KERNEL_MEMCPY(&value, &bits, sizeof(value));
    return value;
}

GB_KERNEL_INLINE float gb_max_float(float a, float b)
{
    /* Preserve NaNs and prefer positive zero when either zero is positive. */
    if (a != a || b != b) return a != a ? a : b;
    if (a == 0.0f && b == 0.0f) return a + b;
    return a > b ? a : b;
}

GB_KERNEL_INLINE float gb_cross_entropy_term(const gb_cross_entropy *args, uint64_t offset,
                                              float maximum)
{
    return GB_KERNEL_EXP(gb_load_float(args->logits, offset, args->logit_dtype) - maximum);
}

GB_KERNEL_INLINE void gb_cross_entropy_loss(const gb_cross_entropy *args, uint64_t row,
                                             int64_t target, float maximum, float others,
                                             float target_term)
{
    float loss = 0.0f;
    if (target >= 0) {
        uint64_t offset = row * args->columns + (uint64_t)target;
        float target_logit = gb_load_float(args->logits, offset, args->logit_dtype);
        loss = target_term == 1.0f && target_logit == maximum
             ? GB_KERNEL_LOG1P(others) : (maximum - target_logit) + GB_KERNEL_LOG(others + target_term);
    }
    gb_store_float(args->loss, row, GB_POINTWISE_F32, loss);
}

GB_KERNEL_INLINE void gb_cross_entropy_grad(const gb_cross_entropy *args, uint64_t start,
                                             uint64_t col, int64_t target, float maximum,
                                             float others, float target_term)
{
    float gradient = 0.0f;
    if (target >= 0) {
        float numerator = col == (uint64_t)target
                        ? -others : gb_cross_entropy_term(args, start + col, maximum);
        gradient = args->gradient_scale * (numerator / (others + target_term));
    }
    gb_store_float(args->gradient, start + col, GB_POINTWISE_F32, gradient);
}

GB_KERNEL_INLINE uint64_t gb_softmax_mask_base(const gb_rowwise *args, uint64_t row)
{
    uint64_t offset = 0;
    if (args->mask) {
        for (uint32_t i = args->rank - 1; i > 0; --i) {
            uint32_t axis = i - 1;
            offset += (row % args->extent[axis]) * args->mask_stride[axis];
            row /= args->extent[axis];
        }
    }
    return offset;
}

GB_KERNEL_INLINE int gb_softmax_keep(const gb_rowwise *args, uint64_t base, uint64_t col)
{
    return !args->mask || args->mask[base + col * args->mask_stride[args->rank - 1]] != 0;
}

GB_KERNEL_INLINE float gb_softmax_term(float value, float maximum)
{
    if (maximum == gb_float_from_bits(0xFF800000u)) return 0.0f;
    if (maximum == gb_float_from_bits(0x7F800000u)) return value == maximum ? 1.0f : 0.0f;
    return GB_KERNEL_EXP(value - maximum);
}

GB_KERNEL_INLINE float gb_pointwise_value(const gb_pointwise *args, uint32_t a_bits,
                                          uint64_t b_index)
{
    float a = gb_float_from_bits(a_bits);
    switch (args->op) {
    case GB_POINTWISE_SILU:
        if (a_bits == 0xFF800000u) return -0.0f;
        if (a >= 0.0f) return a / (1.0f + GB_KERNEL_EXP(-a));
        {
            float exp_a = GB_KERNEL_EXP(a);
            return (a * exp_a) / (1.0f + exp_a);
        }
    case GB_POINTWISE_RELU: return a < 0.0f ? 0.0f : a;
    case GB_POINTWISE_GELU:
        if (a_bits == 0xFF800000u) return -0.0f;
        return (0.5f * a) * (1.0f + GB_KERNEL_ERF(a * 0.7071067811865475f));
    default: {
        float b = gb_load_float(args->b, b_index, args->input_dtype);
        switch (args->op) {
        case GB_POINTWISE_ADD: return a + b;
        case GB_POINTWISE_SUB: return a - b;
        case GB_POINTWISE_MUL: return a * b;
        default: return a / b;
        }
    }
    }
}

GB_KERNEL_INLINE void gb_pointwise_apply(const gb_pointwise *args, uint64_t index)
{
    if (args->op == GB_POINTWISE_AXPBY) {
        float input = args->alpha != 0.0f
                    ? args->alpha * gb_load_float(args->a, index, args->input_dtype) : 0.0f;
        float previous = args->beta != 0.0f
                       ? args->beta * gb_load_float(args->out, index, GB_POINTWISE_F32) : 0.0f;
        gb_store_float(args->out, index, GB_POINTWISE_F32, input + previous);
        return;
    }
    uint64_t a_index, b_index;
    gb_pointwise_offsets(args, index, &a_index, &b_index);
    uint32_t result = gb_pointwise_load(args->a, a_index, args->input_dtype);
    if (args->op != GB_POINTWISE_CAST) {
        float value = gb_pointwise_value(args, result, b_index);
        GB_KERNEL_MEMCPY(&result, &value, sizeof(result));
    }

    gb_store_float_bits(args->out, index, args->output_dtype, result);
}

#endif
