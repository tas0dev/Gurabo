#ifndef GURABO_POINTWISE_H
#define GURABO_POINTWISE_H

#define GB_POINTWISE_MAX_RANK 8

#define GB_POINTWISE_CAST 0u
#define GB_POINTWISE_ADD 1u
#define GB_POINTWISE_MUL 2u
#define GB_POINTWISE_F32 1u
#define GB_POINTWISE_F16 2u
#define GB_POINTWISE_BF16 3u
#define GB_ROW_SUM 1u
#define GB_ROW_MEAN 2u
#define GB_ROW_RMS_NORM 3u
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
} gb_pointwise;

typedef struct {
    void *out;
    const void *input;
    const void *scale;
    uint64_t rows;
    uint64_t columns;
    uint32_t op;
    uint32_t input_dtype;
    uint32_t scale_dtype;
    float epsilon;
} gb_rowwise;

typedef enum { GB_KERNEL_POINTWISE, GB_KERNEL_ROWWISE } gb_kernel_kind;

typedef struct {
    gb_kernel_kind kind;
    union {
        gb_pointwise pointwise;
        gb_rowwise rowwise;
    } args;
} gb_kernel;

#ifndef GB_KERNEL_INLINE
#define GB_KERNEL_INLINE static inline
#endif
#ifndef GB_KERNEL_MEMCPY
#include <string.h>
#define GB_KERNEL_MEMCPY memcpy
#endif

#include "float16_bits.h"

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

GB_KERNEL_INLINE void gb_pointwise_apply(const gb_pointwise *args, uint64_t index)
{
    uint64_t a_index, b_index;
    gb_pointwise_offsets(args, index, &a_index, &b_index);
    uint32_t result = gb_pointwise_load(args->a, a_index, args->input_dtype);
    if (args->op != GB_POINTWISE_CAST) {
        uint32_t b_bits = gb_pointwise_load(args->b, b_index, args->input_dtype);
        float a, b;
        GB_KERNEL_MEMCPY(&a, &result, sizeof(a));
        GB_KERNEL_MEMCPY(&b, &b_bits, sizeof(b));
        float value = args->op == GB_POINTWISE_ADD ? a + b : a * b;
        GB_KERNEL_MEMCPY(&result, &value, sizeof(result));
    }

    gb_store_float_bits(args->out, index, args->output_dtype, result);
}

#endif
