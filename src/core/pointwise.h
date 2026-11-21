#ifndef GURABO_POINTWISE_H
#define GURABO_POINTWISE_H

#define GB_POINTWISE_MAX_RANK 8

#define GB_POINTWISE_CAST 0u
#define GB_POINTWISE_ADD 1u
#define GB_POINTWISE_MUL 2u
#define GB_POINTWISE_F32 1u
#define GB_POINTWISE_F16 2u
#define GB_POINTWISE_BF16 3u

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

    unsigned char *out = (unsigned char *)args->out;
    if (args->output_dtype == GB_POINTWISE_F32) {
        GB_KERNEL_MEMCPY(out + index * sizeof(result), &result, sizeof(result));
    } else {
        uint16_t narrow = args->output_dtype == GB_POINTWISE_F16
                          ? gb_f16_from_f32_bits(result) : gb_bf16_from_f32_bits(result);
        GB_KERNEL_MEMCPY(out + index * sizeof(narrow), &narrow, sizeof(narrow));
    }
}

#endif
