#ifndef GURABO_TENSOR_H
#define GURABO_TENSOR_H
#include "memory.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gurabo_tensor gurabo_tensor;
typedef uint32_t gurabo_dtype;
enum {
    GURABO_F32 = 1, GURABO_F16 = 2, GURABO_BF16 = 3,
    GURABO_I8 = 4, GURABO_U8 = 5, GURABO_I32 = 6, GURABO_I64 = 7
};

/* Host conversion helpers for the 16-bit storage encodings. Narrowing uses
   round-to-nearest, ties-to-even, independent of the host rounding mode. */
uint16_t gurabo_f16_from_f32(float value);
float gurabo_f16_to_f32(uint16_t bits);
uint16_t gurabo_bf16_from_f32(float value);
float gurabo_bf16_to_f32(uint16_t bits);

#define GURABO_MAX_RANK 8

enum { GURABO_TENSOR_EXPLICIT_STRIDES = 1u };

typedef struct {
    uint32_t struct_size;
    gurabo_dtype dtype;
    uint32_t rank;
    uint32_t flags;
    uint64_t extent[GURABO_MAX_RANK];
    uint64_t stride[GURABO_MAX_RANK];
} gurabo_tensor_desc;

#define GURABO_TENSOR_DESC_SIZE ((uint32_t)sizeof(gurabo_tensor_desc))

/* Strides count elements. With flags == 0 the layout is row-major. The
   Tensor retains its Buffer and describes a view starting at byte_offset. */
gurabo_status gurabo_tensor_create(gurabo_buffer *buffer, uint64_t byte_offset,
                                   const gurabo_tensor_desc *desc, gurabo_tensor **out);
gurabo_status gurabo_tensor_release(gurabo_tensor **tensor);
gurabo_status gurabo_tensor_get_desc(const gurabo_tensor *tensor, gurabo_tensor_desc *out);
gurabo_status gurabo_tensor_is_contiguous(const gurabo_tensor *tensor, uint32_t *out);

/* Views retain the input Buffer. The output slot starts at NULL. Axes are
   zero-based. Transpose swaps two axes, including their strides. */
gurabo_status gurabo_tensor_transpose(const gurabo_tensor *input, uint32_t axis_a,
                                       uint32_t axis_b, gurabo_tensor **out);
/* Select length elements at start + i * step. Step is positive.
   An empty slice may begin at the end of its axis. */
gurabo_status gurabo_tensor_slice(const gurabo_tensor *input, uint32_t axis, uint64_t start,
                                   uint64_t length, uint64_t step, gurabo_tensor **out);
/* Change the shape of a contiguous Tensor without moving data. The element
   count must match. Rank zero represents one scalar and permits NULL extent. */
gurabo_status gurabo_tensor_reshape(const gurabo_tensor *input, uint32_t rank,
                                     const uint64_t *extent, gurabo_tensor **out);

#ifdef __cplusplus
}
#endif
#endif
