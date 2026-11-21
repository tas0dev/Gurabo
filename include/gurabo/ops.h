#ifndef GURABO_OPS_H
#define GURABO_OPS_H
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t struct_size;
    uint32_t transpose_a;
    uint32_t transpose_b;
    float alpha;
    float beta;
} gurabo_matmul_desc;

#define GURABO_MATMUL_DESC_SIZE ((uint32_t)sizeof(gurabo_matmul_desc))

/* FP32 contiguous row-major out = alpha * op(a) * op(b) + beta * out.
   Leading batch axes broadcast right-aligned. Matrices occupy the last
   two axes. Rank-2 inputs broadcast across all batch axes. The output
   contains the broadcast batch shape followed by [m, n].
   Matrix extents and the total batch count are positive and fit in INT32_MAX.
   Transpose applies to the final two axes. NULL desc selects no transposes,
   alpha = 1 and beta = 0.
   The output must not overlap an input. Success means Stream acceptance. */
gurabo_status gurabo_matmul_async(gurabo_stream *stream, gurabo_tensor *out,
                                  const gurabo_tensor *a, const gurabo_tensor *b,
                                  const gurabo_matmul_desc *desc);

/* Same shape, layout and transpose/alpha/beta contract as matmul. A and B
   have matching F32, F16 or BF16 storage. Accumulation and output are F32. */
gurabo_status gurabo_gemm_async(gurabo_stream *stream, gurabo_tensor *out,
                                const gurabo_tensor *a, const gurabo_tensor *b,
                                const gurabo_matmul_desc *desc);

/* Convert between F32 and F16/BF16 on the Stream. Both Tensors must have
   the same shape, contiguous storage and disjoint byte ranges. */
gurabo_status gurabo_tensor_cast_async(gurabo_stream *stream, gurabo_tensor *out,
                                       const gurabo_tensor *input);

/* Copy elements in logical row-major order without changing their bits.
   Tensors must match in shape and dtype and belong to the Stream Context.
   The input may be strided. The output must be contiguous.
   Empty tensors and identical contiguous views are no-ops. Other
   overlapping byte spans are rejected, including gaps in strided inputs. */
gurabo_status gurabo_tensor_copy_async(gurabo_stream *stream, gurabo_tensor *out,
                                       const gurabo_tensor *input);

/* Initialize a contiguous Tensor. Fill copies one element in the Tensor's
   storage representation from value before returning. Zero writes zero bits. */
gurabo_status gurabo_tensor_fill_async(gurabo_stream *stream, gurabo_tensor *out,
                                       const void *value);
gurabo_status gurabo_tensor_zero_async(gurabo_stream *stream, gurabo_tensor *out);

/* Uniform values in [0, 1) in a contiguous F32 Tensor. The same seed and
   counter interval produce the same bits on all backends. Counter selects
   the first element in the sequence. The interval must not wrap. */
gurabo_status gurabo_tensor_random_uniform_async(gurabo_stream *stream, gurabo_tensor *out,
                                                 uint64_t seed, uint64_t counter);

typedef uint32_t gurabo_binary_op;
enum { GURABO_ADD = 1, GURABO_MUL = 2, GURABO_SUB = 3, GURABO_DIV = 4 };

/* Contiguous F32, F16 or BF16 Tensors with matching dtypes. Inputs broadcast
   from the trailing axes. Arithmetic is F32, then rounded to the output
   dtype. An output may alias an input with the exact same element mapping. */
gurabo_status gurabo_binary_async(gurabo_stream *stream, gurabo_binary_op op,
                                  gurabo_tensor *out, const gurabo_tensor *a,
                                  const gurabo_tensor *b);

typedef uint32_t gurabo_reduction;
enum { GURABO_SUM = 1, GURABO_MEAN = 2, GURABO_MAX = 3 };

/* Sum, mean or maximum over one or more axes of a contiguous Tensor, with
   F32 accumulation and output. axis_mask has one bit per axis, starting at
   bit 0. keep_dims leaves each reduced axis with extent one. A reduced
   empty axis sums to zero and returns INVALID_ARGUMENT for mean or max.
   Max propagates NaN. */
gurabo_status gurabo_reduce_async(gurabo_stream *stream, gurabo_reduction op,
                                  gurabo_tensor *out, const gurabo_tensor *input,
                                  uint32_t axis_mask, uint32_t keep_dims);

/* Normalize each contiguous row using its F32 mean square and a positive,
   finite epsilon. Output has the input shape and dtype. Optional scale must
   be a contiguous vector over the last axis, in F32 or the input dtype.
   Exact in-place input/output is allowed. */
gurabo_status gurabo_rms_norm_async(gurabo_stream *stream, gurabo_tensor *out,
                                    const gurabo_tensor *input, const gurabo_tensor *scale,
                                    float epsilon);

/* Normalize the last axis using its mean and population variance in F32.
   Scale and bias are independently optional contiguous vectors over that
   axis, in F32 or the input dtype. The output matches the input shape and
   dtype. Exact in-place input/output is allowed. Epsilon is finite and positive. */
gurabo_status gurabo_layer_norm_async(gurabo_stream *stream, gurabo_tensor *out,
                                      const gurabo_tensor *input, const gurabo_tensor *scale,
                                      const gurabo_tensor *bias, float epsilon);

typedef uint32_t gurabo_activation;
enum { GURABO_SILU = 1, GURABO_RELU = 2, GURABO_GELU = 3 };

/* F32 arithmetic on matching contiguous floating-point Tensors. GELU uses
   the erf definition. Exact in-place input/output is allowed. */
gurabo_status gurabo_activation_async(gurabo_stream *stream, gurabo_activation op,
                                      gurabo_tensor *out, const gurabo_tensor *input);

/* Stable F32 softmax over the last axis. The optional contiguous U8 mask
   broadcasts from the trailing axes. Masked elements and rows with only
   masked or negative-infinity entries produce zero. Positive infinities
   share probability equally. An unmasked NaN propagates to unmasked
   entries. Output matches the input shape and dtype. Exact in-place is OK. */
gurabo_status gurabo_softmax_async(gurabo_stream *stream, gurabo_tensor *out,
                                   const gurabo_tensor *input, uint32_t axis,
                                   const gurabo_tensor *keep_mask);

/* Gather rows from a contiguous [vocabulary, width] floating-point table
   using contiguous signed I32 or I64 indices. Output shape is the index
   shape followed by width. Invalid indices write zero rows and report
   OUT_OF_RANGE as a Stream execution error. */
gurabo_status gurabo_embedding_async(gurabo_stream *stream, gurabo_tensor *out,
                                     const gurabo_tensor *table, const gurabo_tensor *indices);

/* Compute out = alpha * input + beta * out in F32. Input may use F32, F16 or
   BF16 storage. Both tensors have the same shape and contiguous storage.
   A zero coefficient skips the corresponding read. Coefficients are
   finite. The ranges are disjoint or have identical F32 element mapping. */
gurabo_status gurabo_axpby_async(gurabo_stream *stream, gurabo_tensor *out,
                                 const gurabo_tensor *input, float alpha, float beta);

#ifdef __cplusplus
}
#endif
#endif
