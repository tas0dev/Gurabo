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

/* FP32 row-major out = alpha * op(a) * op(b) + beta * out.
   NULL desc selects no transposes, alpha = 1, beta = 0. The output must not
   overlap either input. Success means the work was accepted by the Stream. */
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

typedef uint32_t gurabo_binary_op;
enum { GURABO_ADD = 1, GURABO_MUL = 2 };

/* Contiguous F32, F16 or BF16 Tensors with matching dtypes. Inputs broadcast
   from the trailing axes. Arithmetic is F32, then rounded to the output
   dtype. An output may alias an input with the exact same element mapping. */
gurabo_status gurabo_binary_async(gurabo_stream *stream, gurabo_binary_op op,
                                  gurabo_tensor *out, const gurabo_tensor *a,
                                  const gurabo_tensor *b);

typedef uint32_t gurabo_reduction;
enum { GURABO_SUM = 1, GURABO_MEAN = 2 };

/* Sum or mean over the last axis of a contiguous Tensor, with F32
   accumulation and output. axis_mask has one bit per axis, starting at bit 0.
   keep_dims leaves a size-one final axis. A sum over an empty axis is zero. */
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

#ifdef __cplusplus
}
#endif
#endif
