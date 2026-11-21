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

#ifdef __cplusplus
}
#endif
#endif
