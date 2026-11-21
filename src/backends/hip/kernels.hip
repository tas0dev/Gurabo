using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;

#define GB_KERNEL_INLINE static __device__ __forceinline__
#define GB_KERNEL_MEMCPY __builtin_memcpy
#include "pointwise.h"

extern "C" __global__ void gurabo_pointwise(gb_pointwise args)
{
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; index < args.count; index += stride) {
        gb_pointwise_apply(&args, index);
    }
}

extern "C" __global__ void gurabo_rowwise(gb_rowwise args)
{
    __shared__ float partial[GB_ROW_THREADS];
    unsigned tid = threadIdx.x;
    bool rms = args.op == GB_ROW_RMS_NORM;
    for (uint64_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        uint64_t start = row * args.columns;
        float sum = 0.0f;
        for (uint64_t col = tid; col < args.columns; col += blockDim.x) {
            float value = gb_load_float(args.input, start + col, args.input_dtype);
            sum += rms ? value * value : value;
        }
        partial[tid] = sum;
        __syncthreads();
        for (unsigned step = blockDim.x / 2; step > 0; step /= 2) {
            if (tid < step) {
                partial[tid] += partial[tid + step];
            }
            __syncthreads();
        }
        if (rms) {
            float denominator = __builtin_sqrtf(partial[0] / (float)args.columns + args.epsilon);
            for (uint64_t col = tid; col < args.columns; col += blockDim.x) {
                gb_rms_write(&args, start, col, denominator);
            }
        } else if (tid == 0) {
            float value = args.op == GB_ROW_MEAN ? partial[0] / (float)args.columns : partial[0];
            gb_store_float(args.out, row, GB_POINTWISE_F32, value);
        }
        __syncthreads();
    }
}
