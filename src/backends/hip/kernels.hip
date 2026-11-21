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
