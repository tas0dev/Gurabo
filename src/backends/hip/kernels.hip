using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
using int32_t = int;
using int64_t = long long;

extern "C" __device__ float __ocml_erf_f32(float);

#define GB_KERNEL_INLINE static __device__ __forceinline__
#define GB_KERNEL_MEMCPY __builtin_memcpy
#define GB_KERNEL_EXP __builtin_expf
#define GB_KERNEL_ERF __ocml_erf_f32
#include "pointwise.h"

extern "C" __global__ void gurabo_pointwise(gb_pointwise args)
{
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; index < args.count; index += stride) {
        gb_pointwise_apply(&args, index);
    }
}

static __device__ float block_reduce(float *partial, float value, bool max)
{
    unsigned tid = threadIdx.x;
    partial[tid] = value;
    __syncthreads();
    for (unsigned step = blockDim.x / 2; step > 0; step /= 2) {
        if (tid < step) {
            float other = partial[tid + step];
            partial[tid] = max ? gb_max_float(partial[tid], other) : partial[tid] + other;
        }
        __syncthreads();
    }
    float result = partial[0];
    __syncthreads();
    return result;
}

static __device__ void softmax_row(const gb_rowwise *args, uint64_t row, float *partial)
{
    uint64_t start = row * args->columns;
    uint64_t mask = gb_softmax_mask_base(args, row);
    float maximum = gb_float_from_bits(0xFF800000u);
    for (uint64_t col = threadIdx.x; col < args->columns; col += blockDim.x) {
        if (gb_softmax_keep(args, mask, col)) {
            maximum = gb_max_float(maximum, gb_load_float(args->input, start + col, args->input_dtype));
        }
    }
    maximum = block_reduce(partial, maximum, true);
    float sum = 0.0f;
    for (uint64_t col = threadIdx.x; col < args->columns; col += blockDim.x) {
        if (gb_softmax_keep(args, mask, col)) {
            sum += gb_softmax_term(gb_load_float(args->input, start + col, args->input_dtype), maximum);
        }
    }
    sum = block_reduce(partial, sum, false);
    for (uint64_t col = threadIdx.x; col < args->columns; col += blockDim.x) {
        float value = 0.0f;
        if (sum != 0.0f && gb_softmax_keep(args, mask, col)) {
            value = gb_softmax_term(gb_load_float(args->input, start + col, args->input_dtype),
                                    maximum) / sum;
        }
        gb_store_float(args->out, start + col, args->input_dtype, value);
    }
}

extern "C" __global__ void gurabo_rowwise(gb_rowwise args)
{
    __shared__ float partial[GB_ROW_THREADS];
    unsigned tid = threadIdx.x;
    bool rms = args.op == GB_ROW_RMS_NORM;
    bool max = args.op == GB_ROW_MAX;
    for (uint64_t row = blockIdx.x; row < args.rows; row += gridDim.x) {
        if (args.op == GB_ROW_SOFTMAX) {
            softmax_row(&args, row, partial);
            continue;
        }
        uint64_t start = row * args.columns;
        float sum = max ? gb_float_from_bits(0xFF800000u) : 0.0f;
        for (uint64_t col = tid; col < args.columns; col += blockDim.x) {
            float value = gb_load_float(args.input, start + col, args.input_dtype);
            sum = max ? gb_max_float(sum, value) : sum + (rms ? value * value : value);
        }
        sum = block_reduce(partial, sum, max);
        if (rms) {
            float denominator = __builtin_sqrtf(sum / (float)args.columns + args.epsilon);
            for (uint64_t col = tid; col < args.columns; col += blockDim.x) {
                gb_rms_write(&args, start, col, denominator);
            }
        } else if (tid == 0) {
            float value = args.op == GB_ROW_MEAN ? sum / (float)args.columns : sum;
            gb_store_float(args.out, row, GB_POINTWISE_F32, value);
        }
    }
}

extern "C" __global__ void gurabo_embedding(gb_embedding args)
{
    unsigned char *out = (unsigned char *)args.out;
    const unsigned char *table = (const unsigned char *)args.table;
    for (uint64_t row = blockIdx.x; row < args.index_count; row += gridDim.x) {
        int64_t index = gb_embedding_index(&args, row);
        bool valid = index >= 0 && (uint64_t)index < args.vocabulary;
        if (!valid && threadIdx.x == 0) {
            atomicExch(args.invalid_index, 1u);
        }
        for (uint64_t col = threadIdx.x; col < args.width; col += blockDim.x) {
            uint64_t dst = (row * args.width + col) * args.element_bytes;
            uint64_t src = valid ? ((uint64_t)index * args.width + col) * args.element_bytes : 0;
            if (args.element_bytes == sizeof(uint32_t)) {
                uint32_t bits = 0;
                if (valid) __builtin_memcpy(&bits, table + src, sizeof(bits));
                __builtin_memcpy(out + dst, &bits, sizeof(bits));
            } else {
                uint16_t bits = 0;
                if (valid) __builtin_memcpy(&bits, table + src, sizeof(bits));
                __builtin_memcpy(out + dst, &bits, sizeof(bits));
            }
        }
    }
}
