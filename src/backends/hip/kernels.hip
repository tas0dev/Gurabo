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

extern "C" __global__ void gurabo_transpose_copy(gb_pointwise args)
{
    __shared__ uint32_t tile[GB_TRANSPOSE_TILE][GB_TRANSPOSE_TILE + 1];
    const unsigned x = threadIdx.x % GB_TRANSPOSE_TILE;
    const unsigned first_y = threadIdx.x / GB_TRANSPOSE_TILE;
    const unsigned y_step = blockDim.x / GB_TRANSPOSE_TILE;
    const uint64_t rows = args.extent[0], cols = args.extent[1];
    const uint64_t col_tiles = (cols + GB_TRANSPOSE_TILE - 1) / GB_TRANSPOSE_TILE;
    const uint64_t row_tiles = (rows + GB_TRANSPOSE_TILE - 1) / GB_TRANSPOSE_TILE;
    const uint32_t *src = (const uint32_t *)args.a;
    uint32_t *dst = (uint32_t *)args.out;
    for (uint64_t block = blockIdx.x; block < row_tiles * col_tiles; block += gridDim.x) {
        uint64_t row0 = (block / col_tiles) * GB_TRANSPOSE_TILE;
        uint64_t col0 = (block % col_tiles) * GB_TRANSPOSE_TILE;
        for (unsigned y = first_y; y < GB_TRANSPOSE_TILE; y += y_step) {
            if (row0 + x < rows && col0 + y < cols) {
                tile[y][x] = src[(col0 + y) * rows + row0 + x];
            }
        }
        __syncthreads();
        for (unsigned y = first_y; y < GB_TRANSPOSE_TILE; y += y_step) {
            if (row0 + y < rows && col0 + x < cols) {
                dst[(row0 + y) * cols + col0 + x] = tile[x][y];
            }
        }
        __syncthreads();
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

static __device__ void layer_norm_row(const gb_rowwise *args, uint64_t start, float *partial)
{
    float anchor = gb_load_float(args->input, start, args->input_dtype);
    float center = 0.0f;
    for (uint64_t col = threadIdx.x; col < args->columns; col += blockDim.x) {
        center += gb_load_float(args->input, start + col, args->input_dtype) - anchor;
    }
    center = block_reduce(partial, center, false) / (float)args->columns;
    float variance = 0.0f;
    for (uint64_t col = threadIdx.x; col < args->columns; col += blockDim.x) {
        float delta = (gb_load_float(args->input, start + col, args->input_dtype) - anchor) - center;
        variance += delta * delta;
    }
    variance = block_reduce(partial, variance, false) / (float)args->columns;
    float denominator = __builtin_sqrtf(variance + args->epsilon);
    for (uint64_t col = threadIdx.x; col < args->columns; col += blockDim.x) {
        gb_norm_write(args, start, col, anchor, center, denominator);
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
        if (args.op == GB_ROW_LAYER_NORM) {
            layer_norm_row(&args, start, partial);
            continue;
        }
        if (args.reduced_axes && args.columns) {
            start = gb_reduction_offset(&args, row, ~args.reduced_axes);
        }
        float sum = max ? gb_float_from_bits(0xFF800000u) : 0.0f;
        for (uint64_t col = tid; col < args.columns; col += blockDim.x) {
            uint64_t offset = args.reduced_axes ? gb_reduction_offset(&args, col, args.reduced_axes) : col;
            float value = gb_load_float(args.input, start + offset, args.input_dtype);
            sum = max ? gb_max_float(sum, value) : sum + (rms ? value * value : value);
        }
        sum = block_reduce(partial, sum, max);
        if (rms) {
            float denominator = __builtin_sqrtf(sum / (float)args.columns + args.epsilon);
            for (uint64_t col = tid; col < args.columns; col += blockDim.x) {
                gb_norm_write(&args, start, col, 0.0f, 0.0f, denominator);
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
