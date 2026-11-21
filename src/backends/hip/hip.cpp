#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <rocblas/rocblas.h>
#include "backend.h"
#include "gurabo_hip_kernels.h"

namespace {

struct hip_stream {
    uint32_t ordinal;
    hipStream_t stream = nullptr;
    rocblas_handle blas = nullptr;
    void *staging = nullptr;
    size_t staging_size = 0;
};

gb_result hip_result(hipError_t error)
{
    gurabo_status status;
    switch (error) {
    case hipSuccess: return gb_status(GURABO_OK);
    case hipErrorOutOfMemory: status = GURABO_OUT_OF_MEMORY; break;
    case hipErrorNotReady: status = GURABO_NOT_READY; break;
    case hipErrorNotSupported: status = GURABO_UNSUPPORTED; break;
    case hipErrorInsufficientDriver:
    case hipErrorInitializationError: status = GURABO_BACKEND_UNAVAILABLE; break;
    case hipErrorContextIsDestroyed: status = GURABO_DEVICE_LOST; break;
    default: status = GURABO_BACKEND_ERROR; break;
    }
    return {status, GURABO_NATIVE_HIP, error, hipGetErrorString(error)};
}

gb_result blas_result(rocblas_status error)
{
    gurabo_status status;
    switch (error) {
    case rocblas_status_success: return gb_status(GURABO_OK);
    case rocblas_status_memory_error: status = GURABO_OUT_OF_MEMORY; break;
    case rocblas_status_not_implemented:
    case rocblas_status_excluded_from_build:
    case rocblas_status_arch_mismatch: status = GURABO_UNSUPPORTED; break;
    default: status = GURABO_BACKEND_ERROR; break;
    }
    return {status, GURABO_NATIVE_ROCBLAS, error, rocblas_status_to_string(error)};
}

class device_scope {
public:
    explicit device_scope(uint32_t ordinal)
    {
        error = hipGetDevice(&previous);
        if (error == hipSuccess && previous != static_cast<int>(ordinal)) {
            error = hipSetDevice(static_cast<int>(ordinal));
            changed = error == hipSuccess;
        }
    }

    ~device_scope()
    {
        if (changed) {
            (void)hipSetDevice(previous);
        }
    }

    device_scope(const device_scope &) = delete;
    device_scope &operator=(const device_scope &) = delete;
    hipError_t error;

private:
    int previous = 0;
    bool changed = false;
};

struct pointwise_module {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};

gb_result hiprtc_result(hiprtcResult error, hiprtcProgram program)
{
    const char *message = hiprtcGetErrorString(error);
    if (error == HIPRTC_ERROR_COMPILATION) {
        size_t bytes = 0;
        if (hiprtcGetProgramLogSize(program, &bytes) == HIPRTC_SUCCESS && bytes > 1) {
            std::unique_ptr<char[]> log(new (std::nothrow) char[bytes]);
            if (log && hiprtcGetProgramLog(program, log.get()) == HIPRTC_SUCCESS) {
                static thread_local char detail[192];
                std::snprintf(detail, sizeof(detail), "%s: %.140s", message, log.get());
                message = detail;
            }
        }
    }
    gurabo_status status = error == HIPRTC_ERROR_OUT_OF_MEMORY
                           ? GURABO_OUT_OF_MEMORY : GURABO_BACKEND_ERROR;
    return {status, GURABO_NATIVE_HIPRTC, error, message};
}

gb_result compile_pointwise(const char *architecture, std::unique_ptr<char[]> &code)
{
    const char *headers[] = {gurabo_pointwise_header, gurabo_float16_header};
    const char *names[] = {"pointwise.h", "float16_bits.h"};
    hiprtcProgram program = nullptr;
    hiprtcResult error = hiprtcCreateProgram(&program, gurabo_hip_source, "gurabo_kernels.hip",
                                             2, headers, names);
    if (error != HIPRTC_SUCCESS) {
        return hiprtc_result(error, program);
    }
    char target[288];
    std::snprintf(target, sizeof(target), "--gpu-architecture=%s", architecture);
    const char *options[] = {target, "-O3"};
    error = hiprtcCompileProgram(program, 2, options);
    if (error == HIPRTC_SUCCESS) {
        size_t bytes = 0;
        error = hiprtcGetCodeSize(program, &bytes);
        if (error == HIPRTC_SUCCESS) {
            code.reset(new (std::nothrow) char[bytes]);
            error = code ? hiprtcGetCode(program, code.get()) : HIPRTC_ERROR_OUT_OF_MEMORY;
        }
    }
    gb_result result = error == HIPRTC_SUCCESS ? gb_status(GURABO_OK)
                                               : hiprtc_result(error, program);
    hiprtcDestroyProgram(&program);
    return result;
}

gb_result pointwise_prepare(uint32_t ordinal, void **state)
{
    device_scope device(ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    hipDeviceProp_t properties{};
    hipError_t error = hipGetDeviceProperties(&properties, static_cast<int>(ordinal));
    if (error != hipSuccess) {
        return hip_result(error);
    }
    auto module = std::unique_ptr<pointwise_module>(new (std::nothrow) pointwise_module);
    if (!module) {
        return gb_status(GURABO_OUT_OF_MEMORY);
    }
    std::unique_ptr<char[]> code;
    gb_result result = compile_pointwise(properties.gcnArchName, code);
    if (result.status != GURABO_OK) {
        return result;
    }
    error = hipModuleLoadData(&module->module, code.get());
    if (error == hipSuccess) {
        error = hipModuleGetFunction(&module->function, module->module, "gurabo_pointwise");
        if (error != hipSuccess) {
            (void)hipModuleUnload(module->module);
        }
    }
    if (error == hipSuccess) {
        *state = module.release();
    }
    return hip_result(error);
}

gb_result pointwise_release(uint32_t ordinal, void *state)
{
    auto module = std::unique_ptr<pointwise_module>(static_cast<pointwise_module *>(state));
    device_scope device(ordinal);
    return hip_result(device.error == hipSuccess ? hipModuleUnload(module->module) : device.error);
}

gb_result pointwise(void *stream, void *state, const gb_pointwise *args)
{
    auto *native = static_cast<hip_stream *>(stream);
    auto *module = static_cast<pointwise_module *>(state);
    device_scope device(native->ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    constexpr unsigned threads = 256;
    constexpr unsigned max_blocks = 65535;
    uint64_t blocks = (args->count + threads - 1) / threads;
    unsigned grid = blocks < max_blocks ? static_cast<unsigned>(blocks) : max_blocks;
    gb_pointwise parameters = *args;
    void *kernel_args[] = {&parameters};
    return hip_result(hipModuleLaunchKernel(module->function, grid, 1, 1, threads, 1, 1,
                                           0, native->stream, kernel_args, nullptr));
}

gb_result count_devices(uint32_t *out)
{
    int count = 0;
    hipError_t error = hipGetDeviceCount(&count);
    if (error == hipErrorNoDevice) {
        count = 0;
    } else if (error != hipSuccess) {
        return hip_result(error);
    }
    *out = static_cast<uint32_t>(count);
    return gb_status(GURABO_OK);
}

gb_result device_info(uint32_t ordinal, gurabo_device_info *info)
{
    hipDeviceProp_t properties{};
    hipError_t error = hipGetDeviceProperties(&properties, static_cast<int>(ordinal));
    if (error != hipSuccess) {
        return hip_result(error);
    }
    info->total_memory_bytes = properties.totalGlobalMem;
    std::snprintf(info->name, sizeof(info->name), "%.127s", properties.name);
    std::snprintf(info->architecture, sizeof(info->architecture), "%.31s",
                  properties.gcnArchName);
    return gb_status(GURABO_OK);
}

gb_result alloc(uint32_t ordinal, gb_buffer_kind kind, size_t bytes, size_t alignment,
                gb_allocation *out)
{
    const size_t extra = alignment - 1;
    if (bytes > static_cast<size_t>(PTRDIFF_MAX) ||
        extra > static_cast<size_t>(PTRDIFF_MAX) - bytes) {
        return gb_status(GURABO_UNSUPPORTED);
    }
    device_scope device(ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    void *base = nullptr;
    hipError_t error = kind == GB_BUFFER_PINNED
                       ? hipHostMalloc(&base, bytes + extra, hipHostMallocDefault)
                       : hipMalloc(&base, bytes + extra);
    if (error == hipSuccess) {
        uintptr_t address = reinterpret_cast<uintptr_t>(base);
        out->base = base;
        out->data = reinterpret_cast<void *>(address + (-address & extra));
    }
    return hip_result(error);
}

gb_result release(uint32_t ordinal, gb_buffer_kind kind, gb_allocation allocation)
{
    device_scope device(ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    return hip_result(kind == GB_BUFFER_PINNED ? hipFreeHost(allocation.base)
                                               : hipFree(allocation.base));
}

gb_result stream_create(uint32_t ordinal, void **out)
{
    device_scope device(ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    auto *native = new (std::nothrow) hip_stream{ordinal};
    if (!native) {
        return gb_status(GURABO_OUT_OF_MEMORY);
    }
    hipError_t hip_status = hipStreamCreateWithFlags(&native->stream, hipStreamNonBlocking);
    if (hip_status != hipSuccess) {
        delete native;
        return hip_result(hip_status);
    }
    rocblas_status blas_status = rocblas_create_handle(&native->blas);
    if (blas_status == rocblas_status_success) {
        blas_status = rocblas_set_stream(native->blas, native->stream);
    }
    if (blas_status == rocblas_status_success) {
        blas_status = rocblas_set_pointer_mode(native->blas, rocblas_pointer_mode_host);
    }
    if (blas_status != rocblas_status_success) {
        if (native->blas) {
            (void)rocblas_destroy_handle(native->blas);
        }
        (void)hipStreamDestroy(native->stream);
        delete native;
        return blas_result(blas_status);
    }
    *out = native;
    return gb_status(GURABO_OK);
}

gb_result stream_destroy(void *stream)
{
    auto *native = static_cast<hip_stream *>(stream);
    device_scope device(native->ordinal);
    if (device.error != hipSuccess) {
        delete native;
        return hip_result(device.error);
    }
    rocblas_status blas_status = rocblas_destroy_handle(native->blas);
    hipError_t hip_status = hipStreamDestroy(native->stream);
    if (native->staging) {
        hipError_t cleanup = hipFreeHost(native->staging);
        if (hip_status == hipSuccess) {
            hip_status = cleanup;
        }
    }
    delete native;
    return blas_status == rocblas_status_success ? hip_result(hip_status) : blas_result(blas_status);
}

gb_result stream_sync(void *stream)
{
    auto *native = static_cast<hip_stream *>(stream);
    device_scope device(native->ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    return hip_result(hipStreamSynchronize(native->stream));
}

gb_result marker_create(uint32_t ordinal, void **out)
{
    device_scope device(ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    hipEvent_t event = nullptr;
    hipError_t error = hipEventCreateWithFlags(&event, hipEventDisableTiming);
    if (error == hipSuccess) {
        *out = event;
    }
    return hip_result(error);
}

gb_result marker_record(void *stream, void *marker)
{
    auto *native = static_cast<hip_stream *>(stream);
    device_scope device(native->ordinal);
    return hip_result(device.error == hipSuccess
                      ? hipEventRecord(static_cast<hipEvent_t>(marker), native->stream)
                      : device.error);
}

gb_result marker_query(uint32_t ordinal, void *marker)
{
    device_scope device(ordinal);
    return hip_result(device.error == hipSuccess
                      ? hipEventQuery(static_cast<hipEvent_t>(marker)) : device.error);
}

gb_result marker_destroy(uint32_t ordinal, void *marker)
{
    device_scope device(ordinal);
    return hip_result(device.error == hipSuccess
                      ? hipEventDestroy(static_cast<hipEvent_t>(marker)) : device.error);
}

gb_result marker_sync(uint32_t ordinal, void *marker)
{
    device_scope device(ordinal);
    return hip_result(device.error == hipSuccess
                      ? hipEventSynchronize(static_cast<hipEvent_t>(marker)) : device.error);
}

gb_result marker_wait(void *stream, void *marker)
{
    auto *native = static_cast<hip_stream *>(stream);
    device_scope device(native->ordinal);
    return hip_result(device.error == hipSuccess
                      ? hipStreamWaitEvent(native->stream, static_cast<hipEvent_t>(marker), 0)
                      : device.error);
}

gb_result transfer(void *stream, void *dst, const void *src, size_t bytes,
                   int to_device, int *unsafe)
{
    auto *native = static_cast<hip_stream *>(stream);
    device_scope device(native->ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }

    /* Never submit the caller's borrowed address to the device. */
    if (bytes > native->staging_size) {
        if (native->staging) {
            hipError_t error = hipFreeHost(native->staging);
            native->staging = nullptr;
            native->staging_size = 0;
            if (error != hipSuccess) {
                return hip_result(error);
            }
        }
        void *staging = nullptr;
        hipError_t error = hipHostMalloc(&staging, bytes, hipHostMallocDefault);
        if (error != hipSuccess) {
            return hip_result(error);
        }
        native->staging = staging;
        native->staging_size = bytes;
    }
    hipError_t enqueue;
    if (to_device) {
        std::memcpy(native->staging, src, bytes);
        enqueue = hipMemcpyAsync(dst, native->staging, bytes, hipMemcpyHostToDevice,
                                 native->stream);
    } else {
        enqueue = hipMemcpyAsync(native->staging, src, bytes, hipMemcpyDeviceToHost,
                                 native->stream);
    }
    hipError_t sync = hipStreamSynchronize(native->stream);
    if (sync != hipSuccess) {
        *unsafe = 1;
        return hip_result(sync);
    }
    if (enqueue == hipSuccess && !to_device) {
        std::memcpy(dst, native->staging, bytes);
    }
    return hip_result(enqueue);
}

gb_result copy(void *stream, void *dst, const void *src, size_t bytes,
               gb_copy_direction direction)
{
    auto *native = static_cast<hip_stream *>(stream);
    device_scope device(native->ordinal);
    hipMemcpyKind kind = hipMemcpyDeviceToDevice;
    if (direction == GB_COPY_HOST_TO_DEVICE) {
        kind = hipMemcpyHostToDevice;
    } else if (direction == GB_COPY_DEVICE_TO_HOST) {
        kind = hipMemcpyDeviceToHost;
    }
    return hip_result(device.error == hipSuccess
                      ? hipMemcpyAsync(dst, src, bytes, kind, native->stream)
                      : device.error);
}

gb_result matmul(void *stream, const gb_matmul *args)
{
    auto *native = static_cast<hip_stream *>(stream);
    device_scope device(native->ordinal);
    if (device.error != hipSuccess) {
        return hip_result(device.error);
    }
    /* Row-major C equals column-major C transpose, so swap A and B. */
    rocblas_operation trans_b = args->transpose_b ? rocblas_operation_transpose
                                                  : rocblas_operation_none;
    rocblas_operation trans_a = args->transpose_a ? rocblas_operation_transpose
                                                  : rocblas_operation_none;
    if (args->input_dtype == GURABO_F32) {
        return blas_result(rocblas_sgemm(native->blas, trans_b, trans_a,
                                         args->n, args->m, args->k, &args->alpha,
                                         static_cast<const float *>(args->b), args->b_width,
                                         static_cast<const float *>(args->a), args->a_width,
                                         &args->beta, args->out, args->n));
    }
    rocblas_datatype type = args->input_dtype == GURABO_F16 ? rocblas_datatype_f16_r
                                                           : rocblas_datatype_bf16_r;
    // Use the current SDK signature directly, bypassing the legacy workspace macro.
    return blas_result((rocblas_gemm_ex)(native->blas, trans_b, trans_a,
                                         args->n, args->m, args->k, &args->alpha,
                                         args->b, type, args->b_width, args->a, type, args->a_width,
                                         &args->beta, args->out, rocblas_datatype_f32_r, args->n,
                                         args->out, rocblas_datatype_f32_r, args->n,
                                         rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
}

} // namespace

const gb_backend gb_hip = {
    GURABO_HIP,
    256,
    GURABO_FEATURE_F32_MATMUL | GURABO_FEATURE_PINNED_HOST_MEMORY |
        GURABO_FEATURE_F16_GEMM | GURABO_FEATURE_BF16_GEMM |
        GURABO_FEATURE_TENSOR_CAST | GURABO_FEATURE_BINARY_ADD_MUL,
    count_devices,
    device_info,
    alloc,
    release,
    stream_create,
    stream_destroy,
    stream_sync,
    marker_create,
    marker_record,
    marker_query,
    marker_destroy,
    marker_sync,
    marker_wait,
    transfer,
    copy,
    matmul,
    pointwise_prepare,
    pointwise_release,
    pointwise
};
