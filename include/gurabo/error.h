#ifndef GURABO_ERROR_H
#define GURABO_ERROR_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t gurabo_status;
enum {
    GURABO_OK = 0,
    GURABO_INVALID_ARGUMENT = 1,
    GURABO_INVALID_HANDLE = 2,
    GURABO_OUT_OF_RANGE = 3,
    GURABO_OUT_OF_MEMORY = 4,
    GURABO_BACKEND_UNAVAILABLE = 5,
    GURABO_UNSUPPORTED = 6,
    GURABO_CONTEXT_MISMATCH = 7,
    GURABO_BUSY = 8,
    GURABO_NOT_READY = 9,
    GURABO_BACKEND_ERROR = 10,
    GURABO_DEVICE_LOST = 11
};

typedef uint32_t gurabo_backend;
enum {
    GURABO_CPU = 1,
    GURABO_HIP = 2,
    GURABO_CUDA = 3
};

typedef uint32_t gurabo_error_phase;
enum {
    GURABO_ERROR_CALL = 1,
    GURABO_ERROR_EXECUTION = 2,
    GURABO_ERROR_CLEANUP = 3
};

typedef uint32_t gurabo_native_domain;
enum {
    GURABO_NATIVE_GURABO = 0,
    GURABO_NATIVE_HIP = 1,
    GURABO_NATIVE_ROCBLAS = 2,
    GURABO_NATIVE_HIPRTC = 3
};

typedef struct {
    uint32_t struct_size;
    gurabo_status status;
    gurabo_error_phase phase;
    gurabo_backend backend;
    gurabo_native_domain native_domain;
    uint32_t reserved;
    int64_t native_code;
    uint64_t submission_id;
    char message[192];
} gurabo_error_info;

#define GURABO_ERROR_INFO_SIZE ((uint32_t)sizeof(gurabo_error_info))

/* Set struct_size before querying any public output structure.
   Failed queries leave outputs unchanged. Error details are copied out. */
const char *gurabo_status_string(gurabo_status status);
gurabo_status gurabo_get_thread_error(gurabo_error_info *out);

#ifdef __cplusplus
}
#endif
#endif
