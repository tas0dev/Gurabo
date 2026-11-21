#ifndef GURABO_BACKEND_H
#define GURABO_BACKEND_H
#include <stddef.h>
#include <gurabo/gurabo.h>
#include "pointwise.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    gurabo_status status;
    gurabo_native_domain domain;
    int64_t code;
    const char *message;
} gb_result;

typedef struct {
    void *base;
    void *data;
} gb_allocation;

typedef enum {
    GB_BUFFER_DEVICE,
    GB_BUFFER_PINNED
} gb_buffer_kind;

typedef enum {
    GB_COPY_DEVICE_TO_DEVICE,
    GB_COPY_HOST_TO_DEVICE,
    GB_COPY_DEVICE_TO_HOST
} gb_copy_direction;

typedef struct {
    int32_t m, n, k;
    int32_t a_width, b_width;
    uint32_t transpose_a, transpose_b;
    float alpha, beta;
    gurabo_dtype input_dtype;
    const void *a;
    const void *b;
    float *out;
} gb_matmul;

typedef struct {
    gurabo_backend id;
    uint64_t alignment;
    uint64_t features;
    gb_result (*device_count)(uint32_t *);
    gb_result (*device_info)(uint32_t ordinal, gurabo_device_info *);
    gb_result (*alloc)(uint32_t ordinal, gb_buffer_kind, size_t, size_t, gb_allocation *);
    gb_result (*free)(uint32_t ordinal, gb_buffer_kind, gb_allocation);
    gb_result (*stream_create)(uint32_t ordinal, void **);
    gb_result (*stream_destroy)(void *);
    gb_result (*stream_sync)(void *);

    /* Completion markers are absent for immediate-execution backends. */
    gb_result (*marker_create)(uint32_t ordinal, void **);
    gb_result (*marker_record)(void *stream, void *);
    gb_result (*marker_query)(uint32_t ordinal, void *);
    gb_result (*marker_destroy)(uint32_t ordinal, void *);
    gb_result (*marker_sync)(uint32_t ordinal, void *);
    gb_result (*marker_wait)(void *stream, void *);

    /* Host transfer must stop using the borrowed pointer before returning.
       Set unsafe if the device allocation may still be in use after failure. */
    gb_result (*transfer)(void *stream, void *dst, const void *src,
                          size_t bytes, int to_device, int *unsafe);
    gb_result (*copy)(void *stream, void *dst, const void *src, size_t bytes,
                      gb_copy_direction);
    gb_result (*matmul)(void *stream, const gb_matmul *);
    gb_result (*pointwise_prepare)(uint32_t ordinal, void **state);
    gb_result (*pointwise_release)(uint32_t ordinal, void *state);
    gb_result (*pointwise)(void *stream, void *state, const gb_pointwise *);
} gb_backend;

static inline gb_result gb_status(gurabo_status status)
{
    gb_result result = {status, GURABO_NATIVE_GURABO, 0, NULL};
    return result;
}

#ifdef GURABO_WITH_CPU
extern const gb_backend gb_cpu;
#endif
#ifdef GURABO_WITH_HIP
extern const gb_backend gb_hip;
#endif

#ifdef __cplusplus
}
#endif
#endif
