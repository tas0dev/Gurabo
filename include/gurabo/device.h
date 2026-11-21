#ifndef GURABO_DEVICE_H
#define GURABO_DEVICE_H
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gurabo_context gurabo_context;

typedef struct {
    uint32_t struct_size;
    gurabo_backend backend;
    uint32_t ordinal;
    uint32_t reserved;
    uint64_t total_memory_bytes;
    char name[128];
    char architecture[32];
} gurabo_device_info;

typedef struct {
    uint32_t struct_size;
    uint32_t max_tensor_rank;
    uint64_t storage_dtype_mask;
    uint64_t feature_mask;
    uint64_t min_buffer_alignment;
} gurabo_device_caps;

#define GURABO_DEVICE_INFO_SIZE ((uint32_t)sizeof(gurabo_device_info))
#define GURABO_DEVICE_CAPS_SIZE ((uint32_t)sizeof(gurabo_device_caps))
#define GURABO_FEATURE_F32_MATMUL UINT64_C(1)

gurabo_status gurabo_backend_count(uint32_t *out);
gurabo_status gurabo_backend_at(uint32_t index, gurabo_backend *out);
gurabo_status gurabo_device_count(gurabo_backend backend, uint32_t *out);
gurabo_status gurabo_device_get_info(gurabo_backend backend, uint32_t ordinal,
                                    gurabo_device_info *out);
gurabo_status gurabo_device_get_caps(gurabo_backend backend, uint32_t ordinal,
                                    gurabo_device_caps *out);
/* The output slot starts at NULL. A Context selects one device for its lifetime. */
gurabo_status gurabo_context_create(gurabo_backend backend, uint32_t ordinal,
                                   gurabo_context **out);
/* BUSY leaves the Context live while it has public children. Otherwise the
   slot is cleared, including when teardown reports a retained error. */
gurabo_status gurabo_context_destroy(gurabo_context **context);
gurabo_status gurabo_context_synchronize(gurabo_context *context);
gurabo_status gurabo_context_get_error(gurabo_context *context, gurabo_error_info *out);

#ifdef __cplusplus
}
#endif
#endif
