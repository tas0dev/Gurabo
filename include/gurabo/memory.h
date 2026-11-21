#ifndef GURABO_MEMORY_H
#define GURABO_MEMORY_H
#include "stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gurabo_buffer gurabo_buffer;

/* Allocates uninitialized storage. Use zero for the default alignment. */
gurabo_status gurabo_buffer_alloc(gurabo_context *context, uint64_t bytes,
                                 uint64_t alignment, gurabo_buffer **out);
/* Allocates uninitialized host storage. The CPU backend uses ordinary host
   memory. Query GURABO_FEATURE_PINNED_HOST_MEMORY for physical page locking. */
gurabo_status gurabo_pinned_buffer_alloc(gurabo_context *context, uint64_t bytes,
                                        gurabo_buffer **out);
/* Borrows a writable address until the public Buffer is released. The caller
   must synchronize before accessing bytes in conflict with a pending copy. */
gurabo_status gurabo_pinned_buffer_get_ptr(gurabo_buffer *buffer, void **out);
/* Drops the caller's reference. Tensors and accepted work retain storage. */
gurabo_status gurabo_buffer_release(gurabo_buffer **buffer);
/* Synchronous transfers between a Device Buffer and borrowed host memory. */
gurabo_status gurabo_buffer_write(gurabo_stream *stream, gurabo_buffer *dst,
                                  uint64_t offset, const void *src, uint64_t bytes);
gurabo_status gurabo_buffer_read(gurabo_stream *stream, void *dst,
                                 const gurabo_buffer *src, uint64_t offset,
                                 uint64_t bytes);
/* A same-Context copy between two Device Buffers or between a Device Buffer
   and a Pinned Buffer. Both allocations are held until the copy completes. */
gurabo_status gurabo_buffer_copy_async(gurabo_stream *stream, gurabo_buffer *dst,
                                       uint64_t dst_offset, const gurabo_buffer *src,
                                       uint64_t src_offset, uint64_t bytes);
/* Counts requested bytes held by Device and Pinned Buffers in the Context. */
gurabo_status gurabo_context_memory_usage(gurabo_context *context,
                                          uint64_t *allocated_bytes);

#ifdef __cplusplus
}
#endif
#endif
