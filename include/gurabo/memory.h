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
/* Drops the caller's reference. Tensors and accepted work retain storage. */
gurabo_status gurabo_buffer_release(gurabo_buffer **buffer);
/* Synchronous transfers borrow host memory for the duration of the call. */
gurabo_status gurabo_buffer_write(gurabo_stream *stream, gurabo_buffer *dst,
                                  uint64_t offset, const void *src, uint64_t bytes);
gurabo_status gurabo_buffer_read(gurabo_stream *stream, void *dst,
                                 const gurabo_buffer *src, uint64_t offset,
                                 uint64_t bytes);
gurabo_status gurabo_buffer_copy_async(gurabo_stream *stream, gurabo_buffer *dst,
                                       uint64_t dst_offset, const gurabo_buffer *src,
                                       uint64_t src_offset, uint64_t bytes);
gurabo_status gurabo_context_memory_usage(gurabo_context *context,
                                          uint64_t *allocated_bytes);

#ifdef __cplusplus
}
#endif
#endif
