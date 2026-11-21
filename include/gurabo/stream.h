#ifndef GURABO_STREAM_H
#define GURABO_STREAM_H
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gurabo_stream gurabo_stream;

/* A Stream orders submissions in FIFO order. Use synchronization before
   accessing results on the host or from a different Stream. */
gurabo_status gurabo_stream_create(gurabo_context *context, gurabo_stream **out);
/* Waits and consumes the slot even if execution or cleanup fails. */
gurabo_status gurabo_stream_destroy(gurabo_stream **stream);
gurabo_status gurabo_stream_synchronize(gurabo_stream *stream);
gurabo_status gurabo_stream_get_error(gurabo_stream *stream, gurabo_error_info *out);

#ifdef __cplusplus
}
#endif
#endif
