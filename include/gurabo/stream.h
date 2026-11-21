#ifndef GURABO_STREAM_H
#define GURABO_STREAM_H
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gurabo_stream gurabo_stream;
typedef struct gurabo_event gurabo_event;

/* A Stream orders submissions in FIFO order. Use synchronization before
   accessing results on the host or from a different Stream. */
gurabo_status gurabo_stream_create(gurabo_context *context, gurabo_stream **out);
/* Waits and consumes the slot even if execution or cleanup fails. */
gurabo_status gurabo_stream_destroy(gurabo_stream **stream);
gurabo_status gurabo_stream_synchronize(gurabo_stream *stream);
gurabo_status gurabo_stream_get_error(gurabo_stream *stream, gurabo_error_info *out);

/* An Event records a single completion point in a Stream in its Context.
   Query returns NOT_READY until completion. Query, synchronize and wait
   require a recorded Event. */
gurabo_status gurabo_event_create(gurabo_context *context, gurabo_event **out);
gurabo_status gurabo_event_record(gurabo_stream *stream, gurabo_event *event);
gurabo_status gurabo_event_query(const gurabo_event *event);
gurabo_status gurabo_event_synchronize(const gurabo_event *event);
/* Queue the dependency without waiting on the host. A failure at the recorded
   point is reported as an execution error on the waiting Stream. */
gurabo_status gurabo_stream_wait_event(gurabo_stream *stream, const gurabo_event *event);
/* Drops the public reference. Accepted records and waits retain the Event. */
gurabo_status gurabo_event_release(gurabo_event **event);

#ifdef __cplusplus
}
#endif
#endif
