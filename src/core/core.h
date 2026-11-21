#ifndef GURABO_CORE_H
#define GURABO_CORE_H
#include <stdlib.h>
#include <string.h>
#include "backend.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef SRWLOCK gb_mutex;
static inline int gb_mutex_init(gb_mutex *m) { InitializeSRWLock(m); return 0; }
static inline void gb_mutex_destroy(gb_mutex *m) { (void)m; }
static inline void gb_lock(gb_mutex *m) { AcquireSRWLockExclusive(m); }
static inline void gb_unlock(gb_mutex *m) { ReleaseSRWLockExclusive(m); }
#else
#include <pthread.h>
typedef pthread_mutex_t gb_mutex;
static inline int gb_mutex_init(gb_mutex *m) { return pthread_mutex_init(m, NULL); }
static inline void gb_mutex_destroy(gb_mutex *m) { pthread_mutex_destroy(m); }
static inline void gb_lock(gb_mutex *m) { pthread_mutex_lock(m); }
static inline void gb_unlock(gb_mutex *m) { pthread_mutex_unlock(m); }
#endif

typedef struct gb_job {
    struct gb_job *next;
    void *marker;
    uint64_t id;
    int marker_recorded;
    gurabo_event *recorded_event;
    gurabo_event *dependency;
    uint32_t count;
    gurabo_buffer *buffers[3];
} gb_job;

struct gurabo_context {
    const gb_backend *backend;
    uint32_t ordinal;
    gb_mutex gate;
    gurabo_stream *streams;
    gurabo_buffer *retired_buffers;
    uint64_t children;
    uint64_t allocated_bytes;
    uint64_t last_submission;
    gurabo_error_info error;
    int lost;
};

struct gurabo_stream {
    gurabo_context *context;
    gurabo_stream *next;
    void *native;
    gb_job *head;
    gb_job *tail;
    gurabo_buffer *faulted_transfer;
    gurabo_error_info error;
    gurabo_error_info execution_error;
    uint64_t execution_error_at;
};

struct gurabo_event {
    gurabo_context *context;
    gurabo_stream *source;
    void *marker;
    uint64_t references;
    uint64_t submission_id;
    gurabo_error_info error;
    int resolved;
    int quarantined;
};

struct gurabo_buffer {
    gurabo_context *context;
    gurabo_buffer *next_retired;
    gb_allocation allocation;
    gb_buffer_kind kind;
    uint64_t bytes;
    uint64_t references;
    int transfer_in_doubt;
    int quarantined;
};

struct gurabo_tensor {
    gurabo_buffer *buffer;
    gurabo_tensor_desc desc;
    uint64_t offset;
    uint64_t span;
    uint32_t contiguous;
};

gurabo_error_info gb_make_error(gb_result, gurabo_backend, gurabo_error_phase, uint64_t);
gurabo_status gb_report(const gurabo_error_info *);
gurabo_status gb_fail(gurabo_status, gurabo_backend);
gurabo_status gb_call_result(gb_result, gurabo_backend);
gurabo_status gb_get_error(const gurabo_error_info *, gurabo_error_info *out);
void gb_write_struct(void *out, const void *value, uint32_t requested, size_t available);

/* The context gate protects backend admission, pending jobs and all child bookkeeping. */
void gb_context_error(gurabo_context *, const gurabo_error_info *);
gurabo_status gb_stream_ready(gurabo_stream *);
gurabo_status gb_stream_sync_locked(gurabo_stream *, int *completed);
void gb_stream_reap(gurabo_stream *);
gurabo_status gb_job_begin(gurabo_stream *, gb_job **out);
void gb_job_end(gurabo_stream *, gb_job *, gurabo_buffer *const *, uint32_t, gb_result);
gb_result gb_buffer_unref(gurabo_buffer *);
void gb_buffer_unref_deferred(gurabo_buffer *);
void gb_buffer_collect(gurabo_context *);

static inline void *gb_buffer_address(const gurabo_buffer *buffer, uint64_t offset)
{
    return (unsigned char *)buffer->allocation.data + (size_t)offset;
}

static inline int gb_range(const gurabo_buffer *buffer, uint64_t offset, uint64_t bytes)
{
    return offset <= buffer->bytes && bytes <= buffer->bytes - offset;
}

static inline int gb_buffer_unavailable(const gurabo_buffer *buffer)
{
    return buffer->quarantined || buffer->transfer_in_doubt;
}

static inline int gb_overlap(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size)
{
    return a_size && b_size && a < b + b_size && b < a + a_size;
}

#endif
