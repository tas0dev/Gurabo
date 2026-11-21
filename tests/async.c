#include <stdio.h>
#include "core.h"

#define CHECK(condition) do {                                                    \
    if (!(condition)) {                                                          \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);           \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

static int live_allocations;
static int live_markers;
static size_t markers_created;
static gurabo_status marker_status[32];
static gurabo_status create_status;
static gurabo_status copy_status;
static gurabo_status sync_status;

static gb_result alloc_buffer(uint32_t ordinal, size_t bytes, size_t alignment,
                              gb_allocation *out)
{
    (void)ordinal;
    (void)alignment;
    out->data = malloc(bytes);
    CHECK(out->data);
    out->base = out->data;
    ++live_allocations;
    return gb_status(GURABO_OK);
}

static gb_result free_buffer(uint32_t ordinal, gb_allocation allocation)
{
    (void)ordinal;
    free(allocation.base);
    --live_allocations;
    return gb_status(GURABO_OK);
}

static gb_result create_stream(uint32_t ordinal, void **out)
{
    (void)ordinal;
    *out = NULL;
    return gb_status(GURABO_OK);
}

static gb_result destroy_stream(void *stream)
{
    (void)stream;
    return gb_status(GURABO_OK);
}

static gb_result sync_stream(void *stream)
{
    (void)stream;
    return gb_status(sync_status);
}

static gb_result create_marker(uint32_t ordinal, void **out)
{
    (void)ordinal;
    if (create_status == GURABO_OK) {
        CHECK(markers_created < sizeof(marker_status) / sizeof(marker_status[0]));
        marker_status[markers_created] = GURABO_NOT_READY;
        *out = &marker_status[markers_created++];
        ++live_markers;
    }
    return gb_status(create_status);
}

static gb_result record_marker(void *stream, void *marker)
{
    (void)stream;
    (void)marker;
    return gb_status(GURABO_OK);
}

static gb_result query_marker(uint32_t ordinal, void *marker)
{
    (void)ordinal;
    return gb_status(*(gurabo_status *)marker);
}

static gb_result destroy_marker(uint32_t ordinal, void *marker)
{
    (void)ordinal;
    (void)marker;
    --live_markers;
    return gb_status(GURABO_OK);
}

static gb_result wait_marker(void *stream, void *marker)
{
    (void)stream;
    CHECK(marker);
    return gb_status(GURABO_OK);
}

static gb_result copy_buffer(void *stream, void *dst, const void *src, size_t bytes)
{
    (void)stream;
    if (copy_status == GURABO_OK) {
        memcpy(dst, src, bytes);
    }
    gb_result result = {copy_status, GURABO_NATIVE_HIP, 42, "injected copy error"};
    return result;
}

static gurabo_context *mock_context(void)
{
    static const gb_backend backend = {
        .id = GURABO_HIP,
        .alignment = 64,
        .alloc = alloc_buffer,
        .free = free_buffer,
        .stream_create = create_stream,
        .stream_destroy = destroy_stream,
        .stream_sync = sync_stream,
        .marker_create = create_marker,
        .marker_record = record_marker,
        .marker_query = query_marker,
        .marker_destroy = destroy_marker,
        .marker_sync = query_marker,
        .marker_wait = wait_marker,
        .copy = copy_buffer
    };
    gurabo_context *context = calloc(1, sizeof(*context));
    CHECK(context && gb_mutex_init(&context->gate) == 0);
    context->backend = &backend;
    return context;
}

static void check_event_errors(void)
{
    gurabo_context *context = mock_context();
    gurabo_stream *source = NULL, *middle = NULL, *last = NULL;
    gurabo_event *first = NULL, *second = NULL;
    gurabo_buffer *buf = NULL;
    gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE};
    CHECK(gurabo_stream_create(context, &source) == GURABO_OK);
    CHECK(gurabo_stream_create(context, &middle) == GURABO_OK);
    CHECK(gurabo_stream_create(context, &last) == GURABO_OK);
    CHECK(gurabo_buffer_alloc(context, 8, 0, &buf) == GURABO_OK);
    memset(buf->allocation.data, 0, 8);
    CHECK(gurabo_event_create(context, &first) == GURABO_OK);
    CHECK(gurabo_event_create(context, &second) == GURABO_OK);

    size_t failed_copy = markers_created;
    CHECK(gurabo_buffer_copy_async(source, buf, 4, buf, 0, 4) == GURABO_OK);
    CHECK(gurabo_event_record(source, first) == GURABO_OK);
    CHECK(gurabo_event_query(first) == GURABO_NOT_READY);
    CHECK(gurabo_stream_wait_event(middle, first) == GURABO_OK);
    CHECK(gurabo_event_record(middle, second) == GURABO_OK);
    CHECK(gurabo_stream_wait_event(last, second) == GURABO_OK);
    for (size_t i = failed_copy; i < markers_created; ++i) {
        marker_status[i] = GURABO_OK;
    }
    marker_status[failed_copy] = GURABO_BACKEND_ERROR;
    CHECK(gurabo_event_release(&first) == GURABO_OK);
    CHECK(gurabo_event_release(&second) == GURABO_OK);
    CHECK(gurabo_stream_synchronize(last) == GURABO_BACKEND_ERROR);
    CHECK(gurabo_stream_get_error(last, &error) == GURABO_OK);
    CHECK(error.phase == GURABO_ERROR_EXECUTION && error.submission_id == 5);
    CHECK(gurabo_stream_destroy(&last) == GURABO_BACKEND_ERROR);
    CHECK(gurabo_stream_destroy(&middle) == GURABO_BACKEND_ERROR);
    CHECK(gurabo_stream_destroy(&source) == GURABO_BACKEND_ERROR);
    CHECK(gurabo_buffer_release(&buf) == GURABO_OK);
    CHECK(gurabo_context_destroy(&context) == GURABO_BACKEND_ERROR && !context);
    CHECK(live_allocations == 0 && live_markers == 0);
}

static void check_event_prefix(void)
{
    gurabo_context *context = mock_context();
    gurabo_stream *source = NULL;
    gurabo_event *event = NULL;
    gurabo_buffer *buf = NULL;
    CHECK(gurabo_stream_create(context, &source) == GURABO_OK);
    CHECK(gurabo_buffer_alloc(context, 8, 0, &buf) == GURABO_OK);
    CHECK(gurabo_event_create(context, &event) == GURABO_OK);

    size_t point = markers_created;
    CHECK(gurabo_event_record(source, event) == GURABO_OK);
    copy_status = GURABO_OUT_OF_MEMORY;
    CHECK(gurabo_buffer_copy_async(source, buf, 4, buf, 0, 4) == GURABO_OK);
    marker_status[point] = GURABO_OK;
    CHECK(gurabo_event_synchronize(event) == GURABO_OK);
    CHECK(live_markers == 2);
    CHECK(gurabo_stream_destroy(&source) == GURABO_OUT_OF_MEMORY);
    CHECK(gurabo_event_query(event) == GURABO_OK);
    CHECK(gurabo_event_release(&event) == GURABO_OK);
    CHECK(gurabo_buffer_release(&buf) == GURABO_OK);
    CHECK(gurabo_context_destroy(&context) == GURABO_OUT_OF_MEMORY && !context);
    CHECK(live_allocations == 0 && live_markers == 0);
    copy_status = GURABO_OK;
}

static void check_event_poll_cleanup(void)
{
    gurabo_context *context = mock_context();
    gurabo_stream *stream = NULL;
    gurabo_event *event = NULL;
    gurabo_buffer *buffer = NULL;
    CHECK(gurabo_stream_create(context, &stream) == GURABO_OK);
    CHECK(gurabo_event_create(context, &event) == GURABO_OK);
    CHECK(gurabo_buffer_alloc(context, 8, 0, &buffer) == GURABO_OK);
    memset(buffer->allocation.data, 0, 8);
    size_t point = markers_created;
    CHECK(gurabo_buffer_copy_async(stream, buffer, 4, buffer, 0, 4) == GURABO_OK);
    CHECK(gurabo_event_record(stream, event) == GURABO_OK);
    CHECK(gurabo_buffer_release(&buffer) == GURABO_OK);
    marker_status[point] = GURABO_OK;
    marker_status[point + 1] = GURABO_OK;
    CHECK(gurabo_event_query(event) == GURABO_OK);
    CHECK(live_allocations == 1);
    CHECK(gurabo_event_release(&event) == GURABO_OK);
    CHECK(gurabo_stream_destroy(&stream) == GURABO_OK);
    CHECK(gurabo_context_destroy(&context) == GURABO_OK);
    CHECK(live_allocations == 0 && live_markers == 0);
}

int main(void)
{
    check_event_errors();
    check_event_prefix();
    check_event_poll_cleanup();
    gurabo_context *context = mock_context();

    gurabo_stream *stream = NULL, *other = NULL;
    gurabo_buffer *src = NULL, *dst = NULL;
    CHECK(gurabo_stream_create(context, &stream) == GURABO_OK);
    CHECK(gurabo_stream_create(context, &other) == GURABO_OK);
    CHECK(gurabo_buffer_alloc(context, 8, 0, &src) == GURABO_OK);
    CHECK(gurabo_buffer_alloc(context, 8, 0, &dst) == GURABO_OK);
    memset(src->allocation.data, 0xA5, 8);

    create_status = GURABO_OUT_OF_MEMORY;
    CHECK(gurabo_buffer_copy_async(stream, dst, 0, src, 0, 8) == GURABO_OUT_OF_MEMORY);
    gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE};
    CHECK(gurabo_get_thread_error(&error) == GURABO_OK);
    CHECK(error.phase == GURABO_ERROR_CALL && error.submission_id == 0);
    CHECK(gurabo_stream_get_error(stream, &error) == GURABO_NOT_READY);

    create_status = GURABO_OK;
    CHECK(gurabo_buffer_copy_async(stream, dst, 0, src, 0, 8) == GURABO_OK);
    CHECK(gurabo_buffer_release(&src) == GURABO_OK);
    CHECK(live_allocations == 2 && live_markers == 1);
    CHECK(gurabo_stream_synchronize(stream) == GURABO_OK);
    CHECK(live_allocations == 1 && live_markers == 0);

    copy_status = GURABO_DEVICE_LOST;
    CHECK(gurabo_buffer_copy_async(stream, dst, 4, dst, 0, 4) == GURABO_OK);
    CHECK(gurabo_stream_get_error(stream, &error) == GURABO_OK);
    CHECK(error.phase == GURABO_ERROR_EXECUTION && error.submission_id == 2);
    CHECK(error.native_domain == GURABO_NATIVE_HIP && error.native_code == 42);
    CHECK(gurabo_context_get_error(context, &error) == GURABO_OK);
    CHECK(error.status == GURABO_DEVICE_LOST);
    CHECK(gurabo_buffer_alloc(context, 8, 0, &src) == GURABO_DEVICE_LOST && !src);
    CHECK(gurabo_buffer_copy_async(other, dst, 4, dst, 0, 4) == GURABO_DEVICE_LOST);

    CHECK(gurabo_buffer_release(&dst) == GURABO_OK);
    sync_status = GURABO_BACKEND_ERROR;
    CHECK(gurabo_stream_synchronize(stream) == GURABO_DEVICE_LOST);
    CHECK(live_allocations == 1 && live_markers == 1);
    sync_status = GURABO_OK;
    CHECK(gurabo_stream_destroy(&stream) == GURABO_DEVICE_LOST && !stream);
    CHECK(live_allocations == 0 && live_markers == 0);
    CHECK(gurabo_stream_destroy(&other) == GURABO_OK && !other);
    CHECK(gurabo_context_destroy(&context) == GURABO_DEVICE_LOST && !context);
    CHECK(gurabo_get_thread_error(&error) == GURABO_OK);
    CHECK(error.submission_id == 2 && error.native_code == 42);
    return 0;
}
