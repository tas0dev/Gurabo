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

static gb_result create_marker(void *stream, void **out)
{
    (void)stream;
    if (create_status == GURABO_OK) {
        *out = NULL;
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

static gb_result query_marker(void *stream, void *marker)
{
    (void)stream;
    (void)marker;
    return gb_status(GURABO_NOT_READY);
}

static gb_result destroy_marker(void *stream, void *marker)
{
    (void)stream;
    (void)marker;
    --live_markers;
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

int main(void)
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
        .copy = copy_buffer
    };
    gurabo_context *context = calloc(1, sizeof(*context));
    CHECK(context && gb_mutex_init(&context->gate) == 0);
    context->backend = &backend;

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
