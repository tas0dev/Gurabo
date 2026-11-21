#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gurabo/error.h>
#include <gurabo/device.h>
#include <gurabo/stream.h>
#include <gurabo/memory.h>
#include <gurabo/tensor.h>
#include <gurabo/ops.h>
#include <gurabo/gurabo.h>

#define CHECK(condition) do {                                                     \
    if (!(condition)) {                                                           \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);            \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
} while (0)

#define STATUS(expression, expected) do {                                         \
    gurabo_status test_status = (expression);                                     \
    if (test_status != (expected)) {                                              \
        gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE};         \
        gurabo_get_thread_error(&error);                                          \
        fprintf(stderr, "%s:%d: %s returned %s, expected %s (%s)\n",               \
                __FILE__, __LINE__, #expression, gurabo_status_string(test_status), \
                gurabo_status_string(expected), error.message);                  \
        exit(EXIT_FAILURE);                                                       \
    }                                                                             \
} while (0)

#define OK(expression) STATUS(expression, GURABO_OK)

static gurabo_tensor_desc matrix_desc(uint64_t height, uint64_t width)
{
    gurabo_tensor_desc shape = {
        .struct_size = GURABO_TENSOR_DESC_SIZE,
        .dtype = GURABO_F32,
        .rank = 2,
        .extent = {height, width}
    };
    return shape;
}

static void check_discovery(gurabo_backend backend, uint32_t ordinal)
{
    uint32_t count = 0;
    OK(gurabo_backend_count(&count));
    gurabo_backend last = 0;
    uint32_t matches = 0;
    for (uint32_t i = 0; i < count; ++i) {
        gurabo_backend id = 0;
        OK(gurabo_backend_at(i, &id));
        CHECK(id > last);
        matches += id == backend;
        last = id;
    }
    CHECK(matches == 1);
    STATUS(gurabo_backend_at(count, &last), GURABO_OUT_OF_RANGE);

    gurabo_device_info info = {.struct_size = GURABO_DEVICE_INFO_SIZE};
    OK(gurabo_device_get_info(backend, ordinal, &info));
    CHECK(info.backend == backend && info.ordinal == ordinal && info.name[0] != '\0');
    printf("Testing %s (%s), ordinal %u\n", info.name, info.architecture, (unsigned)ordinal);
    gurabo_device_caps caps = {.struct_size = GURABO_DEVICE_CAPS_SIZE};
    OK(gurabo_device_get_caps(backend, ordinal, &caps));
    CHECK(caps.storage_dtype_mask == ((UINT64_C(1) << GURABO_F32) |
                                      (UINT64_C(1) << GURABO_F16) |
                                      (UINT64_C(1) << GURABO_BF16) |
                                      (UINT64_C(1) << GURABO_U8) |
                                      (UINT64_C(1) << GURABO_I32) |
                                      (UINT64_C(1) << GURABO_I64)));
    CHECK(caps.feature_mask & GURABO_FEATURE_F32_MATMUL);
    CHECK(caps.feature_mask & GURABO_FEATURE_F16_GEMM);
    CHECK(caps.feature_mask & GURABO_FEATURE_BF16_GEMM);
    CHECK(caps.feature_mask & GURABO_FEATURE_TENSOR_CAST);
    CHECK(caps.feature_mask & GURABO_FEATURE_BINARY_ADD_MUL);
    CHECK(caps.feature_mask & GURABO_FEATURE_ROW_REDUCTION);
    CHECK(caps.feature_mask & GURABO_FEATURE_RMS_NORM);
    CHECK(caps.feature_mask & GURABO_FEATURE_ACTIVATION);
    CHECK(caps.feature_mask & GURABO_FEATURE_SOFTMAX);
    CHECK(caps.feature_mask & GURABO_FEATURE_EMBEDDING);
    CHECK(caps.feature_mask & GURABO_FEATURE_BINARY_SUB_DIV);
    CHECK(caps.feature_mask & GURABO_FEATURE_BATCHED_GEMM);
    CHECK(caps.feature_mask & GURABO_FEATURE_AXPBY);
    CHECK(!!(caps.feature_mask & GURABO_FEATURE_PINNED_HOST_MEMORY) == (backend == GURABO_HIP));
    CHECK(caps.max_tensor_rank == GURABO_MAX_RANK && caps.min_buffer_alignment >= sizeof(float));

    info.struct_size = (uint32_t)(2 * sizeof(uint32_t));
    info.ordinal = 123;
    OK(gurabo_device_get_info(backend, ordinal, &info));
    CHECK(info.struct_size == 2 * sizeof(uint32_t) && info.ordinal == 123);
    STATUS(gurabo_device_get_info(backend, UINT32_MAX, &info), GURABO_OUT_OF_RANGE);
    CHECK(info.ordinal == 123);

    count = 123;
    STATUS(gurabo_device_count(GURABO_CUDA, &count), GURABO_BACKEND_UNAVAILABLE);
    CHECK(count == 123);
    gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE};
    OK(gurabo_get_thread_error(&error));
    CHECK(error.status == GURABO_BACKEND_UNAVAILABLE && error.phase == GURABO_ERROR_CALL);
    CHECK(error.backend == GURABO_CUDA && error.message[0] != '\0');
}

static void check_memory(gurabo_backend backend, uint32_t ordinal,
                         gurabo_context *context, gurabo_stream *stream)
{
    gurabo_context *other = NULL;
    gurabo_stream *second = NULL;
    gurabo_buffer *src = NULL, *dst = NULL, *foreign = NULL;
    const unsigned char data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned char copy[sizeof(data)] = {0};

    OK(gurabo_context_create(backend, ordinal, &other));
    OK(gurabo_stream_create(context, &second));
    OK(gurabo_buffer_alloc(other, 8, 0, &foreign));
    STATUS(gurabo_buffer_alloc(context, 16, 3, &src), GURABO_INVALID_ARGUMENT);
    CHECK(src == NULL);
    OK(gurabo_buffer_alloc(context, 16, 256, &src));
    OK(gurabo_buffer_alloc(context, 16, 0, &dst));
    gurabo_buffer *saved = src;
    STATUS(gurabo_buffer_alloc(context, 5, 0, &src), GURABO_INVALID_ARGUMENT);
    CHECK(src == saved);
    STATUS(gurabo_buffer_write(stream, src, 15, data, sizeof(data)), GURABO_OUT_OF_RANGE);
    STATUS(gurabo_buffer_read(stream, copy, src, UINT64_MAX, 1), GURABO_OUT_OF_RANGE);
    CHECK(copy[0] == 0);
    STATUS(gurabo_buffer_copy_async(stream, src, 1, src, 0, 2), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_buffer_copy_async(stream, dst, 0, foreign, 0, 8), GURABO_CONTEXT_MISMATCH);
    OK(gurabo_buffer_write(stream, src, 16, NULL, 0));
    OK(gurabo_buffer_write(stream, src, 4, data, sizeof(data)));
    OK(gurabo_buffer_copy_async(stream, dst, 3, src, 4, sizeof(data)));
    OK(gurabo_buffer_release(&src));
    CHECK(src == NULL);
    OK(gurabo_context_synchronize(context));
    OK(gurabo_buffer_read(second, copy, dst, 3, sizeof(copy)));
    CHECK(memcmp(data, copy, sizeof(data)) == 0);

    uint64_t usage = 0;
    OK(gurabo_context_memory_usage(context, &usage));
    CHECK(usage == 16);
    OK(gurabo_buffer_release(&dst));
    OK(gurabo_buffer_release(&foreign));
    OK(gurabo_context_destroy(&other));
    OK(gurabo_stream_destroy(&second));
}

static void check_layout(gurabo_context *context)
{
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *tensor = NULL;
    uint32_t contiguous = 7;
    OK(gurabo_buffer_alloc(context, 64, 0, &buffer));

    gurabo_tensor_desc shape = matrix_desc(2, 3);
    shape.flags = GURABO_TENSOR_EXPLICIT_STRIDES;
    shape.stride[0] = 5;
    shape.stride[1] = 1;
    OK(gurabo_tensor_create(buffer, sizeof(float), &shape, &tensor));
    OK(gurabo_tensor_is_contiguous(tensor, &contiguous));
    CHECK(contiguous == 0);
    gurabo_tensor_desc actual = {.struct_size = GURABO_TENSOR_DESC_SIZE};
    OK(gurabo_tensor_get_desc(tensor, &actual));
    CHECK(actual.stride[0] == 5 && actual.stride[1] == 1);
    OK(gurabo_tensor_release(&tensor));

    shape.stride[0] = UINT64_MAX;
    STATUS(gurabo_tensor_create(buffer, 0, &shape, &tensor), GURABO_OUT_OF_RANGE);
    CHECK(tensor == NULL);
    shape.stride[0] = 0;
    STATUS(gurabo_tensor_create(buffer, 0, &shape, &tensor), GURABO_INVALID_ARGUMENT);
    shape = matrix_desc(2, 0);
    OK(gurabo_tensor_create(buffer, 64, &shape, &tensor));
    OK(gurabo_tensor_is_contiguous(tensor, &contiguous));
    CHECK(contiguous == 1);
    OK(gurabo_tensor_release(&tensor));

    for (gurabo_dtype dtype = GURABO_F16; dtype <= GURABO_BF16; ++dtype) {
        gurabo_tensor_desc half = matrix_desc(1, 2);
        half.dtype = dtype;
        STATUS(gurabo_tensor_create(buffer, 59, &half, &tensor), GURABO_INVALID_ARGUMENT);
        STATUS(gurabo_tensor_create(buffer, 62, &half, &tensor), GURABO_OUT_OF_RANGE);
        OK(gurabo_tensor_create(buffer, 60, &half, &tensor));
        OK(gurabo_tensor_get_desc(tensor, &actual));
        CHECK(actual.dtype == dtype && actual.stride[1] == 1);
        OK(gurabo_tensor_release(&tensor));
    }

    shape.rank = 0;
    STATUS(gurabo_tensor_create(buffer, 64, &shape, &tensor), GURABO_OUT_OF_RANGE);
    OK(gurabo_tensor_create(buffer, 60, &shape, &tensor));
    OK(gurabo_buffer_release(&buffer));
    uint64_t usage = 0;
    OK(gurabo_context_memory_usage(context, &usage));
    CHECK(usage == 64);
    OK(gurabo_tensor_release(&tensor));
    OK(gurabo_context_memory_usage(context, &usage));
    CHECK(usage == 0);
}

static void check_events(gurabo_backend backend, uint32_t ordinal, gurabo_context *context)
{
    gurabo_stream *producer = NULL, *consumer = NULL, *foreign_stream = NULL;
    gurabo_context *foreign = NULL;
    gurabo_event *ready = NULL, *returned = NULL, *finished = NULL;
    gurabo_buffer *input = NULL, *middle = NULL, *result = NULL;
    const uint32_t expected = 0xC0FFEEu;
    uint32_t actual = 0;

    OK(gurabo_stream_create(context, &producer));
    OK(gurabo_stream_create(context, &consumer));
    OK(gurabo_context_create(backend, ordinal, &foreign));
    OK(gurabo_stream_create(foreign, &foreign_stream));
    OK(gurabo_event_create(context, &ready));
    STATUS(gurabo_event_create(context, &ready), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_event_query(ready), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_event_synchronize(ready), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_stream_wait_event(consumer, ready), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_event_record(foreign_stream, ready), GURABO_CONTEXT_MISMATCH);
    STATUS(gurabo_stream_wait_event(foreign_stream, ready), GURABO_CONTEXT_MISMATCH);
    OK(gurabo_stream_destroy(&foreign_stream));
    OK(gurabo_context_destroy(&foreign));

    OK(gurabo_event_create(context, &returned));
    OK(gurabo_event_create(context, &finished));
    OK(gurabo_buffer_alloc(context, sizeof(expected), 0, &input));
    OK(gurabo_buffer_alloc(context, sizeof(expected), 0, &middle));
    OK(gurabo_buffer_alloc(context, sizeof(expected), 0, &result));
    OK(gurabo_buffer_write(producer, input, 0, &expected, sizeof(expected)));
    OK(gurabo_buffer_copy_async(producer, middle, 0, input, 0, sizeof(expected)));
    OK(gurabo_event_record(producer, ready));
    STATUS(gurabo_event_record(producer, ready), GURABO_INVALID_ARGUMENT);
    OK(gurabo_stream_wait_event(consumer, ready));
    OK(gurabo_event_release(&ready));
    OK(gurabo_buffer_copy_async(consumer, result, 0, middle, 0, sizeof(expected)));
    OK(gurabo_event_record(consumer, returned));
    OK(gurabo_stream_wait_event(producer, returned));
    OK(gurabo_event_release(&returned));
    OK(gurabo_event_record(producer, finished));
    OK(gurabo_buffer_release(&input));
    OK(gurabo_buffer_release(&middle));

    gurabo_status status = gurabo_event_query(finished);
    CHECK(status == GURABO_OK || (backend == GURABO_HIP && status == GURABO_NOT_READY));
    OK(gurabo_event_synchronize(finished));
    OK(gurabo_buffer_read(producer, &actual, result, 0, sizeof(actual)));
    CHECK(actual == expected);

    OK(gurabo_buffer_release(&result));
    OK(gurabo_stream_destroy(&consumer));
    OK(gurabo_stream_destroy(&producer));
    OK(gurabo_event_query(finished));
    OK(gurabo_event_release(&finished));

    OK(gurabo_context_create(backend, ordinal, &foreign));
    OK(gurabo_event_create(foreign, &finished));
    STATUS(gurabo_context_destroy(&foreign), GURABO_BUSY);
    OK(gurabo_event_release(&finished));
    OK(gurabo_event_release(&finished));
    OK(gurabo_context_destroy(&foreign));
}

static void check_pinned_memory(gurabo_context *context)
{
    const float values[] = {1, 2, 3, 4};
    const float expected[] = {7, 10, 15, 22};
    const uint64_t offset = sizeof(float);
    const uint64_t host_bytes = sizeof(values) + 2 * offset;
    gurabo_stream *upload = NULL, *compute = NULL;
    gurabo_event *ready = NULL, *done = NULL;
    gurabo_buffer *host_in = NULL, *host_out = NULL, *device_in = NULL, *device_out = NULL;
    gurabo_tensor *input = NULL, *output = NULL;
    gurabo_tensor_desc shape = matrix_desc(2, 2);
    void *src = NULL, *dst = NULL;

    OK(gurabo_stream_create(context, &upload));
    OK(gurabo_stream_create(context, &compute));
    STATUS(gurabo_pinned_buffer_alloc(context, 0, &host_in), GURABO_INVALID_ARGUMENT);
    CHECK(host_in == NULL);
    OK(gurabo_pinned_buffer_alloc(context, host_bytes, &host_in));
    OK(gurabo_pinned_buffer_alloc(context, host_bytes, &host_out));
    OK(gurabo_buffer_alloc(context, sizeof(values), 0, &device_in));
    OK(gurabo_buffer_alloc(context, sizeof(expected), 0, &device_out));
    STATUS(gurabo_pinned_buffer_get_ptr(device_in, &src), GURABO_UNSUPPORTED);
    CHECK(src == NULL);
    STATUS(gurabo_tensor_create(host_in, 0, &shape, &input), GURABO_UNSUPPORTED);
    CHECK(input == NULL);
    STATUS(gurabo_buffer_copy_async(upload, host_out, 0, host_in, 0, 1), GURABO_UNSUPPORTED);
    STATUS(gurabo_buffer_copy_async(upload, device_in, 0, host_in, host_bytes, 1),
           GURABO_OUT_OF_RANGE);
    OK(gurabo_pinned_buffer_get_ptr(host_in, &src));
    OK(gurabo_pinned_buffer_get_ptr(host_out, &dst));
    memcpy((unsigned char *)src + offset, values, sizeof(values));
    memset(dst, 0xA5, (size_t)host_bytes);
    uint64_t usage = 0;
    OK(gurabo_context_memory_usage(context, &usage));
    CHECK(usage == 2 * host_bytes + sizeof(values) + sizeof(expected));

    OK(gurabo_tensor_create(device_in, 0, &shape, &input));
    OK(gurabo_tensor_create(device_out, 0, &shape, &output));
    OK(gurabo_event_create(context, &ready));
    OK(gurabo_event_create(context, &done));
    OK(gurabo_buffer_copy_async(upload, device_in, 0, host_in, offset, sizeof(values)));
    OK(gurabo_buffer_release(&host_in));
    OK(gurabo_event_record(upload, ready));
    OK(gurabo_stream_wait_event(compute, ready));
    OK(gurabo_event_release(&ready));
    OK(gurabo_matmul_async(compute, output, input, input, NULL));
    OK(gurabo_buffer_copy_async(compute, host_out, offset, device_out, 0, sizeof(expected)));
    OK(gurabo_event_record(compute, done));
    OK(gurabo_event_synchronize(done));
    CHECK(memcmp((unsigned char *)dst + offset, expected, sizeof(expected)) == 0);
    CHECK(((unsigned char *)dst)[0] == 0xA5 && ((unsigned char *)dst)[host_bytes - 1] == 0xA5);

    OK(gurabo_event_release(&done));
    OK(gurabo_tensor_release(&input));
    OK(gurabo_tensor_release(&output));
    OK(gurabo_buffer_release(&device_in));
    OK(gurabo_buffer_release(&device_out));
    OK(gurabo_buffer_release(&host_out));
    OK(gurabo_stream_destroy(&upload));
    OK(gurabo_stream_destroy(&compute));
    OK(gurabo_context_memory_usage(context, &usage));
    CHECK(usage == 0);
}

static void check_pointwise_chain(gurabo_context *context, gurabo_stream *stream,
                                  gurabo_dtype dtype)
{
    const float input[] = {1, 2, 3, 4};
    const float row[] = {1, 2};
    const float expected[] = {448, 1344, 896, 2688};
    const float expected_expanded[] = {8, 24, 16, 48};
    float actual[4];
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *input_f32 = NULL, *row_f32 = NULL, *first = NULL, *last = NULL;
    gurabo_tensor *input_16 = NULL, *row_16 = NULL, *intermediate = NULL, *expanded = NULL;
    gurabo_tensor_desc matrix = matrix_desc(2, 2);
    gurabo_tensor_desc row_shape = matrix_desc(1, 2);
    row_shape.rank = 1;
    row_shape.extent[0] = 2;
    OK(gurabo_buffer_alloc(context, 128, 0, &buffer));
    OK(gurabo_tensor_create(buffer, 0, &matrix, &input_f32));
    OK(gurabo_tensor_create(buffer, 16, &matrix, &first));
    OK(gurabo_tensor_create(buffer, 32, &matrix, &last));
    OK(gurabo_tensor_create(buffer, 48, &matrix, &expanded));
    OK(gurabo_tensor_create(buffer, 64, &row_shape, &row_f32));
    matrix.dtype = row_shape.dtype = dtype;
    OK(gurabo_tensor_create(buffer, 80, &matrix, &input_16));
    OK(gurabo_tensor_create(buffer, 96, &matrix, &intermediate));
    OK(gurabo_tensor_create(buffer, 112, &row_shape, &row_16));
    OK(gurabo_buffer_write(stream, buffer, 0, input, sizeof(input)));
    OK(gurabo_buffer_write(stream, buffer, 64, row, sizeof(row)));

    STATUS(gurabo_tensor_cast_async(stream, row_16, input_f32), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_binary_async(stream, GURABO_ADD, intermediate, input_16, row_f32),
           GURABO_UNSUPPORTED);
    OK(gurabo_tensor_cast_async(stream, input_16, input_f32));
    OK(gurabo_tensor_cast_async(stream, row_16, row_f32));
    OK(gurabo_gemm_async(stream, first, input_16, input_16, NULL));
    OK(gurabo_tensor_cast_async(stream, intermediate, first));
    OK(gurabo_binary_async(stream, GURABO_ADD, intermediate, intermediate, row_16));
    OK(gurabo_binary_async(stream, GURABO_MUL, intermediate, row_16, intermediate));
    OK(gurabo_tensor_cast_async(stream, expanded, intermediate));
    OK(gurabo_gemm_async(stream, last, intermediate, intermediate, NULL));

    OK(gurabo_tensor_release(&input_f32));
    OK(gurabo_tensor_release(&row_f32));
    OK(gurabo_tensor_release(&first));
    OK(gurabo_tensor_release(&last));
    OK(gurabo_tensor_release(&input_16));
    OK(gurabo_tensor_release(&row_16));
    OK(gurabo_tensor_release(&intermediate));
    OK(gurabo_tensor_release(&expanded));
    OK(gurabo_buffer_read(stream, actual, buffer, 32, sizeof(actual)));
    CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
    OK(gurabo_buffer_read(stream, actual, buffer, 48, sizeof(actual)));
    CHECK(memcmp(actual, expected_expanded, sizeof(actual)) == 0);
    OK(gurabo_buffer_release(&buffer));
}

static void check_binary_broadcast(gurabo_context *context, gurabo_stream *stream)
{
    const float a[] = {2, 4}, b[] = {3, 5, 7}, bias = 10;
    const float expected[] = {16, 20, 24, 22, 30, 38};
    float actual[6];
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *lhs = NULL, *rhs = NULL, *scalar = NULL, *out = NULL;
    gurabo_tensor *broadcast_alias = NULL, *partial_alias = NULL, *strided = NULL;
    gurabo_tensor_desc a_shape = matrix_desc(2, 1), b_shape = matrix_desc(3, 1);
    gurabo_tensor_desc scalar_shape = matrix_desc(1, 1), out_shape = matrix_desc(2, 3);
    b_shape.rank = 1;
    scalar_shape.rank = 0;
    OK(gurabo_buffer_alloc(context, 160, 0, &buffer));
    OK(gurabo_tensor_create(buffer, 0, &a_shape, &lhs));
    OK(gurabo_tensor_create(buffer, 32, &b_shape, &rhs));
    OK(gurabo_tensor_create(buffer, 48, &scalar_shape, &scalar));
    OK(gurabo_tensor_create(buffer, 64, &out_shape, &out));
    OK(gurabo_tensor_create(buffer, 0, &out_shape, &broadcast_alias));
    OK(gurabo_tensor_create(buffer, 68, &out_shape, &partial_alias));
    out_shape.flags = GURABO_TENSOR_EXPLICIT_STRIDES;
    out_shape.stride[0] = 4;
    out_shape.stride[1] = 1;
    OK(gurabo_tensor_create(buffer, 96, &out_shape, &strided));
    OK(gurabo_buffer_write(stream, buffer, 0, a, sizeof(a)));
    OK(gurabo_buffer_write(stream, buffer, 32, b, sizeof(b)));
    OK(gurabo_buffer_write(stream, buffer, 48, &bias, sizeof(bias)));

    STATUS(gurabo_binary_async(stream, 0, out, lhs, rhs), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_binary_async(stream, GURABO_ADD, lhs, lhs, rhs), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_binary_async(stream, GURABO_ADD, broadcast_alias, lhs, rhs),
           GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_binary_async(stream, GURABO_ADD, partial_alias, out, scalar),
           GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_binary_async(stream, GURABO_ADD, strided, lhs, rhs), GURABO_UNSUPPORTED);

    OK(gurabo_binary_async(stream, GURABO_MUL, out, lhs, rhs));
    OK(gurabo_binary_async(stream, GURABO_ADD, out, out, scalar));
    OK(gurabo_buffer_read(stream, actual, buffer, 64, sizeof(actual)));
    CHECK(memcmp(actual, expected, sizeof(expected)) == 0);
    const float divided[] = {2, 2, 2, 4, 4, 4};
    OK(gurabo_binary_async(stream, GURABO_SUB, out, out, scalar));
    OK(gurabo_binary_async(stream, GURABO_DIV, out, out, rhs));
    OK(gurabo_buffer_read(stream, actual, buffer, 64, sizeof(actual)));
    CHECK(memcmp(actual, divided, sizeof(divided)) == 0);
    OK(gurabo_tensor_release(&strided));
    OK(gurabo_tensor_release(&partial_alias));
    OK(gurabo_tensor_release(&broadcast_alias));
    OK(gurabo_tensor_release(&out));
    OK(gurabo_tensor_release(&scalar));
    OK(gurabo_tensor_release(&rhs));
    OK(gurabo_tensor_release(&lhs));
    OK(gurabo_buffer_release(&buffer));
}

static void check_cast_bits(gurabo_context *context, gurabo_stream *stream, gurabo_dtype dtype)
{
    const size_t count = (size_t)UINT16_MAX + 1;
    const size_t narrow_bytes = count * sizeof(uint16_t);
    const size_t wide_bytes = count * sizeof(float);
    uint16_t *patterns = malloc(narrow_bytes);
    uint32_t *expanded = malloc(wide_bytes);
    CHECK(patterns && expanded);
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *input = NULL, *output = NULL, *wide = NULL;
    gurabo_tensor_desc shape = matrix_desc(count, 1);
    shape.rank = 1;
    shape.dtype = dtype;
    OK(gurabo_buffer_alloc(context, 2 * narrow_bytes + wide_bytes, 0, &buffer));
    OK(gurabo_tensor_create(buffer, 0, &shape, &input));
    OK(gurabo_tensor_create(buffer, narrow_bytes, &shape, &output));
    shape.dtype = GURABO_F32;
    OK(gurabo_tensor_create(buffer, 2 * narrow_bytes, &shape, &wide));
    for (size_t i = 0; i < count; ++i) {
        patterns[i] = (uint16_t)i;
    }
    OK(gurabo_buffer_write(stream, buffer, 0, patterns, narrow_bytes));
    OK(gurabo_tensor_cast_async(stream, wide, input));
    OK(gurabo_tensor_cast_async(stream, output, wide));
    OK(gurabo_buffer_read(stream, expanded, buffer, 2 * narrow_bytes, wide_bytes));
    OK(gurabo_buffer_read(stream, patterns, buffer, narrow_bytes, narrow_bytes));

    float (*decode)(uint16_t) = dtype == GURABO_F16 ? gurabo_f16_to_f32 : gurabo_bf16_to_f32;
    uint16_t (*encode)(float) = dtype == GURABO_F16 ? gurabo_f16_from_f32 : gurabo_bf16_from_f32;
    uint16_t infinity = dtype == GURABO_F16 ? 0x7C00u : 0x7F80u;
    for (size_t i = 0; i < count; ++i) {
        float reference = decode((uint16_t)i);
        uint32_t bits;
        memcpy(&bits, &reference, sizeof(bits));
        if ((i & 0x7FFFu) <= infinity) {
            CHECK(expanded[i] == bits);
        }
        CHECK(patterns[i] == encode(reference));
    }
    OK(gurabo_tensor_release(&input));
    OK(gurabo_tensor_release(&wide));
    OK(gurabo_tensor_release(&output));
    OK(gurabo_buffer_release(&buffer));
    free(expanded);
    free(patterns);
}

static void check_rowwise(gurabo_context *context, gurabo_stream *stream,
                          gurabo_dtype dtype, uint64_t columns)
{
    enum { ROWS = 3, MAX_COLUMNS = 257 };
    float values[ROWS * MAX_COLUMNS], scale[MAX_COLUMNS], normalized[ROWS * MAX_COLUMNS];
    uint16_t encoded[ROWS * MAX_COLUMNS];
    float sum[ROWS], mean[ROWS];
    const float epsilon = 1e-5f;
    CHECK(columns <= MAX_COLUMNS);
    for (uint64_t col = 0; col < columns; ++col) {
        scale[col] = 1.0f + (float)(col % 5) * 0.125f;
    }
    for (uint64_t i = 0; i < ROWS * columns; ++i) {
        values[i] = (float)((int)(i % 11) - 5) * 0.125f;
        if (dtype != GURABO_F32) {
            encoded[i] = dtype == GURABO_F16 ? gurabo_f16_from_f32(values[i])
                                              : gurabo_bf16_from_f32(values[i]);
        }
    }
    size_t element_size = dtype == GURABO_F32 ? sizeof(float) : sizeof(uint16_t);
    size_t input_bytes = (size_t)(ROWS * columns) * element_size;
    size_t scale_offset = sizeof(values);
    size_t sum_offset = scale_offset + sizeof(scale);
    size_t mean_offset = sum_offset + sizeof(sum);
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *input = NULL, *scale_tensor = NULL, *sum_tensor = NULL, *mean_tensor = NULL;
    gurabo_tensor_desc rows = matrix_desc(ROWS, columns);
    gurabo_tensor_desc vector = matrix_desc(columns, 1);
    gurabo_tensor_desc reduced = matrix_desc(ROWS, 1);
    rows.dtype = dtype;
    vector.rank = 1;
    OK(gurabo_buffer_alloc(context, mean_offset + sizeof(mean), 0, &buffer));
    OK(gurabo_tensor_create(buffer, 0, &rows, &input));
    OK(gurabo_tensor_create(buffer, scale_offset, &vector, &scale_tensor));
    OK(gurabo_tensor_create(buffer, mean_offset, &reduced, &mean_tensor));
    reduced.rank = 1;
    OK(gurabo_tensor_create(buffer, sum_offset, &reduced, &sum_tensor));
    OK(gurabo_buffer_write(stream, buffer, 0, dtype == GURABO_F32 ? (void *)values : encoded,
                           input_bytes));
    OK(gurabo_buffer_write(stream, buffer, scale_offset, scale, columns * sizeof(float)));

    STATUS(gurabo_reduce_async(stream, GURABO_SUM, sum_tensor, input, 1u, 0), GURABO_UNSUPPORTED);
    STATUS(gurabo_reduce_async(stream, GURABO_SUM, sum_tensor, input, 2u, 1), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_rms_norm_async(stream, input, input, scale_tensor, 0), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_rms_norm_async(stream, input, input, scale_tensor, INFINITY), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_rms_norm_async(stream, input, input, sum_tensor, epsilon), GURABO_INVALID_ARGUMENT);
    OK(gurabo_reduce_async(stream, GURABO_SUM, sum_tensor, input, 2u, 0));
    OK(gurabo_reduce_async(stream, GURABO_MEAN, mean_tensor, input, 2u, 1));
    int use_scale = dtype != GURABO_BF16;
    OK(gurabo_rms_norm_async(stream, input, input, use_scale ? scale_tensor : NULL, epsilon));

    void *readback = dtype == GURABO_F32 ? (void *)normalized : encoded;
    OK(gurabo_buffer_read(stream, readback, buffer, 0, input_bytes));
    OK(gurabo_buffer_read(stream, sum, buffer, sum_offset, sizeof(sum)));
    OK(gurabo_buffer_read(stream, mean, buffer, mean_offset, sizeof(mean)));
    for (uint64_t row = 0; row < ROWS; ++row) {
        uint64_t start = row * columns;
        double total = 0, squares = 0;
        for (uint64_t col = 0; col < columns; ++col) {
            double value = values[start + col];
            total += value;
            squares += value * value;
        }
        CHECK(sum[row] == (float)total);
        CHECK(fabs(mean[row] - total / (double)columns) < 1e-6);
        double denominator = sqrt(squares / (double)columns + epsilon);
        for (uint64_t col = 0; col < columns; ++col) {
            uint64_t i = start + col;
            double reference = values[i] / denominator * (use_scale ? scale[col] : 1);
            float actual = dtype == GURABO_F32 ? normalized[i]
                         : dtype == GURABO_F16 ? gurabo_f16_to_f32(encoded[i])
                                              : gurabo_bf16_to_f32(encoded[i]);
            double tolerance = dtype == GURABO_F32 ? 1e-6 : dtype == GURABO_F16 ? 1e-3 : 8e-3;
            CHECK(fabs(actual - reference) < tolerance);
        }
    }
    OK(gurabo_tensor_release(&input));
    OK(gurabo_tensor_release(&scale_tensor));
    OK(gurabo_tensor_release(&sum_tensor));
    OK(gurabo_tensor_release(&mean_tensor));
    OK(gurabo_buffer_release(&buffer));
}

static void check_empty_reduction(gurabo_context *context, gurabo_stream *stream)
{
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *empty = NULL, *out = NULL;
    gurabo_tensor_desc shape = matrix_desc(2, 0);
    gurabo_tensor_desc reduced = matrix_desc(2, 1);
    float actual[2] = {1.0f, 1.0f};
    reduced.rank = 1;
    OK(gurabo_buffer_alloc(context, sizeof(actual), 0, &buffer));
    OK(gurabo_tensor_create(buffer, sizeof(actual), &shape, &empty));
    OK(gurabo_tensor_create(buffer, 0, &reduced, &out));
    OK(gurabo_buffer_write(stream, buffer, 0, actual, sizeof(actual)));
    STATUS(gurabo_reduce_async(stream, GURABO_MEAN, out, empty, 2u, 0), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_rms_norm_async(stream, empty, empty, NULL, 1e-5f), GURABO_INVALID_ARGUMENT);
    OK(gurabo_reduce_async(stream, GURABO_SUM, out, empty, 2u, 0));
    OK(gurabo_buffer_read(stream, actual, buffer, 0, sizeof(actual)));
    CHECK(actual[0] == 0 && actual[1] == 0);
    OK(gurabo_tensor_release(&out));
    OK(gurabo_tensor_release(&empty));
    OK(gurabo_buffer_release(&buffer));
}

static void check_rms_chain(gurabo_context *context, gurabo_stream *stream)
{
    const float input[] = {1, 2, 3, 4};
    const float bias[] = {1, 2};
    const float added[] = {8, 12, 16, 24};
    const float epsilon = 1e-5f;
    uint16_t actual[4];
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *x = NULL, *out = NULL, *row = NULL, *cast = NULL;
    gurabo_tensor_desc matrix = matrix_desc(2, 2);
    gurabo_tensor_desc vector = matrix_desc(2, 1);
    vector.rank = 1;
    OK(gurabo_buffer_alloc(context, 48, 0, &buffer));
    OK(gurabo_tensor_create(buffer, 0, &matrix, &x));
    OK(gurabo_tensor_create(buffer, 16, &matrix, &out));
    OK(gurabo_tensor_create(buffer, 32, &vector, &row));
    matrix.dtype = GURABO_F16;
    OK(gurabo_tensor_create(buffer, 40, &matrix, &cast));
    OK(gurabo_buffer_write(stream, buffer, 0, input, sizeof(input)));
    OK(gurabo_buffer_write(stream, buffer, 32, bias, sizeof(bias)));
    OK(gurabo_matmul_async(stream, out, x, x, NULL));
    OK(gurabo_binary_async(stream, GURABO_ADD, out, out, row));
    OK(gurabo_rms_norm_async(stream, out, out, NULL, epsilon));
    OK(gurabo_tensor_cast_async(stream, cast, out));
    OK(gurabo_buffer_read(stream, actual, buffer, 40, sizeof(actual)));
    for (unsigned i = 0; i < 4; ++i) {
        unsigned start = (i / 2) * 2;
        float a = added[start], b = added[start + 1];
        float expected = added[i] / sqrtf((a * a + b * b) / 2 + epsilon);
        CHECK(actual[i] == gurabo_f16_from_f32(expected));
    }
    OK(gurabo_tensor_release(&cast));
    OK(gurabo_tensor_release(&row));
    OK(gurabo_tensor_release(&out));
    OK(gurabo_tensor_release(&x));
    OK(gurabo_buffer_release(&buffer));
}

static void check_float16_conversion(void)
{
    CHECK(gurabo_f16_from_f32(-0.0f) == 0x8000u);
    CHECK(gurabo_f16_from_f32(0x1p-25f) == 0);
    CHECK(gurabo_f16_from_f32(0x1.8p-24f) == 2);
    CHECK(gurabo_f16_from_f32(1.0f + 0x1p-11f) == 0x3C00u);
    CHECK(gurabo_f16_from_f32(1.0f + 0x1.8p-10f) == 0x3C02u);
    CHECK(gurabo_f16_from_f32(65519.0f) == 0x7BFFu);
    CHECK(gurabo_f16_from_f32(65520.0f) == 0x7C00u);
    CHECK(gurabo_bf16_from_f32(-0.0f) == 0x8000u);
    CHECK(gurabo_bf16_from_f32(1.0f + 0x1p-8f) == 0x3F80u);
    CHECK(gurabo_bf16_from_f32(1.0f + 0x1.8p-7f) == 0x3F82u);

    for (uint32_t i = 0; i <= UINT16_MAX; ++i) {
        uint16_t bits = (uint16_t)i;
        uint16_t half = gurabo_f16_from_f32(gurabo_f16_to_f32(bits));
        uint16_t bfloat = gurabo_bf16_from_f32(gurabo_bf16_to_f32(bits));
        CHECK(half == ((bits & 0x7FFFu) > 0x7C00u ? (bits | 0x0200u) : bits));
        CHECK(bfloat == ((bits & 0x7FFFu) > 0x7F80u ? (bits | 0x0040u) : bits));
    }
}

static void check_softmax_activation(gurabo_context *context, gurabo_stream *stream)
{
    const float logits[] = {1000, 1001, 1002, 1, 2, 3, INFINITY, 1, INFINITY, NAN, 2, 3};
    const uint8_t mask[] = {1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1};
    const float activations[] = {-INFINITY, -2, -0.0f, 2, INFINITY, NAN};
    float values[12];
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *x = NULL, *keep = NULL, *activation = NULL, *maxima = NULL, *mask_row = NULL;
    gurabo_tensor_desc shape = matrix_desc(4, 3), vector = matrix_desc(6, 1);
    vector.rank = 1;
    size_t mask_offset = sizeof(logits);
    size_t activation_offset = mask_offset + sizeof(mask);
    size_t maxima_offset = activation_offset + sizeof(activations);
    OK(gurabo_buffer_alloc(context, maxima_offset + 4 * sizeof(float), 0, &buffer));
    OK(gurabo_tensor_create(buffer, 0, &shape, &x));
    OK(gurabo_tensor_create(buffer, activation_offset, &vector, &activation));
    vector.extent[0] = 4;
    OK(gurabo_tensor_create(buffer, maxima_offset, &vector, &maxima));
    shape.dtype = GURABO_U8;
    OK(gurabo_tensor_create(buffer, mask_offset, &shape, &keep));
    vector.dtype = GURABO_U8;
    vector.extent[0] = 3;
    OK(gurabo_tensor_create(buffer, mask_offset, &vector, &mask_row));
    OK(gurabo_buffer_write(stream, buffer, 0, logits, sizeof(logits)));
    OK(gurabo_buffer_write(stream, buffer, mask_offset, mask, sizeof(mask)));
    OK(gurabo_reduce_async(stream, GURABO_MAX, maxima, x, 2u, 0));
    OK(gurabo_buffer_read(stream, values, buffer, maxima_offset, 4 * sizeof(float)));
    CHECK(values[0] == 1002 && values[1] == 3 && isinf(values[2]) && isnan(values[3]));
    STATUS(gurabo_softmax_async(stream, x, x, 0, keep), GURABO_UNSUPPORTED);
    OK(gurabo_softmax_async(stream, x, x, 1, keep));
    OK(gurabo_buffer_read(stream, values, buffer, 0, sizeof(values)));
    CHECK(fabsf(values[0] - 0.11920292f) < 1e-6f && values[1] == 0);
    CHECK(fabsf(values[2] - 0.88079708f) < 1e-6f);
    CHECK(values[3] == 0 && values[4] == 0 && values[5] == 0);
    CHECK(values[6] == 0.5f && values[7] == 0 && values[8] == 0.5f);
    CHECK(values[9] == 0 && fabsf(values[10] - 0.26894142f) < 1e-6f);
    CHECK(fabsf(values[11] - 0.73105858f) < 1e-6f);

    OK(gurabo_buffer_write(stream, buffer, 0, logits, sizeof(logits)));
    OK(gurabo_softmax_async(stream, x, x, 1, mask_row));
    OK(gurabo_buffer_read(stream, values, buffer, 0, sizeof(values)));
    CHECK(fabsf(values[3] - values[0]) < 1e-6f && values[4] == 0);
    CHECK(fabsf(values[5] - values[2]) < 1e-6f);

    OK(gurabo_buffer_write(stream, buffer, 0, logits, sizeof(logits)));
    OK(gurabo_softmax_async(stream, x, x, 1, NULL));
    OK(gurabo_buffer_read(stream, values, buffer, 0, sizeof(values)));
    CHECK(fabsf(values[0] + values[1] + values[2] - 1.0f) < 1e-6f);
    CHECK(isnan(values[9]) && isnan(values[10]) && isnan(values[11]));
    for (gurabo_activation op = GURABO_SILU; op <= GURABO_GELU; ++op) {
        OK(gurabo_buffer_write(stream, buffer, activation_offset, activations, sizeof(activations)));
        OK(gurabo_activation_async(stream, op, activation, activation));
        OK(gurabo_buffer_read(stream, values, buffer, activation_offset, sizeof(activations)));
        CHECK(values[0] == 0 && values[2] == 0 && isinf(values[4]) && isnan(values[5]));
        if (op == GURABO_SILU) {
            CHECK(fabsf(values[1] + 0.23840584f) < 1e-6f);
            CHECK(fabsf(values[3] - 1.76159416f) < 1e-6f);
        } else if (op == GURABO_RELU) {
            CHECK(values[1] == 0 && values[3] == 2);
        } else {
            CHECK(fabsf(values[1] + 0.04550026f) < 1e-6f);
            CHECK(fabsf(values[3] - 1.95449974f) < 1e-6f);
        }
    }
    OK(gurabo_tensor_release(&activation));
    OK(gurabo_tensor_release(&maxima));
    OK(gurabo_tensor_release(&mask_row));
    OK(gurabo_tensor_release(&keep));
    OK(gurabo_tensor_release(&x));
    OK(gurabo_buffer_release(&buffer));
}

static void check_embedding_failure(gurabo_backend backend, uint32_t ordinal)
{
    const float table_values[] = {1, 2, 3, 4};
    const int64_t valid[] = {1, 0, 1}, invalid[] = {-1, 0, INT64_MAX};
    const float expected_valid[] = {3, 4, 1, 2, 3, 4};
    const float expected_invalid[] = {0, 0, 1, 2, 0, 0};
    const size_t index_offset = sizeof(table_values);
    const size_t output_offset = index_offset + sizeof(valid);
    float actual[6];
    gurabo_context *context = NULL;
    gurabo_stream *stream = NULL, *readback = NULL;
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *table = NULL, *indices = NULL, *out = NULL;
    gurabo_tensor_desc table_shape = matrix_desc(2, 2), index_shape = matrix_desc(3, 1);
    gurabo_tensor_desc out_shape = matrix_desc(3, 2);
    index_shape.rank = 1;
    index_shape.dtype = GURABO_I64;
    OK(gurabo_context_create(backend, ordinal, &context));
    OK(gurabo_stream_create(context, &stream));
    OK(gurabo_stream_create(context, &readback));
    OK(gurabo_buffer_alloc(context, output_offset + sizeof(actual), 0, &buffer));
    OK(gurabo_tensor_create(buffer, 0, &table_shape, &table));
    OK(gurabo_tensor_create(buffer, index_offset, &index_shape, &indices));
    OK(gurabo_tensor_create(buffer, output_offset, &out_shape, &out));
    OK(gurabo_buffer_write(stream, buffer, 0, table_values, sizeof(table_values)));
    OK(gurabo_buffer_write(stream, buffer, index_offset, valid, sizeof(valid)));
    OK(gurabo_embedding_async(stream, out, table, indices));
    OK(gurabo_buffer_read(stream, actual, buffer, output_offset, sizeof(actual)));
    CHECK(memcmp(actual, expected_valid, sizeof(actual)) == 0);
    OK(gurabo_buffer_write(stream, buffer, index_offset, invalid, sizeof(invalid)));
    OK(gurabo_embedding_async(stream, out, table, indices));
    STATUS(gurabo_stream_synchronize(stream), GURABO_OUT_OF_RANGE);
    gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE};
    OK(gurabo_stream_get_error(stream, &error));
    CHECK(error.phase == GURABO_ERROR_EXECUTION && error.submission_id != 0);
    OK(gurabo_buffer_read(readback, actual, buffer, output_offset, sizeof(actual)));
    CHECK(memcmp(actual, expected_invalid, sizeof(actual)) == 0);
    OK(gurabo_tensor_release(&out));
    OK(gurabo_tensor_release(&indices));
    OK(gurabo_tensor_release(&table));
    OK(gurabo_buffer_release(&buffer));
    OK(gurabo_stream_destroy(&readback));
    STATUS(gurabo_stream_destroy(&stream), GURABO_OUT_OF_RANGE);
    STATUS(gurabo_context_destroy(&context), GURABO_OUT_OF_RANGE);
}

static void check_axpby(gurabo_context *context, gurabo_stream *stream)
{
    struct accumulation_storage {
        float out[3];
        uint16_t input_a[3], input_b[3];
    };
    const struct accumulation_storage initial = {
        {NAN, NAN, NAN},
        {0x4000, 0xC400, 0x4600}, /* F16: 2, -4, 6 */
        {0x4080, 0x4100, 0xC080}  /* BF16: 4, 8, -4 */
    };
    gurabo_buffer *buffer = NULL;
    gurabo_tensor *out = NULL, *a = NULL, *b = NULL, *overlap = NULL;
    OK(gurabo_buffer_alloc(context, sizeof(initial), 0, &buffer));
    OK(gurabo_buffer_write(stream, buffer, 0, &initial, sizeof(initial)));
    gurabo_tensor_desc shape = {.struct_size = GURABO_TENSOR_DESC_SIZE, .dtype = GURABO_F32,
                                .rank = 1, .extent = {3}};
    OK(gurabo_tensor_create(buffer, 0, &shape, &out));
    OK(gurabo_tensor_create(buffer, sizeof(float), &shape, &overlap));
    shape.dtype = GURABO_F16;
    OK(gurabo_tensor_create(buffer, offsetof(struct accumulation_storage, input_a), &shape, &a));
    shape.dtype = GURABO_BF16;
    OK(gurabo_tensor_create(buffer, offsetof(struct accumulation_storage, input_b), &shape, &b));
    STATUS(gurabo_axpby_async(stream, out, overlap, 1, 0), GURABO_INVALID_ARGUMENT);
    STATUS(gurabo_axpby_async(stream, out, a, NAN, 0), GURABO_INVALID_ARGUMENT);
    OK(gurabo_axpby_async(stream, out, a, 0.5f, 0));
    OK(gurabo_axpby_async(stream, out, b, 0.5f, 1));
    OK(gurabo_axpby_async(stream, out, out, 0, 0.5f));
    float actual[3];
    OK(gurabo_buffer_read(stream, actual, buffer, 0, sizeof(actual)));
    CHECK(actual[0] == 1.5f && actual[1] == 1.0f && actual[2] == 0.5f);
    OK(gurabo_axpby_async(stream, out, out, 0, 0));
    OK(gurabo_buffer_read(stream, actual, buffer, 0, sizeof(actual)));
    CHECK(actual[0] == 0 && actual[1] == 0 && actual[2] == 0);
    OK(gurabo_tensor_release(&overlap));
    OK(gurabo_tensor_release(&b));
    OK(gurabo_tensor_release(&a));
    OK(gurabo_tensor_release(&out));
    OK(gurabo_buffer_release(&buffer));
}

enum { GEMM_MAX_BATCH = 2, GEMM_OUTPUT_SIZE = 8 };

static gurabo_tensor_desc batched_matrix_desc(gurabo_tensor_desc shape, uint32_t batches)
{
    if (batches) {
        shape.rank = 3;
        shape.extent[2] = shape.extent[1];
        shape.extent[1] = shape.extent[0];
        shape.extent[0] = batches;
    }
    return shape;
}

static double check_matmul(gurabo_context *context, gurabo_stream *stream,
                           gurabo_dtype dtype, uint32_t trans_a, uint32_t trans_b,
                           float beta, uint32_t a_batches, uint32_t b_batches, float *actual)
{
    enum { M = 2, N = 4, K = 3 };
    uint32_t batches = a_batches > b_batches ? a_batches : b_batches;
    int a_count = M * K * (a_batches ? (int)a_batches : 1);
    int b_count = K * N * (b_batches ? (int)b_batches : 1);
    int c_count = M * N * (batches ? (int)batches : 1);
    float a[GEMM_MAX_BATCH * M * K], b[GEMM_MAX_BATCH * K * N], c[GEMM_MAX_BATCH * M * N];
    for (int i = 0; i < a_count; ++i) {
        a[i] = (float)(i - 2) * 0.29f;
    }
    for (int i = 0; i < b_count; ++i) {
        b[i] = (float)(i % 7 - 3) * 0.37f;
    }
    for (int i = 0; i < c_count; ++i) {
        c[i] = (float)i * 0.13f;
    }
    float rounded_a[GEMM_MAX_BATCH * M * K], rounded_b[GEMM_MAX_BATCH * K * N];
    memcpy(rounded_a, a, (size_t)a_count * sizeof(*a));
    memcpy(rounded_b, b, (size_t)b_count * sizeof(*b));
    uint16_t encoded_a[GEMM_MAX_BATCH * M * K], encoded_b[GEMM_MAX_BATCH * K * N];
    if (dtype != GURABO_F32) {
        uint16_t (*encode)(float) = dtype == GURABO_F16 ? gurabo_f16_from_f32 : gurabo_bf16_from_f32;
        float (*decode)(uint16_t) = dtype == GURABO_F16 ? gurabo_f16_to_f32 : gurabo_bf16_to_f32;
        for (int i = 0; i < a_count; ++i) {
            encoded_a[i] = encode(a[i]);
            rounded_a[i] = decode(encoded_a[i]);
        }
        for (int i = 0; i < b_count; ++i) {
            encoded_b[i] = encode(b[i]);
            rounded_b[i] = decode(encoded_b[i]);
        }
    }

    gurabo_buffer *ab = NULL, *bb = NULL, *cb = NULL;
    gurabo_tensor *at = NULL, *bt = NULL, *ct = NULL;
    gurabo_tensor_desc ashape = batched_matrix_desc(trans_a ? matrix_desc(K, M) : matrix_desc(M, K),
                                                   a_batches);
    gurabo_tensor_desc bshape = batched_matrix_desc(trans_b ? matrix_desc(N, K) : matrix_desc(K, N),
                                                   b_batches);
    gurabo_tensor_desc cshape = batched_matrix_desc(matrix_desc(M, N), batches);
    ashape.dtype = bshape.dtype = dtype;
    gurabo_matmul_desc config = {GURABO_MATMUL_DESC_SIZE, trans_a, trans_b, -0.75f, beta};

    size_t input_size = dtype == GURABO_F32 ? sizeof(*a) : sizeof(*encoded_a);
    size_t a_bytes = (size_t)a_count * input_size;
    size_t b_bytes = (size_t)b_count * input_size;
    size_t c_bytes = (size_t)c_count * sizeof(*c);
    OK(gurabo_buffer_alloc(context, a_bytes, 0, &ab));
    OK(gurabo_buffer_alloc(context, b_bytes, 0, &bb));
    OK(gurabo_buffer_alloc(context, c_bytes, 0, &cb));
    OK(gurabo_buffer_write(stream, ab, 0, dtype == GURABO_F32 ? (void *)a : encoded_a, a_bytes));
    OK(gurabo_buffer_write(stream, bb, 0, dtype == GURABO_F32 ? (void *)b : encoded_b, b_bytes));
    if (beta != 0) {
        OK(gurabo_buffer_write(stream, cb, 0, c, c_bytes));
    }
    OK(gurabo_tensor_create(ab, 0, &ashape, &at));
    OK(gurabo_tensor_create(bb, 0, &bshape, &bt));
    OK(gurabo_tensor_create(cb, 0, &cshape, &ct));

    if (batches > 1) {
        gurabo_tensor *bad_output = NULL;
        gurabo_tensor_desc bad_shape = batched_matrix_desc(matrix_desc(M, N), 1);
        OK(gurabo_tensor_create(cb, 0, &bad_shape, &bad_output));
        STATUS(gurabo_gemm_async(stream, bad_output, at, bt, &config), GURABO_INVALID_ARGUMENT);
        OK(gurabo_tensor_release(&bad_output));
    }
    if (dtype == GURABO_F32) {
        if (!batches) {
            STATUS(gurabo_matmul_async(stream, at, at, bt, &config), GURABO_INVALID_ARGUMENT);
        }
        OK(gurabo_matmul_async(stream, ct, at, bt, &config));
    } else {
        STATUS(gurabo_matmul_async(stream, ct, at, bt, &config), GURABO_UNSUPPORTED);
        if (!batches) {
            STATUS(gurabo_gemm_async(stream, at, at, bt, &config), GURABO_UNSUPPORTED);
        }
        STATUS(gurabo_gemm_async(stream, ct, at, ct, &config), GURABO_UNSUPPORTED);
        OK(gurabo_gemm_async(stream, ct, at, bt, &config));
    }
    OK(gurabo_tensor_release(&at));
    OK(gurabo_tensor_release(&bt));
    OK(gurabo_tensor_release(&ct));
    OK(gurabo_buffer_release(&ab));
    OK(gurabo_buffer_release(&bb));
    OK(gurabo_buffer_read(stream, actual, cb, 0, c_bytes));
    OK(gurabo_buffer_release(&cb));

    double max_input_rounding = 0;
    for (int index = 0; index < c_count; ++index) {
        int batch = index / (M * N);
        int i = index / N % M;
        int j = index % N;
        int a_start = a_batches > 1 ? batch * M * K : 0;
        int b_start = b_batches > 1 ? batch * K * N : 0;
        double reference = (double)beta * c[index];
        double full_precision = reference;
        for (int k = 0; k < K; ++k) {
            int ai = a_start + (trans_a ? k * M + i : i * K + k);
            int bi = b_start + (trans_b ? j * K + k : k * N + j);
            reference += (double)config.alpha * rounded_a[ai] * rounded_b[bi];
            full_precision += (double)config.alpha * a[ai] * b[bi];
        }
        CHECK(fabs((double)actual[index] - reference) <= 2e-6 * (1 + fabs(reference)));
        double input_rounding = fabs(reference - full_precision);
        if (input_rounding > max_input_rounding) {
            max_input_rounding = input_rounding;
        }
    }
    return max_input_rounding;
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    gurabo_backend backend = strcmp(argv[1], "hip") == 0 ? GURABO_HIP : GURABO_CPU;
    uint32_t count = 0;
    gurabo_status availability = gurabo_device_count(backend, &count);
    if (backend == GURABO_HIP &&
        (availability == GURABO_BACKEND_UNAVAILABLE || (availability == GURABO_OK && !count))) {
        fprintf(stderr, "No HIP device available\n");
        return 77;
    }
    CHECK(availability == GURABO_OK && count > 0);

    uint32_t ordinal = 0;
    if (backend == GURABO_HIP) {
        for (uint32_t i = 0; i < count; ++i) {
            gurabo_device_info info = {.struct_size = GURABO_DEVICE_INFO_SIZE};
            OK(gurabo_device_get_info(backend, i, &info));
            if (strncmp(info.architecture, "gfx1101", 7) == 0 &&
                strstr(info.name, "7700 XT")) {
                ordinal = i;
                break;
            }
        }
    }
    check_discovery(backend, ordinal);
    gurabo_context *context = NULL;
    gurabo_context *cpu_context = NULL;
    gurabo_stream *stream = NULL;
    gurabo_stream *cpu_stream = NULL;
    OK(gurabo_context_create(backend, ordinal, &context));
    OK(gurabo_stream_create(context, &stream));
    if (backend == GURABO_HIP && gurabo_device_count(GURABO_CPU, &count) == GURABO_OK) {
        OK(gurabo_context_create(GURABO_CPU, 0, &cpu_context));
        OK(gurabo_stream_create(cpu_context, &cpu_stream));
    }
    gurabo_context *saved = context;
    STATUS(gurabo_context_destroy(&context), GURABO_BUSY);
    CHECK(context == saved);

    check_memory(backend, ordinal, context, stream);
    check_layout(context);
    check_events(backend, ordinal, context);
    check_pinned_memory(context);
    check_binary_broadcast(context, stream);
    check_cast_bits(context, stream, GURABO_F16);
    check_cast_bits(context, stream, GURABO_BF16);
    check_pointwise_chain(context, stream, GURABO_F16);
    check_pointwise_chain(context, stream, GURABO_BF16);
    check_rowwise(context, stream, GURABO_F32, 33);
    check_rowwise(context, stream, GURABO_F16, 257);
    check_rowwise(context, stream, GURABO_BF16, 33);
    check_empty_reduction(context, stream);
    check_rms_chain(context, stream);
    check_softmax_activation(context, stream);
    check_axpby(context, stream);
    check_float16_conversion();
    for (gurabo_dtype dtype = GURABO_F32; dtype <= GURABO_BF16; ++dtype) {
        double max_input_rounding = 0;
        for (uint32_t ta = 0; ta < 2; ++ta) {
            for (uint32_t tb = 0; tb < 2; ++tb) {
                for (int use_beta = 0; use_beta <= 1; ++use_beta) {
                    float actual[GEMM_OUTPUT_SIZE];
                    float beta = use_beta ? 0.5f : 0.0f;
                    double rounding = check_matmul(context, stream, dtype, ta, tb, beta, 0, 0, actual);
                    if (rounding > max_input_rounding) {
                        max_input_rounding = rounding;
                    }
                    if (cpu_stream) {
                        float cpu[GEMM_OUTPUT_SIZE];
                        check_matmul(cpu_context, cpu_stream, dtype, ta, tb, beta, 0, 0, cpu);
                        for (size_t i = 0; i < sizeof(cpu) / sizeof(cpu[0]); ++i) {
                            CHECK(fabsf(actual[i] - cpu[i]) <= 4e-6f * (1 + fabsf(cpu[i])));
                        }
                    }
                }
            }
        }
        static const uint32_t batch_shapes[][2] = {{2, 2}, {2, 0}, {1, 2}, {0, 2}, {2, 1}, {1, 1}};
        for (size_t i = 0; i < sizeof(batch_shapes) / sizeof(batch_shapes[0]); ++i) {
            float actual[GEMM_MAX_BATCH * GEMM_OUTPUT_SIZE];
            uint32_t a_batches = batch_shapes[i][0], b_batches = batch_shapes[i][1];
            uint32_t ta = (uint32_t)(i % 2), tb = (uint32_t)(i / 2 % 2);
            check_matmul(context, stream, dtype, ta, tb, 0.5f, a_batches, b_batches, actual);
            if (cpu_stream) {
                float cpu[GEMM_MAX_BATCH * GEMM_OUTPUT_SIZE];
                check_matmul(cpu_context, cpu_stream, dtype, ta, tb, 0.5f, a_batches, b_batches, cpu);
                uint32_t batches = a_batches > b_batches ? a_batches : b_batches;
                for (size_t j = 0; j < batches * GEMM_OUTPUT_SIZE; ++j) {
                    CHECK(fabsf(actual[j] - cpu[j]) <= 4e-6f * (1 + fabsf(cpu[j])));
                }
            }
        }
        if (dtype != GURABO_F32) {
            printf("%s maximum absolute output error from rounding inputs: %.8g\n",
                   dtype == GURABO_F16 ? "FP16" : "BF16", max_input_rounding);
        }
    }
    gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE, .status = 123};
    STATUS(gurabo_stream_get_error(stream, &error), GURABO_NOT_READY);
    CHECK(error.status == 123);
    STATUS(gurabo_context_get_error(context, &error), GURABO_NOT_READY);
    CHECK(error.status == 123);
    OK(gurabo_stream_destroy(&stream));
    OK(gurabo_context_destroy(&context));
    OK(gurabo_stream_destroy(&cpu_stream));
    OK(gurabo_context_destroy(&cpu_context));
    CHECK(!context && !stream);
    check_embedding_failure(backend, ordinal);
    return 0;
}
