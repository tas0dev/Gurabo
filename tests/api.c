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
    CHECK(gurabo_abi_version() == GURABO_ABI_VERSION);
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
    CHECK(caps.storage_dtype_mask == (UINT64_C(1) << GURABO_F32));
    CHECK(caps.feature_mask & GURABO_FEATURE_F32_MATMUL);
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

static void check_matmul(gurabo_context *context, gurabo_stream *stream,
                         uint32_t trans_a, uint32_t trans_b, float beta,
                         float actual[8])
{
    enum { M = 2, N = 4, K = 3 };
    float a[M * K], b[K * N], c[M * N];
    for (int i = 0; i < M * K; ++i) {
        a[i] = (float)(i - 2) * 0.29f;
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = (float)(i % 7 - 3) * 0.37f;
    }
    for (int i = 0; i < M * N; ++i) {
        c[i] = (float)i * 0.13f;
    }

    gurabo_buffer *ab = NULL, *bb = NULL, *cb = NULL;
    gurabo_tensor *at = NULL, *bt = NULL, *ct = NULL;
    gurabo_tensor_desc ashape = trans_a ? matrix_desc(K, M) : matrix_desc(M, K);
    gurabo_tensor_desc bshape = trans_b ? matrix_desc(N, K) : matrix_desc(K, N);
    gurabo_tensor_desc cshape = matrix_desc(M, N);
    gurabo_matmul_desc config = {GURABO_MATMUL_DESC_SIZE, trans_a, trans_b, -0.75f, beta};

    OK(gurabo_buffer_alloc(context, sizeof(a), 0, &ab));
    OK(gurabo_buffer_alloc(context, sizeof(b), 0, &bb));
    OK(gurabo_buffer_alloc(context, sizeof(c), 0, &cb));
    OK(gurabo_buffer_write(stream, ab, 0, a, sizeof(a)));
    OK(gurabo_buffer_write(stream, bb, 0, b, sizeof(b)));
    if (beta != 0) {
        OK(gurabo_buffer_write(stream, cb, 0, c, sizeof(c)));
    }
    OK(gurabo_tensor_create(ab, 0, &ashape, &at));
    OK(gurabo_tensor_create(bb, 0, &bshape, &bt));
    OK(gurabo_tensor_create(cb, 0, &cshape, &ct));

    STATUS(gurabo_matmul_async(stream, at, at, bt, &config), GURABO_INVALID_ARGUMENT);
    OK(gurabo_matmul_async(stream, ct, at, bt, &config));
    OK(gurabo_tensor_release(&at));
    OK(gurabo_tensor_release(&bt));
    OK(gurabo_tensor_release(&ct));
    OK(gurabo_buffer_release(&ab));
    OK(gurabo_buffer_release(&bb));
    OK(gurabo_buffer_read(stream, actual, cb, 0, sizeof(c)));
    OK(gurabo_buffer_release(&cb));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int index = i * N + j;
            double reference = (double)beta * c[index];
            for (int k = 0; k < K; ++k) {
                double av = a[trans_a ? k * M + i : i * K + k];
                double bv = b[trans_b ? j * K + k : k * N + j];
                reference += (double)config.alpha * av * bv;
            }
            CHECK(fabs((double)actual[index] - reference) <= 2e-6 * (1 + fabs(reference)));
        }
    }
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
    for (uint32_t ta = 0; ta < 2; ++ta) {
        for (uint32_t tb = 0; tb < 2; ++tb) {
            for (int use_beta = 0; use_beta <= 1; ++use_beta) {
                float actual[8];
                float beta = use_beta ? 0.5f : 0.0f;
                check_matmul(context, stream, ta, tb, beta, actual);
                if (cpu_stream) {
                    float cpu[8];
                    check_matmul(cpu_context, cpu_stream, ta, tb, beta, cpu);
                    for (size_t i = 0; i < sizeof(cpu) / sizeof(cpu[0]); ++i) {
                        CHECK(fabsf(actual[i] - cpu[i]) <= 4e-6f * (1 + fabsf(cpu[i])));
                    }
                }
            }
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
    return 0;
}
