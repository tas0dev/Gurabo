#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gurabo/gurabo.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static void check(gurabo_status status)
{
    if (status != GURABO_OK) {
        gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE};
        gurabo_get_thread_error(&error);
        fprintf(stderr, "%s: %s\n", gurabo_status_string(status), error.message);
        exit(EXIT_FAILURE);
    }
}

static double now(void)
{
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    if (QueryPerformanceFrequency(&frequency) && QueryPerformanceCounter(&counter)) {
        return (double)counter.QuadPart / (double)frequency.QuadPart;
    }
#else
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) == 0) {
        return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
    }
#endif
    fputs("Could not read the benchmark clock\n", stderr);
    exit(EXIT_FAILURE);
}

typedef struct {
    gurabo_buffer *device_in;
    gurabo_buffer *device_out;
    gurabo_buffer *host_in;
    gurabo_buffer *host_out;
    gurabo_tensor *input;
    gurabo_tensor *output;
    gurabo_event *done;
} pipeline_slot;

static void run_pipeline(gurabo_context *context, gurabo_stream *upload,
                         gurabo_stream *compute, pipeline_slot slots[2],
                         size_t bytes, unsigned iterations)
{
    for (unsigned i = 0; i < iterations; ++i) {
        pipeline_slot *slot = &slots[i % 2];
        gurabo_event *ready = NULL;
        if (slot->done) {
            check(gurabo_stream_wait_event(upload, slot->done));
            check(gurabo_event_release(&slot->done));
        }
        check(gurabo_buffer_copy_async(upload, slot->device_in, 0, slot->host_in, 0, bytes));
        check(gurabo_event_create(context, &ready));
        check(gurabo_event_record(upload, ready));
        check(gurabo_stream_wait_event(compute, ready));
        check(gurabo_event_release(&ready));
        check(gurabo_matmul_async(compute, slot->output, slot->input, slot->input, NULL));
        check(gurabo_buffer_copy_async(compute, slot->host_out, 0, slot->device_out, 0, bytes));
        check(gurabo_event_create(context, &slot->done));
        check(gurabo_event_record(compute, slot->done));
    }
    check(gurabo_context_synchronize(context));
}

static void benchmark_pipeline(gurabo_backend backend)
{
    enum { N = 512, ITERATIONS = 64 };
    const size_t elements = (size_t)N * N;
    const size_t bytes = elements * sizeof(float);
    float *storage = malloc(4 * bytes);
    if (!storage) {
        fputs("Could not allocate host data\n", stderr);
        exit(EXIT_FAILURE);
    }
    float *inputs[] = {storage, storage + elements};
    float *references[] = {storage + 2 * elements, storage + 3 * elements};
    gurabo_context *context = NULL;
    gurabo_stream *upload = NULL, *compute = NULL;
    pipeline_slot slots[2] = {0};
    gurabo_device_info info = {.struct_size = GURABO_DEVICE_INFO_SIZE};
    gurabo_tensor_desc shape = {GURABO_TENSOR_DESC_SIZE, GURABO_F32, 2, 0, {N, N}, {0}};
    check(gurabo_device_get_info(backend, 0, &info));
    check(gurabo_context_create(backend, 0, &context));
    check(gurabo_stream_create(context, &upload));
    check(gurabo_stream_create(context, &compute));
    for (unsigned i = 0; i < 2; ++i) {
        pipeline_slot *slot = &slots[i];
        for (size_t j = 0; j < elements; ++j) {
            inputs[i][j] = (float)((i + 1) * (j % 17)) / 34.0f;
        }
        check(gurabo_buffer_alloc(context, bytes, 0, &slot->device_in));
        check(gurabo_buffer_alloc(context, bytes, 0, &slot->device_out));
        check(gurabo_pinned_buffer_alloc(context, bytes, &slot->host_in));
        check(gurabo_pinned_buffer_alloc(context, bytes, &slot->host_out));
        check(gurabo_tensor_create(slot->device_in, 0, &shape, &slot->input));
        check(gurabo_tensor_create(slot->device_out, 0, &shape, &slot->output));
        void *host = NULL;
        check(gurabo_pinned_buffer_get_ptr(slot->host_in, &host));
        memcpy(host, inputs[i], bytes);
    }

    /* Warm up GEMM, both Streams and the synchronous transfer staging area. */
    run_pipeline(context, upload, compute, slots, bytes, 2);
    pipeline_slot *serial = &slots[0];
    check(gurabo_buffer_write(compute, serial->device_in, 0, inputs[0], bytes));
    check(gurabo_matmul_async(compute, serial->output, serial->input, serial->input, NULL));
    check(gurabo_buffer_read(compute, references[0], serial->device_out, 0, bytes));

    double start = now();
    for (unsigned i = 0; i < ITERATIONS; ++i) {
        unsigned input_index = i % 2;
        check(gurabo_buffer_write(compute, serial->device_in, 0, inputs[input_index], bytes));
        check(gurabo_matmul_async(compute, serial->output, serial->input, serial->input, NULL));
        check(gurabo_buffer_read(compute, references[input_index], serial->device_out, 0, bytes));
    }
    double synchronous = now() - start;
    start = now();
    run_pipeline(context, upload, compute, slots, bytes, ITERATIONS);
    double pipelined = now() - start;

    for (unsigned i = 0; i < 2; ++i) {
        pipeline_slot *slot = &slots[i];
        void *host = NULL;
        check(gurabo_pinned_buffer_get_ptr(slot->host_out, &host));
        const float *actual = host;
        for (size_t j = 0; j < elements; ++j) {
            float reference = references[i][j];
            float difference = actual[j] - reference;
            float tolerance = 1e-4f * (1.0f + reference);
            if (!(difference >= -tolerance && difference <= tolerance)) {
                fprintf(stderr, "Pipeline result mismatch in slot %u at element %zu\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
        check(gurabo_event_release(&slot->done));
        check(gurabo_tensor_release(&slot->input));
        check(gurabo_tensor_release(&slot->output));
        check(gurabo_buffer_release(&slot->device_in));
        check(gurabo_buffer_release(&slot->device_out));
        check(gurabo_buffer_release(&slot->host_in));
        check(gurabo_buffer_release(&slot->host_out));
    }
    printf("%s (%s), FP32, %d x %d, %d iterations, H2D + GEMM + D2H\n",
           info.name, info.architecture, N, N, ITERATIONS);
    printf("Synchronous: %.3f us/iteration\n", synchronous * 1e6 / ITERATIONS);
    printf("Double-buffered: %.3f us/iteration (%.2fx speedup), both outputs verified\n",
           pipelined * 1e6 / ITERATIONS, synchronous / pipelined);
    check(gurabo_stream_destroy(&upload));
    check(gurabo_stream_destroy(&compute));
    check(gurabo_context_destroy(&context));
    free(storage);
}

int main(int argc, char **argv)
{
    if (argc > 3 || (argc > 1 && strcmp(argv[1], "cpu") != 0 && strcmp(argv[1], "hip") != 0) ||
        (argc == 3 && strcmp(argv[2], "pipeline") != 0)) {
        fprintf(stderr, "Usage: %s [cpu|hip [pipeline]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    gurabo_backend backend = argc > 1 && strcmp(argv[1], "hip") == 0 ? GURABO_HIP : GURABO_CPU;
    if (argc == 3) {
        benchmark_pipeline(backend);
        return EXIT_SUCCESS;
    }
    enum { N = 256, ITERATIONS = 40 };
    const size_t bytes = (size_t)N * N * sizeof(float);
    float *data = malloc(bytes);
    if (!data) {
        fputs("Could not allocate host data\n", stderr);
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        data[i] = (float)(i % 17) / 17.0f;
    }

    gurabo_device_info info = {.struct_size = GURABO_DEVICE_INFO_SIZE};
    check(gurabo_device_get_info(backend, 0, &info));
    gurabo_context *context = NULL;
    gurabo_stream *stream = NULL;
    gurabo_buffer *input = NULL, *output = NULL;
    gurabo_tensor *a = NULL, *c = NULL;
    gurabo_tensor_desc shape = {GURABO_TENSOR_DESC_SIZE, GURABO_F32, 2, 0, {N, N}, {0}};
    check(gurabo_context_create(backend, 0, &context));
    check(gurabo_stream_create(context, &stream));
    check(gurabo_buffer_alloc(context, bytes, 0, &input));
    check(gurabo_buffer_alloc(context, bytes, 0, &output));
    check(gurabo_buffer_write(stream, input, 0, data, bytes));
    check(gurabo_tensor_create(input, 0, &shape, &a));
    check(gurabo_tensor_create(output, 0, &shape, &c));
    check(gurabo_matmul_async(stream, c, a, a, NULL));
    check(gurabo_stream_synchronize(stream));
    check(gurabo_buffer_read(stream, data, output, 0, bytes));

    double start = now();
    for (int i = 0; i < ITERATIONS; ++i) {
        check(gurabo_matmul_async(stream, c, a, a, NULL));
    }
    double submitted = now();
    check(gurabo_stream_synchronize(stream));
    double finished = now();
    double upload_start = now();
    for (int i = 0; i < ITERATIONS; ++i) {
        check(gurabo_buffer_write(stream, input, 0, data, bytes));
    }
    double upload_elapsed = now() - upload_start;
    double download_start = now();
    for (int i = 0; i < ITERATIONS; ++i) {
        check(gurabo_buffer_read(stream, data, output, 0, bytes));
    }
    double download_elapsed = now() - download_start;

    printf("%s (%s), FP32, %d x %d, %d iterations\n",
           info.name, info.architecture, N, N, ITERATIONS);
    printf("GEMM submission %.3f us/op, including synchronization %.3f us/op\n",
           (submitted - start) * 1e6 / ITERATIONS, (finished - start) * 1e6 / ITERATIONS);
    if (upload_elapsed > 0 && download_elapsed > 0) {
        printf("Host to device %.3f GB/s, device to host %.3f GB/s\n",
               (double)bytes * ITERATIONS / upload_elapsed / 1e9,
               (double)bytes * ITERATIONS / download_elapsed / 1e9);
    }

    check(gurabo_tensor_release(&a));
    check(gurabo_tensor_release(&c));
    check(gurabo_buffer_release(&input));
    check(gurabo_buffer_release(&output));
    check(gurabo_stream_destroy(&stream));
    check(gurabo_context_destroy(&context));
    free(data);
    return EXIT_SUCCESS;
}
