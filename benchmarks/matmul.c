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

int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "cpu") != 0 && strcmp(argv[1], "hip") != 0)) {
        fprintf(stderr, "Usage: %s [cpu|hip]\n", argv[0]);
        return EXIT_FAILURE;
    }
    enum { N = 256, ITERATIONS = 40 };
    gurabo_backend backend = argc == 2 && strcmp(argv[1], "hip") == 0 ? GURABO_HIP : GURABO_CPU;
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
