#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gurabo/gurabo.h>

typedef struct {
    float table[6];
    int32_t tokens[2];
    uint8_t mask[4];
    float embedded[4];
    float normalized[4];
    float scores[4];
    float attention[4];
    float gate[4];
} inference_storage;

static void check(gurabo_status status)
{
    if (status != GURABO_OK) {
        gurabo_error_info error = {.struct_size = GURABO_ERROR_INFO_SIZE};
        gurabo_get_thread_error(&error);
        fprintf(stderr, "%s: %s\n", gurabo_status_string(status), error.message);
        exit(EXIT_FAILURE);
    }
}

static gurabo_tensor *tensor(gurabo_buffer *buffer, uint64_t offset, gurabo_dtype dtype,
                             uint32_t rank, uint64_t height, uint64_t width)
{
    gurabo_tensor_desc desc = {GURABO_TENSOR_DESC_SIZE, dtype, rank, 0, {height, width}, {0}};
    gurabo_tensor *result = NULL;
    check(gurabo_tensor_create(buffer, offset, &desc, &result));
    return result;
}

int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "cpu") != 0 && strcmp(argv[1], "hip") != 0)) {
        fprintf(stderr, "Usage: %s [cpu|hip]\n", argv[0]);
        return EXIT_FAILURE;
    }
    gurabo_backend backend = argc == 2 && strcmp(argv[1], "hip") == 0 ? GURABO_HIP : GURABO_CPU;
    uint32_t count = 0;
    gurabo_status availability = gurabo_device_count(backend, &count);
    if (backend == GURABO_HIP &&
        (availability == GURABO_BACKEND_UNAVAILABLE || (availability == GURABO_OK && count == 0))) {
        fputs("No HIP device available\n", stderr);
        return 77;
    }
    check(availability);
    gurabo_device_info device = {.struct_size = GURABO_DEVICE_INFO_SIZE};
    check(gurabo_device_get_info(backend, 0, &device));

    gurabo_context *context = NULL;
    gurabo_stream *stream = NULL;
    gurabo_buffer *buffer = NULL;
    check(gurabo_context_create(backend, 0, &context));
    check(gurabo_stream_create(context, &stream));
    check(gurabo_buffer_alloc(context, sizeof(inference_storage), 0, &buffer));
    const inference_storage initial = {
        .table = {1, 0, 0, 1, 1, 1},
        .tokens = {0, 1},
        .mask = {1, 0, 1, 1}
    };
    check(gurabo_buffer_write(stream, buffer, 0, &initial, offsetof(inference_storage, embedded)));

    gurabo_tensor *table = tensor(buffer, offsetof(inference_storage, table), GURABO_F32, 2, 3, 2);
    gurabo_tensor *tokens = tensor(buffer, offsetof(inference_storage, tokens), GURABO_I32, 1, 2, 0);
    gurabo_tensor *mask = tensor(buffer, offsetof(inference_storage, mask), GURABO_U8, 2, 2, 2);
    gurabo_tensor *embedded = tensor(buffer, offsetof(inference_storage, embedded), GURABO_F32, 2, 2, 2);
    gurabo_tensor *normalized = tensor(buffer, offsetof(inference_storage, normalized), GURABO_F32, 2, 2, 2);
    gurabo_tensor *scores = tensor(buffer, offsetof(inference_storage, scores), GURABO_F32, 2, 2, 2);
    gurabo_tensor *attention = tensor(buffer, offsetof(inference_storage, attention), GURABO_F32, 2, 2, 2);
    gurabo_tensor *gate = tensor(buffer, offsetof(inference_storage, gate), GURABO_F32, 2, 2, 2);

    const float epsilon = 1e-5f;
    const gurabo_matmul_desc score_options = {GURABO_MATMUL_DESC_SIZE, 0, 1, 0.7071067812f, 0};
    check(gurabo_embedding_async(stream, embedded, table, tokens));
    check(gurabo_rms_norm_async(stream, normalized, embedded, NULL, epsilon));
    /* Use the same normalized representation for Q, K and V in this tiny example. */
    check(gurabo_gemm_async(stream, scores, normalized, normalized, &score_options));
    check(gurabo_softmax_async(stream, scores, scores, 1, mask));
    check(gurabo_gemm_async(stream, attention, scores, normalized, NULL));
    check(gurabo_binary_async(stream, GURABO_ADD, attention, attention, embedded));
    check(gurabo_activation_async(stream, GURABO_SILU, gate, attention));
    check(gurabo_binary_async(stream, GURABO_MUL, gate, gate, attention));
    check(gurabo_binary_async(stream, GURABO_ADD, attention, attention, gate));

    float output[4];
    check(gurabo_buffer_read(stream, output, buffer, offsetof(inference_storage, attention),
                             sizeof(output)));
    double norm = 1.0 / sqrt(0.5 + epsilon);
    double probability = 1.0 / (1.0 + exp(-norm * norm * score_options.alpha));
    double residual[] = {1.0 + norm, 0, (1.0 - probability) * norm, 1.0 + probability * norm};
    int matches = 1;
    for (unsigned i = 0; i < 4; ++i) {
        double x = residual[i];
        double expected = x + x * x / (1.0 + exp(-x));
        if (!(fabs(output[i] - expected) <= 2e-5)) {
            fprintf(stderr, "Output %u: got %.8g, expected %.8g\n", i, output[i], expected);
            matches = 0;
        }
    }
    if (matches) {
        printf("%s: inference output [%.6f, %.6f; %.6f, %.6f] matches the reference\n",
               device.name, output[0], output[1], output[2], output[3]);
    }
    gurabo_tensor **tensors[] = {&table, &tokens, &mask, &embedded, &normalized, &scores, &attention, &gate};
    for (unsigned i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        check(gurabo_tensor_release(tensors[i]));
    }
    check(gurabo_buffer_release(&buffer));
    check(gurabo_stream_destroy(&stream));
    check(gurabo_context_destroy(&context));
    return matches ? EXIT_SUCCESS : EXIT_FAILURE;
}
