// test_matmul.c - Test GPU matrix multiply
// Part of TETYAH-PRIME's native training and inference engine
// Verify: C = A @ B on Radeon GPU

#include "../include/vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// CPU reference implementation
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  🔥⚡ SERAPH VULKAN MATMUL TEST ⚡🔥\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    // Test dimensions (small for verification)
    int M = 64, N = 64, K = 64;
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    printf("[1/6] Initializing Vulkan...\n");
    vulkan_context_t* ctx = vulkan_init(0);  // 0 = use default pool size
    if (!ctx) return 1;

    printf("[2/6] Creating test matrices (%dx%d @ %dx%d)...\n", M, K, K, N);
    float* A = malloc(size_A * sizeof(float));
    float* B = malloc(size_B * sizeof(float));
    float* C_gpu = calloc(size_C, sizeof(float));
    float* C_cpu = calloc(size_C, sizeof(float));

    // Fill with random values
    srand(time(NULL));
    for (size_t i = 0; i < size_A; i++) A[i] = (float)(rand() % 10) - 5.0f;
    for (size_t i = 0; i < size_B; i++) B[i] = (float)(rand() % 10) - 5.0f;

    printf("[3/6] Computing CPU reference...\n");
    clock_t start = clock();
    matmul_cpu(A, B, C_cpu, M, N, K);
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("  CPU: %.2f ms\n", cpu_time);

    printf("[4/6] Creating GPU buffers...\n");
    vulkan_buffer_t* buf_A = vulkan_buffer_create(ctx, size_A * sizeof(float));
    vulkan_buffer_t* buf_B = vulkan_buffer_create(ctx, size_B * sizeof(float));
    vulkan_buffer_t* buf_C = vulkan_buffer_create(ctx, size_C * sizeof(float));

    printf("[5/6] Uploading to GPU and computing...\n");
    vulkan_buffer_upload(ctx, buf_A, A, size_A);
    vulkan_buffer_upload(ctx, buf_B, B, size_B);

    vulkan_matmul(ctx, buf_A, buf_B, buf_C, M, N, K);

    vulkan_buffer_download(ctx, buf_C, C_gpu, size_C);

    printf("[6/6] Verifying results...\n");
    int errors = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_C; i++) {
        float diff = fabsf(C_gpu[i] - C_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) errors++;
    }

    printf("  Max difference: %.6f\n", max_diff);
    printf("  Errors: %d / %zu\n", errors, size_C);

    // Cleanup
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);
    vulkan_buffer_free(ctx, buf_A);
    vulkan_buffer_free(ctx, buf_B);
    vulkan_buffer_free(ctx, buf_C);
    vulkan_free(ctx);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    if (errors == 0) {
        printf("  ✅ MATMUL TEST PASSED!\n");
    } else {
        printf("  ❌ TEST FAILED (%d errors)\n", errors);
    }
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    return errors > 0 ? 1 : 0;
}
