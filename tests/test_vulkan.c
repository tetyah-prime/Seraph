// test_vulkan.c - Test Vulkan backend
// Part of TETYAH-PRIME's native training and inference engine

#include "../include/vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  🔥⚡ SERAPH VULKAN BACKEND TEST ⚡🔥\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    // Test 1: Initialize Vulkan
    printf("[1/4] Initializing Vulkan context...\n");
    vulkan_context_t* ctx = vulkan_init(0);  // 0 = use default pool size
    if (!ctx) {
        fprintf(stderr, "❌ FAILED: Could not initialize Vulkan\n");
        return 1;
    }
    printf("  ✅ Vulkan initialized\n\n");

    // Test 2: Create buffers
    printf("[2/4] Creating GPU buffers...\n");
    size_t test_size = 1024;  // 1K floats = 4KB
    vulkan_buffer_t* buf_a = vulkan_buffer_create(ctx, test_size * sizeof(float));
    vulkan_buffer_t* buf_b = vulkan_buffer_create(ctx, test_size * sizeof(float));

    if (!buf_a || !buf_b) {
        fprintf(stderr, "❌ FAILED: Could not create buffers\n");
        vulkan_free(ctx);
        return 1;
    }
    printf("  ✅ Created 2 buffers (%.2f KB each)\n\n",
           (test_size * sizeof(float)) / 1024.0);

    // Test 3: Upload data to GPU
    printf("[3/4] Testing CPU → GPU transfer...\n");
    float* host_data_a = malloc(test_size * sizeof(float));
    float* host_data_b = malloc(test_size * sizeof(float));

    // Fill with test pattern
    for (size_t i = 0; i < test_size; i++) {
        host_data_a[i] = (float)i;
        host_data_b[i] = (float)(i * 2);
    }

    vulkan_buffer_upload(ctx, buf_a, host_data_a, test_size);
    vulkan_buffer_upload(ctx, buf_b, host_data_b, test_size);
    printf("  ✅ Uploaded %.2f KB to GPU\n\n",
           (test_size * sizeof(float) * 2) / 1024.0);

    // Test 4: Download data from GPU
    printf("[4/4] Testing GPU → CPU transfer...\n");
    float* verify_a = malloc(test_size * sizeof(float));
    float* verify_b = malloc(test_size * sizeof(float));

    vulkan_buffer_download(ctx, buf_a, verify_a, test_size);
    vulkan_buffer_download(ctx, buf_b, verify_b, test_size);

    // Verify data integrity
    int errors = 0;
    for (size_t i = 0; i < test_size; i++) {
        if (fabsf(verify_a[i] - host_data_a[i]) > 1e-5f ||
            fabsf(verify_b[i] - host_data_b[i]) > 1e-5f) {
            errors++;
        }
    }

    if (errors > 0) {
        fprintf(stderr, "❌ FAILED: %d data mismatches\n", errors);
    } else {
        printf("  ✅ Downloaded and verified %.2f KB\n",
               (test_size * sizeof(float) * 2) / 1024.0);
    }

    // Cleanup
    free(host_data_a);
    free(host_data_b);
    free(verify_a);
    free(verify_b);
    vulkan_buffer_free(ctx, buf_a);
    vulkan_buffer_free(ctx, buf_b);
    vulkan_free(ctx);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    if (errors == 0) {
        printf("  ✅ ALL TESTS PASSED - Vulkan backend operational!\n");
    } else {
        printf("  ❌ TEST FAILED\n");
    }
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    return errors > 0 ? 1 : 0;
}
