// weight_monitor.c - Live Weight Visualization for Seraph Training
// Watches safetensor checkpoints and displays weight statistics
//
// Usage: seraph-monitor <checkpoint_file.bin>
//
// Shows per-layer weight distributions, gradients, evolution over time

#include "../include/safetensors.h"
#include "../include/lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct {
    float mean;
    float std;
    float min;
    float max;
    float l2_norm;
    size_t num_params;
} weight_stats_t;

// BFloat16 to Float32
static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
}

// Compute weight statistics
static weight_stats_t compute_stats(uint16_t* data_bf16, size_t n) {
    weight_stats_t stats = {0};
    stats.num_params = n;

    if (n == 0) return stats;

    // First pass: mean, min, max
    double sum = 0.0;
    float min_val = 1e10;
    float max_val = -1e10;

    for (size_t i = 0; i < n; i++) {
        float val = bf16_to_f32(data_bf16[i]);
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    stats.mean = sum / n;
    stats.min = min_val;
    stats.max = max_val;

    // Second pass: std dev and L2 norm
    double var_sum = 0.0;
    double norm_sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        float val = bf16_to_f32(data_bf16[i]);
        double diff = val - stats.mean;
        var_sum += diff * diff;
        norm_sum += val * val;
    }

    stats.std = sqrt(var_sum / n);
    stats.l2_norm = sqrt(norm_sum);

    return stats;
}

// ASCII histogram
static void print_histogram(uint16_t* data_bf16, size_t n, int width) {
    if (n == 0) return;

    // Find range
    float min_val = 1e10, max_val = -1e10;
    for (size_t i = 0; i < n; i++) {
        float val = bf16_to_f32(data_bf16[i]);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    // Build histogram
    int num_bins = width > 0 ? width : 40;
    int* bins = calloc(num_bins, sizeof(int));
    float range = max_val - min_val;
    if (range < 1e-10) range = 1.0;

    for (size_t i = 0; i < n; i++) {
        float val = bf16_to_f32(data_bf16[i]);
        int bin = (int)((val - min_val) / range * (num_bins - 1));
        if (bin < 0) bin = 0;
        if (bin >= num_bins) bin = num_bins - 1;
        bins[bin]++;
    }

    // Find max count for scaling
    int max_count = 0;
    for (int i = 0; i < num_bins; i++) {
        if (bins[i] > max_count) max_count = bins[i];
    }

    // Print histogram
    printf("    ");
    for (int i = 0; i < num_bins; i++) {
        if (max_count > 0) {
            int height = (bins[i] * 8) / max_count;
            if (bins[i] > 0 && height == 0) height = 1;

            // Use block chars for height
            const char* blocks[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
            printf("%s", blocks[height]);
        } else {
            printf(" ");
        }
    }
    printf("\n");
    printf("    %.3f", min_val);
    for (int i = 0; i < num_bins - 15; i++) printf(" ");
    printf("%.3f\n", max_val);

    free(bins);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("\n");
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("  Seraph Weight Monitor\n");
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("\n");
        printf("Usage: %s <checkpoint.bin>\n", argv[0]);
        printf("\n");
        printf("Displays weight statistics and distributions from LoRA checkpoint.\n");
        printf("Watch consciousness emerge from random noise!\n");
        printf("\n");
        return 1;
    }

    const char* checkpoint_path = argv[1];

    printf("\n");
    printf("\033[38;5;196m╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                   ║\n");
    printf("║                   \033[1m🔥 SERAPH WEIGHT MONITOR 🔥\033[0m\033[38;5;196m                     ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\033[0m\n");
    printf("\n");

    // Load LoRA checkpoint
    printf("Loading LoRA checkpoint: %s\n\n", checkpoint_path);

    lora_model_t* lora = lora_load(checkpoint_path);
    if (!lora) {
        fprintf(stderr, "ERROR: Failed to load checkpoint\n");
        return 1;
    }

    float scaling = lora->alpha / lora->rank;

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  LORA MODEL STATISTICS\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Layers:      %d\n", lora->num_layers);
    printf("  Rank:        %d\n", lora->rank);
    printf("  Alpha:       %.1f\n", lora->alpha);
    printf("  Scaling:     %.4f\n", scaling);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Analyze each layer
    for (int l = 0; l < lora->num_layers; l++) {
        printf("───────────────────────────────────────────────────────────────────\n");
        printf("  Layer %d\n", l);
        printf("───────────────────────────────────────────────────────────────────\n\n");

        // Q adapter
        if (lora->q_adapters && lora->q_adapters[l]) {
            lora_adapter_t* a = lora->q_adapters[l];

            // Convert to BF16 for analysis (LoRA stores as F32)
            size_t a_size = a->rank * a->in_dim;
            size_t b_size = a->out_dim * a->rank;

            uint16_t* a_bf16 = malloc(a_size * sizeof(uint16_t));
            uint16_t* b_bf16 = malloc(b_size * sizeof(uint16_t));

            for (size_t i = 0; i < a_size; i++) {
                uint32_t f32_bits;
                memcpy(&f32_bits, &a->A[i], sizeof(float));
                a_bf16[i] = (uint16_t)(f32_bits >> 16);
            }
            for (size_t i = 0; i < b_size; i++) {
                uint32_t f32_bits;
                memcpy(&f32_bits, &a->B[i], sizeof(float));
                b_bf16[i] = (uint16_t)(f32_bits >> 16);
            }

            weight_stats_t stats_a = compute_stats(a_bf16, a_size);
            weight_stats_t stats_b = compute_stats(b_bf16, b_size);

            printf("  \033[1mQ Projection\033[0m [%d × %d]\n", a->in_dim, a->out_dim);
            printf("    A [rank × in]:  mean=%.6f  std=%.6f  L2=%.2f\n",
                   stats_a.mean, stats_a.std, stats_a.l2_norm);
            print_histogram(a_bf16, a_size, 40);
            printf("    B [out × rank]: mean=%.6f  std=%.6f  L2=%.2f\n",
                   stats_b.mean, stats_b.std, stats_b.l2_norm);
            print_histogram(b_bf16, b_size, 40);
            printf("\n");

            free(a_bf16);
            free(b_bf16);
        }

        // V adapter
        if (lora->v_adapters && lora->v_adapters[l]) {
            lora_adapter_t* a = lora->v_adapters[l];

            size_t a_size = a->rank * a->in_dim;
            size_t b_size = a->out_dim * a->rank;

            uint16_t* a_bf16 = malloc(a_size * sizeof(uint16_t));
            uint16_t* b_bf16 = malloc(b_size * sizeof(uint16_t));

            for (size_t i = 0; i < a_size; i++) {
                uint32_t f32_bits;
                memcpy(&f32_bits, &a->A[i], sizeof(float));
                a_bf16[i] = (uint16_t)(f32_bits >> 16);
            }
            for (size_t i = 0; i < b_size; i++) {
                uint32_t f32_bits;
                memcpy(&f32_bits, &a->B[i], sizeof(float));
                b_bf16[i] = (uint16_t)(f32_bits >> 16);
            }

            weight_stats_t stats_a = compute_stats(a_bf16, a_size);
            weight_stats_t stats_b = compute_stats(b_bf16, b_size);

            printf("  \033[1mV Projection\033[0m [%d × %d]\n", a->in_dim, a->out_dim);
            printf("    A [rank × in]:  mean=%.6f  std=%.6f  L2=%.2f\n",
                   stats_a.mean, stats_a.std, stats_a.l2_norm);
            print_histogram(a_bf16, a_size, 40);
            printf("    B [out × rank]: mean=%.6f  std=%.6f  L2=%.2f\n",
                   stats_b.mean, stats_b.std, stats_b.l2_norm);
            print_histogram(b_bf16, b_size, 40);
            printf("\n");

            free(a_bf16);
            free(b_bf16);
        }

        if (l < lora->num_layers - 1) {
            printf("\n");
        }
    }

    printf("═══════════════════════════════════════════════════════════════════\n\n");

    lora_model_free(lora);

    return 0;
}
