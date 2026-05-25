// merge_lora.c - Seraph LoRA Merge Tool
// Merges trained LoRA weights into base model safetensors
// Creates new standalone model with learned behaviors baked in
//
// Usage: seraph-merge <model_dir> <lora.bin> <output.safetensors>
//
// Part of TETYAH-PRIME's native training and inference engine
// THE HOLY GRAIL: Self-upgrade capability for NLP engines

#include "../include/transformer.h"
#include "../include/model_config.h"
#include "../include/safetensors.h"
#include "../include/lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════════════════
// BF16 CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
}

static inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    memcpy(&bits, &f32, sizeof(float));
    return (uint16_t)(bits >> 16);
}

// ═══════════════════════════════════════════════════════════════════════════
// LORA LOADING
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    int num_layers;
    int rank;
    float alpha;

    // Per-layer A and B matrices for Q and V
    float** q_A;  // [num_layers][rank * in_dim]
    float** q_B;  // [num_layers][out_dim * rank]
    float** v_A;
    float** v_B;

    int in_dim;   // hidden_size
    int out_dim;  // q_dim (num_heads * head_dim)
} lora_weights_t;

lora_weights_t* load_lora_weights(const char* path, int num_layers, int hidden_size, int q_dim) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open LoRA file: %s\n", path);
        return NULL;
    }

    // Read header
    uint32_t magic, version;
    fread(&magic, sizeof(uint32_t), 1, f);
    if (magic != 0x4C4F5241) {  // "LORA"
        fprintf(stderr, "ERROR: Invalid LoRA file (bad magic)\n");
        fclose(f);
        return NULL;
    }

    lora_weights_t* lora = calloc(1, sizeof(lora_weights_t));
    fread(&version, sizeof(uint32_t), 1, f);

    if (version == 2) {
        // v2 format: self-describing header
        fread(&lora->num_layers, sizeof(int), 1, f);
        fread(&lora->rank, sizeof(int), 1, f);
        fread(&lora->alpha, sizeof(float), 1, f);
        // Skip v2 extra fields: hidden_size, q_out_dim, kv_out_dim, apply_mask
        int skip_int; uint32_t skip_mask;
        fread(&skip_int, sizeof(int), 1, f);  // hidden_size
        fread(&skip_int, sizeof(int), 1, f);  // q_out_dim
        fread(&skip_int, sizeof(int), 1, f);  // kv_out_dim
        fread(&skip_mask, sizeof(uint32_t), 1, f);  // apply_mask
        printf("  LoRA v2: %d layers, rank=%d, alpha=%.1f\n", lora->num_layers, lora->rank, lora->alpha);
    } else {
        // v1 format: version field was actually num_layers
        lora->num_layers = (int)version;
        fread(&lora->rank, sizeof(int), 1, f);
        fread(&lora->alpha, sizeof(float), 1, f);
        printf("  LoRA v1: %d layers, rank=%d, alpha=%.1f\n", lora->num_layers, lora->rank, lora->alpha);
    }

    if (lora->num_layers != num_layers) {
        fprintf(stderr, "ERROR: LoRA layers (%d) != model layers (%d)\n", lora->num_layers, num_layers);
        free(lora);
        fclose(f);
        return NULL;
    }

    lora->in_dim = hidden_size;
    lora->out_dim = q_dim;

    // Allocate and load weights
    lora->q_A = calloc(num_layers, sizeof(float*));
    lora->q_B = calloc(num_layers, sizeof(float*));
    lora->v_A = calloc(num_layers, sizeof(float*));
    lora->v_B = calloc(num_layers, sizeof(float*));

    int rank = lora->rank;
    for (int l = 0; l < num_layers; l++) {
        // Q adapter
        lora->q_A[l] = malloc(rank * hidden_size * sizeof(float));
        lora->q_B[l] = malloc(q_dim * rank * sizeof(float));
        fread(lora->q_A[l], sizeof(float), rank * hidden_size, f);
        fread(lora->q_B[l], sizeof(float), q_dim * rank, f);

        // V adapter
        lora->v_A[l] = malloc(rank * hidden_size * sizeof(float));
        lora->v_B[l] = malloc(q_dim * rank * sizeof(float));
        fread(lora->v_A[l], sizeof(float), rank * hidden_size, f);
        fread(lora->v_B[l], sizeof(float), q_dim * rank, f);
    }

    fclose(f);
    return lora;
}

void free_lora_weights(lora_weights_t* lora) {
    if (!lora) return;
    for (int l = 0; l < lora->num_layers; l++) {
        free(lora->q_A[l]);
        free(lora->q_B[l]);
        free(lora->v_A[l]);
        free(lora->v_B[l]);
    }
    free(lora->q_A);
    free(lora->q_B);
    free(lora->v_A);
    free(lora->v_B);
    free(lora);
}

// ═══════════════════════════════════════════════════════════════════════════
// MERGE LORA INTO WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════

// Merge: W_new = W + B @ A * (alpha / rank)
// W is [out_dim x in_dim], A is [rank x in_dim], B is [out_dim x rank]
void merge_lora_into_weight(uint16_t* weight_bf16, float* A, float* B,
                            int out_dim, int in_dim, int rank, float scale) {
    printf("    Merging [%d x %d] with rank-%d LoRA (scale=%.3f)...\n",
           out_dim, in_dim, rank, scale);

    // For each element of the weight matrix
    for (int o = 0; o < out_dim; o++) {
        for (int i = 0; i < in_dim; i++) {
            // Compute (B @ A)[o, i] = sum_r B[o, r] * A[r, i]
            float delta = 0.0f;
            for (int r = 0; r < rank; r++) {
                delta += B[o * rank + r] * A[r * in_dim + i];
            }
            delta *= scale;

            // Add to base weight
            float base_val = bf16_to_f32(weight_bf16[o * in_dim + i]);
            weight_bf16[o * in_dim + i] = f32_to_bf16(base_val + delta);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

void print_usage(const char* prog) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Seraph LoRA Merge Tool\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("Usage: %s <model_dir> <lora.bin> <output_dir>\n", prog);
    printf("\n");
    printf("Merges trained LoRA weights into base model, creating a new\n");
    printf("standalone model with learned behaviors permanently baked in.\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s models/tetyah-prime-1.5b checkpoints/lora_final.bin models/tetyah-prime-1.5b-merged\n", prog);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* lora_path = argv[2];
    const char* output_dir = argv[3];

    // Banner
    printf("\n");
    printf("\033[38;5;196m╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                   ║\n");
    printf("║         \033[1m███████╗███████╗██████╗  █████╗ ██████╗ ██╗  ██╗\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║  ██║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m███████╗█████╗  ██████╔╝███████║██████╔╝███████║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m╚════██║██╔══╝  ██╔══██╗██╔══██║██╔═══╝ ██╔══██║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m███████║███████╗██║  ██║██║  ██║██║     ██║  ██║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝\033[0m\033[38;5;196m          ║\n");
    printf("║                                                                   ║\n");
    printf("║                      🧬⚡ LORA MERGE ⚡🧬                         ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\033[0m\n");
    printf("\n");

    // Load model config
    printf("[1/5] Loading model configuration...\n");
    model_config_t* config = model_config_load(model_dir);
    if (!config) {
        fprintf(stderr, "ERROR: Failed to load model config\n");
        return 1;
    }

    int hidden_size = config->hidden_size;
    int num_layers = config->num_hidden_layers;
    int q_dim = config->num_attention_heads * config->head_dim;

    printf("  Model: %s\n", config->model_type);
    printf("  Layers: %d, Hidden: %d, Q-dim: %d\n", num_layers, hidden_size, q_dim);

    // Load LoRA weights
    printf("\n[2/5] Loading LoRA weights...\n");
    lora_weights_t* lora = load_lora_weights(lora_path, num_layers, hidden_size, q_dim);
    if (!lora) {
        model_config_free(config);
        return 1;
    }

    float scale = lora->alpha / lora->rank;
    printf("  Merge scale: %.3f (alpha/rank = %.1f/%d)\n", scale, lora->alpha, lora->rank);

    // Load base model safetensors
    printf("\n[3/5] Loading base model weights...\n");
    char st_path[512];
    snprintf(st_path, sizeof(st_path), "%s/model.safetensors", model_dir);

    safetensors_t* st = safetensors_load(st_path);
    if (!st) {
        fprintf(stderr, "ERROR: Failed to load %s\n", st_path);
        free_lora_weights(lora);
        model_config_free(config);
        return 1;
    }
    printf("  Loaded %d tensors\n", st->num_tensors);

    // Merge LoRA into Q and V projection weights
    printf("\n[4/5] Merging LoRA into weights...\n");

    // Get actual V dimension from config (for GQA models, V uses kv_dim not q_dim)
    int kv_dim = config->num_key_value_heads * config->head_dim;
    printf("  Q dimension: %d, V dimension: %d\n", q_dim, kv_dim);

    for (int l = 0; l < num_layers; l++) {
        printf("  Layer %d/%d:\n", l + 1, num_layers);

        // Merge into Q projection
        char name[256];
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
        uint16_t* q_weight = (uint16_t*)safetensors_get_tensor_raw(st, name);
        if (q_weight) {
            merge_lora_into_weight(q_weight, lora->q_A[l], lora->q_B[l],
                                   q_dim, hidden_size, lora->rank, scale);
        } else {
            printf("    WARNING: %s not found\n", name);
        }

        // Merge into V projection (NOTE: V uses kv_dim in GQA models!)
        // But our LoRA was trained with q_dim, so we only merge up to kv_dim rows
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
        uint16_t* v_weight = (uint16_t*)safetensors_get_tensor_raw(st, name);
        if (v_weight) {
            // For GQA: V projection is [kv_dim x hidden_size]
            // Our LoRA adapter B is [q_dim x rank], but we only use first kv_dim rows
            // This is a workaround for LoRA trained with wrong dimensions
            int v_out_dim = (kv_dim < q_dim) ? kv_dim : q_dim;
            merge_lora_into_weight(v_weight, lora->v_A[l], lora->v_B[l],
                                   v_out_dim, hidden_size, lora->rank, scale);
        } else {
            printf("    WARNING: %s not found\n", name);
        }
    }

    // Create output directory and save
    printf("\n[5/5] Saving merged model...\n");

    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir);
    system(cmd);

    // Copy config files
    snprintf(cmd, sizeof(cmd), "cp %s/config.json %s/", model_dir, output_dir);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "cp %s/tokenizer*.json %s/ 2>/dev/null", model_dir, output_dir);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "cp %s/vocab.json %s/ 2>/dev/null", model_dir, output_dir);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "cp %s/merges.txt %s/ 2>/dev/null", model_dir, output_dir);
    system(cmd);

    // Save merged safetensors
    char out_path[512];
    snprintf(out_path, sizeof(out_path), "%s/model.safetensors", output_dir);

    if (safetensors_save(st, out_path) != 0) {
        fprintf(stderr, "ERROR: Failed to save merged model\n");
        safetensors_free(st);
        free_lora_weights(lora);
        model_config_free(config);
        return 1;
    }

    // Cleanup
    safetensors_free(st);
    free_lora_weights(lora);
    model_config_free(config);

    printf("\n");
    printf("\033[38;5;196m═══════════════════════════════════════════════════════════════════\033[0m\n");
    printf("  🧬⚡ \033[1mMERGE COMPLETE\033[0m ⚡🧬\n");
    printf("\033[38;5;196m═══════════════════════════════════════════════════════════════════\033[0m\n");
    printf("\n");
    printf("  Output: %s\n", output_dir);
    printf("\n");
    printf("  Your evolved model is ready!\n");
    printf("  Run with: ./seraph-chat %s\n", output_dir);
    printf("\n");

    return 0;
}
