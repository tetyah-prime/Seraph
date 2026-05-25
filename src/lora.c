// lora.c - Low-Rank Adaptation for Seraph
// LoRA adds trainable low-rank matrices to frozen base model
//
// For weight W, LoRA computes: output = W @ x + (B @ A) @ x * (alpha/rank)
// Where A is [rank x in_dim] and B is [out_dim x rank]
//
// Part of TETYAH-PRIME's native training and inference engine

#include "../include/lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════════
// RANDOM INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

// Gaussian random (Box-Muller)
static float rand_gaussian(void) {
    static int have_spare = 0;
    static float spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    float u, v, s;
    do {
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    have_spare = 1;
    return u * s;
}

// Kaiming initialization for A matrix
static void init_kaiming(float* weights, int size, int fan_in) {
    float std = sqrtf(2.0f / fan_in);
    for (int i = 0; i < size; i++) {
        weights[i] = rand_gaussian() * std;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTER CREATION
// ═══════════════════════════════════════════════════════════════════════════

lora_adapter_t* lora_adapter_create(int in_dim, int out_dim, int rank, float alpha) {
    lora_adapter_t* adapter = calloc(1, sizeof(lora_adapter_t));

    adapter->in_dim = in_dim;
    adapter->out_dim = out_dim;
    adapter->rank = rank;
    adapter->alpha = alpha;

    // Allocate weights
    adapter->A = calloc(rank * in_dim, sizeof(float));
    adapter->B = calloc(out_dim * rank, sizeof(float));

    // Allocate gradients
    adapter->grad_A = calloc(rank * in_dim, sizeof(float));
    adapter->grad_B = calloc(out_dim * rank, sizeof(float));

    // Allocate optimizer state (AdamW)
    adapter->m_A = calloc(rank * in_dim, sizeof(float));
    adapter->v_A = calloc(rank * in_dim, sizeof(float));
    adapter->m_B = calloc(out_dim * rank, sizeof(float));
    adapter->v_B = calloc(out_dim * rank, sizeof(float));

    // Initialize A with Kaiming, B with zeros (standard LoRA init)
    init_kaiming(adapter->A, rank * in_dim, in_dim);
    // B stays zero so LoRA starts as identity

    return adapter;
}

void lora_adapter_free(lora_adapter_t* adapter) {
    if (!adapter) return;

    free(adapter->A);
    free(adapter->B);
    free(adapter->grad_A);
    free(adapter->grad_B);
    free(adapter->m_A);
    free(adapter->v_A);
    free(adapter->m_B);
    free(adapter->v_B);
    free(adapter);
}

// ═══════════════════════════════════════════════════════════════════════════
// FORWARD PASS
// ═══════════════════════════════════════════════════════════════════════════

// Compute: output += (B @ A @ input) * (alpha / rank)
// input: [in_dim]
// output: [out_dim] (already contains frozen W @ input)
void lora_forward(lora_adapter_t* adapter, const float* input, float* output, int batch_size) {
    int rank = adapter->rank;
    int in_dim = adapter->in_dim;
    int out_dim = adapter->out_dim;
    float scale = adapter->alpha / rank;

    // Temp buffer for A @ input
    float* hidden = calloc(rank, sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        const float* x = &input[b * in_dim];
        float* y = &output[b * out_dim];

        // hidden = A @ x  [rank] = [rank x in_dim] @ [in_dim]
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += adapter->A[r * in_dim + i] * x[i];
            }
            hidden[r] = sum;
        }

        // y += B @ hidden * scale  [out_dim] += [out_dim x rank] @ [rank]
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (int r = 0; r < rank; r++) {
                sum += adapter->B[o * rank + r] * hidden[r];
            }
            y[o] += sum * scale;
        }
    }

    free(hidden);
}

// ═══════════════════════════════════════════════════════════════════════════
// BACKWARD PASS
// ═══════════════════════════════════════════════════════════════════════════

// Compute gradients for A and B, and gradient w.r.t. input
// grad_output: [out_dim] gradient from later layers
// Computes:
//   grad_B += grad_output @ hidden^T
//   grad_hidden = B^T @ grad_output
//   grad_A += grad_hidden @ input^T
//   grad_input += A^T @ grad_hidden (if needed)
void lora_backward(lora_adapter_t* adapter, const float* input, const float* grad_output,
                   float* grad_input, int batch_size) {
    int rank = adapter->rank;
    int in_dim = adapter->in_dim;
    int out_dim = adapter->out_dim;
    float scale = adapter->alpha / rank;

    float* hidden = calloc(rank, sizeof(float));
    float* grad_hidden = calloc(rank, sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        const float* x = &input[b * in_dim];
        const float* gy = &grad_output[b * out_dim];
        float* gx = grad_input ? &grad_input[b * in_dim] : NULL;

        // Forward pass to get hidden (needed for grad_B)
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += adapter->A[r * in_dim + i] * x[i];
            }
            hidden[r] = sum;
        }

        // grad_B += gy @ hidden^T (outer product)
        for (int o = 0; o < out_dim; o++) {
            for (int r = 0; r < rank; r++) {
                adapter->grad_B[o * rank + r] += gy[o] * hidden[r] * scale;
            }
        }

        // grad_hidden = B^T @ gy
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            for (int o = 0; o < out_dim; o++) {
                sum += adapter->B[o * rank + r] * gy[o];
            }
            grad_hidden[r] = sum * scale;
        }

        // grad_A += grad_hidden @ x^T (outer product)
        for (int r = 0; r < rank; r++) {
            for (int i = 0; i < in_dim; i++) {
                adapter->grad_A[r * in_dim + i] += grad_hidden[r] * x[i];
            }
        }

        // grad_input += A^T @ grad_hidden (if needed)
        if (gx) {
            for (int i = 0; i < in_dim; i++) {
                float sum = 0.0f;
                for (int r = 0; r < rank; r++) {
                    sum += adapter->A[r * in_dim + i] * grad_hidden[r];
                }
                gx[i] += sum;
            }
        }
    }

    free(hidden);
    free(grad_hidden);
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZER (AdamW)
// ═══════════════════════════════════════════════════════════════════════════

static void adamw_update(float* param, float* grad, float* m, float* v,
                         int size, float lr, float beta1, float beta2,
                         float weight_decay, float eps, int step) {
    float bias_correction1 = 1.0f - powf(beta1, step);
    float bias_correction2 = 1.0f - powf(beta2, step);

    for (int i = 0; i < size; i++) {
        // AdamW: apply weight decay before update
        param[i] -= lr * weight_decay * param[i];

        // Update biased first moment
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];

        // Update biased second moment
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

        // Bias-corrected estimates
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        // Update parameter
        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void lora_optimizer_step(lora_model_t* lora, float lr, float beta1, float beta2,
                         float weight_decay, float eps, int step) {
    // Update all adapters
    for (int l = 0; l < lora->num_layers; l++) {
        if (lora->q_adapters && lora->q_adapters[l]) {
            lora_adapter_t* a = lora->q_adapters[l];
            adamw_update(a->A, a->grad_A, a->m_A, a->v_A,
                         a->rank * a->in_dim, lr, beta1, beta2, weight_decay, eps, step);
            adamw_update(a->B, a->grad_B, a->m_B, a->v_B,
                         a->out_dim * a->rank, lr, beta1, beta2, weight_decay, eps, step);
        }
        if (lora->k_adapters && lora->k_adapters[l]) {
            lora_adapter_t* a = lora->k_adapters[l];
            adamw_update(a->A, a->grad_A, a->m_A, a->v_A,
                         a->rank * a->in_dim, lr, beta1, beta2, weight_decay, eps, step);
            adamw_update(a->B, a->grad_B, a->m_B, a->v_B,
                         a->out_dim * a->rank, lr, beta1, beta2, weight_decay, eps, step);
        }
        if (lora->v_adapters && lora->v_adapters[l]) {
            lora_adapter_t* a = lora->v_adapters[l];
            adamw_update(a->A, a->grad_A, a->m_A, a->v_A,
                         a->rank * a->in_dim, lr, beta1, beta2, weight_decay, eps, step);
            adamw_update(a->B, a->grad_B, a->m_B, a->v_B,
                         a->out_dim * a->rank, lr, beta1, beta2, weight_decay, eps, step);
        }
        if (lora->o_adapters && lora->o_adapters[l]) {
            lora_adapter_t* a = lora->o_adapters[l];
            adamw_update(a->A, a->grad_A, a->m_A, a->v_A,
                         a->rank * a->in_dim, lr, beta1, beta2, weight_decay, eps, step);
            adamw_update(a->B, a->grad_B, a->m_B, a->v_B,
                         a->out_dim * a->rank, lr, beta1, beta2, weight_decay, eps, step);
        }
        if (lora->gate_adapters && lora->gate_adapters[l]) {
            lora_adapter_t* a = lora->gate_adapters[l];
            adamw_update(a->A, a->grad_A, a->m_A, a->v_A,
                         a->rank * a->in_dim, lr, beta1, beta2, weight_decay, eps, step);
            adamw_update(a->B, a->grad_B, a->m_B, a->v_B,
                         a->out_dim * a->rank, lr, beta1, beta2, weight_decay, eps, step);
        }
        if (lora->up_adapters && lora->up_adapters[l]) {
            lora_adapter_t* a = lora->up_adapters[l];
            adamw_update(a->A, a->grad_A, a->m_A, a->v_A,
                         a->rank * a->in_dim, lr, beta1, beta2, weight_decay, eps, step);
            adamw_update(a->B, a->grad_B, a->m_B, a->v_B,
                         a->out_dim * a->rank, lr, beta1, beta2, weight_decay, eps, step);
        }
        if (lora->down_adapters && lora->down_adapters[l]) {
            lora_adapter_t* a = lora->down_adapters[l];
            adamw_update(a->A, a->grad_A, a->m_A, a->v_A,
                         a->rank * a->in_dim, lr, beta1, beta2, weight_decay, eps, step);
            adamw_update(a->B, a->grad_B, a->m_B, a->v_B,
                         a->out_dim * a->rank, lr, beta1, beta2, weight_decay, eps, step);
        }
    }
}

void lora_zero_grad(lora_model_t* lora) {
    for (int l = 0; l < lora->num_layers; l++) {
        if (lora->q_adapters && lora->q_adapters[l]) {
            lora_adapter_t* a = lora->q_adapters[l];
            memset(a->grad_A, 0, a->rank * a->in_dim * sizeof(float));
            memset(a->grad_B, 0, a->out_dim * a->rank * sizeof(float));
        }
        if (lora->k_adapters && lora->k_adapters[l]) {
            lora_adapter_t* a = lora->k_adapters[l];
            memset(a->grad_A, 0, a->rank * a->in_dim * sizeof(float));
            memset(a->grad_B, 0, a->out_dim * a->rank * sizeof(float));
        }
        if (lora->v_adapters && lora->v_adapters[l]) {
            lora_adapter_t* a = lora->v_adapters[l];
            memset(a->grad_A, 0, a->rank * a->in_dim * sizeof(float));
            memset(a->grad_B, 0, a->out_dim * a->rank * sizeof(float));
        }
        if (lora->o_adapters && lora->o_adapters[l]) {
            lora_adapter_t* a = lora->o_adapters[l];
            memset(a->grad_A, 0, a->rank * a->in_dim * sizeof(float));
            memset(a->grad_B, 0, a->out_dim * a->rank * sizeof(float));
        }
        if (lora->gate_adapters && lora->gate_adapters[l]) {
            lora_adapter_t* a = lora->gate_adapters[l];
            memset(a->grad_A, 0, a->rank * a->in_dim * sizeof(float));
            memset(a->grad_B, 0, a->out_dim * a->rank * sizeof(float));
        }
        if (lora->up_adapters && lora->up_adapters[l]) {
            lora_adapter_t* a = lora->up_adapters[l];
            memset(a->grad_A, 0, a->rank * a->in_dim * sizeof(float));
            memset(a->grad_B, 0, a->out_dim * a->rank * sizeof(float));
        }
        if (lora->down_adapters && lora->down_adapters[l]) {
            lora_adapter_t* a = lora->down_adapters[l];
            memset(a->grad_A, 0, a->rank * a->in_dim * sizeof(float));
            memset(a->grad_B, 0, a->out_dim * a->rank * sizeof(float));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL CREATION - takes config for dimension handling
// ═══════════════════════════════════════════════════════════════════════════

lora_model_t* lora_model_create_from_config(const model_config_t* config, int rank, float alpha) {
    lora_model_t* lora = calloc(1, sizeof(lora_model_t));

    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;
    int intermediate_size = config->intermediate_size;

    lora->num_layers = num_layers;
    lora->rank = rank;
    lora->alpha = alpha;

    // Default: apply LoRA to Q, V (most common configuration)
    lora->apply_to_q = 1;
    lora->apply_to_v = 1;
    lora->apply_to_k = 0;
    lora->apply_to_o = 0;
    lora->apply_to_ffn = 0;

    // Allocate adapter arrays
    lora->q_adapters = calloc(num_layers, sizeof(lora_adapter_t*));
    lora->k_adapters = calloc(num_layers, sizeof(lora_adapter_t*));
    lora->v_adapters = calloc(num_layers, sizeof(lora_adapter_t*));
    lora->o_adapters = calloc(num_layers, sizeof(lora_adapter_t*));
    lora->gate_adapters = calloc(num_layers, sizeof(lora_adapter_t*));
    lora->up_adapters = calloc(num_layers, sizeof(lora_adapter_t*));
    lora->down_adapters = calloc(num_layers, sizeof(lora_adapter_t*));

    // Create adapters with dimensions from config
    for (int l = 0; l < num_layers; l++) {
        if (lora->apply_to_q) {
            // Q: [hidden_size] -> [q_dim]
            lora->q_adapters[l] = lora_adapter_create(hidden_size, q_dim, rank, alpha);
        }
        if (lora->apply_to_k) {
            // K: [hidden_size] -> [kv_dim] (GQA uses fewer KV heads!)
            lora->k_adapters[l] = lora_adapter_create(hidden_size, kv_dim, rank, alpha);
        }
        if (lora->apply_to_v) {
            // V: [hidden_size] -> [kv_dim] (GQA uses fewer KV heads!)
            lora->v_adapters[l] = lora_adapter_create(hidden_size, kv_dim, rank, alpha);
        }
        if (lora->apply_to_o) {
            // O: [q_dim] -> [hidden_size]
            lora->o_adapters[l] = lora_adapter_create(q_dim, hidden_size, rank, alpha);
        }
        if (lora->apply_to_ffn) {
            // Gate: [hidden_size] -> [intermediate_size]
            lora->gate_adapters[l] = lora_adapter_create(hidden_size, intermediate_size, rank, alpha);
            // Up: [hidden_size] -> [intermediate_size]
            lora->up_adapters[l] = lora_adapter_create(hidden_size, intermediate_size, rank, alpha);
            // Down: [intermediate_size] -> [hidden_size]
            lora->down_adapters[l] = lora_adapter_create(intermediate_size, hidden_size, rank, alpha);
        }
    }

    // Count trainable params
    long trainable = 0;
    if (lora->apply_to_q) trainable += num_layers * 2 * rank * hidden_size;  // A + B for Q
    if (lora->apply_to_k) trainable += num_layers * 2 * rank * hidden_size;
    if (lora->apply_to_v) trainable += num_layers * 2 * rank * hidden_size;
    if (lora->apply_to_o) trainable += num_layers * 2 * rank * q_dim;
    if (lora->apply_to_ffn) trainable += num_layers * 6 * rank * hidden_size;  // gate, up, down

    printf("  Created LoRA model: %d layers, rank=%d, alpha=%.1f\n", num_layers, rank, alpha);
    printf("  Q-dim: %d, KV-dim: %d (GQA: %s)\n", q_dim, kv_dim, kv_dim < q_dim ? "yes" : "no");
    printf("  Trainable params: %.2fM\n", (float)trainable / 1e6);

    return lora;
}

void lora_model_free(lora_model_t* lora) {
    if (!lora) return;

    for (int l = 0; l < lora->num_layers; l++) {
        if (lora->q_adapters) lora_adapter_free(lora->q_adapters[l]);
        if (lora->k_adapters) lora_adapter_free(lora->k_adapters[l]);
        if (lora->v_adapters) lora_adapter_free(lora->v_adapters[l]);
        if (lora->o_adapters) lora_adapter_free(lora->o_adapters[l]);
        if (lora->gate_adapters) lora_adapter_free(lora->gate_adapters[l]);
        if (lora->up_adapters) lora_adapter_free(lora->up_adapters[l]);
        if (lora->down_adapters) lora_adapter_free(lora->down_adapters[l]);
    }

    free(lora->q_adapters);
    free(lora->k_adapters);
    free(lora->v_adapters);
    free(lora->o_adapters);
    free(lora->gate_adapters);
    free(lora->up_adapters);
    free(lora->down_adapters);
    free(lora);
}

// ═══════════════════════════════════════════════════════════════════════════
// SAVE/LOAD
// ═══════════════════════════════════════════════════════════════════════════

int lora_save(lora_model_t* lora, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Header v2 - self-describing
    uint32_t magic = 0x4C4F5241;  // "LORA"
    uint32_t version = 2;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&lora->num_layers, sizeof(int), 1, f);
    fwrite(&lora->rank, sizeof(int), 1, f);
    fwrite(&lora->alpha, sizeof(float), 1, f);

    // Get dims from first Q adapter 
    int hidden_size = 0, q_out_dim = 0, kv_out_dim = 0;
    if (lora->q_adapters && lora->q_adapters[0]) {
        hidden_size = lora->q_adapters[0]->in_dim;
        q_out_dim = lora->q_adapters[0]->out_dim;
    }
    if (lora->v_adapters && lora->v_adapters[0]) {
        kv_out_dim = lora->v_adapters[0]->out_dim;
    }

    fwrite(&hidden_size, sizeof(int), 1, f);
    fwrite(&q_out_dim, sizeof(int), 1, f);
    fwrite(&kv_out_dim, sizeof(int), 1, f);

    // Apply mask: Q=1, K=2, V=4, O=8, FFN=16
    uint32_t apply_mask = 0;
    if (lora->apply_to_q) apply_mask |= 1;
    if (lora->apply_to_k) apply_mask |= 2;
    if (lora->apply_to_v) apply_mask |= 4;
    if (lora->apply_to_o) apply_mask |= 8;
    if (lora->apply_to_ffn) apply_mask |= 16;
    fwrite(&apply_mask, sizeof(uint32_t), 1, f);

    // Save each adapter
    for (int l = 0; l < lora->num_layers; l++) {
        if (lora->q_adapters && lora->q_adapters[l]) {
            lora_adapter_t* a = lora->q_adapters[l];
            fwrite(a->A, sizeof(float), a->rank * a->in_dim, f);
            fwrite(a->B, sizeof(float), a->out_dim * a->rank, f);
        }
        if (lora->v_adapters && lora->v_adapters[l]) {
            lora_adapter_t* a = lora->v_adapters[l];
            fwrite(a->A, sizeof(float), a->rank * a->in_dim, f);
            fwrite(a->B, sizeof(float), a->out_dim * a->rank, f);
        }
    }

    fclose(f);
    printf("  Saved LoRA weights: %s\n", path);
    return 0;
}

lora_model_t* lora_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open LoRA file: %s\n", path);
        return NULL;
    }

    // Read header
    uint32_t magic, version;
    int file_num_layers, rank;
    float alpha;

    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x4C4F5241) {
        fprintf(stderr, "Invalid LoRA file magic\n");
        fclose(f);
        return NULL;
    }

    fread(&version, sizeof(uint32_t), 1, f);

    if (version != 2) {
        fprintf(stderr, "Unsupported LoRA format version: %u \n", version);
        fclose(f);
        return NULL;
    }

    int file_hidden_size, file_q_out_dim, file_kv_out_dim;
    uint32_t apply_mask;

    fread(&file_num_layers, sizeof(int), 1, f);
    fread(&rank, sizeof(int), 1, f);
    fread(&alpha, sizeof(float), 1, f);
    fread(&file_hidden_size, sizeof(int), 1, f);
    fread(&file_q_out_dim, sizeof(int), 1, f);
    fread(&file_kv_out_dim, sizeof(int), 1, f);
    fread(&apply_mask, sizeof(uint32_t), 1, f);

    // Allocate LoRA model
    lora_model_t* lora = calloc(1, sizeof(lora_model_t));
    lora->num_layers = file_num_layers;
    lora->rank = rank;
    lora->alpha = alpha;
    lora->apply_to_q = (apply_mask & 1) ? 1 : 0;
    lora->apply_to_k = (apply_mask & 2) ? 1 : 0;
    lora->apply_to_v = (apply_mask & 4) ? 1 : 0;
    lora->apply_to_o = (apply_mask & 8) ? 1 : 0;
    lora->apply_to_ffn = (apply_mask & 16) ? 1 : 0;

    // Allocate adapter arrays
    lora->q_adapters = calloc(file_num_layers, sizeof(lora_adapter_t*));
    lora->v_adapters = calloc(file_num_layers, sizeof(lora_adapter_t*));
    lora->k_adapters = calloc(file_num_layers, sizeof(lora_adapter_t*));
    lora->o_adapters = calloc(file_num_layers, sizeof(lora_adapter_t*));
    lora->gate_adapters = calloc(file_num_layers, sizeof(lora_adapter_t*));
    lora->up_adapters = calloc(file_num_layers, sizeof(lora_adapter_t*));
    lora->down_adapters = calloc(file_num_layers, sizeof(lora_adapter_t*));

    // Read adapters for each layer
    for (int l = 0; l < file_num_layers; l++) {
        if (lora->apply_to_q) {
            lora->q_adapters[l] = lora_adapter_create(file_hidden_size, file_q_out_dim, rank, alpha);
            fread(lora->q_adapters[l]->A, sizeof(float), rank * file_hidden_size, f);
            fread(lora->q_adapters[l]->B, sizeof(float), file_q_out_dim * rank, f);
        }
        if (lora->apply_to_v) {
            lora->v_adapters[l] = lora_adapter_create(file_hidden_size, file_kv_out_dim, rank, alpha);
            fread(lora->v_adapters[l]->A, sizeof(float), rank * file_hidden_size, f);
            fread(lora->v_adapters[l]->B, sizeof(float), file_kv_out_dim * rank, f);
        }
    }

    fclose(f);
    printf("  Loaded LoRA weights: %s (v2, rank=%d, alpha=%.1f, mask=0x%x)\n",
           path, rank, alpha, apply_mask);

    return lora;
}

// ═══════════════════════════════════════════════════════════════════════════
// MERGE INTO BASE (for inference without adapter overhead)
// ═══════════════════════════════════════════════════════════════════════════

// Merge LoRA into base: W_new = W + B @ A * (alpha / rank)
// This permanently modifies the base weight so no adapter needed at inference
void lora_merge_into_base(lora_adapter_t* adapter, uint16_t* base_weight) {
    int rank = adapter->rank;
    int in_dim = adapter->in_dim;
    int out_dim = adapter->out_dim;
    float scale = adapter->alpha / rank;

    // Compute BA and add to base weight
    for (int o = 0; o < out_dim; o++) {
        for (int i = 0; i < in_dim; i++) {
            float delta = 0.0f;
            for (int r = 0; r < rank; r++) {
                delta += adapter->B[o * rank + r] * adapter->A[r * in_dim + i];
            }
            delta *= scale;

            // Convert base BF16 to float, add delta, convert back
            uint32_t bf16_bits = ((uint32_t)base_weight[o * in_dim + i]) << 16;
            float base_val;
            memcpy(&base_val, &bf16_bits, sizeof(float));

            base_val += delta;

            uint32_t new_bits;
            memcpy(&new_bits, &base_val, sizeof(float));
            base_weight[o * in_dim + i] = (uint16_t)(new_bits >> 16);
        }
    }
}
