// lora.h - Low-Rank Adaptation (LoRA) for fine-tuning
// Trainable low-rank matrices applied to frozen base model weights
// Reference: https://arxiv.org/abs/2106.09685
// Part of TETYAH-PRIME's native training and inference engine

#ifndef TETYAH_LORA_H
#define TETYAH_LORA_H

#include <stdint.h>
#include "model_config.h"  // For config-based creation

// LoRA adapter for a single weight matrix
typedef struct {
    int in_dim;         // Input dimension
    int out_dim;        // Output dimension
    int rank;           // LoRA rank (typically 4, 8, 16, 32)
    float alpha;        // Scaling factor (typically rank or 2*rank)

    // Trainable parameters (float32 for training stability)
    float* A;           // [rank x in_dim] - initialized with gaussian
    float* B;           // [out_dim x rank] - initialized with zeros

    // Gradients
    float* grad_A;      // Gradient for A
    float* grad_B;      // Gradient for B

    // Optimizer state (AdamW)
    float* m_A;         // First moment for A
    float* v_A;         // Second moment for A
    float* m_B;         // First moment for B
    float* v_B;         // Second moment for B
} lora_adapter_t;

// LoRA config for a full model
typedef struct {
    int num_layers;
    int rank;
    float alpha;
    float dropout;      // Dropout rate during training

    // Which layers to apply LoRA to
    int apply_to_q;     // Apply to Q projection
    int apply_to_k;     // Apply to K projection
    int apply_to_v;     // Apply to V projection
    int apply_to_o;     // Apply to O projection
    int apply_to_ffn;   // Apply to FFN layers

    // Per-layer adapters
    lora_adapter_t** q_adapters;    // [num_layers]
    lora_adapter_t** k_adapters;    // [num_layers]
    lora_adapter_t** v_adapters;    // [num_layers]
    lora_adapter_t** o_adapters;    // [num_layers]
    lora_adapter_t** gate_adapters; // [num_layers] for FFN gate
    lora_adapter_t** up_adapters;   // [num_layers] for FFN up
    lora_adapter_t** down_adapters; // [num_layers] for FFN down
} lora_model_t;

// Create/free LoRA adapter
lora_adapter_t* lora_adapter_create(int in_dim, int out_dim, int rank, float alpha);
void lora_adapter_free(lora_adapter_t* adapter);

// Create LoRA model from config 
lora_model_t* lora_model_create_from_config(const model_config_t* config, int rank, float alpha);

void lora_model_free(lora_model_t* lora);

// Forward pass: compute BA @ x and add to frozen output
void lora_forward(lora_adapter_t* adapter, const float* input, float* output, int batch_size);

// Backward pass: compute gradients for A and B
void lora_backward(lora_adapter_t* adapter, const float* input, const float* grad_output,
                   float* grad_input, int batch_size);

// Zero gradients before backward pass
void lora_zero_grad(lora_model_t* lora);

// Optimizer step (AdamW)
void lora_optimizer_step(lora_model_t* lora, float lr, float beta1, float beta2,
                         float weight_decay, float eps, int step);

// Save/load LoRA weights 
int lora_save(lora_model_t* lora, const char* path);
lora_model_t* lora_load(const char* path);

// Merge LoRA into base weights (for inference without adapter overhead)
void lora_merge_into_base(lora_adapter_t* adapter, uint16_t* base_weight);

#endif // TETYAH_LORA_H
