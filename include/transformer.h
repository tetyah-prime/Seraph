// transformer.h - Transformer model loading and forward pass
// Memory-optimized BF16 native weights with zero-copy mmap loading
// Supports Llama, Mistral, Qwen, Granite, DeepSeek, and compatible architectures
// Part of TETYAH-PRIME's native training and inference engine

#ifndef SERAPH_TRANSFORMER_H
#define SERAPH_TRANSFORMER_H

#include "safetensors.h"
#include "model_config.h"
#include "lora.h"
#include <stdint.h>

// Tensor location lookup (maps tensor name to safetensors file index)
typedef struct {
    char name[256];    // Normalized name (e.g., "model.layers.0.self_attn.q_proj.weight")
    char orig_name[256]; // Original name in safetensors file (for raw lookup)
    int file_idx;      // Which safetensors file (0-3)
} tensor_location_t;

// KV Cache for autoregressive attention
typedef struct {
    float* k;           // [num_layers, max_seq_len, kv_dim]
    float* v;           // [num_layers, max_seq_len, kv_dim]
    int max_seq_len;
    int cur_seq_len;
    int num_layers;
    int kv_dim;
} kv_cache_t;

// Model handle - BF16 weights with zero-copy mmap
typedef struct {
    safetensors_t* st[8];  // Up to 8 sharded safetensors files
    int num_files;

    tensor_location_t* tensor_map;
    int num_tensors;

    model_config_t* config;
    kv_cache_t* kv_cache;
    lora_model_t* lora;
    void* act_cache;       // Training activation cache (NULL during inference)
} seraph_model_t;

// Load model from safetensors (memory-mapped, zero-copy)
seraph_model_t* seraph_load_model(const model_config_t* config);

// Free model resources
void seraph_free_model(seraph_model_t* model);

// Forward pass - returns logits for last position
// Manages KV cache internally for autoregressive generation
float* seraph_forward(seraph_model_t* model, const model_config_t* config, int* token_ids, int seq_len);

// Reset KV cache (call between conversations)
void seraph_reset_cache(seraph_model_t* model);

#endif // SERAPH_TRANSFORMER_H
