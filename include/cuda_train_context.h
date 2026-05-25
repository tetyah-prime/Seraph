// cuda_train_context.h - CUDA training context for GPU training
// Mirror of vulkan_context_t buffer management, but with CUDA runtime
// Part of TETYAH-PRIME's native training and inference engine

#ifndef CUDA_TRAIN_CONTEXT_H
#define CUDA_TRAIN_CONTEXT_H

#include <stddef.h>
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════════════════
// CUDA BUFFER
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    float* data;           // Device pointer
    size_t size;           // Size in bytes
} cuda_buffer_t;

// ═══════════════════════════════════════════════════════════════════════════
// LORA GPU STATE (defined before context so it can be used)
// ═══════════════════════════════════════════════════════════════════════════
// Per-adapter-type GPU buffer arrays (one entry per layer, NULL if inactive)

typedef struct {
    cuda_buffer_t** A;       // [num_layers] - weight A [rank × in_dim]
    cuda_buffer_t** B;       // [num_layers] - weight B [out_dim × rank]
    cuda_buffer_t** gA;      // [num_layers] - gradient A
    cuda_buffer_t** gB;      // [num_layers] - gradient B
    cuda_buffer_t** mA;      // [num_layers] - optimizer first moment A
    cuda_buffer_t** vA;      // [num_layers] - optimizer second moment A
    cuda_buffer_t** mB;      // [num_layers] - optimizer first moment B
    cuda_buffer_t** vB;      // [num_layers] - optimizer second moment B
    int active;              // 1 if this adapter type has GPU buffers
} cuda_lora_gpu_adapter_t;

typedef struct {
    int num_layers;
    int rank;
    float alpha;
    float scale;             // alpha / rank (precomputed)

    // 7 adapter types matching lora_model_t
    cuda_lora_gpu_adapter_t q;
    cuda_lora_gpu_adapter_t k;
    cuda_lora_gpu_adapter_t v;
    cuda_lora_gpu_adapter_t o;
    cuda_lora_gpu_adapter_t gate;
    cuda_lora_gpu_adapter_t up;
    cuda_lora_gpu_adapter_t down;
} cuda_lora_gpu_state_t;

// ═══════════════════════════════════════════════════════════════════════════
// CUDA TRAINING CONTEXT
// ═══════════════════════════════════════════════════════════════════════════
// Mirrors vulkan_context_t buffer layout exactly for train.c compatibility

typedef struct {
    int device_id;
    void* stream;              // cudaStream_t
    int initialized;

    // ── GPU Hardware Capabilities (populated at init from cudaDeviceProp) ──
    int gpu_sm_count;              // multiProcessorCount
    int gpu_max_threads_per_block; // maxThreadsPerBlock: typically 1024
    int gpu_shared_mem_per_block;  // sharedMemPerBlock in bytes: 48KB (3050), 164KB (A100)
    int gpu_shared_mem_per_sm;     // sharedMemPerMultiprocessor in bytes
    int gpu_warp_size;             // warpSize: 32 
    int gpu_compute_major;         // major compute capability: 8 = Ampere (TF32), 9 = Hopper
    int gpu_compute_minor;         // minor compute capability
    int gpu_l2_cache_bytes;        // l2CacheSize: persistent L2 config on Ampere+
    int gpu_max_shared_per_block_optin; // maxSharedMemoryPerBlockOptin: extended shared mem
    int gpu_max_threads_per_sm;        // maxThreadsPerMultiProcessor
    int gpu_regs_per_sm;               // regsPerMultiprocessor

    // ── Precision mode ──
    int use_bf16;                      // 1 = BF16 mixed precision (wmma tensor cores)
    int use_flash_attn;                // 1 = FlashAttention 

    // ── Batching ──
    int batch_size;                    // current batch size

    // ── Derived launch parameters (computed once at init from hardware caps) ──
    int matmul_tile_dim;           // tile width for shared-mem matmul: 16 (<=48KB) or 32 (>48KB)
    int matmul_block_k;            // K-dimension tile for register blocking
    int parallel_block_size;       // block size for rmsnorm/softmax/cross-entropy: 256 or 512
    int attention_block_size;      // block size for attention kernels: 128 or 256
    int attention_num_warps;       // attention_block_size / warp_size

    // Persistent buffer pool for AdamW optimizer
    cuda_buffer_t* weight_buffer;
    cuda_buffer_t* grad_buffer;
    cuda_buffer_t* m_buffer;
    cuda_buffer_t* v_buffer;
    size_t max_buffer_size;

    // Persistent buffers for forward pass matmul
    cuda_buffer_t* fwd_input_buffer;
    cuda_buffer_t* fwd_weight_buffer;
    cuda_buffer_t* fwd_output_buffer;
    size_t max_fwd_buffer_size;

    // Persistent buffers for batch attention
    cuda_buffer_t* attn_q_buffer;
    cuda_buffer_t* attn_k_buffer;
    cuda_buffer_t* attn_v_buffer;
    cuda_buffer_t* attn_out_buffer;
    size_t max_attn_buffer_size;

    // Persistent buffers for GPU-forward activations
    cuda_buffer_t* fwd_hidden_buffer;
    cuda_buffer_t* fwd_hidden_norm_buffer;
    cuda_buffer_t* fwd_tmp_hidden_buffer;
    cuda_buffer_t* fwd_ffn_gate_buffer;
    cuda_buffer_t* fwd_ffn_up_buffer;
    cuda_buffer_t* fwd_ffn_hidden_buffer;
    cuda_buffer_t* fwd_logits_buffer;
    cuda_buffer_t* fwd_norm_weight_buffer;
    size_t max_fwd_act_bytes;

    // Per-layer forward activation cache
    int fwd_cache_num_layers;
    int fwd_cache_max_seq;
    int fwd_cache_hidden_size;
    int fwd_cache_intermediate_size;
    int fwd_cache_q_dim;
    cuda_buffer_t** fwd_cache_x_norm_attn;    // [num_layers]
    cuda_buffer_t** fwd_cache_x_norm_ffn;     // [num_layers]
    cuda_buffer_t** fwd_cache_attn_out;       // [num_layers]
    cuda_buffer_t** fwd_cache_ffn_gate_out;   // [num_layers]
    cuda_buffer_t** fwd_cache_ffn_up_out;     // [num_layers]
    cuda_buffer_t** fwd_cache_ffn_hidden;     // [num_layers]
    cuda_buffer_t** fwd_cache_layer_input;    // [num_layers]
    cuda_buffer_t** fwd_cache_post_attn_in;   // [num_layers]
    cuda_buffer_t** fwd_cache_silu_out;       // [num_layers]
    cuda_buffer_t** fwd_cache_q;              // [num_layers]
    cuda_buffer_t** fwd_cache_k;              // [num_layers]
    cuda_buffer_t** fwd_cache_v;              // [num_layers]
    cuda_buffer_t** fwd_cache_attn_lse;      // [num_layers] logsumexp for FlashAttention backward

    // Full-training GPU-only persistent buffers
    cuda_buffer_t* train_tokens_u32;
    cuda_buffer_t* train_targets_u32;
    cuda_buffer_t* train_grad_logits;
    cuda_buffer_t* train_loss_rows;
    cuda_buffer_t* train_reduce_tmp_a;
    cuda_buffer_t* train_reduce_tmp_b;
    cuda_buffer_t* train_seq_starts;       // [batch_size+1] uint32 prefix sums for packed sequences
    cuda_buffer_t* train_grad_hidden;
    cuda_buffer_t* train_grad_tmp_hidden;
    cuda_buffer_t* train_grad_x_norm;
    cuda_buffer_t* train_grad_q;
    cuda_buffer_t* train_grad_k;
    cuda_buffer_t* train_grad_v;
    cuda_buffer_t* train_grad_attn;
    cuda_buffer_t* train_grad_ffn;
    cuda_buffer_t* train_grad_gate;
    cuda_buffer_t* train_grad_up;
    cuda_buffer_t* train_final_input;

    // Full-training GPU-resident weights
    int gpu_weights_num_layers;
    int gpu_weights_hidden_size;
    int gpu_weights_intermediate_size;
    int gpu_weights_q_dim;
    int gpu_weights_kv_dim;
    int gpu_weights_vocab_size;

    cuda_buffer_t* gpu_w_embed;
    cuda_buffer_t* gpu_g_embed;
    cuda_buffer_t* gpu_m_embed;
    cuda_buffer_t* gpu_v_embed;

    cuda_buffer_t* gpu_w_final_norm;
    cuda_buffer_t* gpu_g_final_norm;
    cuda_buffer_t* gpu_m_final_norm;
    cuda_buffer_t* gpu_v_final_norm;

    // Per-layer arrays [num_layers]
    cuda_buffer_t** gpu_w_q;
    cuda_buffer_t** gpu_g_q;
    cuda_buffer_t** gpu_m_q;
    cuda_buffer_t** gpu_v_q;

    cuda_buffer_t** gpu_w_k;
    cuda_buffer_t** gpu_g_k;
    cuda_buffer_t** gpu_m_k;
    cuda_buffer_t** gpu_v_k;

    cuda_buffer_t** gpu_w_v;
    cuda_buffer_t** gpu_g_v;
    cuda_buffer_t** gpu_m_v;
    cuda_buffer_t** gpu_v_v;

    cuda_buffer_t** gpu_w_o;
    cuda_buffer_t** gpu_g_o;
    cuda_buffer_t** gpu_m_o;
    cuda_buffer_t** gpu_v_o;

    cuda_buffer_t** gpu_w_gate;
    cuda_buffer_t** gpu_g_gate;
    cuda_buffer_t** gpu_m_gate;
    cuda_buffer_t** gpu_v_gate;

    cuda_buffer_t** gpu_w_up;
    cuda_buffer_t** gpu_g_up;
    cuda_buffer_t** gpu_m_up;
    cuda_buffer_t** gpu_v_up;

    cuda_buffer_t** gpu_w_down;
    cuda_buffer_t** gpu_g_down;
    cuda_buffer_t** gpu_m_down;
    cuda_buffer_t** gpu_v_down;

    cuda_buffer_t** gpu_w_in_norm;
    cuda_buffer_t** gpu_g_in_norm;
    cuda_buffer_t** gpu_m_in_norm;
    cuda_buffer_t** gpu_v_in_norm;

    cuda_buffer_t** gpu_w_post_norm;
    cuda_buffer_t** gpu_g_post_norm;
    cuda_buffer_t** gpu_m_post_norm;
    cuda_buffer_t** gpu_v_post_norm;

    // Debug: NaN/Inf checking
    cuda_buffer_t* debug_nan_flags;

    // LoRA GPU state (for LoRA training mode)
    void* lora_gpu;                        // cuda_lora_gpu_state_t* (opaque)
    cuda_buffer_t* lora_hidden_tmp;        // Temporary buffer for LoRA backward [max_seq × rank]
    size_t lora_hidden_tmp_bytes;

} cuda_train_context_t;

// ═══════════════════════════════════════════════════════════════════════════
// CONTEXT MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

cuda_train_context_t* cuda_train_init(int device_id);
void cuda_train_free(cuda_train_context_t* ctx);
void cuda_train_sync(cuda_train_context_t* ctx);
void cuda_train_get_gpu_info(cuda_train_context_t* ctx, char* name, int name_len, float* memory_gb);

// ═══════════════════════════════════════════════════════════════════════════
// BUFFER MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

cuda_buffer_t* cuda_train_buffer_create(cuda_train_context_t* ctx, size_t size);
void cuda_train_buffer_free(cuda_train_context_t* ctx, cuda_buffer_t* buf);
void cuda_train_buffer_upload(cuda_train_context_t* ctx, cuda_buffer_t* buf, const float* data, size_t count);
void cuda_train_buffer_upload_u32(cuda_train_context_t* ctx, cuda_buffer_t* buf, const uint32_t* data, size_t count);
void cuda_train_buffer_download(cuda_train_context_t* ctx, cuda_buffer_t* buf, float* data, size_t count);
void cuda_train_buffer_download_bytes(cuda_train_context_t* ctx, cuda_buffer_t* buf, void* data, size_t bytes);
void cuda_train_fill_buffer(cuda_train_context_t* ctx, cuda_buffer_t* buf, uint32_t value, size_t bytes);
void cuda_train_fill_buffer_range(cuda_train_context_t* ctx, cuda_buffer_t* buf, size_t offset, uint32_t value, size_t bytes);
void cuda_train_copy_buffer(cuda_train_context_t* ctx, cuda_buffer_t* src, cuda_buffer_t* dst, size_t bytes);

// ═══════════════════════════════════════════════════════════════════════════
// COMPUTE KERNELS
// ═══════════════════════════════════════════════════════════════════════════

// Forward pass
void cuda_train_matmul(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B, cuda_buffer_t* C, int M, int N, int K);
void cuda_train_matmul_transpose(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B, cuda_buffer_t* C, int M, int N, int K, int transpose_A, int transpose_B, int accumulate);
void cuda_train_silu(cuda_train_context_t* ctx, cuda_buffer_t* input, cuda_buffer_t* output, size_t count);
void cuda_train_rmsnorm(cuda_train_context_t* ctx, cuda_buffer_t* input, cuda_buffer_t* scale, cuda_buffer_t* output, int rows, int cols, float eps);
void cuda_train_softmax(cuda_train_context_t* ctx, cuda_buffer_t* input, cuda_buffer_t* output, int rows, int cols);
void cuda_train_rope(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out, int seq_len, int num_heads, int head_dim, float rope_theta);
void cuda_train_add(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B, cuda_buffer_t* C, size_t count);
void cuda_train_mul(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B, cuda_buffer_t* C, size_t count);
void cuda_train_scale(cuda_train_context_t* ctx, cuda_buffer_t* A, size_t count, float alpha);
void cuda_train_batch_attention(cuda_train_context_t* ctx, cuda_buffer_t* Q, cuda_buffer_t* K, cuda_buffer_t* V, cuda_buffer_t* out, cuda_buffer_t* lse_out, int seq_len, int num_heads, int kv_heads, int head_dim);
void cuda_train_batch_attention_packed(cuda_train_context_t* ctx, cuda_buffer_t* Q, cuda_buffer_t* K, cuda_buffer_t* V, cuda_buffer_t* out, cuda_buffer_t* lse_out, cuda_buffer_t* seq_starts, int total_tokens, int batch_size, int num_heads, int kv_heads, int head_dim, int max_seq_in_batch);
void cuda_train_embed_lookup(cuda_train_context_t* ctx, cuda_buffer_t* tokens_u32, cuda_buffer_t* embed_weight, cuda_buffer_t* out_hidden, int seq_len, int hidden_size, int vocab_size);
void cuda_train_rope_packed(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out, cuda_buffer_t* seq_starts, int total_tokens, int batch_size, int num_heads, int head_dim, float rope_theta);

// Loss + backward
void cuda_train_cross_entropy_grad(cuda_train_context_t* ctx, cuda_buffer_t* logits, cuda_buffer_t* targets_u32, cuda_buffer_t* grad_logits, cuda_buffer_t* loss_rows, int rows, int cols);
void cuda_train_reduce_sum(cuda_train_context_t* ctx, cuda_buffer_t* in_buf, cuda_buffer_t* out_buf, size_t count);
void cuda_train_silu_backward(cuda_train_context_t* ctx, cuda_buffer_t* input, cuda_buffer_t* grad_output, cuda_buffer_t* grad_input, size_t count);
void cuda_train_elementwise_mul_backward(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B, cuda_buffer_t* grad_C, cuda_buffer_t* grad_A, cuda_buffer_t* grad_B, size_t count);
void cuda_train_rmsnorm_backward_batch(cuda_train_context_t* ctx, cuda_buffer_t* input, cuda_buffer_t* scale, cuda_buffer_t* grad_output, cuda_buffer_t* grad_input, cuda_buffer_t* grad_scale, int rows, int cols, float eps);
void cuda_train_rope_backward(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out, int seq_len, int num_heads, int head_dim, float rope_theta);
void cuda_train_batch_attention_backward(cuda_train_context_t* ctx, cuda_buffer_t* Q, cuda_buffer_t* K, cuda_buffer_t* V, cuda_buffer_t* attn_out, cuda_buffer_t* lse, cuda_buffer_t* grad_out, cuda_buffer_t* grad_Q, cuda_buffer_t* grad_K, cuda_buffer_t* grad_V, int seq_len, int num_heads, int kv_heads, int head_dim);
void cuda_train_batch_attention_backward_packed(cuda_train_context_t* ctx, cuda_buffer_t* Q, cuda_buffer_t* K, cuda_buffer_t* V, cuda_buffer_t* attn_out, cuda_buffer_t* lse, cuda_buffer_t* grad_out, cuda_buffer_t* grad_Q, cuda_buffer_t* grad_K, cuda_buffer_t* grad_V, cuda_buffer_t* seq_starts, int total_tokens, int batch_size, int num_heads, int kv_heads, int head_dim, int max_seq_in_batch);
void cuda_train_rope_backward_packed(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out, cuda_buffer_t* seq_starts, int total_tokens, int batch_size, int num_heads, int head_dim, float rope_theta);
void cuda_train_embedding_backward(cuda_train_context_t* ctx, cuda_buffer_t* tokens_u32, cuda_buffer_t* grad_hidden, cuda_buffer_t* grad_embed, int seq_len, int hidden_size, int vocab_size);
void cuda_train_cross_entropy_grad_packed(cuda_train_context_t* ctx, cuda_buffer_t* logits, cuda_buffer_t* targets_u32, cuda_buffer_t* grad_logits, cuda_buffer_t* loss_rows, cuda_buffer_t* seq_starts, int total_tokens, int batch_size, int vocab_size, int num_valid_predictions);

// Optimizer
void cuda_train_adamw_update(cuda_train_context_t* ctx, cuda_buffer_t* weight, cuda_buffer_t* grad, cuda_buffer_t* m, cuda_buffer_t* v, float lr, float beta1, float beta2, float weight_decay, float eps, int step, size_t size, const char* tensor_name, int rows, int cols);

// Debug
void cuda_train_nan_check(cuda_train_context_t* ctx, cuda_buffer_t* input, size_t count_floats, cuda_buffer_t* flags_u32, uint32_t flag_index);

// Profiler (set to non-NULL to enable per-kernel timing)
// See cuda_profiler.h for cuda_profiler_t definition
typedef struct cuda_profiler_t cuda_profiler_t;
void cuda_train_set_profiler(cuda_train_context_t* ctx, cuda_profiler_t* prof);

// ═══════════════════════════════════════════════════════════════════════════
// LORA OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

// LoRA backward: computes gradients for A and B matrices
// grad_output [seq, out_dim], input [seq, in_dim], A [rank, in_dim], B [out_dim, rank]
// grad_B += scale * grad_output^T @ hidden  (hidden = input @ A^T recomputed)
// grad_A += scale * (B^T @ grad_output)^T @ input
void cuda_train_lora_backward(cuda_train_context_t* ctx,
                               cuda_buffer_t* grad_output,
                               cuda_buffer_t* input,
                               cuda_buffer_t* A,
                               cuda_buffer_t* B,
                               cuda_buffer_t* gA,
                               cuda_buffer_t* gB,
                               cuda_buffer_t* hidden_tmp,
                               int seq_len, int in_dim, int out_dim, int rank,
                               float scale);

// Zero all LoRA gradients (call at start of accumulation cycle)
void cuda_train_zero_lora_grads(cuda_train_context_t* ctx);

// Download LoRA weights from GPU to CPU (for checkpointing)
// Note: lora_model_t defined in lora.h - include it before using this function
void cuda_download_lora_weights(cuda_train_context_t* ctx, void* lora);

#endif // CUDA_TRAIN_CONTEXT_H
