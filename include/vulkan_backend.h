// vulkan_backend.h - Vulkan compute backend for GPU training
// Hardware-agnostic GPU kernels for transformer forward, backward, and optimization
// Part of TETYAH-PRIME's native training and inference engine

#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

#include <stddef.h>
#include <stdint.h>

// Internal helper (shared across translation units)
uint32_t* load_shader_spirv(const char* shader_name, size_t* size);

// ═══════════════════════════════════════════════════════════════════════════
// VULKAN CONTEXT
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    void* device;              // VkDevice
    void* physical_device;     // VkPhysicalDevice
    void* instance;            // VkInstance
    void* command_pool;        // VkCommandPool
    void* queue;               // VkQueue
    uint32_t queue_family;
    int initialized;

    // ── GPU Hardware Capabilities (populated at init) ──
    char gpu_name[256];
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t device_type;          // 0=other 1=integrated 2=discrete 3=virtual 4=cpu

    // Compute limits
    uint32_t max_workgroup_size[3];        // maxComputeWorkGroupSize[3]
    uint32_t max_workgroup_invocations;    // maxComputeWorkGroupInvocations (total threads)
    uint32_t max_workgroup_count[3];       // maxComputeWorkGroupCount[3] (grid dims)
    uint32_t subgroup_size;                // warp/wavefront size (32 NVIDIA, 64 AMD)
    uint32_t max_subgroups_per_workgroup;  // max subgroups in a workgroup

    // Memory limits
    uint32_t max_shared_memory;            // maxComputeSharedMemorySize
    uint32_t max_push_constant_size;       // maxPushConstantsSize
    uint32_t max_bound_descriptor_sets;    // maxBoundDescriptorSets
    uint32_t max_storage_buffer_range;     // maxStorageBufferRange
    uint64_t vram_bytes;                   // total device-local heap
    uint64_t shared_system_bytes;          // host-visible heap (iGPU shared RAM)

    // Buffer/image limits
    uint32_t min_storage_buffer_offset;    // minStorageBufferOffsetAlignment
    uint32_t min_uniform_buffer_offset;    // minUniformBufferOffsetAlignment
    uint32_t max_storage_buffers_per_stage;
    uint32_t max_compute_units;            // SM count (NVIDIA) or CU count (AMD), 0 if unavailable
    uint32_t warps_per_compute_unit;       // warps/wavefronts per SM/CU (0 if unavailable)
    uint32_t min_subgroup_size;            // min subgroup size (subgroup size control)
    uint32_t max_subgroup_size;            // max subgroup size

    // Extension availability flags
    int has_sm_builtins;                   // VK_NV_shader_sm_builtins
    int has_shader_atomic_float;           // VK_EXT_shader_atomic_float
    int has_cooperative_matrix;            // VK_KHR_cooperative_matrix
    int has_shader_clock;                  // VK_KHR_shader_clock

    // Subgroup capabilities
    uint32_t subgroup_supported_stages;    // VkShaderStageFlags
    uint32_t subgroup_supported_ops;       // VkSubgroupFeatureFlags (ballot, shuffle, arithmetic)
    uint32_t subgroup_quad_ops;            // quad operations supported

    // Timestamps
    float timestamp_period;                // nanoseconds per timestamp tick (for profiling)
    uint32_t timestamp_valid_bits;         // bits of precision in timestamps

    // ── Derived workgroup sizes (computed at init from hardware query) ──
    uint32_t wg_size_1d;                   // optimal 1D workgroup size for elementwise ops
    uint32_t wg_size_attn;                 // attention workgroup size (= subgroup_size)
    uint32_t wg_size_matmul;              // matmul tile size (TILE)
    uint32_t matmul_reg;                   // register blocking factor (REG) — threads = (TILE/REG)²
    uint32_t matmul_bk;                    // K-dimension tile size (BK)

    // ── Profiler (GPU timestamps via VkQueryPool) ──
    int profiling;
    int prof_num_ops;                      // ops recorded this step
    void* prof_ops;                        // vulkan_profiler_op_t[VULKAN_PROFILER_MAX_OPS]
    void* prof_query_pool;                 // VkQueryPool for GPU timestamps
    void* prof_log;                        // FILE* for profile.log output

    // Pipeline cache (avoid recreating every call)
    void* matmul_pipeline;         // VkPipeline
    void* matmul_layout;           // VkPipelineLayout
    void* matmul_desc_layout;      // VkDescriptorSetLayout

    void* matmul_t_pipeline;       // VkPipeline - matmul with transpose
    void* matmul_t_layout;         // VkPipelineLayout
    void* matmul_t_desc_layout;    // VkDescriptorSetLayout

    void* adamw_pipeline;          // VkPipeline
    void* adamw_layout;            // VkPipelineLayout
    void* adamw_desc_layout;       // VkDescriptorSetLayout

    void* softmax_pipeline;        // VkPipeline
    void* softmax_layout;          // VkPipelineLayout
    void* softmax_desc_layout;     // VkDescriptorSetLayout

    void* add_pipeline;            // VkPipeline
    void* add_layout;              // VkPipelineLayout
    void* add_desc_layout;         // VkDescriptorSetLayout

    void* mul_pipeline;            // VkPipeline
    void* mul_layout;              // VkPipelineLayout
    void* mul_desc_layout;         // VkDescriptorSetLayout

    void* silu_pipeline;           // VkPipeline
    void* silu_layout;             // VkPipelineLayout
    void* silu_desc_layout;        // VkDescriptorSetLayout

    void* scale_pipeline;          // VkPipeline - in-place scale
    void* scale_layout;            // VkPipelineLayout
    void* scale_desc_layout;       // VkDescriptorSetLayout

    void* batch_attn_pipeline;     // VkPipeline - fused attention
    void* batch_attn_layout;       // VkPipelineLayout
    void* batch_attn_desc_layout;  // VkDescriptorSetLayout
    void* batch_attn_stream_pipeline;     // VkPipeline - streaming attention (no max seq)
    void* batch_attn_stream_layout;       // VkPipelineLayout
    void* batch_attn_stream_desc_layout;  // VkDescriptorSetLayout

	void* rope_pipeline;           // VkPipeline - rotary embeddings
	void* rope_layout;             // VkPipelineLayout
	void* rope_desc_layout;        // VkDescriptorSetLayout

	void* rmsnorm_pipeline;        // VkPipeline - RMSNorm forward
	void* rmsnorm_layout;          // VkPipelineLayout
	void* rmsnorm_desc_layout;     // VkDescriptorSetLayout

	// Backward/Training pipelines
	void* silu_backward_pipeline;      // VkPipeline
	void* silu_backward_layout;        // VkPipelineLayout
	void* silu_backward_desc_layout;   // VkDescriptorSetLayout

	void* mul_backward_pipeline;       // VkPipeline
	void* mul_backward_layout;         // VkPipelineLayout
	void* mul_backward_desc_layout;    // VkDescriptorSetLayout

	void* rmsnorm_bwd_pipeline;        // VkPipeline - RMSNorm backward (batched)
	void* rmsnorm_bwd_layout;          // VkPipelineLayout
	void* rmsnorm_bwd_desc_layout;     // VkDescriptorSetLayout

	void* embed_lookup_pipeline;       // VkPipeline
	void* embed_lookup_layout;         // VkPipelineLayout
	void* embed_lookup_desc_layout;    // VkDescriptorSetLayout

	void* cross_entropy_pipeline;      // VkPipeline
	void* cross_entropy_layout;        // VkPipelineLayout
	void* cross_entropy_desc_layout;   // VkDescriptorSetLayout

	void* reduce_sum_pipeline;         // VkPipeline
	void* reduce_sum_layout;           // VkPipelineLayout
	void* reduce_sum_desc_layout;      // VkDescriptorSetLayout

	void* rope_backward_pipeline;      // VkPipeline
	void* rope_backward_layout;        // VkPipelineLayout
	void* rope_backward_desc_layout;   // VkDescriptorSetLayout

	void* batch_attn_backward_pipeline;    // VkPipeline
	void* batch_attn_backward_layout;      // VkPipelineLayout
	void* batch_attn_backward_desc_layout; // VkDescriptorSetLayout
	void* batch_attn_backward_stream_pipeline;    // VkPipeline
	void* batch_attn_backward_stream_layout;      // VkPipelineLayout
	void* batch_attn_backward_stream_desc_layout; // VkDescriptorSetLayout

	void* embedding_backward_pipeline;     // VkPipeline
	void* embedding_backward_layout;       // VkPipelineLayout
	void* embedding_backward_desc_layout;  // VkDescriptorSetLayout

	void* descriptor_pool;         // VkDescriptorPool

    // Persistent buffer pool for AdamW optimizer (avoid alloc/free every call)
    void* weight_buffer;           // vulkan_buffer_t* - reusable weight buffer
    void* grad_buffer;             // vulkan_buffer_t* - reusable gradient buffer
    void* m_buffer;                // vulkan_buffer_t* - reusable momentum buffer
    void* v_buffer;                // vulkan_buffer_t* - reusable variance buffer
    size_t max_buffer_size;        // Maximum tensor size for optimizer buffers

    // Persistent buffers for forward pass matmul (separate from optimizer)
    void* fwd_input_buffer;        // vulkan_buffer_t* - forward pass input
    void* fwd_weight_buffer;       // vulkan_buffer_t* - forward pass weights
    void* fwd_output_buffer;       // vulkan_buffer_t* - forward pass output
    size_t max_fwd_buffer_size;    // Maximum size for forward pass buffers

	// Persistent buffers for batch attention
	void* attn_q_buffer;           // vulkan_buffer_t* - Q [seq_len, q_dim]
	void* attn_k_buffer;           // vulkan_buffer_t* - K [seq_len, kv_dim]
	void* attn_v_buffer;           // vulkan_buffer_t* - V [seq_len, kv_dim]
	void* attn_out_buffer;         // vulkan_buffer_t* - output [seq_len, q_dim]
	size_t max_attn_buffer_size;   // Maximum size for attention buffers

	// Persistent buffers for GPU-forward activations (full training)
	void* fwd_hidden_buffer;       // vulkan_buffer_t* - hidden [seq_len, hidden_size]
	void* fwd_hidden_norm_buffer;  // vulkan_buffer_t* - hidden_norm [seq_len, hidden_size]
	void* fwd_tmp_hidden_buffer;   // vulkan_buffer_t* - temp [seq_len, hidden_size]
	void* fwd_ffn_gate_buffer;     // vulkan_buffer_t* - gate [seq_len, intermediate]
	void* fwd_ffn_up_buffer;       // vulkan_buffer_t* - up [seq_len, intermediate]
	void* fwd_ffn_hidden_buffer;   // vulkan_buffer_t* - ffn_hidden [seq_len, intermediate]
	void* fwd_logits_buffer;       // vulkan_buffer_t* - logits [seq_len, vocab]
	void* fwd_norm_weight_buffer;  // vulkan_buffer_t* - norm scale [hidden_size]
	size_t max_fwd_act_bytes;      // Maximum activation bytes for forward buffers

	// Per-layer forward activation cache buffers 
	int fwd_cache_num_layers;
	int fwd_cache_max_seq;
	int fwd_cache_hidden_size;
	int fwd_cache_intermediate_size;
	int fwd_cache_q_dim;
	void* fwd_cache_x_norm_attn;   // vulkan_buffer_t* [num_layers]
	void* fwd_cache_x_norm_ffn;    // vulkan_buffer_t* [num_layers]
	void* fwd_cache_attn_out;      // vulkan_buffer_t* [num_layers]
	void* fwd_cache_ffn_gate_out;  // vulkan_buffer_t* [num_layers]
	void* fwd_cache_ffn_up_out;    // vulkan_buffer_t* [num_layers]
	void* fwd_cache_ffn_hidden;    // vulkan_buffer_t* [num_layers]
	void* fwd_cache_layer_input;   // vulkan_buffer_t* [num_layers] - pre input-norm
	void* fwd_cache_post_attn_in;  // vulkan_buffer_t* [num_layers] - pre post-norm (after attn residual)
	void* fwd_cache_silu_out;      // vulkan_buffer_t* [num_layers] - silu(gate) (before mul)
	void* fwd_cache_q;            // vulkan_buffer_t* [num_layers] - Q post-RoPE
	void* fwd_cache_k;            // vulkan_buffer_t* [num_layers] - K post-RoPE
	void* fwd_cache_v;            // vulkan_buffer_t* [num_layers] - V (no RoPE)

	// Full-training GPU-only persistent buffers (allocated on demand)
	void* train_tokens_u32;        // vulkan_buffer_t* [max_seq] uint32
	void* train_targets_u32;       // vulkan_buffer_t* [max_seq] uint32
	void* train_grad_logits;       // vulkan_buffer_t* [max_seq, vocab] float
	void* train_loss_rows;         // vulkan_buffer_t* [max_seq] float
	void* train_reduce_tmp_a;      // vulkan_buffer_t* [max_seq] float (reduce ping)
	void* train_reduce_tmp_b;      // vulkan_buffer_t* [max_seq] float (reduce pong)
	void* train_grad_hidden;       // vulkan_buffer_t* [max_seq, hidden] float
	void* train_grad_tmp_hidden;   // vulkan_buffer_t* [max_seq, hidden] float
	void* train_grad_x_norm;       // vulkan_buffer_t* [max_seq, hidden] float
	void* train_grad_q;            // vulkan_buffer_t* [max_seq, q_dim] float
	void* train_grad_k;            // vulkan_buffer_t* [max_seq, kv_dim] float
	void* train_grad_v;            // vulkan_buffer_t* [max_seq, kv_dim] float
	void* train_grad_attn;         // vulkan_buffer_t* [max_seq, q_dim] float
	void* train_grad_ffn;          // vulkan_buffer_t* [max_seq, intermediate] float
	void* train_grad_gate;         // vulkan_buffer_t* [max_seq, intermediate] float
	void* train_grad_up;           // vulkan_buffer_t* [max_seq, intermediate] float
	void* train_final_input;       // vulkan_buffer_t* [max_seq, hidden] float (pre final norm)

	// Full-training GPU-resident weights 
	int gpu_weights_num_layers;
	int gpu_weights_hidden_size;
	int gpu_weights_intermediate_size;
	int gpu_weights_q_dim;
	int gpu_weights_kv_dim;
	int gpu_weights_vocab_size;

	void* gpu_w_embed;             // vulkan_buffer_t*
	void* gpu_g_embed;             // vulkan_buffer_t*
	void* gpu_m_embed;             // vulkan_buffer_t*
	void* gpu_v_embed;             // vulkan_buffer_t*

	void* gpu_w_final_norm;        // vulkan_buffer_t*
	void* gpu_g_final_norm;        // vulkan_buffer_t*
	void* gpu_m_final_norm;        // vulkan_buffer_t*
	void* gpu_v_final_norm;        // vulkan_buffer_t*

	// Per-layer arrays (vulkan_buffer_t* [num_layers])
	void* gpu_w_q;
	void* gpu_g_q;
	void* gpu_m_q;
	void* gpu_v_q;

	void* gpu_w_k;
	void* gpu_g_k;
	void* gpu_m_k;
	void* gpu_v_k;

	void* gpu_w_v;
	void* gpu_g_v;
	void* gpu_m_v;
	void* gpu_v_v;

	void* gpu_w_o;
	void* gpu_g_o;
	void* gpu_m_o;
	void* gpu_v_o;

	void* gpu_w_gate;
	void* gpu_g_gate;
	void* gpu_m_gate;
	void* gpu_v_gate;

	void* gpu_w_up;
	void* gpu_g_up;
	void* gpu_m_up;
	void* gpu_v_up;

	void* gpu_w_down;
	void* gpu_g_down;
	void* gpu_m_down;
	void* gpu_v_down;

	void* gpu_w_in_norm;
	void* gpu_g_in_norm;
	void* gpu_m_in_norm;
	void* gpu_v_in_norm;

	void* gpu_w_post_norm;
	void* gpu_g_post_norm;
	void* gpu_m_post_norm;
	void* gpu_v_post_norm;

	// Command batching (single submit for many kernels)
	void* recording_cmd;           // VkCommandBuffer
	int recording;                 // 1 if recording into recording_cmd
	void* deferred_desc_sets;      // VkDescriptorSet[] allocated during recording
	uint32_t deferred_desc_count;  // number of deferred descriptor sets
	uint32_t deferred_desc_cap;    // capacity of deferred descriptor set array

	// LoRA GPU-resident adapter state (opaque, managed by train.c)
	void* lora_gpu;                // lora_gpu_state_t* (NULL when not using GPU LoRA)
	void* lora_hidden_tmp;         // vulkan_buffer_t* [max_seq × rank] temp for LoRA matmuls
	size_t lora_hidden_tmp_bytes;

	// Debug: NaN/Inf checking
	void* nan_check_pipeline;
	void* nan_check_layout;
	void* nan_check_desc_layout;
	void* debug_nan_flags;         // vulkan_buffer_t* (uint32 flags[])

	// QK norm (video/ViT mode) - normalize Q,K across all heads before attention
	void* qk_norm_pipeline;
	void* qk_norm_layout;
	void* qk_norm_desc_layout;

	// Patch extraction (video mode) - extract 3D patches from frame
	void* patch_extract_pipeline;
	void* patch_extract_layout;
	void* patch_extract_desc_layout;

	// Add bias (broadcast add)
	void* add_bias_pipeline;
	void* add_bias_layout;
	void* add_bias_desc_layout;

	// BCE loss + grad (mask segmentation)
	void* bce_loss_pipeline;
	void* bce_loss_layout;
	void* bce_loss_desc_layout;

	// Sum columns (bias gradient)
	void* sum_cols_pipeline;
	void* sum_cols_layout;
	void* sum_cols_desc_layout;
} vulkan_context_t;

// Initialize Vulkan compute backend
// pool_size_hint: expected number of descriptor sets needed (0 = default 500)
vulkan_context_t* vulkan_init(int pool_size_hint);

// Extended init:
// - device_index: -1 = auto select (prefer discrete), otherwise index from vkEnumeratePhysicalDevices
vulkan_context_t* vulkan_init_ex(int pool_size_hint, int device_index);

// Cleanup Vulkan resources
void vulkan_free(vulkan_context_t* ctx);

// Batch multiple compute dispatches into a single submit.
// When recording is active, kernel calls record into one command buffer and do NOT submit/wait.
// Call vulkan_cmd_end_submit() to submit, wait, and free deferred descriptor sets.
int vulkan_cmd_begin(vulkan_context_t* ctx);
int vulkan_cmd_end_submit(vulkan_context_t* ctx);

// ═══════════════════════════════════════════════════════════════════════════
// GPU BUFFER MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    void* buffer;              // VkBuffer
    void* memory;              // VkDeviceMemory
    size_t size;
    void* mapped;              // Host-visible pointer (if mapped)
} vulkan_buffer_t;

// Allocate GPU buffer
vulkan_buffer_t* vulkan_buffer_create(vulkan_context_t* ctx, size_t size);

// Upload data to GPU
void vulkan_buffer_upload(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                          const float* data, size_t count);

// Upload raw bytes to GPU (for uint buffers, etc.)
void vulkan_buffer_upload_bytes(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                                const void* data, size_t bytes);

// Upload uint32 elements to GPU
void vulkan_buffer_upload_u32(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                              const uint32_t* data, size_t count);

// Download data from GPU
void vulkan_buffer_download(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                            float* data, size_t count);

// Download raw bytes from GPU
void vulkan_buffer_download_bytes(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                                  void* data, size_t bytes);

// Free GPU buffer
void vulkan_buffer_free(vulkan_context_t* ctx, vulkan_buffer_t* buf);

// Fill a subrange of a buffer with a 32-bit pattern (like vkCmdFillBuffer).
// offset_bytes and bytes must be multiples of 4; this helper will clamp/align down.
void vulkan_fill_buffer_range(vulkan_context_t* ctx,
                              vulkan_buffer_t* buf,
                              size_t offset_bytes,
                              uint32_t value,
                              size_t bytes);

// Debug: scan a float buffer for NaN/Inf and set flags[flag_index] |= 1 if any found.
// flags must be a uint32 buffer with enough elements.
void vulkan_nan_check(vulkan_context_t* ctx,
                      vulkan_buffer_t* input,
                      size_t count_floats,
                      vulkan_buffer_t* flags_u32,
                      uint32_t flag_index);

// ═══════════════════════════════════════════════════════════════════════════
// COMPUTE KERNELS
// ═══════════════════════════════════════════════════════════════════════════

// Matrix multiply: C = A @ B
// A: [M x K], B: [K x N], C: [M x N]
void vulkan_matmul(vulkan_context_t* ctx,
                   vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                   int M, int N, int K);

// Matrix multiply with transpose: C = op(A) @ op(B)
// transpose_A: 0=A, 1=A^T
// transpose_B: 0=B, 1=B^T
// accumulate: 0=overwrite C, 1=C += result
void vulkan_matmul_transpose(vulkan_context_t* ctx,
                             vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                             int M, int N, int K,
                             int transpose_A, int transpose_B, int accumulate);

// AdamW optimizer update for single tensor
void vulkan_adamw_update(vulkan_context_t* ctx,
                         vulkan_buffer_t* weight,
                         vulkan_buffer_t* grad,
                         vulkan_buffer_t* m,
                         vulkan_buffer_t* v,
                         float lr, float beta1, float beta2,
                         float weight_decay, float eps,
                         int step, size_t size);

// Stable softmax: output[i] = exp(input[i]) / sum(exp(input))
void vulkan_softmax(vulkan_context_t* ctx,
                    vulkan_buffer_t* input,
                    vulkan_buffer_t* output,
                    int rows, int cols);

// Element-wise add: C = A + B
void vulkan_add(vulkan_context_t* ctx,
                vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                size_t count);

// Element-wise multiply: C = A * B
void vulkan_mul(vulkan_context_t* ctx,
                vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                size_t count);

// In-place scale: A = A * alpha
void vulkan_scale(vulkan_context_t* ctx,
                  vulkan_buffer_t* A,
                  size_t count, float alpha);

// SiLU activation: output = x * sigmoid(x)
void vulkan_silu(vulkan_context_t* ctx,
                 vulkan_buffer_t* input, vulkan_buffer_t* output,
                 size_t count);

// Fused multi-head attention with causal mask
// Q: [seq_len, num_heads * head_dim]
// K: [seq_len, kv_heads * head_dim]
// V: [seq_len, kv_heads * head_dim]
// out: [seq_len, num_heads * head_dim]
void vulkan_batch_attention(vulkan_context_t* ctx,
                            vulkan_buffer_t* Q, vulkan_buffer_t* K,
                            vulkan_buffer_t* V, vulkan_buffer_t* out,
                            int seq_len, int num_heads, int kv_heads, int head_dim);

// Rotary Position Embedding (RoPE)
// Applies in-place rotation to Q or K based on position
// x: [seq_len, num_heads * head_dim]
void vulkan_rope(vulkan_context_t* ctx,
                 vulkan_buffer_t* x_in, vulkan_buffer_t* x_out,
                 int seq_len, int num_heads, int head_dim, float rope_theta);

// RMSNorm forward
// input: [rows, cols], scale: [cols], output: [rows, cols]
void vulkan_rmsnorm(vulkan_context_t* ctx,
                    vulkan_buffer_t* input,
                    vulkan_buffer_t* scale,
                    vulkan_buffer_t* output,
                    int rows, int cols, float eps);

// Buffer copy (GPU-side)
// Copies `bytes` from src to dst (dst must be >= bytes).
void vulkan_copy_buffer(vulkan_context_t* ctx,
                        vulkan_buffer_t* src,
                        vulkan_buffer_t* dst,
                        size_t bytes);

// Cache copy: compute→transfer barrier + copy, but NO transfer→compute post-barrier.
// Use for forward-pass activation caching where dst is only read in backward pass.
// MUST call vulkan_barrier_transfer_to_compute() before backward pass begins!
void vulkan_copy_buffer_cache(vulkan_context_t* ctx,
                              vulkan_buffer_t* src,
                              vulkan_buffer_t* dst,
                              size_t bytes);

// Standalone transfer→compute barrier. Call once before backward pass to ensure
// all cache copies from forward pass are visible to compute shaders.
void vulkan_barrier_transfer_to_compute(vulkan_context_t* ctx);

// Fill buffer with a 32-bit pattern (commonly 0 for memset)
void vulkan_fill_buffer(vulkan_context_t* ctx,
                        vulkan_buffer_t* buf,
                        uint32_t value,
                        size_t bytes);

// Embedding lookup: out_hidden = embed[tokens]
void vulkan_embed_lookup(vulkan_context_t* ctx,
                         vulkan_buffer_t* tokens_u32,
                         vulkan_buffer_t* embed_weight,
                         vulkan_buffer_t* out_hidden,
                         int seq_len, int hidden_size, int vocab_size);

// Fused cross-entropy + gradient
void vulkan_cross_entropy_grad(vulkan_context_t* ctx,
                               vulkan_buffer_t* logits,
                               vulkan_buffer_t* targets_u32,
                               vulkan_buffer_t* grad_logits,
                               vulkan_buffer_t* loss_rows,
                               int rows, int cols, int valid_rows);

// Reduce sum: out has ceil(count/512) elements
void vulkan_reduce_sum(vulkan_context_t* ctx,
                       vulkan_buffer_t* in_buf,
                       vulkan_buffer_t* out_buf,
                       size_t count);

// RMSNorm backward (batched): computes grad_input and accumulates grad_scale
void vulkan_rmsnorm_backward_batch(vulkan_context_t* ctx,
                                   vulkan_buffer_t* input,
                                   vulkan_buffer_t* scale,
                                   vulkan_buffer_t* grad_output,
                                   vulkan_buffer_t* grad_input,
                                   vulkan_buffer_t* grad_scale,
                                   int rows, int cols, float eps);

// RoPE backward (inverse rotation)
void vulkan_rope_backward(vulkan_context_t* ctx,
                          vulkan_buffer_t* x_in, vulkan_buffer_t* x_out,
                          int seq_len, int num_heads, int head_dim, float rope_theta);

// Attention backward
void vulkan_batch_attention_backward(vulkan_context_t* ctx,
                                     vulkan_buffer_t* Q, vulkan_buffer_t* K, vulkan_buffer_t* V,
                                     vulkan_buffer_t* grad_out,
                                     vulkan_buffer_t* grad_Q,
                                     vulkan_buffer_t* grad_K,
                                     vulkan_buffer_t* grad_V,
                                     int seq_len, int num_heads, int kv_heads, int head_dim);

// Embedding backward (scatter-add): grad_embed += grad_hidden scattered by token ids
void vulkan_embedding_backward(vulkan_context_t* ctx,
                               vulkan_buffer_t* tokens_u32,
                               vulkan_buffer_t* grad_hidden,
                               vulkan_buffer_t* grad_embed,
                               int seq_len, int hidden_size, int vocab_size);

// SiLU backward: grad_input = grad_output * silu'(input)
void vulkan_silu_backward(vulkan_context_t* ctx,
                          vulkan_buffer_t* input,
                          vulkan_buffer_t* grad_output,
                          vulkan_buffer_t* grad_input,
                          size_t count);

// Elementwise multiply backward:
// grad_A = grad_C * B
// grad_B = grad_C * A
void vulkan_elementwise_mul_backward(vulkan_context_t* ctx,
                                     vulkan_buffer_t* A,
                                     vulkan_buffer_t* B,
                                     vulkan_buffer_t* grad_C,
                                     vulkan_buffer_t* grad_A,
                                     vulkan_buffer_t* grad_B,
                                     size_t count);

// QK RMSNorm (video/ViT mode)
void vulkan_qk_norm(vulkan_context_t* ctx,
                    vulkan_buffer_t* qk,
                    int seq_len, int num_heads, int head_dim,
                    int use_qk_norm, float eps);

// Patch extraction (video mode)
// frame: [temporal * channels * H * W] input frame(s)
// patches: [num_patches, patch_dim] output patches
void vulkan_patch_extract(vulkan_context_t* ctx,
                          vulkan_buffer_t* frame,
                          vulkan_buffer_t* patches,
                          int frame_h, int frame_w, int channels,
                          int patch_t, int patch_h, int patch_w);

// Add bias with broadcast: data[i,j] += bias[j]
void vulkan_add_bias(vulkan_context_t* ctx,
                     vulkan_buffer_t* data,
                     vulkan_buffer_t* bias,
                     int rows, int cols);

// BCE loss + gradient for mask segmentation
// loss[i] = BCE(logits[i], targets[i])
// grad[i] = sigmoid(logits[i]) - targets[i]
void vulkan_bce_loss_grad(vulkan_context_t* ctx,
                          vulkan_buffer_t* logits,
                          vulkan_buffer_t* targets,
                          vulkan_buffer_t* grad_logits,
                          vulkan_buffer_t* loss_out,
                          size_t count);

// Sum columns: out[col] = sum_row(data[row, col])
// For bias gradient: grad_b = sum over queries of grad_logits
void vulkan_sum_cols(vulkan_context_t* ctx,
                     vulkan_buffer_t* data,
                     vulkan_buffer_t* out,
                     int rows, int cols);

// Vulkan profiler
void vulkan_profiler_enable(vulkan_context_t* ctx);
void vulkan_profiler_disable(vulkan_context_t* ctx);
void vulkan_profiler_set_log(vulkan_context_t* ctx, void* file_ptr);
void vulkan_profiler_begin_step(vulkan_context_t* ctx);
void vulkan_profiler_end_step(vulkan_context_t* ctx, int step, int seq_len, float loss);

#endif // VULKAN_BACKEND_H
