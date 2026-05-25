// train.h - Training pipeline and gradient computation
// Full-weight and LoRA fine-tuning with Vulkan GPU acceleration
// Part of TETYAH-PRIME's native training and inference engine

#ifndef TETYAH_TRAIN_H
#define TETYAH_TRAIN_H

#include "transformer.h"
#include "model_config.h"
#include "tokenizer.h"
#include "lora.h"
#include <stdint.h>
#include <stdio.h>

// ═══════════════════════════════════════════════════════════════════════════
// ACTIVATION CACHE - stores intermediate values for backprop
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    int num_layers;
    int hidden_size;
    int intermediate_size;
    int q_dim;
    int kv_dim;
    int max_seq_len;
    int cur_seq_len;

    // Per-layer, per-position activations
    // Shape: [num_layers][seq_len][dim]
    float** layer_inputs;           // Input to each layer (after prev residual)
    float** x_norm_attn;            // After input layernorm (input to Q/K/V)
    float** attn_output;            // Combined attention output (input to O proj)
    float** x_norm_ffn;             // After post-attn layernorm (input to FFN)
    float** ffn_gate_out;           // Gate projection output (before SiLU)
    float** ffn_up_out;             // Up projection output
    float** ffn_hidden;             // After SiLU * up (input to down proj)

    // Final hidden state
    float* final_hidden;            // [hidden_size] - input to LM head
} activation_cache_t;

activation_cache_t* activation_cache_alloc(int num_layers, int hidden_size,
                                            int intermediate_size, int q_dim,
                                            int kv_dim, int max_seq_len);
void activation_cache_free(activation_cache_t* cache);
void activation_cache_reset(activation_cache_t* cache);

// ═══════════════════════════════════════════════════════════════════════════
// TRAINABLE WEIGHTS - for full weight training (not LoRA)
// ═══════════════════════════════════════════════════════════════════════════

// Single weight tensor with gradient and optimizer state
typedef struct {
    float* weight;      // FP32 weights
    float* grad;        // FP32 gradients
    float* m;           // Adam first moment (momentum)
    float* v;           // Adam second moment (variance)
    size_t size;        // Number of elements
} weight_tensor_t;

// Per-layer trainable weights
typedef struct {
    weight_tensor_t q_proj;
    weight_tensor_t k_proj;
    weight_tensor_t v_proj;
    weight_tensor_t o_proj;
    weight_tensor_t gate_proj;
    weight_tensor_t up_proj;
    weight_tensor_t down_proj;
    weight_tensor_t input_norm;
    weight_tensor_t post_norm;
} layer_weights_t;

// Full model trainable weights (FP32 for all parameters)
typedef struct {
    int num_layers;
    int hidden_size;
    int intermediate_size;
    int vocab_size;

    // Per-layer weights
    layer_weights_t* layers;  // [num_layers]

    // Global weights
    weight_tensor_t embed_tokens;
    weight_tensor_t final_norm;

    // Total parameter count
    size_t total_params;
} trainable_weights_t;

trainable_weights_t* trainable_weights_alloc(const model_config_t* config);
void trainable_weights_free(trainable_weights_t* weights);
void trainable_weights_load_from_model(trainable_weights_t* weights, seraph_model_t* model);
void trainable_weights_save_to_safetensors(trainable_weights_t* weights, const char* path);
int trainable_weights_load_from_safetensors(trainable_weights_t* weights, const char* path);

// ═══════════════════════════════════════════════════════════════════════════
// GRADIENT SNAPSHOT SYSTEM - Electromagnetic Field Visualization
// ═══════════════════════════════════════════════════════════════════════════
// Samples gradient vectors at solfeggio-harmonic intervals for field monitoring
// Based on Jefimenko's retarded time - field observed between discrete snapshots

#define GRADIENT_SNAPSHOT_MAGIC 0x47524144  // "GRAD"
#define GRADIENT_SAMPLES_PER_TENSOR 528     // Solfeggio frequency (cellular-communication harmonic)

typedef struct {
    FILE* fp;
    uint32_t num_snapshots;
    int enabled;
} gradient_snapshot_writer_t;

gradient_snapshot_writer_t* gradient_snapshot_writer_init(const char* path, int num_layers);
void gradient_snapshot_write(gradient_snapshot_writer_t* writer, trainable_weights_t* weights,
                             uint32_t step, float loss, float lr);
void gradient_snapshot_writer_close(gradient_snapshot_writer_t* writer);

// Training configuration
typedef struct {
    // Data
    char* train_data_path;      // Path to .ttok file (LLM) or .vdat file (video)
    char* val_data_path;        // Optional validation .ttok (LLM) or .vdat (video)

    // Training params
    int epochs;
    int batch_size;             // Usually 1 for CPU
    int max_seq_len;            // Maximum sequence length
    int fixed_seq_windows;      // 1 = split samples into fixed max_seq_len windows, drop remainders
    float learning_rate;        // Base learning rate
    float lr_warmup_ratio;      // Warmup steps as ratio of total
    float weight_decay;         // AdamW weight decay

    // Training mode
    int use_full_training;  // 0 = LoRA mode, 1 = Full weight training

    // LoRA params (if use_full_training == 0)
    int lora_rank;
    float lora_alpha;
    float lora_dropout;

    // Checkpointing
    int save_every_n_steps;
    char* checkpoint_dir;

    // Logging
    int log_every_n_steps;
    int eval_every_n_steps;

    // Gradient accumulation
    int gradient_accumulation_steps;  // Effective batch = batch_size * accumulation_steps

    // Gradient snapshots for field visualization
    int snapshot_every_n_steps;  // 0 = disabled

    // Debugging / diagnostics
    int debug;                   // 1 = enable extra debug output
    int debug_weight_delta;      // 1 = print w_embed[0] delta after optimizer steps (GPU full training)
    int debug_nan_check;         // 1 = run GPU NaN/Inf checks and stop before poisoning weights

    // Precision
    int use_bf16;                // 1 = BF16 mixed precision (tensor cores on Ampere+)

    // GPU selection
    int cpu_only;                // 1 = force CPU paths even if GPU is available
    int cuda_backend;            // 1 = using CUDA backend (skip Vulkan init)
    int gpu_device_index;        // -1 = auto (prefer discrete), otherwise device index

    // Video/ViT mode
    int use_qk_norm;             // 1 = QK normalization across heads before attention
    int video_shuffle_frames;    // 0 = sequential (default, required for query propagation)
                                 // 1 = shuffle (breaks Q_t = f(Q_{t-1}), use for independent frames)
    int video_batch_size;        // Frames per batch (default 1)
    int max_video_frames;        // 0 = load all, >0 = limit frames loaded (memory saver)

    // CUDA init function pointer (set by CUDA frontend, called at step 7)
    void* (*cuda_init_fn)(int device_id);  // Returns cuda_train_context_t*

    // Logging to file
    char* log_path;              // NULL = no file logging, otherwise path to append training log
} train_config_t;

// Training state
typedef struct {
    // Model
    seraph_model_t* model;
    model_config_t* config;
    tokenizer_t* tokenizer;
    train_config_t* train_cfg;  // Training config (for max_video_frames, etc.)

    // Training mode (mutually exclusive)
    int use_full_training;              // 0 = LoRA, 1 = Full weights
    lora_model_t* lora;                 // LoRA adapters (if use_full_training == 0)
    trainable_weights_t* full_weights;  // Full FP32 weights (if use_full_training == 1)

    // GPU acceleration (optional - NULL for CPU-only)
    void* vulkan_ctx;                   // vulkan_context_t* (NULL = CPU mode)
    void* cuda_ctx;                     // cuda_train_context_t* (NULL = no CUDA)
    size_t max_tensor_size;             // Largest tensor size in bytes (for GPU buffer sizing)

    // Activation cache for backprop
    activation_cache_t* act_cache;

    // Video model state (query propagation)
    void* prev_query_out;           // vulkan_buffer_t* or float* - Q_out[t-1]
    void* query_learnable;          // vulkan_buffer_t* or float* - learned queries
    void* query_fusion_weight;      // vulkan_buffer_t* or float* - fusion linear weight
    void* video_patch_embed;        // vulkan_buffer_t* - embedded patches (K/V source, persists through layers)
    void* video_pos_embed;          // vulkan_buffer_t* - sinusoidal 2D position embeddings [num_patches, hidden]
    void* video_grad_patches;       // vulkan_buffer_t* - gradients for patch embeddings
    void* video_grad_queries;       // vulkan_buffer_t* - gradients for learnable queries
    void* video_grad_fusion;        // vulkan_buffer_t* - gradients for fusion weight
    void* video_m_queries;          // vulkan_buffer_t* - Adam m state for queries
    void* video_v_queries;          // vulkan_buffer_t* - Adam v state for queries
    void* video_m_fusion;           // vulkan_buffer_t* - Adam m state for fusion
    void* video_v_fusion;           // vulkan_buffer_t* - Adam v state for fusion
    int current_frame_idx;          // 0 = first frame (no propagation)
    int valid_query_count;          // number of queries with valid class targets this frame

    // Video output heads
    void* class_head_weight;        // vulkan_buffer_t* - [query_dim, num_classes]
    void* class_head_bias;          // vulkan_buffer_t* - [num_classes]
    void* class_head_grad_w;        // vulkan_buffer_t* - gradient for weight
    void* class_head_grad_b;        // vulkan_buffer_t* - gradient for bias
    void* class_head_m_w;           // vulkan_buffer_t* - Adam m for weight
    void* class_head_v_w;           // vulkan_buffer_t* - Adam v for weight
    void* class_head_m_b;           // vulkan_buffer_t* - Adam m for bias
    void* class_head_v_b;           // vulkan_buffer_t* - Adam v for bias
    void* class_logits_buf;         // vulkan_buffer_t* - [num_queries, num_classes] for metrics

    // Mask head: query_embed -> mask_mlp -> dot with patches -> masks
    void* mask_mlp_w1;              // vulkan_buffer_t* - [query_dim, mask_dim]
    void* mask_mlp_w2;              // vulkan_buffer_t* - [mask_dim, query_dim]
    void* mask_logits;              // vulkan_buffer_t* - [num_queries, num_patches] output
    void* mask_targets;             // vulkan_buffer_t* - [num_queries, num_patches] ground truth
    void* mask_grad_logits;         // vulkan_buffer_t* - [num_queries, num_patches] BCE grad
    void* mask_loss_per_elem;       // vulkan_buffer_t* - [num_queries * num_patches] per-element loss
    void* mask_grad_w1;             // vulkan_buffer_t* - gradient
    void* mask_grad_w2;             // vulkan_buffer_t* - gradient
    void* mask_m_w1;                // vulkan_buffer_t* - Adam m
    void* mask_v_w1;                // vulkan_buffer_t* - Adam v
    void* mask_m_w2;                // vulkan_buffer_t* - Adam m
    void* mask_v_w2;                // vulkan_buffer_t* - Adam v

    // CLIP semantic alignment
    void* clip_embeddings;          // vulkan_buffer_t* - [num_classes, clip_embed_dim] pre-computed
    void* clip_proj_weight;         // vulkan_buffer_t* - [query_dim, clip_embed_dim] projection (if dims differ)
    void* clip_proj_grad;           // vulkan_buffer_t* - gradient for projection
    void* clip_proj_m;              // vulkan_buffer_t* - Adam m for projection
    void* clip_proj_v;              // vulkan_buffer_t* - Adam v for projection
    void* clip_query_proj;          // vulkan_buffer_t* - [num_queries, clip_embed_dim] projected queries
    void* clip_similarity;          // vulkan_buffer_t* - [num_queries, num_classes] similarity scores
    void* clip_grad_similarity;     // vulkan_buffer_t* - [num_queries, num_classes] grad for similarity
    void* clip_loss_per_query;      // vulkan_buffer_t* - [num_queries] per-query contrastive loss

    // Gradient snapshot writer for field visualization
    gradient_snapshot_writer_t* snapshot_writer;

    // Data (LLM mode)
    int* train_tokens;          // All training tokens
    int num_train_tokens;
    int* sample_offsets;        // Start of each sample
    int num_samples;

    // Data (Video mode - training)
    float* video_frames;        // All frames [num_frames, C, H, W]
    int* video_labels;          // Labels per frame [num_frames, num_labels]
    float* video_masks;         // Mask targets [num_frames, num_queries, num_patches] - binary
    int num_video_frames;
    int num_labels_per_frame;   // e.g., 1 for classification, N for detection
    void* vdat_mmap;            // mmap'd VDAT file (NULL if not using mmap)
    size_t vdat_mmap_size;      // Size of mmap'd region
    void* gpu_mask_targets;     // vulkan_buffer_t* - uploaded mask targets for current frame
    void* gpu_mask_loss;        // vulkan_buffer_t* - per-element BCE loss
    void* gpu_grad_mask;        // vulkan_buffer_t* - gradient of mask logits

    // Data (Video mode - validation)
    float* val_video_frames;    // Validation frames [num_val_frames, C, H, W]
    int* val_video_labels;      // Validation labels
    float* val_video_masks;     // Validation mask targets
    int num_val_video_frames;

    // Training progress
    int current_epoch;
    int current_step;
    int total_steps;
    float current_loss;
    float best_val_loss;
    int nan_detected;            // set when debug nan check trips

    // Optimizer state
    int optimizer_step;
    int accumulation_counter;   // Tracks steps since last optimizer update
    int gradient_accumulation_steps;  // Copy from config for easy access
    float learning_rate;        // Copy from config for optimizer

    // Debugging / diagnostics
    int debug;
    int debug_weight_delta;
    int debug_nan_check;

    // Vision metrics (IoU, mAP, vAP, Accuracy)
    float current_iou;           // Mean IoU for current frame
    float current_map;           // Mean AP for current frame
    float current_vap;           // Video AP (tracking consistency across frames)
    float current_acc;           // Top-1 accuracy for current frame
    float current_acc5;          // Top-5 accuracy for current frame
    float running_iou_sum;       // Sum for epoch average
    float running_map_sum;       // Sum for epoch average
    float running_vap_sum;       // Sum for epoch average
    float running_acc_sum;       // Sum for epoch average
    float running_acc5_sum;      // Sum for epoch average
    int metrics_count;           // Number of samples in running sum

    // vAP tracking state (query ID consistency across frames)
    int* prev_query_classes;     // [num_queries] - predicted class per query last frame
    float* prev_query_conf;      // [num_queries] - confidence per query last frame
    int vap_id_switches;         // Count of ID switches (query changed class unexpectedly)
    int vap_total_tracks;        // Total tracked objects across frames

} train_state_t;

// Load training data from .ttok file (LLM mode)
int train_load_data(train_state_t* state, const char* ttok_path);

// Load training data from .vdat file (Video mode)
// Format: [magic:4][version:4][num_frames:4][height:4][width:4][channels:4][labels_per_frame:4]
//         [frame_data: num_frames * C * H * W floats]
//         [labels: num_frames * labels_per_frame int32s]
int train_load_video_data(train_state_t* state, const char* vdat_path);

// Initialize training state
train_state_t* train_init(const char* model_dir, train_config_t* config);

// Free training state
void train_free(train_state_t* state);

// Single training step: forward, loss, backward, optimize
float train_step(train_state_t* state, int* tokens, int num_tokens);

// Video training step: uploads frame, runs forward/backward
float train_step_video(train_state_t* state, int frame_idx);

// Compute cross-entropy loss
float compute_loss(float* logits, int* targets, int seq_len, int vocab_size);

// Full training loop
void train_loop(train_state_t* state, train_config_t* config);

// Evaluation on validation set (LLM)
float evaluate(train_state_t* state, int* val_tokens, int num_val_tokens);

// Evaluation on validation set (Video) - returns avg loss over all val frames
float evaluate_video(train_state_t* state);

// ═══════════════════════════════════════════════════════════════════════════
// VISION METRICS - IoU, mAP, vAP for video segmentation
// ═══════════════════════════════════════════════════════════════════════════

// Compute IoU (Intersection over Union) for mask predictions
float compute_iou(float* mask_pred, float* mask_gt, int num_queries, int num_patches,
                  float threshold, float* iou_out);

// Compute mAP@[0.5:0.95] (COCO-style mean Average Precision)
float compute_map(float* class_logits, int* class_gt, float* iou_scores,
                  int num_queries, int num_classes);

// Compute vAP (video AP) - tracking consistency across frames
float compute_vap(train_state_t* state, float* class_logits, int* class_gt,
                  int num_queries, int num_classes);

// Compute all vision metrics for current frame
void compute_vision_metrics(train_state_t* state);

// Load validation video data
int train_load_val_video_data(train_state_t* state, const char* vdat_path);

// Save checkpoint
int save_checkpoint(train_state_t* state, const char* path);

// Load checkpoint and resume
int load_checkpoint(train_state_t* state, const char* path);

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING FORWARD/BACKWARD (saves activations for gradient computation)
// ═══════════════════════════════════════════════════════════════════════════

// Training forward pass - saves activations for backward pass
// Returns logits [vocab_size] for the last position
float* train_forward(train_state_t* state, int* tokens, int num_tokens);

// Compute loss gradient w.r.t. logits
// grad_logits[i] = softmax(logits)[i] - (1 if i == target else 0)
void compute_loss_gradient(float* grad_logits, float* logits, int target, int vocab_size);

// Backward pass - computes gradients for LoRA adapters
// Uses saved activations from train_forward
void train_backward(train_state_t* state, float* grad_logits, int seq_len);

#endif // TETYAH_TRAIN_H
