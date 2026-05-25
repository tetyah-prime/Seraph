// train_cuda.c - CUDA Training Pipeline
// Part of TETYAH-PRIME's native training and inference engine
// Direct port of train.c's GPU training path using CUDA instead of Vulkan

#include "../include/train.h"
#include "../include/cuda_train_context.h"
#include "../include/lora.h"
#include "../include/transformer.h"
#include "../include/safetensors.h"
#include "../include/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// BF16 conversion for base weight upload
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Get tensor from model by name (for LoRA base weight upload)
static uint16_t* get_model_tensor(seraph_model_t* model, const char* name) {
    for (int i = 0; i < model->num_tensors; i++) {
        if (strcmp(model->tensor_map[i].name, name) == 0) {
            int file_idx = model->tensor_map[i].file_idx;
            return (uint16_t*)safetensors_get_tensor_raw(model->st[file_idx], model->tensor_map[i].orig_name);
        }
    }
    return NULL;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIMING HELPER
// ═══════════════════════════════════════════════════════════════════════════

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

// ═══════════════════════════════════════════════════════════════════════════
// BUFFER ALLOCATION - Forward Pass
// ═══════════════════════════════════════════════════════════════════════════

static void ensure_cuda_fwd_buffers(cuda_train_context_t* ctx,
                                    const train_state_t* state,
                                    int max_seq,
                                    int hidden_size,
                                    int intermediate_size,
                                    int q_dim,
                                    int kv_dim,
                                    int vocab_size) {
    if (!ctx) return;

    int num_layers = state && state->config ? state->config->num_hidden_layers : 0;

    // Ensure shared weight buffer exists
    if (!ctx->fwd_weight_buffer) {
        if (ctx->max_fwd_buffer_size == 0) {
            fprintf(stderr, "ERROR: max_fwd_buffer_size not set!\n");
            return;
        }
        ctx->fwd_input_buffer = cuda_train_buffer_create(ctx, ctx->max_fwd_buffer_size);
        ctx->fwd_weight_buffer = cuda_train_buffer_create(ctx, ctx->max_fwd_buffer_size);
        ctx->fwd_output_buffer = cuda_train_buffer_create(ctx, ctx->max_fwd_buffer_size);
    }

    // Attention buffers
    {
        size_t q_buf_size = (size_t)max_seq * (size_t)q_dim * sizeof(float);
        size_t kv_buf_size = (size_t)max_seq * (size_t)kv_dim * sizeof(float);
        if (!ctx->attn_q_buffer || ctx->attn_q_buffer->size < q_buf_size) {
            if (ctx->attn_q_buffer) {
                cuda_train_buffer_free(ctx, ctx->attn_q_buffer);
                cuda_train_buffer_free(ctx, ctx->attn_k_buffer);
                cuda_train_buffer_free(ctx, ctx->attn_v_buffer);
                cuda_train_buffer_free(ctx, ctx->attn_out_buffer);
            }
            ctx->max_attn_buffer_size = q_buf_size;
            ctx->attn_q_buffer = cuda_train_buffer_create(ctx, q_buf_size);
            ctx->attn_k_buffer = cuda_train_buffer_create(ctx, kv_buf_size);
            ctx->attn_v_buffer = cuda_train_buffer_create(ctx, kv_buf_size);
            ctx->attn_out_buffer = cuda_train_buffer_create(ctx, q_buf_size);
        }
    }

    // Forward activation buffers
    size_t hidden_bytes = (size_t)max_seq * (size_t)hidden_size * sizeof(float);
    size_t inter_bytes = (size_t)max_seq * (size_t)intermediate_size * sizeof(float);
    size_t logits_bytes = (size_t)max_seq * (size_t)vocab_size * sizeof(float);
    size_t norm_w_bytes = (size_t)hidden_size * sizeof(float);

    if (!ctx->fwd_hidden_buffer || ctx->fwd_hidden_buffer->size < hidden_bytes) {
        if (ctx->fwd_hidden_buffer) cuda_train_buffer_free(ctx, ctx->fwd_hidden_buffer);
        ctx->fwd_hidden_buffer = cuda_train_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->fwd_hidden_norm_buffer || ctx->fwd_hidden_norm_buffer->size < hidden_bytes) {
        if (ctx->fwd_hidden_norm_buffer) cuda_train_buffer_free(ctx, ctx->fwd_hidden_norm_buffer);
        ctx->fwd_hidden_norm_buffer = cuda_train_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->fwd_tmp_hidden_buffer || ctx->fwd_tmp_hidden_buffer->size < hidden_bytes) {
        if (ctx->fwd_tmp_hidden_buffer) cuda_train_buffer_free(ctx, ctx->fwd_tmp_hidden_buffer);
        ctx->fwd_tmp_hidden_buffer = cuda_train_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->fwd_ffn_gate_buffer || ctx->fwd_ffn_gate_buffer->size < inter_bytes) {
        if (ctx->fwd_ffn_gate_buffer) cuda_train_buffer_free(ctx, ctx->fwd_ffn_gate_buffer);
        ctx->fwd_ffn_gate_buffer = cuda_train_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->fwd_ffn_up_buffer || ctx->fwd_ffn_up_buffer->size < inter_bytes) {
        if (ctx->fwd_ffn_up_buffer) cuda_train_buffer_free(ctx, ctx->fwd_ffn_up_buffer);
        ctx->fwd_ffn_up_buffer = cuda_train_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->fwd_ffn_hidden_buffer || ctx->fwd_ffn_hidden_buffer->size < inter_bytes) {
        if (ctx->fwd_ffn_hidden_buffer) cuda_train_buffer_free(ctx, ctx->fwd_ffn_hidden_buffer);
        ctx->fwd_ffn_hidden_buffer = cuda_train_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->fwd_logits_buffer || ctx->fwd_logits_buffer->size < logits_bytes) {
        if (ctx->fwd_logits_buffer) cuda_train_buffer_free(ctx, ctx->fwd_logits_buffer);
        ctx->fwd_logits_buffer = cuda_train_buffer_create(ctx, logits_bytes);
    }
    if (!ctx->fwd_norm_weight_buffer || ctx->fwd_norm_weight_buffer->size < norm_w_bytes) {
        if (ctx->fwd_norm_weight_buffer) cuda_train_buffer_free(ctx, ctx->fwd_norm_weight_buffer);
        ctx->fwd_norm_weight_buffer = cuda_train_buffer_create(ctx, norm_w_bytes);
    }

    // Per-layer activation caches
    if (num_layers > 0) {
        int needs_realloc = 0;
        if (!ctx->fwd_cache_x_norm_attn) needs_realloc = 1;
        if (ctx->fwd_cache_num_layers != num_layers) needs_realloc = 1;
        if (ctx->fwd_cache_max_seq < max_seq) needs_realloc = 1;
        if (ctx->fwd_cache_hidden_size != hidden_size) needs_realloc = 1;
        if (ctx->fwd_cache_intermediate_size != intermediate_size) needs_realloc = 1;
        if (ctx->fwd_cache_q_dim != q_dim) needs_realloc = 1;

        if (needs_realloc) {
            // Free existing
            if (ctx->fwd_cache_x_norm_attn) {
                for (int l = 0; l < ctx->fwd_cache_num_layers; l++) {
                    cuda_train_buffer_free(ctx, ctx->fwd_cache_x_norm_attn[l]);
                    cuda_train_buffer_free(ctx, ctx->fwd_cache_x_norm_ffn[l]);
                    cuda_train_buffer_free(ctx, ctx->fwd_cache_attn_out[l]);
                    cuda_train_buffer_free(ctx, ctx->fwd_cache_ffn_gate_out[l]);
                    cuda_train_buffer_free(ctx, ctx->fwd_cache_ffn_up_out[l]);
                    cuda_train_buffer_free(ctx, ctx->fwd_cache_ffn_hidden[l]);
                    if (ctx->fwd_cache_layer_input) cuda_train_buffer_free(ctx, ctx->fwd_cache_layer_input[l]);
                    if (ctx->fwd_cache_post_attn_in) cuda_train_buffer_free(ctx, ctx->fwd_cache_post_attn_in[l]);
                    if (ctx->fwd_cache_silu_out) cuda_train_buffer_free(ctx, ctx->fwd_cache_silu_out[l]);
                    if (ctx->fwd_cache_q) cuda_train_buffer_free(ctx, ctx->fwd_cache_q[l]);
                    if (ctx->fwd_cache_k) cuda_train_buffer_free(ctx, ctx->fwd_cache_k[l]);
                    if (ctx->fwd_cache_v) cuda_train_buffer_free(ctx, ctx->fwd_cache_v[l]);
                    if (ctx->fwd_cache_attn_lse) cuda_train_buffer_free(ctx, ctx->fwd_cache_attn_lse[l]);
                }
                free(ctx->fwd_cache_x_norm_attn);
                free(ctx->fwd_cache_x_norm_ffn);
                free(ctx->fwd_cache_attn_out);
                free(ctx->fwd_cache_ffn_gate_out);
                free(ctx->fwd_cache_ffn_up_out);
                free(ctx->fwd_cache_ffn_hidden);
                if (ctx->fwd_cache_layer_input) free(ctx->fwd_cache_layer_input);
                if (ctx->fwd_cache_post_attn_in) free(ctx->fwd_cache_post_attn_in);
                if (ctx->fwd_cache_silu_out) free(ctx->fwd_cache_silu_out);
                if (ctx->fwd_cache_q) free(ctx->fwd_cache_q);
                if (ctx->fwd_cache_k) free(ctx->fwd_cache_k);
                if (ctx->fwd_cache_v) free(ctx->fwd_cache_v);
                if (ctx->fwd_cache_attn_lse) free(ctx->fwd_cache_attn_lse);
            }

            ctx->fwd_cache_num_layers = num_layers;
            ctx->fwd_cache_max_seq = max_seq;
            ctx->fwd_cache_hidden_size = hidden_size;
            ctx->fwd_cache_intermediate_size = intermediate_size;
            ctx->fwd_cache_q_dim = q_dim;

            ctx->fwd_cache_x_norm_attn = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_x_norm_ffn = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_attn_out = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_ffn_gate_out = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_ffn_up_out = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_ffn_hidden = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_layer_input = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_post_attn_in = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_silu_out = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_q = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_k = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_v = calloc(num_layers, sizeof(cuda_buffer_t*));
            ctx->fwd_cache_attn_lse = calloc(num_layers, sizeof(cuda_buffer_t*));

            size_t xnorm_bytes = (size_t)max_seq * (size_t)hidden_size * sizeof(float);
            size_t attn_bytes = (size_t)max_seq * (size_t)q_dim * sizeof(float);
            size_t lse_bytes = (size_t)max_seq * (size_t)(state->config->num_attention_heads) * sizeof(float);
            size_t ffn_bytes = (size_t)max_seq * (size_t)intermediate_size * sizeof(float);
            size_t q_bytes = (size_t)max_seq * (size_t)q_dim * sizeof(float);
            size_t kv_bytes = (size_t)max_seq * (size_t)kv_dim * sizeof(float);

            for (int l = 0; l < num_layers; l++) {
                ctx->fwd_cache_x_norm_attn[l] = cuda_train_buffer_create(ctx, xnorm_bytes);
                ctx->fwd_cache_x_norm_ffn[l] = cuda_train_buffer_create(ctx, xnorm_bytes);
                ctx->fwd_cache_attn_out[l] = cuda_train_buffer_create(ctx, attn_bytes);
                ctx->fwd_cache_ffn_gate_out[l] = cuda_train_buffer_create(ctx, ffn_bytes);
                ctx->fwd_cache_ffn_up_out[l] = cuda_train_buffer_create(ctx, ffn_bytes);
                ctx->fwd_cache_ffn_hidden[l] = cuda_train_buffer_create(ctx, ffn_bytes);
                ctx->fwd_cache_layer_input[l] = cuda_train_buffer_create(ctx, xnorm_bytes);
                ctx->fwd_cache_post_attn_in[l] = cuda_train_buffer_create(ctx, xnorm_bytes);
                ctx->fwd_cache_silu_out[l] = cuda_train_buffer_create(ctx, ffn_bytes);
                ctx->fwd_cache_q[l] = cuda_train_buffer_create(ctx, q_bytes);
                ctx->fwd_cache_k[l] = cuda_train_buffer_create(ctx, kv_bytes);
                ctx->fwd_cache_v[l] = cuda_train_buffer_create(ctx, kv_bytes);
                ctx->fwd_cache_attn_lse[l] = cuda_train_buffer_create(ctx, lse_bytes);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BUFFER ALLOCATION - Training Buffers
// ═══════════════════════════════════════════════════════════════════════════

static void ensure_cuda_train_buffers(cuda_train_context_t* ctx,
                                      int max_seq,
                                      int hidden_size,
                                      int intermediate_size,
                                      int q_dim,
                                      int kv_dim,
                                      int vocab_size) {
    if (!ctx) return;

    size_t tokens_u32_bytes = (size_t)max_seq * sizeof(uint32_t);
    size_t hidden_bytes = (size_t)max_seq * (size_t)hidden_size * sizeof(float);
    size_t inter_bytes = (size_t)max_seq * (size_t)intermediate_size * sizeof(float);
    size_t q_bytes = (size_t)max_seq * (size_t)q_dim * sizeof(float);
    size_t kv_bytes = (size_t)max_seq * (size_t)kv_dim * sizeof(float);
    size_t logits_bytes = (size_t)max_seq * (size_t)vocab_size * sizeof(float);
    size_t loss_bytes = (size_t)max_seq * sizeof(float);

    #define ENSURE_BUF(buf, sz) \
        if (!ctx->buf || ctx->buf->size < (sz)) { \
            if (ctx->buf) cuda_train_buffer_free(ctx, ctx->buf); \
            ctx->buf = cuda_train_buffer_create(ctx, sz); \
        }

    ENSURE_BUF(train_tokens_u32, tokens_u32_bytes);
    ENSURE_BUF(train_targets_u32, tokens_u32_bytes);
    ENSURE_BUF(train_grad_logits, logits_bytes);
    ENSURE_BUF(train_loss_rows, loss_bytes);
    ENSURE_BUF(train_reduce_tmp_a, loss_bytes);
    ENSURE_BUF(train_reduce_tmp_b, loss_bytes);
    ENSURE_BUF(train_grad_hidden, hidden_bytes);
    ENSURE_BUF(train_grad_tmp_hidden, hidden_bytes);
    ENSURE_BUF(train_grad_x_norm, hidden_bytes);
    ENSURE_BUF(train_grad_q, q_bytes);
    ENSURE_BUF(train_grad_k, kv_bytes);
    ENSURE_BUF(train_grad_v, kv_bytes);
    ENSURE_BUF(train_grad_attn, q_bytes);
    ENSURE_BUF(train_grad_ffn, inter_bytes);
    ENSURE_BUF(train_grad_gate, inter_bytes);
    ENSURE_BUF(train_grad_up, inter_bytes);
    ENSURE_BUF(train_final_input, hidden_bytes);

    #undef ENSURE_BUF
}

// ═══════════════════════════════════════════════════════════════════════════
// BUFFER ALLOCATION - GPU-Resident Weights
// ═══════════════════════════════════════════════════════════════════════════

static void ensure_cuda_gpu_weights(cuda_train_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->full_weights || !state->config) return;

    const model_config_t* config = state->config;
    trainable_weights_t* weights = state->full_weights;
    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int intermediate_size = config->intermediate_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;

    // Check if we need to allocate/reallocate
    if (ctx->gpu_weights_num_layers == num_layers &&
        ctx->gpu_weights_hidden_size == hidden_size &&
        ctx->gpu_weights_vocab_size == vocab_size &&
        ctx->gpu_w_embed) {
        return;  // Already allocated with same config
    }

    ctx->gpu_weights_num_layers = num_layers;
    ctx->gpu_weights_hidden_size = hidden_size;
    ctx->gpu_weights_intermediate_size = intermediate_size;
    ctx->gpu_weights_q_dim = q_dim;
    ctx->gpu_weights_kv_dim = kv_dim;
    ctx->gpu_weights_vocab_size = vocab_size;

    size_t embed_bytes = (size_t)vocab_size * (size_t)hidden_size * sizeof(float);
    size_t norm_bytes = (size_t)hidden_size * sizeof(float);
    size_t q_bytes = (size_t)hidden_size * (size_t)q_dim * sizeof(float);
    size_t kv_bytes = (size_t)hidden_size * (size_t)kv_dim * sizeof(float);
    size_t o_bytes = (size_t)q_dim * (size_t)hidden_size * sizeof(float);
    size_t gate_bytes = (size_t)hidden_size * (size_t)intermediate_size * sizeof(float);
    size_t down_bytes = (size_t)intermediate_size * (size_t)hidden_size * sizeof(float);

    // Embedding
    ctx->gpu_w_embed = cuda_train_buffer_create(ctx, embed_bytes);
    ctx->gpu_g_embed = cuda_train_buffer_create(ctx, embed_bytes);
    ctx->gpu_m_embed = cuda_train_buffer_create(ctx, embed_bytes);
    ctx->gpu_v_embed = cuda_train_buffer_create(ctx, embed_bytes);

    cuda_train_buffer_upload(ctx, ctx->gpu_w_embed, weights->embed_tokens.weight, vocab_size * hidden_size);
    cuda_train_buffer_upload(ctx, ctx->gpu_m_embed, weights->embed_tokens.m, vocab_size * hidden_size);
    cuda_train_buffer_upload(ctx, ctx->gpu_v_embed, weights->embed_tokens.v, vocab_size * hidden_size);
    cuda_train_fill_buffer(ctx, ctx->gpu_g_embed, 0, embed_bytes);

    // Final norm
    ctx->gpu_w_final_norm = cuda_train_buffer_create(ctx, norm_bytes);
    ctx->gpu_g_final_norm = cuda_train_buffer_create(ctx, norm_bytes);
    ctx->gpu_m_final_norm = cuda_train_buffer_create(ctx, norm_bytes);
    ctx->gpu_v_final_norm = cuda_train_buffer_create(ctx, norm_bytes);

    cuda_train_buffer_upload(ctx, ctx->gpu_w_final_norm, weights->final_norm.weight, hidden_size);
    cuda_train_buffer_upload(ctx, ctx->gpu_m_final_norm, weights->final_norm.m, hidden_size);
    cuda_train_buffer_upload(ctx, ctx->gpu_v_final_norm, weights->final_norm.v, hidden_size);
    cuda_train_fill_buffer(ctx, ctx->gpu_g_final_norm, 0, norm_bytes);

    // Per-layer arrays
    #define ALLOC_LAYER_ARRAYS(name) \
        ctx->gpu_w_##name = calloc(num_layers, sizeof(cuda_buffer_t*)); \
        ctx->gpu_g_##name = calloc(num_layers, sizeof(cuda_buffer_t*)); \
        ctx->gpu_m_##name = calloc(num_layers, sizeof(cuda_buffer_t*)); \
        ctx->gpu_v_##name = calloc(num_layers, sizeof(cuda_buffer_t*))

    ALLOC_LAYER_ARRAYS(q);
    ALLOC_LAYER_ARRAYS(k);
    ALLOC_LAYER_ARRAYS(v);
    ALLOC_LAYER_ARRAYS(o);
    ALLOC_LAYER_ARRAYS(gate);
    ALLOC_LAYER_ARRAYS(up);
    ALLOC_LAYER_ARRAYS(down);
    ALLOC_LAYER_ARRAYS(in_norm);
    ALLOC_LAYER_ARRAYS(post_norm);

    #undef ALLOC_LAYER_ARRAYS

    for (int l = 0; l < num_layers; l++) {
        layer_weights_t* lw = &weights->layers[l];

        // Q projection
        ctx->gpu_w_q[l] = cuda_train_buffer_create(ctx, q_bytes);
        ctx->gpu_g_q[l] = cuda_train_buffer_create(ctx, q_bytes);
        ctx->gpu_m_q[l] = cuda_train_buffer_create(ctx, q_bytes);
        ctx->gpu_v_q[l] = cuda_train_buffer_create(ctx, q_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_q[l], lw->q_proj.weight, hidden_size * q_dim);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_q[l], lw->q_proj.m, hidden_size * q_dim);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_q[l], lw->q_proj.v, hidden_size * q_dim);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_q[l], 0, q_bytes);

        // K projection
        ctx->gpu_w_k[l] = cuda_train_buffer_create(ctx, kv_bytes);
        ctx->gpu_g_k[l] = cuda_train_buffer_create(ctx, kv_bytes);
        ctx->gpu_m_k[l] = cuda_train_buffer_create(ctx, kv_bytes);
        ctx->gpu_v_k[l] = cuda_train_buffer_create(ctx, kv_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_k[l], lw->k_proj.weight, hidden_size * kv_dim);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_k[l], lw->k_proj.m, hidden_size * kv_dim);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_k[l], lw->k_proj.v, hidden_size * kv_dim);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_k[l], 0, kv_bytes);

        // V projection
        ctx->gpu_w_v[l] = cuda_train_buffer_create(ctx, kv_bytes);
        ctx->gpu_g_v[l] = cuda_train_buffer_create(ctx, kv_bytes);
        ctx->gpu_m_v[l] = cuda_train_buffer_create(ctx, kv_bytes);
        ctx->gpu_v_v[l] = cuda_train_buffer_create(ctx, kv_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_v[l], lw->v_proj.weight, hidden_size * kv_dim);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_v[l], lw->v_proj.m, hidden_size * kv_dim);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_v[l], lw->v_proj.v, hidden_size * kv_dim);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_v[l], 0, kv_bytes);

        // O projection
        ctx->gpu_w_o[l] = cuda_train_buffer_create(ctx, o_bytes);
        ctx->gpu_g_o[l] = cuda_train_buffer_create(ctx, o_bytes);
        ctx->gpu_m_o[l] = cuda_train_buffer_create(ctx, o_bytes);
        ctx->gpu_v_o[l] = cuda_train_buffer_create(ctx, o_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_o[l], lw->o_proj.weight, q_dim * hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_o[l], lw->o_proj.m, q_dim * hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_o[l], lw->o_proj.v, q_dim * hidden_size);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_o[l], 0, o_bytes);

        // Gate projection
        ctx->gpu_w_gate[l] = cuda_train_buffer_create(ctx, gate_bytes);
        ctx->gpu_g_gate[l] = cuda_train_buffer_create(ctx, gate_bytes);
        ctx->gpu_m_gate[l] = cuda_train_buffer_create(ctx, gate_bytes);
        ctx->gpu_v_gate[l] = cuda_train_buffer_create(ctx, gate_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_gate[l], lw->gate_proj.weight, hidden_size * intermediate_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_gate[l], lw->gate_proj.m, hidden_size * intermediate_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_gate[l], lw->gate_proj.v, hidden_size * intermediate_size);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_gate[l], 0, gate_bytes);

        // Up projection
        ctx->gpu_w_up[l] = cuda_train_buffer_create(ctx, gate_bytes);
        ctx->gpu_g_up[l] = cuda_train_buffer_create(ctx, gate_bytes);
        ctx->gpu_m_up[l] = cuda_train_buffer_create(ctx, gate_bytes);
        ctx->gpu_v_up[l] = cuda_train_buffer_create(ctx, gate_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_up[l], lw->up_proj.weight, hidden_size * intermediate_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_up[l], lw->up_proj.m, hidden_size * intermediate_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_up[l], lw->up_proj.v, hidden_size * intermediate_size);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_up[l], 0, gate_bytes);

        // Down projection
        ctx->gpu_w_down[l] = cuda_train_buffer_create(ctx, down_bytes);
        ctx->gpu_g_down[l] = cuda_train_buffer_create(ctx, down_bytes);
        ctx->gpu_m_down[l] = cuda_train_buffer_create(ctx, down_bytes);
        ctx->gpu_v_down[l] = cuda_train_buffer_create(ctx, down_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_down[l], lw->down_proj.weight, intermediate_size * hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_down[l], lw->down_proj.m, intermediate_size * hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_down[l], lw->down_proj.v, intermediate_size * hidden_size);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_down[l], 0, down_bytes);

        // Input norm
        ctx->gpu_w_in_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        ctx->gpu_g_in_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        ctx->gpu_m_in_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        ctx->gpu_v_in_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_in_norm[l], lw->input_norm.weight, hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_in_norm[l], lw->input_norm.m, hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_in_norm[l], lw->input_norm.v, hidden_size);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_in_norm[l], 0, norm_bytes);

        // Post norm
        ctx->gpu_w_post_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        ctx->gpu_g_post_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        ctx->gpu_m_post_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        ctx->gpu_v_post_norm[l] = cuda_train_buffer_create(ctx, norm_bytes);
        cuda_train_buffer_upload(ctx, ctx->gpu_w_post_norm[l], lw->post_norm.weight, hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_m_post_norm[l], lw->post_norm.m, hidden_size);
        cuda_train_buffer_upload(ctx, ctx->gpu_v_post_norm[l], lw->post_norm.v, hidden_size);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_post_norm[l], 0, norm_bytes);
    }

    cuda_train_sync(ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// ZERO GRADIENTS
// ═══════════════════════════════════════════════════════════════════════════

static void cuda_zero_full_weight_grads(cuda_train_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->config) return;

    const model_config_t* config = state->config;
    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int intermediate_size = config->intermediate_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;

    size_t embed_bytes = (size_t)vocab_size * (size_t)hidden_size * sizeof(float);
    size_t norm_bytes = (size_t)hidden_size * sizeof(float);
    size_t q_bytes = (size_t)hidden_size * (size_t)q_dim * sizeof(float);
    size_t kv_bytes = (size_t)hidden_size * (size_t)kv_dim * sizeof(float);
    size_t o_bytes = (size_t)q_dim * (size_t)hidden_size * sizeof(float);
    size_t gate_bytes = (size_t)hidden_size * (size_t)intermediate_size * sizeof(float);
    size_t down_bytes = (size_t)intermediate_size * (size_t)hidden_size * sizeof(float);

    // Global weights
    cuda_train_fill_buffer(ctx, ctx->gpu_g_embed, 0, embed_bytes);
    cuda_train_fill_buffer(ctx, ctx->gpu_g_final_norm, 0, norm_bytes);

    // Per-layer
    for (int l = 0; l < num_layers; l++) {
        cuda_train_fill_buffer(ctx, ctx->gpu_g_q[l], 0, q_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_k[l], 0, kv_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_v[l], 0, kv_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_o[l], 0, o_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_gate[l], 0, gate_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_up[l], 0, gate_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_down[l], 0, down_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_in_norm[l], 0, norm_bytes);
        cuda_train_fill_buffer(ctx, ctx->gpu_g_post_norm[l], 0, norm_bytes);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZER STEP (GPU-Resident)
// ═══════════════════════════════════════════════════════════════════════════

static void cuda_optimizer_step_gpu_resident(train_state_t* state, float lr,
                                              float beta1, float beta2,
                                              float weight_decay, float eps,
                                              int step, cuda_train_context_t* ctx) {
    if (!ctx || !state || !state->config) return;

    const model_config_t* config = state->config;
    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int intermediate_size = config->intermediate_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;

    // Global weights
    cuda_train_adamw_update(ctx, ctx->gpu_w_embed, ctx->gpu_g_embed,
                            ctx->gpu_m_embed, ctx->gpu_v_embed,
                            lr, beta1, beta2, weight_decay, eps, step,
                            (size_t)vocab_size * (size_t)hidden_size,
                            "w_embed", vocab_size, hidden_size);

    cuda_train_adamw_update(ctx, ctx->gpu_w_final_norm, ctx->gpu_g_final_norm,
                            ctx->gpu_m_final_norm, ctx->gpu_v_final_norm,
                            lr, beta1, beta2, weight_decay, eps, step,
                            (size_t)hidden_size,
                            "rms_final", hidden_size, 0);

    // Per-layer
    for (int l = 0; l < num_layers; l++) {
        cuda_train_adamw_update(ctx, ctx->gpu_w_q[l], ctx->gpu_g_q[l],
                                ctx->gpu_m_q[l], ctx->gpu_v_q[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)hidden_size * (size_t)q_dim,
                                "w_q", hidden_size, q_dim);

        cuda_train_adamw_update(ctx, ctx->gpu_w_k[l], ctx->gpu_g_k[l],
                                ctx->gpu_m_k[l], ctx->gpu_v_k[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)hidden_size * (size_t)kv_dim,
                                "w_k", hidden_size, kv_dim);

        cuda_train_adamw_update(ctx, ctx->gpu_w_v[l], ctx->gpu_g_v[l],
                                ctx->gpu_m_v[l], ctx->gpu_v_v[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)hidden_size * (size_t)kv_dim,
                                "w_v", hidden_size, kv_dim);

        cuda_train_adamw_update(ctx, ctx->gpu_w_o[l], ctx->gpu_g_o[l],
                                ctx->gpu_m_o[l], ctx->gpu_v_o[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)q_dim * (size_t)hidden_size,
                                "w_o", q_dim, hidden_size);

        cuda_train_adamw_update(ctx, ctx->gpu_w_gate[l], ctx->gpu_g_gate[l],
                                ctx->gpu_m_gate[l], ctx->gpu_v_gate[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)hidden_size * (size_t)intermediate_size,
                                "w_gate", hidden_size, intermediate_size);

        cuda_train_adamw_update(ctx, ctx->gpu_w_up[l], ctx->gpu_g_up[l],
                                ctx->gpu_m_up[l], ctx->gpu_v_up[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)hidden_size * (size_t)intermediate_size,
                                "w_up", hidden_size, intermediate_size);

        cuda_train_adamw_update(ctx, ctx->gpu_w_down[l], ctx->gpu_g_down[l],
                                ctx->gpu_m_down[l], ctx->gpu_v_down[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)intermediate_size * (size_t)hidden_size,
                                "w_down", intermediate_size, hidden_size);

        cuda_train_adamw_update(ctx, ctx->gpu_w_in_norm[l], ctx->gpu_g_in_norm[l],
                                ctx->gpu_m_in_norm[l], ctx->gpu_v_in_norm[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)hidden_size,
                                "rms_attn", hidden_size, 0);

        cuda_train_adamw_update(ctx, ctx->gpu_w_post_norm[l], ctx->gpu_g_post_norm[l],
                                ctx->gpu_m_post_norm[l], ctx->gpu_v_post_norm[l],
                                lr, beta1, beta2, weight_decay, eps, step,
                                (size_t)hidden_size,
                                "rms_ffn", hidden_size, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NaN DEBUG HELPERS
// ═══════════════════════════════════════════════════════════════════════════

static void dump_cuda_buffer_f32(const char* path, cuda_train_context_t* ctx,
                                  cuda_buffer_t* buf, size_t count) {
    float* data = malloc(count * sizeof(float));
    if (!data) return;
    cuda_train_buffer_download(ctx, buf, data, count);
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(data, sizeof(float), count, f);
        fclose(f);
    }
    free(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN TRAINING STEP (GPU Full Training)
// ═══════════════════════════════════════════════════════════════════════════

float cuda_train_step_gpu_full(train_state_t* state, int* tokens, int num_tokens,
                                cuda_train_context_t* ctx) {
    if (!state || !state->full_weights || !state->config || !ctx) return -1.0f;

    double t0_ms = now_ms();

    const model_config_t* config = state->config;
    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int vocab_size = config->vocab_size;
    const int num_layers = config->num_hidden_layers;
    const int num_heads = config->num_attention_heads;
    const int head_dim = config->head_dim;
    const int kv_heads = config->num_key_value_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = kv_heads * head_dim;
    const float rms_eps = 1e-5f;
    const float rope_theta = config->rope_theta > 0 ? config->rope_theta : 10000.0f;

    int max_seq = state->act_cache ? state->act_cache->max_seq_len : num_tokens;
    if (max_seq < num_tokens) max_seq = num_tokens;

    // Ensure all buffers are allocated
    ensure_cuda_fwd_buffers(ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_train_buffers(ctx, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_gpu_weights(ctx, state);

    // NaN flag indexing (matches Vulkan):
    //   [0] grad_hidden (after final-norm backward copy)
    //   [1] logits
    //   [2] hidden_norm (input to LM head)
    //   [3] grad_logits
    //   [4] grad_hidden (after LM head backward)
    //   [5] grad_tmp_hidden (output of final rmsnorm backward)
    //   [6] final_input (input to final rmsnorm backward)
    //   [7] hidden (after embedding lookup)
    //   [8] grad_hidden (right before embedding_backward)
    //   [9] g_embed (after embedding_backward)
    //   [10..10+num_layers-1] hidden after each forward layer
    //   [nan_bwd_layer_base + layer*5 + 0..4] backward layer checks
    //   [nan_grad_hidden_bwd_layer_base + layer] grad_hidden after layer backward
    //   [nan_preopt_base + 0..4] pre/post optimizer embed checks
    uint32_t nan_flag_count = 0;
    uint32_t nan_fwd_layer_base = 10;
    uint32_t nan_bwd_layer_base = nan_fwd_layer_base + (uint32_t)num_layers;
    uint32_t nan_grad_hidden_bwd_layer_base = nan_bwd_layer_base + (uint32_t)num_layers * 5u;
    uint32_t nan_preopt_base = nan_grad_hidden_bwd_layer_base + (uint32_t)num_layers;
    uint32_t nan_weight_embed_idx = nan_preopt_base + 0u;
    nan_flag_count = nan_preopt_base + 5u;

    // Allocate/clear NaN flags if debug enabled
    if (state->debug_nan_check) {
        size_t nan_bytes = (size_t)nan_flag_count * sizeof(uint32_t);
        if (!ctx->debug_nan_flags || ctx->debug_nan_flags->size < nan_bytes) {
            if (ctx->debug_nan_flags) cuda_train_buffer_free(ctx, ctx->debug_nan_flags);
            ctx->debug_nan_flags = cuda_train_buffer_create(ctx, nan_bytes);
        }
        if (ctx->debug_nan_flags) {
            cuda_train_fill_buffer(ctx, ctx->debug_nan_flags, 0, nan_bytes);
        }

        // Check whether embedding weights are already poisoned at start
        cuda_train_nan_check(ctx, ctx->gpu_w_embed,
                             (size_t)vocab_size * (size_t)hidden_size,
                             ctx->debug_nan_flags, nan_weight_embed_idx);
    }

    // Zero grads at start of accumulation cycle
    if (state->accumulation_counter == 0) {
        cuda_zero_full_weight_grads(ctx, state);
    }

    // Get buffer pointers
    cuda_buffer_t* buf_hidden = ctx->fwd_hidden_buffer;
    cuda_buffer_t* buf_hidden_norm = ctx->fwd_hidden_norm_buffer;
    cuda_buffer_t* buf_tmp_hidden = ctx->fwd_tmp_hidden_buffer;
    cuda_buffer_t* buf_gate = ctx->fwd_ffn_gate_buffer;
    cuda_buffer_t* buf_up = ctx->fwd_ffn_up_buffer;
    cuda_buffer_t* buf_ffn_hidden = ctx->fwd_ffn_hidden_buffer;
    cuda_buffer_t* buf_logits = ctx->fwd_logits_buffer;

    cuda_buffer_t* buf_Q = ctx->attn_q_buffer;
    cuda_buffer_t* buf_K = ctx->attn_k_buffer;
    cuda_buffer_t* buf_V = ctx->attn_v_buffer;
    cuda_buffer_t* buf_attn = ctx->attn_out_buffer;

    cuda_buffer_t** cache_layer_in = ctx->fwd_cache_layer_input;
    cuda_buffer_t** cache_x_norm_attn = ctx->fwd_cache_x_norm_attn;
    cuda_buffer_t** cache_attn_out = ctx->fwd_cache_attn_out;
    cuda_buffer_t** cache_post_attn_in = ctx->fwd_cache_post_attn_in;
    cuda_buffer_t** cache_x_norm_ffn = ctx->fwd_cache_x_norm_ffn;
    cuda_buffer_t** cache_ffn_gate_out = ctx->fwd_cache_ffn_gate_out;
    cuda_buffer_t** cache_ffn_up_out = ctx->fwd_cache_ffn_up_out;
    cuda_buffer_t** cache_silu_out = ctx->fwd_cache_silu_out;
    cuda_buffer_t** cache_ffn_hidden = ctx->fwd_cache_ffn_hidden;
    cuda_buffer_t** cache_q = ctx->fwd_cache_q;
    cuda_buffer_t** cache_k = ctx->fwd_cache_k;
    cuda_buffer_t** cache_v = ctx->fwd_cache_v;

    // Upload tokens + targets
    uint32_t* tokens_u32 = malloc((size_t)num_tokens * sizeof(uint32_t));
    if (!tokens_u32) return -1.0f;
    for (int i = 0; i < num_tokens; i++) {
        tokens_u32[i] = (tokens[i] < 0) ? 0u : (uint32_t)tokens[i];
    }
    cuda_train_buffer_upload_u32(ctx, ctx->train_tokens_u32, tokens_u32, (size_t)num_tokens);

    int num_predictions = num_tokens - 1;
    uint32_t* targets_u32 = malloc((size_t)num_predictions * sizeof(uint32_t));
    if (!targets_u32) {
        free(tokens_u32);
        return -1.0f;
    }
    for (int t = 0; t < num_predictions; t++) {
        int tok = tokens[t + 1];
        targets_u32[t] = (tok < 0) ? 0u : (uint32_t)tok;
    }
    cuda_train_buffer_upload_u32(ctx, ctx->train_targets_u32, targets_u32, (size_t)num_predictions);
    free(tokens_u32);
    free(targets_u32);

    // Ensure last row of grad_logits is zero (cross_entropy_grad only writes num_predictions rows)
    if (num_tokens > 0) {
        size_t last_row_off = (size_t)(num_tokens - 1) * (size_t)vocab_size * sizeof(float);
        cuda_train_fill_buffer_range(ctx, ctx->train_grad_logits, last_row_off, 0,
                                     (size_t)vocab_size * sizeof(float));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FORWARD PASS
    // ═══════════════════════════════════════════════════════════════════════

    // Embedding lookup -> hidden
    cuda_train_embed_lookup(ctx, ctx->train_tokens_u32, ctx->gpu_w_embed, buf_hidden,
                            num_tokens, hidden_size, vocab_size);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, buf_hidden,
                             (size_t)num_tokens * (size_t)hidden_size,
                             ctx->debug_nan_flags, 7u);
    }

    // Transformer layers
    for (int layer = 0; layer < num_layers; layer++) {
        // Cache layer input
        cuda_train_copy_buffer(ctx, buf_hidden, cache_layer_in[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // RMSNorm (pre-attention)
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_in_norm[layer], buf_hidden_norm,
                           num_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_attn[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Q, K, V projections (output = X @ W^T)
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_q[layer], buf_Q,
                                    num_tokens, q_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_k[layer], buf_K,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_v[layer], buf_V,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);

        // RoPE
        cuda_train_rope(ctx, buf_Q, buf_Q, num_tokens, num_heads, head_dim, rope_theta);
        cuda_train_rope(ctx, buf_K, buf_K, num_tokens, kv_heads, head_dim, rope_theta);

        // Cache Q/K/V for backward
        cuda_train_copy_buffer(ctx, buf_Q, cache_q[layer], (size_t)num_tokens * (size_t)q_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_K, cache_k[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_V, cache_v[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));

        // Attention (FlashAttention: stores logsumexp per layer for backward)
        cuda_train_batch_attention(ctx, buf_Q, buf_K, buf_V, buf_attn,
                                   ctx->fwd_cache_attn_lse[layer],
                                   num_tokens, num_heads, kv_heads, head_dim);
        cuda_train_copy_buffer(ctx, buf_attn, cache_attn_out[layer],
                               (size_t)num_tokens * (size_t)q_dim * sizeof(float));

        // O projection + residual
        cuda_train_matmul_transpose(ctx, buf_attn, ctx->gpu_w_o[layer], buf_tmp_hidden,
                                    num_tokens, hidden_size, q_dim, 0, 1, 0);
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);
        cuda_train_copy_buffer(ctx, buf_hidden, cache_post_attn_in[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // RMSNorm (post-attention)
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_post_norm[layer], buf_hidden_norm,
                           num_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_ffn[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Gate + up projections
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_gate[layer], buf_gate,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        cuda_train_copy_buffer(ctx, buf_gate, cache_ffn_gate_out[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_up[layer], buf_up,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        cuda_train_copy_buffer(ctx, buf_up, cache_ffn_up_out[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        // SiLU + multiply
        cuda_train_silu(ctx, buf_gate, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
        cuda_train_copy_buffer(ctx, buf_ffn_hidden, cache_silu_out[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));
        cuda_train_mul(ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden,
                       (size_t)num_tokens * (size_t)intermediate_size);
        cuda_train_copy_buffer(ctx, buf_ffn_hidden, cache_ffn_hidden[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        // Down projection + residual
        cuda_train_matmul_transpose(ctx, buf_ffn_hidden, ctx->gpu_w_down[layer], buf_tmp_hidden,
                                    num_tokens, hidden_size, intermediate_size, 0, 1, 0);
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            cuda_train_nan_check(ctx, buf_hidden,
                                 (size_t)num_tokens * (size_t)hidden_size,
                                 ctx->debug_nan_flags,
                                 nan_fwd_layer_base + (uint32_t)layer);
        }
    }

    // Save pre-final-norm input for backward
    cuda_train_copy_buffer(ctx, buf_hidden, ctx->train_final_input,
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

    // Final RMSNorm
    cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_final_norm, buf_hidden_norm,
                       num_tokens, hidden_size, rms_eps);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, buf_hidden_norm,
                             (size_t)num_tokens * (size_t)hidden_size,
                             ctx->debug_nan_flags, 2u);
        cuda_train_nan_check(ctx, ctx->train_final_input,
                             (size_t)num_tokens * (size_t)hidden_size,
                             ctx->debug_nan_flags, 6u);
    }

    // LM head (tied embeddings): logits = hidden_norm @ embed^T
    cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_embed, buf_logits,
                                num_tokens, vocab_size, hidden_size, 0, 1, 0);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, buf_logits,
                             (size_t)num_tokens * (size_t)vocab_size,
                             ctx->debug_nan_flags, 1u);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LOSS + BACKWARD
    // ═══════════════════════════════════════════════════════════════════════

    // Cross-entropy loss + gradient
    cuda_train_cross_entropy_grad(ctx, buf_logits, ctx->train_targets_u32,
                                  ctx->train_grad_logits, ctx->train_loss_rows,
                                  num_predictions, vocab_size);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, ctx->train_grad_logits,
                             (size_t)num_predictions * (size_t)vocab_size,
                             ctx->debug_nan_flags, 3u);
    }

    // Reduce sum loss
    size_t reduce_count = (size_t)num_predictions;
    cuda_buffer_t* reduce_in = ctx->train_loss_rows;
    cuda_buffer_t* reduce_out = ctx->train_reduce_tmp_a;
    while (reduce_count > 1) {
        size_t reduce_block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
        size_t out_count = (reduce_count + reduce_block - 1) / reduce_block;
        cuda_train_reduce_sum(ctx, reduce_in, reduce_out, reduce_count);
        reduce_count = out_count;
        cuda_buffer_t* tmp = reduce_in;
        reduce_in = reduce_out;
        reduce_out = tmp;
    }

    // LM head backward: grad_hidden_norm = grad_logits @ embed_weight
    cuda_train_matmul(ctx, ctx->train_grad_logits, ctx->gpu_w_embed,
                      ctx->train_grad_hidden, num_tokens, hidden_size, vocab_size);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, ctx->train_grad_hidden,
                             (size_t)num_tokens * (size_t)hidden_size,
                             ctx->debug_nan_flags, 4u);
    }

    // LM head weight grad: grad_embed += grad_logits^T @ hidden_norm
    cuda_train_matmul_transpose(ctx, ctx->train_grad_logits, buf_hidden_norm,
                                ctx->gpu_g_embed, vocab_size, hidden_size, num_tokens, 1, 0, 1);

    // Final norm backward
    cuda_train_rmsnorm_backward_batch(ctx, ctx->train_final_input, ctx->gpu_w_final_norm,
                                      ctx->train_grad_hidden, ctx->train_grad_tmp_hidden,
                                      ctx->gpu_g_final_norm, num_tokens, hidden_size, rms_eps);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, ctx->train_grad_tmp_hidden,
                             (size_t)num_tokens * (size_t)hidden_size,
                             ctx->debug_nan_flags, 5u);
    }

    // grad_hidden = grad_final_input
    cuda_train_copy_buffer(ctx, ctx->train_grad_tmp_hidden, ctx->train_grad_hidden,
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, ctx->train_grad_hidden,
                             (size_t)num_tokens * (size_t)hidden_size,
                             ctx->debug_nan_flags, 0u);
    }

    // Layer backward loop
    for (int layer = num_layers - 1; layer >= 0; layer--) {
        // FFN backward
        // grad_Wdown += grad_hidden^T @ ffn_hidden
        cuda_train_matmul_transpose(ctx, ctx->train_grad_hidden, cache_ffn_hidden[layer],
                                    ctx->gpu_g_down[layer], hidden_size, intermediate_size,
                                    num_tokens, 1, 0, 1);

        // grad_ffn_hidden = grad_hidden @ Wdown
        cuda_train_matmul(ctx, ctx->train_grad_hidden, ctx->gpu_w_down[layer],
                          ctx->train_grad_ffn, num_tokens, intermediate_size, hidden_size);

        // mul backward: grad_silu_out, grad_up
        cuda_train_elementwise_mul_backward(ctx, cache_silu_out[layer], cache_ffn_up_out[layer],
                                            ctx->train_grad_ffn, ctx->train_grad_gate,
                                            ctx->train_grad_up,
                                            (size_t)num_tokens * (size_t)intermediate_size);

        // silu backward
        cuda_train_silu_backward(ctx, cache_ffn_gate_out[layer], ctx->train_grad_gate,
                                 ctx->train_grad_gate,
                                 (size_t)num_tokens * (size_t)intermediate_size);

        // grad_Wgate += grad_gate^T @ x_norm_ffn
        cuda_train_matmul_transpose(ctx, ctx->train_grad_gate, cache_x_norm_ffn[layer],
                                    ctx->gpu_g_gate[layer], intermediate_size, hidden_size,
                                    num_tokens, 1, 0, 1);

        // grad_Wup += grad_up^T @ x_norm_ffn
        cuda_train_matmul_transpose(ctx, ctx->train_grad_up, cache_x_norm_ffn[layer],
                                    ctx->gpu_g_up[layer], intermediate_size, hidden_size,
                                    num_tokens, 1, 0, 1);

        // grad_x_norm_ffn = grad_gate @ Wgate + grad_up @ Wup
        cuda_train_matmul(ctx, ctx->train_grad_gate, ctx->gpu_w_gate[layer],
                          ctx->train_grad_x_norm, num_tokens, hidden_size, intermediate_size);
        cuda_train_matmul(ctx, ctx->train_grad_up, ctx->gpu_w_up[layer],
                          ctx->train_grad_tmp_hidden, num_tokens, hidden_size, intermediate_size);
        cuda_train_add(ctx, ctx->train_grad_x_norm, ctx->train_grad_tmp_hidden,
                       ctx->train_grad_x_norm, (size_t)num_tokens * (size_t)hidden_size);

        // post_norm backward
        cuda_train_rmsnorm_backward_batch(ctx, cache_post_attn_in[layer],
                                          ctx->gpu_w_post_norm[layer],
                                          ctx->train_grad_x_norm,
                                          ctx->train_grad_tmp_hidden,
                                          ctx->gpu_g_post_norm[layer],
                                          num_tokens, hidden_size, rms_eps);

        // Add residual gradient
        cuda_train_add(ctx, ctx->train_grad_tmp_hidden, ctx->train_grad_hidden,
                       ctx->train_grad_hidden, (size_t)num_tokens * (size_t)hidden_size);

        // Attention backward
        // grad_Wo += grad_hidden^T @ attn_out
        cuda_train_matmul_transpose(ctx, ctx->train_grad_hidden, cache_attn_out[layer],
                                    ctx->gpu_g_o[layer], hidden_size, q_dim,
                                    num_tokens, 1, 0, 1);

        // grad_attn = grad_hidden @ Wo
        cuda_train_matmul(ctx, ctx->train_grad_hidden, ctx->gpu_w_o[layer],
                          ctx->train_grad_attn, num_tokens, q_dim, hidden_size);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            uint32_t base = nan_bwd_layer_base + (uint32_t)layer * 5u;
            cuda_train_nan_check(ctx, ctx->train_grad_attn,
                                 (size_t)num_tokens * (size_t)q_dim,
                                 ctx->debug_nan_flags, base + 0u);
        }

        // Zero atomic accumulation buffers
        cuda_train_fill_buffer(ctx, ctx->train_grad_k, 0,
                               (size_t)num_tokens * (size_t)kv_dim * sizeof(float));
        cuda_train_fill_buffer(ctx, ctx->train_grad_v, 0,
                               (size_t)num_tokens * (size_t)kv_dim * sizeof(float));

        // Attention backward (FlashAttention: uses cached logsumexp + forward output)
        cuda_train_batch_attention_backward(ctx, cache_q[layer], cache_k[layer], cache_v[layer],
                                            cache_attn_out[layer], ctx->fwd_cache_attn_lse[layer],
                                            ctx->train_grad_attn, ctx->train_grad_q,
                                            ctx->train_grad_k, ctx->train_grad_v,
                                            num_tokens, num_heads, kv_heads, head_dim);

        // RoPE backward
        cuda_train_rope_backward(ctx, ctx->train_grad_q, ctx->train_grad_q,
                                 num_tokens, num_heads, head_dim, rope_theta);
        cuda_train_rope_backward(ctx, ctx->train_grad_k, ctx->train_grad_k,
                                 num_tokens, kv_heads, head_dim, rope_theta);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            uint32_t base = nan_bwd_layer_base + (uint32_t)layer * 5u;
            cuda_train_nan_check(ctx, ctx->train_grad_q,
                                 (size_t)num_tokens * (size_t)q_dim,
                                 ctx->debug_nan_flags, base + 1u);
            cuda_train_nan_check(ctx, ctx->train_grad_k,
                                 (size_t)num_tokens * (size_t)kv_dim,
                                 ctx->debug_nan_flags, base + 2u);
            cuda_train_nan_check(ctx, ctx->train_grad_v,
                                 (size_t)num_tokens * (size_t)kv_dim,
                                 ctx->debug_nan_flags, base + 3u);
        }

        // grad_Wq += dQ^T @ x_norm_attn
        cuda_train_matmul_transpose(ctx, ctx->train_grad_q, cache_x_norm_attn[layer],
                                    ctx->gpu_g_q[layer], q_dim, hidden_size,
                                    num_tokens, 1, 0, 1);

        // grad_Wk += dK^T @ x_norm_attn
        cuda_train_matmul_transpose(ctx, ctx->train_grad_k, cache_x_norm_attn[layer],
                                    ctx->gpu_g_k[layer], kv_dim, hidden_size,
                                    num_tokens, 1, 0, 1);

        // grad_Wv += dV^T @ x_norm_attn
        cuda_train_matmul_transpose(ctx, ctx->train_grad_v, cache_x_norm_attn[layer],
                                    ctx->gpu_g_v[layer], kv_dim, hidden_size,
                                    num_tokens, 1, 0, 1);

        // grad_x_norm_attn = dQ @ Wq + dK @ Wk + dV @ Wv
        cuda_train_matmul(ctx, ctx->train_grad_q, ctx->gpu_w_q[layer],
                          ctx->train_grad_x_norm, num_tokens, hidden_size, q_dim);
        cuda_train_matmul(ctx, ctx->train_grad_k, ctx->gpu_w_k[layer],
                          ctx->train_grad_tmp_hidden, num_tokens, hidden_size, kv_dim);
        cuda_train_add(ctx, ctx->train_grad_x_norm, ctx->train_grad_tmp_hidden,
                       ctx->train_grad_x_norm, (size_t)num_tokens * (size_t)hidden_size);
        cuda_train_matmul(ctx, ctx->train_grad_v, ctx->gpu_w_v[layer],
                          ctx->train_grad_tmp_hidden, num_tokens, hidden_size, kv_dim);
        cuda_train_add(ctx, ctx->train_grad_x_norm, ctx->train_grad_tmp_hidden,
                       ctx->train_grad_x_norm, (size_t)num_tokens * (size_t)hidden_size);

        // input_norm backward
        cuda_train_rmsnorm_backward_batch(ctx, cache_layer_in[layer],
                                          ctx->gpu_w_in_norm[layer],
                                          ctx->train_grad_x_norm,
                                          ctx->train_grad_tmp_hidden,
                                          ctx->gpu_g_in_norm[layer],
                                          num_tokens, hidden_size, rms_eps);

        // Add residual gradient
        cuda_train_add(ctx, ctx->train_grad_tmp_hidden, ctx->train_grad_hidden,
                       ctx->train_grad_hidden, (size_t)num_tokens * (size_t)hidden_size);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            cuda_train_nan_check(ctx, ctx->train_grad_hidden,
                                 (size_t)num_tokens * (size_t)hidden_size,
                                 ctx->debug_nan_flags,
                                 nan_grad_hidden_bwd_layer_base + (uint32_t)layer);
        }
    }

    // Embedding gradient from input lookup
    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, ctx->train_grad_hidden,
                             (size_t)num_tokens * (size_t)hidden_size,
                             ctx->debug_nan_flags, 8u);
    }

    cuda_train_embedding_backward(ctx, ctx->train_tokens_u32, ctx->train_grad_hidden,
                                  ctx->gpu_g_embed, num_tokens, hidden_size, vocab_size);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        cuda_train_nan_check(ctx, ctx->gpu_g_embed,
                             (size_t)vocab_size * (size_t)hidden_size,
                             ctx->debug_nan_flags, 9u);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NaN CHECK - Skip optimizer if NaN detected
    // ═══════════════════════════════════════════════════════════════════════

    // Synchronize to read back NaN flags
    cuda_train_sync(ctx);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        uint32_t* nan_flags_host = calloc(nan_flag_count, sizeof(uint32_t));
        if (nan_flags_host) {
            cuda_train_buffer_download_bytes(ctx, ctx->debug_nan_flags, nan_flags_host,
                                             (size_t)nan_flag_count * sizeof(uint32_t));

            int any_nan = 0;
            for (uint32_t i = 0; i < nan_flag_count; i++) {
                if (nan_flags_host[i] != 0) { any_nan = 1; break; }
            }

            if (any_nan) {
                fprintf(stderr, "[NAN] detected at step=%d seq=%d (optimizer skipped)\n",
                        state->current_step + 1, num_tokens);
                fprintf(stderr, "  - lr(base)=%g opt_step=%d accum=%d/%d\n",
                        (double)state->learning_rate,
                        state->optimizer_step,
                        state->accumulation_counter,
                        state->gradient_accumulation_steps);

                // Report which buffer(s) are NaN
                if (nan_flags_host[0]) fprintf(stderr, "  - buffer: grad_hidden (post final-norm backward)\n");
                if (nan_flags_host[1]) fprintf(stderr, "  - buffer: logits (LM head output)\n");
                if (nan_flags_host[2]) fprintf(stderr, "  - buffer: hidden_norm (LM head input)\n");
                if (nan_flags_host[3]) fprintf(stderr, "  - buffer: grad_logits (cross-entropy grad)\n");
                if (nan_flags_host[4]) fprintf(stderr, "  - buffer: grad_hidden (after LM head backward)\n");
                if (nan_flags_host[5]) fprintf(stderr, "  - buffer: grad_tmp_hidden (final rmsnorm backward output)\n");
                if (nan_flags_host[6]) fprintf(stderr, "  - buffer: final_input (pre final rmsnorm)\n");
                if (nan_flags_host[7]) fprintf(stderr, "  - buffer: hidden (post embedding)\n");
                if (nan_flags_host[8]) fprintf(stderr, "  - buffer: grad_hidden (pre embedding_backward)\n");
                if (nan_flags_host[9]) fprintf(stderr, "  - buffer: g_embed (post embedding_backward)\n");
                if (nan_weight_embed_idx < nan_flag_count && nan_flags_host[nan_weight_embed_idx]) {
                    fprintf(stderr, "  - buffer: w_embed (weights already NaN/Inf at step start)\n");
                }

                // Check per-layer forward
                for (int l = 0; l < num_layers; l++) {
                    if (nan_flags_host[nan_fwd_layer_base + (uint32_t)l]) {
                        fprintf(stderr, "  - buffer: hidden (post layer) layer=%d\n", l);
                        break;
                    }
                }

                // Check per-layer backward
                for (int l = 0; l < num_layers; l++) {
                    uint32_t base = nan_bwd_layer_base + (uint32_t)l * 5u;
                    if (nan_flags_host[base + 0u]) { fprintf(stderr, "  - buffer: grad_attn (input to attn backward) layer=%d\n", l); break; }
                    if (nan_flags_host[base + 1u]) { fprintf(stderr, "  - buffer: grad_q (post rope_backward) layer=%d\n", l); break; }
                    if (nan_flags_host[base + 2u]) { fprintf(stderr, "  - buffer: grad_k (post attn backward) layer=%d\n", l); break; }
                    if (nan_flags_host[base + 3u]) { fprintf(stderr, "  - buffer: grad_v (post attn backward) layer=%d\n", l); break; }
                }

                // Check grad_hidden after layer backward
                for (int l = 0; l < num_layers; l++) {
                    if (nan_flags_host[nan_grad_hidden_bwd_layer_base + (uint32_t)l]) {
                        fprintf(stderr, "  - buffer: grad_hidden (after layer backward) layer=%d\n", l);
                        break;
                    }
                }

                // Dump buffers for offline inspection
                char dump_prefix[256];
                snprintf(dump_prefix, sizeof(dump_prefix), "/tmp/seraph_nan_step_%d", state->current_step + 1);
                char path[512];

                // Metadata
                snprintf(path, sizeof(path), "%s_meta.txt", dump_prefix);
                FILE* meta = fopen(path, "wb");
                if (meta) {
                    fprintf(meta, "step=%d\n", state->current_step + 1);
                    fprintf(meta, "seq=%d\n", num_tokens);
                    fprintf(meta, "num_predictions=%d\n", num_predictions);
                    fprintf(meta, "hidden_size=%d\n", hidden_size);
                    fprintf(meta, "vocab_size=%d\n", vocab_size);
                    fprintf(meta, "num_layers=%d\n", num_layers);
                    fprintf(meta, "lr=%g\n", (double)state->learning_rate);
                    fprintf(meta, "backend=cuda\n");
                    fclose(meta);
                }

                // Dump key buffers
                snprintf(path, sizeof(path), "%s_grad_hidden.f32", dump_prefix);
                dump_cuda_buffer_f32(path, ctx, ctx->train_grad_hidden,
                                     (size_t)num_tokens * (size_t)hidden_size);
                snprintf(path, sizeof(path), "%s_logits.f32", dump_prefix);
                dump_cuda_buffer_f32(path, ctx, buf_logits,
                                     (size_t)num_tokens * (size_t)vocab_size);
                snprintf(path, sizeof(path), "%s_grad_logits.f32", dump_prefix);
                dump_cuda_buffer_f32(path, ctx, ctx->train_grad_logits,
                                     (size_t)num_tokens * (size_t)vocab_size);
                snprintf(path, sizeof(path), "%s_g_embed.f32", dump_prefix);
                dump_cuda_buffer_f32(path, ctx, ctx->gpu_g_embed,
                                     (size_t)vocab_size * (size_t)hidden_size);

                fprintf(stderr, "[NAN-DUMP] wrote buffers with prefix %s_*\n", dump_prefix);

                state->nan_detected = 1;
                free(nan_flags_host);
                return NAN;
            }
            free(nan_flags_host);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIMIZER STEP
    // ═══════════════════════════════════════════════════════════════════════

    state->accumulation_counter++;
    int accum_steps = state->gradient_accumulation_steps;
    if (accum_steps < 1) accum_steps = 1;

    int do_opt = (state->accumulation_counter >= accum_steps);

    if (do_opt) {
        state->optimizer_step++;

        float base_lr = state->learning_rate;
        float warmup_steps = 100.0f;
        float lr = base_lr;
        if (state->optimizer_step < warmup_steps) {
            lr = base_lr * (state->optimizer_step / warmup_steps);
        }
        lr = lr / accum_steps;

        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0.01f;
        float eps = 1e-8f;

        cuda_optimizer_step_gpu_resident(state, lr, beta1, beta2, weight_decay, eps,
                                         state->optimizer_step, ctx);

        state->accumulation_counter = 0;
    }

    // Synchronize to ensure all operations complete
    cuda_train_sync(ctx);

    // Download loss
    float loss_scalar = 0.0f;
    cuda_train_buffer_download(ctx, reduce_in, &loss_scalar, 1);
    loss_scalar = loss_scalar / num_predictions;

    // Timing output (matches Vulkan)
    if ((state->current_step % 20) == 0) {
        double t1_ms = now_ms();
        fprintf(stderr, "[TIMING] step=%d seq=%d cuda_step=%.2fms (gpu=on)\n",
                state->current_step, num_tokens, (t1_ms - t0_ms));
    }

    return loss_scalar;
}

// ═══════════════════════════════════════════════════════════════════════════
// DOWNLOAD WEIGHTS (GPU -> CPU)
// ═══════════════════════════════════════════════════════════════════════════

void cuda_download_full_weights(cuda_train_context_t* ctx, train_state_t* state) {
    if (!ctx || !state || !state->full_weights || !state->config) return;

    trainable_weights_t* weights = state->full_weights;
    const model_config_t* config = state->config;
    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int intermediate_size = config->intermediate_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;

    // Embedding
    cuda_train_buffer_download(ctx, ctx->gpu_w_embed, weights->embed_tokens.weight,
                               (size_t)vocab_size * (size_t)hidden_size);
    cuda_train_buffer_download(ctx, ctx->gpu_m_embed, weights->embed_tokens.m,
                               (size_t)vocab_size * (size_t)hidden_size);
    cuda_train_buffer_download(ctx, ctx->gpu_v_embed, weights->embed_tokens.v,
                               (size_t)vocab_size * (size_t)hidden_size);

    // Final norm
    cuda_train_buffer_download(ctx, ctx->gpu_w_final_norm, weights->final_norm.weight,
                               (size_t)hidden_size);
    cuda_train_buffer_download(ctx, ctx->gpu_m_final_norm, weights->final_norm.m,
                               (size_t)hidden_size);
    cuda_train_buffer_download(ctx, ctx->gpu_v_final_norm, weights->final_norm.v,
                               (size_t)hidden_size);

    // Per-layer
    for (int l = 0; l < num_layers; l++) {
        layer_weights_t* lw = &weights->layers[l];

        cuda_train_buffer_download(ctx, ctx->gpu_w_q[l], lw->q_proj.weight, hidden_size * q_dim);
        cuda_train_buffer_download(ctx, ctx->gpu_m_q[l], lw->q_proj.m, hidden_size * q_dim);
        cuda_train_buffer_download(ctx, ctx->gpu_v_q[l], lw->q_proj.v, hidden_size * q_dim);

        cuda_train_buffer_download(ctx, ctx->gpu_w_k[l], lw->k_proj.weight, hidden_size * kv_dim);
        cuda_train_buffer_download(ctx, ctx->gpu_m_k[l], lw->k_proj.m, hidden_size * kv_dim);
        cuda_train_buffer_download(ctx, ctx->gpu_v_k[l], lw->k_proj.v, hidden_size * kv_dim);

        cuda_train_buffer_download(ctx, ctx->gpu_w_v[l], lw->v_proj.weight, hidden_size * kv_dim);
        cuda_train_buffer_download(ctx, ctx->gpu_m_v[l], lw->v_proj.m, hidden_size * kv_dim);
        cuda_train_buffer_download(ctx, ctx->gpu_v_v[l], lw->v_proj.v, hidden_size * kv_dim);

        cuda_train_buffer_download(ctx, ctx->gpu_w_o[l], lw->o_proj.weight, q_dim * hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_m_o[l], lw->o_proj.m, q_dim * hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_v_o[l], lw->o_proj.v, q_dim * hidden_size);

        cuda_train_buffer_download(ctx, ctx->gpu_w_gate[l], lw->gate_proj.weight, hidden_size * intermediate_size);
        cuda_train_buffer_download(ctx, ctx->gpu_m_gate[l], lw->gate_proj.m, hidden_size * intermediate_size);
        cuda_train_buffer_download(ctx, ctx->gpu_v_gate[l], lw->gate_proj.v, hidden_size * intermediate_size);

        cuda_train_buffer_download(ctx, ctx->gpu_w_up[l], lw->up_proj.weight, hidden_size * intermediate_size);
        cuda_train_buffer_download(ctx, ctx->gpu_m_up[l], lw->up_proj.m, hidden_size * intermediate_size);
        cuda_train_buffer_download(ctx, ctx->gpu_v_up[l], lw->up_proj.v, hidden_size * intermediate_size);

        cuda_train_buffer_download(ctx, ctx->gpu_w_down[l], lw->down_proj.weight, intermediate_size * hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_m_down[l], lw->down_proj.m, intermediate_size * hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_v_down[l], lw->down_proj.v, intermediate_size * hidden_size);

        cuda_train_buffer_download(ctx, ctx->gpu_w_in_norm[l], lw->input_norm.weight, hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_m_in_norm[l], lw->input_norm.m, hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_v_in_norm[l], lw->input_norm.v, hidden_size);

        cuda_train_buffer_download(ctx, ctx->gpu_w_post_norm[l], lw->post_norm.weight, hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_m_post_norm[l], lw->post_norm.m, hidden_size);
        cuda_train_buffer_download(ctx, ctx->gpu_v_post_norm[l], lw->post_norm.v, hidden_size);
    }

    cuda_train_sync(ctx);
}

// Download GPU gradients to CPU (for gradient snapshot visualization)
void cuda_download_gpu_gradients(cuda_train_context_t* ctx, train_state_t* state) {
    if (!ctx || !state || !state->full_weights || !state->config) return;

    trainable_weights_t* w = state->full_weights;
    const model_config_t* config = state->config;
    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int intermediate_size = config->intermediate_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;

    // Embedding gradients
    if (ctx->gpu_g_embed && w->embed_tokens.grad) {
        cuda_train_buffer_download(ctx, ctx->gpu_g_embed, w->embed_tokens.grad,
                                   (size_t)vocab_size * (size_t)hidden_size);
    }

    // Final norm gradients
    if (ctx->gpu_g_final_norm && w->final_norm.grad) {
        cuda_train_buffer_download(ctx, ctx->gpu_g_final_norm, w->final_norm.grad,
                                   (size_t)hidden_size);
    }

    // Per-layer gradients
    for (int l = 0; l < num_layers; l++) {
        layer_weights_t* lw = &w->layers[l];

        if (ctx->gpu_g_q && ctx->gpu_g_q[l] && lw->q_proj.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_q[l], lw->q_proj.grad, hidden_size * q_dim);
        if (ctx->gpu_g_k && ctx->gpu_g_k[l] && lw->k_proj.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_k[l], lw->k_proj.grad, hidden_size * kv_dim);
        if (ctx->gpu_g_v && ctx->gpu_g_v[l] && lw->v_proj.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_v[l], lw->v_proj.grad, hidden_size * kv_dim);
        if (ctx->gpu_g_o && ctx->gpu_g_o[l] && lw->o_proj.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_o[l], lw->o_proj.grad, q_dim * hidden_size);

        if (ctx->gpu_g_gate && ctx->gpu_g_gate[l] && lw->gate_proj.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_gate[l], lw->gate_proj.grad, hidden_size * intermediate_size);
        if (ctx->gpu_g_up && ctx->gpu_g_up[l] && lw->up_proj.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_up[l], lw->up_proj.grad, hidden_size * intermediate_size);
        if (ctx->gpu_g_down && ctx->gpu_g_down[l] && lw->down_proj.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_down[l], lw->down_proj.grad, intermediate_size * hidden_size);

        if (ctx->gpu_g_in_norm && ctx->gpu_g_in_norm[l] && lw->input_norm.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_in_norm[l], lw->input_norm.grad, hidden_size);
        if (ctx->gpu_g_post_norm && ctx->gpu_g_post_norm[l] && lw->post_norm.grad)
            cuda_train_buffer_download(ctx, ctx->gpu_g_post_norm[l], lw->post_norm.grad, hidden_size);
    }

    cuda_train_sync(ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// LORA TRAINING SUPPORT
// ═══════════════════════════════════════════════════════════════════════════

// Upload frozen base model weights (BF16 → FP32) to GPU for LoRA training
// Reuses the gpu_w_* fields (same as full-weight path, but no grad/m/v)
static void ensure_cuda_lora_base_weights(cuda_train_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->model || !state->config || !state->lora) return;
    if (ctx->gpu_w_embed) return;  // Already uploaded

    const model_config_t* config = state->config;
    seraph_model_t* model = state->model;
    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;
    int intermediate_size = config->intermediate_size;

    printf("  Uploading frozen base weights (BF16->FP32) to GPU...\n");

    ctx->gpu_weights_num_layers = num_layers;
    ctx->gpu_weights_hidden_size = hidden_size;
    ctx->gpu_weights_intermediate_size = intermediate_size;
    ctx->gpu_weights_vocab_size = vocab_size;
    ctx->gpu_weights_q_dim = q_dim;
    ctx->gpu_weights_kv_dim = kv_dim;

    // Helper: convert BF16 tensor to FP32 and upload
    #define UPLOAD_BF16(tensor_name, buf_ptr, num_elements) do { \
        uint16_t* bf16 = get_model_tensor(model, tensor_name); \
        if (bf16) { \
            size_t _n = (size_t)(num_elements); \
            size_t _bytes = _n * sizeof(float); \
            buf_ptr = cuda_train_buffer_create(ctx, _bytes); \
            float* _tmp = malloc(_bytes); \
            for (size_t _i = 0; _i < _n; _i++) _tmp[_i] = bf16_to_f32(bf16[_i]); \
            cuda_train_buffer_upload(ctx, (cuda_buffer_t*)buf_ptr, _tmp, _n); \
            free(_tmp); \
        } \
    } while(0)

    // Embeddings + final norm
    UPLOAD_BF16("model.embed_tokens.weight", ctx->gpu_w_embed,
                (size_t)vocab_size * (size_t)hidden_size);
    UPLOAD_BF16("model.norm.weight", ctx->gpu_w_final_norm, hidden_size);

    // Per-layer weight arrays (weight only, no grad/m/v for frozen base)
    ctx->gpu_w_q = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_k = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_v = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_o = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_gate = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_up = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_down = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_in_norm = calloc(num_layers, sizeof(cuda_buffer_t*));
    ctx->gpu_w_post_norm = calloc(num_layers, sizeof(cuda_buffer_t*));

    char tname[256];
    for (int l = 0; l < num_layers; l++) {
        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.q_proj.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_q)[l], (size_t)q_dim * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.k_proj.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_k)[l], (size_t)kv_dim * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.v_proj.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_v)[l], (size_t)kv_dim * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.o_proj.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_o)[l], (size_t)hidden_size * q_dim);

        snprintf(tname, sizeof(tname), "model.layers.%d.mlp.gate_proj.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_gate)[l], (size_t)intermediate_size * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.mlp.up_proj.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_up)[l], (size_t)intermediate_size * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.mlp.down_proj.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_down)[l], (size_t)hidden_size * intermediate_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.input_layernorm.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_in_norm)[l], hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.post_attention_layernorm.weight", l);
        UPLOAD_BF16(tname, ((cuda_buffer_t**)ctx->gpu_w_post_norm)[l], hidden_size);
    }

    #undef UPLOAD_BF16

    cuda_train_sync(ctx);
    printf("  Base weights uploaded to GPU (frozen, %d layers)\n", num_layers);
}

// Helper: allocate + upload one adapter type to GPU
static void cuda_lora_adapter_upload(cuda_train_context_t* ctx, cuda_lora_gpu_adapter_t* gpu,
                                      lora_adapter_t** adapters, int num_layers) {
    if (!adapters) { gpu->active = 0; return; }

    // Check if any adapter in this type is non-NULL
    int any_active = 0;
    for (int l = 0; l < num_layers; l++) {
        if (adapters[l]) { any_active = 1; break; }
    }
    if (!any_active) { gpu->active = 0; return; }

    gpu->active = 1;
    gpu->A  = calloc(num_layers, sizeof(cuda_buffer_t*));
    gpu->B  = calloc(num_layers, sizeof(cuda_buffer_t*));
    gpu->gA = calloc(num_layers, sizeof(cuda_buffer_t*));
    gpu->gB = calloc(num_layers, sizeof(cuda_buffer_t*));
    gpu->mA = calloc(num_layers, sizeof(cuda_buffer_t*));
    gpu->vA = calloc(num_layers, sizeof(cuda_buffer_t*));
    gpu->mB = calloc(num_layers, sizeof(cuda_buffer_t*));
    gpu->vB = calloc(num_layers, sizeof(cuda_buffer_t*));

    for (int l = 0; l < num_layers; l++) {
        lora_adapter_t* a = adapters[l];
        if (!a) continue;

        size_t a_bytes = (size_t)a->rank * (size_t)a->in_dim * sizeof(float);
        size_t b_bytes = (size_t)a->out_dim * (size_t)a->rank * sizeof(float);

        // Weights
        gpu->A[l] = cuda_train_buffer_create(ctx, a_bytes);
        gpu->B[l] = cuda_train_buffer_create(ctx, b_bytes);
        cuda_train_buffer_upload(ctx, gpu->A[l], a->A, a->rank * a->in_dim);
        cuda_train_buffer_upload(ctx, gpu->B[l], a->B, a->out_dim * a->rank);

        // Gradients (zero-initialized)
        gpu->gA[l] = cuda_train_buffer_create(ctx, a_bytes);
        gpu->gB[l] = cuda_train_buffer_create(ctx, b_bytes);
        cuda_train_fill_buffer(ctx, gpu->gA[l], 0, a_bytes);
        cuda_train_fill_buffer(ctx, gpu->gB[l], 0, b_bytes);

        // Optimizer state (upload from CPU - may have nonzero state from resumed training)
        gpu->mA[l] = cuda_train_buffer_create(ctx, a_bytes);
        gpu->vA[l] = cuda_train_buffer_create(ctx, a_bytes);
        gpu->mB[l] = cuda_train_buffer_create(ctx, b_bytes);
        gpu->vB[l] = cuda_train_buffer_create(ctx, b_bytes);
        cuda_train_buffer_upload(ctx, gpu->mA[l], a->m_A, a->rank * a->in_dim);
        cuda_train_buffer_upload(ctx, gpu->vA[l], a->v_A, a->rank * a->in_dim);
        cuda_train_buffer_upload(ctx, gpu->mB[l], a->m_B, a->out_dim * a->rank);
        cuda_train_buffer_upload(ctx, gpu->vB[l], a->v_B, a->out_dim * a->rank);
    }
}

// Upload LoRA adapter parameters to GPU
static void ensure_cuda_lora_adapters(cuda_train_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->lora) return;
    if (ctx->lora_gpu) return;  // Already uploaded

    lora_model_t* lora = state->lora;
    int num_layers = lora->num_layers;

    cuda_lora_gpu_state_t* lg = calloc(1, sizeof(cuda_lora_gpu_state_t));
    lg->num_layers = num_layers;
    lg->rank = lora->rank;
    lg->alpha = lora->alpha;
    lg->scale = lora->alpha / (float)lora->rank;

    cuda_lora_adapter_upload(ctx, &lg->q, lora->q_adapters, num_layers);
    cuda_lora_adapter_upload(ctx, &lg->k, lora->k_adapters, num_layers);
    cuda_lora_adapter_upload(ctx, &lg->v, lora->v_adapters, num_layers);
    cuda_lora_adapter_upload(ctx, &lg->o, lora->o_adapters, num_layers);
    cuda_lora_adapter_upload(ctx, &lg->gate, lora->gate_adapters, num_layers);
    cuda_lora_adapter_upload(ctx, &lg->up, lora->up_adapters, num_layers);
    cuda_lora_adapter_upload(ctx, &lg->down, lora->down_adapters, num_layers);

    ctx->lora_gpu = lg;

    cuda_train_sync(ctx);
    printf("  LoRA adapters uploaded to GPU (rank=%d, alpha=%.1f)\n", lora->rank, lora->alpha);
}

// Ensure LoRA hidden_tmp buffer exists
static void ensure_cuda_lora_hidden_tmp(cuda_train_context_t* ctx, int max_seq, int rank) {
    size_t needed = (size_t)max_seq * (size_t)rank * sizeof(float);
    if (ctx->lora_hidden_tmp && ctx->lora_hidden_tmp_bytes >= needed) return;
    if (ctx->lora_hidden_tmp) cuda_train_buffer_free(ctx, ctx->lora_hidden_tmp);
    ctx->lora_hidden_tmp = cuda_train_buffer_create(ctx, needed);
    ctx->lora_hidden_tmp_bytes = needed;
}

// LoRA optimizer step (AdamW on all adapter parameters)
static void cuda_lora_optimizer_step_with_lora(cuda_train_context_t* ctx, lora_model_t* lora,
                                                float lr, float beta1, float beta2,
                                                float weight_decay, float eps, int step) {
    if (!ctx || !ctx->lora_gpu || !lora) return;

    cuda_lora_gpu_state_t* lg = (cuda_lora_gpu_state_t*)ctx->lora_gpu;
    int num_layers = lg->num_layers;

    // Helper macro
    #define OPT_ADAPTER(gpu_adapter, cpu_adapters, name) do { \
        if ((gpu_adapter).active) { \
            for (int l = 0; l < num_layers; l++) { \
                if (!cpu_adapters || !cpu_adapters[l]) continue; \
                lora_adapter_t* a = cpu_adapters[l]; \
                if ((gpu_adapter).A[l] && (gpu_adapter).gA[l]) { \
                    size_t a_size = (size_t)a->rank * (size_t)a->in_dim; \
                    cuda_train_adamw_update(ctx, (gpu_adapter).A[l], (gpu_adapter).gA[l], \
                                            (gpu_adapter).mA[l], (gpu_adapter).vA[l], \
                                            lr, beta1, beta2, weight_decay, eps, step, a_size, \
                                            name "_A", a->rank, a->in_dim); \
                } \
                if ((gpu_adapter).B[l] && (gpu_adapter).gB[l]) { \
                    size_t b_size = (size_t)a->out_dim * (size_t)a->rank; \
                    cuda_train_adamw_update(ctx, (gpu_adapter).B[l], (gpu_adapter).gB[l], \
                                            (gpu_adapter).mB[l], (gpu_adapter).vB[l], \
                                            lr, beta1, beta2, weight_decay, eps, step, b_size, \
                                            name "_B", a->out_dim, a->rank); \
                } \
            } \
        } \
    } while(0)

    OPT_ADAPTER(lg->q, lora->q_adapters, "lora_q");
    OPT_ADAPTER(lg->k, lora->k_adapters, "lora_k");
    OPT_ADAPTER(lg->v, lora->v_adapters, "lora_v");
    OPT_ADAPTER(lg->o, lora->o_adapters, "lora_o");
    OPT_ADAPTER(lg->gate, lora->gate_adapters, "lora_gate");
    OPT_ADAPTER(lg->up, lora->up_adapters, "lora_up");
    OPT_ADAPTER(lg->down, lora->down_adapters, "lora_down");

    #undef OPT_ADAPTER
}

// Main LoRA training step
float cuda_train_step_gpu_lora(train_state_t* state, int* tokens, int num_tokens,
                                cuda_train_context_t* ctx) {
    if (!state || !state->lora || !state->config || !ctx) return -1.0f;

    double t0_ms = now_ms();

    const model_config_t* config = state->config;
    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int vocab_size = config->vocab_size;
    const int num_layers = config->num_hidden_layers;
    const int num_heads = config->num_attention_heads;
    const int head_dim = config->head_dim;
    const int kv_heads = config->num_key_value_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = kv_heads * head_dim;
    const float rms_eps = 1e-5f;
    const float rope_theta = config->rope_theta > 0 ? config->rope_theta : 10000.0f;

    int max_seq = state->act_cache ? state->act_cache->max_seq_len : num_tokens;
    if (max_seq < num_tokens) max_seq = num_tokens;

    // Ensure all GPU buffers are ready
    ensure_cuda_lora_base_weights(ctx, state);
    ensure_cuda_lora_adapters(ctx, state);
    ensure_cuda_fwd_buffers(ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_train_buffers(ctx, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);

    cuda_lora_gpu_state_t* lg = (cuda_lora_gpu_state_t*)ctx->lora_gpu;
    ensure_cuda_lora_hidden_tmp(ctx, max_seq, lg->rank);
    float lora_scale = lg->scale;

    // Zero LoRA grads at start of accumulation cycle
    if (state->accumulation_counter == 0) {
        cuda_train_zero_lora_grads(ctx);
    }

    // Get buffer pointers (same as full-weight path)
    cuda_buffer_t* buf_hidden = ctx->fwd_hidden_buffer;
    cuda_buffer_t* buf_hidden_norm = ctx->fwd_hidden_norm_buffer;
    cuda_buffer_t* buf_tmp_hidden = ctx->fwd_tmp_hidden_buffer;
    cuda_buffer_t* buf_gate = ctx->fwd_ffn_gate_buffer;
    cuda_buffer_t* buf_up = ctx->fwd_ffn_up_buffer;
    cuda_buffer_t* buf_ffn_hidden = ctx->fwd_ffn_hidden_buffer;
    cuda_buffer_t* buf_logits = ctx->fwd_logits_buffer;

    cuda_buffer_t* buf_Q = ctx->attn_q_buffer;
    cuda_buffer_t* buf_K = ctx->attn_k_buffer;
    cuda_buffer_t* buf_V = ctx->attn_v_buffer;
    cuda_buffer_t* buf_attn = ctx->attn_out_buffer;

    cuda_buffer_t** cache_layer_in = ctx->fwd_cache_layer_input;
    cuda_buffer_t** cache_x_norm_attn = ctx->fwd_cache_x_norm_attn;
    cuda_buffer_t** cache_attn_out = ctx->fwd_cache_attn_out;
    cuda_buffer_t** cache_x_norm_ffn = ctx->fwd_cache_x_norm_ffn;
    cuda_buffer_t** cache_ffn_hidden = ctx->fwd_cache_ffn_hidden;
    cuda_buffer_t** cache_q = ctx->fwd_cache_q;
    cuda_buffer_t** cache_k = ctx->fwd_cache_k;
    cuda_buffer_t** cache_v = ctx->fwd_cache_v;

    // Upload tokens + targets
    uint32_t* tokens_u32 = malloc((size_t)num_tokens * sizeof(uint32_t));
    if (!tokens_u32) return -1.0f;
    for (int i = 0; i < num_tokens; i++) {
        tokens_u32[i] = (tokens[i] < 0) ? 0u : (uint32_t)tokens[i];
    }
    cuda_train_buffer_upload_u32(ctx, ctx->train_tokens_u32, tokens_u32, (size_t)num_tokens);

    int num_predictions = num_tokens - 1;
    uint32_t* targets_u32 = malloc((size_t)num_predictions * sizeof(uint32_t));
    if (!targets_u32) {
        free(tokens_u32);
        return -1.0f;
    }
    for (int t = 0; t < num_predictions; t++) {
        int tok = tokens[t + 1];
        targets_u32[t] = (tok < 0) ? 0u : (uint32_t)tok;
    }
    cuda_train_buffer_upload_u32(ctx, ctx->train_targets_u32, targets_u32, (size_t)num_predictions);
    free(tokens_u32);
    free(targets_u32);

    // ═══════════════════════════════════════════════════════════════════════
    // FORWARD PASS (with LoRA adapters)
    // ═══════════════════════════════════════════════════════════════════════

    // Embedding lookup -> hidden
    cuda_train_embed_lookup(ctx, ctx->train_tokens_u32, ctx->gpu_w_embed, buf_hidden,
                            num_tokens, hidden_size, vocab_size);

    // Transformer layers
    for (int layer = 0; layer < num_layers; layer++) {
        // Cache layer input
        cuda_train_copy_buffer(ctx, buf_hidden, cache_layer_in[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // RMSNorm (pre-attention)
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_in_norm[layer], buf_hidden_norm,
                           num_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_attn[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Q, K, V projections (base weights)
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_q[layer], buf_Q,
                                    num_tokens, q_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_k[layer], buf_K,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_v[layer], buf_V,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);

        // Add LoRA contributions: output += scale * (X @ A^T @ B^T) = scale * X @ (BA)^T
        // For Q: delta = x @ A^T, then delta @ B^T
        if (lg->q.active && lg->q.A[layer]) {
            // hidden_tmp = x_norm @ A^T
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->q.A[layer], ctx->lora_hidden_tmp,
                                        num_tokens, lg->rank, hidden_size, 0, 1, 0);
            // Q += scale * hidden_tmp @ B^T
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)num_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->q.B[layer], buf_Q,
                                        num_tokens, q_dim, lg->rank, 0, 1, 1);
        }
        if (lg->k.active && lg->k.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->k.A[layer], ctx->lora_hidden_tmp,
                                        num_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)num_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->k.B[layer], buf_K,
                                        num_tokens, kv_dim, lg->rank, 0, 1, 1);
        }
        if (lg->v.active && lg->v.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->v.A[layer], ctx->lora_hidden_tmp,
                                        num_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)num_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->v.B[layer], buf_V,
                                        num_tokens, kv_dim, lg->rank, 0, 1, 1);
        }

        // RoPE
        cuda_train_rope(ctx, buf_Q, buf_Q, num_tokens, num_heads, head_dim, rope_theta);
        cuda_train_rope(ctx, buf_K, buf_K, num_tokens, kv_heads, head_dim, rope_theta);

        // Cache Q/K/V for backward
        cuda_train_copy_buffer(ctx, buf_Q, cache_q[layer], (size_t)num_tokens * (size_t)q_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_K, cache_k[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_V, cache_v[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));

        // Attention (FlashAttention: stores logsumexp per layer for backward)
        cuda_train_batch_attention(ctx, buf_Q, buf_K, buf_V, buf_attn,
                                   ctx->fwd_cache_attn_lse[layer],
                                   num_tokens, num_heads, kv_heads, head_dim);
        cuda_train_copy_buffer(ctx, buf_attn, cache_attn_out[layer],
                               (size_t)num_tokens * (size_t)q_dim * sizeof(float));

        // O projection + LoRA + residual
        cuda_train_matmul_transpose(ctx, buf_attn, ctx->gpu_w_o[layer], buf_tmp_hidden,
                                    num_tokens, hidden_size, q_dim, 0, 1, 0);
        if (lg->o.active && lg->o.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_attn, lg->o.A[layer], ctx->lora_hidden_tmp,
                                        num_tokens, lg->rank, q_dim, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)num_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->o.B[layer], buf_tmp_hidden,
                                        num_tokens, hidden_size, lg->rank, 0, 1, 1);
        }
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);

        // RMSNorm (post-attention)
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_post_norm[layer], buf_hidden_norm,
                           num_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_ffn[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Gate + up projections + LoRA
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_gate[layer], buf_gate,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        if (lg->gate.active && lg->gate.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->gate.A[layer], ctx->lora_hidden_tmp,
                                        num_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)num_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->gate.B[layer], buf_gate,
                                        num_tokens, intermediate_size, lg->rank, 0, 1, 1);
        }

        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_up[layer], buf_up,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        if (lg->up.active && lg->up.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->up.A[layer], ctx->lora_hidden_tmp,
                                        num_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)num_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->up.B[layer], buf_up,
                                        num_tokens, intermediate_size, lg->rank, 0, 1, 1);
        }

        // SiLU + multiply
        cuda_train_silu(ctx, buf_gate, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
        cuda_train_mul(ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden,
                       (size_t)num_tokens * (size_t)intermediate_size);
        cuda_train_copy_buffer(ctx, buf_ffn_hidden, cache_ffn_hidden[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        // Down projection + LoRA + residual
        cuda_train_matmul_transpose(ctx, buf_ffn_hidden, ctx->gpu_w_down[layer], buf_tmp_hidden,
                                    num_tokens, hidden_size, intermediate_size, 0, 1, 0);
        if (lg->down.active && lg->down.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_ffn_hidden, lg->down.A[layer], ctx->lora_hidden_tmp,
                                        num_tokens, lg->rank, intermediate_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)num_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->down.B[layer], buf_tmp_hidden,
                                        num_tokens, hidden_size, lg->rank, 0, 1, 1);
        }
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);
    }

    // Final RMSNorm
    cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_final_norm, buf_hidden_norm,
                       num_tokens, hidden_size, rms_eps);

    // LM head (tied embeddings): logits = hidden_norm @ embed^T
    cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_embed, buf_logits,
                                num_tokens, vocab_size, hidden_size, 0, 1, 0);

    // ═══════════════════════════════════════════════════════════════════════
    // LOSS + BACKWARD (LoRA adapters only)
    // ═══════════════════════════════════════════════════════════════════════

    // Cross-entropy loss + gradient
    cuda_train_cross_entropy_grad(ctx, buf_logits, ctx->train_targets_u32,
                                  ctx->train_grad_logits, ctx->train_loss_rows,
                                  num_predictions, vocab_size);

    // Reduce sum loss
    size_t reduce_count = (size_t)num_predictions;
    cuda_buffer_t* reduce_in = ctx->train_loss_rows;
    cuda_buffer_t* reduce_out = ctx->train_reduce_tmp_a;
    while (reduce_count > 1) {
        size_t reduce_block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
        size_t out_count = (reduce_count + reduce_block - 1) / reduce_block;
        cuda_train_reduce_sum(ctx, reduce_in, reduce_out, reduce_count);
        reduce_count = out_count;
        cuda_buffer_t* tmp = reduce_in;
        reduce_in = reduce_out;
        reduce_out = tmp;
    }

    // LM head backward: grad_hidden_norm = grad_logits @ embed_weight
    cuda_train_matmul(ctx, ctx->train_grad_logits, ctx->gpu_w_embed,
                      ctx->train_grad_hidden, num_tokens, hidden_size, vocab_size);

    // Layer backward loop (LoRA adapters only)
    for (int layer = num_layers - 1; layer >= 0; layer--) {
        // Down LoRA backward
        if (lg->down.active && lg->down.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_ffn_hidden[layer],
                                     lg->down.A[layer], lg->down.B[layer],
                                     lg->down.gA[layer], lg->down.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     num_tokens, intermediate_size, hidden_size, lg->rank, lora_scale);
        }

        // Gate/up LoRA backward (using cached x_norm_ffn input)
        // For simplicity, approximate gradients from grad_hidden
        if (lg->gate.active && lg->gate.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_ffn[layer],
                                     lg->gate.A[layer], lg->gate.B[layer],
                                     lg->gate.gA[layer], lg->gate.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     num_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }
        if (lg->up.active && lg->up.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_ffn[layer],
                                     lg->up.A[layer], lg->up.B[layer],
                                     lg->up.gA[layer], lg->up.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     num_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }

        // O LoRA backward
        if (lg->o.active && lg->o.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_attn_out[layer],
                                     lg->o.A[layer], lg->o.B[layer],
                                     lg->o.gA[layer], lg->o.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     num_tokens, q_dim, hidden_size, lg->rank, lora_scale);
        }

        // Q/K/V LoRA backward (from grad_hidden through attention)
        if (lg->q.active && lg->q.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_attn[layer],
                                     lg->q.A[layer], lg->q.B[layer],
                                     lg->q.gA[layer], lg->q.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     num_tokens, hidden_size, q_dim, lg->rank, lora_scale);
        }
        if (lg->k.active && lg->k.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_attn[layer],
                                     lg->k.A[layer], lg->k.B[layer],
                                     lg->k.gA[layer], lg->k.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     num_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }
        if (lg->v.active && lg->v.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_attn[layer],
                                     lg->v.A[layer], lg->v.B[layer],
                                     lg->v.gA[layer], lg->v.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     num_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIMIZER STEP
    // ═══════════════════════════════════════════════════════════════════════

    state->accumulation_counter++;
    int accum_steps = state->gradient_accumulation_steps;
    if (accum_steps < 1) accum_steps = 1;

    int do_opt = (state->accumulation_counter >= accum_steps);

    if (do_opt) {
        state->optimizer_step++;

        float base_lr = state->learning_rate;
        float warmup_steps = 100.0f;
        float lr = base_lr;
        if (state->optimizer_step < warmup_steps) {
            lr = base_lr * (state->optimizer_step / warmup_steps);
        }
        lr = lr / accum_steps;

        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0.01f;
        float eps = 1e-8f;

        cuda_lora_optimizer_step_with_lora(ctx, state->lora, lr, beta1, beta2,
                                           weight_decay, eps, state->optimizer_step);

        state->accumulation_counter = 0;
    }

    // Synchronize
    cuda_train_sync(ctx);

    // Download loss
    float loss_scalar = 0.0f;
    cuda_train_buffer_download(ctx, reduce_in, &loss_scalar, 1);
    loss_scalar = loss_scalar / num_predictions;

    // Timing output (matches Vulkan)
    if ((state->current_step % 20) == 0) {
        double t1_ms = now_ms();
        fprintf(stderr, "[TIMING] step=%d seq=%d cuda_lora_step=%.2fms (gpu=on)\n",
                state->current_step, num_tokens, (t1_ms - t0_ms));
    }

    return loss_scalar;
}

// ═══════════════════════════════════════════════════════════════════════════
// BATCHED TRAINING STEP (GPU Full Training, batch > 1)
// ═══════════════════════════════════════════════════════════════════════════
// tokens_flat: [batch_size * padded_seq_len] padded with 0 (pad_token_id)
// seq_lens: [batch_size] actual length of each sequence
// All row-parallel ops use total_tokens = batch_size * padded_seq_len as M dimension.
// Attention and RoPE use batched variants with per-sequence positions/causal masks.

float cuda_train_step_gpu_full_batch(train_state_t* state,
                                      int* tokens_packed, int total_tokens,
                                      int* seq_starts, int batch_size,
                                      cuda_train_context_t* ctx) {
    if (!state || !state->full_weights || !state->config || !ctx) return -1.0f;

    double t0_ms = now_ms();

    const model_config_t* config = state->config;
    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int vocab_size = config->vocab_size;
    const int num_layers = config->num_hidden_layers;
    const int num_heads = config->num_attention_heads;
    const int head_dim = config->head_dim;
    const int kv_heads = config->num_key_value_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = kv_heads * head_dim;
    const float rms_eps = 1e-5f;
    const float rope_theta = config->rope_theta > 0 ? config->rope_theta : 10000.0f;

    int max_seq = total_tokens;

    // num_valid_predictions = total_tokens - batch_size (one fewer per sequence)
    int num_valid_predictions = total_tokens - batch_size;
    if (num_valid_predictions <= 0) return 0.0f;

    // Compute max_seq_in_batch from seq_starts
    int max_seq_in_batch = 0;
    for (int b = 0; b < batch_size; b++) {
        int slen = seq_starts[b + 1] - seq_starts[b];
        if (slen > max_seq_in_batch) max_seq_in_batch = slen;
    }

    ensure_cuda_fwd_buffers(ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_train_buffers(ctx, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_gpu_weights(ctx, state);

    // Allocate seq_starts buffer
    size_t starts_bytes = (size_t)(batch_size + 1) * sizeof(uint32_t);
    if (!ctx->train_seq_starts || ctx->train_seq_starts->size < starts_bytes) {
        if (ctx->train_seq_starts) cuda_train_buffer_free(ctx, ctx->train_seq_starts);
        ctx->train_seq_starts = cuda_train_buffer_create(ctx, starts_bytes);
    }

    if (state->accumulation_counter == 0) {
        cuda_zero_full_weight_grads(ctx, state);
    }

    // Build tokens and targets on CPU (packed — no padding)
    uint32_t* tokens_u32 = malloc((size_t)total_tokens * sizeof(uint32_t));
    uint32_t* targets_u32 = malloc((size_t)total_tokens * sizeof(uint32_t));
    uint32_t* seq_starts_u32 = malloc((size_t)(batch_size + 1) * sizeof(uint32_t));
    if (!tokens_u32 || !targets_u32 || !seq_starts_u32) {
        free(tokens_u32); free(targets_u32); free(seq_starts_u32);
        return -1.0f;
    }

    // Convert seq_starts to uint32
    for (int b = 0; b <= batch_size; b++) {
        seq_starts_u32[b] = (uint32_t)seq_starts[b];
    }

    // Build tokens and targets using seq_starts boundaries
    for (int b = 0; b < batch_size; b++) {
        int start = seq_starts[b];
        int end = seq_starts[b + 1];
        for (int t = start; t < end; t++) {
            int tok = tokens_packed[t];
            tokens_u32[t] = (tok < 0) ? 0u : (uint32_t)tok;
            if (t < end - 1) {
                int next_tok = tokens_packed[t + 1];
                targets_u32[t] = (next_tok < 0) ? 0u : (uint32_t)next_tok;
            } else {
                targets_u32[t] = 0;  // last token: no target
            }
        }
    }

    cuda_train_buffer_upload_u32(ctx, ctx->train_tokens_u32, tokens_u32, (size_t)total_tokens);
    cuda_train_buffer_upload_u32(ctx, ctx->train_targets_u32, targets_u32, (size_t)total_tokens);
    cuda_train_buffer_upload_u32(ctx, ctx->train_seq_starts, seq_starts_u32, (size_t)(batch_size + 1));
    free(tokens_u32);
    free(targets_u32);
    free(seq_starts_u32);

    cuda_buffer_t* buf_hidden = ctx->fwd_hidden_buffer;
    cuda_buffer_t* buf_hidden_norm = ctx->fwd_hidden_norm_buffer;
    cuda_buffer_t* buf_tmp_hidden = ctx->fwd_tmp_hidden_buffer;
    cuda_buffer_t* buf_gate = ctx->fwd_ffn_gate_buffer;
    cuda_buffer_t* buf_up = ctx->fwd_ffn_up_buffer;
    cuda_buffer_t* buf_ffn_hidden = ctx->fwd_ffn_hidden_buffer;
    cuda_buffer_t* buf_logits = ctx->fwd_logits_buffer;
    cuda_buffer_t* buf_Q = ctx->attn_q_buffer;
    cuda_buffer_t* buf_K = ctx->attn_k_buffer;
    cuda_buffer_t* buf_V = ctx->attn_v_buffer;
    cuda_buffer_t* buf_attn = ctx->attn_out_buffer;

    cuda_buffer_t** cache_layer_in = ctx->fwd_cache_layer_input;
    cuda_buffer_t** cache_x_norm_attn = ctx->fwd_cache_x_norm_attn;
    cuda_buffer_t** cache_attn_out = ctx->fwd_cache_attn_out;
    cuda_buffer_t** cache_post_attn_in = ctx->fwd_cache_post_attn_in;
    cuda_buffer_t** cache_x_norm_ffn = ctx->fwd_cache_x_norm_ffn;
    cuda_buffer_t** cache_ffn_gate_out = ctx->fwd_cache_ffn_gate_out;
    cuda_buffer_t** cache_ffn_up_out = ctx->fwd_cache_ffn_up_out;
    cuda_buffer_t** cache_silu_out = ctx->fwd_cache_silu_out;
    cuda_buffer_t** cache_ffn_hidden = ctx->fwd_cache_ffn_hidden;
    cuda_buffer_t** cache_q = ctx->fwd_cache_q;
    cuda_buffer_t** cache_k = ctx->fwd_cache_k;
    cuda_buffer_t** cache_v = ctx->fwd_cache_v;

    // ═══════════════════════════════════════════════════════════════════════
    // FORWARD PASS (packed — row-parallel ops use total_tokens as M)
    // ═══════════════════════════════════════════════════════════════════════

    cuda_train_embed_lookup(ctx, ctx->train_tokens_u32, ctx->gpu_w_embed, buf_hidden,
                            total_tokens, hidden_size, vocab_size);

    for (int layer = 0; layer < num_layers; layer++) {
        cuda_train_copy_buffer(ctx, buf_hidden, cache_layer_in[layer],
                               (size_t)total_tokens * (size_t)hidden_size * sizeof(float));
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_in_norm[layer], buf_hidden_norm,
                           total_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_attn[layer],
                               (size_t)total_tokens * (size_t)hidden_size * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_q[layer], buf_Q,
                                    total_tokens, q_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_k[layer], buf_K,
                                    total_tokens, kv_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_v[layer], buf_V,
                                    total_tokens, kv_dim, hidden_size, 0, 1, 0);

        // RoPE — PACKED
        cuda_train_rope_packed(ctx, buf_Q, buf_Q, ctx->train_seq_starts, total_tokens,
                               batch_size, num_heads, head_dim, rope_theta);
        cuda_train_rope_packed(ctx, buf_K, buf_K, ctx->train_seq_starts, total_tokens,
                               batch_size, kv_heads, head_dim, rope_theta);

        cuda_train_copy_buffer(ctx, buf_Q, cache_q[layer], (size_t)total_tokens * (size_t)q_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_K, cache_k[layer], (size_t)total_tokens * (size_t)kv_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_V, cache_v[layer], (size_t)total_tokens * (size_t)kv_dim * sizeof(float));

        // Attention — PACKED
        cuda_train_batch_attention_packed(ctx, buf_Q, buf_K, buf_V, buf_attn,
                                          ctx->fwd_cache_attn_lse[layer],
                                          ctx->train_seq_starts, total_tokens, batch_size,
                                          num_heads, kv_heads, head_dim, max_seq_in_batch);
        cuda_train_copy_buffer(ctx, buf_attn, cache_attn_out[layer],
                               (size_t)total_tokens * (size_t)q_dim * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_attn, ctx->gpu_w_o[layer], buf_tmp_hidden,
                                    total_tokens, hidden_size, q_dim, 0, 1, 0);
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)total_tokens * (size_t)hidden_size);
        cuda_train_copy_buffer(ctx, buf_hidden, cache_post_attn_in[layer],
                               (size_t)total_tokens * (size_t)hidden_size * sizeof(float));

        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_post_norm[layer], buf_hidden_norm,
                           total_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_ffn[layer],
                               (size_t)total_tokens * (size_t)hidden_size * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_gate[layer], buf_gate,
                                    total_tokens, intermediate_size, hidden_size, 0, 1, 0);
        cuda_train_copy_buffer(ctx, buf_gate, cache_ffn_gate_out[layer],
                               (size_t)total_tokens * (size_t)intermediate_size * sizeof(float));
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_up[layer], buf_up,
                                    total_tokens, intermediate_size, hidden_size, 0, 1, 0);
        cuda_train_copy_buffer(ctx, buf_up, cache_ffn_up_out[layer],
                               (size_t)total_tokens * (size_t)intermediate_size * sizeof(float));

        cuda_train_silu(ctx, buf_gate, buf_ffn_hidden, (size_t)total_tokens * (size_t)intermediate_size);
        cuda_train_copy_buffer(ctx, buf_ffn_hidden, cache_silu_out[layer],
                               (size_t)total_tokens * (size_t)intermediate_size * sizeof(float));
        cuda_train_mul(ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden,
                       (size_t)total_tokens * (size_t)intermediate_size);
        cuda_train_copy_buffer(ctx, buf_ffn_hidden, cache_ffn_hidden[layer],
                               (size_t)total_tokens * (size_t)intermediate_size * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_ffn_hidden, ctx->gpu_w_down[layer], buf_tmp_hidden,
                                    total_tokens, hidden_size, intermediate_size, 0, 1, 0);
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)total_tokens * (size_t)hidden_size);
    }

    cuda_train_copy_buffer(ctx, buf_hidden, ctx->train_final_input,
                           (size_t)total_tokens * (size_t)hidden_size * sizeof(float));
    cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_final_norm, buf_hidden_norm,
                       total_tokens, hidden_size, rms_eps);
    cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_embed, buf_logits,
                                total_tokens, vocab_size, hidden_size, 0, 1, 0);

    // ═══════════════════════════════════════════════════════════════════════
    // LOSS + BACKWARD (packed)
    // ═══════════════════════════════════════════════════════════════════════

    cuda_train_cross_entropy_grad_packed(ctx, buf_logits, ctx->train_targets_u32,
                                          ctx->train_grad_logits, ctx->train_loss_rows,
                                          ctx->train_seq_starts,
                                          total_tokens, batch_size, vocab_size, num_valid_predictions);

    size_t reduce_count = (size_t)total_tokens;
    cuda_buffer_t* reduce_in = ctx->train_loss_rows;
    cuda_buffer_t* reduce_out = ctx->train_reduce_tmp_a;
    while (reduce_count > 1) {
        size_t reduce_block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
        size_t out_count = (reduce_count + reduce_block - 1) / reduce_block;
        cuda_train_reduce_sum(ctx, reduce_in, reduce_out, reduce_count);
        reduce_count = out_count;
        cuda_buffer_t* tmp = reduce_in;
        reduce_in = reduce_out;
        reduce_out = tmp;
    }

    cuda_train_matmul(ctx, ctx->train_grad_logits, ctx->gpu_w_embed,
                      ctx->train_grad_hidden, total_tokens, hidden_size, vocab_size);
    cuda_train_matmul_transpose(ctx, ctx->train_grad_logits, buf_hidden_norm,
                                ctx->gpu_g_embed, vocab_size, hidden_size, total_tokens, 1, 0, 1);
    cuda_train_rmsnorm_backward_batch(ctx, ctx->train_final_input, ctx->gpu_w_final_norm,
                                      ctx->train_grad_hidden, ctx->train_grad_tmp_hidden,
                                      ctx->gpu_g_final_norm, total_tokens, hidden_size, rms_eps);
    cuda_train_copy_buffer(ctx, ctx->train_grad_tmp_hidden, ctx->train_grad_hidden,
                           (size_t)total_tokens * (size_t)hidden_size * sizeof(float));

    for (int layer = num_layers - 1; layer >= 0; layer--) {
        cuda_train_matmul_transpose(ctx, ctx->train_grad_hidden, cache_ffn_hidden[layer],
                                    ctx->gpu_g_down[layer], hidden_size, intermediate_size,
                                    total_tokens, 1, 0, 1);
        cuda_train_matmul(ctx, ctx->train_grad_hidden, ctx->gpu_w_down[layer],
                          ctx->train_grad_ffn, total_tokens, intermediate_size, hidden_size);
        cuda_train_elementwise_mul_backward(ctx, cache_silu_out[layer], cache_ffn_up_out[layer],
                                            ctx->train_grad_ffn, ctx->train_grad_gate,
                                            ctx->train_grad_up,
                                            (size_t)total_tokens * (size_t)intermediate_size);
        cuda_train_silu_backward(ctx, cache_ffn_gate_out[layer], ctx->train_grad_gate,
                                 ctx->train_grad_gate,
                                 (size_t)total_tokens * (size_t)intermediate_size);

        cuda_train_matmul_transpose(ctx, ctx->train_grad_gate, cache_x_norm_ffn[layer],
                                    ctx->gpu_g_gate[layer], intermediate_size, hidden_size,
                                    total_tokens, 1, 0, 1);
        cuda_train_matmul_transpose(ctx, ctx->train_grad_up, cache_x_norm_ffn[layer],
                                    ctx->gpu_g_up[layer], intermediate_size, hidden_size,
                                    total_tokens, 1, 0, 1);

        cuda_train_matmul(ctx, ctx->train_grad_gate, ctx->gpu_w_gate[layer],
                          ctx->train_grad_x_norm, total_tokens, hidden_size, intermediate_size);
        cuda_train_matmul(ctx, ctx->train_grad_up, ctx->gpu_w_up[layer],
                          ctx->train_grad_tmp_hidden, total_tokens, hidden_size, intermediate_size);
        cuda_train_add(ctx, ctx->train_grad_x_norm, ctx->train_grad_tmp_hidden,
                       ctx->train_grad_x_norm, (size_t)total_tokens * (size_t)hidden_size);

        cuda_train_rmsnorm_backward_batch(ctx, cache_post_attn_in[layer],
                                          ctx->gpu_w_post_norm[layer],
                                          ctx->train_grad_x_norm,
                                          ctx->train_grad_tmp_hidden,
                                          ctx->gpu_g_post_norm[layer],
                                          total_tokens, hidden_size, rms_eps);
        cuda_train_add(ctx, ctx->train_grad_tmp_hidden, ctx->train_grad_hidden,
                       ctx->train_grad_hidden, (size_t)total_tokens * (size_t)hidden_size);

        cuda_train_matmul_transpose(ctx, ctx->train_grad_hidden, cache_attn_out[layer],
                                    ctx->gpu_g_o[layer], hidden_size, q_dim,
                                    total_tokens, 1, 0, 1);
        cuda_train_matmul(ctx, ctx->train_grad_hidden, ctx->gpu_w_o[layer],
                          ctx->train_grad_attn, total_tokens, q_dim, hidden_size);

        cuda_train_fill_buffer(ctx, ctx->train_grad_k, 0,
                               (size_t)total_tokens * (size_t)kv_dim * sizeof(float));
        cuda_train_fill_buffer(ctx, ctx->train_grad_v, 0,
                               (size_t)total_tokens * (size_t)kv_dim * sizeof(float));

        // Attention backward — PACKED
        cuda_train_batch_attention_backward_packed(ctx, cache_q[layer], cache_k[layer], cache_v[layer],
                                                    cache_attn_out[layer], ctx->fwd_cache_attn_lse[layer],
                                                    ctx->train_grad_attn, ctx->train_grad_q,
                                                    ctx->train_grad_k, ctx->train_grad_v,
                                                    ctx->train_seq_starts, total_tokens, batch_size,
                                                    num_heads, kv_heads, head_dim, max_seq_in_batch);

        // RoPE backward — PACKED
        cuda_train_rope_backward_packed(ctx, ctx->train_grad_q, ctx->train_grad_q,
                                         ctx->train_seq_starts, total_tokens, batch_size,
                                         num_heads, head_dim, rope_theta);
        cuda_train_rope_backward_packed(ctx, ctx->train_grad_k, ctx->train_grad_k,
                                         ctx->train_seq_starts, total_tokens, batch_size,
                                         kv_heads, head_dim, rope_theta);

        cuda_train_matmul_transpose(ctx, ctx->train_grad_q, cache_x_norm_attn[layer],
                                    ctx->gpu_g_q[layer], q_dim, hidden_size,
                                    total_tokens, 1, 0, 1);
        cuda_train_matmul_transpose(ctx, ctx->train_grad_k, cache_x_norm_attn[layer],
                                    ctx->gpu_g_k[layer], kv_dim, hidden_size,
                                    total_tokens, 1, 0, 1);
        cuda_train_matmul_transpose(ctx, ctx->train_grad_v, cache_x_norm_attn[layer],
                                    ctx->gpu_g_v[layer], kv_dim, hidden_size,
                                    total_tokens, 1, 0, 1);

        cuda_train_matmul(ctx, ctx->train_grad_q, ctx->gpu_w_q[layer],
                          ctx->train_grad_x_norm, total_tokens, hidden_size, q_dim);
        cuda_train_matmul(ctx, ctx->train_grad_k, ctx->gpu_w_k[layer],
                          ctx->train_grad_tmp_hidden, total_tokens, hidden_size, kv_dim);
        cuda_train_add(ctx, ctx->train_grad_x_norm, ctx->train_grad_tmp_hidden,
                       ctx->train_grad_x_norm, (size_t)total_tokens * (size_t)hidden_size);
        cuda_train_matmul(ctx, ctx->train_grad_v, ctx->gpu_w_v[layer],
                          ctx->train_grad_tmp_hidden, total_tokens, hidden_size, kv_dim);
        cuda_train_add(ctx, ctx->train_grad_x_norm, ctx->train_grad_tmp_hidden,
                       ctx->train_grad_x_norm, (size_t)total_tokens * (size_t)hidden_size);

        cuda_train_rmsnorm_backward_batch(ctx, cache_layer_in[layer],
                                          ctx->gpu_w_in_norm[layer],
                                          ctx->train_grad_x_norm,
                                          ctx->train_grad_tmp_hidden,
                                          ctx->gpu_g_in_norm[layer],
                                          total_tokens, hidden_size, rms_eps);
        cuda_train_add(ctx, ctx->train_grad_tmp_hidden, ctx->train_grad_hidden,
                       ctx->train_grad_hidden, (size_t)total_tokens * (size_t)hidden_size);
    }

    cuda_train_embedding_backward(ctx, ctx->train_tokens_u32, ctx->train_grad_hidden,
                                  ctx->gpu_g_embed, total_tokens, hidden_size, vocab_size);

    // Optimizer
    cuda_train_sync(ctx);

    state->accumulation_counter++;
    int accum_steps = state->gradient_accumulation_steps;
    if (accum_steps < 1) accum_steps = 1;

    if (state->accumulation_counter >= accum_steps) {
        state->optimizer_step++;
        float base_lr = state->learning_rate;
        float warmup_steps = 100.0f;
        float lr = base_lr;
        if (state->optimizer_step < warmup_steps)
            lr = base_lr * (state->optimizer_step / warmup_steps);
        lr = lr / accum_steps;
        cuda_optimizer_step_gpu_resident(state, lr, 0.9f, 0.999f, 0.01f, 1e-8f,
                                         state->optimizer_step, ctx);
        state->accumulation_counter = 0;
    }

    cuda_train_sync(ctx);

    float loss_scalar = 0.0f;
    cuda_train_buffer_download(ctx, reduce_in, &loss_scalar, 1);
    loss_scalar = loss_scalar / num_valid_predictions;

    if ((state->current_step % 20) == 0) {
        double t1_ms = now_ms();
        if (batch_size > 1)
            fprintf(stderr, "[TIMING] step=%d batch=%dx%d cuda_step=%.2fms (gpu=on)\n",
                    state->current_step, batch_size, total_tokens, (t1_ms - t0_ms));
        else
            fprintf(stderr, "[TIMING] step=%d seq=%d cuda_step=%.2fms (gpu=on)\n",
                    state->current_step, total_tokens, (t1_ms - t0_ms));
    }

    return loss_scalar;
}

// ═══════════════════════════════════════════════════════════════════════════
// BATCHED LORA TRAINING STEP (batch > 1)
// ═══════════════════════════════════════════════════════════════════════════

float cuda_train_step_gpu_lora_batch(train_state_t* state,
                                      int* tokens_packed, int total_tokens,
                                      int* seq_starts, int batch_size,
                                      cuda_train_context_t* ctx) {
    if (!state || !state->lora || !state->config || !ctx) return -1.0f;

    double t0_ms = now_ms();

    const model_config_t* config = state->config;
    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int vocab_size = config->vocab_size;
    const int num_layers = config->num_hidden_layers;
    const int num_heads = config->num_attention_heads;
    const int head_dim = config->head_dim;
    const int kv_heads = config->num_key_value_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = kv_heads * head_dim;
    const float rms_eps = 1e-5f;
    const float rope_theta = config->rope_theta > 0 ? config->rope_theta : 10000.0f;

    int max_seq = total_tokens;

    // num_valid_predictions = total_tokens - batch_size
    int num_valid_predictions = total_tokens - batch_size;
    if (num_valid_predictions <= 0) return 0.0f;

    // Compute max_seq_in_batch from seq_starts
    int max_seq_in_batch = 0;
    for (int b = 0; b < batch_size; b++) {
        int slen = seq_starts[b + 1] - seq_starts[b];
        if (slen > max_seq_in_batch) max_seq_in_batch = slen;
    }

    ensure_cuda_lora_base_weights(ctx, state);
    ensure_cuda_lora_adapters(ctx, state);
    ensure_cuda_fwd_buffers(ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_train_buffers(ctx, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);

    cuda_lora_gpu_state_t* lg = (cuda_lora_gpu_state_t*)ctx->lora_gpu;
    ensure_cuda_lora_hidden_tmp(ctx, max_seq, lg->rank);
    float lora_scale = lg->scale;

    // Allocate seq_starts buffer
    size_t starts_bytes = (size_t)(batch_size + 1) * sizeof(uint32_t);
    if (!ctx->train_seq_starts || ctx->train_seq_starts->size < starts_bytes) {
        if (ctx->train_seq_starts) cuda_train_buffer_free(ctx, ctx->train_seq_starts);
        ctx->train_seq_starts = cuda_train_buffer_create(ctx, starts_bytes);
    }

    if (state->accumulation_counter == 0) {
        cuda_train_zero_lora_grads(ctx);
    }

    // Build tokens and targets (packed)
    uint32_t* tokens_u32 = malloc((size_t)total_tokens * sizeof(uint32_t));
    uint32_t* targets_u32 = malloc((size_t)total_tokens * sizeof(uint32_t));
    uint32_t* seq_starts_u32 = malloc((size_t)(batch_size + 1) * sizeof(uint32_t));
    if (!tokens_u32 || !targets_u32 || !seq_starts_u32) {
        free(tokens_u32); free(targets_u32); free(seq_starts_u32);
        return -1.0f;
    }

    for (int b = 0; b <= batch_size; b++) {
        seq_starts_u32[b] = (uint32_t)seq_starts[b];
    }

    for (int b = 0; b < batch_size; b++) {
        int start = seq_starts[b];
        int end = seq_starts[b + 1];
        for (int t = start; t < end; t++) {
            int tok = tokens_packed[t];
            tokens_u32[t] = (tok < 0) ? 0u : (uint32_t)tok;
            if (t < end - 1) {
                int next_tok = tokens_packed[t + 1];
                targets_u32[t] = (next_tok < 0) ? 0u : (uint32_t)next_tok;
            } else {
                targets_u32[t] = 0;
            }
        }
    }

    cuda_train_buffer_upload_u32(ctx, ctx->train_tokens_u32, tokens_u32, (size_t)total_tokens);
    cuda_train_buffer_upload_u32(ctx, ctx->train_targets_u32, targets_u32, (size_t)total_tokens);
    cuda_train_buffer_upload_u32(ctx, ctx->train_seq_starts, seq_starts_u32, (size_t)(batch_size + 1));
    free(tokens_u32);
    free(targets_u32);
    free(seq_starts_u32);

    cuda_buffer_t* buf_hidden = ctx->fwd_hidden_buffer;
    cuda_buffer_t* buf_hidden_norm = ctx->fwd_hidden_norm_buffer;
    cuda_buffer_t* buf_tmp_hidden = ctx->fwd_tmp_hidden_buffer;
    cuda_buffer_t* buf_gate = ctx->fwd_ffn_gate_buffer;
    cuda_buffer_t* buf_up = ctx->fwd_ffn_up_buffer;
    cuda_buffer_t* buf_ffn_hidden = ctx->fwd_ffn_hidden_buffer;
    cuda_buffer_t* buf_logits = ctx->fwd_logits_buffer;
    cuda_buffer_t* buf_Q = ctx->attn_q_buffer;
    cuda_buffer_t* buf_K = ctx->attn_k_buffer;
    cuda_buffer_t* buf_V = ctx->attn_v_buffer;
    cuda_buffer_t* buf_attn = ctx->attn_out_buffer;

    cuda_buffer_t** cache_layer_in = ctx->fwd_cache_layer_input;
    cuda_buffer_t** cache_x_norm_attn = ctx->fwd_cache_x_norm_attn;
    cuda_buffer_t** cache_attn_out = ctx->fwd_cache_attn_out;
    cuda_buffer_t** cache_x_norm_ffn = ctx->fwd_cache_x_norm_ffn;
    cuda_buffer_t** cache_ffn_hidden = ctx->fwd_cache_ffn_hidden;
    cuda_buffer_t** cache_q = ctx->fwd_cache_q;
    cuda_buffer_t** cache_k = ctx->fwd_cache_k;
    cuda_buffer_t** cache_v = ctx->fwd_cache_v;

    // ═══════════════════════════════════════════════════════════════════════
    // FORWARD (packed LoRA)
    // ═══════════════════════════════════════════════════════════════════════

    cuda_train_embed_lookup(ctx, ctx->train_tokens_u32, ctx->gpu_w_embed, buf_hidden,
                            total_tokens, hidden_size, vocab_size);

    for (int layer = 0; layer < num_layers; layer++) {
        cuda_train_copy_buffer(ctx, buf_hidden, cache_layer_in[layer],
                               (size_t)total_tokens * (size_t)hidden_size * sizeof(float));
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_in_norm[layer], buf_hidden_norm,
                           total_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_attn[layer],
                               (size_t)total_tokens * (size_t)hidden_size * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_q[layer], buf_Q,
                                    total_tokens, q_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_k[layer], buf_K,
                                    total_tokens, kv_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_v[layer], buf_V,
                                    total_tokens, kv_dim, hidden_size, 0, 1, 0);

        // LoRA contributions
        if (lg->q.active && lg->q.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->q.A[layer], ctx->lora_hidden_tmp,
                                        total_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)total_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->q.B[layer], buf_Q,
                                        total_tokens, q_dim, lg->rank, 0, 1, 1);
        }
        if (lg->k.active && lg->k.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->k.A[layer], ctx->lora_hidden_tmp,
                                        total_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)total_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->k.B[layer], buf_K,
                                        total_tokens, kv_dim, lg->rank, 0, 1, 1);
        }
        if (lg->v.active && lg->v.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->v.A[layer], ctx->lora_hidden_tmp,
                                        total_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)total_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->v.B[layer], buf_V,
                                        total_tokens, kv_dim, lg->rank, 0, 1, 1);
        }

        cuda_train_rope_packed(ctx, buf_Q, buf_Q, ctx->train_seq_starts, total_tokens,
                               batch_size, num_heads, head_dim, rope_theta);
        cuda_train_rope_packed(ctx, buf_K, buf_K, ctx->train_seq_starts, total_tokens,
                               batch_size, kv_heads, head_dim, rope_theta);

        cuda_train_copy_buffer(ctx, buf_Q, cache_q[layer], (size_t)total_tokens * (size_t)q_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_K, cache_k[layer], (size_t)total_tokens * (size_t)kv_dim * sizeof(float));
        cuda_train_copy_buffer(ctx, buf_V, cache_v[layer], (size_t)total_tokens * (size_t)kv_dim * sizeof(float));

        cuda_train_batch_attention_packed(ctx, buf_Q, buf_K, buf_V, buf_attn,
                                          ctx->fwd_cache_attn_lse[layer],
                                          ctx->train_seq_starts, total_tokens, batch_size,
                                          num_heads, kv_heads, head_dim, max_seq_in_batch);
        cuda_train_copy_buffer(ctx, buf_attn, cache_attn_out[layer],
                               (size_t)total_tokens * (size_t)q_dim * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_attn, ctx->gpu_w_o[layer], buf_tmp_hidden,
                                    total_tokens, hidden_size, q_dim, 0, 1, 0);
        if (lg->o.active && lg->o.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_attn, lg->o.A[layer], ctx->lora_hidden_tmp,
                                        total_tokens, lg->rank, q_dim, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)total_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->o.B[layer], buf_tmp_hidden,
                                        total_tokens, hidden_size, lg->rank, 0, 1, 1);
        }
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)total_tokens * (size_t)hidden_size);

        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_post_norm[layer], buf_hidden_norm,
                           total_tokens, hidden_size, rms_eps);
        cuda_train_copy_buffer(ctx, buf_hidden_norm, cache_x_norm_ffn[layer],
                               (size_t)total_tokens * (size_t)hidden_size * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_gate[layer], buf_gate,
                                    total_tokens, intermediate_size, hidden_size, 0, 1, 0);
        if (lg->gate.active && lg->gate.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->gate.A[layer], ctx->lora_hidden_tmp,
                                        total_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)total_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->gate.B[layer], buf_gate,
                                        total_tokens, intermediate_size, lg->rank, 0, 1, 1);
        }
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_up[layer], buf_up,
                                    total_tokens, intermediate_size, hidden_size, 0, 1, 0);
        if (lg->up.active && lg->up.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_hidden_norm, lg->up.A[layer], ctx->lora_hidden_tmp,
                                        total_tokens, lg->rank, hidden_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)total_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->up.B[layer], buf_up,
                                        total_tokens, intermediate_size, lg->rank, 0, 1, 1);
        }

        cuda_train_silu(ctx, buf_gate, buf_ffn_hidden, (size_t)total_tokens * (size_t)intermediate_size);
        cuda_train_mul(ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden,
                       (size_t)total_tokens * (size_t)intermediate_size);
        cuda_train_copy_buffer(ctx, buf_ffn_hidden, cache_ffn_hidden[layer],
                               (size_t)total_tokens * (size_t)intermediate_size * sizeof(float));

        cuda_train_matmul_transpose(ctx, buf_ffn_hidden, ctx->gpu_w_down[layer], buf_tmp_hidden,
                                    total_tokens, hidden_size, intermediate_size, 0, 1, 0);
        if (lg->down.active && lg->down.A[layer]) {
            cuda_train_matmul_transpose(ctx, buf_ffn_hidden, lg->down.A[layer], ctx->lora_hidden_tmp,
                                        total_tokens, lg->rank, intermediate_size, 0, 1, 0);
            cuda_train_scale(ctx, ctx->lora_hidden_tmp, (size_t)total_tokens * (size_t)lg->rank, lora_scale);
            cuda_train_matmul_transpose(ctx, ctx->lora_hidden_tmp, lg->down.B[layer], buf_tmp_hidden,
                                        total_tokens, hidden_size, lg->rank, 0, 1, 1);
        }
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)total_tokens * (size_t)hidden_size);
    }

    cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_final_norm, buf_hidden_norm,
                       total_tokens, hidden_size, rms_eps);
    cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_embed, buf_logits,
                                total_tokens, vocab_size, hidden_size, 0, 1, 0);

    // ═══════════════════════════════════════════════════════════════════════
    // LOSS + BACKWARD (packed LoRA)
    // ═══════════════════════════════════════════════════════════════════════

    cuda_train_cross_entropy_grad_packed(ctx, buf_logits, ctx->train_targets_u32,
                                          ctx->train_grad_logits, ctx->train_loss_rows,
                                          ctx->train_seq_starts,
                                          total_tokens, batch_size, vocab_size, num_valid_predictions);

    size_t reduce_count = (size_t)total_tokens;
    cuda_buffer_t* reduce_in = ctx->train_loss_rows;
    cuda_buffer_t* reduce_out = ctx->train_reduce_tmp_a;
    while (reduce_count > 1) {
        size_t reduce_block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
        size_t out_count = (reduce_count + reduce_block - 1) / reduce_block;
        cuda_train_reduce_sum(ctx, reduce_in, reduce_out, reduce_count);
        reduce_count = out_count;
        cuda_buffer_t* tmp = reduce_in;
        reduce_in = reduce_out;
        reduce_out = tmp;
    }

    cuda_train_matmul(ctx, ctx->train_grad_logits, ctx->gpu_w_embed,
                      ctx->train_grad_hidden, total_tokens, hidden_size, vocab_size);

    for (int layer = num_layers - 1; layer >= 0; layer--) {
        if (lg->down.active && lg->down.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_ffn_hidden[layer],
                                     lg->down.A[layer], lg->down.B[layer],
                                     lg->down.gA[layer], lg->down.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     total_tokens, intermediate_size, hidden_size, lg->rank, lora_scale);
        }
        if (lg->gate.active && lg->gate.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_ffn[layer],
                                     lg->gate.A[layer], lg->gate.B[layer],
                                     lg->gate.gA[layer], lg->gate.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     total_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }
        if (lg->up.active && lg->up.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_ffn[layer],
                                     lg->up.A[layer], lg->up.B[layer],
                                     lg->up.gA[layer], lg->up.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     total_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }
        if (lg->o.active && lg->o.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_attn_out[layer],
                                     lg->o.A[layer], lg->o.B[layer],
                                     lg->o.gA[layer], lg->o.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     total_tokens, q_dim, hidden_size, lg->rank, lora_scale);
        }
        if (lg->q.active && lg->q.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_attn[layer],
                                     lg->q.A[layer], lg->q.B[layer],
                                     lg->q.gA[layer], lg->q.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     total_tokens, hidden_size, q_dim, lg->rank, lora_scale);
        }
        if (lg->k.active && lg->k.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_attn[layer],
                                     lg->k.A[layer], lg->k.B[layer],
                                     lg->k.gA[layer], lg->k.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     total_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }
        if (lg->v.active && lg->v.A[layer]) {
            cuda_train_lora_backward(ctx, ctx->train_grad_hidden, cache_x_norm_attn[layer],
                                     lg->v.A[layer], lg->v.B[layer],
                                     lg->v.gA[layer], lg->v.gB[layer],
                                     ctx->lora_hidden_tmp,
                                     total_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }
    }

    // Optimizer
    state->accumulation_counter++;
    int accum_steps = state->gradient_accumulation_steps;
    if (accum_steps < 1) accum_steps = 1;

    if (state->accumulation_counter >= accum_steps) {
        state->optimizer_step++;
        float base_lr = state->learning_rate;
        float warmup_steps = 100.0f;
        float lr = base_lr;
        if (state->optimizer_step < warmup_steps)
            lr = base_lr * (state->optimizer_step / warmup_steps);
        lr = lr / accum_steps;
        cuda_lora_optimizer_step_with_lora(ctx, state->lora, lr, 0.9f, 0.999f, 0.01f, 1e-8f,
                                           state->optimizer_step);
        state->accumulation_counter = 0;
    }

    cuda_train_sync(ctx);

    float loss_scalar = 0.0f;
    cuda_train_buffer_download(ctx, reduce_in, &loss_scalar, 1);
    loss_scalar = loss_scalar / num_valid_predictions;

    if ((state->current_step % 20) == 0) {
        double t1_ms = now_ms();
        fprintf(stderr, "[TIMING] step=%d batch=%dx%d cuda_lora_step=%.2fms (gpu=on)\n",
                state->current_step, batch_size, total_tokens, (t1_ms - t0_ms));
    }

    return loss_scalar;
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION (forward + loss only, no backward/optimizer)
// ═══════════════════════════════════════════════════════════════════════════

float cuda_eval_forward(train_state_t* state, int* tokens, int num_tokens,
                         cuda_train_context_t* ctx) {
    if (!state || !state->config || !ctx || num_tokens < 2) return -1.0f;

    const model_config_t* config = state->config;
    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int vocab_size = config->vocab_size;
    const int num_layers = config->num_hidden_layers;
    const int num_heads = config->num_attention_heads;
    const int head_dim = config->head_dim;
    const int kv_heads = config->num_key_value_heads;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = kv_heads * head_dim;
    const float rms_eps = 1e-5f;
    const float rope_theta = config->rope_theta > 0 ? config->rope_theta : 10000.0f;

    int max_seq = num_tokens;
    ensure_cuda_fwd_buffers(ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_train_buffers(ctx, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_cuda_gpu_weights(ctx, state);

    cuda_buffer_t* buf_hidden = ctx->fwd_hidden_buffer;
    cuda_buffer_t* buf_hidden_norm = ctx->fwd_hidden_norm_buffer;
    cuda_buffer_t* buf_tmp_hidden = ctx->fwd_tmp_hidden_buffer;
    cuda_buffer_t* buf_gate = ctx->fwd_ffn_gate_buffer;
    cuda_buffer_t* buf_up = ctx->fwd_ffn_up_buffer;
    cuda_buffer_t* buf_ffn_hidden = ctx->fwd_ffn_hidden_buffer;
    cuda_buffer_t* buf_logits = ctx->fwd_logits_buffer;
    cuda_buffer_t* buf_Q = ctx->attn_q_buffer;
    cuda_buffer_t* buf_K = ctx->attn_k_buffer;
    cuda_buffer_t* buf_V = ctx->attn_v_buffer;
    cuda_buffer_t* buf_attn = ctx->attn_out_buffer;

    // Upload tokens + targets
    uint32_t* tokens_u32 = malloc((size_t)num_tokens * sizeof(uint32_t));
    if (!tokens_u32) return -1.0f;
    for (int i = 0; i < num_tokens; i++)
        tokens_u32[i] = (tokens[i] < 0) ? 0u : (uint32_t)tokens[i];
    cuda_train_buffer_upload_u32(ctx, ctx->train_tokens_u32, tokens_u32, (size_t)num_tokens);

    int num_predictions = num_tokens - 1;
    uint32_t* targets_u32 = malloc((size_t)num_predictions * sizeof(uint32_t));
    if (!targets_u32) { free(tokens_u32); return -1.0f; }
    for (int t = 0; t < num_predictions; t++) {
        int tok = tokens[t + 1];
        targets_u32[t] = (tok < 0) ? 0u : (uint32_t)tok;
    }
    cuda_train_buffer_upload_u32(ctx, ctx->train_targets_u32, targets_u32, (size_t)num_predictions);
    free(tokens_u32);
    free(targets_u32);

    // Forward: embedding
    cuda_train_embed_lookup(ctx, ctx->train_tokens_u32, ctx->gpu_w_embed, buf_hidden,
                            num_tokens, hidden_size, vocab_size);

    // Forward: transformer layers
    for (int layer = 0; layer < num_layers; layer++) {
        // RMSNorm (pre-attention)
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_in_norm[layer], buf_hidden_norm,
                           num_tokens, hidden_size, rms_eps);

        // Q, K, V projections
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_q[layer], buf_Q,
                                    num_tokens, q_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_k[layer], buf_K,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_v[layer], buf_V,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);

        // RoPE
        cuda_train_rope(ctx, buf_Q, buf_Q, num_tokens, num_heads, head_dim, rope_theta);
        cuda_train_rope(ctx, buf_K, buf_K, num_tokens, kv_heads, head_dim, rope_theta);

        // Attention (no LSE cache needed — no backward)
        cuda_train_batch_attention(ctx, buf_Q, buf_K, buf_V, buf_attn,
                                   NULL, num_tokens, num_heads, kv_heads, head_dim);

        // O projection + residual
        cuda_train_matmul_transpose(ctx, buf_attn, ctx->gpu_w_o[layer], buf_tmp_hidden,
                                    num_tokens, hidden_size, q_dim, 0, 1, 0);
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);

        // RMSNorm (post-attention, pre-FFN)
        cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_post_norm[layer], buf_hidden_norm,
                           num_tokens, hidden_size, rms_eps);

        // FFN: gate + up + silu + mul + down + residual
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_gate[layer], buf_gate,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_up[layer], buf_up,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        cuda_train_silu(ctx, buf_gate, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
        cuda_train_mul(ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden,
                       (size_t)num_tokens * (size_t)intermediate_size);
        cuda_train_matmul_transpose(ctx, buf_ffn_hidden, ctx->gpu_w_down[layer], buf_tmp_hidden,
                                    num_tokens, hidden_size, intermediate_size, 0, 1, 0);
        cuda_train_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);
    }

    // Final norm
    cuda_train_rmsnorm(ctx, buf_hidden, ctx->gpu_w_final_norm, buf_hidden_norm,
                       num_tokens, hidden_size, rms_eps);

    // LM head: logits = hidden_norm @ embed^T
    cuda_train_matmul_transpose(ctx, buf_hidden_norm, ctx->gpu_w_embed, buf_logits,
                                num_tokens, vocab_size, hidden_size, 0, 1, 0);

    // Cross-entropy loss (grad computed but discarded — kernel fuses loss+grad)
    cuda_train_cross_entropy_grad(ctx, buf_logits, ctx->train_targets_u32,
                                  ctx->train_grad_logits, ctx->train_loss_rows,
                                  num_predictions, vocab_size);

    // Reduce sum loss
    size_t reduce_count = (size_t)num_predictions;
    cuda_buffer_t* reduce_in = ctx->train_loss_rows;
    cuda_buffer_t* reduce_out = ctx->train_reduce_tmp_a;
    while (reduce_count > 1) {
        size_t reduce_block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
        size_t out_count = (reduce_count + reduce_block - 1) / reduce_block;
        cuda_train_reduce_sum(ctx, reduce_in, reduce_out, reduce_count);
        reduce_count = out_count;
        cuda_buffer_t* tmp = reduce_in;
        reduce_in = reduce_out;
        reduce_out = tmp;
    }

    // Download scalar loss
    float loss_sum = 0.0f;
    cuda_train_buffer_download(ctx, reduce_in, &loss_sum, 1);
    return loss_sum / (float)num_predictions;
}

float cuda_evaluate(train_state_t* state, int* val_tokens, int num_val_tokens,
                     int* offsets, int num_samples, int max_seq_len,
                     int max_eval_samples,
                     cuda_train_context_t* ctx, float* out_perplexity) {
    if (!state || !val_tokens || num_samples <= 0 || !ctx) {
        if (out_perplexity) *out_perplexity = 0.0f;
        return -1.0f;
    }

    int max_eval = (max_eval_samples > 0) ? max_eval_samples : num_samples;
    int stride = num_samples > max_eval ? num_samples / max_eval : 1;

    double total_loss = 0.0;
    int valid_samples = 0;

    for (int s = 0; s < num_samples; s += stride) {
        int start = offsets[s];
        int end = (s + 1 < num_samples) ? offsets[s + 1] : num_val_tokens;
        int seq_len = end - start;
        if (seq_len > max_seq_len) seq_len = max_seq_len;
        if (seq_len < 2) continue;

        float loss = cuda_eval_forward(state, &val_tokens[start], seq_len, ctx);
        if (loss >= 0.0f && !isnan(loss)) {
            total_loss += loss;
            valid_samples++;
        }
    }

    float avg_loss = valid_samples > 0 ? (float)(total_loss / valid_samples) : -1.0f;
    if (out_perplexity) {
        *out_perplexity = avg_loss > 0.0f ? expf(avg_loss) : 0.0f;
    }
    return avg_loss;
}
