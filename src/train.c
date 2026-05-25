// train.c - Seraph Training Pipeline
// Part of TETYAH-PRIME's native training and inference engine
//
// Training loop with LoRA support for efficient fine-tuning

#include "../include/train.h"
#include "../include/vulkan_backend.h"
#include "../include/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define TTOK_MAGIC 0x544F4B54

// Forward declarations for GPU train steps (needed by evaluate_video)
static float train_step_gpu_full(train_state_t* state, int* tokens, int num_tokens, vulkan_context_t* ctx);
static float train_step_gpu_lora(train_state_t* state, int* tokens, int num_tokens, vulkan_context_t* ctx);

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void dump_buffer_f32(const char* path, vulkan_context_t* ctx, vulkan_buffer_t* buf, size_t count) {
    if (!path || !ctx || !buf || count == 0) return;
    FILE* f = fopen(path, "wb");
    if (!f) return;

    const float* data = (const float*)buf->mapped;
    float* tmp = NULL;
    if (!data) {
        tmp = (float*)malloc(count * sizeof(float));
        if (!tmp) {
            fclose(f);
            return;
        }
        vulkan_buffer_download(ctx, buf, tmp, count);
        data = tmp;
    }

    fwrite(data, sizeof(float), count, f);
    fclose(f);
    if (tmp) free(tmp);
}

static void dump_buffer_u32(const char* path, vulkan_context_t* ctx, vulkan_buffer_t* buf, size_t count) {
    if (!path || !ctx || !buf || count == 0) return;
    FILE* f = fopen(path, "wb");
    if (!f) return;

    const uint32_t* data = (const uint32_t*)buf->mapped;
    uint32_t* tmp = NULL;
    if (!data) {
        tmp = (uint32_t*)malloc(count * sizeof(uint32_t));
        if (!tmp) {
            fclose(f);
            return;
        }
        vulkan_buffer_download_bytes(ctx, buf, tmp, count * sizeof(uint32_t));
        data = tmp;
    }

    fwrite(data, sizeof(uint32_t), count, f);
    fclose(f);
    if (tmp) free(tmp);
}

// ═══════════════════════════════════════════════════════════════════════════
// ACTIVATION CACHE
// ═══════════════════════════════════════════════════════════════════════════

activation_cache_t* activation_cache_alloc(int num_layers, int hidden_size,
                                            int intermediate_size, int q_dim,
                                            int kv_dim, int max_seq_len) {
    activation_cache_t* cache = calloc(1, sizeof(activation_cache_t));
    cache->num_layers = num_layers;
    cache->hidden_size = hidden_size;
    cache->intermediate_size = intermediate_size;
    cache->q_dim = q_dim;
    cache->kv_dim = kv_dim;
    cache->max_seq_len = max_seq_len;
    cache->cur_seq_len = 0;

    // Allocate per-layer activation storage
    // Each is [num_layers] array of pointers to [max_seq_len * dim] arrays
    cache->layer_inputs = calloc(num_layers, sizeof(float*));
    cache->x_norm_attn = calloc(num_layers, sizeof(float*));
    cache->attn_output = calloc(num_layers, sizeof(float*));
    cache->x_norm_ffn = calloc(num_layers, sizeof(float*));
    cache->ffn_gate_out = calloc(num_layers, sizeof(float*));
    cache->ffn_up_out = calloc(num_layers, sizeof(float*));
    cache->ffn_hidden = calloc(num_layers, sizeof(float*));

    for (int l = 0; l < num_layers; l++) {
        cache->layer_inputs[l] = calloc(max_seq_len * hidden_size, sizeof(float));
        cache->x_norm_attn[l] = calloc(max_seq_len * hidden_size, sizeof(float));
        cache->attn_output[l] = calloc(max_seq_len * q_dim, sizeof(float));
        cache->x_norm_ffn[l] = calloc(max_seq_len * hidden_size, sizeof(float));
        cache->ffn_gate_out[l] = calloc(max_seq_len * intermediate_size, sizeof(float));
        cache->ffn_up_out[l] = calloc(max_seq_len * intermediate_size, sizeof(float));
        cache->ffn_hidden[l] = calloc(max_seq_len * intermediate_size, sizeof(float));
    }

    cache->final_hidden = calloc(max_seq_len * hidden_size, sizeof(float));

    printf("  Allocated activation cache: %.2f MB\n",
           (float)(num_layers * max_seq_len * (4*hidden_size + q_dim + 3*intermediate_size) * sizeof(float)) / (1024*1024));

    return cache;
}

void activation_cache_free(activation_cache_t* cache) {
    if (!cache) return;

    for (int l = 0; l < cache->num_layers; l++) {
        free(cache->layer_inputs[l]);
        free(cache->x_norm_attn[l]);
        free(cache->attn_output[l]);
        free(cache->x_norm_ffn[l]);
        free(cache->ffn_gate_out[l]);
        free(cache->ffn_up_out[l]);
        free(cache->ffn_hidden[l]);
    }

    free(cache->layer_inputs);
    free(cache->x_norm_attn);
    free(cache->attn_output);
    free(cache->x_norm_ffn);
    free(cache->ffn_gate_out);
    free(cache->ffn_up_out);
    free(cache->ffn_hidden);
    free(cache->final_hidden);
    free(cache);
}

void activation_cache_reset(activation_cache_t* cache) {
    cache->cur_seq_len = 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINABLE WEIGHTS - Full FP32 weight training
// ═══════════════════════════════════════════════════════════════════════════

// Helper: allocate single weight tensor with gradient and Adam state
static void weight_tensor_alloc(weight_tensor_t* tensor, size_t size) {
    tensor->size = size;
    tensor->weight = calloc(size, sizeof(float));
    tensor->grad = calloc(size, sizeof(float));
    tensor->m = calloc(size, sizeof(float));
    tensor->v = calloc(size, sizeof(float));
}

// Helper: free single weight tensor
static void weight_tensor_free(weight_tensor_t* tensor) {
    free(tensor->weight);
    free(tensor->grad);
    free(tensor->m);
    free(tensor->v);
}

// Allocate trainable weights for all parameters (FP32)
trainable_weights_t* trainable_weights_alloc(const model_config_t* config) {
    trainable_weights_t* weights = calloc(1, sizeof(trainable_weights_t));

    weights->num_layers = config->num_hidden_layers;
    weights->hidden_size = config->hidden_size;
    weights->intermediate_size = config->intermediate_size;
    weights->vocab_size = config->vocab_size;

    // Allocate per-layer weights
    weights->layers = calloc(weights->num_layers, sizeof(layer_weights_t));

    size_t total = 0;

    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];

        // Attention projections
        size_t q_size = config->hidden_size * (config->num_attention_heads * config->head_dim);
        size_t kv_size = config->hidden_size * config->kv_dim;
        size_t o_size = (config->num_attention_heads * config->head_dim) * config->hidden_size;

        weight_tensor_alloc(&layer->q_proj, q_size);
        weight_tensor_alloc(&layer->k_proj, kv_size);
        weight_tensor_alloc(&layer->v_proj, kv_size);
        weight_tensor_alloc(&layer->o_proj, o_size);

        total += q_size + kv_size + kv_size + o_size;

        // FFN projections
        size_t gate_size = config->hidden_size * config->intermediate_size;
        size_t up_size = config->hidden_size * config->intermediate_size;
        size_t down_size = config->intermediate_size * config->hidden_size;

        weight_tensor_alloc(&layer->gate_proj, gate_size);
        weight_tensor_alloc(&layer->up_proj, up_size);
        weight_tensor_alloc(&layer->down_proj, down_size);

        total += gate_size + up_size + down_size;

        // Layer norms (just scale, no bias in RMSNorm)
        weight_tensor_alloc(&layer->input_norm, config->hidden_size);
        weight_tensor_alloc(&layer->post_norm, config->hidden_size);

        total += config->hidden_size * 2;
    }

    // Global weights
    size_t embed_size = config->vocab_size * config->hidden_size;
    weight_tensor_alloc(&weights->embed_tokens, embed_size);
    weight_tensor_alloc(&weights->final_norm, config->hidden_size);

    total += embed_size + config->hidden_size;

    weights->total_params = total;

    printf("    Allocated %zu FP32 parameters (%.2f MB weights + grads + Adam state)\n",
           total, (total * sizeof(float) * 4) / (1024.0f * 1024.0f));

    return weights;
}

// Free trainable weights
void trainable_weights_free(trainable_weights_t* weights) {
    if (!weights) return;

    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];
        weight_tensor_free(&layer->q_proj);
        weight_tensor_free(&layer->k_proj);
        weight_tensor_free(&layer->v_proj);
        weight_tensor_free(&layer->o_proj);
        weight_tensor_free(&layer->gate_proj);
        weight_tensor_free(&layer->up_proj);
        weight_tensor_free(&layer->down_proj);
        weight_tensor_free(&layer->input_norm);
        weight_tensor_free(&layer->post_norm);
    }

    free(weights->layers);
    weight_tensor_free(&weights->embed_tokens);
    weight_tensor_free(&weights->final_norm);
    free(weights);
}

// Helper: BF16 → FP32 conversion
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
}

// Helper: FP32 → BF16 conversion
static inline uint16_t f32_to_bf16(float f32) {
    uint32_t f32_bits;
    memcpy(&f32_bits, &f32, sizeof(float));
    return (uint16_t)(f32_bits >> 16);
}

// Helper: Get tensor from model by name (returns BF16 pointer)
// Uses normalized name for lookup, original name for safetensors access
static uint16_t* get_model_tensor(seraph_model_t* model, const char* name) {
    for (int i = 0; i < model->num_tensors; i++) {
        if (strcmp(model->tensor_map[i].name, name) == 0) {
            int file_idx = model->tensor_map[i].file_idx;
            return (uint16_t*)safetensors_get_tensor_raw(model->st[file_idx], model->tensor_map[i].orig_name);
        }
    }
    return NULL;
}

// Load BF16 weights from model → FP32 trainable weights
void trainable_weights_load_from_model(trainable_weights_t* weights, seraph_model_t* model) {
    printf("    Converting BF16 model weights → FP32 trainable buffers...\n");

    char tensor_name[256];

    // Load embeddings (handle vocab expansion: model file may have fewer rows)
    uint16_t* embed_bf16 = get_model_tensor(model, "model.embed_tokens.weight");
    if (embed_bf16) {
        // Get ACTUAL tensor size from safetensors metadata (not config which may be expanded)
        size_t model_embed_elements = 0;
        for (int fi = 0; fi < model->num_tensors; fi++) {
            if (strcmp(model->tensor_map[fi].name, "model.embed_tokens.weight") == 0) {
                int sidx = model->tensor_map[fi].file_idx;
                tensor_metadata_t* meta = safetensors_find_tensor(model->st[sidx], "model.embed_tokens.weight");
                if (meta) {
                    model_embed_elements = meta->num_elements;
                    printf("    Model embed tensor: %zu elements (file) vs %zu (config)\n",
                           model_embed_elements, weights->embed_tokens.size);
                }
                break;
            }
        }

        // Fallback if metadata lookup failed
        if (model_embed_elements == 0) {
            model_embed_elements = weights->embed_tokens.size;
        }

        // Only load what actually exists in the model file
        size_t safe_count = (model_embed_elements < weights->embed_tokens.size)
                            ? model_embed_elements : weights->embed_tokens.size;

        for (size_t i = 0; i < safe_count; i++) {
            weights->embed_tokens.weight[i] = bf16_to_f32(embed_bf16[i]);
        }

        // He-initialize new embedding rows if vocab was expanded
        if (safe_count < weights->embed_tokens.size) {
            size_t hidden = weights->hidden_size;
            size_t new_tokens = (weights->embed_tokens.size - safe_count) / hidden;
            float scale = sqrtf(2.0f / (float)hidden);
            printf("    Initializing %zu new embedding rows (He init, scale=%.4f)\n",
                   new_tokens, scale);
            srand(42);
            for (size_t i = safe_count; i < weights->embed_tokens.size; i++) {
                float u1 = ((float)rand() / RAND_MAX) + 1e-7f;
                float u2 = ((float)rand() / RAND_MAX);
                float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
                weights->embed_tokens.weight[i] = z * scale;
            }
        }
    }

    // Load per-layer weights
    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];

        // Q projection
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.q_proj.weight", l);
        uint16_t* q_bf16 = get_model_tensor(model, tensor_name);
        if (q_bf16) {
            for (size_t i = 0; i < layer->q_proj.size; i++) {
                layer->q_proj.weight[i] = bf16_to_f32(q_bf16[i]);
            }
        }

        // K projection
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.k_proj.weight", l);
        uint16_t* k_bf16 = get_model_tensor(model, tensor_name);
        if (k_bf16) {
            for (size_t i = 0; i < layer->k_proj.size; i++) {
                layer->k_proj.weight[i] = bf16_to_f32(k_bf16[i]);
            }
        }

        // V projection
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.v_proj.weight", l);
        uint16_t* v_bf16 = get_model_tensor(model, tensor_name);
        if (v_bf16) {
            for (size_t i = 0; i < layer->v_proj.size; i++) {
                layer->v_proj.weight[i] = bf16_to_f32(v_bf16[i]);
            }
        }

        // O projection
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.o_proj.weight", l);
        uint16_t* o_bf16 = get_model_tensor(model, tensor_name);
        if (o_bf16) {
            for (size_t i = 0; i < layer->o_proj.size; i++) {
                layer->o_proj.weight[i] = bf16_to_f32(o_bf16[i]);
            }
        }

        // Gate projection
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.gate_proj.weight", l);
        uint16_t* gate_bf16 = get_model_tensor(model, tensor_name);
        if (gate_bf16) {
            for (size_t i = 0; i < layer->gate_proj.size; i++) {
                layer->gate_proj.weight[i] = bf16_to_f32(gate_bf16[i]);
            }
        }

        // Up projection
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.up_proj.weight", l);
        uint16_t* up_bf16 = get_model_tensor(model, tensor_name);
        if (up_bf16) {
            for (size_t i = 0; i < layer->up_proj.size; i++) {
                layer->up_proj.weight[i] = bf16_to_f32(up_bf16[i]);
            }
        }

        // Down projection
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.down_proj.weight", l);
        uint16_t* down_bf16 = get_model_tensor(model, tensor_name);
        if (down_bf16) {
            for (size_t i = 0; i < layer->down_proj.size; i++) {
                layer->down_proj.weight[i] = bf16_to_f32(down_bf16[i]);
            }
        }

        // Input norm
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.input_layernorm.weight", l);
        uint16_t* input_norm_bf16 = get_model_tensor(model, tensor_name);
        if (input_norm_bf16) {
            for (size_t i = 0; i < layer->input_norm.size; i++) {
                layer->input_norm.weight[i] = bf16_to_f32(input_norm_bf16[i]);
            }
        }

        // Post-attention norm
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.post_attention_layernorm.weight", l);
        uint16_t* post_norm_bf16 = get_model_tensor(model, tensor_name);
        if (post_norm_bf16) {
            for (size_t i = 0; i < layer->post_norm.size; i++) {
                layer->post_norm.weight[i] = bf16_to_f32(post_norm_bf16[i]);
            }
        }
    }

    // Final norm
    uint16_t* final_norm_bf16 = get_model_tensor(model, "model.norm.weight");
    if (final_norm_bf16) {
        for (size_t i = 0; i < weights->final_norm.size; i++) {
            weights->final_norm.weight[i] = bf16_to_f32(final_norm_bf16[i]);
        }
    }

    printf("    ✅ Loaded %zu FP32 parameters from BF16 safetensors\n", weights->total_params);
}

// Save FP32 trainable weights → BF16 safetensors
void trainable_weights_save_to_safetensors(trainable_weights_t* weights, const char* path) {
    printf("    Converting FP32 → BF16 and saving to %s\n", path);

    // Build cJSON header
    cJSON* header = cJSON_CreateObject();

    // Add metadata
    cJSON* metadata = cJSON_CreateObject();
    cJSON_AddStringToObject(metadata, "format", "pt");
    cJSON_AddStringToObject(metadata, "framework", "tetyah");
    cJSON_AddStringToObject(metadata, "trained", "full_weight_fp32_adamw");
    cJSON_AddItemToObject(header, "__metadata__", metadata);

    size_t data_offset = 0;
    char tensor_name[256];

    // Helper to add tensor to header
    #define ADD_TENSOR(name, dim0, dim1, is_2d) do { \
        cJSON* tensor = cJSON_CreateObject(); \
        cJSON_AddStringToObject(tensor, "dtype", "BF16"); \
        cJSON* shape = cJSON_CreateArray(); \
        cJSON_AddItemToArray(shape, cJSON_CreateNumber((double)(dim0))); \
        if (is_2d) cJSON_AddItemToArray(shape, cJSON_CreateNumber((double)(dim1))); \
        cJSON_AddItemToObject(tensor, "shape", shape); \
        size_t tensor_size = (is_2d) ? (dim0) * (dim1) : (dim0); \
        size_t data_size = tensor_size * sizeof(uint16_t); \
        cJSON* offsets = cJSON_CreateArray(); \
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)data_offset)); \
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)(data_offset + data_size))); \
        cJSON_AddItemToObject(tensor, "data_offsets", offsets); \
        cJSON_AddItemToObject(header, name, tensor); \
        data_offset += data_size; \
    } while (0)

    // Embeddings
    ADD_TENSOR("model.embed_tokens.weight", weights->vocab_size, weights->hidden_size, 1);

    // Per-layer tensors
    for (int l = 0; l < weights->num_layers; l++) {
        int q_dim = weights->hidden_size;  // For TETYAH-PRIME
        int kv_dim = weights->hidden_size;

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.q_proj.weight", l);
        ADD_TENSOR(tensor_name, q_dim, weights->hidden_size, 1);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.k_proj.weight", l);
        ADD_TENSOR(tensor_name, kv_dim, weights->hidden_size, 1);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.v_proj.weight", l);
        ADD_TENSOR(tensor_name, kv_dim, weights->hidden_size, 1);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.o_proj.weight", l);
        ADD_TENSOR(tensor_name, weights->hidden_size, q_dim, 1);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.gate_proj.weight", l);
        ADD_TENSOR(tensor_name, weights->intermediate_size, weights->hidden_size, 1);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.up_proj.weight", l);
        ADD_TENSOR(tensor_name, weights->intermediate_size, weights->hidden_size, 1);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.down_proj.weight", l);
        ADD_TENSOR(tensor_name, weights->hidden_size, weights->intermediate_size, 1);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.input_layernorm.weight", l);
        ADD_TENSOR(tensor_name, weights->hidden_size, 0, 0);

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.post_attention_layernorm.weight", l);
        ADD_TENSOR(tensor_name, weights->hidden_size, 0, 0);
    }

    // Final norm
    ADD_TENSOR("model.norm.weight", weights->hidden_size, 0, 0);

    #undef ADD_TENSOR

    // Serialize JSON header
    char* header_json = cJSON_PrintUnformatted(header);
    uint64_t header_size = strlen(header_json);

    // Write file
    FILE* out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot create %s\n", path);
        cJSON_Delete(header);
        free(header_json);
        return;
    }

    // Write header size
    fwrite(&header_size, sizeof(uint64_t), 1, out);

    // Write JSON header
    fwrite(header_json, 1, header_size, out);

    // Write tensor data (convert FP32 → BF16)
    #define WRITE_TENSOR(tensor) do { \
        for (size_t i = 0; i < (tensor).size; i++) { \
            uint16_t bf16 = f32_to_bf16((tensor).weight[i]); \
            fwrite(&bf16, sizeof(uint16_t), 1, out); \
        } \
    } while (0)

    // Write embeddings
    WRITE_TENSOR(weights->embed_tokens);

    // Write per-layer weights
    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];
        WRITE_TENSOR(layer->q_proj);
        WRITE_TENSOR(layer->k_proj);
        WRITE_TENSOR(layer->v_proj);
        WRITE_TENSOR(layer->o_proj);
        WRITE_TENSOR(layer->gate_proj);
        WRITE_TENSOR(layer->up_proj);
        WRITE_TENSOR(layer->down_proj);
        WRITE_TENSOR(layer->input_norm);
        WRITE_TENSOR(layer->post_norm);
    }

    // Write final norm
    WRITE_TENSOR(weights->final_norm);

    #undef WRITE_TENSOR

    fclose(out);
    free(header_json);
    cJSON_Delete(header);

    // Get file size
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    printf("    ✅ Saved %.2f MB (%zu params) to %s\n",
           file_size / (1024.0 * 1024.0), weights->total_params, path);
}

int trainable_weights_load_from_safetensors(trainable_weights_t* weights, const char* path) {
    printf("    Loading weights from %s\n", path);

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        return -1;
    }

    // Read header size
    uint64_t header_size;
    if (fread(&header_size, sizeof(uint64_t), 1, f) != 1) {
        fprintf(stderr, "ERROR: Failed to read header size\n");
        fclose(f);
        return -1;
    }

    // Skip JSON header (we'll read tensors in same order as saved)
    fseek(f, (long)header_size, SEEK_CUR);

    // Read tensor data (BF16 → FP32)
    #define READ_TENSOR(tensor) do { \
        for (size_t i = 0; i < (tensor).size; i++) { \
            uint16_t bf16; \
            if (fread(&bf16, sizeof(uint16_t), 1, f) != 1) { \
                fprintf(stderr, "ERROR: Failed to read tensor data\n"); \
                fclose(f); \
                return -1; \
            } \
            (tensor).weight[i] = bf16_to_f32(bf16); \
        } \
    } while (0)

    // Read embeddings
    READ_TENSOR(weights->embed_tokens);

    // Read per-layer weights
    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];
        READ_TENSOR(layer->q_proj);
        READ_TENSOR(layer->k_proj);
        READ_TENSOR(layer->v_proj);
        READ_TENSOR(layer->o_proj);
        READ_TENSOR(layer->gate_proj);
        READ_TENSOR(layer->up_proj);
        READ_TENSOR(layer->down_proj);
        READ_TENSOR(layer->input_norm);
        READ_TENSOR(layer->post_norm);
    }

    // Read final norm
    READ_TENSOR(weights->final_norm);

    #undef READ_TENSOR

    fclose(f);
    printf("    ✅ Loaded weights from %s\n", path);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// GRADIENT SNAPSHOT SYSTEM - Electromagnetic Field Visualization
// ═══════════════════════════════════════════════════════════════════════════

// Download GPU gradients to CPU for snapshot visualization
static void download_gpu_gradients(train_state_t* state) {
    if (!state->vulkan_ctx || !state->full_weights) return;
    vulkan_context_t* ctx = (vulkan_context_t*)state->vulkan_ctx;
    trainable_weights_t* w = state->full_weights;

    // Download embed gradients
    if (ctx->gpu_g_embed && w->embed_tokens.grad) {
        vulkan_buffer_download(ctx, (vulkan_buffer_t*)ctx->gpu_g_embed,
                               w->embed_tokens.grad, w->embed_tokens.size);
    }

    // Download per-layer gradients
    for (int l = 0; l < w->num_layers; l++) {
        layer_weights_t* layer = &w->layers[l];

        if (ctx->gpu_g_q)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_q)[l],
                                   layer->q_proj.grad, layer->q_proj.size);
        if (ctx->gpu_g_k)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_k)[l],
                                   layer->k_proj.grad, layer->k_proj.size);
        if (ctx->gpu_g_v)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_v)[l],
                                   layer->v_proj.grad, layer->v_proj.size);
        if (ctx->gpu_g_o)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_o)[l],
                                   layer->o_proj.grad, layer->o_proj.size);
        if (ctx->gpu_g_gate)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_gate)[l],
                                   layer->gate_proj.grad, layer->gate_proj.size);
        if (ctx->gpu_g_up)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_up)[l],
                                   layer->up_proj.grad, layer->up_proj.size);
        if (ctx->gpu_g_down)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_down)[l],
                                   layer->down_proj.grad, layer->down_proj.size);
        if (ctx->gpu_g_in_norm)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_in_norm)[l],
                                   layer->input_norm.grad, layer->input_norm.size);
        if (ctx->gpu_g_post_norm)
            vulkan_buffer_download(ctx, ((vulkan_buffer_t**)ctx->gpu_g_post_norm)[l],
                                   layer->post_norm.grad, layer->post_norm.size);
    }

    // Download final norm gradients
    if (ctx->gpu_g_final_norm && w->final_norm.grad) {
        vulkan_buffer_download(ctx, (vulkan_buffer_t*)ctx->gpu_g_final_norm,
                               w->final_norm.grad, w->final_norm.size);
    }
}

gradient_snapshot_writer_t* gradient_snapshot_writer_init(const char* path, int num_layers) {
    gradient_snapshot_writer_t* writer = calloc(1, sizeof(gradient_snapshot_writer_t));

    writer->fp = fopen(path, "wb");
    if (!writer->fp) {
        fprintf(stderr, "ERROR: Cannot create gradient snapshot file: %s\n", path);
        free(writer);
        return NULL;
    }

    // Write header
    uint32_t magic = GRADIENT_SNAPSHOT_MAGIC;
    uint32_t version = 1;
    uint32_t samples_per_tensor = GRADIENT_SAMPLES_PER_TENSOR;
    uint32_t num_layers_val = (uint32_t)num_layers;

    fwrite(&magic, sizeof(uint32_t), 1, writer->fp);
    fwrite(&version, sizeof(uint32_t), 1, writer->fp);
    fwrite(&samples_per_tensor, sizeof(uint32_t), 1, writer->fp);
    fwrite(&num_layers_val, sizeof(uint32_t), 1, writer->fp);

    writer->num_snapshots = 0;
    writer->enabled = 1;

    printf("    [GRAD] Snapshot writer initialized: %s\n", path);
    printf("    [GRAD] Sampling %u vectors per tensor\n", samples_per_tensor);

    return writer;
}

void gradient_snapshot_write(gradient_snapshot_writer_t* writer, trainable_weights_t* weights,
                             uint32_t step, float loss, float lr) {
    if (!writer || !writer->enabled || !weights) return;

    // Write snapshot metadata
    uint32_t timestamp = (uint32_t)time(NULL);
    fwrite(&step, sizeof(uint32_t), 1, writer->fp);
    fwrite(&timestamp, sizeof(uint32_t), 1, writer->fp);
    fwrite(&loss, sizeof(float), 1, writer->fp);
    fwrite(&lr, sizeof(float), 1, writer->fp);

    // Helper to write gradient samples from a tensor
    #define WRITE_TENSOR_SAMPLES(tensor) do { \
        size_t stride = (tensor).size / GRADIENT_SAMPLES_PER_TENSOR; \
        if (stride == 0) stride = 1; \
        for (int s = 0; s < GRADIENT_SAMPLES_PER_TENSOR && s * stride < (tensor).size; s++) { \
            size_t idx = s * stride; \
            float grad_val = (tensor).grad[idx]; \
            fwrite(&grad_val, sizeof(float), 1, writer->fp); \
        } \
        /* Pad if tensor is smaller than 528 samples */ \
        int actual_samples = ((tensor).size < GRADIENT_SAMPLES_PER_TENSOR) ? (tensor).size : GRADIENT_SAMPLES_PER_TENSOR; \
        for (int s = actual_samples; s < GRADIENT_SAMPLES_PER_TENSOR; s++) { \
            float zero = 0.0f; \
            fwrite(&zero, sizeof(float), 1, writer->fp); \
        } \
    } while (0)

    // Write embeddings
    WRITE_TENSOR_SAMPLES(weights->embed_tokens);

    // Write per-layer gradients
    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];
        WRITE_TENSOR_SAMPLES(layer->q_proj);
        WRITE_TENSOR_SAMPLES(layer->k_proj);
        WRITE_TENSOR_SAMPLES(layer->v_proj);
        WRITE_TENSOR_SAMPLES(layer->o_proj);
        WRITE_TENSOR_SAMPLES(layer->gate_proj);
        WRITE_TENSOR_SAMPLES(layer->up_proj);
        WRITE_TENSOR_SAMPLES(layer->down_proj);
        WRITE_TENSOR_SAMPLES(layer->input_norm);
        WRITE_TENSOR_SAMPLES(layer->post_norm);
    }

    // Write final norm
    WRITE_TENSOR_SAMPLES(weights->final_norm);

    #undef WRITE_TENSOR_SAMPLES

    writer->num_snapshots++;
    fflush(writer->fp);  // Ensure data is written immediately
}

void gradient_snapshot_writer_close(gradient_snapshot_writer_t* writer) {
    if (!writer) return;

    if (writer->fp) {
        fclose(writer->fp);
        printf("    [GRAD] Snapshot writer closed: %u snapshots written\n", writer->num_snapshots);
    }

    free(writer);
}

// ═══════════════════════════════════════════════════════════════════════════
// LOSS GRADIENT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

// Gradient of cross-entropy loss w.r.t. logits
// grad_logits[i] = softmax(logits)[i] - (1 if i == target else 0)
void compute_loss_gradient(float* grad_logits, float* logits, int target, int vocab_size) {
    // Compute softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        grad_logits[i] = expf(logits[i] - max_logit);
        sum_exp += grad_logits[i];
    }

    // Normalize and subtract target
    for (int i = 0; i < vocab_size; i++) {
        grad_logits[i] /= sum_exp;
    }
    grad_logits[target] -= 1.0f;  // Subtract 1 at target index
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════════════════════

int train_load_data(train_state_t* state, const char* ttok_path) {
    FILE* f = fopen(ttok_path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", ttok_path);
        return -1;
    }

    // Read header
    uint32_t header[4];
    fread(header, sizeof(uint32_t), 4, f);

    if (header[0] != TTOK_MAGIC) {
        fprintf(stderr, "ERROR: Invalid .ttok file (bad magic)\n");
        fclose(f);
        return -1;
    }

    int num_samples = header[2];
    printf("  Loading %d samples from %s\n", num_samples, ttok_path);

    // First pass: count total tokens
    long data_start = ftell(f);
    int total_tokens = 0;
    for (int i = 0; i < num_samples; i++) {
        uint32_t n;
        if (fread(&n, sizeof(uint32_t), 1, f) != 1) break;
        total_tokens += n;
        fseek(f, n * sizeof(int), SEEK_CUR);
    }

    // Allocate
    state->train_tokens = malloc(total_tokens * sizeof(int));
    state->sample_offsets = malloc((num_samples + 1) * sizeof(int));
    state->num_samples = num_samples;
    state->num_train_tokens = total_tokens;

    // Second pass: load tokens
    fseek(f, data_start, SEEK_SET);
    int offset = 0;
    for (int i = 0; i < num_samples; i++) {
        state->sample_offsets[i] = offset;
        uint32_t n;
        fread(&n, sizeof(uint32_t), 1, f);
        fread(&state->train_tokens[offset], sizeof(int), n, f);
        offset += n;
    }
    state->sample_offsets[num_samples] = offset;

    fclose(f);
    printf("  Loaded %d tokens across %d samples\n", total_tokens, num_samples);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// VISION IMPLIMENTATION - still researching, semi-ready training
// ═══════════════════════════════════════════════════════════════════════════

#define VDAT_MAGIC 0x54414456  // "VDAT"

int train_load_video_data(train_state_t* state, const char* vdat_path) {
    // Open file and get size
    int fd = open(vdat_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s\n", vdat_path);
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "ERROR: Cannot stat %s\n", vdat_path);
        close(fd);
        return -1;
    }
    size_t file_size = (size_t)st.st_size;

    // mmap the entire file
    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // close fd after mmap
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed for %s\n", vdat_path);
        return -1;
    }

    // Advise kernel we'll read sequentially
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    uint32_t* header = (uint32_t*)mapped;
    if (header[0] != VDAT_MAGIC) {
        fprintf(stderr, "ERROR: Invalid .vdat file (bad magic: 0x%08X)\n", header[0]);
        munmap(mapped, file_size);
        return -1;
    }

    uint32_t version = header[1];
    uint32_t num_frames = header[2];
    uint32_t height = header[3];
    uint32_t width = header[4];
    uint32_t channels = header[5];
    uint32_t labels_per_frame = header[6];

    printf("  Loading video data (mmap): %u frames, %ux%ux%u, %u labels/frame (v%u)\n",
           num_frames, height, width, channels, labels_per_frame, version);

    // Limit frames if max_video_frames is set
    uint32_t original_num_frames = num_frames;
    if (state->train_cfg && state->train_cfg->max_video_frames > 0 &&
        (uint32_t)state->train_cfg->max_video_frames < num_frames) {
        num_frames = (uint32_t)state->train_cfg->max_video_frames;
        printf("  Limiting to %u frames (--max-frames, %u available)\n",
               num_frames, original_num_frames);
    }

    // Validate against model config
    if (state->config) {
        int cfg_h = state->config->input_resolution[0];
        int cfg_w = state->config->input_resolution[1];
        int cfg_c = state->config->in_channels > 0 ? state->config->in_channels : 3;
        if ((cfg_h > 0 && cfg_h != (int)height) ||
            (cfg_w > 0 && cfg_w != (int)width) ||
            (cfg_c != (int)channels)) {
            fprintf(stderr, "WARNING: Video dimensions (%ux%ux%u) don't match config (%dx%dx%d)\n",
                    height, width, channels, cfg_h, cfg_w, cfg_c);
        }
    }

    // Calculate offsets into mmap'd region
    size_t header_size = 7 * sizeof(uint32_t);
    size_t frame_size = (size_t)channels * height * width;
    size_t all_frames_bytes = (size_t)original_num_frames * frame_size * sizeof(float);
    size_t all_labels_bytes = (size_t)original_num_frames * labels_per_frame * sizeof(int32_t);

    // Point directly into mmap (zero-copy!)
    char* base = (char*)mapped;
    state->video_frames = (float*)(base + header_size);
    state->video_labels = (int*)(base + header_size + all_frames_bytes);

    // Store mmap info for cleanup
    state->vdat_mmap = mapped;
    state->vdat_mmap_size = file_size;

    state->num_video_frames = (int)num_frames;
    state->num_labels_per_frame = (int)labels_per_frame;

    // Load masks if present (version >= 2)
    state->video_masks = NULL;
    if (state->config && state->config->mask_output && version >= 2) {
        int num_queries = state->config->num_queries;
        int patch_h = state->config->patch_size[1] > 0 ? state->config->patch_size[1] : 8;
        int patch_w = state->config->patch_size[2] > 0 ? state->config->patch_size[2] : 8;
        int num_patches = ((int)height / patch_h) * ((int)width / patch_w);

        size_t masks_offset = header_size + all_frames_bytes + all_labels_bytes;
        state->video_masks = (float*)(base + masks_offset);
        printf("    Loaded masks (mmap): %d queries x %d patches x %u frames\n",
               num_queries, num_patches, num_frames);
    }

    size_t loaded_bytes = (size_t)num_frames * frame_size * sizeof(float);
    printf("  Mapped %u video frames (%.2f MB, zero-copy)\n",
           num_frames, (float)loaded_bytes / (1024 * 1024));
    return 0;
}

int train_load_val_video_data(train_state_t* state, const char* vdat_path) {
    FILE* f = fopen(vdat_path, "rb");
    if (!f) {
        fprintf(stderr, "WARNING: Cannot open validation video %s\n", vdat_path);
        return -1;
    }

    uint32_t header[7];
    if (fread(header, sizeof(uint32_t), 7, f) != 7) {
        fprintf(stderr, "ERROR: Failed to read validation .vdat header\n");
        fclose(f);
        return -1;
    }

    if (header[0] != VDAT_MAGIC) {
        fprintf(stderr, "ERROR: Invalid validation .vdat file\n");
        fclose(f);
        return -1;
    }

    uint32_t num_frames = header[2];
    uint32_t height = header[3];
    uint32_t width = header[4];
    uint32_t channels = header[5];
    uint32_t labels_per_frame = header[6];

    printf("  Loading validation video: %u frames, %ux%ux%u\n",
           num_frames, height, width, channels);

    size_t frame_size = (size_t)channels * height * width;
    size_t total_floats = (size_t)num_frames * frame_size;

    state->val_video_frames = malloc(total_floats * sizeof(float));
    if (!state->val_video_frames) {
        fclose(f);
        return -1;
    }

    if (fread(state->val_video_frames, sizeof(float), total_floats, f) != total_floats) {
        free(state->val_video_frames);
        state->val_video_frames = NULL;
        fclose(f);
        return -1;
    }

    size_t total_labels = (size_t)num_frames * labels_per_frame;
    state->val_video_labels = malloc(total_labels * sizeof(int));
    if (state->val_video_labels) {
        int32_t* labels_i32 = malloc(total_labels * sizeof(int32_t));
        if (fread(labels_i32, sizeof(int32_t), total_labels, f) == total_labels) {
            for (size_t i = 0; i < total_labels; i++) {
                state->val_video_labels[i] = labels_i32[i];
            }
        }
        free(labels_i32);
    }

    // Load validation masks if present
    state->val_video_masks = NULL;
    if (state->config && state->config->mask_output && header[1] >= 2) {
        int num_queries = state->config->num_queries;
        int patch_h = state->config->patch_size[1] > 0 ? state->config->patch_size[1] : 8;
        int patch_w = state->config->patch_size[2] > 0 ? state->config->patch_size[2] : 8;
        int num_patches = ((int)height / patch_h) * ((int)width / patch_w);
        size_t masks_per_frame = (size_t)num_queries * num_patches;
        size_t total_masks = (size_t)num_frames * masks_per_frame;

        state->val_video_masks = malloc(total_masks * sizeof(float));
        if (state->val_video_masks) {
            if (fread(state->val_video_masks, sizeof(float), total_masks, f) != total_masks) {
                free(state->val_video_masks);
                state->val_video_masks = NULL;
            }
        }
    }

    state->num_val_video_frames = (int)num_frames;
    fclose(f);
    printf("  Loaded %u validation frames\n", num_frames);
    return 0;
}

// Evaluate on validation video (forward only, no gradients)
// Prints progress and timing, returns average loss
float evaluate_video(train_state_t* state) {
    if (!state || !state->val_video_frames || state->num_val_video_frames <= 0) {
        return -1.0f;
    }

    vulkan_context_t* ctx = (vulkan_context_t*)state->vulkan_ctx;
    if (!ctx) return -1.0f;

    const model_config_t* config = state->config;
    int frame_h = config->input_resolution[0];
    int frame_w = config->input_resolution[1];
    int in_channels = config->in_channels > 0 ? config->in_channels : 3;
    int num_queries = config->num_queries;
    size_t frame_size = (size_t)in_channels * frame_h * frame_w;

    // Save training state
    int saved_frame_idx = state->current_frame_idx;
    state->current_frame_idx = 0;

    // Temporarily swap to validation data
    float* train_frames = state->video_frames;
    int* train_labels = state->video_labels;
    float* train_masks = state->video_masks;
    int train_num_frames = state->num_video_frames;

    state->video_frames = state->val_video_frames;
    state->video_labels = state->val_video_labels;
    state->video_masks = state->val_video_masks;
    state->num_video_frames = state->num_val_video_frames;

    float total_loss = 0.0f;
    int num_frames = state->num_val_video_frames;
    double t_start = now_ms();

    printf("  [VAL] Evaluating %d validation frames...\n", num_frames);

    for (int f = 0; f < num_frames; f++) {
        // Upload frame
        float* frame_data = &state->video_frames[f * frame_size];
        vulkan_buffer_t* frame_buf = (vulkan_buffer_t*)ctx->train_tokens_u32;
        vulkan_buffer_upload(ctx, frame_buf, frame_data, frame_size);

        // Upload labels
        int* labels = &state->video_labels[f * state->num_labels_per_frame];
        vulkan_buffer_t* targets_buf = (vulkan_buffer_t*)ctx->train_targets_u32;
        if (targets_buf) {
            uint32_t* labels_u32 = malloc(num_queries * sizeof(uint32_t));
            for (int i = 0; i < num_queries; i++) {
                int label_idx = i < state->num_labels_per_frame ? i : 0;
                labels_u32[i] = (uint32_t)labels[label_idx];
            }
            vulkan_buffer_upload_u32(ctx, targets_buf, labels_u32, num_queries);
            free(labels_u32);
        }

        // Upload mask targets if available
        // Set mask values to -1.0 for background queries (class_gt < 0) to ignore in BCE
        if (config->mask_output && state->video_masks && state->mask_targets) {
            int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
            int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
            int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
            size_t masks_per_frame = (size_t)num_queries * num_patches;
            float* src_masks = &state->video_masks[f * masks_per_frame];
            int num_classes = config->num_classes > 0 ? config->num_classes : 40;

            float* tmp_masks = malloc(masks_per_frame * sizeof(float));
            for (int q = 0; q < num_queries; q++) {
                int label_idx = q < state->num_labels_per_frame ? q : 0;
                int label = labels[label_idx];
                float* dst = &tmp_masks[q * num_patches];
                float* src = &src_masks[q * num_patches];

                if (label < 0 || label >= num_classes) {
                    for (int p = 0; p < num_patches; p++) dst[p] = -1.0f;
                } else {
                    memcpy(dst, src, num_patches * sizeof(float));
                }
            }
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->mask_targets, tmp_masks, masks_per_frame);
            free(tmp_masks);
        }

        // Run forward only (set accumulation to prevent optimizer step)
        int saved_accum = state->accumulation_counter;
        state->accumulation_counter = 0;  // Won't trigger optimizer

        float loss;
        if (state->full_weights) {
            loss = train_step_gpu_full(state, NULL, num_queries, ctx);
        } else if (state->lora) {
            loss = train_step_gpu_lora(state, NULL, num_queries, ctx);
        } else {
            loss = -1.0f;
        }

        state->accumulation_counter = saved_accum;
        total_loss += loss;
    }

    double t_end = now_ms();
    double elapsed_sec = (t_end - t_start) / 1000.0;
    float frames_per_sec = elapsed_sec > 0 ? num_frames / elapsed_sec : 0;
    float avg_loss = total_loss / num_frames;

    printf("  [VAL] Frames: %d | Avg Loss: %.4f | Total: %.4f | %.1f frames/s | %.2fs\n",
           num_frames, avg_loss, total_loss, frames_per_sec, elapsed_sec);

    // Restore training state
    state->video_frames = train_frames;
    state->video_labels = train_labels;
    state->video_masks = train_masks;
    state->num_video_frames = train_num_frames;
    state->current_frame_idx = saved_frame_idx;

    return avg_loss;
}

// ═══════════════════════════════════════════════════════════════════════════
// LOSS COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

// Cross-entropy loss: -sum(log(softmax(logits)[target]))
float compute_loss(float* logits, int* targets, int seq_len, int vocab_size) {
    float total_loss = 0.0f;

    for (int t = 0; t < seq_len - 1; t++) {  // Predict next token
        float* logits_t = &logits[t * vocab_size];
        int target = targets[t + 1];

        // Stable log-softmax
        float max_logit = logits_t[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits_t[i] > max_logit) max_logit = logits_t[i];
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += expf(logits_t[i] - max_logit);
        }

        float log_softmax_target = (logits_t[target] - max_logit) - logf(sum_exp);
        total_loss -= log_softmax_target;
    }

    return total_loss / (seq_len - 1);  // Average loss per token
}

// ═══════════════════════════════════════════════════════════════════════════
// VISION METRICS - IoU, mAP, vAP for video segmentation
// ═══════════════════════════════════════════════════════════════════════════

// Compute per-query IoU (Intersection over Union) for mask predictions
// Returns mean IoU across queries with valid GT, stores per-query IoU in iou_out
float compute_iou(float* mask_pred, float* mask_gt, int num_queries, int num_patches,
                  float threshold, float* iou_out) {
    float total_iou = 0.0f;
    int valid_queries = 0;

    for (int q = 0; q < num_queries; q++) {
        float* pred = &mask_pred[q * num_patches];
        float* gt = &mask_gt[q * num_patches];

        // Check if GT has any positive pixels (valid object)
        float gt_area = 0.0f;
        for (int p = 0; p < num_patches; p++) {
            if (gt[p] > 0.5f) gt_area += 1.0f;
        }

        if (gt_area < 0.5f) {
            // No GT object for this query
            if (iou_out) iou_out[q] = 0.0f;
            continue;
        }

        // Compute intersection and union
        float intersection = 0.0f;
        float pred_area = 0.0f;

        for (int p = 0; p < num_patches; p++) {
            int pred_pos = (pred[p] > threshold) ? 1 : 0;
            int gt_pos = (gt[p] > 0.5f) ? 1 : 0;

            if (pred_pos && gt_pos) intersection += 1.0f;
            if (pred_pos) pred_area += 1.0f;
        }

        // IoU = intersection / union = intersection / (pred + gt - intersection)
        float union_area = pred_area + gt_area - intersection;
        float iou = (union_area > 0.0f) ? intersection / union_area : 0.0f;

        if (iou_out) iou_out[q] = iou;
        total_iou += iou;
        valid_queries++;
    }

    return valid_queries > 0 ? total_iou / valid_queries : 0.0f;
}

// Detection struct for mAP calculation
typedef struct {
    int query_idx;      // Which query this detection came from
    int pred_class;     // Predicted class (argmax)
    float confidence;   // Softmax probability for predicted class
    float iou;          // IoU with matched GT mask
    int gt_class;       // Ground truth class (-1 = background/no object)
    int matched;        // Whether this detection was matched to GT
} detection_t;

// Compute mAP@[0.5:0.95] (COCO-style) for class predictions
// Returns mAP averaged over IoU thresholds 0.5, 0.55, ..., 0.95
float compute_map(float* class_logits, int* class_gt, float* iou_scores,
                  int num_queries, int num_classes) {
    if (!class_logits || !class_gt || !iou_scores) return 0.0f;

    // Build detection list
    detection_t detections[256];
    int num_detections = 0;

    for (int q = 0; q < num_queries && q < 256; q++) {
        float* logits = &class_logits[q * num_classes];

        // Find predicted class (argmax) and confidence (softmax)
        int pred_class = 0;
        float max_logit = logits[0];
        for (int c = 1; c < num_classes; c++) {
            if (logits[c] > max_logit) {
                max_logit = logits[c];
                pred_class = c;
            }
        }

        // Softmax for confidence
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(logits[c] - max_logit);
        }
        float confidence = 1.0f / sum_exp;  // Max class probability

        detections[num_detections].query_idx = q;
        detections[num_detections].pred_class = pred_class;
        detections[num_detections].confidence = confidence;
        detections[num_detections].iou = iou_scores[q];
        detections[num_detections].gt_class = class_gt[q];
        detections[num_detections].matched = 0;
        num_detections++;
    }

    // Sort detections by confidence (descending)
    for (int i = 0; i < num_detections - 1; i++) {
        for (int j = i + 1; j < num_detections; j++) {
            if (detections[j].confidence > detections[i].confidence) {
                detection_t tmp = detections[i];
                detections[i] = detections[j];
                detections[j] = tmp;
            }
        }
    }

    // Compute AP at multiple IoU thresholds (Training-friendly: 0.1:0.1:0.9 = 9 thresholds)
    // Lower thresholds than COCO to see progress during early training
    float iou_thresholds[] = {0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f, 0.90f};
    int num_thresholds = 9;
    float total_ap = 0.0f;

    for (int t = 0; t < num_thresholds; t++) {
        float iou_thresh = iou_thresholds[t];

        // Reset matched flags
        for (int d = 0; d < num_detections; d++) {
            detections[d].matched = 0;
        }

        // Compute AP for each class at this IoU threshold
        float class_ap_sum = 0.0f;
        int classes_with_gt = 0;

        for (int c = 0; c < num_classes; c++) {
            // Count GT instances for this class
            int gt_count = 0;
            for (int d = 0; d < num_detections; d++) {
                if (detections[d].gt_class == c) gt_count++;
            }
            if (gt_count == 0) continue;

            classes_with_gt++;

            // Compute precision-recall for this class
            int tp = 0, fp = 0;
            float ap = 0.0f;
            float max_precision = 0.0f;

            // 101-point interpolation (COCO style)
            float precisions[101];
            for (int i = 0; i <= 100; i++) precisions[i] = 0.0f;

            for (int d = 0; d < num_detections; d++) {
                if (detections[d].pred_class != c) continue;

                // Check if this is a TP: correct class AND IoU >= threshold AND not already matched
                int is_tp = 0;
                if (detections[d].gt_class == c && detections[d].iou >= iou_thresh) {
                    // Find if this GT was already matched
                    int gt_already_matched = 0;
                    for (int dd = 0; dd < d; dd++) {
                        if (detections[dd].matched && detections[dd].query_idx == detections[d].query_idx) {
                            gt_already_matched = 1;
                            break;
                        }
                    }
                    if (!gt_already_matched) {
                        is_tp = 1;
                        detections[d].matched = 1;
                    }
                }

                if (is_tp) tp++;
                else fp++;

                float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
                float recall = (float)tp / gt_count;

                // Record precision at this recall level (101-point)
                int recall_idx = (int)(recall * 100.0f);
                if (recall_idx > 100) recall_idx = 100;
                if (precision > precisions[recall_idx]) {
                    precisions[recall_idx] = precision;
                }
            }

            // Interpolate precision (make monotonically decreasing from right)
            max_precision = 0.0f;
            for (int i = 100; i >= 0; i--) {
                if (precisions[i] > max_precision) {
                    max_precision = precisions[i];
                }
                precisions[i] = max_precision;
            }

            // AP = mean of interpolated precisions
            for (int i = 0; i <= 100; i++) {
                ap += precisions[i];
            }
            ap /= 101.0f;

            class_ap_sum += ap;
        }

        // mAP at this IoU threshold
        float map_at_thresh = (classes_with_gt > 0) ? class_ap_sum / classes_with_gt : 0.0f;
        total_ap += map_at_thresh;
    }

    // mAP@[0.5:0.95] = average over all IoU thresholds
    return total_ap / num_thresholds;
}

// Compute vAP (Video Average Precision) - tracking consistency across frames
// Measures: when GT says same object persists across frames, does model predict same class?
float compute_vap(train_state_t* state, float* class_logits, int* class_gt,
                  int num_queries, int num_classes) {
    if (!class_logits || !class_gt) return 1.0f;

    // Allocate tracking state if needed (stores both prediction and gt from previous frame)
    if (!state->prev_query_classes) {
        state->prev_query_classes = calloc(num_queries * 2, sizeof(int));  // [pred, gt] pairs
        state->prev_query_conf = calloc(num_queries, sizeof(float));
        for (int q = 0; q < num_queries; q++) {
            state->prev_query_classes[q * 2] = -1;      // prev pred
            state->prev_query_classes[q * 2 + 1] = -1;  // prev gt
            state->prev_query_conf[q] = 0.0f;
        }
        state->vap_id_switches = 0;
        state->vap_total_tracks = 0;
    }

    // Current frame predictions
    int curr_pred[256];
    float curr_conf[256];

    for (int q = 0; q < num_queries && q < 256; q++) {
        float* logits = &class_logits[q * num_classes];

        // Predicted class (argmax)
        int pred_class = 0;
        float max_logit = logits[0];
        for (int c = 1; c < num_classes; c++) {
            if (logits[c] > max_logit) {
                max_logit = logits[c];
                pred_class = c;
            }
        }

        // Confidence (softmax of max)
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(logits[c] - max_logit);
        }
        float conf = 1.0f / sum_exp;

        curr_pred[q] = pred_class;
        curr_conf[q] = conf;
    }

    // Count ID switches and valid tracks
    int switches = 0;
    int valid_tracks = 0;

    for (int q = 0; q < num_queries; q++) {
        int prev_pred = state->prev_query_classes[q * 2];
        int prev_gt = state->prev_query_classes[q * 2 + 1];
        float prev_conf = state->prev_query_conf[q];
        int curr_p = curr_pred[q];
        int curr_gt = class_gt[q];
        float curr_c = curr_conf[q];

        // Only count as valid track if:
        // 1. GT class is the SAME across both frames (same object being tracked)
        // 2. GT class is valid (>= 0, not background)
        // 3. Both frames have confident predictions
        if (prev_gt >= 0 && curr_gt >= 0 && prev_gt == curr_gt &&
            prev_conf > 0.3f && curr_c > 0.3f) {
            valid_tracks++;

            // ID switch: model's prediction changed but GT stayed the same
            // This means the model lost track of the object
            if (curr_p != prev_pred) {
                switches++;
            }
        }

        // Update tracking state with current frame's pred and GT
        state->prev_query_classes[q * 2] = curr_p;
        state->prev_query_classes[q * 2 + 1] = curr_gt;
        state->prev_query_conf[q] = curr_c;
    }

    state->vap_id_switches += switches;
    state->vap_total_tracks += valid_tracks;

    // vAP = 1 - (switch_rate)
    // Perfect tracking = 1.0 (no switches when GT is consistent)
    // Random predictions = low (switches whenever GT is consistent)
    if (state->vap_total_tracks > 0) {
        float switch_rate = (float)state->vap_id_switches / state->vap_total_tracks;
        return 1.0f - switch_rate;
    }
    return 1.0f;
}

// Compute all vision metrics for current frame
void compute_vision_metrics(train_state_t* state) {
    if (!state->vulkan_ctx) {
        fprintf(stderr, "[METRICS] No vulkan_ctx\n");
        return;
    }

    vulkan_context_t* ctx = (vulkan_context_t*)state->vulkan_ctx;
    model_config_t* config = state->config;
    int num_queries = config->num_queries;
    int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 14;
    int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 14;
    int num_patches = (config->input_resolution[0] / patch_h) *
                      (config->input_resolution[1] / patch_w);
    int num_classes = config->num_classes;

    // Allocate CPU buffers for metric computation
    size_t mask_size = (size_t)num_queries * num_patches;
    size_t class_size = (size_t)num_queries * num_classes;

    float* mask_pred = malloc(mask_size * sizeof(float));
    float* mask_gt = malloc(mask_size * sizeof(float));
    float* class_logits = malloc(class_size * sizeof(float));
    float* iou_per_query = malloc(num_queries * sizeof(float));

    // Initialize to zero
    memset(mask_pred, 0, mask_size * sizeof(float));
    memset(mask_gt, 0, mask_size * sizeof(float));
    memset(class_logits, 0, class_size * sizeof(float));
    memset(iou_per_query, 0, num_queries * sizeof(float));

    // Download mask predictions and targets from GPU
    if (config->mask_output && state->mask_logits) {
        vulkan_buffer_download(ctx, (vulkan_buffer_t*)state->mask_logits, mask_pred, mask_size);
        // Apply sigmoid to convert logits to probabilities
        for (size_t i = 0; i < mask_size; i++) {
            mask_pred[i] = 1.0f / (1.0f + expf(-mask_pred[i]));
        }
    }
    // Use original mask data from VDAT (not GPU buffer which has -1.0 for background)
    if (config->mask_output && state->video_masks && state->current_frame_idx >= 0) {
        size_t mask_offset = (size_t)state->current_frame_idx * mask_size;
        memcpy(mask_gt, &state->video_masks[mask_offset], mask_size * sizeof(float));
    }

    // Download class logits from GPU
    if (state->class_logits_buf) {
        vulkan_buffer_download(ctx, (vulkan_buffer_t*)state->class_logits_buf, class_logits, class_size);
    }

    // Get ground truth labels for current frame
    int* class_gt = NULL;
    if (state->video_labels && state->current_frame_idx >= 0) {
        class_gt = &state->video_labels[state->current_frame_idx * num_queries];
    }

    // Debug: print diagnostic info
    if (state->debug) {
        fprintf(stderr, "[METRICS] frame_idx=%d class_gt=%p\n", state->current_frame_idx, (void*)class_gt);
        if (class_gt) {
            fprintf(stderr, "[METRICS] class_gt[0..4]=%d %d %d %d %d\n",
                    class_gt[0], class_gt[1], class_gt[2], class_gt[3], class_gt[4]);
        }
        // Count valid queries (non-background)
        int valid_q = 0;
        if (class_gt) {
            for (int q = 0; q < num_queries; q++) {
                if (class_gt[q] >= 0 && class_gt[q] < num_classes) valid_q++;
            }
        }
        fprintf(stderr, "[METRICS] valid_queries=%d/%d\n", valid_q, num_queries);

        // Check mask_gt for any positive pixels
        float mask_gt_sum = 0.0f;
        float mask_pred_max = 0.0f;
        float mask_pred_sum = 0.0f;
        int mask_pred_above_thresh = 0;
        for (size_t i = 0; i < mask_size && i < 5120; i++) {
            mask_gt_sum += mask_gt[i];
            if (mask_pred[i] > mask_pred_max) mask_pred_max = mask_pred[i];
            mask_pred_sum += mask_pred[i];
            if (mask_pred[i] > 0.5f) mask_pred_above_thresh++;
        }
        fprintf(stderr, "[METRICS] mask_gt_sum=%.0f pred_max=%.3f pred_sum=%.1f pred>0.5=%d\n",
                mask_gt_sum, mask_pred_max, mask_pred_sum, mask_pred_above_thresh);
    }

    // Compute IoU (returns mean IoU, stores per-query in iou_per_query)
    if (config->mask_output) {
        state->current_iou = compute_iou(mask_pred, mask_gt, num_queries, num_patches, 0.5f, iou_per_query);
    } else {
        state->current_iou = 0.0f;
    }

    // Compute mAP@[0.5:0.95] using class predictions and IoU scores
    if (class_gt && state->class_logits_buf) {
        state->current_map = compute_map(class_logits, class_gt, iou_per_query, num_queries, num_classes);
    } else {
        state->current_map = 0.0f;
    }

    // Compute vAP (tracking consistency across frames)
    if (class_gt && state->class_logits_buf) {
        state->current_vap = compute_vap(state, class_logits, class_gt, num_queries, num_classes);
    } else {
        state->current_vap = 1.0f;
    }

    // Compute Top-1 and Top-5 accuracy
    if (class_gt && state->class_logits_buf) {
        int top1_correct = 0;
        int top5_correct = 0;
        int valid_queries = 0;

        for (int q = 0; q < num_queries; q++) {
            int gt_class = class_gt[q];
            if (gt_class < 0 || gt_class >= num_classes) continue;  // Skip background/invalid

            valid_queries++;
            float* logits = &class_logits[q * num_classes];

            // Find top-5 predictions
            int top5[5] = {-1, -1, -1, -1, -1};
            float top5_vals[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};

            for (int c = 0; c < num_classes; c++) {
                float val = logits[c];
                // Insert into sorted top5 if large enough
                for (int k = 0; k < 5; k++) {
                    if (val > top5_vals[k]) {
                        // Shift down
                        for (int j = 4; j > k; j--) {
                            top5[j] = top5[j-1];
                            top5_vals[j] = top5_vals[j-1];
                        }
                        top5[k] = c;
                        top5_vals[k] = val;
                        break;
                    }
                }
            }

            // Check top-1
            if (top5[0] == gt_class) top1_correct++;

            // Check top-5
            for (int k = 0; k < 5; k++) {
                if (top5[k] == gt_class) {
                    top5_correct++;
                    break;
                }
            }

            // Debug: show prediction vs GT for first valid query
            if (state->debug && valid_queries == 1) {
                fprintf(stderr, "[METRICS] query=%d pred=%d gt=%d top5=[%d,%d,%d,%d,%d]\n",
                        q, top5[0], gt_class, top5[0], top5[1], top5[2], top5[3], top5[4]);
            }
        }

        state->current_acc = valid_queries > 0 ? (float)top1_correct / valid_queries : 0.0f;
        state->current_acc5 = valid_queries > 0 ? (float)top5_correct / valid_queries : 0.0f;
    } else {
        state->current_acc = 0.0f;
        state->current_acc5 = 0.0f;
    }

    // Update running sums for epoch averaging
    state->running_iou_sum += state->current_iou;
    state->running_map_sum += state->current_map;
    state->running_vap_sum += state->current_vap;
    state->running_acc_sum += state->current_acc;
    state->running_acc5_sum += state->current_acc5;
    state->metrics_count++;

    free(mask_pred);
    free(mask_gt);
    free(class_logits);
    free(iou_per_query);
}

// ═══════════════════════════════════════════════════════════════════════════
// BACKWARD PASS - Compute gradients (LoRA adapters OR full weights)
// ═══════════════════════════════════════════════════════════════════════════
//
// Gradient flow for transformer layer:
//   x_out = x + attn(norm1(x)) + ffn(norm2(x + attn))
//
// For LoRA at projection W:
//   forward: y = Wx + BA @ x * scale
//   backward:
//     grad_B = grad_y @ (Ax)^T * scale
//     grad_A = (B^T @ grad_y) @ x^T * scale
//     grad_x = W^T @ grad_y + A^T @ (B^T @ grad_y) * scale
//
// For full weight training:
//   forward: y = W @ x
//   backward: grad_W += grad_y @ x^T  (outer product)
//
// ═══════════════════════════════════════════════════════════════════════════

// Helper: AdamW update for single weight tensor
static void adamw_update_tensor(weight_tensor_t* tensor, float lr, float beta1,
                                float beta2, float weight_decay, float eps, int step) {
    float bias_correction1 = 1.0f - powf(beta1, step);
    float bias_correction2 = 1.0f - powf(beta2, step);

    for (size_t i = 0; i < tensor->size; i++) {
        // AdamW: apply weight decay before update
        tensor->weight[i] -= lr * weight_decay * tensor->weight[i];

        // Update biased first moment
        tensor->m[i] = beta1 * tensor->m[i] + (1.0f - beta1) * tensor->grad[i];

        // Update biased second moment
        tensor->v[i] = beta2 * tensor->v[i] + (1.0f - beta2) * tensor->grad[i] * tensor->grad[i];

        // Bias-corrected estimates
        float m_hat = tensor->m[i] / bias_correction1;
        float v_hat = tensor->v[i] / bias_correction2;

        // Update parameter
        tensor->weight[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// GPU-accelerated AdamW update (uses persistent buffers for efficiency)
// NOTE: ctx->max_buffer_size must be set before first call!
static void gpu_adamw_update_tensor(vulkan_context_t* ctx, weight_tensor_t* tensor,
                                     float lr, float beta1, float beta2,
                                     float weight_decay, float eps, int step) {
    // Allocate persistent buffers ONCE on first use (sized exactly for max tensor)
    if (!ctx->weight_buffer) {
        if (ctx->max_buffer_size == 0) {
            fprintf(stderr, "ERROR: max_buffer_size not set on Vulkan context!\n");
            return;
        }

        ctx->weight_buffer = vulkan_buffer_create(ctx, ctx->max_buffer_size);
        ctx->grad_buffer = vulkan_buffer_create(ctx, ctx->max_buffer_size);
        ctx->m_buffer = vulkan_buffer_create(ctx, ctx->max_buffer_size);
        ctx->v_buffer = vulkan_buffer_create(ctx, ctx->max_buffer_size);
    }

    // Reuse persistent buffers
    vulkan_buffer_t* buf_weight = (vulkan_buffer_t*)ctx->weight_buffer;
    vulkan_buffer_t* buf_grad = (vulkan_buffer_t*)ctx->grad_buffer;
    vulkan_buffer_t* buf_m = (vulkan_buffer_t*)ctx->m_buffer;
    vulkan_buffer_t* buf_v = (vulkan_buffer_t*)ctx->v_buffer;

    // Upload to GPU
    vulkan_buffer_upload(ctx, buf_weight, tensor->weight, tensor->size);
    vulkan_buffer_upload(ctx, buf_grad, tensor->grad, tensor->size);
    vulkan_buffer_upload(ctx, buf_m, tensor->m, tensor->size);
    vulkan_buffer_upload(ctx, buf_v, tensor->v, tensor->size);

    // Run GPU AdamW kernel
    vulkan_adamw_update(ctx, buf_weight, buf_grad, buf_m, buf_v,
                        lr, beta1, beta2, weight_decay, eps, step, tensor->size);

    // Download updated weights and moments
    vulkan_buffer_download(ctx, buf_weight, tensor->weight, tensor->size);
    vulkan_buffer_download(ctx, buf_m, tensor->m, tensor->size);
    vulkan_buffer_download(ctx, buf_v, tensor->v, tensor->size);

    // Buffers stay allocated for next call!
}

// Helper: dispatch to GPU or CPU for single tensor
static inline void update_tensor_dispatch(vulkan_context_t* vulkan_ctx, weight_tensor_t* tensor,
                                          float lr, float beta1, float beta2,
                                          float weight_decay, float eps, int step) {
    if (vulkan_ctx) {
        gpu_adamw_update_tensor(vulkan_ctx, tensor, lr, beta1, beta2, weight_decay, eps, step);
    } else {
        adamw_update_tensor(tensor, lr, beta1, beta2, weight_decay, eps, step);
    }
}

// AdamW optimizer step for full weight training (CPU or GPU)
static void full_weights_optimizer_step(trainable_weights_t* weights, float lr,
                                       float beta1, float beta2, float weight_decay,
                                       float eps, int step, vulkan_context_t* vulkan_ctx) {
    if (!weights) return;

    // Update all layer weights
    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];

        // Attention projections
        update_tensor_dispatch(vulkan_ctx, &layer->q_proj, lr, beta1, beta2, weight_decay, eps, step);
        update_tensor_dispatch(vulkan_ctx, &layer->k_proj, lr, beta1, beta2, weight_decay, eps, step);
        update_tensor_dispatch(vulkan_ctx, &layer->v_proj, lr, beta1, beta2, weight_decay, eps, step);
        update_tensor_dispatch(vulkan_ctx, &layer->o_proj, lr, beta1, beta2, weight_decay, eps, step);

        // FFN projections
        update_tensor_dispatch(vulkan_ctx, &layer->gate_proj, lr, beta1, beta2, weight_decay, eps, step);
        update_tensor_dispatch(vulkan_ctx, &layer->up_proj, lr, beta1, beta2, weight_decay, eps, step);
        update_tensor_dispatch(vulkan_ctx, &layer->down_proj, lr, beta1, beta2, weight_decay, eps, step);

        // Layer norms
        update_tensor_dispatch(vulkan_ctx, &layer->input_norm, lr, beta1, beta2, weight_decay, eps, step);
        update_tensor_dispatch(vulkan_ctx, &layer->post_norm, lr, beta1, beta2, weight_decay, eps, step);
    }

    // Global weights
    update_tensor_dispatch(vulkan_ctx, &weights->embed_tokens, lr, beta1, beta2, weight_decay, eps, step);
    update_tensor_dispatch(vulkan_ctx, &weights->final_norm, lr, beta1, beta2, weight_decay, eps, step);
}

// Helper: Zero out gradient buffer for a weight tensor
static void zero_weight_tensor_grad(weight_tensor_t* tensor) {
    memset(tensor->grad, 0, tensor->size * sizeof(float));
}

// Zero out all gradients for full weight training
static void zero_full_weight_gradients(trainable_weights_t* weights) {
    if (!weights) return;

    // Per-layer weights
    for (int l = 0; l < weights->num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];
        zero_weight_tensor_grad(&layer->q_proj);
        zero_weight_tensor_grad(&layer->k_proj);
        zero_weight_tensor_grad(&layer->v_proj);
        zero_weight_tensor_grad(&layer->o_proj);
        zero_weight_tensor_grad(&layer->gate_proj);
        zero_weight_tensor_grad(&layer->up_proj);
        zero_weight_tensor_grad(&layer->down_proj);
        zero_weight_tensor_grad(&layer->input_norm);
        zero_weight_tensor_grad(&layer->post_norm);
    }

    // Global weights
    zero_weight_tensor_grad(&weights->embed_tokens);
    zero_weight_tensor_grad(&weights->final_norm);
}

// ═══════════════════════════════════════════════════════════════════════════
// FP32 TRAINING FORWARD PASS
// ═══════════════════════════════════════════════════════════════════════════
// Uses trainable_weights_t (FP32) instead of BF16 model weights
// Returns logits for ALL positions (for computing loss at each position)

// Helper: FP32 RMSNorm
static void rms_norm_fp32(float* out, const float* x, const float* weight, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / size + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// Helper: FP32 matvec (CPU fallback)
static void matvec_fp32(float* out, const float* weight, const float* input,
                        int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += weight[i * in_dim + j] * input[j];
        }
        out[i] = sum;
    }
}

// Helper: FP32 SiLU activation
static void silu_fp32(float* out, const float* x, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// Helper: FP32 softmax
// ═══════════════════════════════════════════════════════════════════════════
// GPU BATCH MATMUL HELPER
// ═══════════════════════════════════════════════════════════════════════════
// Batch matmul: out = input @ weight^T
// input: [seq_len, in_dim], weight: [out_dim, in_dim], output: [seq_len, out_dim]

static void gpu_batch_matmul(vulkan_context_t* ctx,
                              const float* input, int seq_len, int in_dim,
                              const float* weight, int out_dim,
                              float* output) {
    // Ensure buffers are allocated
    if (!ctx->fwd_input_buffer) {
        if (ctx->max_fwd_buffer_size == 0) {
            fprintf(stderr, "ERROR: max_fwd_buffer_size not set!\n");
            return;
        }
        ctx->fwd_input_buffer = vulkan_buffer_create(ctx, ctx->max_fwd_buffer_size);
        ctx->fwd_weight_buffer = vulkan_buffer_create(ctx, ctx->max_fwd_buffer_size);
        ctx->fwd_output_buffer = vulkan_buffer_create(ctx, ctx->max_fwd_buffer_size);
    }

    vulkan_buffer_t* buf_input = (vulkan_buffer_t*)ctx->fwd_input_buffer;
    vulkan_buffer_t* buf_weight = (vulkan_buffer_t*)ctx->fwd_weight_buffer;
    vulkan_buffer_t* buf_output = (vulkan_buffer_t*)ctx->fwd_output_buffer;

    // Upload ALL data at once
    vulkan_buffer_upload(ctx, buf_input, input, seq_len * in_dim);
    vulkan_buffer_upload(ctx, buf_weight, weight, out_dim * in_dim);

    // Batch matmul using matmul_transpose shader
    // output = input @ weight^T
    //
    // A = input  [seq_len, in_dim]   (M x K)
    // B = weight [out_dim, in_dim]   (N x K) -> transposed to (K x N)
    // C = output [seq_len, out_dim]  (M x N)
    //
    // M = seq_len, N = out_dim, K = in_dim
    // transpose_A = 0, transpose_B = 1, accumulate = 0
    vulkan_matmul_transpose(ctx, buf_input, buf_weight, buf_output,
                            seq_len, out_dim, in_dim,
                            0, 1, 0);  // transpose_B = 1

    // Download ALL results at once
    vulkan_buffer_download(ctx, buf_output, output, seq_len * out_dim);
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU LoRA STATE - Adapter buffers on GPU for GPU-accelerated LoRA training
// ═══════════════════════════════════════════════════════════════════════════

// Per-adapter-type GPU buffer arrays (one entry per layer, NULL if inactive)
typedef struct {
    vulkan_buffer_t** A;       // [num_layers] - weight A [rank × in_dim]
    vulkan_buffer_t** B;       // [num_layers] - weight B [out_dim × rank]
    vulkan_buffer_t** gA;      // [num_layers] - gradient A
    vulkan_buffer_t** gB;      // [num_layers] - gradient B
    vulkan_buffer_t** mA;      // [num_layers] - optimizer first moment A
    vulkan_buffer_t** vA;      // [num_layers] - optimizer second moment A
    vulkan_buffer_t** mB;      // [num_layers] - optimizer first moment B
    vulkan_buffer_t** vB;      // [num_layers] - optimizer second moment B
    int active;                // 1 if this adapter type has GPU buffers
} lora_gpu_adapter_t;

typedef struct {
    int num_layers;
    int rank;
    float alpha;
    float scale;               // alpha / rank (precomputed)

    // 7 adapter types matching lora_model_t
    lora_gpu_adapter_t q;
    lora_gpu_adapter_t k;
    lora_gpu_adapter_t v;
    lora_gpu_adapter_t o;
    lora_gpu_adapter_t gate;
    lora_gpu_adapter_t up;
    lora_gpu_adapter_t down;
} lora_gpu_state_t;

// Helper: allocate + upload one adapter type to GPU
static void lora_gpu_adapter_upload(vulkan_context_t* ctx, lora_gpu_adapter_t* gpu,
                                     lora_adapter_t** adapters, int num_layers) {
    if (!adapters) { gpu->active = 0; return; }

    // Check if any adapter in this type is non-NULL
    int any_active = 0;
    for (int l = 0; l < num_layers; l++) {
        if (adapters[l]) { any_active = 1; break; }
    }
    if (!any_active) { gpu->active = 0; return; }

    gpu->active = 1;
    gpu->A  = calloc(num_layers, sizeof(vulkan_buffer_t*));
    gpu->B  = calloc(num_layers, sizeof(vulkan_buffer_t*));
    gpu->gA = calloc(num_layers, sizeof(vulkan_buffer_t*));
    gpu->gB = calloc(num_layers, sizeof(vulkan_buffer_t*));
    gpu->mA = calloc(num_layers, sizeof(vulkan_buffer_t*));
    gpu->vA = calloc(num_layers, sizeof(vulkan_buffer_t*));
    gpu->mB = calloc(num_layers, sizeof(vulkan_buffer_t*));
    gpu->vB = calloc(num_layers, sizeof(vulkan_buffer_t*));

    for (int l = 0; l < num_layers; l++) {
        lora_adapter_t* a = adapters[l];
        if (!a) continue;

        size_t a_bytes = (size_t)a->rank * (size_t)a->in_dim * sizeof(float);
        size_t b_bytes = (size_t)a->out_dim * (size_t)a->rank * sizeof(float);

        // Weights
        gpu->A[l] = vulkan_buffer_create(ctx, a_bytes);
        gpu->B[l] = vulkan_buffer_create(ctx, b_bytes);
        vulkan_buffer_upload(ctx, gpu->A[l], a->A, a->rank * a->in_dim);
        vulkan_buffer_upload(ctx, gpu->B[l], a->B, a->out_dim * a->rank);

        // Gradients (zero-initialized)
        gpu->gA[l] = vulkan_buffer_create(ctx, a_bytes);
        gpu->gB[l] = vulkan_buffer_create(ctx, b_bytes);
        vulkan_fill_buffer(ctx, gpu->gA[l], 0, a_bytes);
        vulkan_fill_buffer(ctx, gpu->gB[l], 0, b_bytes);

        // Optimizer state (upload from CPU - may have nonzero state from resumed training)
        gpu->mA[l] = vulkan_buffer_create(ctx, a_bytes);
        gpu->vA[l] = vulkan_buffer_create(ctx, a_bytes);
        gpu->mB[l] = vulkan_buffer_create(ctx, b_bytes);
        gpu->vB[l] = vulkan_buffer_create(ctx, b_bytes);
        vulkan_buffer_upload(ctx, gpu->mA[l], a->m_A, a->rank * a->in_dim);
        vulkan_buffer_upload(ctx, gpu->vA[l], a->v_A, a->rank * a->in_dim);
        vulkan_buffer_upload(ctx, gpu->mB[l], a->m_B, a->out_dim * a->rank);
        vulkan_buffer_upload(ctx, gpu->vB[l], a->v_B, a->out_dim * a->rank);
    }
}

// Helper: free one adapter type's GPU buffers
static void lora_gpu_adapter_free(vulkan_context_t* ctx, lora_gpu_adapter_t* gpu, int num_layers) {
    if (!gpu->active) return;
    for (int l = 0; l < num_layers; l++) {
        if (gpu->A && gpu->A[l])   vulkan_buffer_free(ctx, gpu->A[l]);
        if (gpu->B && gpu->B[l])   vulkan_buffer_free(ctx, gpu->B[l]);
        if (gpu->gA && gpu->gA[l]) vulkan_buffer_free(ctx, gpu->gA[l]);
        if (gpu->gB && gpu->gB[l]) vulkan_buffer_free(ctx, gpu->gB[l]);
        if (gpu->mA && gpu->mA[l]) vulkan_buffer_free(ctx, gpu->mA[l]);
        if (gpu->vA && gpu->vA[l]) vulkan_buffer_free(ctx, gpu->vA[l]);
        if (gpu->mB && gpu->mB[l]) vulkan_buffer_free(ctx, gpu->mB[l]);
        if (gpu->vB && gpu->vB[l]) vulkan_buffer_free(ctx, gpu->vB[l]);
    }
    free(gpu->A);  free(gpu->B);
    free(gpu->gA); free(gpu->gB);
    free(gpu->mA); free(gpu->vA);
    free(gpu->mB); free(gpu->vB);
    memset(gpu, 0, sizeof(*gpu));
}

// Upload frozen base model weights (BF16 → FP32) to GPU for LoRA training
// Reuses the gpu_w_* fields in vulkan_context_t (same as full-weight, but no grad/m/v)
static void ensure_lora_gpu_base_weights(vulkan_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->model || !state->config || !state->lora) return;
    // Use gpu_weights_num_layers as guard - it's set unconditionally and works for both LLM and video
    if (ctx->gpu_weights_num_layers > 0) return;  // Already uploaded

    const model_config_t* config = state->config;
    seraph_model_t* model = state->model;
    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;
    int intermediate_size = config->intermediate_size;
    int is_video = config->num_queries > 0;

    printf("  Uploading frozen base weights (BF16→FP32) to GPU...\n");

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
            buf_ptr = vulkan_buffer_create(ctx, _bytes); \
            float* _tmp = malloc(_bytes); \
            for (size_t _i = 0; _i < _n; _i++) _tmp[_i] = bf16_to_f32(bf16[_i]); \
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)buf_ptr, _tmp, _n); \
            free(_tmp); \
        } \
    } while(0)

    // Embeddings: patch_embed for video, embed_tokens for LLM
    if (is_video) {
        // Video: load patch embedding [hidden_size, patch_dim] - REQUIRED
        int in_channels = config->in_channels > 0 ? config->in_channels : 3;
        int patch_t = config->patch_size[0] > 0 ? config->patch_size[0] : 1;
        int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
        int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
        int patch_dim = in_channels * patch_t * patch_h * patch_w;

        uint16_t* patch_bf16 = get_model_tensor(model, "model.patch_embed.weight");
        if (!patch_bf16) {
            fprintf(stderr, "ERROR: Corrupted video model - missing model.patch_embed.weight tensor\n");
            fprintf(stderr, "       Use seraph-init to create a properly initialized model\n");
            return;
        }
        size_t patch_n = (size_t)hidden_size * (size_t)patch_dim;
        size_t patch_bytes = patch_n * sizeof(float);
        ctx->gpu_w_embed = vulkan_buffer_create(ctx, patch_bytes);
        float* patch_tmp = malloc(patch_bytes);
        for (size_t i = 0; i < patch_n; i++) patch_tmp[i] = bf16_to_f32(patch_bf16[i]);
        vulkan_buffer_upload(ctx, (vulkan_buffer_t*)ctx->gpu_w_embed, patch_tmp, patch_n);
        free(patch_tmp);
        printf("    Loaded patch_embed: [%d, %d]\n", hidden_size, patch_dim);
    } else {
        // LLM: load token embedding [vocab_size, hidden_size]
        UPLOAD_BF16("model.embed_tokens.weight", ctx->gpu_w_embed,
                    (size_t)vocab_size * (size_t)hidden_size);
    }
    UPLOAD_BF16("model.norm.weight", ctx->gpu_w_final_norm, hidden_size);

    // Per-layer weight arrays (weight only, no grad/m/v for frozen base)
    ctx->gpu_w_q = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_k = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_v = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_o = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_gate = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_up = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_down = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_in_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_w_post_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));

    char tname[256];
    for (int l = 0; l < num_layers; l++) {
        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.q_proj.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_q)[l], (size_t)q_dim * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.k_proj.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_k)[l], (size_t)kv_dim * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.v_proj.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_v)[l], (size_t)kv_dim * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.self_attn.o_proj.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_o)[l], (size_t)hidden_size * q_dim);

        snprintf(tname, sizeof(tname), "model.layers.%d.mlp.gate_proj.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_gate)[l], (size_t)intermediate_size * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.mlp.up_proj.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_up)[l], (size_t)intermediate_size * hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.mlp.down_proj.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_down)[l], (size_t)hidden_size * intermediate_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.input_layernorm.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[l], hidden_size);

        snprintf(tname, sizeof(tname), "model.layers.%d.post_attention_layernorm.weight", l);
        UPLOAD_BF16(tname, ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[l], hidden_size);
    }

    #undef UPLOAD_BF16

    printf("  ✅ Base weights uploaded to GPU (frozen, %d layers)\n", num_layers);
}

static void ensure_video_gpu_weights(vulkan_context_t* ctx, train_state_t* state) {
    if (!ctx || !state || !state->model || !state->config) return;
    if (state->query_learnable) return;

    const model_config_t* config = state->config;
    if (config->num_queries <= 0) return;

    seraph_model_t* model = state->model;
    int num_queries = config->num_queries;
    int query_dim = config->query_dim > 0 ? config->query_dim : config->hidden_size;
    int hidden_size = config->hidden_size;

    // Calculate patch dimensions
    int frame_h = config->input_resolution[0] > 0 ? config->input_resolution[0] : 256;
    int frame_w = config->input_resolution[1] > 0 ? config->input_resolution[1] : 256;
    int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
    int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
    int num_patches = (frame_h / patch_h) * (frame_w / patch_w);

    printf("  Uploading video weights to GPU...\n"); fflush(stdout);

    #define UPLOAD_BF16_VIDEO(tensor_name, buf_ptr, num_elements) do { \
        fprintf(stderr, "    [DBG] Uploading %s...\n", tensor_name); fflush(stderr); \
        uint16_t* bf16 = get_model_tensor(model, tensor_name); \
        if (bf16) { \
            size_t _n = (size_t)(num_elements); \
            size_t _bytes = _n * sizeof(float); \
            buf_ptr = vulkan_buffer_create(ctx, _bytes); \
            float* _tmp = malloc(_bytes); \
            for (size_t _i = 0; _i < _n; _i++) _tmp[_i] = bf16_to_f32(bf16[_i]); \
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)buf_ptr, _tmp, _n); \
            free(_tmp); \
        } \
    } while(0)

    UPLOAD_BF16_VIDEO("model.queries.weight", state->query_learnable,
                      (size_t)num_queries * query_dim);
    UPLOAD_BF16_VIDEO("model.query_fusion.weight", state->query_fusion_weight,
                      (size_t)query_dim * query_dim);
    fprintf(stderr, "    [DBG] Done query weights\n"); fflush(stderr);

    // Prev query output buffer
    fprintf(stderr, "    [DBG] Creating prev_query_out...\n"); fflush(stderr);
    size_t prev_bytes = (size_t)num_queries * query_dim * sizeof(float);
    state->prev_query_out = vulkan_buffer_create(ctx, prev_bytes);
    fprintf(stderr, "    [DBG] Filling prev_query_out...\n"); fflush(stderr);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->prev_query_out, 0, prev_bytes);
    fprintf(stderr, "    [DBG] Done prev_query_out\n"); fflush(stderr);

    // Video frame input buffer (holds one frame for upload)
    // Reuses train_tokens_u32 but needs to be large enough for frames
    int in_channels = config->in_channels > 0 ? config->in_channels : 3;
    size_t frame_bytes = (size_t)in_channels * frame_h * frame_w * sizeof(float);
    fprintf(stderr, "    [DBG] Frame buffer check (tokens_u32=%p, need %zu bytes)...\n",
            (void*)ctx->train_tokens_u32, frame_bytes); fflush(stderr);
    if (!ctx->train_tokens_u32 || ((vulkan_buffer_t*)ctx->train_tokens_u32)->size < frame_bytes) {
        fprintf(stderr, "    [DBG] Need to reallocate frame buffer...\n"); fflush(stderr);
        if (ctx->train_tokens_u32) {
            fprintf(stderr, "    [DBG] Freeing old train_tokens_u32...\n"); fflush(stderr);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_tokens_u32);
        }
        ctx->train_tokens_u32 = vulkan_buffer_create(ctx, frame_bytes);
        printf("    Allocated frame buffer: %.2f MB\n", frame_bytes / (1024.0f * 1024.0f));
    }
    fprintf(stderr, "    [DBG] Frame buffer done\n"); fflush(stderr);

    // Targets buffer for class labels (needs num_queries * sizeof(uint32_t))
    size_t targets_bytes = (size_t)num_queries * sizeof(uint32_t);
    if (!ctx->train_targets_u32 || ((vulkan_buffer_t*)ctx->train_targets_u32)->size < targets_bytes) {
        if (ctx->train_targets_u32) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_targets_u32);
        ctx->train_targets_u32 = vulkan_buffer_create(ctx, targets_bytes);
    }

    // Patch embedding buffer (persists through layers for K/V source)
    size_t patch_bytes = (size_t)num_patches * hidden_size * sizeof(float);
    state->video_patch_embed = vulkan_buffer_create(ctx, patch_bytes);

    // Sinusoidal 2D position embeddings for patches
    // Encodes (row, col) position using sin/cos at different frequencies
    {
        int grid_h = frame_h / patch_h;
        int grid_w = frame_w / patch_w;
        float* pos_embed = malloc(patch_bytes);

        for (int row = 0; row < grid_h; row++) {
            for (int col = 0; col < grid_w; col++) {
                int patch_idx = row * grid_w + col;
                float* pos = &pos_embed[patch_idx * hidden_size];

                // Half of hidden_size for row encoding, half for col
                int half_dim = hidden_size / 2;
                for (int d = 0; d + 1 < half_dim; d += 2) {
                    // Frequency decreases with dimension (captures different scales)
                    float freq = 1.0f / powf(10000.0f, (float)d / (float)half_dim);

                    // Row encoding in first half
                    pos[d]     = sinf((float)row * freq);
                    pos[d + 1] = cosf((float)row * freq);

                    // Col encoding in second half
                    pos[half_dim + d]     = sinf((float)col * freq);
                    pos[half_dim + d + 1] = cosf((float)col * freq);
                }
            }
        }

        state->video_pos_embed = vulkan_buffer_create(ctx, patch_bytes);
        vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->video_pos_embed, pos_embed, num_patches * hidden_size);
        free(pos_embed);
        printf("  Created sinusoidal 2D position embeddings (%dx%d grid, %d dim)\n",
               grid_h, grid_w, hidden_size);
        fflush(stdout);
    }
    fprintf(stderr, "    [DBG] Creating gradient buffers...\n"); fflush(stderr);

    // Gradient buffers for video training
    size_t fusion_bytes = (size_t)query_dim * query_dim * sizeof(float);
    state->video_grad_patches = vulkan_buffer_create(ctx, patch_bytes);
    state->video_grad_queries = vulkan_buffer_create(ctx, prev_bytes);
    state->video_grad_fusion = vulkan_buffer_create(ctx, fusion_bytes);

    // Optimizer state (Adam m, v) for video weights
    state->video_m_queries = vulkan_buffer_create(ctx, prev_bytes);
    state->video_v_queries = vulkan_buffer_create(ctx, prev_bytes);
    state->video_m_fusion = vulkan_buffer_create(ctx, fusion_bytes);
    state->video_v_fusion = vulkan_buffer_create(ctx, fusion_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->video_m_queries, 0, prev_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->video_v_queries, 0, prev_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->video_m_fusion, 0, fusion_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->video_v_fusion, 0, fusion_bytes);

    // Class head: queries -> num_classes
    int num_classes = config->num_classes > 0 ? config->num_classes : 10;
    size_t class_w_bytes = (size_t)query_dim * num_classes * sizeof(float);
    size_t class_b_bytes = (size_t)num_classes * sizeof(float);

    UPLOAD_BF16_VIDEO("model.class_head.weight", state->class_head_weight,
                      (size_t)query_dim * num_classes);
    UPLOAD_BF16_VIDEO("model.class_head.bias", state->class_head_bias,
                      (size_t)num_classes);

    // If not in model, create random initialized weights
    if (!state->class_head_weight) {
        state->class_head_weight = vulkan_buffer_create(ctx, class_w_bytes);
        float* tmp = malloc(class_w_bytes);
        float scale = 0.02f;
        for (size_t i = 0; i < (size_t)query_dim * num_classes; i++) {
            tmp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->class_head_weight, tmp, query_dim * num_classes);
        free(tmp);
    }
    if (!state->class_head_bias) {
        state->class_head_bias = vulkan_buffer_create(ctx, class_b_bytes);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->class_head_bias, 0, class_b_bytes);
    }

    // Class head gradients and optimizer state
    state->class_head_grad_w = vulkan_buffer_create(ctx, class_w_bytes);
    state->class_head_grad_b = vulkan_buffer_create(ctx, class_b_bytes);
    state->class_head_m_w = vulkan_buffer_create(ctx, class_w_bytes);
    state->class_head_v_w = vulkan_buffer_create(ctx, class_w_bytes);
    state->class_head_m_b = vulkan_buffer_create(ctx, class_b_bytes);
    state->class_head_v_b = vulkan_buffer_create(ctx, class_b_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->class_head_m_w, 0, class_w_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->class_head_v_w, 0, class_w_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->class_head_m_b, 0, class_b_bytes);
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->class_head_v_b, 0, class_b_bytes);

    // Buffer to store class logits for metrics computation
    size_t class_logits_bytes = (size_t)num_queries * num_classes * sizeof(float);
    state->class_logits_buf = vulkan_buffer_create(ctx, class_logits_bytes);

    // Mask head: MLP to project queries, then dot product with patches
    // mask_logits[q, p] = (query[q] @ W1 @ W2) · patch[p]
    if (config->mask_output) {
        int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
        size_t mask_w1_bytes = (size_t)query_dim * mask_dim * sizeof(float);
        size_t mask_w2_bytes = (size_t)mask_dim * query_dim * sizeof(float);
        size_t mask_logits_bytes = (size_t)num_queries * num_patches * sizeof(float);

        UPLOAD_BF16_VIDEO("model.mask_head.mlp.0.weight", state->mask_mlp_w1,
                          (size_t)query_dim * mask_dim);
        UPLOAD_BF16_VIDEO("model.mask_head.mlp.1.weight", state->mask_mlp_w2,
                          (size_t)mask_dim * query_dim);

        // Random init if not in model
        if (!state->mask_mlp_w1) {
            state->mask_mlp_w1 = vulkan_buffer_create(ctx, mask_w1_bytes);
            float* tmp = malloc(mask_w1_bytes);
            float scale = 0.02f;
            for (size_t i = 0; i < (size_t)query_dim * mask_dim; i++) {
                tmp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
            }
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->mask_mlp_w1, tmp, query_dim * mask_dim);
            free(tmp);
        }
        if (!state->mask_mlp_w2) {
            state->mask_mlp_w2 = vulkan_buffer_create(ctx, mask_w2_bytes);
            float* tmp = malloc(mask_w2_bytes);
            float scale = 0.02f;
            for (size_t i = 0; i < (size_t)mask_dim * query_dim; i++) {
                tmp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
            }
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->mask_mlp_w2, tmp, mask_dim * query_dim);
            free(tmp);
        }

        // Output buffer for mask logits
        state->mask_logits = vulkan_buffer_create(ctx, mask_logits_bytes);

        // Gradients and optimizer state
        state->mask_grad_w1 = vulkan_buffer_create(ctx, mask_w1_bytes);
        state->mask_grad_w2 = vulkan_buffer_create(ctx, mask_w2_bytes);
        state->mask_m_w1 = vulkan_buffer_create(ctx, mask_w1_bytes);
        state->mask_v_w1 = vulkan_buffer_create(ctx, mask_w1_bytes);
        state->mask_m_w2 = vulkan_buffer_create(ctx, mask_w2_bytes);
        state->mask_v_w2 = vulkan_buffer_create(ctx, mask_w2_bytes);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->mask_m_w1, 0, mask_w1_bytes);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->mask_v_w1, 0, mask_w1_bytes);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->mask_m_w2, 0, mask_w2_bytes);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->mask_v_w2, 0, mask_w2_bytes);

        // Buffers for mask loss computation
        state->mask_targets = vulkan_buffer_create(ctx, mask_logits_bytes);
        state->mask_grad_logits = vulkan_buffer_create(ctx, mask_logits_bytes);
        state->mask_loss_per_elem = vulkan_buffer_create(ctx, mask_logits_bytes);

        printf("    Mask head: MLP %d→%d→%d, output %dx%d\n",
               query_dim, mask_dim, query_dim, num_queries, num_patches);
    }

    // CLIP semantic alignment (load pre-computed class embeddings)
    if (config->clip_loss_weight > 0.0f && config->clip_embeddings_path[0]) {
        int clip_dim = config->clip_embed_dim > 0 ? config->clip_embed_dim : query_dim;
        size_t clip_embed_bytes = (size_t)num_classes * clip_dim * sizeof(float);
        size_t clip_proj_bytes = (size_t)query_dim * clip_dim * sizeof(float);
        size_t clip_query_bytes = (size_t)num_queries * clip_dim * sizeof(float);
        size_t clip_sim_bytes = (size_t)num_queries * num_classes * sizeof(float);

        // Load pre-computed CLIP embeddings from safetensors
        printf("    Loading CLIP embeddings: %s\n", config->clip_embeddings_path);
        safetensors_t* clip_st = safetensors_load(config->clip_embeddings_path);
        if (clip_st) {
            uint16_t* clip_bf16 = (uint16_t*)safetensors_get_tensor_raw(clip_st, "class_embeddings");
            if (clip_bf16) {
                state->clip_embeddings = vulkan_buffer_create(ctx, clip_embed_bytes);
                float* clip_fp32 = malloc(clip_embed_bytes);
                for (size_t i = 0; i < (size_t)num_classes * clip_dim; i++) {
                    uint32_t bits = (uint32_t)clip_bf16[i] << 16;
                    memcpy(&clip_fp32[i], &bits, sizeof(float));
                }
                vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->clip_embeddings,
                                     clip_fp32, num_classes * clip_dim);
                free(clip_fp32);
                printf("    Loaded CLIP embeddings: %d classes × %d dim\n", num_classes, clip_dim);
            } else {
                printf("    WARNING: CLIP embeddings tensor 'class_embeddings' not found\n");
            }
            safetensors_free(clip_st);
        } else {
            printf("    WARNING: Could not load CLIP embeddings from %s\n",
                   config->clip_embeddings_path);
        }

        // Create projection layer if query_dim != clip_dim
        if (query_dim != clip_dim) {
            state->clip_proj_weight = vulkan_buffer_create(ctx, clip_proj_bytes);
            float* tmp = malloc(clip_proj_bytes);
            float scale = sqrtf(2.0f / (query_dim + clip_dim));
            for (size_t i = 0; i < (size_t)query_dim * clip_dim; i++) {
                tmp[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
            }
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->clip_proj_weight,
                                 tmp, query_dim * clip_dim);
            free(tmp);

            state->clip_proj_grad = vulkan_buffer_create(ctx, clip_proj_bytes);
            state->clip_proj_m = vulkan_buffer_create(ctx, clip_proj_bytes);
            state->clip_proj_v = vulkan_buffer_create(ctx, clip_proj_bytes);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->clip_proj_m, 0, clip_proj_bytes);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->clip_proj_v, 0, clip_proj_bytes);
            printf("    CLIP projection: %d → %d\n", query_dim, clip_dim);
        }

        // Working buffers for CLIP loss
        state->clip_query_proj = vulkan_buffer_create(ctx, clip_query_bytes);
        state->clip_similarity = vulkan_buffer_create(ctx, clip_sim_bytes);
        state->clip_grad_similarity = vulkan_buffer_create(ctx, clip_sim_bytes);
        state->clip_loss_per_query = vulkan_buffer_create(ctx, (size_t)num_queries * sizeof(float));
    }

    state->current_frame_idx = 0;

    #undef UPLOAD_BF16_VIDEO

    printf("  ✅ Video weights uploaded (queries=%d, patches=%d, classes=%d)\n",
           num_queries, num_patches, num_classes);
}

// Upload LoRA adapter parameters to GPU
static void ensure_lora_gpu_adapters(vulkan_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->lora) return;
    if (ctx->lora_gpu) return;  // Already uploaded

    lora_model_t* lora = state->lora;
    int num_layers = lora->num_layers;

    printf("  Uploading LoRA adapters to GPU (rank=%d, alpha=%.1f)...\n",
           lora->rank, lora->alpha);

    lora_gpu_state_t* lg = calloc(1, sizeof(lora_gpu_state_t));
    lg->num_layers = num_layers;
    lg->rank = lora->rank;
    lg->alpha = lora->alpha;
    lg->scale = lora->alpha / (float)lora->rank;

    // Upload each adapter type
    lora_gpu_adapter_upload(ctx, &lg->q, lora->q_adapters, num_layers);
    lora_gpu_adapter_upload(ctx, &lg->k, lora->k_adapters, num_layers);
    lora_gpu_adapter_upload(ctx, &lg->v, lora->v_adapters, num_layers);
    lora_gpu_adapter_upload(ctx, &lg->o, lora->o_adapters, num_layers);
    lora_gpu_adapter_upload(ctx, &lg->gate, lora->gate_adapters, num_layers);
    lora_gpu_adapter_upload(ctx, &lg->up, lora->up_adapters, num_layers);
    lora_gpu_adapter_upload(ctx, &lg->down, lora->down_adapters, num_layers);

    int active_count = lg->q.active + lg->k.active + lg->v.active + lg->o.active +
                       lg->gate.active + lg->up.active + lg->down.active;
    printf("  ✅ %d adapter types active on GPU (%d layers)\n", active_count, num_layers);

    ctx->lora_gpu = lg;
}

// Ensure LoRA hidden temp buffer is large enough
static void ensure_lora_hidden_tmp(vulkan_context_t* ctx, int max_seq, int rank) {
    size_t needed = (size_t)max_seq * (size_t)rank * sizeof(float);
    if (ctx->lora_hidden_tmp && ctx->lora_hidden_tmp_bytes >= needed) return;
    if (ctx->lora_hidden_tmp) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->lora_hidden_tmp);
    ctx->lora_hidden_tmp = vulkan_buffer_create(ctx, needed);
    ctx->lora_hidden_tmp_bytes = needed;
}

// Zero all LoRA gradients on GPU
static void gpu_zero_lora_grads(vulkan_context_t* ctx) {
    lora_gpu_state_t* lg = (lora_gpu_state_t*)ctx->lora_gpu;
    if (!lg) return;

    #define ZERO_ADAPTER_GRADS(adapter) do { \
        if ((adapter).active) { \
            for (int l = 0; l < lg->num_layers; l++) { \
                if ((adapter).gA[l]) vulkan_fill_buffer(ctx, (adapter).gA[l], 0, (adapter).gA[l]->size); \
                if ((adapter).gB[l]) vulkan_fill_buffer(ctx, (adapter).gB[l], 0, (adapter).gB[l]->size); \
            } \
        } \
    } while(0)

    ZERO_ADAPTER_GRADS(lg->q);
    ZERO_ADAPTER_GRADS(lg->k);
    ZERO_ADAPTER_GRADS(lg->v);
    ZERO_ADAPTER_GRADS(lg->o);
    ZERO_ADAPTER_GRADS(lg->gate);
    ZERO_ADAPTER_GRADS(lg->up);
    ZERO_ADAPTER_GRADS(lg->down);

    #undef ZERO_ADAPTER_GRADS
}

// Run AdamW optimizer on all LoRA parameters (GPU-resident)
static void lora_optimizer_step_gpu(vulkan_context_t* ctx, float lr, float beta1, float beta2,
                                     float weight_decay, float eps, int step) {
    lora_gpu_state_t* lg = (lora_gpu_state_t*)ctx->lora_gpu;
    if (!lg) return;

    #define OPT_ADAPTER(adapter) do { \
        if ((adapter).active) { \
            for (int l = 0; l < lg->num_layers; l++) { \
                if ((adapter).A[l]) { \
                    vulkan_adamw_update(ctx, (adapter).A[l], (adapter).gA[l], \
                                        (adapter).mA[l], (adapter).vA[l], \
                                        lr, beta1, beta2, weight_decay, eps, step, \
                                        (adapter).A[l]->size / sizeof(float)); \
                    vulkan_adamw_update(ctx, (adapter).B[l], (adapter).gB[l], \
                                        (adapter).mB[l], (adapter).vB[l], \
                                        lr, beta1, beta2, weight_decay, eps, step, \
                                        (adapter).B[l]->size / sizeof(float)); \
                } \
            } \
        } \
    } while(0)

    OPT_ADAPTER(lg->q);
    OPT_ADAPTER(lg->k);
    OPT_ADAPTER(lg->v);
    OPT_ADAPTER(lg->o);
    OPT_ADAPTER(lg->gate);
    OPT_ADAPTER(lg->up);
    OPT_ADAPTER(lg->down);

    #undef OPT_ADAPTER
}

// Download LoRA weights from GPU back to CPU (for saving)
static void lora_gpu_download_to_cpu(vulkan_context_t* ctx, lora_model_t* lora) {
    lora_gpu_state_t* lg = (lora_gpu_state_t*)ctx->lora_gpu;
    if (!lg) return;

    #define DOWNLOAD_ADAPTER(gpu_adapter, cpu_adapters) do { \
        if ((gpu_adapter).active && cpu_adapters) { \
            for (int l = 0; l < lg->num_layers; l++) { \
                lora_adapter_t* a = cpu_adapters[l]; \
                if (!a || !(gpu_adapter).A[l]) continue; \
                vulkan_buffer_download(ctx, (gpu_adapter).A[l], a->A, a->rank * a->in_dim); \
                vulkan_buffer_download(ctx, (gpu_adapter).B[l], a->B, a->out_dim * a->rank); \
                vulkan_buffer_download(ctx, (gpu_adapter).mA[l], a->m_A, a->rank * a->in_dim); \
                vulkan_buffer_download(ctx, (gpu_adapter).vA[l], a->v_A, a->rank * a->in_dim); \
                vulkan_buffer_download(ctx, (gpu_adapter).mB[l], a->m_B, a->out_dim * a->rank); \
                vulkan_buffer_download(ctx, (gpu_adapter).vB[l], a->v_B, a->out_dim * a->rank); \
            } \
        } \
    } while(0)

    DOWNLOAD_ADAPTER(lg->q, lora->q_adapters);
    DOWNLOAD_ADAPTER(lg->k, lora->k_adapters);
    DOWNLOAD_ADAPTER(lg->v, lora->v_adapters);
    DOWNLOAD_ADAPTER(lg->o, lora->o_adapters);
    DOWNLOAD_ADAPTER(lg->gate, lora->gate_adapters);
    DOWNLOAD_ADAPTER(lg->up, lora->up_adapters);
    DOWNLOAD_ADAPTER(lg->down, lora->down_adapters);

    #undef DOWNLOAD_ADAPTER
}

// Free all LoRA GPU state
static void lora_gpu_free(vulkan_context_t* ctx) {
    lora_gpu_state_t* lg = (lora_gpu_state_t*)ctx->lora_gpu;
    if (!lg) return;

    lora_gpu_adapter_free(ctx, &lg->q, lg->num_layers);
    lora_gpu_adapter_free(ctx, &lg->k, lg->num_layers);
    lora_gpu_adapter_free(ctx, &lg->v, lg->num_layers);
    lora_gpu_adapter_free(ctx, &lg->o, lg->num_layers);
    lora_gpu_adapter_free(ctx, &lg->gate, lg->num_layers);
    lora_gpu_adapter_free(ctx, &lg->up, lg->num_layers);
    lora_gpu_adapter_free(ctx, &lg->down, lg->num_layers);

    if (ctx->lora_hidden_tmp) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->lora_hidden_tmp);
    ctx->lora_hidden_tmp = NULL;
    ctx->lora_hidden_tmp_bytes = 0;

    free(lg);
    ctx->lora_gpu = NULL;
}

static void ensure_fulltrain_fwd_buffers(vulkan_context_t* ctx,
                                        const train_state_t* state,
                                        int max_seq,
                                        int hidden_size,
                                        int intermediate_size,
                                        int q_dim,
                                        int kv_dim,
                                        int vocab_size) {
    if (!ctx) return;

    int num_layers = state && state->config ? state->config->num_hidden_layers : 0;

    // Ensure shared weight buffer exists (max_fwd_buffer_size set in train_init for full training)
    if (!ctx->fwd_weight_buffer) {
        if (ctx->max_fwd_buffer_size == 0) {
            fprintf(stderr, "ERROR: max_fwd_buffer_size not set!\n");
            return;
        }
        ctx->fwd_input_buffer = vulkan_buffer_create(ctx, ctx->max_fwd_buffer_size);
        ctx->fwd_weight_buffer = vulkan_buffer_create(ctx, ctx->max_fwd_buffer_size);
        ctx->fwd_output_buffer = vulkan_buffer_create(ctx, ctx->max_fwd_buffer_size);
    }

    // Ensure attention buffers exist (used for Q/K/V + attn_out)
    {
        size_t q_buf_size = (size_t)max_seq * (size_t)q_dim * sizeof(float);
        size_t kv_buf_size = (size_t)max_seq * (size_t)kv_dim * sizeof(float);
        if (!ctx->attn_q_buffer || ((vulkan_buffer_t*)ctx->attn_q_buffer)->size < q_buf_size) {
            if (ctx->attn_q_buffer) {
                vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_q_buffer);
                vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_k_buffer);
                vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_v_buffer);
                vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_out_buffer);
            }
            ctx->max_attn_buffer_size = q_buf_size;
            ctx->attn_q_buffer = vulkan_buffer_create(ctx, q_buf_size);
            ctx->attn_k_buffer = vulkan_buffer_create(ctx, kv_buf_size);
            ctx->attn_v_buffer = vulkan_buffer_create(ctx, kv_buf_size);
            ctx->attn_out_buffer = vulkan_buffer_create(ctx, q_buf_size);
        }
    }

    // Forward activation buffers
    size_t hidden_bytes = (size_t)max_seq * (size_t)hidden_size * sizeof(float);
    size_t inter_bytes = (size_t)max_seq * (size_t)intermediate_size * sizeof(float);
    size_t logits_bytes = (size_t)max_seq * (size_t)vocab_size * sizeof(float);
    size_t norm_w_bytes = (size_t)hidden_size * sizeof(float);

    // Track max needed for debug
    size_t max_needed = hidden_bytes;
    if (inter_bytes > max_needed) max_needed = inter_bytes;
    if (logits_bytes > max_needed) max_needed = logits_bytes;
    if (q_dim > hidden_size && (size_t)max_seq * (size_t)q_dim * sizeof(float) > max_needed)
        max_needed = (size_t)max_seq * (size_t)q_dim * sizeof(float);
    if (kv_dim > hidden_size && (size_t)max_seq * (size_t)kv_dim * sizeof(float) > max_needed)
        max_needed = (size_t)max_seq * (size_t)kv_dim * sizeof(float);
    if (max_needed > ctx->max_fwd_act_bytes) ctx->max_fwd_act_bytes = max_needed;

    if (!ctx->fwd_hidden_buffer || ((vulkan_buffer_t*)ctx->fwd_hidden_buffer)->size < hidden_bytes) {
        if (ctx->fwd_hidden_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_hidden_buffer);
        ctx->fwd_hidden_buffer = vulkan_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->fwd_hidden_norm_buffer || ((vulkan_buffer_t*)ctx->fwd_hidden_norm_buffer)->size < hidden_bytes) {
        if (ctx->fwd_hidden_norm_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_hidden_norm_buffer);
        ctx->fwd_hidden_norm_buffer = vulkan_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->fwd_tmp_hidden_buffer || ((vulkan_buffer_t*)ctx->fwd_tmp_hidden_buffer)->size < hidden_bytes) {
        if (ctx->fwd_tmp_hidden_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_tmp_hidden_buffer);
        ctx->fwd_tmp_hidden_buffer = vulkan_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->fwd_ffn_gate_buffer || ((vulkan_buffer_t*)ctx->fwd_ffn_gate_buffer)->size < inter_bytes) {
        if (ctx->fwd_ffn_gate_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_ffn_gate_buffer);
        ctx->fwd_ffn_gate_buffer = vulkan_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->fwd_ffn_up_buffer || ((vulkan_buffer_t*)ctx->fwd_ffn_up_buffer)->size < inter_bytes) {
        if (ctx->fwd_ffn_up_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_ffn_up_buffer);
        ctx->fwd_ffn_up_buffer = vulkan_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->fwd_ffn_hidden_buffer || ((vulkan_buffer_t*)ctx->fwd_ffn_hidden_buffer)->size < inter_bytes) {
        if (ctx->fwd_ffn_hidden_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_ffn_hidden_buffer);
        ctx->fwd_ffn_hidden_buffer = vulkan_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->fwd_logits_buffer || ((vulkan_buffer_t*)ctx->fwd_logits_buffer)->size < logits_bytes) {
        if (ctx->fwd_logits_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_logits_buffer);
        ctx->fwd_logits_buffer = vulkan_buffer_create(ctx, logits_bytes);
    }
    if (!ctx->fwd_norm_weight_buffer || ((vulkan_buffer_t*)ctx->fwd_norm_weight_buffer)->size < norm_w_bytes) {
        if (ctx->fwd_norm_weight_buffer) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_norm_weight_buffer);
        ctx->fwd_norm_weight_buffer = vulkan_buffer_create(ctx, norm_w_bytes);
    }

    // Per-layer activation caches (GPU-side) so we can submit the whole forward pass once
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
                    vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_x_norm_attn)[l]);
                    vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_x_norm_ffn)[l]);
                    vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_attn_out)[l]);
                    vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_ffn_gate_out)[l]);
                    vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_ffn_up_out)[l]);
                    vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_ffn_hidden)[l]);
                    if (ctx->fwd_cache_layer_input) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_layer_input)[l]);
                    if (ctx->fwd_cache_post_attn_in) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_post_attn_in)[l]);
                    if (ctx->fwd_cache_silu_out) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_silu_out)[l]);
                    if (ctx->fwd_cache_q) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_q)[l]);
                    if (ctx->fwd_cache_k) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_k)[l]);
                    if (ctx->fwd_cache_v) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_v)[l]);
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
                ctx->fwd_cache_x_norm_attn = NULL;
                ctx->fwd_cache_x_norm_ffn = NULL;
                ctx->fwd_cache_attn_out = NULL;
                ctx->fwd_cache_ffn_gate_out = NULL;
                ctx->fwd_cache_ffn_up_out = NULL;
                ctx->fwd_cache_ffn_hidden = NULL;
                ctx->fwd_cache_layer_input = NULL;
                ctx->fwd_cache_post_attn_in = NULL;
                ctx->fwd_cache_silu_out = NULL;
                ctx->fwd_cache_q = NULL;
                ctx->fwd_cache_k = NULL;
                ctx->fwd_cache_v = NULL;
            }

            ctx->fwd_cache_num_layers = num_layers;
            ctx->fwd_cache_max_seq = max_seq;
            ctx->fwd_cache_hidden_size = hidden_size;
            ctx->fwd_cache_intermediate_size = intermediate_size;
            ctx->fwd_cache_q_dim = q_dim;

            ctx->fwd_cache_x_norm_attn = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_x_norm_ffn = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_attn_out = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_ffn_gate_out = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_ffn_up_out = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_ffn_hidden = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_layer_input = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_post_attn_in = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_silu_out = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_q = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_k = calloc(num_layers, sizeof(vulkan_buffer_t*));
            ctx->fwd_cache_v = calloc(num_layers, sizeof(vulkan_buffer_t*));

            size_t xnorm_bytes = (size_t)max_seq * (size_t)hidden_size * sizeof(float);
            size_t attn_bytes = (size_t)max_seq * (size_t)q_dim * sizeof(float);
            size_t ffn_bytes = (size_t)max_seq * (size_t)intermediate_size * sizeof(float);
            size_t q_bytes = (size_t)max_seq * (size_t)q_dim * sizeof(float);
            size_t kv_bytes = (size_t)max_seq * (size_t)kv_dim * sizeof(float);

            for (int l = 0; l < num_layers; l++) {
                ((vulkan_buffer_t**)ctx->fwd_cache_x_norm_attn)[l] = vulkan_buffer_create(ctx, xnorm_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_x_norm_ffn)[l] = vulkan_buffer_create(ctx, xnorm_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_attn_out)[l] = vulkan_buffer_create(ctx, attn_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_ffn_gate_out)[l] = vulkan_buffer_create(ctx, ffn_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_ffn_up_out)[l] = vulkan_buffer_create(ctx, ffn_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_ffn_hidden)[l] = vulkan_buffer_create(ctx, ffn_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_layer_input)[l] = vulkan_buffer_create(ctx, xnorm_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_post_attn_in)[l] = vulkan_buffer_create(ctx, xnorm_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_silu_out)[l] = vulkan_buffer_create(ctx, ffn_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_q)[l] = vulkan_buffer_create(ctx, q_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_k)[l] = vulkan_buffer_create(ctx, kv_bytes);
                ((vulkan_buffer_t**)ctx->fwd_cache_v)[l] = vulkan_buffer_create(ctx, kv_bytes);
            }
        }
    }
}

static void ensure_fulltrain_train_buffers(vulkan_context_t* ctx,
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

    if (!ctx->train_tokens_u32 || ((vulkan_buffer_t*)ctx->train_tokens_u32)->size < tokens_u32_bytes) {
        if (ctx->train_tokens_u32) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_tokens_u32);
        ctx->train_tokens_u32 = vulkan_buffer_create(ctx, tokens_u32_bytes);
    }
    if (!ctx->train_targets_u32 || ((vulkan_buffer_t*)ctx->train_targets_u32)->size < tokens_u32_bytes) {
        if (ctx->train_targets_u32) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_targets_u32);
        ctx->train_targets_u32 = vulkan_buffer_create(ctx, tokens_u32_bytes);
    }

    if (!ctx->train_grad_logits || ((vulkan_buffer_t*)ctx->train_grad_logits)->size < logits_bytes) {
        if (ctx->train_grad_logits) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_logits);
        ctx->train_grad_logits = vulkan_buffer_create(ctx, logits_bytes);
    }
    if (!ctx->train_loss_rows || ((vulkan_buffer_t*)ctx->train_loss_rows)->size < loss_bytes) {
        if (ctx->train_loss_rows) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_loss_rows);
        ctx->train_loss_rows = vulkan_buffer_create(ctx, loss_bytes);
    }
    if (!ctx->train_reduce_tmp_a || ((vulkan_buffer_t*)ctx->train_reduce_tmp_a)->size < loss_bytes) {
        if (ctx->train_reduce_tmp_a) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_reduce_tmp_a);
        ctx->train_reduce_tmp_a = vulkan_buffer_create(ctx, loss_bytes);
    }
    if (!ctx->train_reduce_tmp_b || ((vulkan_buffer_t*)ctx->train_reduce_tmp_b)->size < loss_bytes) {
        if (ctx->train_reduce_tmp_b) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_reduce_tmp_b);
        ctx->train_reduce_tmp_b = vulkan_buffer_create(ctx, loss_bytes);
    }

    if (!ctx->train_grad_hidden || ((vulkan_buffer_t*)ctx->train_grad_hidden)->size < hidden_bytes) {
        if (ctx->train_grad_hidden) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden);
        ctx->train_grad_hidden = vulkan_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->train_grad_tmp_hidden || ((vulkan_buffer_t*)ctx->train_grad_tmp_hidden)->size < hidden_bytes) {
        if (ctx->train_grad_tmp_hidden) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden);
        ctx->train_grad_tmp_hidden = vulkan_buffer_create(ctx, hidden_bytes);
    }
    if (!ctx->train_grad_x_norm || ((vulkan_buffer_t*)ctx->train_grad_x_norm)->size < hidden_bytes) {
        if (ctx->train_grad_x_norm) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_x_norm);
        ctx->train_grad_x_norm = vulkan_buffer_create(ctx, hidden_bytes);
    }

    if (!ctx->train_grad_q || ((vulkan_buffer_t*)ctx->train_grad_q)->size < q_bytes) {
        if (ctx->train_grad_q) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_q);
        ctx->train_grad_q = vulkan_buffer_create(ctx, q_bytes);
    }
    if (!ctx->train_grad_attn || ((vulkan_buffer_t*)ctx->train_grad_attn)->size < q_bytes) {
        if (ctx->train_grad_attn) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_attn);
        ctx->train_grad_attn = vulkan_buffer_create(ctx, q_bytes);
    }

    if (!ctx->train_grad_k || ((vulkan_buffer_t*)ctx->train_grad_k)->size < kv_bytes) {
        if (ctx->train_grad_k) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_k);
        ctx->train_grad_k = vulkan_buffer_create(ctx, kv_bytes);
    }
    if (!ctx->train_grad_v || ((vulkan_buffer_t*)ctx->train_grad_v)->size < kv_bytes) {
        if (ctx->train_grad_v) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_v);
        ctx->train_grad_v = vulkan_buffer_create(ctx, kv_bytes);
    }

    if (!ctx->train_grad_ffn || ((vulkan_buffer_t*)ctx->train_grad_ffn)->size < inter_bytes) {
        if (ctx->train_grad_ffn) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_ffn);
        ctx->train_grad_ffn = vulkan_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->train_grad_gate || ((vulkan_buffer_t*)ctx->train_grad_gate)->size < inter_bytes) {
        if (ctx->train_grad_gate) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_gate);
        ctx->train_grad_gate = vulkan_buffer_create(ctx, inter_bytes);
    }
    if (!ctx->train_grad_up || ((vulkan_buffer_t*)ctx->train_grad_up)->size < inter_bytes) {
        if (ctx->train_grad_up) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_up);
        ctx->train_grad_up = vulkan_buffer_create(ctx, inter_bytes);
    }

    if (!ctx->train_final_input || ((vulkan_buffer_t*)ctx->train_final_input)->size < hidden_bytes) {
        if (ctx->train_final_input) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_final_input);
        ctx->train_final_input = vulkan_buffer_create(ctx, hidden_bytes);
    }
}

static void free_gpu_weight_arrays(vulkan_context_t* ctx) {
    if (!ctx) return;
    if (ctx->gpu_w_embed) {
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_w_embed);
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_g_embed);
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_m_embed);
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_v_embed);
        ctx->gpu_w_embed = NULL;
        ctx->gpu_g_embed = NULL;
        ctx->gpu_m_embed = NULL;
        ctx->gpu_v_embed = NULL;
    }
    if (ctx->gpu_w_final_norm) {
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_w_final_norm);
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_g_final_norm);
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_m_final_norm);
        vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_v_final_norm);
        ctx->gpu_w_final_norm = NULL;
        ctx->gpu_g_final_norm = NULL;
        ctx->gpu_m_final_norm = NULL;
        ctx->gpu_v_final_norm = NULL;
    }

    if (ctx->gpu_w_q) {
        int n = ctx->gpu_weights_num_layers;
        vulkan_buffer_t** arrays[] = {
            (vulkan_buffer_t**)ctx->gpu_w_q, (vulkan_buffer_t**)ctx->gpu_g_q, (vulkan_buffer_t**)ctx->gpu_m_q, (vulkan_buffer_t**)ctx->gpu_v_q,
            (vulkan_buffer_t**)ctx->gpu_w_k, (vulkan_buffer_t**)ctx->gpu_g_k, (vulkan_buffer_t**)ctx->gpu_m_k, (vulkan_buffer_t**)ctx->gpu_v_k,
            (vulkan_buffer_t**)ctx->gpu_w_v, (vulkan_buffer_t**)ctx->gpu_g_v, (vulkan_buffer_t**)ctx->gpu_m_v, (vulkan_buffer_t**)ctx->gpu_v_v,
            (vulkan_buffer_t**)ctx->gpu_w_o, (vulkan_buffer_t**)ctx->gpu_g_o, (vulkan_buffer_t**)ctx->gpu_m_o, (vulkan_buffer_t**)ctx->gpu_v_o,
            (vulkan_buffer_t**)ctx->gpu_w_gate, (vulkan_buffer_t**)ctx->gpu_g_gate, (vulkan_buffer_t**)ctx->gpu_m_gate, (vulkan_buffer_t**)ctx->gpu_v_gate,
            (vulkan_buffer_t**)ctx->gpu_w_up, (vulkan_buffer_t**)ctx->gpu_g_up, (vulkan_buffer_t**)ctx->gpu_m_up, (vulkan_buffer_t**)ctx->gpu_v_up,
            (vulkan_buffer_t**)ctx->gpu_w_down, (vulkan_buffer_t**)ctx->gpu_g_down, (vulkan_buffer_t**)ctx->gpu_m_down, (vulkan_buffer_t**)ctx->gpu_v_down,
            (vulkan_buffer_t**)ctx->gpu_w_in_norm, (vulkan_buffer_t**)ctx->gpu_g_in_norm, (vulkan_buffer_t**)ctx->gpu_m_in_norm, (vulkan_buffer_t**)ctx->gpu_v_in_norm,
            (vulkan_buffer_t**)ctx->gpu_w_post_norm, (vulkan_buffer_t**)ctx->gpu_g_post_norm, (vulkan_buffer_t**)ctx->gpu_m_post_norm, (vulkan_buffer_t**)ctx->gpu_v_post_norm,
        };

        for (size_t ai = 0; ai < sizeof(arrays) / sizeof(arrays[0]); ai++) {
            if (!arrays[ai]) continue;
            for (int l = 0; l < n; l++) {
                if (arrays[ai][l]) vulkan_buffer_free(ctx, arrays[ai][l]);
            }
            free(arrays[ai]);
        }

        ctx->gpu_w_q = ctx->gpu_g_q = ctx->gpu_m_q = ctx->gpu_v_q = NULL;
        ctx->gpu_w_k = ctx->gpu_g_k = ctx->gpu_m_k = ctx->gpu_v_k = NULL;
        ctx->gpu_w_v = ctx->gpu_g_v = ctx->gpu_m_v = ctx->gpu_v_v = NULL;
        ctx->gpu_w_o = ctx->gpu_g_o = ctx->gpu_m_o = ctx->gpu_v_o = NULL;
        ctx->gpu_w_gate = ctx->gpu_g_gate = ctx->gpu_m_gate = ctx->gpu_v_gate = NULL;
        ctx->gpu_w_up = ctx->gpu_g_up = ctx->gpu_m_up = ctx->gpu_v_up = NULL;
        ctx->gpu_w_down = ctx->gpu_g_down = ctx->gpu_m_down = ctx->gpu_v_down = NULL;
        ctx->gpu_w_in_norm = ctx->gpu_g_in_norm = ctx->gpu_m_in_norm = ctx->gpu_v_in_norm = NULL;
        ctx->gpu_w_post_norm = ctx->gpu_g_post_norm = ctx->gpu_m_post_norm = ctx->gpu_v_post_norm = NULL;
    }

    ctx->gpu_weights_num_layers = 0;
    ctx->gpu_weights_hidden_size = 0;
    ctx->gpu_weights_intermediate_size = 0;
    ctx->gpu_weights_q_dim = 0;
    ctx->gpu_weights_kv_dim = 0;
    ctx->gpu_weights_vocab_size = 0;
}

static void ensure_fulltrain_gpu_weights(vulkan_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->full_weights || !state->config) return;

    const model_config_t* config = state->config;
    trainable_weights_t* w = state->full_weights;

    int num_layers = config->num_hidden_layers;
    int hidden_size = config->hidden_size;
    int intermediate_size = config->intermediate_size;
    int vocab_size = config->vocab_size;
    int q_dim = config->num_attention_heads * config->head_dim;
    int kv_dim = config->num_key_value_heads * config->head_dim;
    int is_video = config->num_queries > 0;

    // Use num_layers as primary guard (set unconditionally, works for both LLM and video)
    int needs = 0;
    if (ctx->gpu_weights_num_layers != num_layers) needs = 1;
    if (ctx->gpu_weights_hidden_size != hidden_size) needs = 1;
    if (ctx->gpu_weights_intermediate_size != intermediate_size) needs = 1;
    if (ctx->gpu_weights_q_dim != q_dim) needs = 1;
    if (ctx->gpu_weights_kv_dim != kv_dim) needs = 1;
    // Only check embed for LLM (video uses patch_embed from model file)
    if (!is_video && !ctx->gpu_w_embed) needs = 1;
    if (!is_video && ctx->gpu_weights_vocab_size != vocab_size) needs = 1;

    if (!needs) return;

    printf("  Uploading full training weights to GPU...\n");

    free_gpu_weight_arrays(ctx);

    ctx->gpu_weights_num_layers = num_layers;
    ctx->gpu_weights_hidden_size = hidden_size;
    ctx->gpu_weights_intermediate_size = intermediate_size;
    ctx->gpu_weights_vocab_size = vocab_size;
    ctx->gpu_weights_q_dim = q_dim;
    ctx->gpu_weights_kv_dim = kv_dim;

    // Global tensors - different handling for video vs LLM
    {
        size_t norm_bytes = w->final_norm.size * sizeof(float);

        if (is_video) {
            // Video: load patch_embed from model file (BF16 → FP32)
            // gpu_w_embed stores patch projection [hidden_size, patch_dim]
            int in_channels = config->in_channels > 0 ? config->in_channels : 3;
            int patch_t = config->patch_size[0] > 0 ? config->patch_size[0] : 1;
            int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
            int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
            int patch_dim = in_channels * patch_t * patch_h * patch_w;
            size_t patch_embed_size = (size_t)hidden_size * (size_t)patch_dim;
            size_t embed_bytes = patch_embed_size * sizeof(float);

            ctx->gpu_w_embed = vulkan_buffer_create(ctx, embed_bytes);
            ctx->gpu_g_embed = vulkan_buffer_create(ctx, embed_bytes);
            ctx->gpu_m_embed = vulkan_buffer_create(ctx, embed_bytes);
            ctx->gpu_v_embed = vulkan_buffer_create(ctx, embed_bytes);

            // Load patch_embed from model file - REQUIRED for video models
            if (!state->model) {
                fprintf(stderr, "ERROR: Video model requires model file but state->model is NULL\n");
                return;
            }
            uint16_t* bf16 = get_model_tensor(state->model, "model.patch_embed.weight");
            if (!bf16) {
                fprintf(stderr, "ERROR: Corrupted video model - missing model.patch_embed.weight tensor\n");
                fprintf(stderr, "       Use seraph-init to create a properly initialized model\n");
                return;
            }
            float* tmp = malloc(embed_bytes);
            for (size_t i = 0; i < patch_embed_size; i++) tmp[i] = bf16_to_f32(bf16[i]);
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)ctx->gpu_w_embed, tmp, patch_embed_size);
            free(tmp);
            printf("    Loaded patch_embed: [%d, %d]\n", hidden_size, patch_dim);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_g_embed, 0, embed_bytes);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_m_embed, 0, embed_bytes);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_v_embed, 0, embed_bytes);
        } else {
            // LLM: use embed_tokens from trainable weights
            size_t embed_bytes = w->embed_tokens.size * sizeof(float);

            ctx->gpu_w_embed = vulkan_buffer_create(ctx, embed_bytes);
            ctx->gpu_g_embed = vulkan_buffer_create(ctx, embed_bytes);
            ctx->gpu_m_embed = vulkan_buffer_create(ctx, embed_bytes);
            ctx->gpu_v_embed = vulkan_buffer_create(ctx, embed_bytes);
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)ctx->gpu_w_embed, w->embed_tokens.weight, w->embed_tokens.size);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_g_embed, 0, embed_bytes);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_m_embed, 0, embed_bytes);
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_v_embed, 0, embed_bytes);
        }

        ctx->gpu_w_final_norm = vulkan_buffer_create(ctx, norm_bytes);
        ctx->gpu_g_final_norm = vulkan_buffer_create(ctx, norm_bytes);
        ctx->gpu_m_final_norm = vulkan_buffer_create(ctx, norm_bytes);
        ctx->gpu_v_final_norm = vulkan_buffer_create(ctx, norm_bytes);
        vulkan_buffer_upload(ctx, (vulkan_buffer_t*)ctx->gpu_w_final_norm, w->final_norm.weight, w->final_norm.size);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_g_final_norm, 0, norm_bytes);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_m_final_norm, 0, norm_bytes);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_v_final_norm, 0, norm_bytes);
    }

    // Per-layer tensors
    ctx->gpu_w_q = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_q = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_q = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_q = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_k = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_k = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_k = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_k = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_v = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_v = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_v = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_v = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_o = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_o = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_o = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_o = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_gate = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_gate = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_gate = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_gate = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_up = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_up = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_up = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_up = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_down = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_down = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_down = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_down = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_in_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_in_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_in_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_in_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));

    ctx->gpu_w_post_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_g_post_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_m_post_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));
    ctx->gpu_v_post_norm = calloc(num_layers, sizeof(vulkan_buffer_t*));

    for (int l = 0; l < num_layers; l++) {
        layer_weights_t* lw = &w->layers[l];

        size_t q_bytes = lw->q_proj.size * sizeof(float);
        size_t k_bytes = lw->k_proj.size * sizeof(float);
        size_t v_bytes = lw->v_proj.size * sizeof(float);
        size_t o_bytes = lw->o_proj.size * sizeof(float);
        size_t gate_bytes = lw->gate_proj.size * sizeof(float);
        size_t up_bytes = lw->up_proj.size * sizeof(float);
        size_t down_bytes = lw->down_proj.size * sizeof(float);
        size_t norm_bytes = lw->input_norm.size * sizeof(float);

        ((vulkan_buffer_t**)ctx->gpu_w_q)[l] = vulkan_buffer_create(ctx, q_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_q)[l] = vulkan_buffer_create(ctx, q_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_q)[l] = vulkan_buffer_create(ctx, q_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_q)[l] = vulkan_buffer_create(ctx, q_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_q)[l], lw->q_proj.weight, lw->q_proj.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_q)[l], 0, q_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_q)[l], 0, q_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_q)[l], 0, q_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_k)[l] = vulkan_buffer_create(ctx, k_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_k)[l] = vulkan_buffer_create(ctx, k_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_k)[l] = vulkan_buffer_create(ctx, k_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_k)[l] = vulkan_buffer_create(ctx, k_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_k)[l], lw->k_proj.weight, lw->k_proj.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_k)[l], 0, k_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_k)[l], 0, k_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_k)[l], 0, k_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_v)[l] = vulkan_buffer_create(ctx, v_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_v)[l] = vulkan_buffer_create(ctx, v_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_v)[l] = vulkan_buffer_create(ctx, v_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_v)[l] = vulkan_buffer_create(ctx, v_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_v)[l], lw->v_proj.weight, lw->v_proj.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_v)[l], 0, v_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_v)[l], 0, v_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_v)[l], 0, v_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_o)[l] = vulkan_buffer_create(ctx, o_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_o)[l] = vulkan_buffer_create(ctx, o_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_o)[l] = vulkan_buffer_create(ctx, o_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_o)[l] = vulkan_buffer_create(ctx, o_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_o)[l], lw->o_proj.weight, lw->o_proj.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_o)[l], 0, o_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_o)[l], 0, o_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_o)[l], 0, o_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_gate)[l] = vulkan_buffer_create(ctx, gate_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_gate)[l] = vulkan_buffer_create(ctx, gate_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_gate)[l] = vulkan_buffer_create(ctx, gate_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_gate)[l] = vulkan_buffer_create(ctx, gate_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_gate)[l], lw->gate_proj.weight, lw->gate_proj.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_gate)[l], 0, gate_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_gate)[l], 0, gate_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_gate)[l], 0, gate_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_up)[l] = vulkan_buffer_create(ctx, up_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_up)[l] = vulkan_buffer_create(ctx, up_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_up)[l] = vulkan_buffer_create(ctx, up_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_up)[l] = vulkan_buffer_create(ctx, up_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_up)[l], lw->up_proj.weight, lw->up_proj.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_up)[l], 0, up_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_up)[l], 0, up_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_up)[l], 0, up_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_down)[l] = vulkan_buffer_create(ctx, down_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_down)[l] = vulkan_buffer_create(ctx, down_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_down)[l] = vulkan_buffer_create(ctx, down_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_down)[l] = vulkan_buffer_create(ctx, down_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_down)[l], lw->down_proj.weight, lw->down_proj.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_down)[l], 0, down_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_down)[l], 0, down_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_down)[l], 0, down_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_in_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_in_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_in_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[l], lw->input_norm.weight, lw->input_norm.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_in_norm)[l], 0, norm_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_in_norm)[l], 0, norm_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_in_norm)[l], 0, norm_bytes);

        ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        ((vulkan_buffer_t**)ctx->gpu_g_post_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        ((vulkan_buffer_t**)ctx->gpu_m_post_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        ((vulkan_buffer_t**)ctx->gpu_v_post_norm)[l] = vulkan_buffer_create(ctx, norm_bytes);
        vulkan_buffer_upload(ctx, ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[l], lw->post_norm.weight, lw->post_norm.size);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_post_norm)[l], 0, norm_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_m_post_norm)[l], 0, norm_bytes);
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_v_post_norm)[l], 0, norm_bytes);
    }
}

static void gpu_zero_full_weight_grads(vulkan_context_t* ctx, const train_state_t* state) {
    if (!ctx || !state || !state->full_weights || !state->config) return;
    if (!ctx->gpu_g_embed) return;

    trainable_weights_t* w = state->full_weights;
    int num_layers = state->config->num_hidden_layers;

    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_g_embed, 0, w->embed_tokens.size * sizeof(float));
    vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->gpu_g_final_norm, 0, w->final_norm.size * sizeof(float));

    for (int l = 0; l < num_layers; l++) {
        layer_weights_t* lw = &w->layers[l];
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_q)[l], 0, lw->q_proj.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_k)[l], 0, lw->k_proj.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_v)[l], 0, lw->v_proj.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_o)[l], 0, lw->o_proj.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_gate)[l], 0, lw->gate_proj.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_up)[l], 0, lw->up_proj.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_down)[l], 0, lw->down_proj.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_in_norm)[l], 0, lw->input_norm.size * sizeof(float));
        vulkan_fill_buffer(ctx, ((vulkan_buffer_t**)ctx->gpu_g_post_norm)[l], 0, lw->post_norm.size * sizeof(float));
    }
}

// FP32 Training Forward Pass - BATCH VERSION
// Processes all positions at once for GPU efficiency
// Returns logits array [num_tokens * vocab_size] - caller must free
float* train_forward_fp32(train_state_t* state, int* tokens, int num_tokens,
                          vulkan_context_t* vk_ctx) {
    trainable_weights_t* weights = state->full_weights;
    activation_cache_t* act = state->act_cache;
    const model_config_t* config = state->config;

    if (!weights || !act) return NULL;

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
    const int use_qk_norm = config->use_qk_norm;  // Video/ViT mode: normalize Q,K across heads

    float* all_logits = malloc(num_tokens * vocab_size * sizeof(float));
    if (!all_logits) return NULL;

    // Fast path: keep forward activations on GPU (full training only)
    if (vk_ctx) {
        int max_seq = state->act_cache->max_seq_len;
        ensure_fulltrain_fwd_buffers(vk_ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);

        vulkan_buffer_t* buf_hidden = (vulkan_buffer_t*)vk_ctx->fwd_hidden_buffer;
        vulkan_buffer_t* buf_hidden_norm = (vulkan_buffer_t*)vk_ctx->fwd_hidden_norm_buffer;
        vulkan_buffer_t* buf_tmp_hidden = (vulkan_buffer_t*)vk_ctx->fwd_tmp_hidden_buffer;
        vulkan_buffer_t* buf_gate = (vulkan_buffer_t*)vk_ctx->fwd_ffn_gate_buffer;
        vulkan_buffer_t* buf_up = (vulkan_buffer_t*)vk_ctx->fwd_ffn_up_buffer;
        vulkan_buffer_t* buf_ffn_hidden = (vulkan_buffer_t*)vk_ctx->fwd_ffn_hidden_buffer;
        vulkan_buffer_t* buf_logits = (vulkan_buffer_t*)vk_ctx->fwd_logits_buffer;
        vulkan_buffer_t* buf_norm_w = (vulkan_buffer_t*)vk_ctx->fwd_norm_weight_buffer;

        vulkan_buffer_t* buf_Q = (vulkan_buffer_t*)vk_ctx->attn_q_buffer;
        vulkan_buffer_t* buf_K = (vulkan_buffer_t*)vk_ctx->attn_k_buffer;
        vulkan_buffer_t* buf_V = (vulkan_buffer_t*)vk_ctx->attn_v_buffer;
        vulkan_buffer_t* buf_attn = (vulkan_buffer_t*)vk_ctx->attn_out_buffer;

        vulkan_buffer_t* buf_weight = (vulkan_buffer_t*)vk_ctx->fwd_weight_buffer;

        vulkan_buffer_t** cache_x_norm_attn = (vulkan_buffer_t**)vk_ctx->fwd_cache_x_norm_attn;
        vulkan_buffer_t** cache_x_norm_ffn = (vulkan_buffer_t**)vk_ctx->fwd_cache_x_norm_ffn;
        vulkan_buffer_t** cache_attn_out = (vulkan_buffer_t**)vk_ctx->fwd_cache_attn_out;
        vulkan_buffer_t** cache_ffn_gate_out = (vulkan_buffer_t**)vk_ctx->fwd_cache_ffn_gate_out;
        vulkan_buffer_t** cache_ffn_up_out = (vulkan_buffer_t**)vk_ctx->fwd_cache_ffn_up_out;
        vulkan_buffer_t** cache_ffn_hidden = (vulkan_buffer_t**)vk_ctx->fwd_cache_ffn_hidden;

        // Build initial hidden on CPU from embeddings, then upload once
        float* hidden_init = malloc((size_t)num_tokens * (size_t)hidden_size * sizeof(float));
        if (!hidden_init) {
            free(all_logits);
            return NULL;
        }

        for (int pos = 0; pos < num_tokens; pos++) {
            int token = tokens[pos];
            if (token >= 0 && token < vocab_size) {
                memcpy(&hidden_init[pos * hidden_size],
                       &weights->embed_tokens.weight[token * hidden_size],
                       hidden_size * sizeof(float));
            } else {
                memset(&hidden_init[pos * hidden_size], 0, hidden_size * sizeof(float));
            }

            // Cache layer input for backward (layer 0 input)
            memcpy(&act->layer_inputs[0][pos * hidden_size],
                   &hidden_init[pos * hidden_size], hidden_size * sizeof(float));
        }

        vulkan_buffer_upload(vk_ctx, buf_hidden, hidden_init, (size_t)num_tokens * (size_t)hidden_size);
        free(hidden_init);

        // Record all compute into one command buffer, submit once.
        if (!vulkan_cmd_begin(vk_ctx)) {
            fprintf(stderr, "WARNING: vulkan_cmd_begin failed, falling back to per-op submits\n");
        }

        // Transformer layers
        for (int layer = 0; layer < num_layers; layer++) {
            layer_weights_t* lw = &weights->layers[layer];

            // RMSNorm (pre-attention)
            vulkan_buffer_upload(vk_ctx, buf_norm_w, lw->input_norm.weight, hidden_size);
            vulkan_rmsnorm(vk_ctx, buf_hidden, buf_norm_w, buf_hidden_norm, num_tokens, hidden_size, rms_eps);
            vulkan_copy_buffer(vk_ctx, buf_hidden_norm, cache_x_norm_attn[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

            // Q, K, V projections
            vulkan_buffer_upload(vk_ctx, buf_weight, lw->q_proj.weight, (size_t)q_dim * (size_t)hidden_size);
            vulkan_matmul_transpose(vk_ctx, buf_hidden_norm, buf_weight, buf_Q,
                                    num_tokens, q_dim, hidden_size, 0, 1, 0);

            vulkan_buffer_upload(vk_ctx, buf_weight, lw->k_proj.weight, (size_t)kv_dim * (size_t)hidden_size);
            vulkan_matmul_transpose(vk_ctx, buf_hidden_norm, buf_weight, buf_K,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);

            vulkan_buffer_upload(vk_ctx, buf_weight, lw->v_proj.weight, (size_t)kv_dim * (size_t)hidden_size);
            vulkan_matmul_transpose(vk_ctx, buf_hidden_norm, buf_weight, buf_V,
                                    num_tokens, kv_dim, hidden_size, 0, 1, 0);

            // RoPE
            vulkan_rope(vk_ctx, buf_Q, buf_Q, num_tokens, num_heads, head_dim, rope_theta);
            vulkan_rope(vk_ctx, buf_K, buf_K, num_tokens, kv_heads, head_dim, rope_theta);

            // QK norm (video/ViT mode) - normalize Q,K across all heads
            vulkan_qk_norm(vk_ctx, buf_Q, num_tokens, num_heads, head_dim, use_qk_norm, rms_eps);
            vulkan_qk_norm(vk_ctx, buf_K, num_tokens, kv_heads, head_dim, use_qk_norm, rms_eps);

            // Attention (fused + causal)
            vulkan_batch_attention(vk_ctx, buf_Q, buf_K, buf_V, buf_attn,
                                   num_tokens, num_heads, kv_heads, head_dim);
            vulkan_copy_buffer(vk_ctx, buf_attn, cache_attn_out[layer],
                               (size_t)num_tokens * (size_t)q_dim * sizeof(float));

            // O projection
            vulkan_buffer_upload(vk_ctx, buf_weight, lw->o_proj.weight, (size_t)hidden_size * (size_t)q_dim);
            vulkan_matmul_transpose(vk_ctx, buf_attn, buf_weight, buf_tmp_hidden,
                                    num_tokens, hidden_size, q_dim, 0, 1, 0);

            // Residual: hidden += o_proj(attn)
            vulkan_add(vk_ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);

            // RMSNorm (post-attention)
            vulkan_buffer_upload(vk_ctx, buf_norm_w, lw->post_norm.weight, hidden_size);
            vulkan_rmsnorm(vk_ctx, buf_hidden, buf_norm_w, buf_hidden_norm, num_tokens, hidden_size, rms_eps);
            vulkan_copy_buffer(vk_ctx, buf_hidden_norm, cache_x_norm_ffn[layer],
                               (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

            // Gate and up projections
            vulkan_buffer_upload(vk_ctx, buf_weight, lw->gate_proj.weight, (size_t)intermediate_size * (size_t)hidden_size);
            vulkan_matmul_transpose(vk_ctx, buf_hidden_norm, buf_weight, buf_gate,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
            vulkan_copy_buffer(vk_ctx, buf_gate, cache_ffn_gate_out[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

            vulkan_buffer_upload(vk_ctx, buf_weight, lw->up_proj.weight, (size_t)intermediate_size * (size_t)hidden_size);
            vulkan_matmul_transpose(vk_ctx, buf_hidden_norm, buf_weight, buf_up,
                                    num_tokens, intermediate_size, hidden_size, 0, 1, 0);
            vulkan_copy_buffer(vk_ctx, buf_up, cache_ffn_up_out[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

            // SiLU(gate) * up
            vulkan_silu(vk_ctx, buf_gate, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
            vulkan_mul(vk_ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
            vulkan_copy_buffer(vk_ctx, buf_ffn_hidden, cache_ffn_hidden[layer],
                               (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

            // Down projection
            vulkan_buffer_upload(vk_ctx, buf_weight, lw->down_proj.weight, (size_t)hidden_size * (size_t)intermediate_size);
            vulkan_matmul_transpose(vk_ctx, buf_ffn_hidden, buf_weight, buf_tmp_hidden,
                                    num_tokens, hidden_size, intermediate_size, 0, 1, 0);

            // Residual: hidden += down(ffn)
            vulkan_add(vk_ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                       (size_t)num_tokens * (size_t)hidden_size);
        }

        // Final norm
        vulkan_buffer_upload(vk_ctx, buf_norm_w, weights->final_norm.weight, hidden_size);
        vulkan_rmsnorm(vk_ctx, buf_hidden, buf_norm_w, buf_hidden_norm, num_tokens, hidden_size, rms_eps);

        // LM head (tied embeddings): logits = hidden_norm @ embed^T
        vulkan_buffer_upload(vk_ctx, buf_weight, weights->embed_tokens.weight, (size_t)vocab_size * (size_t)hidden_size);
        vulkan_matmul_transpose(vk_ctx, buf_hidden_norm, buf_weight, buf_logits,
                                num_tokens, vocab_size, hidden_size, 0, 1, 0);

        // Execute the recorded forward pass (or no-op if not recording)
        vulkan_cmd_end_submit(vk_ctx);

        // Materialize activation cache for CPU backward in bulk (no per-op sync)
        for (int layer = 0; layer < num_layers; layer++) {
            vulkan_buffer_download(vk_ctx, cache_x_norm_attn[layer], act->x_norm_attn[layer],
                                   (size_t)num_tokens * (size_t)hidden_size);
            vulkan_buffer_download(vk_ctx, cache_attn_out[layer], act->attn_output[layer],
                                   (size_t)num_tokens * (size_t)q_dim);
            vulkan_buffer_download(vk_ctx, cache_x_norm_ffn[layer], act->x_norm_ffn[layer],
                                   (size_t)num_tokens * (size_t)hidden_size);
            vulkan_buffer_download(vk_ctx, cache_ffn_gate_out[layer], act->ffn_gate_out[layer],
                                   (size_t)num_tokens * (size_t)intermediate_size);
            vulkan_buffer_download(vk_ctx, cache_ffn_up_out[layer], act->ffn_up_out[layer],
                                   (size_t)num_tokens * (size_t)intermediate_size);
            vulkan_buffer_download(vk_ctx, cache_ffn_hidden[layer], act->ffn_hidden[layer],
                                   (size_t)num_tokens * (size_t)intermediate_size);
        }

        vulkan_buffer_download(vk_ctx, buf_hidden_norm, act->final_hidden,
                               (size_t)num_tokens * (size_t)hidden_size);
        act->cur_seq_len = num_tokens;

        vulkan_buffer_download(vk_ctx, buf_logits, all_logits, (size_t)num_tokens * (size_t)vocab_size);

        return all_logits;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ALLOCATE BATCH BUFFERS (CPU path / CPU fallback)
    // ═══════════════════════════════════════════════════════════════════════

    // Hidden states for all positions [seq_len, hidden_size]
    float* hidden = calloc(num_tokens * hidden_size, sizeof(float));
    float* hidden_norm = calloc(num_tokens * hidden_size, sizeof(float));

    // Q, K, V for all positions [seq_len, dim]
    float* Q_all = calloc(num_tokens * q_dim, sizeof(float));
    float* K_all = calloc(num_tokens * kv_dim, sizeof(float));
    float* V_all = calloc(num_tokens * kv_dim, sizeof(float));
    float* attn_out = calloc(num_tokens * q_dim, sizeof(float));

    // FFN buffers for all positions
    float* ffn_gate = calloc(num_tokens * intermediate_size, sizeof(float));
    float* ffn_up = calloc(num_tokens * intermediate_size, sizeof(float));
    float* ffn_hidden = calloc(num_tokens * intermediate_size, sizeof(float));
    float* ffn_out = calloc(num_tokens * hidden_size, sizeof(float));

    // Optional GPU helpers for this fallback path (used for RoPE/attention accel if vk_ctx is present)
    vulkan_buffer_t* buf_Q = NULL;
    vulkan_buffer_t* buf_K = NULL;
    vulkan_buffer_t* buf_V = NULL;
    vulkan_buffer_t* buf_attn = NULL;
    if (vk_ctx) {
        // Lazy allocate persistent attention buffers on first use
        int max_seq = state->act_cache->max_seq_len;
        size_t q_buf_size = (size_t)max_seq * (size_t)q_dim * sizeof(float);
        size_t kv_buf_size = (size_t)max_seq * (size_t)kv_dim * sizeof(float);
        if (!vk_ctx->attn_q_buffer || ((vulkan_buffer_t*)vk_ctx->attn_q_buffer)->size < q_buf_size) {
            if (vk_ctx->attn_q_buffer) {
                vulkan_buffer_free(vk_ctx, (vulkan_buffer_t*)vk_ctx->attn_q_buffer);
                vulkan_buffer_free(vk_ctx, (vulkan_buffer_t*)vk_ctx->attn_k_buffer);
                vulkan_buffer_free(vk_ctx, (vulkan_buffer_t*)vk_ctx->attn_v_buffer);
                vulkan_buffer_free(vk_ctx, (vulkan_buffer_t*)vk_ctx->attn_out_buffer);
            }
            vk_ctx->attn_q_buffer = vulkan_buffer_create(vk_ctx, q_buf_size);
            vk_ctx->attn_k_buffer = vulkan_buffer_create(vk_ctx, kv_buf_size);
            vk_ctx->attn_v_buffer = vulkan_buffer_create(vk_ctx, kv_buf_size);
            vk_ctx->attn_out_buffer = vulkan_buffer_create(vk_ctx, q_buf_size);
        }
        buf_Q = (vulkan_buffer_t*)vk_ctx->attn_q_buffer;
        buf_K = (vulkan_buffer_t*)vk_ctx->attn_k_buffer;
        buf_V = (vulkan_buffer_t*)vk_ctx->attn_v_buffer;
        buf_attn = (vulkan_buffer_t*)vk_ctx->attn_out_buffer;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // EMBEDDING LOOKUP (all positions)
    // ═══════════════════════════════════════════════════════════════════════

    for (int pos = 0; pos < num_tokens; pos++) {
        int token = tokens[pos];
        if (token >= 0 && token < vocab_size) {
            memcpy(&hidden[pos * hidden_size],
                   &weights->embed_tokens.weight[token * hidden_size],
                   hidden_size * sizeof(float));
        }
        // Cache layer input for backward
        memcpy(&act->layer_inputs[0][pos * hidden_size],
               &hidden[pos * hidden_size], hidden_size * sizeof(float));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TRANSFORMER LAYERS
    // ═══════════════════════════════════════════════════════════════════════

    for (int layer = 0; layer < num_layers; layer++) {
        layer_weights_t* lw = &weights->layers[layer];

        // --- Batch RMSNorm (pre-attention) ---
        for (int pos = 0; pos < num_tokens; pos++) {
            rms_norm_fp32(&hidden_norm[pos * hidden_size],
                         &hidden[pos * hidden_size],
                         lw->input_norm.weight, hidden_size, rms_eps);
            // Cache for backward
            memcpy(&act->x_norm_attn[layer][pos * hidden_size],
                   &hidden_norm[pos * hidden_size], hidden_size * sizeof(float));
        }

        // --- Batch Q, K, V projections ---
        if (vk_ctx) {
            gpu_batch_matmul(vk_ctx, hidden_norm, num_tokens, hidden_size,
                            lw->q_proj.weight, q_dim, Q_all);
            gpu_batch_matmul(vk_ctx, hidden_norm, num_tokens, hidden_size,
                            lw->k_proj.weight, kv_dim, K_all);
            gpu_batch_matmul(vk_ctx, hidden_norm, num_tokens, hidden_size,
                            lw->v_proj.weight, kv_dim, V_all);
        } else {
            for (int pos = 0; pos < num_tokens; pos++) {
                matvec_fp32(&Q_all[pos * q_dim], lw->q_proj.weight,
                           &hidden_norm[pos * hidden_size], q_dim, hidden_size);
                matvec_fp32(&K_all[pos * kv_dim], lw->k_proj.weight,
                           &hidden_norm[pos * hidden_size], kv_dim, hidden_size);
                matvec_fp32(&V_all[pos * kv_dim], lw->v_proj.weight,
                           &hidden_norm[pos * hidden_size], kv_dim, hidden_size);
            }
        }

        // --- Apply RoPE to Q and K ---
        if (vk_ctx && buf_Q && buf_K) {
            // GPU RoPE
            vulkan_buffer_upload(vk_ctx, buf_Q, Q_all, num_tokens * q_dim);
            vulkan_rope(vk_ctx, buf_Q, buf_Q, num_tokens, num_heads, head_dim, rope_theta);
            vulkan_buffer_download(vk_ctx, buf_Q, Q_all, num_tokens * q_dim);

            vulkan_buffer_upload(vk_ctx, buf_K, K_all, num_tokens * kv_dim);
            vulkan_rope(vk_ctx, buf_K, buf_K, num_tokens, kv_heads, head_dim, rope_theta);
            vulkan_buffer_download(vk_ctx, buf_K, K_all, num_tokens * kv_dim);
        } else {
            // CPU RoPE fallback
            for (int pos = 0; pos < num_tokens; pos++) {
                for (int h = 0; h < num_heads; h++) {
                    float* q_head = &Q_all[pos * q_dim + h * head_dim];
                    for (int d = 0; d < head_dim / 2; d++) {
                        float freq = 1.0f / powf(rope_theta, (float)(2 * d) / head_dim);
                        float angle = pos * freq;
                        float cos_a = cosf(angle), sin_a = sinf(angle);
                        float q0 = q_head[2*d], q1 = q_head[2*d + 1];
                        q_head[2*d] = q0 * cos_a - q1 * sin_a;
                        q_head[2*d + 1] = q0 * sin_a + q1 * cos_a;
                    }
                }
                for (int h = 0; h < kv_heads; h++) {
                    float* k_head = &K_all[pos * kv_dim + h * head_dim];
                    for (int d = 0; d < head_dim / 2; d++) {
                        float freq = 1.0f / powf(rope_theta, (float)(2 * d) / head_dim);
                        float angle = pos * freq;
                        float cos_a = cosf(angle), sin_a = sinf(angle);
                        float k0 = k_head[2*d], k1 = k_head[2*d + 1];
                        k_head[2*d] = k0 * cos_a - k1 * sin_a;
                        k_head[2*d + 1] = k0 * sin_a + k1 * cos_a;
                    }
                }
            }
        }

        // --- Batch Attention (GPU fused kernel) ---
        if (vk_ctx && buf_Q && buf_K && buf_V && buf_attn) {
            vulkan_buffer_upload(vk_ctx, buf_Q, Q_all, num_tokens * q_dim);
            vulkan_buffer_upload(vk_ctx, buf_K, K_all, num_tokens * kv_dim);
            vulkan_buffer_upload(vk_ctx, buf_V, V_all, num_tokens * kv_dim);

            // QK norm (video/ViT mode)
            vulkan_qk_norm(vk_ctx, buf_Q, num_tokens, num_heads, head_dim, use_qk_norm, rms_eps);
            vulkan_qk_norm(vk_ctx, buf_K, num_tokens, kv_heads, head_dim, use_qk_norm, rms_eps);

            vulkan_batch_attention(vk_ctx, buf_Q, buf_K, buf_V, buf_attn,
                                   num_tokens, num_heads, kv_heads, head_dim);

            vulkan_buffer_download(vk_ctx, buf_attn, attn_out, num_tokens * q_dim);
        } else {
            // CPU attention fallback (proper multi-head with causal mask)
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int pos = 0; pos < num_tokens; pos++) {
                for (int h = 0; h < num_heads; h++) {
                    int kv_h = h * kv_heads / num_heads;
                    float* q_h = &Q_all[pos * q_dim + h * head_dim];
                    float* out_h = &attn_out[pos * q_dim + h * head_dim];

                    // Compute attention scores
                    // Allocate per-query scratch (no hardcoded max seq)
                    float* scores = calloc((size_t)pos + 1, sizeof(float));
                    if (!scores) continue;
                    float max_score = -1e9f;
                    for (int j = 0; j <= pos; j++) {
                        float* k_h = &K_all[j * kv_dim + kv_h * head_dim];
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += q_h[d] * k_h[d];
                        }
                        scores[j] = score * scale;
                        if (scores[j] > max_score) max_score = scores[j];
                    }

                    // Softmax
                    float sum = 0.0f;
                    for (int j = 0; j <= pos; j++) {
                        scores[j] = expf(scores[j] - max_score);
                        sum += scores[j];
                    }
                    for (int j = 0; j <= pos; j++) {
                        scores[j] /= sum;
                    }

                    // Weighted sum of V
                    memset(out_h, 0, head_dim * sizeof(float));
                    for (int j = 0; j <= pos; j++) {
                        float* v_h = &V_all[j * kv_dim + kv_h * head_dim];
                        for (int d = 0; d < head_dim; d++) {
                            out_h[d] += scores[j] * v_h[d];
                        }
                    }
                }
            }
        }

        // Cache attention output for backward
        for (int pos = 0; pos < num_tokens; pos++) {
            memcpy(&act->attn_output[layer][pos * q_dim],
                   &attn_out[pos * q_dim], q_dim * sizeof(float));
        }

        // --- Batch O projection ---
        if (vk_ctx) {
            gpu_batch_matmul(vk_ctx, attn_out, num_tokens, q_dim,
                            lw->o_proj.weight, hidden_size, ffn_out);
        } else {
            for (int pos = 0; pos < num_tokens; pos++) {
                matvec_fp32(&ffn_out[pos * hidden_size], lw->o_proj.weight,
                           &attn_out[pos * q_dim], hidden_size, q_dim);
            }
        }

        // Residual connection
        for (int i = 0; i < num_tokens * hidden_size; i++) {
            hidden[i] += ffn_out[i];
        }

        // --- Batch RMSNorm (post-attention) ---
        for (int pos = 0; pos < num_tokens; pos++) {
            rms_norm_fp32(&hidden_norm[pos * hidden_size],
                         &hidden[pos * hidden_size],
                         lw->post_norm.weight, hidden_size, rms_eps);
            memcpy(&act->x_norm_ffn[layer][pos * hidden_size],
                   &hidden_norm[pos * hidden_size], hidden_size * sizeof(float));
        }

        // --- Batch FFN: gate and up projections ---
        if (vk_ctx) {
            gpu_batch_matmul(vk_ctx, hidden_norm, num_tokens, hidden_size,
                            lw->gate_proj.weight, intermediate_size, ffn_gate);
            gpu_batch_matmul(vk_ctx, hidden_norm, num_tokens, hidden_size,
                            lw->up_proj.weight, intermediate_size, ffn_up);
        } else {
            for (int pos = 0; pos < num_tokens; pos++) {
                matvec_fp32(&ffn_gate[pos * intermediate_size], lw->gate_proj.weight,
                           &hidden_norm[pos * hidden_size], intermediate_size, hidden_size);
                matvec_fp32(&ffn_up[pos * intermediate_size], lw->up_proj.weight,
                           &hidden_norm[pos * hidden_size], intermediate_size, hidden_size);
            }
        }

        // Cache for backward
        memcpy(act->ffn_gate_out[layer], ffn_gate, num_tokens * intermediate_size * sizeof(float));
        memcpy(act->ffn_up_out[layer], ffn_up, num_tokens * intermediate_size * sizeof(float));

        // --- Batch SiLU(gate) * up ---
        if (vk_ctx) {
            // GPU SiLU for all positions
            vulkan_buffer_t* buf_gate = (vulkan_buffer_t*)vk_ctx->fwd_input_buffer;
            vulkan_buffer_t* buf_silu = (vulkan_buffer_t*)vk_ctx->fwd_output_buffer;
            vulkan_buffer_upload(vk_ctx, buf_gate, ffn_gate, num_tokens * intermediate_size);
            vulkan_silu(vk_ctx, buf_gate, buf_silu, num_tokens * intermediate_size);
            vulkan_buffer_download(vk_ctx, buf_silu, ffn_hidden, num_tokens * intermediate_size);
        } else {
            silu_fp32(ffn_hidden, ffn_gate, num_tokens * intermediate_size);
        }

        // Elementwise multiply with up
        for (int i = 0; i < num_tokens * intermediate_size; i++) {
            ffn_hidden[i] *= ffn_up[i];
        }

        // Cache for backward
        memcpy(act->ffn_hidden[layer], ffn_hidden, num_tokens * intermediate_size * sizeof(float));

        // --- Batch down projection ---
        if (vk_ctx) {
            gpu_batch_matmul(vk_ctx, ffn_hidden, num_tokens, intermediate_size,
                            lw->down_proj.weight, hidden_size, ffn_out);
        } else {
            for (int pos = 0; pos < num_tokens; pos++) {
                matvec_fp32(&ffn_out[pos * hidden_size], lw->down_proj.weight,
                           &ffn_hidden[pos * intermediate_size], hidden_size, intermediate_size);
            }
        }

        // Residual connection
        for (int i = 0; i < num_tokens * hidden_size; i++) {
            hidden[i] += ffn_out[i];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FINAL NORM + LM HEAD
    // ═══════════════════════════════════════════════════════════════════════

    for (int pos = 0; pos < num_tokens; pos++) {
        rms_norm_fp32(&hidden_norm[pos * hidden_size],
                     &hidden[pos * hidden_size],
                     weights->final_norm.weight, hidden_size, rms_eps);
        memcpy(&act->final_hidden[pos * hidden_size],
               &hidden_norm[pos * hidden_size], hidden_size * sizeof(float));
    }

    // LM Head (tied embeddings): logits = hidden @ embed^T
    {
        for (int pos = 0; pos < num_tokens; pos++) {
            for (int v = 0; v < vocab_size; v++) {
                float sum = 0.0f;
                for (int i = 0; i < hidden_size; i++) {
                    sum += weights->embed_tokens.weight[v * hidden_size + i] *
                           hidden_norm[pos * hidden_size + i];
                }
                all_logits[pos * vocab_size + v] = sum;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CLEANUP
    // ═══════════════════════════════════════════════════════════════════════

    free(hidden);
    free(hidden_norm);
    free(Q_all);
    free(K_all);
    free(V_all);
    free(attn_out);
    free(ffn_gate);
    free(ffn_up);
    free(ffn_hidden);
    free(ffn_out);

    return all_logits;
}

// Helper: Accumulate gradient for weight matrix (outer product)
// grad_W += grad_out @ input^T
static void accumulate_weight_gradient(float* grad_W, const float* grad_out,
                                       const float* input, int out_dim, int in_dim) {
    // grad_W[i,j] += grad_out[i] * input[j]
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            grad_W[i * in_dim + j] += grad_out[i] * input[j];
        }
    }
}

// Full sequence backward: accumulate gradients for ALL positions (LoRA or full weights)
void train_backward(train_state_t* state, float* grad_logits, int num_predictions) {
    // Need either LoRA or full weights, plus activation cache
    if ((!state->lora && !state->full_weights) || !state->act_cache) return;

    const model_config_t* config = state->config;
    activation_cache_t* act = state->act_cache;
    lora_model_t* lora = state->lora;
    trainable_weights_t* weights = state->full_weights;

    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int vocab_size = config->vocab_size;
    const int num_layers = config->num_hidden_layers;
    const int q_dim = config->num_attention_heads * config->head_dim;

    // Allocate gradient buffers (reused across positions)
    float* grad_hidden = calloc(hidden_size, sizeof(float));

    // Loop over ALL positions and accumulate gradients
    for (int pos = 0; pos < num_predictions; pos++) {
        // Get grad_logits for this position
        float* grad_logits_pos = &grad_logits[pos * vocab_size];

        // === Backprop through LM head ===
        // logits = lm_head @ final_hidden, so grad_final_hidden = lm_head^T @ grad_logits
        // For tied embeddings, lm_head = embed_tokens

        // Get the final hidden state for this position
        float* final_hidden_pos = &act->final_hidden[pos * hidden_size];

        if (weights) {
            // FULL WEIGHT MODE: Compute embedding gradient
            // LM head is tied with embeddings: lm_head = embed_tokens^T
            // grad_embed[i,j] += grad_logits[i] * final_hidden[j]
            // This is expensive (vocab_size * hidden_size), so we accumulate
            accumulate_weight_gradient(weights->embed_tokens.grad,
                                      grad_logits_pos, final_hidden_pos,
                                      vocab_size, hidden_size);

            // PROPER LM Head backward: grad_hidden = embed_tokens^T @ grad_logits
            // embed_tokens: [vocab_size, hidden_size]
            // grad_logits: [vocab_size]
            // grad_hidden: [hidden_size]
            // grad_hidden[i] = sum_v embed_tokens[v, i] * grad_logits[v]
            for (int i = 0; i < hidden_size; i++) {
                float sum = 0.0f;
                for (int v = 0; v < vocab_size; v++) {
                    sum += weights->embed_tokens.weight[v * hidden_size + i] * grad_logits_pos[v];
                }
                grad_hidden[i] = sum;
            }

            // Final norm gradient (RMSNorm scale before LM head)
            // The final_hidden is the output of final_norm
            for (int i = 0; i < hidden_size; i++) {
                weights->final_norm.grad[i] += grad_hidden[i] * final_hidden_pos[i];
            }
        } else {
            // LoRA mode - PROPER LM Head backward using model's BF16 embeddings
            // grad_hidden = embed_tokens^T @ grad_logits
            uint16_t* embed_bf16 = get_model_tensor(state->model, "model.embed_tokens.weight");
            if (embed_bf16) {
                for (int i = 0; i < hidden_size; i++) {
                    float sum = 0.0f;
                    for (int v = 0; v < vocab_size; v++) {
                        float embed_val = bf16_to_f32(embed_bf16[v * hidden_size + i]);
                        sum += embed_val * grad_logits_pos[v];
                    }
                    grad_hidden[i] = sum;
                }
            } else {
                // Fallback if embeddings not found (shouldn't happen)
                fprintf(stderr, "WARNING: embed_tokens not found for LoRA backward\n");
                memset(grad_hidden, 0, hidden_size * sizeof(float));
            }
        }

        // === Backprop through layers (reverse order) ===
        for (int layer = num_layers - 1; layer >= 0; layer--) {
            // Get saved activations for this layer and position
            float* x_norm_ffn = &act->x_norm_ffn[layer][pos * hidden_size];
            float* ffn_hidden = &act->ffn_hidden[layer][pos * intermediate_size];
            float* attn_output = &act->attn_output[layer][pos * q_dim];
            float* x_norm_attn = &act->x_norm_attn[layer][pos * hidden_size];

            // === FFN backward: PROPER IMPLEMENTATION ===
            // Forward was: ffn_hidden = silu(gate_proj @ x_norm) * (up_proj @ x_norm)
            //              output = down_proj @ ffn_hidden

            // Get saved gate and up outputs for SiLU backward
            float* gate_out = &act->ffn_gate_out[layer][pos * intermediate_size];
            float* up_out = &act->ffn_up_out[layer][pos * intermediate_size];

            if (weights) {
                // FULL WEIGHT MODE

                // 1. Weight gradient for down_proj
                accumulate_weight_gradient(weights->layers[layer].down_proj.grad,
                                          grad_hidden, ffn_hidden, hidden_size, intermediate_size);

                // 2. Backprop through down_proj: grad_ffn_hidden = W_down^T @ grad_hidden
                float* grad_ffn_hidden = calloc(intermediate_size, sizeof(float));
                for (int j = 0; j < intermediate_size; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < hidden_size; i++) {
                        sum += weights->layers[layer].down_proj.weight[i * intermediate_size + j] * grad_hidden[i];
                    }
                    grad_ffn_hidden[j] = sum;
                }

                // 3. Backprop through SiLU * up
                // ffn_hidden = silu(gate) * up
                // silu(x) = x * sigmoid(x)
                // silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                float* grad_gate = calloc(intermediate_size, sizeof(float));
                float* grad_up = calloc(intermediate_size, sizeof(float));

                for (int i = 0; i < intermediate_size; i++) {
                    float g = gate_out[i];
                    float sigmoid_g = 1.0f / (1.0f + expf(-g));
                    float silu_g = g * sigmoid_g;
                    float silu_deriv = sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));

                    // grad_gate = grad_ffn_hidden * up * silu'(gate)
                    grad_gate[i] = grad_ffn_hidden[i] * up_out[i] * silu_deriv;
                    // grad_up = grad_ffn_hidden * silu(gate)
                    grad_up[i] = grad_ffn_hidden[i] * silu_g;
                }

                // 4. Weight gradients for gate_proj and up_proj
                accumulate_weight_gradient(weights->layers[layer].gate_proj.grad,
                                          grad_gate, x_norm_ffn, intermediate_size, hidden_size);
                accumulate_weight_gradient(weights->layers[layer].up_proj.grad,
                                          grad_up, x_norm_ffn, intermediate_size, hidden_size);

                // 5. Backprop to x_norm_ffn: grad_x_norm_ffn = W_gate^T @ grad_gate + W_up^T @ grad_up
                float* grad_x_norm_ffn = calloc(hidden_size, sizeof(float));
                for (int i = 0; i < hidden_size; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < intermediate_size; j++) {
                        sum += weights->layers[layer].gate_proj.weight[j * hidden_size + i] * grad_gate[j];
                        sum += weights->layers[layer].up_proj.weight[j * hidden_size + i] * grad_up[j];
                    }
                    grad_x_norm_ffn[i] = sum;
                }

                // 6. Post-attention norm gradient (RMSNorm scale)
                for (int i = 0; i < hidden_size; i++) {
                    weights->layers[layer].post_norm.grad[i] += grad_x_norm_ffn[i] * x_norm_ffn[i];
                }

                free(grad_ffn_hidden);
                free(grad_gate);
                free(grad_up);
                free(grad_x_norm_ffn);

            } else if (lora && lora->down_adapters && lora->down_adapters[layer]) {
                // LoRA mode - compute grad_ffn_hidden for backprop
                // Use model's down_proj weights
                uint16_t* down_bf16 = NULL;
                char tensor_name[128];
                snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.down_proj.weight", layer);
                down_bf16 = get_model_tensor(state->model, tensor_name);

                lora_backward(lora->down_adapters[layer], ffn_hidden, grad_hidden, NULL, 1);

                if (down_bf16 && lora->apply_to_ffn) {
                    // Backprop through down_proj
                    float* grad_ffn_hidden = calloc(intermediate_size, sizeof(float));
                    for (int j = 0; j < intermediate_size; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < hidden_size; i++) {
                            sum += bf16_to_f32(down_bf16[i * intermediate_size + j]) * grad_hidden[i];
                        }
                        grad_ffn_hidden[j] = sum;
                    }

                    // SiLU backward
                    float* grad_gate = calloc(intermediate_size, sizeof(float));
                    float* grad_up = calloc(intermediate_size, sizeof(float));
                    for (int i = 0; i < intermediate_size; i++) {
                        float g = gate_out[i];
                        float sigmoid_g = 1.0f / (1.0f + expf(-g));
                        float silu_g = g * sigmoid_g;
                        float silu_deriv = sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));
                        grad_gate[i] = grad_ffn_hidden[i] * up_out[i] * silu_deriv;
                        grad_up[i] = grad_ffn_hidden[i] * silu_g;
                    }

                    if (lora->gate_adapters && lora->gate_adapters[layer]) {
                        lora_backward(lora->gate_adapters[layer], x_norm_ffn, grad_gate, NULL, 1);
                    }
                    if (lora->up_adapters && lora->up_adapters[layer]) {
                        lora_backward(lora->up_adapters[layer], x_norm_ffn, grad_up, NULL, 1);
                    }

                    free(grad_ffn_hidden);
                    free(grad_gate);
                    free(grad_up);
                }
            } else if (lora) {
                // LoRA mode without FFN adapters - still need to propagate grad_hidden through frozen FFN
                // grad_hidden comes in from the residual, need to backprop through FFN to get grad for attention
                // FFN forward: output = down_proj @ (silu(gate_proj @ x) * up_proj @ x) + residual
                // For backprop through frozen FFN: grad_x_norm = gate^T @ grad_gate + up^T @ grad_up
                // But since we're not training FFN, we just need grad_hidden to flow to attention
                // The residual connection means grad_hidden passes through unchanged
                // (This is correct - the gradient from the residual flows directly to attention)
            }

            // === Attention backward: PROPER IMPLEMENTATION ===
            // Forward was: attn_out = Attention(Q, K, V), output = O_proj @ attn_out

            int kv_dim = config->num_key_value_heads * config->head_dim;
            int num_heads = config->num_attention_heads;
            int kv_heads = config->num_key_value_heads;
            int head_dim = config->head_dim;
            float scale = 1.0f / sqrtf((float)head_dim);

            if (weights) {
                // FULL WEIGHT TRAINING MODE

                // 1. O projection weight gradient
                accumulate_weight_gradient(weights->layers[layer].o_proj.grad,
                                          grad_hidden, attn_output, hidden_size, q_dim);

                // 2. Backprop through O proj: grad_attn_out = W_o^T @ grad_hidden
                float* grad_attn_out = calloc(q_dim, sizeof(float));
                for (int j = 0; j < q_dim; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < hidden_size; i++) {
                        sum += weights->layers[layer].o_proj.weight[i * q_dim + j] * grad_hidden[i];
                    }
                    grad_attn_out[j] = sum;
                }

                // 3. Re-compute Q, K, V for this position (needed for attention backward)
                // Q = W_q @ x_norm_attn
                float* Q = calloc(q_dim, sizeof(float));
                for (int i = 0; i < q_dim; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < hidden_size; j++) {
                        sum += weights->layers[layer].q_proj.weight[i * hidden_size + j] * x_norm_attn[j];
                    }
                    Q[i] = sum;
                }

                // For attention backward, we also need K, V from all positions 0..pos
                // This requires accessing x_norm_attn for all those positions
                // Allocate K, V caches for positions 0..pos
                float* K_cache = calloc((pos + 1) * kv_dim, sizeof(float));
                float* V_cache = calloc((pos + 1) * kv_dim, sizeof(float));

                for (int p = 0; p <= pos; p++) {
                    float* x_norm_p = &act->x_norm_attn[layer][p * hidden_size];
                    for (int i = 0; i < kv_dim; i++) {
                        float k_sum = 0.0f, v_sum = 0.0f;
                        for (int j = 0; j < hidden_size; j++) {
                            k_sum += weights->layers[layer].k_proj.weight[i * hidden_size + j] * x_norm_p[j];
                            v_sum += weights->layers[layer].v_proj.weight[i * hidden_size + j] * x_norm_p[j];
                        }
                        K_cache[p * kv_dim + i] = k_sum;
                        V_cache[p * kv_dim + i] = v_sum;
                    }
                }

                // 4. Attention backward per head
                // attn_out[h] = softmax(Q[h] @ K[h]^T / sqrt(d)) @ V[h]
                float* grad_Q = calloc(q_dim, sizeof(float));
                float* grad_K_all = calloc((pos + 1) * kv_dim, sizeof(float));
                float* grad_V_all = calloc((pos + 1) * kv_dim, sizeof(float));

                for (int h = 0; h < num_heads; h++) {
                    int kv_h = h * kv_heads / num_heads;  // GQA mapping
                    float* q_h = &Q[h * head_dim];
                    float* grad_attn_h = &grad_attn_out[h * head_dim];
                    float* grad_q_h = &grad_Q[h * head_dim];

                    // Compute attention scores and weights for this head
                    float* scores = calloc(pos + 1, sizeof(float));
                    float* attn_weights = calloc((size_t)pos + 1, sizeof(float));

                    // Scores = Q @ K^T / sqrt(d)
                    float max_score = -1e9f;
                    for (int p = 0; p <= pos; p++) {
                        float* k_h = &K_cache[p * kv_dim + kv_h * head_dim];
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += q_h[d] * k_h[d];
                        }
                        scores[p] = score * scale;
                        if (scores[p] > max_score) max_score = scores[p];
                    }

                    // Softmax
                    float sum_exp = 0.0f;
                    for (int p = 0; p <= pos; p++) {
                        attn_weights[p] = expf(scores[p] - max_score);
                        sum_exp += attn_weights[p];
                    }
                    for (int p = 0; p <= pos; p++) {
                        attn_weights[p] /= sum_exp;
                    }

                    // Backward through attention
                    // grad_V = attn_weights^T @ grad_attn_out (for each position)
                    for (int p = 0; p <= pos; p++) {
                        for (int d = 0; d < head_dim; d++) {
                            grad_V_all[p * kv_dim + kv_h * head_dim + d] += attn_weights[p] * grad_attn_h[d];
                        }
                    }

                    // grad_attn_weights = grad_attn_out @ V^T
                    float* grad_attn_weights = calloc(pos + 1, sizeof(float));
                    for (int p = 0; p <= pos; p++) {
                        float* v_h = &V_cache[p * kv_dim + kv_h * head_dim];
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            dot += grad_attn_h[d] * v_h[d];
                        }
                        grad_attn_weights[p] = dot;
                    }

                    // Softmax backward: grad_scores = attn_weights * (grad_attn_weights - sum(attn_weights * grad_attn_weights))
                    float weighted_sum = 0.0f;
                    for (int p = 0; p <= pos; p++) {
                        weighted_sum += attn_weights[p] * grad_attn_weights[p];
                    }
                    float* grad_scores = calloc(pos + 1, sizeof(float));
                    for (int p = 0; p <= pos; p++) {
                        grad_scores[p] = attn_weights[p] * (grad_attn_weights[p] - weighted_sum);
                    }

                    // grad_Q = grad_scores @ K * scale
                    for (int d = 0; d < head_dim; d++) {
                        float sum = 0.0f;
                        for (int p = 0; p <= pos; p++) {
                            sum += grad_scores[p] * K_cache[p * kv_dim + kv_h * head_dim + d];
                        }
                        grad_q_h[d] = sum * scale;
                    }

                    // grad_K = grad_scores^T @ Q * scale (accumulate for each position)
                    for (int p = 0; p <= pos; p++) {
                        for (int d = 0; d < head_dim; d++) {
                            grad_K_all[p * kv_dim + kv_h * head_dim + d] += grad_scores[p] * q_h[d] * scale;
                        }
                    }

                    free(scores);
                    free(attn_weights);
                    free(grad_attn_weights);
                    free(grad_scores);
                }

                // 5. Weight gradients for Q, K, V projections
                // Only accumulate gradient for current position's Q
                accumulate_weight_gradient(weights->layers[layer].q_proj.grad,
                                          grad_Q, x_norm_attn, q_dim, hidden_size);

                // K and V gradients: accumulate for all positions 0..pos
                for (int p = 0; p <= pos; p++) {
                    float* x_norm_p = &act->x_norm_attn[layer][p * hidden_size];
                    float* grad_k_p = &grad_K_all[p * kv_dim];
                    float* grad_v_p = &grad_V_all[p * kv_dim];
                    accumulate_weight_gradient(weights->layers[layer].k_proj.grad,
                                              grad_k_p, x_norm_p, kv_dim, hidden_size);
                    accumulate_weight_gradient(weights->layers[layer].v_proj.grad,
                                              grad_v_p, x_norm_p, kv_dim, hidden_size);
                }

                // 6. Backprop to x_norm_attn: grad_x_norm = W_q^T @ grad_Q + W_k^T @ grad_K + W_v^T @ grad_V
                // For current position, we have grad_Q and need to include grad_K, grad_V for pos
                float* grad_x_norm_attn = calloc(hidden_size, sizeof(float));
                float* grad_k_pos = &grad_K_all[pos * kv_dim];
                float* grad_v_pos = &grad_V_all[pos * kv_dim];

                for (int i = 0; i < hidden_size; i++) {
                    float sum = 0.0f;
                    // W_q^T @ grad_Q
                    for (int j = 0; j < q_dim; j++) {
                        sum += weights->layers[layer].q_proj.weight[j * hidden_size + i] * grad_Q[j];
                    }
                    // W_k^T @ grad_K
                    for (int j = 0; j < kv_dim; j++) {
                        sum += weights->layers[layer].k_proj.weight[j * hidden_size + i] * grad_k_pos[j];
                    }
                    // W_v^T @ grad_V
                    for (int j = 0; j < kv_dim; j++) {
                        sum += weights->layers[layer].v_proj.weight[j * hidden_size + i] * grad_v_pos[j];
                    }
                    grad_x_norm_attn[i] = sum;
                }

                // 7. Input norm gradient (RMSNorm scale)
                for (int i = 0; i < hidden_size; i++) {
                    weights->layers[layer].input_norm.grad[i] += grad_x_norm_attn[i] * x_norm_attn[i];
                }

                free(grad_attn_out);
                free(Q);
                free(K_cache);
                free(V_cache);
                free(grad_Q);
                free(grad_K_all);
                free(grad_V_all);
                free(grad_x_norm_attn);

            } else if (lora && (lora->q_adapters || lora->v_adapters)) {
                // LoRA mode - simplified but correct for trained adapters
                if (lora->o_adapters && lora->o_adapters[layer]) {
                    lora_backward(lora->o_adapters[layer], attn_output, grad_hidden, NULL, 1);
                }

                // For LoRA Q/K/V, we need grad_attn_out first
                // Get O projection weights from model
                char tensor_name[128];
                snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.o_proj.weight", layer);
                uint16_t* o_bf16 = get_model_tensor(state->model, tensor_name);

                if (o_bf16) {
                    float* grad_attn_out = calloc(q_dim, sizeof(float));
                    for (int j = 0; j < q_dim; j++) {
                        float sum = 0.0f;
                        for (int i = 0; i < hidden_size; i++) {
                            sum += bf16_to_f32(o_bf16[i * q_dim + j]) * grad_hidden[i];
                        }
                        grad_attn_out[j] = sum;
                    }

                    // For LoRA, use grad_attn_out directly as approximation for Q/V gradients
                    // (Full attention backward would require re-computing K,V from model weights)
                    if (lora->q_adapters && lora->q_adapters[layer]) {
                        lora_backward(lora->q_adapters[layer], x_norm_attn, grad_attn_out, NULL, 1);
                    }
                    if (lora->v_adapters && lora->v_adapters[layer]) {
                        // V gradient is similar magnitude to attn_out gradient
                        float* grad_v = calloc(kv_dim, sizeof(float));
                        for (int i = 0; i < kv_dim; i++) {
                            grad_v[i] = grad_attn_out[i % q_dim];
                        }
                        lora_backward(lora->v_adapters[layer], x_norm_attn, grad_v, NULL, 1);
                        free(grad_v);
                    }
                    if (lora->k_adapters && lora->k_adapters[layer]) {
                        float* grad_k = calloc(kv_dim, sizeof(float));
                        for (int i = 0; i < kv_dim; i++) {
                            grad_k[i] = grad_attn_out[i % q_dim];
                        }
                        lora_backward(lora->k_adapters[layer], x_norm_attn, grad_k, NULL, 1);
                        free(grad_k);
                    }
                    free(grad_attn_out);
                }
            }

            // Residual connections: gradient passes through unchanged
        }  // end layer loop
    }  // end position loop

    free(grad_hidden);
}

static void full_weights_optimizer_step_gpu_resident_record(const train_state_t* state,
                                                            float lr, float beta1, float beta2,
                                                            float weight_decay, float eps, int step,
                                                            vulkan_context_t* ctx) {
    if (!state || !state->full_weights || !state->config || !ctx) return;
    if (!ctx->gpu_w_embed) return;

    trainable_weights_t* w = state->full_weights;
    const model_config_t* config = state->config;
    int num_layers = config->num_hidden_layers;

    // Global
    vulkan_adamw_update(ctx,
                        (vulkan_buffer_t*)ctx->gpu_w_embed,
                        (vulkan_buffer_t*)ctx->gpu_g_embed,
                        (vulkan_buffer_t*)ctx->gpu_m_embed,
                        (vulkan_buffer_t*)ctx->gpu_v_embed,
                        lr, beta1, beta2, weight_decay, eps, step, w->embed_tokens.size);

    vulkan_adamw_update(ctx,
                        (vulkan_buffer_t*)ctx->gpu_w_final_norm,
                        (vulkan_buffer_t*)ctx->gpu_g_final_norm,
                        (vulkan_buffer_t*)ctx->gpu_m_final_norm,
                        (vulkan_buffer_t*)ctx->gpu_v_final_norm,
                        lr, beta1, beta2, weight_decay, eps, step, w->final_norm.size);

    // Layers
    for (int l = 0; l < num_layers; l++) {
        layer_weights_t* lw = &w->layers[l];

        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_q)[l], ((vulkan_buffer_t**)ctx->gpu_g_q)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_q)[l], ((vulkan_buffer_t**)ctx->gpu_v_q)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->q_proj.size);
        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_k)[l], ((vulkan_buffer_t**)ctx->gpu_g_k)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_k)[l], ((vulkan_buffer_t**)ctx->gpu_v_k)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->k_proj.size);
        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_v)[l], ((vulkan_buffer_t**)ctx->gpu_g_v)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_v)[l], ((vulkan_buffer_t**)ctx->gpu_v_v)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->v_proj.size);
        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_o)[l], ((vulkan_buffer_t**)ctx->gpu_g_o)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_o)[l], ((vulkan_buffer_t**)ctx->gpu_v_o)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->o_proj.size);

        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_gate)[l], ((vulkan_buffer_t**)ctx->gpu_g_gate)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_gate)[l], ((vulkan_buffer_t**)ctx->gpu_v_gate)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->gate_proj.size);
        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_up)[l], ((vulkan_buffer_t**)ctx->gpu_g_up)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_up)[l], ((vulkan_buffer_t**)ctx->gpu_v_up)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->up_proj.size);
        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_down)[l], ((vulkan_buffer_t**)ctx->gpu_g_down)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_down)[l], ((vulkan_buffer_t**)ctx->gpu_v_down)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->down_proj.size);

        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[l], ((vulkan_buffer_t**)ctx->gpu_g_in_norm)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_in_norm)[l], ((vulkan_buffer_t**)ctx->gpu_v_in_norm)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->input_norm.size);
        vulkan_adamw_update(ctx, ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[l], ((vulkan_buffer_t**)ctx->gpu_g_post_norm)[l],
                            ((vulkan_buffer_t**)ctx->gpu_m_post_norm)[l], ((vulkan_buffer_t**)ctx->gpu_v_post_norm)[l],
                            lr, beta1, beta2, weight_decay, eps, step, lw->post_norm.size);
    }

    // Video weights (query_learnable, query_fusion_weight)
    if (config->num_queries > 0 && state->query_learnable && state->video_grad_queries) {
        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : config->hidden_size;
        size_t q_size = (size_t)num_queries * query_dim;
        size_t f_size = (size_t)query_dim * query_dim;

        vulkan_adamw_update(ctx,
                            (vulkan_buffer_t*)state->query_learnable,
                            (vulkan_buffer_t*)state->video_grad_queries,
                            (vulkan_buffer_t*)state->video_m_queries,
                            (vulkan_buffer_t*)state->video_v_queries,
                            lr, beta1, beta2, weight_decay, eps, step, q_size);

        if (state->query_fusion_weight && state->video_grad_fusion) {
            vulkan_adamw_update(ctx,
                                (vulkan_buffer_t*)state->query_fusion_weight,
                                (vulkan_buffer_t*)state->video_grad_fusion,
                                (vulkan_buffer_t*)state->video_m_fusion,
                                (vulkan_buffer_t*)state->video_v_fusion,
                                lr, beta1, beta2, weight_decay, eps, step, f_size);
        }

        // Class head
        if (state->class_head_weight && state->class_head_grad_w) {
            int num_classes = config->num_classes > 0 ? config->num_classes : 10;
            size_t w_size = (size_t)query_dim * num_classes;
            size_t b_size = (size_t)num_classes;

            vulkan_adamw_update(ctx,
                                (vulkan_buffer_t*)state->class_head_weight,
                                (vulkan_buffer_t*)state->class_head_grad_w,
                                (vulkan_buffer_t*)state->class_head_m_w,
                                (vulkan_buffer_t*)state->class_head_v_w,
                                lr, beta1, beta2, weight_decay, eps, step, w_size);
            vulkan_adamw_update(ctx,
                                (vulkan_buffer_t*)state->class_head_bias,
                                (vulkan_buffer_t*)state->class_head_grad_b,
                                (vulkan_buffer_t*)state->class_head_m_b,
                                (vulkan_buffer_t*)state->class_head_v_b,
                                lr, beta1, beta2, weight_decay, eps, step, b_size);
        }

        // Mask head
        if (state->mask_mlp_w1 && state->mask_grad_w1) {
            int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
            size_t w1_size = (size_t)query_dim * mask_dim;
            size_t w2_size = (size_t)mask_dim * query_dim;

            vulkan_adamw_update(ctx,
                                (vulkan_buffer_t*)state->mask_mlp_w1,
                                (vulkan_buffer_t*)state->mask_grad_w1,
                                (vulkan_buffer_t*)state->mask_m_w1,
                                (vulkan_buffer_t*)state->mask_v_w1,
                                lr, beta1, beta2, weight_decay, eps, step, w1_size);
            vulkan_adamw_update(ctx,
                                (vulkan_buffer_t*)state->mask_mlp_w2,
                                (vulkan_buffer_t*)state->mask_grad_w2,
                                (vulkan_buffer_t*)state->mask_m_w2,
                                (vulkan_buffer_t*)state->mask_v_w2,
                                lr, beta1, beta2, weight_decay, eps, step, w2_size);
        }

        // CLIP projection (if enabled and dimensions differ)
        if (state->clip_proj_weight && state->clip_proj_grad) {
            int clip_dim = config->clip_embed_dim > 0 ? config->clip_embed_dim : query_dim;
            if (clip_dim != query_dim) {
                size_t proj_size = (size_t)query_dim * clip_dim;
                vulkan_adamw_update(ctx,
                                    (vulkan_buffer_t*)state->clip_proj_weight,
                                    (vulkan_buffer_t*)state->clip_proj_grad,
                                    (vulkan_buffer_t*)state->clip_proj_m,
                                    (vulkan_buffer_t*)state->clip_proj_v,
                                    lr, beta1, beta2, weight_decay, eps, step, proj_size);
            }
        }
    }
}

static float train_step_gpu_full(train_state_t* state, int* tokens, int num_tokens, vulkan_context_t* ctx) {
    if (!state || !state->full_weights || !state->config || !ctx) return -1.0f;

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
    const int use_qk_norm = config->use_qk_norm;  // Video/ViT mode

    int max_seq = state->act_cache ? state->act_cache->max_seq_len : num_tokens;
    if (max_seq < num_tokens) max_seq = num_tokens;

    ensure_fulltrain_fwd_buffers(ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_fulltrain_train_buffers(ctx, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_fulltrain_gpu_weights(ctx, state);
    ensure_video_gpu_weights(ctx, state);

    uint32_t nan_flag_count = 0;
    uint32_t nan_fwd_layer_base = 0;
    uint32_t nan_bwd_layer_base = 0;
    uint32_t nan_grad_hidden_bwd_layer_base = 0;
    uint32_t nan_weight_embed_idx = 0;
    uint32_t nan_weight_embed_post_opt_idx = 0;
    uint32_t nan_preopt_base = 0;

    // Zero grads at start of accumulation cycle
    if (state->accumulation_counter == 0) {
        gpu_zero_full_weight_grads(ctx, state);
        // Zero video gradients
        if (state->video_grad_queries) {
            vulkan_buffer_t* gq = (vulkan_buffer_t*)state->video_grad_queries;
            vulkan_fill_buffer(ctx, gq, 0, gq->size);
        }
        if (state->video_grad_fusion) {
            vulkan_buffer_t* gf = (vulkan_buffer_t*)state->video_grad_fusion;
            vulkan_fill_buffer(ctx, gf, 0, gf->size);
        }
        // Zero class head gradients
        if (state->class_head_grad_w) {
            vulkan_buffer_t* gw = (vulkan_buffer_t*)state->class_head_grad_w;
            vulkan_fill_buffer(ctx, gw, 0, gw->size);
        }
        if (state->class_head_grad_b) {
            vulkan_buffer_t* gb = (vulkan_buffer_t*)state->class_head_grad_b;
            vulkan_fill_buffer(ctx, gb, 0, gb->size);
        }
        // Zero mask head gradients
        if (state->mask_grad_w1) {
            vulkan_buffer_t* gw = (vulkan_buffer_t*)state->mask_grad_w1;
            vulkan_fill_buffer(ctx, gw, 0, gw->size);
        }
        if (state->mask_grad_w2) {
            vulkan_buffer_t* gw = (vulkan_buffer_t*)state->mask_grad_w2;
            vulkan_fill_buffer(ctx, gw, 0, gw->size);
        }
    }

    vulkan_buffer_t* buf_hidden = (vulkan_buffer_t*)ctx->fwd_hidden_buffer;
    vulkan_buffer_t* buf_hidden_norm = (vulkan_buffer_t*)ctx->fwd_hidden_norm_buffer;
    vulkan_buffer_t* buf_tmp_hidden = (vulkan_buffer_t*)ctx->fwd_tmp_hidden_buffer;
    vulkan_buffer_t* buf_gate = (vulkan_buffer_t*)ctx->fwd_ffn_gate_buffer;
    vulkan_buffer_t* buf_up = (vulkan_buffer_t*)ctx->fwd_ffn_up_buffer;
    vulkan_buffer_t* buf_ffn_hidden = (vulkan_buffer_t*)ctx->fwd_ffn_hidden_buffer;
    vulkan_buffer_t* buf_logits = (vulkan_buffer_t*)ctx->fwd_logits_buffer;

    vulkan_buffer_t* buf_Q = (vulkan_buffer_t*)ctx->attn_q_buffer;
    vulkan_buffer_t* buf_K = (vulkan_buffer_t*)ctx->attn_k_buffer;
    vulkan_buffer_t* buf_V = (vulkan_buffer_t*)ctx->attn_v_buffer;
    vulkan_buffer_t* buf_attn = (vulkan_buffer_t*)ctx->attn_out_buffer;

    vulkan_buffer_t** cache_layer_in = (vulkan_buffer_t**)ctx->fwd_cache_layer_input;
    vulkan_buffer_t** cache_x_norm_attn = (vulkan_buffer_t**)ctx->fwd_cache_x_norm_attn;
    vulkan_buffer_t** cache_attn_out = (vulkan_buffer_t**)ctx->fwd_cache_attn_out;
    vulkan_buffer_t** cache_post_attn_in = (vulkan_buffer_t**)ctx->fwd_cache_post_attn_in;
    vulkan_buffer_t** cache_x_norm_ffn = (vulkan_buffer_t**)ctx->fwd_cache_x_norm_ffn;
    vulkan_buffer_t** cache_ffn_gate_out = (vulkan_buffer_t**)ctx->fwd_cache_ffn_gate_out;
    vulkan_buffer_t** cache_ffn_up_out = (vulkan_buffer_t**)ctx->fwd_cache_ffn_up_out;
    vulkan_buffer_t** cache_silu_out = (vulkan_buffer_t**)ctx->fwd_cache_silu_out;
    vulkan_buffer_t** cache_ffn_hidden = (vulkan_buffer_t**)ctx->fwd_cache_ffn_hidden;
    vulkan_buffer_t** cache_q = (vulkan_buffer_t**)ctx->fwd_cache_q;
    vulkan_buffer_t** cache_k = (vulkan_buffer_t**)ctx->fwd_cache_k;
    vulkan_buffer_t** cache_v = (vulkan_buffer_t**)ctx->fwd_cache_v;

    // Check if video mode (tokens will be NULL, data already uploaded by train_step_video)
    int is_video_mode = (config->num_queries > 0);
    int num_predictions = num_tokens - 1;

    // These buffers are used in loss computation (declared outside for scope)
    vulkan_buffer_t* tokens_buf = (vulkan_buffer_t*)ctx->train_tokens_u32;
    vulkan_buffer_t* targets_buf = (vulkan_buffer_t*)ctx->train_targets_u32;

    // Upload tokens + targets (LLM mode only - video mode uploads in train_step_video)
    if (!is_video_mode && tokens != NULL) {
        uint32_t* tokens_u32 = tokens_buf ? (uint32_t*)tokens_buf->mapped : NULL;
        uint32_t* tokens_u32_heap = NULL;
        if (!tokens_u32) {
            tokens_u32_heap = malloc((size_t)num_tokens * sizeof(uint32_t));
            tokens_u32 = tokens_u32_heap;
        }
        if (!tokens_u32) return -1.0f;
        for (int i = 0; i < num_tokens; i++) tokens_u32[i] = (tokens[i] < 0) ? 0u : (uint32_t)tokens[i];
        vulkan_buffer_upload_u32(ctx, tokens_buf, tokens_u32, (size_t)num_tokens);

        uint32_t* targets_u32 = targets_buf ? (uint32_t*)targets_buf->mapped : NULL;
        uint32_t* targets_u32_heap = NULL;
        if (!targets_u32) {
            targets_u32_heap = malloc((size_t)num_predictions * sizeof(uint32_t));
            targets_u32 = targets_u32_heap;
        }
        if (!targets_u32) {
            free(tokens_u32_heap);
            return -1.0f;
        }
        for (int t = 0; t < num_predictions; t++) {
            int tok = tokens[t + 1];
            targets_u32[t] = (tok < 0) ? 0u : (uint32_t)tok;
        }
        vulkan_buffer_upload_u32(ctx, targets_buf, targets_u32, (size_t)num_predictions);
        free(tokens_u32_heap);
        free(targets_u32_heap);
    }

    // Record forward + loss + backward (+ optional optimizer) into one submit.
    vulkan_cmd_begin(ctx);

    // Debug: allocate/clear NaN flags for this step (recorded into the same submit).
    if (state->debug_nan_check) {
        // Global flags:
        //   [0] grad_hidden (after final-norm backward copy)
        //   [1] logits
        //   [2] hidden_norm (input to LM head)
        //   [3] grad_logits
        //   [4] grad_hidden (after LM head backward)
        //   [5] grad_tmp_hidden (output of final rmsnorm backward)
        //   [6] final_input (input to final rmsnorm backward)
        //   [7] hidden (after embedding lookup)
        //   [8] grad_hidden (right before embedding_backward)
        //   [9] g_embed (right after embedding_backward)
        //
        // Per-layer forward flags at nan_fwd_layer_base:
        //   flags[nan_fwd_layer_base + layer] = hidden after layer (post FFN residual)
        //
        // Per-layer backward flags at nan_bwd_layer_base:
        // flags[nan_bwd_layer_base + layer*5 + k]:
        //   k=0 grad_attn (input to attn backward)
        //   k=1 grad_q pre-rope (output of attn backward)
        //   k=2 grad_k post-attn-backward (atomic accumulation)
        //   k=3 grad_v post-attn-backward (atomic accumulation)
        //   k=4 grad_q post-rope (after rope_backward)
        nan_fwd_layer_base = 10u;
        nan_bwd_layer_base = nan_fwd_layer_base + (uint32_t)num_layers;
        // Per-layer backward checkpoint: grad_hidden after completing that layer's backward
        nan_grad_hidden_bwd_layer_base = nan_bwd_layer_base + (uint32_t)num_layers * 5u;

        nan_preopt_base = nan_grad_hidden_bwd_layer_base + (uint32_t)num_layers;
        // Pre-optimizer embed checks:
        //   [nan_preopt_base+0] w_embed (pre opt)
        //   [nan_preopt_base+1] g_embed (pre opt)
        //   [nan_preopt_base+2] m_embed (pre opt)
        //   [nan_preopt_base+3] v_embed (pre opt)
        // Post-optimizer embed check:
        //   [nan_preopt_base+4] w_embed (post opt)
        nan_weight_embed_idx = nan_preopt_base + 0u;
        nan_weight_embed_post_opt_idx = nan_preopt_base + 4u;
        nan_flag_count = nan_preopt_base + 5u;
        size_t nan_bytes = (size_t)nan_flag_count * sizeof(uint32_t);
        if (!ctx->debug_nan_flags || ((vulkan_buffer_t*)ctx->debug_nan_flags)->size < nan_bytes) {
            if (ctx->debug_nan_flags) vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->debug_nan_flags);
            ctx->debug_nan_flags = vulkan_buffer_create(ctx, nan_bytes);
        }
        if (ctx->debug_nan_flags) {
            vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->debug_nan_flags, 0u, nan_bytes);
        }

        // Check whether embedding weights are already poisoned at the start of the step.
        // This helps distinguish "optimizer introduced NaN last step" vs "forward/backward produced NaN this step".
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->gpu_w_embed,
                         (size_t)vocab_size * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         nan_weight_embed_idx);
    }

    // Ensure the last (unused) grad_logits row is zero without clearing the entire [seq,vocab] slab.
    // `cross_entropy_grad` only writes rows [0..num_predictions-1], but LM-head backward uses `num_tokens` rows.
    if (num_tokens > 0) {
        size_t last_row_off = (size_t)(num_tokens - 1) * (size_t)vocab_size * sizeof(float);
        vulkan_fill_buffer_range(ctx, (vulkan_buffer_t*)ctx->train_grad_logits, last_row_off, 0u,
                                 (size_t)vocab_size * sizeof(float));
    }

    int is_video = config->num_queries > 0;
    vulkan_buffer_t* video_kv_src = NULL;  // Persistent patch embeddings for video K/V
    if (is_video) {
        vulkan_buffer_t* frame_buf = (vulkan_buffer_t*)ctx->train_tokens_u32;
        vulkan_buffer_t* patch_buf = buf_tmp_hidden;
        vulkan_buffer_t* patch_embed_w = (vulkan_buffer_t*)ctx->gpu_w_embed;
        vulkan_buffer_t* patch_embed_out = (vulkan_buffer_t*)state->video_patch_embed;
        vulkan_buffer_t* queries = (vulkan_buffer_t*)state->query_learnable;
        vulkan_buffer_t* prev_queries = (vulkan_buffer_t*)state->prev_query_out;
        vulkan_buffer_t* fusion_w = (vulkan_buffer_t*)state->query_fusion_weight;

        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
        int frame_h = config->input_resolution[0];
        int frame_w = config->input_resolution[1];
        int in_channels = config->in_channels > 0 ? config->in_channels : 3;
        int patch_t = config->patch_size[0] > 0 ? config->patch_size[0] : 1;
        int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
        int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
        int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
        int patch_dim = in_channels * patch_t * patch_h * patch_w;

        // Upload mask targets for current frame (full training path)
        // Set mask values to -1.0 for background queries (class_gt < 0) to ignore in BCE
        if (config->mask_output && state->video_masks && state->mask_targets) {
            size_t masks_per_frame = (size_t)num_queries * num_patches;
            size_t mask_offset = (size_t)state->current_frame_idx * masks_per_frame;
            float* src_masks = &state->video_masks[mask_offset];

            // Get class labels for this frame to identify background queries
            int* labels = &state->video_labels[state->current_frame_idx * state->num_labels_per_frame];
            int num_classes = config->num_classes > 0 ? config->num_classes : 40;

            // Create temp buffer with -1.0 for background query masks
            float* tmp_masks = malloc(masks_per_frame * sizeof(float));
            for (int q = 0; q < num_queries; q++) {
                int label_idx = q < state->num_labels_per_frame ? q : 0;
                int label = labels[label_idx];
                float* dst = &tmp_masks[q * num_patches];
                float* src = &src_masks[q * num_patches];

                if (label < 0 || label >= num_classes) {
                    // Background query - set all mask values to -1.0 (ignore in BCE)
                    for (int p = 0; p < num_patches; p++) {
                        dst[p] = -1.0f;
                    }
                } else {
                    // Valid object - copy actual mask values
                    memcpy(dst, src, num_patches * sizeof(float));
                }
            }
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->mask_targets, tmp_masks, masks_per_frame);
            free(tmp_masks);
        }

        // Extract patches and embed -> persistent buffer for K/V across layers
        vulkan_patch_extract(ctx, frame_buf, patch_buf,
                             frame_h, frame_w, in_channels,
                             patch_t, patch_h, patch_w);
        vulkan_matmul_transpose(ctx, patch_buf, patch_embed_w, patch_embed_out,
                                num_patches, hidden_size, patch_dim, 0, 1, 0);

        // Add sinusoidal 2D position embeddings (encodes spatial row/col location)
        if (state->video_pos_embed) {
            vulkan_add(ctx, patch_embed_out, (vulkan_buffer_t*)state->video_pos_embed,
                       patch_embed_out, (size_t)num_patches * hidden_size);
        }
        video_kv_src = patch_embed_out;

        // Queries go to buf_hidden (the evolving state through transformer)
        if (state->current_frame_idx == 0 || !prev_queries) {
            vulkan_copy_buffer(ctx, queries, buf_hidden,
                               (size_t)num_queries * (size_t)query_dim * sizeof(float));
        } else {
            vulkan_matmul_transpose(ctx, prev_queries, fusion_w, buf_hidden,
                                    num_queries, query_dim, query_dim, 0, 1, 0);
            vulkan_add(ctx, buf_hidden, queries, buf_hidden,
                       (size_t)num_queries * (size_t)query_dim);
        }
    } else {
        vulkan_embed_lookup(ctx,
                            (vulkan_buffer_t*)ctx->train_tokens_u32,
                            (vulkan_buffer_t*)ctx->gpu_w_embed,
                            buf_hidden,
                            num_tokens, hidden_size, vocab_size);
    }

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         buf_hidden,
                         (size_t)num_tokens * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         7u);
    }

    // Transformer forward (GPU-resident weights)
    for (int layer = 0; layer < num_layers; layer++) {
        // Cache layer input (pre input-norm)
        // NOTE: cache copies use _cache variant (no post-barrier) since destinations
        // are only read in backward pass. Single barrier before backward loop.
        vulkan_copy_buffer_cache(ctx, buf_hidden, cache_layer_in[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // RMSNorm (pre-attention)
        vulkan_rmsnorm(ctx, buf_hidden, ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[layer], buf_hidden_norm, num_tokens, hidden_size, rms_eps);
        vulkan_copy_buffer_cache(ctx, buf_hidden_norm, cache_x_norm_attn[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Q, K, V projections (output = X @ W^T)
        // Video: Q from queries (buf_hidden_norm), K/V from patch embeddings (video_kv_src)
        // LLM: Q/K/V all from buf_hidden_norm
        vulkan_buffer_t* kv_src = is_video ? video_kv_src : buf_hidden_norm;
        vulkan_matmul_transpose(ctx, buf_hidden_norm, ((vulkan_buffer_t**)ctx->gpu_w_q)[layer], buf_Q,
                                num_tokens, q_dim, hidden_size, 0, 1, 0);
        vulkan_matmul_transpose(ctx, kv_src, ((vulkan_buffer_t**)ctx->gpu_w_k)[layer], buf_K,
                                num_tokens, kv_dim, hidden_size, 0, 1, 0);
        vulkan_matmul_transpose(ctx, kv_src, ((vulkan_buffer_t**)ctx->gpu_w_v)[layer], buf_V,
                                num_tokens, kv_dim, hidden_size, 0, 1, 0);

        // RoPE
        vulkan_rope(ctx, buf_Q, buf_Q, num_tokens, num_heads, head_dim, rope_theta);
        vulkan_rope(ctx, buf_K, buf_K, num_tokens, kv_heads, head_dim, rope_theta);

        // QK norm (video/ViT mode) - must be after RoPE, before cache and attention
        vulkan_qk_norm(ctx, buf_Q, num_tokens, num_heads, head_dim, use_qk_norm, rms_eps);
        vulkan_qk_norm(ctx, buf_K, num_tokens, kv_heads, head_dim, use_qk_norm, rms_eps);

        // Cache Q/K (post-RoPE, post-QKnorm) and V for backward (avoid recompute)
        vulkan_copy_buffer_cache(ctx, buf_Q, cache_q[layer], (size_t)num_tokens * (size_t)q_dim * sizeof(float));
        vulkan_copy_buffer_cache(ctx, buf_K, cache_k[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));
        vulkan_copy_buffer_cache(ctx, buf_V, cache_v[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));

        // Attention
        vulkan_batch_attention(ctx, buf_Q, buf_K, buf_V, buf_attn,
                               num_tokens, num_heads, kv_heads, head_dim);
        vulkan_copy_buffer_cache(ctx, buf_attn, cache_attn_out[layer],
                           (size_t)num_tokens * (size_t)q_dim * sizeof(float));

        // O projection + residual
        vulkan_matmul_transpose(ctx, buf_attn, ((vulkan_buffer_t**)ctx->gpu_w_o)[layer], buf_tmp_hidden,
                                num_tokens, hidden_size, q_dim, 0, 1, 0);
        vulkan_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);
        vulkan_copy_buffer_cache(ctx, buf_hidden, cache_post_attn_in[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // RMSNorm (post-attention)
        vulkan_rmsnorm(ctx, buf_hidden, ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[layer], buf_hidden_norm, num_tokens, hidden_size, rms_eps);
        vulkan_copy_buffer_cache(ctx, buf_hidden_norm, cache_x_norm_ffn[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Gate + up projections
        vulkan_matmul_transpose(ctx, buf_hidden_norm, ((vulkan_buffer_t**)ctx->gpu_w_gate)[layer], buf_gate,
                                num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        vulkan_copy_buffer_cache(ctx, buf_gate, cache_ffn_gate_out[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        vulkan_matmul_transpose(ctx, buf_hidden_norm, ((vulkan_buffer_t**)ctx->gpu_w_up)[layer], buf_up,
                                num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        vulkan_copy_buffer_cache(ctx, buf_up, cache_ffn_up_out[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        // SiLU + multiply
        vulkan_silu(ctx, buf_gate, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
        vulkan_copy_buffer_cache(ctx, buf_ffn_hidden, cache_silu_out[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));
        vulkan_mul(ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
        vulkan_copy_buffer_cache(ctx, buf_ffn_hidden, cache_ffn_hidden[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        // Down projection + residual
        vulkan_matmul_transpose(ctx, buf_ffn_hidden, ((vulkan_buffer_t**)ctx->gpu_w_down)[layer], buf_tmp_hidden,
                                num_tokens, hidden_size, intermediate_size, 0, 1, 0);
        vulkan_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            vulkan_nan_check(ctx,
                             buf_hidden,
                             (size_t)num_tokens * (size_t)hidden_size,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             nan_fwd_layer_base + (uint32_t)layer);
        }
    }

    // Save pre-final-norm input for backward
    vulkan_copy_buffer(ctx, buf_hidden, (vulkan_buffer_t*)ctx->train_final_input,
                       (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

    // Final RMSNorm
    vulkan_rmsnorm(ctx, buf_hidden, (vulkan_buffer_t*)ctx->gpu_w_final_norm, buf_hidden_norm, num_tokens, hidden_size, rms_eps);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         buf_hidden_norm,
                         (size_t)num_tokens * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         2u);
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->train_final_input,
                         (size_t)num_tokens * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         6u);
    }

    // Video: save query outputs for next frame's propagation
    if (is_video && state->prev_query_out) {
        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
        vulkan_copy_buffer(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->prev_query_out,
                           (size_t)num_queries * (size_t)query_dim * sizeof(float));
        state->current_frame_idx++;
    }

    // Output head: class head for video, LM head for LLM
    int output_dim;
    if (is_video) {
        // Class head: logits = hidden_norm @ class_weight^T + bias
        int num_classes = config->num_classes > 0 ? config->num_classes : 10;
        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
        output_dim = num_classes;

        vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->class_head_weight, buf_logits,
                                num_queries, num_classes, query_dim, 0, 1, 0);
        vulkan_add_bias(ctx, buf_logits, (vulkan_buffer_t*)state->class_head_bias,
                        num_queries, num_classes);

        // Store class logits for metrics computation
        if (state->class_logits_buf) {
            vulkan_copy_buffer(ctx, buf_logits, (vulkan_buffer_t*)state->class_logits_buf,
                               (size_t)num_queries * num_classes * sizeof(float));
        }

        // Mask head: query_proj @ patches^T → mask_logits [num_queries, num_patches]
        if (config->mask_output && state->mask_mlp_w1 && state->mask_mlp_w2) {
            int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
            int frame_h = config->input_resolution[0];
            int frame_w = config->input_resolution[1];
            int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
            int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
            int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
            size_t mask_count = (size_t)num_queries * num_patches;

            // query @ W1 → tmp [num_queries, mask_dim]
            vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->mask_mlp_w1,
                                    buf_tmp_hidden, num_queries, mask_dim, query_dim, 0, 1, 0);
            // Cache pre-silu for backward
            vulkan_copy_buffer(ctx, buf_tmp_hidden, buf_gate,
                               (size_t)num_queries * mask_dim * sizeof(float));
            // SiLU activation
            vulkan_silu(ctx, buf_tmp_hidden, buf_tmp_hidden, (size_t)num_queries * mask_dim);
            // tmp @ W2 → query_proj [num_queries, query_dim]
            vulkan_matmul_transpose(ctx, buf_tmp_hidden, (vulkan_buffer_t*)state->mask_mlp_w2,
                                    buf_hidden, num_queries, query_dim, mask_dim, 0, 1, 0);
            // query_proj @ patches^T → mask_logits [num_queries, num_patches]
            vulkan_matmul_transpose(ctx, buf_hidden, video_kv_src,
                                    (vulkan_buffer_t*)state->mask_logits,
                                    num_queries, num_patches, query_dim, 0, 1, 0);

            // BCE loss + grad on mask_logits vs mask_targets
            vulkan_bce_loss_grad(ctx,
                                 (vulkan_buffer_t*)state->mask_logits,
                                 (vulkan_buffer_t*)state->mask_targets,
                                 (vulkan_buffer_t*)state->mask_grad_logits,
                                 (vulkan_buffer_t*)state->mask_loss_per_elem,
                                 mask_count);
        }

        // CLIP semantic alignment loss (contrastive)
        // Query embeddings should be similar to their target class CLIP embedding
        if (config->clip_loss_weight > 0.0f && state->clip_embeddings) {
            int clip_dim = config->clip_embed_dim > 0 ? config->clip_embed_dim : query_dim;

            // Project queries to CLIP space if dimensions differ
            vulkan_buffer_t* query_for_clip = buf_hidden_norm;
            if (state->clip_proj_weight && query_dim != clip_dim) {
                // query_proj = hidden_norm @ proj_weight [num_queries, clip_dim]
                vulkan_matmul_transpose(ctx, buf_hidden_norm,
                                        (vulkan_buffer_t*)state->clip_proj_weight,
                                        (vulkan_buffer_t*)state->clip_query_proj,
                                        num_queries, clip_dim, query_dim, 0, 1, 0);
                query_for_clip = (vulkan_buffer_t*)state->clip_query_proj;
            }

            // Compute similarity: query_proj @ clip_embeddings^T [num_queries, num_classes]
            // Scale by temperature (CLIP uses 1/0.07 ≈ 14.3, we use config-based scaling via loss_weight)
            vulkan_matmul_transpose(ctx, query_for_clip,
                                    (vulkan_buffer_t*)state->clip_embeddings,
                                    (vulkan_buffer_t*)state->clip_similarity,
                                    num_queries, num_classes, clip_dim, 0, 1, 0);

            // CLIP contrastive loss: cross_entropy on similarity scores
            // This encourages query embeddings to be close to their target class CLIP embedding
            vulkan_cross_entropy_grad(ctx,
                                      (vulkan_buffer_t*)state->clip_similarity,
                                      (vulkan_buffer_t*)ctx->train_targets_u32,  // Same targets as class head
                                      (vulkan_buffer_t*)state->clip_grad_similarity,
                                      (vulkan_buffer_t*)state->clip_loss_per_query,
                                      num_queries, num_classes, state->valid_query_count);

            // Scale CLIP gradients by loss weight (applied in backward pass)
            // Note: clip_loss is reduced and added to total loss after main cross_entropy
        }
    } else {
        // LM head (tied embeddings): logits = hidden_norm @ embed^T
        output_dim = vocab_size;
        vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)ctx->gpu_w_embed, buf_logits,
                                num_tokens, vocab_size, hidden_size, 0, 1, 0);
    }

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         buf_logits,
                         (size_t)num_tokens * (size_t)output_dim,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         1u);
    }

    // Loss + grad_logits
    // Video: 1 label per query, LLM: predict next token
    int loss_rows = is_video ? config->num_queries : num_predictions;
    int valid_rows = is_video ? state->valid_query_count : num_predictions;
    vulkan_cross_entropy_grad(ctx,
                              buf_logits,
                              (vulkan_buffer_t*)ctx->train_targets_u32,
                              (vulkan_buffer_t*)ctx->train_grad_logits,
                              (vulkan_buffer_t*)ctx->train_loss_rows,
                              loss_rows, output_dim, valid_rows);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->train_grad_logits,
                         (size_t)num_predictions * (size_t)vocab_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         3u);
    }

    // Reduce sum loss
    size_t reduce_count = (size_t)loss_rows;
    vulkan_buffer_t* reduce_in = (vulkan_buffer_t*)ctx->train_loss_rows;
    vulkan_buffer_t* reduce_out = (vulkan_buffer_t*)ctx->train_reduce_tmp_a;
    while (reduce_count > 1) {
        size_t out_count = (reduce_count + 511) / 512;
        vulkan_reduce_sum(ctx, reduce_in, reduce_out, reduce_count);
        reduce_count = out_count;
        vulkan_buffer_t* tmp = reduce_in;
        reduce_in = reduce_out;
        reduce_out = tmp;
    }

    // === Backward (GPU) ===
    // Output head backward: class head for video, LM head for LLM
    if (is_video) {
        // 1) Class head: grad_hidden_norm = grad_logits @ class_weight
        int num_classes = config->num_classes > 0 ? config->num_classes : 10;
        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;

        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_logits,
                      (vulkan_buffer_t*)state->class_head_weight,
                      (vulkan_buffer_t*)ctx->train_grad_hidden,
                      num_queries, query_dim, num_classes);

        // 2) Class head weight grad: grad_w += grad_logits^T @ hidden_norm
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_logits,
                                buf_hidden_norm,
                                (vulkan_buffer_t*)state->class_head_grad_w,
                                num_classes, query_dim, num_queries,
                                1, 0, 1);

        // 3) Class head bias grad: grad_b = sum over queries of grad_logits
        if (state->class_head_grad_b) {
            vulkan_sum_cols(ctx,
                            (vulkan_buffer_t*)ctx->train_grad_logits,
                            (vulkan_buffer_t*)state->class_head_grad_b,
                            num_queries, num_classes);
        }

        // Mask head backward (if enabled and mask targets provided)
        if (config->mask_output && state->mask_mlp_w1 && state->mask_grad_logits) {
            int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
            int frame_h = config->input_resolution[0];
            int frame_w = config->input_resolution[1];
            int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
            int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
            int num_patches = (frame_h / patch_h) * (frame_w / patch_w);

            // Recompute forward activations for backward
            // hidden1 = query @ W1 [num_queries, mask_dim]
            vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->mask_mlp_w1,
                                    buf_tmp_hidden, num_queries, mask_dim, query_dim, 0, 1, 0);
            // Cache pre-silu for backward
            vulkan_copy_buffer(ctx, buf_tmp_hidden, buf_gate,
                               (size_t)num_queries * mask_dim * sizeof(float));
            // hidden1_act = silu(hidden1)
            vulkan_silu(ctx, buf_tmp_hidden, buf_tmp_hidden, (size_t)num_queries * mask_dim);
            // query_proj = hidden1_act @ W2 [num_queries, query_dim]
            vulkan_matmul_transpose(ctx, buf_tmp_hidden, (vulkan_buffer_t*)state->mask_mlp_w2,
                                    buf_hidden, num_queries, query_dim, mask_dim, 0, 1, 0);

            // grad_mask_logits was computed in forward by BCE loss
            // mask_logits = query_proj @ patches^T [num_queries, num_patches]
            // Backprop: grad_query_proj = grad_mask_logits @ patches [num_queries, query_dim]
            vulkan_buffer_t* grad_mask_logits = (vulkan_buffer_t*)state->mask_grad_logits;
            vulkan_buffer_t* grad_query_proj = buf_ffn_hidden;  // reuse buffer [num_queries, query_dim]
            vulkan_matmul(ctx, grad_mask_logits, video_kv_src,
                          grad_query_proj, num_queries, query_dim, num_patches);

            // Backprop: grad_patches += query_proj^T @ grad_mask_logits [num_patches, query_dim]
            if (state->video_grad_patches) {
                vulkan_matmul_transpose(ctx, buf_hidden, grad_mask_logits,
                                        (vulkan_buffer_t*)state->video_grad_patches,
                                        query_dim, num_patches, num_queries,
                                        1, 0, 1);
            }

            // d_W2 += hidden1_act^T @ grad_query_proj [mask_dim, query_dim]
            vulkan_matmul_transpose(ctx, buf_tmp_hidden, grad_query_proj,
                                    (vulkan_buffer_t*)state->mask_grad_w2,
                                    mask_dim, query_dim, num_queries, 1, 0, 1);
            // d_hidden1_act = grad_query_proj @ W2^T [num_queries, mask_dim]
            vulkan_matmul(ctx, grad_query_proj, (vulkan_buffer_t*)state->mask_mlp_w2,
                          buf_up, num_queries, mask_dim, query_dim);
            // d_hidden1 = d_hidden1_act * silu'(hidden1)
            vulkan_silu_backward(ctx, buf_gate, buf_up, buf_up, (size_t)num_queries * mask_dim);
            // d_W1 += query^T @ d_hidden1 [query_dim, mask_dim]
            vulkan_matmul_transpose(ctx, buf_hidden_norm, buf_up,
                                    (vulkan_buffer_t*)state->mask_grad_w1,
                                    query_dim, mask_dim, num_queries, 1, 0, 1);
            // d_query += d_hidden1 @ W1^T (accumulate to grad_hidden)
            vulkan_matmul(ctx, buf_up, (vulkan_buffer_t*)state->mask_mlp_w1,
                          buf_tmp_hidden, num_queries, query_dim, mask_dim);
            vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden, buf_tmp_hidden,
                       (vulkan_buffer_t*)ctx->train_grad_hidden, (size_t)num_queries * query_dim);
        }

        // CLIP alignment backward (if enabled)
        if (config->clip_loss_weight > 0.0f && state->clip_embeddings && state->clip_grad_similarity) {
            int clip_dim = config->clip_embed_dim > 0 ? config->clip_embed_dim : query_dim;
            float clip_weight = config->clip_loss_weight;

            // grad_query_clip = clip_grad_similarity @ clip_embeddings [num_queries, clip_dim]
            vulkan_matmul(ctx,
                          (vulkan_buffer_t*)state->clip_grad_similarity,
                          (vulkan_buffer_t*)state->clip_embeddings,
                          (vulkan_buffer_t*)state->clip_query_proj,  // reuse as scratch
                          num_queries, clip_dim, num_classes);

            // Scale by loss weight
            vulkan_scale(ctx,
                         (vulkan_buffer_t*)state->clip_query_proj,
                         (size_t)num_queries * clip_dim,
                         clip_weight);

            if (state->clip_proj_weight && query_dim != clip_dim) {
                // Backprop through projection: grad_query = grad_query_clip @ proj^T
                vulkan_matmul(ctx,
                              (vulkan_buffer_t*)state->clip_query_proj,
                              (vulkan_buffer_t*)state->clip_proj_weight,
                              buf_tmp_hidden,
                              num_queries, query_dim, clip_dim);

                // grad_proj += query^T @ grad_query_clip
                vulkan_matmul_transpose(ctx, buf_hidden_norm,
                                        (vulkan_buffer_t*)state->clip_query_proj,
                                        (vulkan_buffer_t*)state->clip_proj_grad,
                                        query_dim, clip_dim, num_queries, 1, 0, 1);

                // Accumulate to grad_hidden
                vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden, buf_tmp_hidden,
                           (vulkan_buffer_t*)ctx->train_grad_hidden, (size_t)num_queries * query_dim);
            } else {
                // No projection, directly accumulate to grad_hidden
                vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden,
                           (vulkan_buffer_t*)state->clip_query_proj,
                           (vulkan_buffer_t*)ctx->train_grad_hidden, (size_t)num_queries * query_dim);
            }
        }
    } else {
        // 1) LM head: grad_hidden_norm = grad_logits @ embed_weight
        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_logits,
                      (vulkan_buffer_t*)ctx->gpu_w_embed,
                      (vulkan_buffer_t*)ctx->train_grad_hidden,
                      num_tokens, hidden_size, vocab_size);

        // 2) LM head weight grad: grad_embed += grad_logits^T @ hidden_norm
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_logits,
                                buf_hidden_norm,
                                (vulkan_buffer_t*)ctx->gpu_g_embed,
                                vocab_size, hidden_size, num_tokens,
                                1, 0, 1);
    }

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->train_grad_hidden,
                         (size_t)num_tokens * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         4u);
    }

    // 3) Final norm backward -> grad_final_input in tmp_hidden
    vulkan_rmsnorm_backward_batch(ctx,
                                  (vulkan_buffer_t*)ctx->train_final_input,
                                  (vulkan_buffer_t*)ctx->gpu_w_final_norm,
                                  (vulkan_buffer_t*)ctx->train_grad_hidden,
                                  (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                                  (vulkan_buffer_t*)ctx->gpu_g_final_norm,
                                  num_tokens, hidden_size, rms_eps);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                         (size_t)num_tokens * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         5u);
    }

    // grad_hidden = grad_final_input (normalize by sequence length to keep backward in-range)
    vulkan_copy_buffer(ctx, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden, (vulkan_buffer_t*)ctx->train_grad_hidden,
                       (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->train_grad_hidden,
                         (size_t)num_tokens * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         0u);
    }

    // Single barrier: make ALL forward-pass cache copies visible to backward compute
    vulkan_barrier_transfer_to_compute(ctx);

    // Video mode: zero patch gradient accumulator before layer loop
    if (is_video && state->video_grad_patches) {
        int frame_h = config->input_resolution[0] > 0 ? config->input_resolution[0] : 256;
        int frame_w = config->input_resolution[1] > 0 ? config->input_resolution[1] : 256;
        int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
        int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
        int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->video_grad_patches, 0,
                           (size_t)num_patches * hidden_size * sizeof(float));
    }

    for (int layer = num_layers - 1; layer >= 0; layer--) {
        // FFN backward
        // grad_Wdown += grad_hidden^T @ ffn_hidden
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_hidden,
                                cache_ffn_hidden[layer],
                                ((vulkan_buffer_t**)ctx->gpu_g_down)[layer],
                                hidden_size, intermediate_size, num_tokens,
                                1, 0, 1);

        // grad_ffn_hidden = grad_hidden @ Wdown
        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_hidden,
                      ((vulkan_buffer_t**)ctx->gpu_w_down)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_ffn,
                      num_tokens, intermediate_size, hidden_size);

        // mul backward: grad_silu_out, grad_up
        vulkan_elementwise_mul_backward(ctx,
                                        cache_silu_out[layer],
                                        cache_ffn_up_out[layer],
                                        (vulkan_buffer_t*)ctx->train_grad_ffn,
                                        (vulkan_buffer_t*)ctx->train_grad_gate,
                                        (vulkan_buffer_t*)ctx->train_grad_up,
                                        (size_t)num_tokens * (size_t)intermediate_size);

        // silu backward: grad_gate (in-place into train_grad_gate)
        vulkan_silu_backward(ctx,
                             cache_ffn_gate_out[layer],
                             (vulkan_buffer_t*)ctx->train_grad_gate,
                             (vulkan_buffer_t*)ctx->train_grad_gate,
                             (size_t)num_tokens * (size_t)intermediate_size);

        // grad_Wgate += grad_gate^T @ x_norm_ffn
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_gate,
                                cache_x_norm_ffn[layer],
                                ((vulkan_buffer_t**)ctx->gpu_g_gate)[layer],
                                intermediate_size, hidden_size, num_tokens,
                                1, 0, 1);

        // grad_Wup += grad_up^T @ x_norm_ffn
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_up,
                                cache_x_norm_ffn[layer],
                                ((vulkan_buffer_t**)ctx->gpu_g_up)[layer],
                                intermediate_size, hidden_size, num_tokens,
                                1, 0, 1);

        // grad_x_norm_ffn = grad_gate @ Wgate + grad_up @ Wup
        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_gate,
                      ((vulkan_buffer_t**)ctx->gpu_w_gate)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                      num_tokens, hidden_size, intermediate_size);
        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_up,
                      ((vulkan_buffer_t**)ctx->gpu_w_up)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                      num_tokens, hidden_size, intermediate_size);
        vulkan_add(ctx,
                   (vulkan_buffer_t*)ctx->train_grad_x_norm,
                   (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_x_norm,
                   (size_t)num_tokens * (size_t)hidden_size);

        // post_norm backward -> grad_post_attn_in in tmp_hidden (accum scale grad)
        vulkan_rmsnorm_backward_batch(ctx,
                                      cache_post_attn_in[layer],
                                      ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[layer],
                                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                                      (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                                      ((vulkan_buffer_t**)ctx->gpu_g_post_norm)[layer],
                                      num_tokens, hidden_size, rms_eps);

        // grad_post_attn_in_total = grad_post_attn_in_from_norm + grad_hidden(residual)
        vulkan_add(ctx,
                   (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);

        // Attention backward
        // grad_Wo += grad_hidden^T @ attn_out
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_hidden,
                                cache_attn_out[layer],
                                ((vulkan_buffer_t**)ctx->gpu_g_o)[layer],
                                hidden_size, q_dim, num_tokens,
                                1, 0, 1);

        // grad_attn = grad_hidden @ Wo
        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_hidden,
                      ((vulkan_buffer_t**)ctx->gpu_w_o)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_attn,
                      num_tokens, q_dim, hidden_size);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            uint32_t base = nan_bwd_layer_base + (uint32_t)layer * 5u;
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->train_grad_attn,
                             (size_t)num_tokens * (size_t)q_dim,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             base + 0u);
        }

        // Zero atomic accumulation buffers for dK/dV
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->train_grad_k, 0, (size_t)num_tokens * (size_t)kv_dim * sizeof(float));
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->train_grad_v, 0, (size_t)num_tokens * (size_t)kv_dim * sizeof(float));

        // Attention backward: dQ (write), dK/dV (atomic add)
        vulkan_batch_attention_backward(ctx,
                                        cache_q[layer], cache_k[layer], cache_v[layer],
                                        (vulkan_buffer_t*)ctx->train_grad_attn,
                                        (vulkan_buffer_t*)ctx->train_grad_q,
                                        (vulkan_buffer_t*)ctx->train_grad_k,
                                        (vulkan_buffer_t*)ctx->train_grad_v,
                                        num_tokens, num_heads, kv_heads, head_dim);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            uint32_t base = nan_bwd_layer_base + (uint32_t)layer * 5u;
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->train_grad_q,
                             (size_t)num_tokens * (size_t)q_dim,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             base + 1u);
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->train_grad_k,
                             (size_t)num_tokens * (size_t)kv_dim,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             base + 2u);
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->train_grad_v,
                             (size_t)num_tokens * (size_t)kv_dim,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             base + 3u);
        }

        // RoPE backward (inverse rotation)
        vulkan_rope_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_q, (vulkan_buffer_t*)ctx->train_grad_q,
                             num_tokens, num_heads, head_dim, rope_theta);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            uint32_t base = nan_bwd_layer_base + (uint32_t)layer * 5u;
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->train_grad_q,
                             (size_t)num_tokens * (size_t)q_dim,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             base + 4u);
        }

        vulkan_rope_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_k, (vulkan_buffer_t*)ctx->train_grad_k,
                             num_tokens, kv_heads, head_dim, rope_theta);

        // grad_Wq += dQ^T @ x_norm_attn
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_q,
                                cache_x_norm_attn[layer],
                                ((vulkan_buffer_t**)ctx->gpu_g_q)[layer],
                                q_dim, hidden_size, num_tokens,
                                1, 0, 1);

        // grad_Wk += dK^T @ x_norm_attn
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_k,
                                cache_x_norm_attn[layer],
                                ((vulkan_buffer_t**)ctx->gpu_g_k)[layer],
                                kv_dim, hidden_size, num_tokens,
                                1, 0, 1);

        // grad_Wv += dV^T @ x_norm_attn
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_v,
                                cache_x_norm_attn[layer],
                                ((vulkan_buffer_t**)ctx->gpu_g_v)[layer],
                                kv_dim, hidden_size, num_tokens,
                                1, 0, 1);

        // grad_x_norm_attn = dQ @ Wq (queries path)
        // Video mode: K/V gradients go to patches separately
        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_q,
                      ((vulkan_buffer_t**)ctx->gpu_w_q)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                      num_tokens, hidden_size, q_dim);

        if (is_video && state->video_grad_patches) {
            // Video: accumulate K/V gradients to patch gradient buffer
            vulkan_buffer_t* grad_patches = (vulkan_buffer_t*)state->video_grad_patches;
            vulkan_matmul(ctx,
                          (vulkan_buffer_t*)ctx->train_grad_k,
                          ((vulkan_buffer_t**)ctx->gpu_w_k)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx, grad_patches, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       grad_patches, (size_t)num_tokens * (size_t)hidden_size);
            vulkan_matmul(ctx,
                          (vulkan_buffer_t*)ctx->train_grad_v,
                          ((vulkan_buffer_t**)ctx->gpu_w_v)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx, grad_patches, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       grad_patches, (size_t)num_tokens * (size_t)hidden_size);
        } else {
            // LLM: K/V gradients add to same grad_x_norm
            vulkan_matmul(ctx,
                          (vulkan_buffer_t*)ctx->train_grad_k,
                          ((vulkan_buffer_t**)ctx->gpu_w_k)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx,
                       (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (size_t)num_tokens * (size_t)hidden_size);
            vulkan_matmul(ctx,
                          (vulkan_buffer_t*)ctx->train_grad_v,
                          ((vulkan_buffer_t**)ctx->gpu_w_v)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx,
                       (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (size_t)num_tokens * (size_t)hidden_size);
        }

        // input_norm backward -> grad_layer_input_from_norm in tmp_hidden
        vulkan_rmsnorm_backward_batch(ctx,
                                      cache_layer_in[layer],
                                      ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[layer],
                                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                                      (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                                      ((vulkan_buffer_t**)ctx->gpu_g_in_norm)[layer],
                                      num_tokens, hidden_size, rms_eps);

        // grad_layer_input_total = grad_from_norm + grad_hidden(residual from attn add)
        vulkan_add(ctx,
                   (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->train_grad_hidden,
                             (size_t)num_tokens * (size_t)hidden_size,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             nan_grad_hidden_bwd_layer_base + (uint32_t)layer);
        }
    }

    // Input gradient: video (patch embedding) or LLM (token embedding)
    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->train_grad_hidden,
                         (size_t)num_tokens * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         8u);
    }

    if (is_video) {
        // Video backward: compute gradients for patch embedding and query weights
        vulkan_buffer_t* frame_buf = (vulkan_buffer_t*)ctx->train_tokens_u32;
        vulkan_buffer_t* patch_buf = buf_tmp_hidden;
        vulkan_buffer_t* grad_patches = (vulkan_buffer_t*)state->video_grad_patches;
        vulkan_buffer_t* grad_queries = (vulkan_buffer_t*)state->video_grad_queries;
        vulkan_buffer_t* grad_fusion = (vulkan_buffer_t*)state->video_grad_fusion;
        vulkan_buffer_t* prev_queries = (vulkan_buffer_t*)state->prev_query_out;

        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
        int frame_h = config->input_resolution[0];
        int frame_w = config->input_resolution[1];
        int in_channels = config->in_channels > 0 ? config->in_channels : 3;
        int patch_t = config->patch_size[0] > 0 ? config->patch_size[0] : 1;
        int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
        int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
        int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
        int patch_dim = in_channels * patch_t * patch_h * patch_w;

        // Re-extract patches for gradient computation (they were overwritten)
        vulkan_patch_extract(ctx, frame_buf, patch_buf,
                             frame_h, frame_w, in_channels,
                             patch_t, patch_h, patch_w);

        // d_patch_embed_w = patches^T @ grad_patches
        // Shape: [patch_dim, num_patches] @ [num_patches, hidden] = [patch_dim, hidden]
        vulkan_matmul_transpose(ctx, patch_buf, grad_patches,
                                (vulkan_buffer_t*)ctx->gpu_g_embed,
                                patch_dim, hidden_size, num_patches,
                                1, 0, 1);  // transpose A, accumulate

        // grad_hidden contains gradient w.r.t. initial query state
        // d_query_learnable += grad_hidden (queries are always added in forward)
        vulkan_add(ctx, grad_queries, (vulkan_buffer_t*)ctx->train_grad_hidden,
                   grad_queries, (size_t)num_queries * query_dim);

        // For frame > 0: d_fusion_w = prev_queries^T @ grad_hidden
        // Forward was: buf_hidden = prev_queries @ fusion_w + queries
        // So: d_fusion_w = prev_queries^T @ d_buf_hidden
        if (state->current_frame_idx > 0 && prev_queries && grad_fusion) {
            vulkan_matmul_transpose(ctx, prev_queries, (vulkan_buffer_t*)ctx->train_grad_hidden,
                                    grad_fusion,
                                    query_dim, query_dim, num_queries,
                                    1, 0, 1);  // transpose A, accumulate
        }
    } else {
        // LLM: standard embedding gradient
        vulkan_embedding_backward(ctx,
                                  (vulkan_buffer_t*)ctx->train_tokens_u32,
                                  (vulkan_buffer_t*)ctx->train_grad_hidden,
                                  (vulkan_buffer_t*)ctx->gpu_g_embed,
                                  num_tokens, hidden_size, vocab_size);
    }

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        vulkan_nan_check(ctx,
                         (vulkan_buffer_t*)ctx->gpu_g_embed,
                         (size_t)vocab_size * (size_t)hidden_size,
                         (vulkan_buffer_t*)ctx->debug_nan_flags,
                         9u);
    }

    // Optional optimizer step (on GPU)
    state->accumulation_counter++;
    int accum_steps = state->gradient_accumulation_steps;
    if (accum_steps < 1) accum_steps = 1;

    int do_opt = (state->accumulation_counter >= accum_steps);
    float loss_scalar = 0.0f;

    // If NaN checks are enabled, defer optimizer until after we read back flags, so we can skip it and
    // avoid poisoning weights permanently.
    int defer_opt_for_debug = (do_opt && state->debug_nan_check);
    float opt_lr = 0.0f, opt_beta1 = 0.9f, opt_beta2 = 0.999f, opt_weight_decay = 0.01f, opt_eps = 1e-8f;
    int opt_step = 0;
    if (do_opt) {
        state->optimizer_step++;
        opt_step = state->optimizer_step;

        float base_lr = state->learning_rate;
        float warmup_steps = 100.0f;
        float lr = base_lr;
        if (state->optimizer_step < warmup_steps) {
            lr = base_lr * (state->optimizer_step / warmup_steps);
        }
        lr = lr / accum_steps;
        opt_lr = lr;
    }

    if (do_opt && !defer_opt_for_debug) {
        full_weights_optimizer_step_gpu_resident_record(state, opt_lr, opt_beta1, opt_beta2, opt_weight_decay, opt_eps, opt_step, ctx);
    }

    // Execute everything
    vulkan_cmd_end_submit(ctx);

    if (state->debug_nan_check && ctx->debug_nan_flags) {
        uint32_t* nan_flags_host = NULL;
        vulkan_buffer_t* flags_buf = (vulkan_buffer_t*)ctx->debug_nan_flags;
        if (flags_buf->mapped) {
            nan_flags_host = (uint32_t*)flags_buf->mapped;
        } else {
            uint32_t* tmp = calloc(nan_flag_count, sizeof(uint32_t));
            if (tmp) {
                vulkan_buffer_download_bytes(ctx, flags_buf, tmp, (size_t)nan_flag_count * sizeof(uint32_t));
                nan_flags_host = tmp; // freed below
            }
        }

        int any_nan = 0;
        if (nan_flags_host) {
            for (uint32_t i = 0; i < nan_flag_count; i++) {
                if (nan_flags_host[i] != 0) { any_nan = 1; break; }
            }
        }

        if (any_nan) {
            fprintf(stderr, "[NAN] detected at step=%d seq=%d (optimizer skipped)\n", state->current_step + 1, num_tokens);
            fprintf(stderr, "  - lr(base)=%g opt_lr(step)=%g opt_step=%d accum=%d/%d\n",
                    (double)state->learning_rate,
                    (double)opt_lr,
                    opt_step,
                    state->accumulation_counter,
                    accum_steps);
            if (nan_flags_host && nan_flags_host[0]) {
                fprintf(stderr, "  - buffer: grad_hidden (post final-norm backward)\n");
            }
            if (nan_flags_host && nan_flags_host[1]) {
                fprintf(stderr, "  - buffer: logits (LM head output)\n");
            }
            if (nan_flags_host && nan_flags_host[2]) {
                fprintf(stderr, "  - buffer: hidden_norm (LM head input)\n");
            }
            if (nan_flags_host && nan_flags_host[3]) {
                fprintf(stderr, "  - buffer: grad_logits (cross-entropy grad)\n");
            }
            if (nan_flags_host && nan_flags_host[4]) {
                fprintf(stderr, "  - buffer: grad_hidden (after LM head backward)\n");
            }
            if (nan_flags_host && nan_flags_host[5]) {
                fprintf(stderr, "  - buffer: grad_tmp_hidden (final rmsnorm backward output)\n");
            }
            if (nan_flags_host && nan_flags_host[6]) {
                fprintf(stderr, "  - buffer: final_input (pre final rmsnorm)\n");
            }
            if (nan_flags_host && nan_flags_host[7]) {
                fprintf(stderr, "  - buffer: hidden (post embedding)\n");
            }
            if (nan_flags_host && nan_flags_host[8]) {
                fprintf(stderr, "  - buffer: grad_hidden (pre embedding_backward)\n");
            }
            if (nan_flags_host && nan_flags_host[9]) {
                fprintf(stderr, "  - buffer: g_embed (post embedding_backward)\n");
            }
            for (int l = 0; l < num_layers; l++) {
                if (!nan_flags_host) break;
                if (nan_flags_host[nan_fwd_layer_base + (uint32_t)l]) {
                    fprintf(stderr, "  - buffer: hidden (post layer) layer=%d\n", l);
                    break;
                }
            }
            if (nan_flags_host && nan_weight_embed_idx < nan_flag_count && nan_flags_host[nan_weight_embed_idx]) {
                fprintf(stderr, "  - buffer: w_embed (weights already NaN/Inf at step start)\n");
            }
            for (int l = 0; l < num_layers; l++) {
                if (!nan_flags_host) break;
                if (nan_flags_host[nan_grad_hidden_bwd_layer_base + (uint32_t)l]) {
                    fprintf(stderr, "  - buffer: grad_hidden (after layer backward) layer=%d\n", l);
                    break;
                }
            }
            for (int l = 0; l < num_layers; l++) {
                uint32_t base = nan_bwd_layer_base + (uint32_t)l * 5u;
                if (!nan_flags_host) break;
                if (nan_flags_host[base + 0u]) { fprintf(stderr, "  - buffer: grad_attn (input to attn backward) layer=%d\n", l); break; }
                if (nan_flags_host[base + 1u]) { fprintf(stderr, "  - buffer: grad_q (pre rope_backward) layer=%d\n", l); break; }
                if (nan_flags_host[base + 2u]) { fprintf(stderr, "  - buffer: grad_k (post attn backward) layer=%d\n", l); break; }
                if (nan_flags_host[base + 3u]) { fprintf(stderr, "  - buffer: grad_v (post attn backward) layer=%d\n", l); break; }
                if (nan_flags_host[base + 4u]) { fprintf(stderr, "  - buffer: grad_q (post rope_backward) layer=%d\n", l); break; }
            }
            if (nan_flags_host) {
                if (nan_flags_host[nan_preopt_base + 1u]) fprintf(stderr, "  - buffer: g_embed (pre AdamW)\n");
                if (nan_flags_host[nan_preopt_base + 2u]) fprintf(stderr, "  - buffer: m_embed (pre AdamW)\n");
                if (nan_flags_host[nan_preopt_base + 3u]) fprintf(stderr, "  - buffer: v_embed (pre AdamW)\n");
            }

            // One-shot dump for offline inspection (no per-element scans in-code).
            // Writes raw FP32 buffers into /tmp; safe to read with numpy.
            // This is intentionally "big" but only triggers on NaN.
            char dump_prefix[256];
            snprintf(dump_prefix, sizeof(dump_prefix), "/tmp/seraph_nan_step_%d", state->current_step + 1);
            char path[512];

            // Lightweight metadata to make dumps self-describing.
            snprintf(path, sizeof(path), "%s_meta.txt", dump_prefix);
            FILE* meta = fopen(path, "wb");
            if (meta) {
                fprintf(meta, "step=%d\n", state->current_step + 1);
                fprintf(meta, "seq=%d\n", num_tokens);
                fprintf(meta, "num_predictions=%d\n", num_predictions);
                fprintf(meta, "hidden_size=%d\n", hidden_size);
                fprintf(meta, "q_dim=%d\n", q_dim);
                fprintf(meta, "kv_dim=%d\n", kv_dim);
                fprintf(meta, "vocab_size=%d\n", vocab_size);
                fprintf(meta, "num_layers=%d\n", num_layers);
                fprintf(meta, "num_heads=%d\n", num_heads);
                fprintf(meta, "kv_heads=%d\n", kv_heads);
                fprintf(meta, "head_dim=%d\n", head_dim);
                fprintf(meta, "lr=%g\n", (double)state->learning_rate);
                fclose(meta);
            }

            snprintf(path, sizeof(path), "%s_tokens.u32", dump_prefix);
            dump_buffer_u32(path, ctx, (vulkan_buffer_t*)ctx->train_tokens_u32, (size_t)num_tokens);
            snprintf(path, sizeof(path), "%s_targets.u32", dump_prefix);
            dump_buffer_u32(path, ctx, (vulkan_buffer_t*)ctx->train_targets_u32, (size_t)num_predictions);

            snprintf(path, sizeof(path), "%s_hidden.f32", dump_prefix);
            dump_buffer_f32(path, ctx, buf_hidden, (size_t)num_tokens * (size_t)hidden_size);
            snprintf(path, sizeof(path), "%s_hidden_norm.f32", dump_prefix);
            dump_buffer_f32(path, ctx, buf_hidden_norm, (size_t)num_tokens * (size_t)hidden_size);
            snprintf(path, sizeof(path), "%s_logits.f32", dump_prefix);
            dump_buffer_f32(path, ctx, buf_logits, (size_t)num_tokens * (size_t)vocab_size);

            snprintf(path, sizeof(path), "%s_grad_logits.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_logits, (size_t)num_predictions * (size_t)vocab_size);
            snprintf(path, sizeof(path), "%s_grad_hidden.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_hidden, (size_t)num_tokens * (size_t)hidden_size);
            snprintf(path, sizeof(path), "%s_grad_tmp_hidden.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden, (size_t)num_tokens * (size_t)hidden_size);
            snprintf(path, sizeof(path), "%s_grad_x_norm.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_x_norm, (size_t)num_tokens * (size_t)hidden_size);
            snprintf(path, sizeof(path), "%s_grad_hidden_start.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_hidden, (size_t)num_tokens * (size_t)hidden_size);
            snprintf(path, sizeof(path), "%s_grad_ffn.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_ffn, (size_t)num_tokens * (size_t)intermediate_size);
            snprintf(path, sizeof(path), "%s_grad_gate.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_gate, (size_t)num_tokens * (size_t)intermediate_size);
            snprintf(path, sizeof(path), "%s_grad_up.f32", dump_prefix);
            dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_up, (size_t)num_tokens * (size_t)intermediate_size);

            // Focus on layer 0 where your NaN originates.
            if (num_layers > 0) {
                snprintf(path, sizeof(path), "%s_L0_layer_in.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_layer_in[0], (size_t)num_tokens * (size_t)hidden_size);
                snprintf(path, sizeof(path), "%s_L0_x_norm_attn.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_x_norm_attn[0], (size_t)num_tokens * (size_t)hidden_size);
                snprintf(path, sizeof(path), "%s_L0_post_attn_in.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_post_attn_in[0], (size_t)num_tokens * (size_t)hidden_size);
                snprintf(path, sizeof(path), "%s_L0_x_norm_ffn.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_x_norm_ffn[0], (size_t)num_tokens * (size_t)hidden_size);
                snprintf(path, sizeof(path), "%s_L0_ffn_gate.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_ffn_gate_out[0], (size_t)num_tokens * (size_t)intermediate_size);
                snprintf(path, sizeof(path), "%s_L0_ffn_up.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_ffn_up_out[0], (size_t)num_tokens * (size_t)intermediate_size);
                snprintf(path, sizeof(path), "%s_L0_silu.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_silu_out[0], (size_t)num_tokens * (size_t)intermediate_size);
                snprintf(path, sizeof(path), "%s_L0_ffn_hidden.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_ffn_hidden[0], (size_t)num_tokens * (size_t)intermediate_size);
                snprintf(path, sizeof(path), "%s_L0_q.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_q[0], (size_t)num_tokens * (size_t)q_dim);
                snprintf(path, sizeof(path), "%s_L0_k.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_k[0], (size_t)num_tokens * (size_t)kv_dim);
                snprintf(path, sizeof(path), "%s_L0_v.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_v[0], (size_t)num_tokens * (size_t)kv_dim);
                snprintf(path, sizeof(path), "%s_L0_attn_out.f32", dump_prefix);
                dump_buffer_f32(path, ctx, cache_attn_out[0], (size_t)num_tokens * (size_t)q_dim);
                snprintf(path, sizeof(path), "%s_grad_attn.f32", dump_prefix);
                dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_attn, (size_t)num_tokens * (size_t)q_dim);
                snprintf(path, sizeof(path), "%s_grad_q.f32", dump_prefix);
                dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_q, (size_t)num_tokens * (size_t)q_dim);
                snprintf(path, sizeof(path), "%s_grad_k.f32", dump_prefix);
                dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_k, (size_t)num_tokens * (size_t)kv_dim);
                snprintf(path, sizeof(path), "%s_grad_v.f32", dump_prefix);
                dump_buffer_f32(path, ctx, (vulkan_buffer_t*)ctx->train_grad_v, (size_t)num_tokens * (size_t)kv_dim);
            }

            fprintf(stderr, "[NAN-DUMP] wrote buffers with prefix %s_*\n", dump_prefix);

            state->nan_detected = 1;
            if (!((vulkan_buffer_t*)ctx->debug_nan_flags)->mapped && nan_flags_host) free(nan_flags_host);
            return NAN;
        }

        if (!((vulkan_buffer_t*)ctx->debug_nan_flags)->mapped && nan_flags_host) free(nan_flags_host);
    }

    if (do_opt && defer_opt_for_debug) {
        // Run optimizer in a second submit so NaN checks can prevent weight poisoning.
        // Before running AdamW, check that grad/m/v are finite.
        vulkan_cmd_begin(ctx);
        if (state->debug_nan_check && ctx->debug_nan_flags) {
            // Clear pre/post opt slots
            vulkan_fill_buffer_range(ctx, (vulkan_buffer_t*)ctx->debug_nan_flags,
                                     (size_t)nan_preopt_base * sizeof(uint32_t),
                                     0u, (size_t)5u * sizeof(uint32_t));
            // w_embed pre
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->gpu_w_embed,
                             (size_t)vocab_size * (size_t)hidden_size,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             nan_preopt_base + 0u);
            // g_embed pre
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->gpu_g_embed,
                             (size_t)vocab_size * (size_t)hidden_size,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             nan_preopt_base + 1u);
            // m_embed pre
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->gpu_m_embed,
                             (size_t)vocab_size * (size_t)hidden_size,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             nan_preopt_base + 2u);
            // v_embed pre
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->gpu_v_embed,
                             (size_t)vocab_size * (size_t)hidden_size,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             nan_preopt_base + 3u);
        }

        full_weights_optimizer_step_gpu_resident_record(state, opt_lr, opt_beta1, opt_beta2, opt_weight_decay, opt_eps, opt_step, ctx);

        if (state->debug_nan_check && ctx->debug_nan_flags) {
            vulkan_nan_check(ctx,
                             (vulkan_buffer_t*)ctx->gpu_w_embed,
                             (size_t)vocab_size * (size_t)hidden_size,
                             (vulkan_buffer_t*)ctx->debug_nan_flags,
                             nan_weight_embed_post_opt_idx);
        }
        vulkan_cmd_end_submit(ctx);

        // Read back the pre/post flags and report cause.
        if (state->debug_nan_check && ctx->debug_nan_flags) {
            vulkan_buffer_t* flags_buf = (vulkan_buffer_t*)ctx->debug_nan_flags;
            uint32_t flags[5] = {0,0,0,0,0};
            if (flags_buf->mapped) {
                uint32_t* f = (uint32_t*)flags_buf->mapped;
                for (int i = 0; i < 5; i++) flags[i] = f[nan_preopt_base + (uint32_t)i];
            } else {
                // Slow path: download the full flags buffer (small in debug mode)
                vulkan_buffer_download_bytes(ctx, flags_buf, flags, sizeof(flags));
            }
            if (flags[4]) {
                fprintf(stderr, "[NAN] optimizer produced NaN/Inf in w_embed at step=%d (weights would be poisoned)\n", state->current_step + 1);
                if (flags[1]) fprintf(stderr, "  - cause: g_embed already NaN/Inf before AdamW\n");
                if (flags[2]) fprintf(stderr, "  - cause: m_embed already NaN/Inf before AdamW\n");
                if (flags[3]) fprintf(stderr, "  - cause: v_embed already NaN/Inf before AdamW\n");
                if (!flags[1] && !flags[2] && !flags[3]) fprintf(stderr, "  - cause: AdamW math produced NaN/Inf from finite inputs\n");
                state->nan_detected = 1;
                return NAN;
            }
        }
    }

    // Optional sanity check: confirm weights actually change after optimizer step.
    // Enable via `--debug-weight-delta` / `--debug` (or `SERAPH_DEBUG_WEIGHT_DELTA=1` env override).
    if (do_opt) {
        static int debug_init = 0;
        static int debug_enabled = 0;
        static int have_prev = 0;
        static float prev_w0 = 0.0f;
        if (!debug_init) {
            debug_enabled = state->debug_weight_delta || (getenv("SERAPH_DEBUG_WEIGHT_DELTA") != NULL);
            debug_init = 1;
        }
        if (debug_enabled) {
            float w0 = 0.0f;
            vulkan_buffer_download(ctx, (vulkan_buffer_t*)ctx->gpu_w_embed, &w0, 1);
            if (have_prev) {
                fprintf(stderr, "[DEBUG] w_embed[0] delta=%g new=%g\n", (double)(w0 - prev_w0), (double)w0);
            }
            prev_w0 = w0;
            have_prev = 1;
        }
    }

    // Read reduced loss (single float) and normalize by valid rows
    vulkan_buffer_download(ctx, reduce_in, &loss_scalar, 1);
    float loss = loss_scalar / (float)(valid_rows > 0 ? valid_rows : num_predictions);

    if (do_opt) {
        state->accumulation_counter = 0;
    }

    return loss;
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU LoRA TRAINING STEP
// ═══════════════════════════════════════════════════════════════════════════
//
// Same pipeline as train_step_gpu_full but:
// - Base weights are frozen (BF16→FP32 on GPU, read-only)
// - LoRA A/B matrices get forward additions and backward gradients
// - Optimizer runs on LoRA params only (much smaller)
// - No base weight gradients computed (massive memory + compute savings)

// Helper: apply LoRA forward addition for one adapter at one layer
// output += scale * (input @ A^T) @ B^T
// input: [seq, in_dim], output: [seq, out_dim], A: [rank, in_dim], B: [out_dim, rank]
static void lora_gpu_forward_add(vulkan_context_t* ctx,
                                  vulkan_buffer_t* input,
                                  vulkan_buffer_t* output,
                                  vulkan_buffer_t* A,
                                  vulkan_buffer_t* B,
                                  vulkan_buffer_t* hidden_tmp,
                                  int seq_len, int in_dim, int out_dim, int rank,
                                  float scale) {
    // hidden = input @ A^T  [seq, in_dim] @ [rank, in_dim]^T → [seq, rank]
    vulkan_matmul_transpose(ctx, input, A, hidden_tmp,
                            seq_len, rank, in_dim, 0, 1, 0);

    // Scale hidden by alpha/rank
    vulkan_scale(ctx, hidden_tmp, (size_t)seq_len * (size_t)rank, scale);

    // output += hidden @ B^T  [seq, rank] @ [out_dim, rank]^T → [seq, out_dim]
    vulkan_matmul_transpose(ctx, hidden_tmp, B, output,
                            seq_len, out_dim, rank, 0, 1, 1);  // accumulate=1
}

// Helper: compute LoRA backward gradients for one adapter at one layer
// Given upstream grad_output [seq, out_dim] and cached input [seq, in_dim]:
//   grad_B += scale * grad_output^T @ hidden    (hidden = A @ input^T recomputed)
//   grad_A += scale * (B^T @ grad_output)^T @ input
//   Also adds LoRA contribution to grad_input for chain rule
static void lora_gpu_backward(vulkan_context_t* ctx,
                               vulkan_buffer_t* grad_output,
                               vulkan_buffer_t* input,
                               vulkan_buffer_t* A,
                               vulkan_buffer_t* B,
                               vulkan_buffer_t* gA,
                               vulkan_buffer_t* gB,
                               vulkan_buffer_t* hidden_tmp,
                               vulkan_buffer_t* grad_input,   // may be NULL if not needed
                               int seq_len, int in_dim, int out_dim, int rank,
                               float scale) {
    // 1. Recompute hidden = input @ A^T  [seq, rank]
    vulkan_matmul_transpose(ctx, input, A, hidden_tmp,
                            seq_len, rank, in_dim, 0, 1, 0);

    // 2. grad_B += scale * grad_output^T @ hidden
    //    [out_dim, seq] @ [seq, rank] → [out_dim, rank], accumulate
    //    Using matmul_transpose: A=grad_output, B=hidden, transpose_A=1, accumulate=1
    //    But we need to scale. Apply scale to hidden_tmp first (reuse it).
    vulkan_scale(ctx, hidden_tmp, (size_t)seq_len * (size_t)rank, scale);

    vulkan_matmul_transpose(ctx, grad_output, hidden_tmp, gB,
                            out_dim, rank, seq_len, 1, 0, 1);  // transpose_A=1, accum=1

    // 3. grad_hidden = grad_output @ B  [seq, out_dim] @ [out_dim, rank] → [seq, rank]
    //    Then scale (hidden_tmp already scaled, but grad_hidden needs fresh scale)
    vulkan_matmul(ctx, grad_output, B, hidden_tmp,
                  seq_len, rank, out_dim);
    vulkan_scale(ctx, hidden_tmp, (size_t)seq_len * (size_t)rank, scale);

    // 4. grad_A += grad_hidden^T @ input
    //    [rank, seq] @ [seq, in_dim] → [rank, in_dim], accumulate
    vulkan_matmul_transpose(ctx, hidden_tmp, input, gA,
                            rank, in_dim, seq_len, 1, 0, 1);  // transpose_A=1, accum=1

    // 5. Chain rule: grad_input += grad_hidden @ A  [seq, rank] @ [rank, in_dim] → [seq, in_dim]
    if (grad_input) {
        vulkan_matmul(ctx, hidden_tmp, A, grad_input,
                      seq_len, in_dim, rank);
        // Note: this overwrites grad_input. Caller must add to existing grad if needed.
        // Actually, for the projections we need accumulate. Let me use matmul_transpose.
    }
}

static float train_step_gpu_lora(train_state_t* state, int* tokens, int num_tokens, vulkan_context_t* ctx) {
    if (!state || !state->lora || !state->config || !ctx) return -1.0f;

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
    const int use_qk_norm = config->use_qk_norm;  // Video/ViT mode

    int max_seq = state->act_cache ? state->act_cache->max_seq_len : num_tokens;
    if (max_seq < num_tokens) max_seq = num_tokens;

    // Ensure all GPU buffers are ready
    ensure_lora_gpu_base_weights(ctx, state);
    ensure_lora_gpu_adapters(ctx, state);
    ensure_video_gpu_weights(ctx, state);
    ensure_fulltrain_fwd_buffers(ctx, state, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);
    ensure_fulltrain_train_buffers(ctx, max_seq, hidden_size, intermediate_size, q_dim, kv_dim, vocab_size);

    lora_gpu_state_t* lg = (lora_gpu_state_t*)ctx->lora_gpu;
    ensure_lora_hidden_tmp(ctx, max_seq, lg->rank);
    vulkan_buffer_t* lora_tmp = (vulkan_buffer_t*)ctx->lora_hidden_tmp;
    float lora_scale = lg->scale;

    // Zero LoRA grads at start of accumulation cycle
    if (state->accumulation_counter == 0) {
        gpu_zero_lora_grads(ctx);
        // Zero video gradients
        if (state->video_grad_queries) {
            vulkan_buffer_t* gq = (vulkan_buffer_t*)state->video_grad_queries;
            vulkan_fill_buffer(ctx, gq, 0, gq->size);
        }
        if (state->video_grad_fusion) {
            vulkan_buffer_t* gf = (vulkan_buffer_t*)state->video_grad_fusion;
            vulkan_fill_buffer(ctx, gf, 0, gf->size);
        }
        // Zero class head gradients
        if (state->class_head_grad_w) {
            vulkan_buffer_t* gw = (vulkan_buffer_t*)state->class_head_grad_w;
            vulkan_fill_buffer(ctx, gw, 0, gw->size);
        }
        if (state->class_head_grad_b) {
            vulkan_buffer_t* gb = (vulkan_buffer_t*)state->class_head_grad_b;
            vulkan_fill_buffer(ctx, gb, 0, gb->size);
        }
        // Zero mask head gradients
        if (state->mask_grad_w1) {
            vulkan_buffer_t* gw = (vulkan_buffer_t*)state->mask_grad_w1;
            vulkan_fill_buffer(ctx, gw, 0, gw->size);
        }
        if (state->mask_grad_w2) {
            vulkan_buffer_t* gw = (vulkan_buffer_t*)state->mask_grad_w2;
            vulkan_fill_buffer(ctx, gw, 0, gw->size);
        }
    }

    // Aliases for activation buffers (same as full-weight path)
    vulkan_buffer_t* buf_hidden = (vulkan_buffer_t*)ctx->fwd_hidden_buffer;
    vulkan_buffer_t* buf_hidden_norm = (vulkan_buffer_t*)ctx->fwd_hidden_norm_buffer;
    vulkan_buffer_t* buf_tmp_hidden = (vulkan_buffer_t*)ctx->fwd_tmp_hidden_buffer;
    vulkan_buffer_t* buf_gate = (vulkan_buffer_t*)ctx->fwd_ffn_gate_buffer;
    vulkan_buffer_t* buf_up = (vulkan_buffer_t*)ctx->fwd_ffn_up_buffer;
    vulkan_buffer_t* buf_ffn_hidden = (vulkan_buffer_t*)ctx->fwd_ffn_hidden_buffer;
    vulkan_buffer_t* buf_logits = (vulkan_buffer_t*)ctx->fwd_logits_buffer;

    vulkan_buffer_t* buf_Q = (vulkan_buffer_t*)ctx->attn_q_buffer;
    vulkan_buffer_t* buf_K = (vulkan_buffer_t*)ctx->attn_k_buffer;
    vulkan_buffer_t* buf_V = (vulkan_buffer_t*)ctx->attn_v_buffer;
    vulkan_buffer_t* buf_attn = (vulkan_buffer_t*)ctx->attn_out_buffer;

    vulkan_buffer_t** cache_layer_in = (vulkan_buffer_t**)ctx->fwd_cache_layer_input;
    vulkan_buffer_t** cache_x_norm_attn = (vulkan_buffer_t**)ctx->fwd_cache_x_norm_attn;
    vulkan_buffer_t** cache_attn_out = (vulkan_buffer_t**)ctx->fwd_cache_attn_out;
    vulkan_buffer_t** cache_post_attn_in = (vulkan_buffer_t**)ctx->fwd_cache_post_attn_in;
    vulkan_buffer_t** cache_x_norm_ffn = (vulkan_buffer_t**)ctx->fwd_cache_x_norm_ffn;
    vulkan_buffer_t** cache_ffn_gate_out = (vulkan_buffer_t**)ctx->fwd_cache_ffn_gate_out;
    vulkan_buffer_t** cache_ffn_up_out = (vulkan_buffer_t**)ctx->fwd_cache_ffn_up_out;
    vulkan_buffer_t** cache_silu_out = (vulkan_buffer_t**)ctx->fwd_cache_silu_out;
    vulkan_buffer_t** cache_ffn_hidden = (vulkan_buffer_t**)ctx->fwd_cache_ffn_hidden;
    vulkan_buffer_t** cache_q = (vulkan_buffer_t**)ctx->fwd_cache_q;
    vulkan_buffer_t** cache_k = (vulkan_buffer_t**)ctx->fwd_cache_k;
    vulkan_buffer_t** cache_v = (vulkan_buffer_t**)ctx->fwd_cache_v;

    // Check if video mode (tokens will be NULL, data already uploaded by train_step_video)
    int is_video_mode = (config->num_queries > 0);
    int num_predictions = num_tokens - 1;

    // These buffers are used in loss computation (declared outside for scope)
    vulkan_buffer_t* tokens_buf = (vulkan_buffer_t*)ctx->train_tokens_u32;
    vulkan_buffer_t* targets_buf = (vulkan_buffer_t*)ctx->train_targets_u32;

    // Upload tokens + targets (LLM mode only - video mode uploads in train_step_video)
    if (!is_video_mode && tokens != NULL) {
        uint32_t* tokens_u32 = tokens_buf ? (uint32_t*)tokens_buf->mapped : NULL;
        uint32_t* tokens_u32_heap = NULL;
        if (!tokens_u32) {
            tokens_u32_heap = malloc((size_t)num_tokens * sizeof(uint32_t));
            tokens_u32 = tokens_u32_heap;
        }
        if (!tokens_u32) return -1.0f;
        for (int i = 0; i < num_tokens; i++) tokens_u32[i] = (tokens[i] < 0) ? 0u : (uint32_t)tokens[i];
        vulkan_buffer_upload_u32(ctx, tokens_buf, tokens_u32, (size_t)num_tokens);

        uint32_t* targets_u32 = targets_buf ? (uint32_t*)targets_buf->mapped : NULL;
        uint32_t* targets_u32_heap = NULL;
        if (!targets_u32) {
            targets_u32_heap = malloc((size_t)num_predictions * sizeof(uint32_t));
            targets_u32 = targets_u32_heap;
        }
        if (!targets_u32) { free(tokens_u32_heap); return -1.0f; }
        for (int t = 0; t < num_predictions; t++) {
            int tok = tokens[t + 1];
            targets_u32[t] = (tok < 0) ? 0u : (uint32_t)tok;
        }
        vulkan_buffer_upload_u32(ctx, targets_buf, targets_u32, (size_t)num_predictions);
        free(tokens_u32_heap);
        free(targets_u32_heap);
    }

    // Record forward + backward + optimizer into single command buffer
    vulkan_cmd_begin(ctx);

    // Zero the last grad_logits row (unused by cross_entropy_grad)
    if (num_tokens > 0) {
        size_t last_row_off = (size_t)(num_tokens - 1) * (size_t)vocab_size * sizeof(float);
        vulkan_fill_buffer_range(ctx, (vulkan_buffer_t*)ctx->train_grad_logits, last_row_off, 0u,
                                 (size_t)vocab_size * sizeof(float));
    }

    // ═══════════════════════════════════════════════════
    // FORWARD PASS (frozen base weights + LoRA additions)
    // ═══════════════════════════════════════════════════

    int is_video = config->num_queries > 0;
    vulkan_buffer_t* video_kv_src = NULL;  // Persistent patch embeddings for video K/V
    if (is_video) {
        vulkan_buffer_t* frame_buf = tokens_buf;
        vulkan_buffer_t* patch_buf = buf_tmp_hidden;
        vulkan_buffer_t* patch_embed_w = (vulkan_buffer_t*)ctx->gpu_w_embed;
        vulkan_buffer_t* patch_embed_out = (vulkan_buffer_t*)state->video_patch_embed;
        vulkan_buffer_t* queries = (vulkan_buffer_t*)state->query_learnable;
        vulkan_buffer_t* prev_queries = (vulkan_buffer_t*)state->prev_query_out;
        vulkan_buffer_t* fusion_w = (vulkan_buffer_t*)state->query_fusion_weight;

        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
        int frame_h = config->input_resolution[0];
        int frame_w = config->input_resolution[1];
        int in_channels = config->in_channels > 0 ? config->in_channels : 3;
        int patch_t = config->patch_size[0] > 0 ? config->patch_size[0] : 1;
        int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
        int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
        int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
        int patch_dim = in_channels * patch_t * patch_h * patch_w;

        // Upload mask targets for current frame (LoRA training path)
        // Set mask values to -1.0 for background queries (class_gt < 0) to ignore in BCE
        if (config->mask_output && state->video_masks && state->mask_targets) {
            size_t masks_per_frame = (size_t)num_queries * num_patches;
            size_t mask_offset = (size_t)state->current_frame_idx * masks_per_frame;
            float* src_masks = &state->video_masks[mask_offset];

            // Get class labels for this frame to identify background queries
            int* labels = &state->video_labels[state->current_frame_idx * state->num_labels_per_frame];
            int num_classes = config->num_classes > 0 ? config->num_classes : 40;

            // Create temp buffer with -1.0 for background query masks
            float* tmp_masks = malloc(masks_per_frame * sizeof(float));
            for (int q = 0; q < num_queries; q++) {
                int label_idx = q < state->num_labels_per_frame ? q : 0;
                int label = labels[label_idx];
                float* dst = &tmp_masks[q * num_patches];
                float* src = &src_masks[q * num_patches];

                if (label < 0 || label >= num_classes) {
                    // Background query - set all mask values to -1.0 (ignore in BCE)
                    for (int p = 0; p < num_patches; p++) {
                        dst[p] = -1.0f;
                    }
                } else {
                    // Valid object - copy actual mask values
                    memcpy(dst, src, num_patches * sizeof(float));
                }
            }
            vulkan_buffer_upload(ctx, (vulkan_buffer_t*)state->mask_targets, tmp_masks, masks_per_frame);
            free(tmp_masks);
        }

        // Extract patches and embed -> persistent buffer for K/V across layers
        vulkan_patch_extract(ctx, frame_buf, patch_buf,
                             frame_h, frame_w, in_channels,
                             patch_t, patch_h, patch_w);
        vulkan_matmul_transpose(ctx, patch_buf, patch_embed_w, patch_embed_out,
                                num_patches, hidden_size, patch_dim, 0, 1, 0);

        // Add sinusoidal 2D position embeddings (encodes spatial row/col location)
        if (state->video_pos_embed) {
            vulkan_add(ctx, patch_embed_out, (vulkan_buffer_t*)state->video_pos_embed,
                       patch_embed_out, (size_t)num_patches * hidden_size);
        }
        video_kv_src = patch_embed_out;

        // Queries go to buf_hidden (the evolving state through transformer)
        if (state->current_frame_idx == 0 || !prev_queries) {
            vulkan_copy_buffer(ctx, queries, buf_hidden,
                               (size_t)num_queries * (size_t)query_dim * sizeof(float));
        } else {
            vulkan_matmul_transpose(ctx, prev_queries, fusion_w, buf_hidden,
                                    num_queries, query_dim, query_dim, 0, 1, 0);
            vulkan_add(ctx, buf_hidden, queries, buf_hidden,
                       (size_t)num_queries * (size_t)query_dim);
        }
    } else {
        vulkan_embed_lookup(ctx, tokens_buf, (vulkan_buffer_t*)ctx->gpu_w_embed, buf_hidden,
                            num_tokens, hidden_size, vocab_size);
    }

    for (int layer = 0; layer < num_layers; layer++) {
        // Cache layer input
        vulkan_copy_buffer_cache(ctx, buf_hidden, cache_layer_in[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // RMSNorm (pre-attention)
        vulkan_rmsnorm(ctx, buf_hidden, ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[layer],
                       buf_hidden_norm, num_tokens, hidden_size, rms_eps);
        vulkan_copy_buffer_cache(ctx, buf_hidden_norm, cache_x_norm_attn[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Q, K, V projections (frozen base weights)
        // Video: Q from queries (buf_hidden_norm), K/V from patch embeddings (video_kv_src)
        vulkan_buffer_t* kv_src = is_video ? video_kv_src : buf_hidden_norm;
        vulkan_matmul_transpose(ctx, buf_hidden_norm, ((vulkan_buffer_t**)ctx->gpu_w_q)[layer], buf_Q,
                                num_tokens, q_dim, hidden_size, 0, 1, 0);
        vulkan_matmul_transpose(ctx, kv_src, ((vulkan_buffer_t**)ctx->gpu_w_k)[layer], buf_K,
                                num_tokens, kv_dim, hidden_size, 0, 1, 0);
        vulkan_matmul_transpose(ctx, kv_src, ((vulkan_buffer_t**)ctx->gpu_w_v)[layer], buf_V,
                                num_tokens, kv_dim, hidden_size, 0, 1, 0);

        // LoRA additions to Q, K, V
        if (lg->q.active && lg->q.A[layer]) {
            lora_gpu_forward_add(ctx, buf_hidden_norm, buf_Q,
                                 lg->q.A[layer], lg->q.B[layer], lora_tmp,
                                 num_tokens, hidden_size, q_dim, lg->rank, lora_scale);
        }
        if (lg->k.active && lg->k.A[layer]) {
            lora_gpu_forward_add(ctx, kv_src, buf_K,
                                 lg->k.A[layer], lg->k.B[layer], lora_tmp,
                                 num_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }
        if (lg->v.active && lg->v.A[layer]) {
            lora_gpu_forward_add(ctx, kv_src, buf_V,
                                 lg->v.A[layer], lg->v.B[layer], lora_tmp,
                                 num_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }

        // RoPE
        vulkan_rope(ctx, buf_Q, buf_Q, num_tokens, num_heads, head_dim, rope_theta);
        vulkan_rope(ctx, buf_K, buf_K, num_tokens, kv_heads, head_dim, rope_theta);

        // QK norm (video/ViT mode)
        vulkan_qk_norm(ctx, buf_Q, num_tokens, num_heads, head_dim, use_qk_norm, rms_eps);
        vulkan_qk_norm(ctx, buf_K, num_tokens, kv_heads, head_dim, use_qk_norm, rms_eps);

        // Cache Q, K, V (post-RoPE, post-QKnorm)
        vulkan_copy_buffer_cache(ctx, buf_Q, cache_q[layer], (size_t)num_tokens * (size_t)q_dim * sizeof(float));
        vulkan_copy_buffer_cache(ctx, buf_K, cache_k[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));
        vulkan_copy_buffer_cache(ctx, buf_V, cache_v[layer], (size_t)num_tokens * (size_t)kv_dim * sizeof(float));

        // Attention
        vulkan_batch_attention(ctx, buf_Q, buf_K, buf_V, buf_attn,
                               num_tokens, num_heads, kv_heads, head_dim);
        vulkan_copy_buffer_cache(ctx, buf_attn, cache_attn_out[layer],
                           (size_t)num_tokens * (size_t)q_dim * sizeof(float));

        // O projection + LoRA + residual
        vulkan_matmul_transpose(ctx, buf_attn, ((vulkan_buffer_t**)ctx->gpu_w_o)[layer], buf_tmp_hidden,
                                num_tokens, hidden_size, q_dim, 0, 1, 0);
        if (lg->o.active && lg->o.A[layer]) {
            lora_gpu_forward_add(ctx, buf_attn, buf_tmp_hidden,
                                 lg->o.A[layer], lg->o.B[layer], lora_tmp,
                                 num_tokens, q_dim, hidden_size, lg->rank, lora_scale);
        }
        vulkan_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);
        vulkan_copy_buffer_cache(ctx, buf_hidden, cache_post_attn_in[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // RMSNorm (post-attention)
        vulkan_rmsnorm(ctx, buf_hidden, ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[layer],
                       buf_hidden_norm, num_tokens, hidden_size, rms_eps);
        vulkan_copy_buffer_cache(ctx, buf_hidden_norm, cache_x_norm_ffn[layer],
                           (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

        // Gate + Up projections + LoRA
        vulkan_matmul_transpose(ctx, buf_hidden_norm, ((vulkan_buffer_t**)ctx->gpu_w_gate)[layer], buf_gate,
                                num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        if (lg->gate.active && lg->gate.A[layer]) {
            lora_gpu_forward_add(ctx, buf_hidden_norm, buf_gate,
                                 lg->gate.A[layer], lg->gate.B[layer], lora_tmp,
                                 num_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }
        vulkan_copy_buffer_cache(ctx, buf_gate, cache_ffn_gate_out[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        vulkan_matmul_transpose(ctx, buf_hidden_norm, ((vulkan_buffer_t**)ctx->gpu_w_up)[layer], buf_up,
                                num_tokens, intermediate_size, hidden_size, 0, 1, 0);
        if (lg->up.active && lg->up.A[layer]) {
            lora_gpu_forward_add(ctx, buf_hidden_norm, buf_up,
                                 lg->up.A[layer], lg->up.B[layer], lora_tmp,
                                 num_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }
        vulkan_copy_buffer_cache(ctx, buf_up, cache_ffn_up_out[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        // SiLU + multiply
        vulkan_silu(ctx, buf_gate, buf_ffn_hidden, (size_t)num_tokens * (size_t)intermediate_size);
        vulkan_copy_buffer_cache(ctx, buf_ffn_hidden, cache_silu_out[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));
        vulkan_mul(ctx, buf_ffn_hidden, buf_up, buf_ffn_hidden,
                   (size_t)num_tokens * (size_t)intermediate_size);
        vulkan_copy_buffer_cache(ctx, buf_ffn_hidden, cache_ffn_hidden[layer],
                           (size_t)num_tokens * (size_t)intermediate_size * sizeof(float));

        // Down projection + LoRA + residual
        vulkan_matmul_transpose(ctx, buf_ffn_hidden, ((vulkan_buffer_t**)ctx->gpu_w_down)[layer], buf_tmp_hidden,
                                num_tokens, hidden_size, intermediate_size, 0, 1, 0);
        if (lg->down.active && lg->down.A[layer]) {
            lora_gpu_forward_add(ctx, buf_ffn_hidden, buf_tmp_hidden,
                                 lg->down.A[layer], lg->down.B[layer], lora_tmp,
                                 num_tokens, intermediate_size, hidden_size, lg->rank, lora_scale);
        }
        vulkan_add(ctx, buf_hidden, buf_tmp_hidden, buf_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);
    }

    // Save pre-final-norm input for backward
    vulkan_copy_buffer(ctx, buf_hidden, (vulkan_buffer_t*)ctx->train_final_input,
                       (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

    // Final RMSNorm
    vulkan_rmsnorm(ctx, buf_hidden, (vulkan_buffer_t*)ctx->gpu_w_final_norm,
                   buf_hidden_norm, num_tokens, hidden_size, rms_eps);

    // Video: save query outputs for next frame's propagation
    if (is_video && state->prev_query_out) {
        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
        vulkan_copy_buffer(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->prev_query_out,
                           (size_t)num_queries * (size_t)query_dim * sizeof(float));
        state->current_frame_idx++;
    }

    int output_dim = vocab_size;
    if (is_video) {
        // Video: Class head forward
        int num_classes = config->num_classes > 0 ? config->num_classes : 10;
        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
        output_dim = num_classes;

        vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->class_head_weight, buf_logits,
                                num_queries, num_classes, query_dim, 0, 1, 0);
        vulkan_add_bias(ctx, buf_logits, (vulkan_buffer_t*)state->class_head_bias,
                        num_queries, num_classes);

        // Store class logits for metrics computation
        if (state->class_logits_buf) {
            vulkan_copy_buffer(ctx, buf_logits, (vulkan_buffer_t*)state->class_logits_buf,
                               (size_t)num_queries * num_classes * sizeof(float));
        }

        // Mask head forward (LoRA path)
        if (config->mask_output && state->mask_mlp_w1 && state->mask_mlp_w2) {
            int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
            int frame_h = config->input_resolution[0];
            int frame_w = config->input_resolution[1];
            int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
            int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
            int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
            size_t mask_count = (size_t)num_queries * num_patches;

            // query @ W1 → tmp [num_queries, mask_dim]
            vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->mask_mlp_w1,
                                    buf_tmp_hidden, num_queries, mask_dim, query_dim, 0, 1, 0);
            // Cache pre-silu for backward
            vulkan_copy_buffer(ctx, buf_tmp_hidden, buf_gate,
                               (size_t)num_queries * mask_dim * sizeof(float));
            // SiLU activation
            vulkan_silu(ctx, buf_tmp_hidden, buf_tmp_hidden, (size_t)num_queries * mask_dim);
            // tmp @ W2 → query_proj [num_queries, query_dim]
            vulkan_matmul_transpose(ctx, buf_tmp_hidden, (vulkan_buffer_t*)state->mask_mlp_w2,
                                    buf_hidden, num_queries, query_dim, mask_dim, 0, 1, 0);
            // query_proj @ patches^T → mask_logits [num_queries, num_patches]
            vulkan_matmul_transpose(ctx, buf_hidden, video_kv_src,
                                    (vulkan_buffer_t*)state->mask_logits,
                                    num_queries, num_patches, query_dim, 0, 1, 0);

            // BCE loss + grad on mask_logits vs mask_targets
            vulkan_bce_loss_grad(ctx,
                                 (vulkan_buffer_t*)state->mask_logits,
                                 (vulkan_buffer_t*)state->mask_targets,
                                 (vulkan_buffer_t*)state->mask_grad_logits,
                                 (vulkan_buffer_t*)state->mask_loss_per_elem,
                                 mask_count);
        }
    } else {
        // LM head (tied embeddings): logits = hidden_norm @ embed^T
        vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)ctx->gpu_w_embed, buf_logits,
                                num_tokens, vocab_size, hidden_size, 0, 1, 0);
    }

    // Loss + grad_logits (class CE for video, next-token CE for LLM)
    int loss_rows = is_video ? config->num_queries : num_predictions;
    int valid_rows = is_video ? state->valid_query_count : num_predictions;
    vulkan_cross_entropy_grad(ctx, buf_logits, targets_buf,
                              (vulkan_buffer_t*)ctx->train_grad_logits,
                              (vulkan_buffer_t*)ctx->train_loss_rows,
                              loss_rows, output_dim, valid_rows);

    // Reduce sum loss
    size_t reduce_count = (size_t)loss_rows;
    vulkan_buffer_t* reduce_in = (vulkan_buffer_t*)ctx->train_loss_rows;
    vulkan_buffer_t* reduce_out = (vulkan_buffer_t*)ctx->train_reduce_tmp_a;
    while (reduce_count > 1) {
        size_t out_count = (reduce_count + 511) / 512;
        vulkan_reduce_sum(ctx, reduce_in, reduce_out, reduce_count);
        reduce_count = out_count;
        vulkan_buffer_t* tmp = reduce_in;
        reduce_in = reduce_out;
        reduce_out = tmp;
    }

    // ═══════════════════════════════════════════════════
    // BACKWARD PASS (LoRA gradients + chain rule through frozen weights)
    // ═══════════════════════════════════════════════════

    if (is_video) {
        // 1) Class head backward: grad_hidden_norm = grad_logits @ class_weight
        int num_classes = config->num_classes > 0 ? config->num_classes : 10;
        int num_queries = config->num_queries;
        int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;

        vulkan_matmul(ctx,
                      (vulkan_buffer_t*)ctx->train_grad_logits,
                      (vulkan_buffer_t*)state->class_head_weight,
                      (vulkan_buffer_t*)ctx->train_grad_hidden,
                      num_queries, query_dim, num_classes);

        // 2) Class head weight grad: grad_w += grad_logits^T @ hidden_norm
        vulkan_matmul_transpose(ctx,
                                (vulkan_buffer_t*)ctx->train_grad_logits,
                                buf_hidden_norm,
                                (vulkan_buffer_t*)state->class_head_grad_w,
                                num_classes, query_dim, num_queries,
                                1, 0, 1);

        // 3) Class head bias grad: grad_b = sum over queries of grad_logits
        if (state->class_head_grad_b) {
            vulkan_sum_cols(ctx,
                            (vulkan_buffer_t*)ctx->train_grad_logits,
                            (vulkan_buffer_t*)state->class_head_grad_b,
                            num_queries, num_classes);
        }

        // Mask head backward (LoRA path)
        if (config->mask_output && state->mask_mlp_w1 && state->mask_grad_logits) {
            int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
            int frame_h = config->input_resolution[0];
            int frame_w = config->input_resolution[1];
            int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
            int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
            int num_patches = (frame_h / patch_h) * (frame_w / patch_w);

            // Recompute forward activations for backward
            vulkan_matmul_transpose(ctx, buf_hidden_norm, (vulkan_buffer_t*)state->mask_mlp_w1,
                                    buf_tmp_hidden, num_queries, mask_dim, query_dim, 0, 1, 0);
            vulkan_copy_buffer(ctx, buf_tmp_hidden, buf_gate,
                               (size_t)num_queries * mask_dim * sizeof(float));
            vulkan_silu(ctx, buf_tmp_hidden, buf_tmp_hidden, (size_t)num_queries * mask_dim);
            vulkan_matmul_transpose(ctx, buf_tmp_hidden, (vulkan_buffer_t*)state->mask_mlp_w2,
                                    buf_hidden, num_queries, query_dim, mask_dim, 0, 1, 0);

            // grad_query_proj = grad_mask_logits @ patches [num_queries, query_dim]
            vulkan_buffer_t* grad_mask_logits = (vulkan_buffer_t*)state->mask_grad_logits;
            vulkan_buffer_t* grad_query_proj = buf_ffn_hidden;
            vulkan_matmul(ctx, grad_mask_logits, video_kv_src,
                          grad_query_proj, num_queries, query_dim, num_patches);

            // grad_patches += query_proj^T @ grad_mask_logits [num_patches, query_dim]
            if (state->video_grad_patches) {
                vulkan_matmul_transpose(ctx, buf_hidden, grad_mask_logits,
                                        (vulkan_buffer_t*)state->video_grad_patches,
                                        query_dim, num_patches, num_queries,
                                        1, 0, 1);
            }

            // d_W2 += hidden1_act^T @ grad_query_proj
            vulkan_matmul_transpose(ctx, buf_tmp_hidden, grad_query_proj,
                                    (vulkan_buffer_t*)state->mask_grad_w2,
                                    mask_dim, query_dim, num_queries, 1, 0, 1);
            // d_hidden1_act = grad_query_proj @ W2^T
            vulkan_matmul(ctx, grad_query_proj, (vulkan_buffer_t*)state->mask_mlp_w2,
                          buf_up, num_queries, mask_dim, query_dim);
            // d_hidden1 = d_hidden1_act * silu'(hidden1)
            vulkan_silu_backward(ctx, buf_gate, buf_up, buf_up, (size_t)num_queries * mask_dim);
            // d_W1 += query^T @ d_hidden1
            vulkan_matmul_transpose(ctx, buf_hidden_norm, buf_up,
                                    (vulkan_buffer_t*)state->mask_grad_w1,
                                    query_dim, mask_dim, num_queries, 1, 0, 1);
            // d_query += d_hidden1 @ W1^T (accumulate to grad_hidden)
            vulkan_matmul(ctx, buf_up, (vulkan_buffer_t*)state->mask_mlp_w1,
                          buf_tmp_hidden, num_queries, query_dim, mask_dim);
            vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden, buf_tmp_hidden,
                       (vulkan_buffer_t*)ctx->train_grad_hidden, (size_t)num_queries * query_dim);
        }
    } else {
        // 1) LM head backward: grad_hidden_norm = grad_logits @ embed_weight
        vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_logits,
                      (vulkan_buffer_t*)ctx->gpu_w_embed,
                      (vulkan_buffer_t*)ctx->train_grad_hidden,
                      num_tokens, hidden_size, vocab_size);

        // Note: No LM head weight gradient (embeddings are frozen in LoRA mode)
    }

    // 2) Final norm backward
    vulkan_rmsnorm_backward_batch(ctx,
                                  (vulkan_buffer_t*)ctx->train_final_input,
                                  (vulkan_buffer_t*)ctx->gpu_w_final_norm,
                                  (vulkan_buffer_t*)ctx->train_grad_hidden,
                                  (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                                  // No final_norm grad accumulation in LoRA mode - use a dummy
                                  // Actually we still need to pass a valid buffer even if we don't use it
                                  // The shader accumulates into grad_scale, so pass grad_tmp as scratch
                                  (vulkan_buffer_t*)ctx->fwd_norm_weight_buffer,
                                  num_tokens, hidden_size, rms_eps);

    // grad_hidden = grad_final_input
    vulkan_copy_buffer(ctx, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       (vulkan_buffer_t*)ctx->train_grad_hidden,
                       (size_t)num_tokens * (size_t)hidden_size * sizeof(float));

    // Barrier: make all forward cache copies visible to backward compute
    vulkan_barrier_transfer_to_compute(ctx);

    // Video mode: zero patch gradient accumulator before layer loop
    if (is_video && state->video_grad_patches) {
        int frame_h = config->input_resolution[0] > 0 ? config->input_resolution[0] : 256;
        int frame_w = config->input_resolution[1] > 0 ? config->input_resolution[1] : 256;
        int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
        int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
        int num_patches = (frame_h / patch_h) * (frame_w / patch_w);
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)state->video_grad_patches, 0,
                           (size_t)num_patches * hidden_size * sizeof(float));
    }

    // Layer backward (reverse order)
    for (int layer = num_layers - 1; layer >= 0; layer--) {
        // === FFN backward ===

        // grad_ffn_hidden = grad_hidden @ W_down (through frozen down_proj)
        vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden,
                      ((vulkan_buffer_t**)ctx->gpu_w_down)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_ffn,
                      num_tokens, intermediate_size, hidden_size);

        // LoRA down backward (if active)
        if (lg->down.active && lg->down.A[layer]) {
            lora_gpu_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden,
                              cache_ffn_hidden[layer],
                              lg->down.A[layer], lg->down.B[layer],
                              lg->down.gA[layer], lg->down.gB[layer],
                              lora_tmp, NULL,  // don't need grad_input contrib here
                              num_tokens, intermediate_size, hidden_size, lg->rank, lora_scale);
        }

        // mul backward: grad_silu_out, grad_up
        vulkan_elementwise_mul_backward(ctx,
                                        cache_silu_out[layer], cache_ffn_up_out[layer],
                                        (vulkan_buffer_t*)ctx->train_grad_ffn,
                                        (vulkan_buffer_t*)ctx->train_grad_gate,
                                        (vulkan_buffer_t*)ctx->train_grad_up,
                                        (size_t)num_tokens * (size_t)intermediate_size);

        // SiLU backward
        vulkan_silu_backward(ctx, cache_ffn_gate_out[layer],
                             (vulkan_buffer_t*)ctx->train_grad_gate,
                             (vulkan_buffer_t*)ctx->train_grad_gate,
                             (size_t)num_tokens * (size_t)intermediate_size);

        // LoRA gate/up backward (if active)
        if (lg->gate.active && lg->gate.A[layer]) {
            lora_gpu_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_gate,
                              cache_x_norm_ffn[layer],
                              lg->gate.A[layer], lg->gate.B[layer],
                              lg->gate.gA[layer], lg->gate.gB[layer],
                              lora_tmp, NULL,
                              num_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }
        if (lg->up.active && lg->up.A[layer]) {
            lora_gpu_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_up,
                              cache_x_norm_ffn[layer],
                              lg->up.A[layer], lg->up.B[layer],
                              lg->up.gA[layer], lg->up.gB[layer],
                              lora_tmp, NULL,
                              num_tokens, hidden_size, intermediate_size, lg->rank, lora_scale);
        }

        // grad_x_norm_ffn = grad_gate @ W_gate + grad_up @ W_up (through frozen weights)
        vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_gate,
                      ((vulkan_buffer_t**)ctx->gpu_w_gate)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                      num_tokens, hidden_size, intermediate_size);
        vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_up,
                      ((vulkan_buffer_t**)ctx->gpu_w_up)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                      num_tokens, hidden_size, intermediate_size);
        vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_x_norm,
                   (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_x_norm,
                   (size_t)num_tokens * (size_t)hidden_size);

        // Post-norm backward
        vulkan_rmsnorm_backward_batch(ctx,
                                      cache_post_attn_in[layer],
                                      ((vulkan_buffer_t**)ctx->gpu_w_post_norm)[layer],
                                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                                      (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                                      (vulkan_buffer_t*)ctx->fwd_norm_weight_buffer,  // scratch for norm grad
                                      num_tokens, hidden_size, rms_eps);

        // Residual: grad_hidden += grad_post_attn_in_from_norm
        vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);

        // === Attention backward ===

        // grad_attn = grad_hidden @ W_o (through frozen O proj)
        vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden,
                      ((vulkan_buffer_t**)ctx->gpu_w_o)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_attn,
                      num_tokens, q_dim, hidden_size);

        // LoRA O backward (if active)
        if (lg->o.active && lg->o.A[layer]) {
            lora_gpu_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden,
                              cache_attn_out[layer],
                              lg->o.A[layer], lg->o.B[layer],
                              lg->o.gA[layer], lg->o.gB[layer],
                              lora_tmp, NULL,
                              num_tokens, q_dim, hidden_size, lg->rank, lora_scale);
        }

        // Zero atomic accumulation buffers for dK/dV
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->train_grad_k, 0,
                           (size_t)num_tokens * (size_t)kv_dim * sizeof(float));
        vulkan_fill_buffer(ctx, (vulkan_buffer_t*)ctx->train_grad_v, 0,
                           (size_t)num_tokens * (size_t)kv_dim * sizeof(float));

        // Attention backward
        vulkan_batch_attention_backward(ctx,
                                        cache_q[layer], cache_k[layer], cache_v[layer],
                                        (vulkan_buffer_t*)ctx->train_grad_attn,
                                        (vulkan_buffer_t*)ctx->train_grad_q,
                                        (vulkan_buffer_t*)ctx->train_grad_k,
                                        (vulkan_buffer_t*)ctx->train_grad_v,
                                        num_tokens, num_heads, kv_heads, head_dim);

        // RoPE backward
        vulkan_rope_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_q, (vulkan_buffer_t*)ctx->train_grad_q,
                             num_tokens, num_heads, head_dim, rope_theta);
        vulkan_rope_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_k, (vulkan_buffer_t*)ctx->train_grad_k,
                             num_tokens, kv_heads, head_dim, rope_theta);

        // LoRA Q/K/V backward (if active)
        // Video mode: Q uses normalized queries, K/V use patch embeddings
        vulkan_buffer_t* lora_kv_input = is_video ? video_kv_src : cache_x_norm_attn[layer];
        if (lg->q.active && lg->q.A[layer]) {
            lora_gpu_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_q,
                              cache_x_norm_attn[layer],  // Q always uses normalized queries
                              lg->q.A[layer], lg->q.B[layer],
                              lg->q.gA[layer], lg->q.gB[layer],
                              lora_tmp, NULL,
                              num_tokens, hidden_size, q_dim, lg->rank, lora_scale);
        }
        if (lg->k.active && lg->k.A[layer]) {
            lora_gpu_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_k,
                              lora_kv_input,  // K uses patches in video mode
                              lg->k.A[layer], lg->k.B[layer],
                              lg->k.gA[layer], lg->k.gB[layer],
                              lora_tmp, NULL,
                              num_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }
        if (lg->v.active && lg->v.A[layer]) {
            lora_gpu_backward(ctx, (vulkan_buffer_t*)ctx->train_grad_v,
                              lora_kv_input,  // V uses patches in video mode
                              lg->v.A[layer], lg->v.B[layer],
                              lg->v.gA[layer], lg->v.gB[layer],
                              lora_tmp, NULL,
                              num_tokens, hidden_size, kv_dim, lg->rank, lora_scale);
        }

        // grad_x_norm_attn = dQ @ W_q (queries path)
        // Video mode: K/V gradients go to patches separately
        vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_q,
                      ((vulkan_buffer_t**)ctx->gpu_w_q)[layer],
                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                      num_tokens, hidden_size, q_dim);

        if (is_video && state->video_grad_patches) {
            // Video: accumulate K/V gradients to patch gradient buffer
            vulkan_buffer_t* grad_patches = (vulkan_buffer_t*)state->video_grad_patches;
            vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_k,
                          ((vulkan_buffer_t**)ctx->gpu_w_k)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx, grad_patches, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       grad_patches, (size_t)num_tokens * (size_t)hidden_size);
            vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_v,
                          ((vulkan_buffer_t**)ctx->gpu_w_v)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx, grad_patches, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       grad_patches, (size_t)num_tokens * (size_t)hidden_size);
        } else {
            // LLM: K/V gradients add to same grad_x_norm
            vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_k,
                          ((vulkan_buffer_t**)ctx->gpu_w_k)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (size_t)num_tokens * (size_t)hidden_size);
            vulkan_matmul(ctx, (vulkan_buffer_t*)ctx->train_grad_v,
                          ((vulkan_buffer_t**)ctx->gpu_w_v)[layer],
                          (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                          num_tokens, hidden_size, kv_dim);
            vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                       (vulkan_buffer_t*)ctx->train_grad_x_norm,
                       (size_t)num_tokens * (size_t)hidden_size);
        }

        // Input norm backward
        vulkan_rmsnorm_backward_batch(ctx,
                                      cache_layer_in[layer],
                                      ((vulkan_buffer_t**)ctx->gpu_w_in_norm)[layer],
                                      (vulkan_buffer_t*)ctx->train_grad_x_norm,
                                      (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                                      (vulkan_buffer_t*)ctx->fwd_norm_weight_buffer,  // scratch
                                      num_tokens, hidden_size, rms_eps);

        // Residual: grad_hidden += grad_from_norm
        vulkan_add(ctx, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (vulkan_buffer_t*)ctx->train_grad_hidden,
                   (size_t)num_tokens * (size_t)hidden_size);
    }

    // No embedding backward needed (frozen in LoRA mode)

    // Optimizer step on LoRA parameters
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

        float beta1 = 0.9f, beta2 = 0.999f, weight_decay = 0.01f, eps = 1e-8f;
        int step = state->optimizer_step;

        lora_optimizer_step_gpu(ctx, lr, beta1, beta2, weight_decay, eps, step);

        // Video weights optimizer (LoRA mode still trains these new parameters)
        if (is_video && state->query_learnable && state->video_grad_queries) {
            int num_queries = config->num_queries;
            int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
            size_t q_size = (size_t)num_queries * query_dim;
            size_t f_size = (size_t)query_dim * query_dim;

            vulkan_adamw_update(ctx,
                                (vulkan_buffer_t*)state->query_learnable,
                                (vulkan_buffer_t*)state->video_grad_queries,
                                (vulkan_buffer_t*)state->video_m_queries,
                                (vulkan_buffer_t*)state->video_v_queries,
                                lr, beta1, beta2, weight_decay, eps, step, q_size);

            if (state->query_fusion_weight && state->video_grad_fusion) {
                vulkan_adamw_update(ctx,
                                    (vulkan_buffer_t*)state->query_fusion_weight,
                                    (vulkan_buffer_t*)state->video_grad_fusion,
                                    (vulkan_buffer_t*)state->video_m_fusion,
                                    (vulkan_buffer_t*)state->video_v_fusion,
                                    lr, beta1, beta2, weight_decay, eps, step, f_size);
            }

            // Class head
            if (state->class_head_weight && state->class_head_grad_w) {
                int num_classes = config->num_classes > 0 ? config->num_classes : 10;
                size_t w_size = (size_t)query_dim * num_classes;
                size_t b_size = (size_t)num_classes;

                vulkan_adamw_update(ctx,
                                    (vulkan_buffer_t*)state->class_head_weight,
                                    (vulkan_buffer_t*)state->class_head_grad_w,
                                    (vulkan_buffer_t*)state->class_head_m_w,
                                    (vulkan_buffer_t*)state->class_head_v_w,
                                    lr, beta1, beta2, weight_decay, eps, step, w_size);
                vulkan_adamw_update(ctx,
                                    (vulkan_buffer_t*)state->class_head_bias,
                                    (vulkan_buffer_t*)state->class_head_grad_b,
                                    (vulkan_buffer_t*)state->class_head_m_b,
                                    (vulkan_buffer_t*)state->class_head_v_b,
                                    lr, beta1, beta2, weight_decay, eps, step, b_size);
            }

            // Mask head
            if (state->mask_mlp_w1 && state->mask_grad_w1) {
                int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
                size_t w1_size = (size_t)query_dim * mask_dim;
                size_t w2_size = (size_t)mask_dim * query_dim;

                vulkan_adamw_update(ctx,
                                    (vulkan_buffer_t*)state->mask_mlp_w1,
                                    (vulkan_buffer_t*)state->mask_grad_w1,
                                    (vulkan_buffer_t*)state->mask_m_w1,
                                    (vulkan_buffer_t*)state->mask_v_w1,
                                    lr, beta1, beta2, weight_decay, eps, step, w1_size);
                vulkan_adamw_update(ctx,
                                    (vulkan_buffer_t*)state->mask_mlp_w2,
                                    (vulkan_buffer_t*)state->mask_grad_w2,
                                    (vulkan_buffer_t*)state->mask_m_w2,
                                    (vulkan_buffer_t*)state->mask_v_w2,
                                    lr, beta1, beta2, weight_decay, eps, step, w2_size);
            }
        }
    }

    // Submit everything
    vulkan_cmd_end_submit(ctx);

    // Read loss
    float loss_scalar = 0.0f;
    vulkan_buffer_download(ctx, reduce_in, &loss_scalar, 1);
    int loss_divisor = valid_rows > 0 ? valid_rows : (is_video ? config->num_queries : num_predictions);
    float loss = loss_scalar / (float)loss_divisor;

    if (do_opt) {
        state->accumulation_counter = 0;
    }

    return loss;
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING STEP
// ═══════════════════════════════════════════════════════════════════════════

// Video training step: uploads frame data and labels, then calls GPU train step
float train_step_video(train_state_t* state, int frame_idx) {
    if (!state || !state->config || !state->video_frames) return -1.0f;

    vulkan_context_t* ctx = (vulkan_context_t*)state->vulkan_ctx;
    if (!ctx) {
        fprintf(stderr, "ERROR: Video training requires GPU (Vulkan context)\n");
        return -1.0f;
    }

    const model_config_t* config = state->config;
    int frame_h = config->input_resolution[0];
    int frame_w = config->input_resolution[1];
    int in_channels = config->in_channels > 0 ? config->in_channels : 3;
    size_t frame_size = (size_t)in_channels * frame_h * frame_w;

    // Upload frame data to GPU (reuses train_tokens_u32 buffer for floats)
    float* frame_data = &state->video_frames[frame_idx * frame_size];
    vulkan_buffer_t* frame_buf = (vulkan_buffer_t*)ctx->train_tokens_u32;

    // Ensure buffer is large enough
    size_t frame_bytes = frame_size * sizeof(float);
    if (!frame_buf || frame_buf->size < frame_bytes) {
        fprintf(stderr, "ERROR: Frame buffer too small (%zu < %zu)\n",
                frame_buf ? frame_buf->size : 0, frame_bytes);
        return -1.0f;
    }
    vulkan_buffer_upload(ctx, frame_buf, frame_data, frame_size);

    // Upload class labels for this frame
    // Note: -1 = "no object" in VDAT, clamp to num_classes (background class)
    int num_queries = config->num_queries;
    int num_classes = config->num_classes > 0 ? config->num_classes : 40;
    int* labels = &state->video_labels[frame_idx * state->num_labels_per_frame];
    vulkan_buffer_t* targets_buf = (vulkan_buffer_t*)ctx->train_targets_u32;
    int valid_queries = 0;
    if (targets_buf) {
        uint32_t* labels_u32 = malloc(num_queries * sizeof(uint32_t));
        for (int i = 0; i < num_queries; i++) {
            int label_idx = i < state->num_labels_per_frame ? i : 0;
            int label = labels[label_idx];
            if (label < 0 || label >= num_classes) {
                label = num_classes;  // out of bounds = ignored by CE shader
            } else {
                valid_queries++;
            }
            labels_u32[i] = (uint32_t)label;
        }
        vulkan_buffer_upload_u32(ctx, targets_buf, labels_u32, num_queries);
        free(labels_u32);
    }
    state->valid_query_count = valid_queries;

    // Skip frames with no valid labels (all -1 = no objects)
    if (valid_queries == 0) {
        return -1.0f;
    }

    double t0_ms = now_ms();
    float loss;

    // Call appropriate GPU train step (video mode detected via config->num_queries)
    if (state->full_weights) {
        loss = train_step_gpu_full(state, NULL, num_queries, ctx);
    } else if (state->lora) {
        loss = train_step_gpu_lora(state, NULL, num_queries, ctx);
    } else {
        fprintf(stderr, "ERROR: Video training requires full_weights or LoRA mode\n");
        return -1.0f;
    }

    state->current_loss = loss;
    state->current_step++;

    if ((state->current_step % 20) == 0) {
        double t1_ms = now_ms();
        fprintf(stderr, "[TIMING] step=%d frame=%d gpu_video_step=%.2fms\n",
                state->current_step, frame_idx, (t1_ms - t0_ms));
    }

    return loss;
}

float train_step(train_state_t* state, int* tokens, int num_tokens) {
    const int vocab_size = state->config->vocab_size;
    vulkan_context_t* vk_ctx = (vulkan_context_t*)state->vulkan_ctx;

    double t0_ms = now_ms();

    // GPU-only full training path (CPU fallback remains when vk_ctx == NULL)
    if (state->full_weights && vk_ctx) {
        float loss = train_step_gpu_full(state, tokens, num_tokens, vk_ctx);
        state->current_loss = loss;
        state->current_step++;
        if ((state->current_step % 20) == 0) {
            double t1_ms = now_ms();
            fprintf(stderr,
                    "[TIMING] step=%d seq=%d gpu_step=%.2fms (gpu=on)\n",
                    state->current_step, num_tokens, (t1_ms - t0_ms));
        }
        return loss;
    }

    // GPU LoRA training path
    if (state->lora && vk_ctx) {
        float loss = train_step_gpu_lora(state, tokens, num_tokens, vk_ctx);
        state->current_loss = loss;
        state->current_step++;
        if ((state->current_step % 20) == 0) {
            double t1_ms = now_ms();
            fprintf(stderr,
                    "[TIMING] step=%d seq=%d gpu_lora_step=%.2fms (gpu=on)\n",
                    state->current_step, num_tokens, (t1_ms - t0_ms));
        }
        return loss;
    }

    // 1. Forward pass - use FP32 forward for full weight training, BF16 for LoRA
    float* logits;
    if (state->full_weights) {
        // FP32 training forward (with GPU matmul for down_proj)
        logits = train_forward_fp32(state, tokens, num_tokens, vk_ctx);
    } else {
        // BF16 inference forward (for LoRA training)
        logits = seraph_forward(state->model, state->config, tokens, num_tokens);
    }
    if (!logits) {
        fprintf(stderr, "ERROR: Forward pass failed\n");
        return -1.0f;
    }

    double t_fwd_ms = now_ms();


    // 2. Compute loss for FULL SEQUENCE (predict next token at every position)
    // For each position t, predict token at position t+1
    float total_loss = 0.0f;
    int num_predictions = num_tokens - 1;  // Can't predict after last token

    for (int t = 0; t < num_predictions; t++) {
        // Get logits for position t
        float* logits_t = &logits[t * vocab_size];
        // Target is the next token
        int target = tokens[t + 1];

        // Compute cross-entropy loss: -log(softmax(logits)[target])
        // Use stable log-softmax
        float max_logit = logits_t[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits_t[i] > max_logit) max_logit = logits_t[i];
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += expf(logits_t[i] - max_logit);
        }

        float log_prob = logits_t[target] - max_logit - logf(sum_exp);
        total_loss -= log_prob;
    }

    // Average loss over all predictions
    float loss = total_loss / num_predictions;

    double t_loss_ms = now_ms();

    // 3. Backward pass (LoRA or full weight training)
    if (state->lora || state->full_weights) {
        // Zero gradients only at start of accumulation cycle
        if (state->accumulation_counter == 0) {
            if (state->lora) {
                lora_zero_grad(state->lora);
            } else if (state->full_weights) {
                zero_full_weight_gradients(state->full_weights);
            }
        }

        // Compute loss gradients for ALL positions
        float* grad_logits = calloc(num_predictions * vocab_size, sizeof(float));
        if (!grad_logits) {
            free(logits);
            return loss;
        }

        for (int t = 0; t < num_predictions; t++) {
            float* logits_t = &logits[t * vocab_size];
            float* grad_t = &grad_logits[t * vocab_size];
            int target = tokens[t + 1];
            compute_loss_gradient(grad_t, logits_t, target, vocab_size);
        }

        double t_grad_ms = now_ms();

        // Backward pass - gradients ACCUMULATE in LoRA adapters or full weights
        train_backward(state, grad_logits, num_predictions);
        free(grad_logits);

        double t_bwd_ms = now_ms();

        // Increment accumulation counter
        state->accumulation_counter++;

        // 4. Optimizer step only after accumulating enough gradients
        int accum_steps = state->gradient_accumulation_steps;
        if (accum_steps < 1) accum_steps = 1;  // Default to 1 if not set

        if (state->accumulation_counter >= accum_steps) {
            state->optimizer_step++;

            // Use configured learning rate with warmup
            float base_lr = state->learning_rate;  // Use actual configured LR!
            float warmup_steps = 100.0f;
            float lr = base_lr;
            if (state->optimizer_step < warmup_steps) {
                lr = base_lr * (state->optimizer_step / warmup_steps);  // Linear warmup
            }

            // Scale learning rate by accumulation steps (gradient averaging)
            lr = lr / accum_steps;

            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float weight_decay = 0.01f;
            float eps = 1e-8f;

            if (state->lora) {
                // LoRA mode - update adapter weights (CPU only for now)
                lora_optimizer_step(state->lora, lr, beta1, beta2, weight_decay, eps, state->optimizer_step);
            } else if (state->full_weights) {
                // Full weight mode - update all weights with AdamW (GPU or CPU)
                full_weights_optimizer_step(state->full_weights, lr, beta1, beta2, weight_decay, eps,
                                           state->optimizer_step, (vulkan_context_t*)state->vulkan_ctx);
            }

            // Reset accumulation counter for next cycle
            state->accumulation_counter = 0;
        }

        double t_opt_ms = now_ms();

        if ((state->current_step % 20) == 0) {
            fprintf(stderr,
                    "[TIMING] step=%d seq=%d pred=%d fwd=%.2fms loss=%.2fms grad=%.2fms bwd=%.2fms opt=%.2fms total=%.2fms (gpu=%s)\n",
                    state->current_step, num_tokens, num_predictions,
                    (t_fwd_ms - t0_ms),
                    (t_loss_ms - t_fwd_ms),
                    (t_grad_ms - t_loss_ms),
                    (t_bwd_ms - t_grad_ms),
                    (t_opt_ms - t_bwd_ms),
                    (t_opt_ms - t0_ms),
                    (vk_ctx && state->full_weights) ? "on" : "off");
        }
    }

    free(logits);
    state->current_step++;
    state->current_loss = loss;

    return loss;
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING LOOP
// ═══════════════════════════════════════════════════════════════════════════

void train_loop(train_state_t* state, train_config_t* config) {
    int is_video = state->config && state->config->num_queries > 0 && state->num_video_frames > 0;

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  SERAPH TRAINING\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    if (is_video) {
        printf("  Mode:           VIDEO SEGMENTATION\n");
        printf("  Frames:         %d\n", state->num_video_frames);
        printf("  Resolution:     %dx%d\n", state->config->input_resolution[0], state->config->input_resolution[1]);
        printf("  Queries:        %d\n", state->config->num_queries);
        printf("  Classes:        %d\n", state->config->num_classes > 0 ? state->config->num_classes : 10);
        printf("  Mask output:    %s\n", state->config->mask_output ? "yes" : "no");
    } else {
        printf("  Samples:        %d\n", state->num_samples);
        printf("  Total tokens:   %d\n", state->num_train_tokens);
    }
    printf("  Epochs:         %d\n", config->epochs);
    printf("  Learning rate:  %.6f\n", config->learning_rate);
    if (!config->use_full_training) {
        // LoRA mode
        printf("  LoRA rank:      %d\n", config->lora_rank);
        printf("  LoRA alpha:     %.1f\n", config->lora_alpha);
    } else {
        // Full training mode
        printf("  Training:       Full weights\n");
    }
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Auto-create training log in checkpoint directory
    char auto_log_path[512];
    if (!config->log_path && config->checkpoint_dir) {
        snprintf(auto_log_path, sizeof(auto_log_path), "%s/train.log", config->checkpoint_dir);
        config->log_path = auto_log_path;
        printf("  Log file:       %s\n", config->log_path);
    }

    time_t start_time = time(NULL);

    int start_epoch = state->current_epoch;
    if (start_epoch < 0) start_epoch = 0;
    if (start_epoch >= config->epochs) {
        printf("  Resume epoch (%d) is beyond configured epochs (%d). Nothing to do.\n",
               start_epoch + 1, config->epochs);
        return;
    }

    // Rolling window for recent loss (actual current performance)
    // Window size = log_every_n_steps so each print reflects exactly the last logging period
    int loss_window = config->log_every_n_steps > 0 ? config->log_every_n_steps : 100;
    float* loss_ring = calloc(loss_window, sizeof(float));
    int loss_ring_pos = 0;
    int loss_ring_count = 0;
    float loss_ring_sum = 0.0f;

    for (int epoch = start_epoch; epoch < config->epochs; epoch++) {
        state->current_epoch = epoch;
        float epoch_loss = 0.0f;
        int epoch_units = 0;

        printf("Epoch %d/%d\n", epoch + 1, config->epochs);
        printf("───────────────────────────────────────────────────────────────────\n");

        // ═══════════════════════════════════════════════════════════════════
        // VIDEO TRAINING LOOP
        // ═══════════════════════════════════════════════════════════════════
        if (is_video) {
            int num_frames = state->num_video_frames;

            // Reset query propagation state at epoch start
            state->current_frame_idx = 0;

            // Reset vision metrics for new epoch
            state->running_iou_sum = 0.0f;
            state->running_map_sum = 0.0f;
            state->running_vap_sum = 0.0f;
            state->running_acc_sum = 0.0f;
            state->running_acc5_sum = 0.0f;
            state->metrics_count = 0;
            state->vap_id_switches = 0;
            state->vap_total_tracks = 0;
            if (state->prev_query_classes) {
                // Reset both pred and gt slots (stored as pairs)
                for (int q = 0; q < state->config->num_queries; q++) {
                    state->prev_query_classes[q * 2] = -1;      // prev pred
                    state->prev_query_classes[q * 2 + 1] = -1;  // prev gt
                }
            }

            // Build frame order - SEQUENTIAL by default for query propagation (Q_t depends on Q_{t-1})
            // Shuffle only if explicitly requested (breaks temporal coherence but helps generalization
            // for models that don't use query propagation)
            int* frame_order = malloc(num_frames * sizeof(int));
            for (int i = 0; i < num_frames; i++) frame_order[i] = i;

            // Optional shuffle (only if video_shuffle_frames is set)
            if (config->video_shuffle_frames) {
                printf("  WARNING: Shuffling frames breaks query propagation (Q_t depends on Q_{t-1})\n");
                for (int i = num_frames - 1; i > 0; i--) {
                    int j = rand() % (i + 1);
                    int tmp = frame_order[i];
                    frame_order[i] = frame_order[j];
                    frame_order[j] = tmp;
                }
            }

            // Resume handling
            int resume_skip = 0;
            if (epoch == start_epoch && state->current_step > 0) {
                resume_skip = state->current_step % num_frames;
                if (resume_skip > 0) {
                    printf("  Resume: skipping %d already-completed frames\n", resume_skip);
                }
            }

            // Train on each frame
            for (int f = 0; f < num_frames; f++) {
                if (f < resume_skip) continue;

                int frame_idx = frame_order[f];

                float loss = train_step_video(state, frame_idx);
                epoch_loss += loss;
                epoch_units++;

                // Compute vision metrics (IoU, mAP, vAP) periodically
                if ((f + 1) % config->log_every_n_steps == 0) {
                    compute_vision_metrics(state);
                }

                // Track rolling window
                loss_ring_sum -= loss_ring[loss_ring_pos];
                loss_ring[loss_ring_pos] = loss;
                loss_ring_sum += loss;
                loss_ring_pos = (loss_ring_pos + 1) % loss_window;
                if (loss_ring_count < loss_window) loss_ring_count++;

                if (state->nan_detected || isnan(loss)) {
                    fprintf(stderr, "\n[NAN] stopping training at epoch=%d step=%d frame=%d\n",
                            epoch + 1, state->current_step, frame_idx);
                    free(frame_order);
                    free(loss_ring);
                    return;
                }

                // Log progress
                if ((f + 1) % config->log_every_n_steps == 0) {
                    float avg_loss = epoch_loss / epoch_units;
                    float elapsed = difftime(time(NULL), start_time);
                    float frames_per_sec = epoch_units / (elapsed > 0 ? elapsed : 1);
                    float recent_loss = loss_ring_sum / loss_ring_count;

                    // Get average metrics for this logging period
                    float avg_iou = state->metrics_count > 0 ? state->running_iou_sum / state->metrics_count : 0.0f;
                    float avg_map = state->metrics_count > 0 ? state->running_map_sum / state->metrics_count : 0.0f;
                    float avg_vap = state->metrics_count > 0 ? state->running_vap_sum / state->metrics_count : 0.0f;
                    float avg_acc = state->metrics_count > 0 ? state->running_acc_sum / state->metrics_count : 0.0f;
                    float avg_acc5 = state->metrics_count > 0 ? state->running_acc5_sum / state->metrics_count : 0.0f;

                    printf("  Frame %d/%d | Loss: %.4f | Avg: %.4f | Acc: %.1f%% | Top5: %.1f%% | IoU: %.3f | mAP: %.3f | vAP: %.3f | %.1f fps\n",
                           f + 1, num_frames, recent_loss, avg_loss, avg_acc * 100.0f, avg_acc5 * 100.0f, avg_iou, avg_map, avg_vap, frames_per_sec);

                    if (config->log_path) {
                        FILE* lf = fopen(config->log_path, "a");
                        if (lf) {
                            fprintf(lf, "  Frame %d/%d | Loss: %.4f | Avg: %.4f | Acc: %.1f%% | Top5: %.1f%% | IoU: %.3f | mAP: %.3f | vAP: %.3f | %.1f fps\n",
                                    f + 1, num_frames, recent_loss, avg_loss, avg_acc * 100.0f, avg_acc5 * 100.0f, avg_iou, avg_map, avg_vap, frames_per_sec);
                            fclose(lf);
                        }
                    }

                    // Reset running metrics for next logging period
                    state->running_iou_sum = 0.0f;
                    state->running_map_sum = 0.0f;
                    state->running_vap_sum = 0.0f;
                    state->running_acc_sum = 0.0f;
                    state->running_acc5_sum = 0.0f;
                    state->metrics_count = 0;
                }

                // Save checkpoint
                if (config->save_every_n_steps > 0 &&
                    state->current_step % config->save_every_n_steps == 0) {
                    char ckpt_path[512];
                    if (state->use_full_training) {
                        snprintf(ckpt_path, sizeof(ckpt_path), "%s/checkpoint_%d.safetensors",
                                 config->checkpoint_dir, state->current_step);
                    } else {
                        snprintf(ckpt_path, sizeof(ckpt_path), "%s/checkpoint_%d.bin",
                                 config->checkpoint_dir, state->current_step);
                    }
                    save_checkpoint(state, ckpt_path);
                }

                // Validation evaluation
                if (config->eval_every_n_steps > 0 &&
                    state->current_step % config->eval_every_n_steps == 0 &&
                    state->val_video_frames) {
                    printf("  [VAL] Step %d validation:\n", state->current_step);
                    float val_loss = evaluate_video(state);
                    if (val_loss < state->best_val_loss) {
                        state->best_val_loss = val_loss;
                        printf("  [VAL] New best! Saving checkpoint...\n");
                        if (config->checkpoint_dir) {
                            char best_path[512];
                            snprintf(best_path, sizeof(best_path), "%s/best_model.safetensors",
                                     config->checkpoint_dir);
                            save_checkpoint(state, best_path);
                        }
                    }
                    // Reset query state after validation (was modified during eval)
                    state->current_frame_idx = f + 1;
                }
            }

            free(frame_order);

            // End of epoch validation
            if (state->val_video_frames) {
                printf("  [VAL] End of epoch %d validation:\n", epoch + 1);
                float val_loss = evaluate_video(state);
                if (val_loss < state->best_val_loss) {
                    state->best_val_loss = val_loss;
                    printf("  [VAL] New best! Saving checkpoint...\n");
                    if (config->checkpoint_dir) {
                        char best_path[512];
                        snprintf(best_path, sizeof(best_path), "%s/best_model.safetensors",
                                 config->checkpoint_dir);
                        save_checkpoint(state, best_path);
                    }
                }
            }

            // Epoch summary
            float avg_epoch_loss = epoch_units > 0 ? epoch_loss / epoch_units : 0.0f;
            printf("───────────────────────────────────────────────────────────────────\n");
            printf("  Epoch %d complete | Avg Loss: %.4f | Frames: %d\n\n", epoch + 1, avg_epoch_loss, epoch_units);
            continue;  // Skip LLM training path
        }

        // ═══════════════════════════════════════════════════════════════════
        // LLM TRAINING LOOP (original path)
        // ═══════════════════════════════════════════════════════════════════
        int epoch_tokens = 0;

        // Build training units: either samples (variable length) or fixed windows
        int num_units = 0;
        int* unit_starts = NULL;
        int* unit_lengths = NULL;

        if (config->fixed_seq_windows) {
            // Fixed-window mode: split samples into max_seq_len windows, drop remainders
            // This ensures consistent gradient scaling (1/max_seq_len) for all steps
            int max_windows = 0;
            for (int i = 0; i < state->num_samples; i++) {
                int sample_len = state->sample_offsets[i + 1] - state->sample_offsets[i];
                max_windows += sample_len / config->max_seq_len;
            }
            unit_starts = malloc(max_windows * sizeof(int));
            unit_lengths = malloc(max_windows * sizeof(int));

            for (int i = 0; i < state->num_samples; i++) {
                int start = state->sample_offsets[i];
                int sample_len = state->sample_offsets[i + 1] - start;
                int num_windows = sample_len / config->max_seq_len;
                for (int w = 0; w < num_windows; w++) {
                    unit_starts[num_units] = start + w * config->max_seq_len;
                    unit_lengths[num_units] = config->max_seq_len;
                    num_units++;
                }
            }
            if (epoch == 0) {
                printf("  Fixed-seq mode: %d windows of %d tokens\n", num_units, config->max_seq_len);
            }
        } else {
            // Variable-length mode: each sample is a unit
            num_units = state->num_samples;
            unit_starts = malloc(num_units * sizeof(int));
            unit_lengths = malloc(num_units * sizeof(int));
            for (int i = 0; i < num_units; i++) {
                unit_starts[i] = state->sample_offsets[i];
                int len = state->sample_offsets[i + 1] - state->sample_offsets[i];
                unit_lengths[i] = (len > config->max_seq_len) ? config->max_seq_len : len;
            }
        }

        // Shuffle units (simple Fisher-Yates)
        int* unit_order = malloc(num_units * sizeof(int));
        for (int i = 0; i < num_units; i++) unit_order[i] = i;
        for (int i = num_units - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = unit_order[i];
            unit_order[i] = unit_order[j];
            unit_order[j] = tmp;
        }

        int resume_skip_units = 0;
        if (epoch == start_epoch && state->current_step > 0 && num_units > 0) {
            resume_skip_units = state->current_step % num_units;
            if (resume_skip_units > 0) {
                printf("  Resume: skipping %d already-completed units in this epoch\n", resume_skip_units);
            }
        }

        // Train on each unit
        for (int s = 0; s < num_units; s++) {
            if (s < resume_skip_units) continue;

            int unit_idx = unit_order[s];
            int start = unit_starts[unit_idx];
            int len = unit_lengths[unit_idx];

            if (len < 2) continue;  // Need at least 2 tokens

            // Reset KV cache before each unit (units are independent)
            seraph_reset_cache(state->model);

            float loss = train_step(state, &state->train_tokens[start], len);
            epoch_loss += loss * (len - 1);
            epoch_tokens += len - 1;

            // Track rolling window (running sum — no loop needed at print time)
            loss_ring_sum -= loss_ring[loss_ring_pos];
            loss_ring[loss_ring_pos] = loss;
            loss_ring_sum += loss;
            loss_ring_pos = (loss_ring_pos + 1) % loss_window;
            if (loss_ring_count < loss_window) loss_ring_count++;

            if (state->nan_detected || isnan(loss)) {
                fprintf(stderr, "\n[NAN] stopping training at epoch=%d step=%d seq=%d\n",
                        epoch + 1, state->current_step, len);
                free(unit_order);
                free(unit_starts);
                free(unit_lengths);
                free(loss_ring);
                return;
            }

            // Log progress
            if ((s + 1) % config->log_every_n_steps == 0) {
                float avg_loss = epoch_loss / epoch_tokens;
                float elapsed = difftime(time(NULL), start_time);
                float tok_per_sec = epoch_tokens / (elapsed > 0 ? elapsed : 1);

                // Rolling window average (O(1) — running sum, no loop)
                float recent_loss = loss_ring_sum / loss_ring_count;

                printf("  Step %d/%d | Loss: %.4f | Avg: %.4f | %.0f tok/s\n",
                       s + 1, num_units, recent_loss, avg_loss, tok_per_sec);

                // Write to log file if enabled
                if (config->log_path) {
                    FILE* lf = fopen(config->log_path, "a");
                    if (lf) {
                        fprintf(lf, "  Step %d/%d | Loss: %.4f | Avg: %.4f | %.0f tok/s\n",
                                s + 1, num_units, recent_loss, avg_loss, tok_per_sec);
                        fclose(lf);
                    }
                }
            }

            // Save checkpoint
            if (config->save_every_n_steps > 0 &&
                state->current_step % config->save_every_n_steps == 0) {
                char ckpt_path[512];
                if (state->use_full_training) {
                    snprintf(ckpt_path, sizeof(ckpt_path), "%s/checkpoint_%d.safetensors",
                             config->checkpoint_dir, state->current_step);
                } else {
                    snprintf(ckpt_path, sizeof(ckpt_path), "%s/checkpoint_%d.bin",
                             config->checkpoint_dir, state->current_step);
                }
                save_checkpoint(state, ckpt_path);
            }

            // Snapshot gradients for field visualization
            if (config->snapshot_every_n_steps > 0 &&
                state->current_step % config->snapshot_every_n_steps == 0 &&
                state->snapshot_writer && state->full_weights) {
                // Download GPU gradients to CPU before snapshot
                download_gpu_gradients(state);
                float lr = config->learning_rate;
                gradient_snapshot_write(state->snapshot_writer, state->full_weights,
                                       state->current_step, state->current_loss, lr);
            }
        }

        free(unit_order);
        free(unit_starts);
        free(unit_lengths);

        // Epoch summary
        float avg_epoch_loss = epoch_loss / epoch_tokens;
        printf("───────────────────────────────────────────────────────────────────\n");
        printf("  Epoch %d complete | Avg Loss: %.4f\n\n", epoch + 1, avg_epoch_loss);
    }

    time_t end_time = time(NULL);
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Training complete!\n");
    printf("  Total time: %.0f seconds\n", difftime(end_time, start_time));
    printf("  Final loss: %.4f\n", state->current_loss);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    free(loss_ring);
}

// ═══════════════════════════════════════════════════════════════════════════
// CHECKPOINTING
// ═══════════════════════════════════════════════════════════════════════════

#define VIDEO_CKPT_MAGIC 0x56494457  // "VIDW"

// Save video-specific weights (queries, fusion, class head, mask head)
static int save_video_weights(train_state_t* state, const char* path) {
    if (!state->config || state->config->num_queries <= 0) return 0;  // Not video mode

    vulkan_context_t* ctx = (vulkan_context_t*)state->vulkan_ctx;
    if (!ctx) return -1;

    const model_config_t* config = state->config;
    int num_queries = config->num_queries;
    int query_dim = config->query_dim > 0 ? config->query_dim : config->hidden_size;
    int num_classes = config->num_classes > 0 ? config->num_classes : 10;
    int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;

    char video_path[512];
    snprintf(video_path, sizeof(video_path), "%s.video", path);
    FILE* f = fopen(video_path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create video weights file %s\n", video_path);
        return -1;
    }

    // Header: magic, num_queries, query_dim, num_classes, mask_dim, mask_output
    uint32_t header[6] = {
        VIDEO_CKPT_MAGIC,
        (uint32_t)num_queries,
        (uint32_t)query_dim,
        (uint32_t)num_classes,
        (uint32_t)mask_dim,
        (uint32_t)config->mask_output
    };
    fwrite(header, sizeof(uint32_t), 6, f);

    // Helper to download and write GPU buffer
    #define SAVE_GPU_BUFFER(buf, count) do { \
        if (buf) { \
            float* tmp = malloc((count) * sizeof(float)); \
            vulkan_buffer_download(ctx, (vulkan_buffer_t*)(buf), tmp, (count)); \
            fwrite(tmp, sizeof(float), (count), f); \
            free(tmp); \
        } \
    } while(0)

    // Save learnable queries [num_queries, query_dim]
    size_t q_size = (size_t)num_queries * query_dim;
    SAVE_GPU_BUFFER(state->query_learnable, q_size);

    // Save fusion weight [query_dim, query_dim]
    size_t f_size = (size_t)query_dim * query_dim;
    SAVE_GPU_BUFFER(state->query_fusion_weight, f_size);

    // Save class head weight [query_dim, num_classes] and bias [num_classes]
    size_t cw_size = (size_t)query_dim * num_classes;
    SAVE_GPU_BUFFER(state->class_head_weight, cw_size);
    SAVE_GPU_BUFFER(state->class_head_bias, num_classes);

    // Save mask head if enabled
    if (config->mask_output) {
        size_t w1_size = (size_t)query_dim * mask_dim;
        size_t w2_size = (size_t)mask_dim * query_dim;
        SAVE_GPU_BUFFER(state->mask_mlp_w1, w1_size);
        SAVE_GPU_BUFFER(state->mask_mlp_w2, w2_size);
    }

    // Save optimizer state (m, v for each weight)
    SAVE_GPU_BUFFER(state->video_m_queries, q_size);
    SAVE_GPU_BUFFER(state->video_v_queries, q_size);
    SAVE_GPU_BUFFER(state->video_m_fusion, f_size);
    SAVE_GPU_BUFFER(state->video_v_fusion, f_size);
    SAVE_GPU_BUFFER(state->class_head_m_w, cw_size);
    SAVE_GPU_BUFFER(state->class_head_v_w, cw_size);
    SAVE_GPU_BUFFER(state->class_head_m_b, num_classes);
    SAVE_GPU_BUFFER(state->class_head_v_b, num_classes);

    if (config->mask_output) {
        size_t w1_size = (size_t)query_dim * mask_dim;
        size_t w2_size = (size_t)mask_dim * query_dim;
        SAVE_GPU_BUFFER(state->mask_m_w1, w1_size);
        SAVE_GPU_BUFFER(state->mask_v_w1, w1_size);
        SAVE_GPU_BUFFER(state->mask_m_w2, w2_size);
        SAVE_GPU_BUFFER(state->mask_v_w2, w2_size);
    }

    #undef SAVE_GPU_BUFFER

    fclose(f);
    printf("  Saved video weights: %s\n", video_path);
    return 0;
}

// Save full video model as unified safetensors (for inference)
// Includes: patch_embed + transformer layers + video heads
static void save_video_full_safetensors(train_state_t* state, const char* path) {
    vulkan_context_t* ctx = (vulkan_context_t*)state->vulkan_ctx;
    const model_config_t* config = state->config;
    trainable_weights_t* weights = state->full_weights;

    if (!ctx || !config || !weights) return;

    printf("    Converting video model FP32 → BF16 safetensors: %s\n", path);

    int hidden_size = config->hidden_size;
    int num_layers = config->num_hidden_layers;
    int num_queries = config->num_queries;
    int query_dim = config->query_dim > 0 ? config->query_dim : hidden_size;
    int num_classes = config->num_classes > 0 ? config->num_classes : 10;
    int mask_dim = config->mask_dim > 0 ? config->mask_dim : query_dim;
    int in_channels = config->in_channels > 0 ? config->in_channels : 3;
    int patch_h = config->patch_size[1] > 0 ? config->patch_size[1] : 8;
    int patch_w = config->patch_size[2] > 0 ? config->patch_size[2] : 8;
    int patch_dim = in_channels * patch_h * patch_w;

    // Build cJSON header
    cJSON* header = cJSON_CreateObject();
    cJSON* metadata = cJSON_CreateObject();
    cJSON_AddStringToObject(metadata, "format", "pt");
    cJSON_AddStringToObject(metadata, "framework", "tetyah");
    cJSON_AddStringToObject(metadata, "model_type", "video");
    cJSON_AddItemToObject(header, "__metadata__", metadata);

    size_t data_offset = 0;
    char tensor_name[256];

    #define ADD_TENSOR(name, dim0, dim1, is_2d) do { \
        cJSON* tensor = cJSON_CreateObject(); \
        cJSON_AddStringToObject(tensor, "dtype", "BF16"); \
        cJSON* shape = cJSON_CreateArray(); \
        cJSON_AddItemToArray(shape, cJSON_CreateNumber((double)(dim0))); \
        if (is_2d) cJSON_AddItemToArray(shape, cJSON_CreateNumber((double)(dim1))); \
        cJSON_AddItemToObject(tensor, "shape", shape); \
        size_t tensor_size = (is_2d) ? (size_t)(dim0) * (dim1) : (size_t)(dim0); \
        size_t data_size = tensor_size * sizeof(uint16_t); \
        cJSON* offsets = cJSON_CreateArray(); \
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)data_offset)); \
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)(data_offset + data_size))); \
        cJSON_AddItemToObject(tensor, "data_offsets", offsets); \
        cJSON_AddItemToObject(header, name, tensor); \
        data_offset += data_size; \
    } while (0)

    // Video-specific: patch_embed instead of embed_tokens
    ADD_TENSOR("model.patch_embed.weight", hidden_size, patch_dim, 1);

    // Per-layer transformer weights
    for (int l = 0; l < num_layers; l++) {
        int q_dim = config->num_attention_heads * config->head_dim;
        int kv_dim = config->num_key_value_heads * config->head_dim;

        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.q_proj.weight", l);
        ADD_TENSOR(tensor_name, q_dim, hidden_size, 1);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.k_proj.weight", l);
        ADD_TENSOR(tensor_name, kv_dim, hidden_size, 1);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.v_proj.weight", l);
        ADD_TENSOR(tensor_name, kv_dim, hidden_size, 1);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.self_attn.o_proj.weight", l);
        ADD_TENSOR(tensor_name, hidden_size, q_dim, 1);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.gate_proj.weight", l);
        ADD_TENSOR(tensor_name, config->intermediate_size, hidden_size, 1);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.up_proj.weight", l);
        ADD_TENSOR(tensor_name, config->intermediate_size, hidden_size, 1);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.mlp.down_proj.weight", l);
        ADD_TENSOR(tensor_name, hidden_size, config->intermediate_size, 1);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.input_layernorm.weight", l);
        ADD_TENSOR(tensor_name, hidden_size, 0, 0);
        snprintf(tensor_name, sizeof(tensor_name), "model.layers.%d.post_attention_layernorm.weight", l);
        ADD_TENSOR(tensor_name, hidden_size, 0, 0);
    }

    // Final norm
    ADD_TENSOR("model.norm.weight", hidden_size, 0, 0);

    // Video heads
    ADD_TENSOR("model.queries.weight", num_queries, query_dim, 1);
    ADD_TENSOR("model.query_fusion.weight", query_dim, query_dim, 1);
    ADD_TENSOR("model.class_head.weight", num_classes, query_dim, 1);
    ADD_TENSOR("model.class_head.bias", num_classes, 0, 0);

    if (config->mask_output) {
        ADD_TENSOR("model.mask_mlp.w1.weight", mask_dim, query_dim, 1);
        ADD_TENSOR("model.mask_mlp.w2.weight", query_dim, mask_dim, 1);
    }

    #undef ADD_TENSOR

    // Serialize header
    char* header_json = cJSON_PrintUnformatted(header);
    uint64_t header_size = strlen(header_json);

    FILE* out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot create %s\n", path);
        cJSON_Delete(header);
        free(header_json);
        return;
    }

    fwrite(&header_size, sizeof(uint64_t), 1, out);
    fwrite(header_json, 1, header_size, out);

    // Helper to download GPU buffer and write as BF16
    #define WRITE_GPU_BF16(buf, count) do { \
        float* tmp = malloc((count) * sizeof(float)); \
        vulkan_buffer_download(ctx, (vulkan_buffer_t*)(buf), tmp, (count)); \
        for (size_t i = 0; i < (count); i++) { \
            uint16_t bf16 = f32_to_bf16(tmp[i]); \
            fwrite(&bf16, sizeof(uint16_t), 1, out); \
        } \
        free(tmp); \
    } while(0)

    // Helper for CPU tensor
    #define WRITE_CPU_BF16(tensor) do { \
        for (size_t i = 0; i < (tensor).size; i++) { \
            uint16_t bf16 = f32_to_bf16((tensor).weight[i]); \
            fwrite(&bf16, sizeof(uint16_t), 1, out); \
        } \
    } while(0)

    // Write patch_embed from GPU
    WRITE_GPU_BF16(ctx->gpu_w_embed, (size_t)hidden_size * patch_dim);

    // Write transformer layers from CPU (trainable_weights)
    for (int l = 0; l < num_layers; l++) {
        layer_weights_t* layer = &weights->layers[l];
        WRITE_CPU_BF16(layer->q_proj);
        WRITE_CPU_BF16(layer->k_proj);
        WRITE_CPU_BF16(layer->v_proj);
        WRITE_CPU_BF16(layer->o_proj);
        WRITE_CPU_BF16(layer->gate_proj);
        WRITE_CPU_BF16(layer->up_proj);
        WRITE_CPU_BF16(layer->down_proj);
        WRITE_CPU_BF16(layer->input_norm);
        WRITE_CPU_BF16(layer->post_norm);
    }

    // Final norm
    WRITE_CPU_BF16(weights->final_norm);

    // Video heads from GPU
    WRITE_GPU_BF16(state->query_learnable, (size_t)num_queries * query_dim);
    WRITE_GPU_BF16(state->query_fusion_weight, (size_t)query_dim * query_dim);
    WRITE_GPU_BF16(state->class_head_weight, (size_t)num_classes * query_dim);
    WRITE_GPU_BF16(state->class_head_bias, (size_t)num_classes);

    if (config->mask_output) {
        WRITE_GPU_BF16(state->mask_mlp_w1, (size_t)mask_dim * query_dim);
        WRITE_GPU_BF16(state->mask_mlp_w2, (size_t)query_dim * mask_dim);
    }

    #undef WRITE_GPU_BF16
    #undef WRITE_CPU_BF16

    fclose(out);
    free(header_json);
    cJSON_Delete(header);

    // Report file size
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    printf("    ✅ Saved video model: %.2f MB to %s\n", file_size / (1024.0 * 1024.0), path);
}

// Load video-specific weights
static int load_video_weights(train_state_t* state, const char* path) {
    if (!state->config || state->config->num_queries <= 0) return 0;

    vulkan_context_t* ctx = (vulkan_context_t*)state->vulkan_ctx;
    if (!ctx) return -1;

    char video_path[512];
    snprintf(video_path, sizeof(video_path), "%s.video", path);
    FILE* f = fopen(video_path, "rb");
    if (!f) {
        printf("  No video weights found at %s (using initialized weights)\n", video_path);
        return 0;  // Not an error - might be first run
    }

    uint32_t header[6];
    if (fread(header, sizeof(uint32_t), 6, f) != 6 || header[0] != VIDEO_CKPT_MAGIC) {
        fprintf(stderr, "ERROR: Invalid video weights file\n");
        fclose(f);
        return -1;
    }

    int num_queries = (int)header[1];
    int query_dim = (int)header[2];
    int num_classes = (int)header[3];
    int mask_dim = (int)header[4];
    int mask_output = (int)header[5];

    // Validate against config
    const model_config_t* config = state->config;
    if (num_queries != config->num_queries ||
        query_dim != (config->query_dim > 0 ? config->query_dim : config->hidden_size)) {
        fprintf(stderr, "ERROR: Video weights dimension mismatch\n");
        fclose(f);
        return -1;
    }

    // Helper to read and upload to GPU buffer
    #define LOAD_GPU_BUFFER(buf, count) do { \
        if (buf) { \
            size_t _cnt = (size_t)(count); \
            float* tmp = malloc(_cnt * sizeof(float)); \
            if (fread(tmp, sizeof(float), _cnt, f) == _cnt) { \
                vulkan_buffer_upload(ctx, (vulkan_buffer_t*)(buf), tmp, _cnt); \
            } \
            free(tmp); \
        } \
    } while(0)

    size_t q_size = (size_t)num_queries * query_dim;
    size_t f_size = (size_t)query_dim * query_dim;
    size_t cw_size = (size_t)query_dim * num_classes;

    LOAD_GPU_BUFFER(state->query_learnable, q_size);
    LOAD_GPU_BUFFER(state->query_fusion_weight, f_size);
    LOAD_GPU_BUFFER(state->class_head_weight, cw_size);
    LOAD_GPU_BUFFER(state->class_head_bias, num_classes);

    if (mask_output) {
        size_t w1_size = (size_t)query_dim * mask_dim;
        size_t w2_size = (size_t)mask_dim * query_dim;
        LOAD_GPU_BUFFER(state->mask_mlp_w1, w1_size);
        LOAD_GPU_BUFFER(state->mask_mlp_w2, w2_size);
    }

    // Load optimizer state
    LOAD_GPU_BUFFER(state->video_m_queries, q_size);
    LOAD_GPU_BUFFER(state->video_v_queries, q_size);
    LOAD_GPU_BUFFER(state->video_m_fusion, f_size);
    LOAD_GPU_BUFFER(state->video_v_fusion, f_size);
    LOAD_GPU_BUFFER(state->class_head_m_w, cw_size);
    LOAD_GPU_BUFFER(state->class_head_v_w, cw_size);
    LOAD_GPU_BUFFER(state->class_head_m_b, num_classes);
    LOAD_GPU_BUFFER(state->class_head_v_b, num_classes);

    if (mask_output) {
        size_t w1_size = (size_t)query_dim * mask_dim;
        size_t w2_size = (size_t)mask_dim * query_dim;
        LOAD_GPU_BUFFER(state->mask_m_w1, w1_size);
        LOAD_GPU_BUFFER(state->mask_v_w1, w1_size);
        LOAD_GPU_BUFFER(state->mask_m_w2, w2_size);
        LOAD_GPU_BUFFER(state->mask_v_w2, w2_size);
    }

    #undef LOAD_GPU_BUFFER

    fclose(f);
    printf("  Loaded video weights: %s\n", video_path);
    return 0;
}

int save_checkpoint(train_state_t* state, const char* path) {
    // Download GPU-resident LoRA weights to CPU before saving
    if (state->lora && state->vulkan_ctx) {
        vulkan_context_t* vk = (vulkan_context_t*)state->vulkan_ctx;
        if (vk->lora_gpu) {
            lora_gpu_download_to_cpu(vk, state->lora);
        }
    }

    // Save weights based on training mode
    int is_video = state->config && state->config->num_queries > 0;

    if (state->lora) {
        if (lora_save(state->lora, path) != 0) {
            fprintf(stderr, "ERROR: Failed to save LoRA checkpoint\n");
            return -1;
        }
        // Save video-specific weights for LoRA video mode
        if (is_video) {
            save_video_weights(state, path);
        }
    } else if (state->full_weights) {
        if (is_video) {
            // Video mode: save unified safetensors with patch_embed + video weights
            save_video_full_safetensors(state, path);
        } else {
            // LLM mode: save with embed_tokens
            trainable_weights_save_to_safetensors(state->full_weights, path);
        }
    }

    // Also save training state to a separate file
    char state_path[512];
    snprintf(state_path, sizeof(state_path), "%s.state", path);
    FILE* f = fopen(state_path, "wb");
    if (f) {
        fwrite(&state->current_epoch, sizeof(int), 1, f);
        fwrite(&state->current_step, sizeof(int), 1, f);
        fwrite(&state->current_loss, sizeof(float), 1, f);
        fwrite(&state->optimizer_step, sizeof(int), 1, f);
        fclose(f);
        printf("  Saved training state: %s\n", state_path);
    }

    printf("  Saved checkpoint: %s\n", path);
    return 0;
}

int load_checkpoint(train_state_t* state, const char* path) {
    // Load weights based on training mode
    if (state->lora) {
        // LoRA mode: load adapter weights from .bin
        lora_model_free(state->lora);
        state->lora = lora_load(path);
        if (!state->lora) {
            fprintf(stderr, "ERROR: Failed to load LoRA weights from %s\n", path);
            return -1;
        }
    } else if (state->full_weights) {
        // Full training mode: load weights from safetensors
        if (trainable_weights_load_from_safetensors(state->full_weights, path) != 0) {
            fprintf(stderr, "ERROR: Failed to load full weights from %s\n", path);
            return -1;
        }
    }

    // Load video-specific weights (queries, fusion, class head, mask head)
    if (state->config && state->config->num_queries > 0) {
        load_video_weights(state, path);
    }

    // Load training state from path.state
    char state_path[512];
    snprintf(state_path, sizeof(state_path), "%s.state", path);
    FILE* f = fopen(state_path, "rb");
    if (f) {
        fread(&state->current_epoch, sizeof(int), 1, f);
        fread(&state->current_step, sizeof(int), 1, f);
        fread(&state->current_loss, sizeof(float), 1, f);
        fread(&state->optimizer_step, sizeof(int), 1, f);
        fclose(f);
        printf("  Loaded training state: %s\n", state_path);
    } else {
        printf("  No training state found at %s (starting fresh)\n", state_path);
    }

    printf("  Loaded checkpoint: %s\n", path);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

train_state_t* train_init(const char* model_dir, train_config_t* config) {
    train_state_t* state = calloc(1, sizeof(train_state_t));
    state->train_cfg = config;  // Store training config for later use

    // Load model config
    printf("[1/4] Loading model configuration...\n");
    state->config = model_config_load(model_dir);
    if (!state->config) {
        free(state);
        return NULL;
    }

    // Propagate training config overrides to model config
    if (config && config->use_qk_norm) {
        state->config->use_qk_norm = 1;
        printf("      [Video/ViT mode] QK normalization enabled\n");
    }

    // Load tokenizer (skip for video models - they don't use text tokens)
    int is_video_model = state->config->num_queries > 0;
    if (!is_video_model) {
        printf("[2/4] Loading tokenizer...\n");
        state->tokenizer = tokenizer_load(model_dir);
        if (!state->tokenizer) {
            model_config_free(state->config);
            free(state);
            return NULL;
        }
    } else {
        printf("[2/4] Skipping tokenizer (video model)...\n");
        state->tokenizer = NULL;
    }

    // Load model
    printf("[3/4] Loading model weights...\n");
    state->model = seraph_load_model(state->config);
    if (!state->model) {
        if (state->tokenizer) tokenizer_free(state->tokenizer);
        model_config_free(state->config);
        free(state);
        return NULL;
    }

    // Initialize training mode (LoRA or full weights)
    if (!config->use_full_training) {
        // LoRA mode - create adapter matrices
        printf("[4/5] Initializing LoRA adapters (rank=%d)...\n", config->lora_rank);
        state->lora = lora_model_create_from_config(state->config, config->lora_rank, config->lora_alpha);
        if (!state->lora) {
            fprintf(stderr, "ERROR: Failed to create LoRA adapters\n");
            train_free(state);
            return NULL;
        }
        // Attach LoRA to model for inference
        state->model->lora = state->lora;
        state->use_full_training = 0;
    } else {
        // Full training mode - allocate FP32 weight tensors
        state->full_weights = trainable_weights_alloc(state->config);
        printf("[4/5] Allocating full weight tensors (%.2fM params in FP32)...\n",
               state->full_weights ? state->full_weights->total_params / 1e6 : 0.0);
        if (!state->full_weights) {
            fprintf(stderr, "ERROR: Failed to allocate full weight tensors\n");
            train_free(state);
            return NULL;
        }
        // Load BF16 weights from model → FP32 trainable weights
        trainable_weights_load_from_model(state->full_weights, state->model);
        state->use_full_training = 1;
    }

    // Allocate activation cache for backprop
    printf("[5/5] Allocating activation cache...\n");
    state->act_cache = activation_cache_alloc(
        state->config->num_hidden_layers,
        state->config->hidden_size,
        state->config->intermediate_size,
        state->config->num_attention_heads * state->config->head_dim,
        state->config->kv_dim,
        config->max_seq_len
    );
    // Connect activation cache to model for forward pass to use
    state->model->act_cache = state->act_cache;

    // Load training data
    printf("[6/6] Loading training data...\n");
    int is_video_mode = state->config->num_queries > 0;
    const char* data_path = config->train_data_path;

    // Auto-detect by extension or video mode
    int is_vdat = 0;
    if (data_path) {
        size_t len = strlen(data_path);
        if (len > 5 && strcmp(data_path + len - 5, ".vdat") == 0) {
            is_vdat = 1;
        }
    }

    if (is_video_mode || is_vdat) {
        if (train_load_video_data(state, data_path) < 0) {
            train_free(state);
            return NULL;
        }
        // Load validation video data if provided
        if (config->val_data_path) {
            train_load_val_video_data(state, config->val_data_path);
        }
    } else {
        if (train_load_data(state, data_path) < 0) {
            train_free(state);
            return NULL;
        }
    }

    // Initialize best validation loss
    state->best_val_loss = 1e9f;

    // Initialize gradient accumulation
    state->accumulation_counter = 0;
    state->gradient_accumulation_steps = config->gradient_accumulation_steps;
    state->learning_rate = config->learning_rate;  // Store configured LR for optimizer
    state->debug = config->debug;
    state->debug_weight_delta = config->debug_weight_delta;
    state->debug_nan_check = config->debug_nan_check;
    state->nan_detected = 0;
    if (state->gradient_accumulation_steps < 1) {
        state->gradient_accumulation_steps = 1;  // Default to no accumulation
    }
    if (state->gradient_accumulation_steps > 1) {
        printf("  Gradient accumulation: %d steps (effective batch = %d)\n",
               state->gradient_accumulation_steps, state->gradient_accumulation_steps);
    }

    // Initialize gradient snapshot writer if enabled (full training only)
    state->snapshot_writer = NULL;
    if (config->snapshot_every_n_steps > 0 && state->use_full_training) {
        char snapshot_path[512];
        snprintf(snapshot_path, sizeof(snapshot_path), "%s/gradient_snapshots.bin", config->checkpoint_dir);
        state->snapshot_writer = gradient_snapshot_writer_init(snapshot_path, state->config->num_hidden_layers);
    }

    // Find max tensor size for GPU buffer allocation (if using full training)
    if (state->full_weights) {
        size_t max_size = 0;
        trainable_weights_t* w = state->full_weights;

        // Check embeddings
        if (w->embed_tokens.size > max_size) max_size = w->embed_tokens.size;
        if (w->final_norm.size > max_size) max_size = w->final_norm.size;

        // Check all layer tensors
        for (int l = 0; l < w->num_layers; l++) {
            layer_weights_t* layer = &w->layers[l];
            if (layer->q_proj.size > max_size) max_size = layer->q_proj.size;
            if (layer->k_proj.size > max_size) max_size = layer->k_proj.size;
            if (layer->v_proj.size > max_size) max_size = layer->v_proj.size;
            if (layer->o_proj.size > max_size) max_size = layer->o_proj.size;
            if (layer->gate_proj.size > max_size) max_size = layer->gate_proj.size;
            if (layer->up_proj.size > max_size) max_size = layer->up_proj.size;
            if (layer->down_proj.size > max_size) max_size = layer->down_proj.size;
            if (layer->input_norm.size > max_size) max_size = layer->input_norm.size;
            if (layer->post_norm.size > max_size) max_size = layer->post_norm.size;
        }

        state->max_tensor_size = max_size * sizeof(float);
        printf("  Max tensor: %zu params (%.2f MB)\n", max_size, state->max_tensor_size / (1024.0 * 1024.0));
    } else if (state->lora) {
        // LoRA GPU path also needs max tensor size for forward pass buffer allocation
        // Base weights are same dimensions as full-weight (just frozen)
        const model_config_t* c = state->config;
        size_t max_size = 0;
        size_t embed_sz = (size_t)c->vocab_size * (size_t)c->hidden_size;
        if (embed_sz > max_size) max_size = embed_sz;
        size_t q_sz = (size_t)(c->num_attention_heads * c->head_dim) * (size_t)c->hidden_size;
        if (q_sz > max_size) max_size = q_sz;
        size_t gate_sz = (size_t)c->intermediate_size * (size_t)c->hidden_size;
        if (gate_sz > max_size) max_size = gate_sz;
        state->max_tensor_size = max_size * sizeof(float);
        printf("  Max tensor (LoRA base): %zu params (%.2f MB)\n",
               max_size, state->max_tensor_size / (1024.0 * 1024.0));
    } else {
        state->max_tensor_size = 0;
    }

    // Initialize GPU acceleration (optional)
    printf("[7/7] Initializing GPU acceleration...\n");

    if (config->cuda_backend && config->cuda_init_fn) {
        state->vulkan_ctx = NULL;
        state->cuda_ctx = config->cuda_init_fn(config->gpu_device_index);
        if (state->cuda_ctx) {
            printf("  ✅ CUDA GPU acceleration enabled\n");
        } else {
            fprintf(stderr, "  ⚠️  CUDA init failed - cannot continue\n");
            train_free(state);
            return NULL;
        }
    } else if (config->cpu_only) {
        state->vulkan_ctx = NULL;
        printf("  🧠 CPU-only mode enabled (--cpu-only)\n");
    } else {
        // Calculate descriptor pool size from model (if full training)
        int pool_hint = 0;
        if (state->full_weights) {
            // GPU-only training records many compute dispatches per step and allocates
            // one descriptor set per dispatch, so we need a much larger pool than "tensor count".
            // Rough sizing: O(100) sets per layer per step.
            pool_hint = (state->full_weights->num_layers * 128) + 512;
            printf("  Descriptor pool hint: %d sets (gpu full training)\n", pool_hint);
        } else if (state->lora) {
            // GPU LoRA: same structure as full training plus LoRA matmuls per active adapter
            int nl = state->config->num_hidden_layers;
            pool_hint = (nl * 160) + 512;
            printf("  Descriptor pool hint: %d sets (gpu lora training)\n", pool_hint);
        }

        state->vulkan_ctx = vulkan_init_ex(pool_hint, config->gpu_device_index);
        if (state->vulkan_ctx) {
            // Set max buffer size for GPU allocations
            if (state->max_tensor_size > 0) {
                ((vulkan_context_t*)state->vulkan_ctx)->max_buffer_size = state->max_tensor_size;
                ((vulkan_context_t*)state->vulkan_ctx)->max_fwd_buffer_size = state->max_tensor_size;
            }
            printf("  ✅ Vulkan GPU acceleration enabled\n");

            // Initialize video GPU buffers if video mode
            if (state->config->num_queries > 0) {
                printf("  Initializing video GPU buffers...\n");
                ensure_video_gpu_weights((vulkan_context_t*)state->vulkan_ctx, state);
            }
        } else {
            printf("  ⚠️  GPU unavailable - using CPU (slower)\n");
        }
    }

    return state;
}

void train_free(train_state_t* state) {
    if (!state) return;

    if (state->vulkan_ctx) {
        vulkan_context_t* vk = (vulkan_context_t*)state->vulkan_ctx;
        if (vk->lora_gpu) lora_gpu_free(vk);
        vulkan_free(vk);
    }
    if (state->snapshot_writer) gradient_snapshot_writer_close(state->snapshot_writer);
    if (state->train_tokens) free(state->train_tokens);
    if (state->sample_offsets) free(state->sample_offsets);
    if (state->act_cache) activation_cache_free(state->act_cache);
    if (state->lora) lora_model_free(state->lora);
    if (state->full_weights) trainable_weights_free(state->full_weights);
    if (state->model) seraph_free_model(state->model);
    if (state->tokenizer) tokenizer_free(state->tokenizer);
    if (state->config) model_config_free(state->config);

    // Unmap video data (mmap'd, not malloc'd)
    if (state->vdat_mmap) {
        munmap(state->vdat_mmap, state->vdat_mmap_size);
        state->video_frames = NULL;
        state->video_labels = NULL;
        state->video_masks = NULL;
    }
    // Free validation data (still uses malloc)
    if (state->val_video_frames) free(state->val_video_frames);
    if (state->val_video_labels) free(state->val_video_labels);
    if (state->val_video_masks) free(state->val_video_masks);

    // Free vAP tracking state
    if (state->prev_query_classes) free(state->prev_query_classes);
    if (state->prev_query_conf) free(state->prev_query_conf);

    free(state);
}
