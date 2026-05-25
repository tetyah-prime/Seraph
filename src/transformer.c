// transformer.c - SERAPH UNIVERSAL INFERENCE ENGINE
// ═══════════════════════════════════════════════════════════════════════════
// Pure BF16 Native Implementation with AVX-512 BF16 Acceleration
// Supports: Qwen, Mistral, Granite, Llama, and all Llama-derived architectures
// ═══════════════════════════════════════════════════════════════════════════
//
// Architecture: AMD Ryzen AI 9 365 (AVX-512 BF16, 70 AI TOPS)
// Strategy:
//   1. Model weights: BF16 mmap'd (zero-copy)
//   2. All computation: AVX-512 BF16 native (_mm512_dpbf16_ps)
//   3. Dynamic dimensions from config.json (no hardcoded sizes)
//   4. Single codebase for all Llama-style models
//
// ═══════════════════════════════════════════════════════════════════════════

#define _GNU_SOURCE
#include "../include/transformer.h"
#include "../include/model_config.h"
#include "../include/safetensors.h"
#include "../include/lora.h"
#include "../include/train.h"  // For activation_cache_t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// Architecture-specific SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

// OpenMP (optional - may not be available on all platforms)
#ifdef _OPENMP
    #include <omp.h>
#endif

// Auto-configure OpenMP threads (called once at startup)
#ifdef _OPENMP
__attribute__((constructor))
static void configure_threads(void) {
    // Use half of available processors (physical cores, not hyperthreads)
    int num_procs = sysconf(_SC_NPROCESSORS_ONLN);
    int num_threads = num_procs / 2;
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 16) num_threads = 16;  // Cap at 16 for sanity
    omp_set_num_threads(num_threads);
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// BF16 CORE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

// BF16 to F32 (for final output only)
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float f32;
    memcpy(&f32, &f32_bits, sizeof(float));
    return f32;
}

// F32 to BF16
static inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    memcpy(&bits, &f32, sizeof(float));
    return (uint16_t)(bits >> 16);
}

// BF16 add
static inline uint16_t bf16_add(uint16_t a, uint16_t b) {
    return f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
}

// BF16 multiply
static inline uint16_t bf16_mul(uint16_t a, uint16_t b) {
    return f32_to_bf16(bf16_to_f32(a) * bf16_to_f32(b));
}

// BF16 SiLU activation: x * sigmoid(x)
static inline uint16_t bf16_silu(uint16_t x) {
    float fx = bf16_to_f32(x);
    return f32_to_bf16(fx / (1.0f + expf(-fx)));
}

// ═══════════════════════════════════════════════════════════════════════════
// ARCHITECTURE-SPECIFIC BF16 KERNELS
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512BF16__) && defined(__AVX512F__)
// AVX-512 BF16 dot product - 32 elements per iteration!
static inline float bf16_dot(const uint16_t* a, const uint16_t* b, int n) {
    __m512 acc = _mm512_setzero_ps();

    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m512i va = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i vb = _mm512_loadu_si512((__m512i*)(b + i));
        acc = _mm512_dpbf16_ps(acc, (__m512bh)va, (__m512bh)vb);
    }

    // Horizontal sum
    __m256 acc_low = _mm512_castps512_ps256(acc);
    __m256 acc_high = _mm512_extractf32x8_ps(acc, 1);
    __m256 acc256 = _mm256_add_ps(acc_low, acc_high);
    __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(acc256), _mm256_extractf128_ps(acc256, 1));
    acc128 = _mm_hadd_ps(acc128, acc128);
    acc128 = _mm_hadd_ps(acc128, acc128);
    float result = _mm_cvtss_f32(acc128);

    // Remainder
    for (; i < n; i++) {
        result += bf16_to_f32(a[i]) * bf16_to_f32(b[i]);
    }

    return result;
}
#else
// x86/x64 fallback when AVX-512 BF16 compile flags are not enabled
static inline float bf16_dot(const uint16_t* a, const uint16_t* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += bf16_to_f32(a[i]) * bf16_to_f32(b[i]);
    }
    return result;
}
#endif

#elif defined(__aarch64__)
// ARM64 BF16 dot product - uses NEON with BF16 conversion
static inline float bf16_dot(const uint16_t* a, const uint16_t* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 4 elements at a time with NEON
    for (; i + 3 < n; i += 4) {
        // Load BF16 values and convert to FP32
        float32x4_t va = {bf16_to_f32(a[i]), bf16_to_f32(a[i+1]), bf16_to_f32(a[i+2]), bf16_to_f32(a[i+3])};
        float32x4_t vb = {bf16_to_f32(b[i]), bf16_to_f32(b[i+1]), bf16_to_f32(b[i+2]), bf16_to_f32(b[i+3])};
        acc = vmlaq_f32(acc, va, vb);
    }

    // Horizontal sum
    float result = vaddvq_f32(acc);

    // Remainder
    for (; i < n; i++) {
        result += bf16_to_f32(a[i]) * bf16_to_f32(b[i]);
    }

    return result;
}

#else
// Scalar fallback for other architectures
static inline float bf16_dot(const uint16_t* a, const uint16_t* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += bf16_to_f32(a[i]) * bf16_to_f32(b[i]);
    }
    return result;
}
#endif

// Matrix-vector multiply: out = matrix @ vec + bias (PARALLELIZED for large dims)
static void matvec(uint16_t* out, const uint16_t* matrix, const uint16_t* vec,
                   const uint16_t* bias, int out_dim, int in_dim) {
    #pragma omp parallel for schedule(static) if(out_dim >= 1024)
    for (int i = 0; i < out_dim; i++) {
        float result = bf16_dot(&matrix[i * in_dim], vec, in_dim);
        if (bias) result += bf16_to_f32(bias[i]);
        out[i] = f32_to_bf16(result);
    }
}

// RMS LayerNorm (sequential - too small to parallelize)
static void rms_norm(uint16_t* out, const uint16_t* x, const uint16_t* weight, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        float v = bf16_to_f32(x[i]);
        sum_sq += v * v;
    }
    float rms = sqrtf(sum_sq / size + eps);
    for (int i = 0; i < size; i++) {
        float v = bf16_to_f32(x[i]) / rms * bf16_to_f32(weight[i]);
        out[i] = f32_to_bf16(v);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ROPE (ROTARY POSITION EMBEDDINGS)
// ═══════════════════════════════════════════════════════════════════════════

// Apply RoPE to a single head's Q or K vector
// pos = position in sequence, head_dim = dimension of each head
static void apply_rope(float* vec, int pos, int head_dim, float rope_theta) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(rope_theta, (float)i / (float)head_dim);
        float theta = pos * freq;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float x0 = vec[i];
        float x1 = vec[i + 1];
        vec[i]     = x0 * cos_theta - x1 * sin_theta;
        vec[i + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION MECHANISM
// ═══════════════════════════════════════════════════════════════════════════

// Softmax over attention scores (in-place)
static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FORWARD DECLARATIONS
// ═══════════════════════════════════════════════════════════════════════════

static void kv_cache_free(kv_cache_t* cache);

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR ACCESS
// ═══════════════════════════════════════════════════════════════════════════

static int find_tensor_file(seraph_model_t* model, const char* name) {
    for (int i = 0; i < model->num_tensors; i++) {
        if (strcmp(model->tensor_map[i].name, name) == 0) {
            return model->tensor_map[i].file_idx;
        }
    }
    return -1;
}

// Debug flag - set to 1 to see all tensor loading warnings
#define TETYAH_DEBUG 0

// Find tensor by normalized name, return original name for safetensors lookup
static const char* find_tensor_orig_name(seraph_model_t* model, const char* name) {
    for (int i = 0; i < model->num_tensors; i++) {
        if (strcmp(model->tensor_map[i].name, name) == 0) {
            return model->tensor_map[i].orig_name;
        }
    }
    return NULL;
}

static uint16_t* get_tensor(seraph_model_t* model, const char* name) {
    int file_idx = find_tensor_file(model, name);
    if (file_idx < 0) {
        #if TETYAH_DEBUG
        fprintf(stderr, "DEBUG: Tensor '%s' not found\n", name);
        #endif
        return NULL;
    }
    const char* orig = find_tensor_orig_name(model, name);
    return (uint16_t*)safetensors_get_tensor_raw(model->st[file_idx], orig ? orig : name);
}

// Get required tensor - warns if missing (for critical weights)
static uint16_t* get_tensor_required(seraph_model_t* model, const char* name) {
    int file_idx = find_tensor_file(model, name);
    if (file_idx < 0) {
        fprintf(stderr, "ERROR: Required tensor '%s' not found!\n", name);
        return NULL;
    }
    const char* orig = find_tensor_orig_name(model, name);
    return (uint16_t*)safetensors_get_tensor_raw(model->st[file_idx], orig ? orig : name);
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════════════════════════════════════

seraph_model_t* seraph_load_model(const model_config_t* config) {
    if (!config) {
        fprintf(stderr, "ERROR: NULL config\n");
        return NULL;
    }

    printf("Loading %s model (PURE BF16 + AVX-512)...\n", config->model_type);

    seraph_model_t* model = calloc(1, sizeof(seraph_model_t));
    if (!model) return NULL;

    // Store config reference
    model->config = (model_config_t*)config;

    char path[1024];
    int total_tensors = 0;

    // Try single file first
    snprintf(path, sizeof(path), "%s/model.safetensors", config->model_dir);
    if (access(path, F_OK) == 0) {
        printf("  Loading: %s\n", path);
        model->st[0] = safetensors_load(path);
        if (!model->st[0]) {
            fprintf(stderr, "ERROR: Failed to load %s\n", path);
            free(model);
            return NULL;
        }
        model->num_files = 1;
        total_tensors = model->st[0]->num_tensors;
    } else {
        // Try sharded files
        for (int i = 0; i < 8; i++) {
            snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.safetensors",
                     config->model_dir, i + 1, config->num_model_files);
            if (access(path, F_OK) != 0) {
                // Try 4-file format
                snprintf(path, sizeof(path), "%s/model-%05d-of-00004.safetensors",
                         config->model_dir, i + 1);
                if (access(path, F_OK) != 0) break;
            }

            printf("  Loading: %s\n", path);
            model->st[i] = safetensors_load(path);
            if (!model->st[i]) {
                fprintf(stderr, "ERROR: Failed to load %s\n", path);
                seraph_free_model(model);
                return NULL;
            }
            model->num_files++;
            total_tensors += model->st[i]->num_tensors;
        }
    }

    if (model->num_files == 0) {
        fprintf(stderr, "ERROR: No model files found in %s\n", config->model_dir);
        free(model);
        return NULL;
    }

    // Build tensor map — normalize names by stripping any prefix before "model."
    // This handles multimodal models (language_model.model.X → model.X),
    // encoder-decoder models, or any future wrapping convention
    printf("  Building tensor map (%d tensors)...\n", total_tensors);
    model->tensor_map = malloc(total_tensors * sizeof(tensor_location_t));
    model->num_tensors = total_tensors;

    int stripped = 0;
    int idx = 0;
    for (int f = 0; f < model->num_files; f++) {
        for (int t = 0; t < model->st[f]->num_tensors; t++) {
            const char* orig_name = model->st[f]->tensors[t].name;

            // Store original name (needed for safetensors raw lookup)
            strncpy(model->tensor_map[idx].orig_name, orig_name, 255);

            // Normalize: strip everything before the last "model." prefix
            // e.g. "language_model.model.layers.0..." → "model.layers.0..."
            // Must match "model." preceded by '.' or at start, not inside a word
            const char* normalized = NULL;
            const char* search = orig_name;
            while ((search = strstr(search, "model.")) != NULL) {
                // Accept if at start of string or preceded by '.'
                if (search == orig_name || *(search - 1) == '.') {
                    normalized = search;
                }
                search += 6;  // Skip past "model."
            }
            if (normalized && normalized != orig_name) {
                strncpy(model->tensor_map[idx].name, normalized, 255);
                stripped++;
            } else {
                strncpy(model->tensor_map[idx].name, orig_name, 255);
            }
            model->tensor_map[idx].file_idx = f;
            idx++;
        }
    }

    if (stripped > 0) {
        printf("  📦 Normalized %d tensor names (stripped wrapper prefix)\n", stripped);
    }
    printf("✅ Model loaded (%d files, %d tensors)\n", model->num_files, model->num_tensors);
    return model;
}

void seraph_free_model(seraph_model_t* model) {
    if (!model) return;
    for (int i = 0; i < model->num_files; i++) {
        if (model->st[i]) safetensors_free(model->st[i]);
    }
    if (model->tensor_map) free(model->tensor_map);
    if (model->kv_cache) kv_cache_free(model->kv_cache);
    free(model);
}

// ═══════════════════════════════════════════════════════════════════════════
// KV CACHE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

static kv_cache_t* kv_cache_alloc(int num_layers, int kv_dim, int max_seq_len) {
    kv_cache_t* cache = calloc(1, sizeof(kv_cache_t));
    cache->num_layers = num_layers;
    cache->kv_dim = kv_dim;
    cache->max_seq_len = max_seq_len;
    cache->cur_seq_len = 0;

    size_t size = (size_t)num_layers * max_seq_len * kv_dim * sizeof(float);
    cache->k = calloc(1, size);
    cache->v = calloc(1, size);

    #if TETYAH_DEBUG
    printf("  Allocated KV cache: %zu MB\n", (size * 2) / (1024 * 1024));
    #endif

    return cache;
}

static void kv_cache_free(kv_cache_t* cache) {
    if (!cache) return;
    free(cache->k);
    free(cache->v);
    free(cache);
}

void seraph_reset_cache(seraph_model_t* model) {
    if (model && model->kv_cache) {
        model->kv_cache->cur_seq_len = 0;
    }
}

// Get pointer to K cache for a specific layer and position
static float* kv_cache_k(kv_cache_t* cache, int layer, int pos) {
    return &cache->k[(size_t)layer * cache->max_seq_len * cache->kv_dim + pos * cache->kv_dim];
}

// Get pointer to V cache for a specific layer and position
static float* kv_cache_v(kv_cache_t* cache, int layer, int pos) {
    return &cache->v[(size_t)layer * cache->max_seq_len * cache->kv_dim + pos * cache->kv_dim];
}

// ═══════════════════════════════════════════════════════════════════════════
// FORWARD PASS - WITH PROPER ATTENTION AND KV CACHE
// ═══════════════════════════════════════════════════════════════════════════

float* seraph_forward(seraph_model_t* model, const model_config_t* config, int* token_ids, int seq_len) {
    // Use model's config if not provided
    if (!config) config = model->config;
    if (!config) {
        fprintf(stderr, "ERROR: No config available\n");
        return NULL;
    }

    // Extract dimensions from config
    const int hidden_size = config->hidden_size;
    const int intermediate_size = config->intermediate_size;
    const int num_layers = config->num_hidden_layers;
    const int vocab_size = config->vocab_size;
    const int num_heads = config->num_attention_heads;
    const int num_kv_heads = config->num_key_value_heads;
    const int head_dim = config->head_dim;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = config->kv_dim;
    const float rms_eps = config->rms_norm_eps;
    const float rope_theta = config->rope_theta;
    const int num_groups = num_heads / num_kv_heads;  // For GQA

    // Lazy allocate KV cache
    // Cap at reasonable inference size — max_position_embeddings can be 262K+
    // which would require 50-100GB+ of RAM for the cache alone
    if (!model->kv_cache) {
        int max_seq = 4096;  // Default for inference/chat
        if (config && config->max_position_embeddings > 0 && config->max_position_embeddings <= 8192) {
            max_seq = config->max_position_embeddings;
        }
        model->kv_cache = kv_cache_alloc(num_layers, kv_dim, max_seq);
        if (!model->kv_cache || !model->kv_cache->k || !model->kv_cache->v) {
            fprintf(stderr, "ERROR: KV cache allocation failed (%d layers × %d seq × %d kv_dim)\n",
                    num_layers, max_seq, kv_dim);
            return NULL;
        }
    }
    kv_cache_t* cache = model->kv_cache;

    // Determine which positions to process
    // If cache has content and seq_len > cache length, only process new tokens
    int start_pos = cache->cur_seq_len;
    if (start_pos > 0 && seq_len <= start_pos) {
        // Generating: only process the last (new) token
        start_pos = seq_len - 1;
    } else if (start_pos == 0) {
        // Fresh start: process all tokens
        start_pos = 0;
    }

    // Get embedding table
    uint16_t* embed_table = get_tensor(model, "model.embed_tokens.weight");
    if (!embed_table) return NULL;

    // Allocate working buffers
    uint16_t* x_bf16 = malloc(hidden_size * sizeof(uint16_t));
    uint16_t* x_norm_bf16 = malloc(hidden_size * sizeof(uint16_t));
    uint16_t* q_proj_bf16 = malloc(q_dim * sizeof(uint16_t));
    uint16_t* k_proj_bf16 = malloc(kv_dim * sizeof(uint16_t));
    uint16_t* v_proj_bf16 = malloc(kv_dim * sizeof(uint16_t));
    uint16_t* attn_out_bf16 = malloc(hidden_size * sizeof(uint16_t));

    // Storage for all positions' hidden states (for full sequence training)
    uint16_t* all_hidden_bf16 = malloc(seq_len * hidden_size * sizeof(uint16_t));
    uint16_t* mlp_gate = malloc(intermediate_size * sizeof(uint16_t));
    uint16_t* mlp_up = malloc(intermediate_size * sizeof(uint16_t));

    // Float buffers for attention computation (per-head for parallelization)
    float* q_float = malloc(q_dim * sizeof(float));
    float* attn_head_out = malloc(num_heads * head_dim * sizeof(float));  // Per-head output buffers
    float* attn_combined = malloc(q_dim * sizeof(float));
    // Score buffer sized to current sequence length (config-driven at call site)
    float* attn_scores = malloc((size_t)num_heads * (size_t)seq_len * sizeof(float));  // Per-head score buffers

    // LoRA working buffers (for training, avoids alloca() stack overflow)
    float* lora_buf1 = malloc(intermediate_size * sizeof(float));  // Largest needed size
    float* lora_buf2 = malloc(intermediate_size * sizeof(float));

    if (!x_bf16 || !x_norm_bf16 || !q_proj_bf16 || !k_proj_bf16 || !v_proj_bf16 ||
        !attn_out_bf16 || !mlp_gate || !mlp_up || !q_float || !attn_head_out ||
        !attn_combined || !attn_scores || !lora_buf1 || !lora_buf2 || !all_hidden_bf16) {
        fprintf(stderr, "ERROR: Allocation failed\n");
        return NULL;
    }

    // Process each position from start_pos to seq_len-1
    for (int pos = start_pos; pos < seq_len; pos++) {
        int token_id = token_ids[pos];

        // Get embedding for this token
        memcpy(x_bf16, &embed_table[token_id * hidden_size], hidden_size * sizeof(uint16_t));

        // Process through all layers
        for (int layer = 0; layer < num_layers; layer++) {
            char name[256];

            // === Load weights for this layer ===
            snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer);
            uint16_t* ln1_w = get_tensor_required(model, name);

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer);
            uint16_t* q_w = get_tensor_required(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer);
            uint16_t* k_w = get_tensor_required(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer);
            uint16_t* v_w = get_tensor_required(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer);
            uint16_t* o_w = get_tensor_required(model, name);

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", layer);
            uint16_t* q_b = get_tensor(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", layer);
            uint16_t* k_b = get_tensor(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", layer);
            uint16_t* v_b = get_tensor(model, name);

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer);
            uint16_t* q_norm_w = get_tensor(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer);
            uint16_t* k_norm_w = get_tensor(model, name);

            snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer);
            uint16_t* ln2_w = get_tensor_required(model, name);

            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer);
            uint16_t* gate_w = get_tensor_required(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer);
            uint16_t* up_w = get_tensor_required(model, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer);
            uint16_t* down_w = get_tensor_required(model, name);

            // === Pre-attention LayerNorm ===
            rms_norm(x_norm_bf16, x_bf16, ln1_w, hidden_size, rms_eps);

            // Cache x_norm for LoRA backward (input to Q/K/V)
            if (model->act_cache) {
                activation_cache_t* ac = (activation_cache_t*)model->act_cache;
                float* dst = &ac->x_norm_attn[layer][pos * hidden_size];
                for (int i = 0; i < hidden_size; i++) {
                    dst[i] = bf16_to_f32(x_norm_bf16[i]);
                }
            }

            // === Q, K, V Projections ===
            matvec(q_proj_bf16, q_w, x_norm_bf16, q_b, q_dim, hidden_size);
            matvec(k_proj_bf16, k_w, x_norm_bf16, k_b, kv_dim, hidden_size);
            matvec(v_proj_bf16, v_w, x_norm_bf16, v_b, kv_dim, hidden_size);

            // === Apply LoRA to Q and V if present ===
            if (model->lora) {
                // Convert normalized input to float for LoRA (reuse q_float buffer)
                float* x_norm_float = q_float;  // Reuse existing buffer
                for (int i = 0; i < hidden_size; i++) {
                    x_norm_float[i] = bf16_to_f32(x_norm_bf16[i]);
                }

                // Apply LoRA to Q projection
                if (model->lora->q_adapters && model->lora->q_adapters[layer]) {
                    // Use attn_combined as temp buffer (same size as q_dim)
                    float* q_float_tmp = attn_combined;
                    for (int i = 0; i < q_dim; i++) {
                        q_float_tmp[i] = bf16_to_f32(q_proj_bf16[i]);
                    }
                    lora_forward(model->lora->q_adapters[layer], x_norm_float, q_float_tmp, 1);
                    for (int i = 0; i < q_dim; i++) {
                        q_proj_bf16[i] = f32_to_bf16(q_float_tmp[i]);
                    }
                }

                // Apply LoRA to K projection (if enabled)
                if (model->lora->k_adapters && model->lora->k_adapters[layer]) {
                    // Reuse part of attn_head_out as temp
                    float* k_float_tmp = attn_head_out;
                    for (int i = 0; i < kv_dim; i++) {
                        k_float_tmp[i] = bf16_to_f32(k_proj_bf16[i]);
                    }
                    lora_forward(model->lora->k_adapters[layer], x_norm_float, k_float_tmp, 1);
                    for (int i = 0; i < kv_dim; i++) {
                        k_proj_bf16[i] = f32_to_bf16(k_float_tmp[i]);
                    }
                }

                // Apply LoRA to V projection
                if (model->lora->v_adapters && model->lora->v_adapters[layer]) {
                    float* v_float_tmp = attn_head_out;
                    for (int i = 0; i < kv_dim; i++) {
                        v_float_tmp[i] = bf16_to_f32(v_proj_bf16[i]);
                    }
                    lora_forward(model->lora->v_adapters[layer], x_norm_float, v_float_tmp, 1);
                    for (int i = 0; i < kv_dim; i++) {
                        v_proj_bf16[i] = f32_to_bf16(v_float_tmp[i]);
                    }
                }
            }

            // === Apply QK norm if present (Qwen3) ===
            if (q_norm_w) {
                for (int h = 0; h < num_heads; h++) {
                    rms_norm(&q_proj_bf16[h * head_dim], &q_proj_bf16[h * head_dim], q_norm_w, head_dim, rms_eps);
                }
            }
            if (k_norm_w) {
                for (int h = 0; h < num_kv_heads; h++) {
                    rms_norm(&k_proj_bf16[h * head_dim], &k_proj_bf16[h * head_dim], k_norm_w, head_dim, rms_eps);
                }
            }

            // === Convert to float and apply RoPE ===
            // Q: convert and apply RoPE per head
            for (int i = 0; i < q_dim; i++) {
                q_float[i] = bf16_to_f32(q_proj_bf16[i]);
            }
            for (int h = 0; h < num_heads; h++) {
                apply_rope(&q_float[h * head_dim], pos, head_dim, rope_theta);
            }

            // K: convert, apply RoPE, store in cache
            float* k_cache_pos = kv_cache_k(cache, layer, pos);
            for (int i = 0; i < kv_dim; i++) {
                k_cache_pos[i] = bf16_to_f32(k_proj_bf16[i]);
            }
            for (int h = 0; h < num_kv_heads; h++) {
                apply_rope(&k_cache_pos[h * head_dim], pos, head_dim, rope_theta);
            }

            // V: convert and store in cache (no RoPE for V)
            float* v_cache_pos = kv_cache_v(cache, layer, pos);
            for (int i = 0; i < kv_dim; i++) {
                v_cache_pos[i] = bf16_to_f32(v_proj_bf16[i]);
            }

            // === Multi-head Attention with GQA (PARALLELIZED for 8+ heads) ===
            float* k_cache_layer = kv_cache_k(cache, layer, 0);
            float* v_cache_layer = kv_cache_v(cache, layer, 0);
            float scale = 1.0f / sqrtf((float)head_dim);

            #pragma omp parallel for schedule(static) if(num_heads >= 8)
            for (int h = 0; h < num_heads; h++) {
                int kv_head = h / num_groups;  // Which KV head this Q head uses

                // Get per-head buffers (each thread uses its own slice)
                float* my_scores = &attn_scores[(size_t)h * (size_t)seq_len];
                float* my_out = &attn_head_out[h * head_dim];

                // Get Q for this head
                float* q_head = &q_float[h * head_dim];

                // Compute Q @ K^T for all positions
                for (int p = 0; p <= pos; p++) {
                    float* k_p = &k_cache_layer[p * kv_dim + kv_head * head_dim];
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += q_head[d] * k_p[d];
                    }
                    my_scores[p] = score * scale;
                }

                // Softmax
                softmax(my_scores, pos + 1);

                // Weighted sum of V
                for (int d = 0; d < head_dim; d++) {
                    my_out[d] = 0.0f;
                }
                for (int p = 0; p <= pos; p++) {
                    float* v_p = &v_cache_layer[p * kv_dim + kv_head * head_dim];
                    for (int d = 0; d < head_dim; d++) {
                        my_out[d] += my_scores[p] * v_p[d];
                    }
                }

                // Store in combined output
                for (int d = 0; d < head_dim; d++) {
                    attn_combined[h * head_dim + d] = my_out[d];
                }
            }

            // Cache attention output for LoRA backward (input to O projection)
            if (model->act_cache) {
                activation_cache_t* ac = (activation_cache_t*)model->act_cache;
                float* dst = &ac->attn_output[layer][pos * q_dim];
                for (int i = 0; i < q_dim; i++) {
                    dst[i] = attn_combined[i];
                }
            }

            // === Output projection (convert back to BF16) ===
            for (int i = 0; i < q_dim; i++) {
                q_proj_bf16[i] = f32_to_bf16(attn_combined[i]);
            }
            matvec(attn_out_bf16, o_w, q_proj_bf16, NULL, hidden_size, q_dim);

            // Apply LoRA to O projection if present
            if (model->lora && model->lora->o_adapters && model->lora->o_adapters[layer]) {
                for (int i = 0; i < hidden_size; i++) {
                    lora_buf1[i] = bf16_to_f32(attn_out_bf16[i]);
                }
                lora_forward(model->lora->o_adapters[layer], attn_combined, lora_buf1, 1);
                for (int i = 0; i < hidden_size; i++) {
                    attn_out_bf16[i] = f32_to_bf16(lora_buf1[i]);
                }
            }

            // === Residual connection 1 ===
            for (int i = 0; i < hidden_size; i++) {
                x_bf16[i] = bf16_add(x_bf16[i], attn_out_bf16[i]);
            }

            // === Post-attention LayerNorm ===
            rms_norm(x_norm_bf16, x_bf16, ln2_w, hidden_size, rms_eps);

            // Cache x_norm_ffn for LoRA backward (input to FFN)
            if (model->act_cache) {
                activation_cache_t* ac = (activation_cache_t*)model->act_cache;
                float* dst = &ac->x_norm_ffn[layer][pos * hidden_size];
                for (int i = 0; i < hidden_size; i++) {
                    dst[i] = bf16_to_f32(x_norm_bf16[i]);
                }
            }

            // === MLP: SwiGLU ===
            matvec(mlp_gate, gate_w, x_norm_bf16, NULL, intermediate_size, hidden_size);
            matvec(mlp_up, up_w, x_norm_bf16, NULL, intermediate_size, hidden_size);

            // Apply LoRA to FFN if present
            if (model->lora && model->lora->apply_to_ffn) {
                // Convert input to float once for FFN LoRA (reuse q_float)
                for (int i = 0; i < hidden_size; i++) {
                    q_float[i] = bf16_to_f32(x_norm_bf16[i]);
                }

                // Gate LoRA
                if (model->lora->gate_adapters && model->lora->gate_adapters[layer]) {
                    for (int i = 0; i < intermediate_size; i++) {
                        lora_buf1[i] = bf16_to_f32(mlp_gate[i]);
                    }
                    lora_forward(model->lora->gate_adapters[layer], q_float, lora_buf1, 1);
                    for (int i = 0; i < intermediate_size; i++) {
                        mlp_gate[i] = f32_to_bf16(lora_buf1[i]);
                    }
                }

                // Up LoRA
                if (model->lora->up_adapters && model->lora->up_adapters[layer]) {
                    for (int i = 0; i < intermediate_size; i++) {
                        lora_buf1[i] = bf16_to_f32(mlp_up[i]);
                    }
                    lora_forward(model->lora->up_adapters[layer], q_float, lora_buf1, 1);
                    for (int i = 0; i < intermediate_size; i++) {
                        mlp_up[i] = f32_to_bf16(lora_buf1[i]);
                    }
                }
            }

            // Cache gate and up BEFORE silu (needed for proper backprop)
            if (model->act_cache) {
                activation_cache_t* ac = (activation_cache_t*)model->act_cache;
                float* gate_dst = &ac->ffn_gate_out[layer][pos * intermediate_size];
                float* up_dst = &ac->ffn_up_out[layer][pos * intermediate_size];
                for (int i = 0; i < intermediate_size; i++) {
                    gate_dst[i] = bf16_to_f32(mlp_gate[i]);
                    up_dst[i] = bf16_to_f32(mlp_up[i]);
                }
            }

            // SiLU elementwise (sequential - memory bound, not compute bound)
            for (int i = 0; i < intermediate_size; i++) {
                mlp_gate[i] = bf16_mul(bf16_silu(mlp_gate[i]), mlp_up[i]);
            }

            // Cache ffn_hidden AFTER silu (input to down projection)
            if (model->act_cache) {
                activation_cache_t* ac = (activation_cache_t*)model->act_cache;
                float* dst = &ac->ffn_hidden[layer][pos * intermediate_size];
                for (int i = 0; i < intermediate_size; i++) {
                    dst[i] = bf16_to_f32(mlp_gate[i]);
                }
            }

            matvec(attn_out_bf16, down_w, mlp_gate, NULL, hidden_size, intermediate_size);

            // Apply LoRA to down projection if present
            if (model->lora && model->lora->apply_to_ffn &&
                model->lora->down_adapters && model->lora->down_adapters[layer]) {
                for (int i = 0; i < intermediate_size; i++) {
                    lora_buf1[i] = bf16_to_f32(mlp_gate[i]);
                }
                for (int i = 0; i < hidden_size; i++) {
                    lora_buf2[i] = bf16_to_f32(attn_out_bf16[i]);
                }
                lora_forward(model->lora->down_adapters[layer], lora_buf1, lora_buf2, 1);
                for (int i = 0; i < hidden_size; i++) {
                    attn_out_bf16[i] = f32_to_bf16(lora_buf2[i]);
                }
            }

            // === Residual connection 2 ===
            for (int i = 0; i < hidden_size; i++) {
                x_bf16[i] = bf16_add(x_bf16[i], attn_out_bf16[i]);
            }
        }  // end layer loop

        // Save this position's hidden state for full sequence training
        memcpy(&all_hidden_bf16[pos * hidden_size], x_bf16, hidden_size * sizeof(uint16_t));
    }  // end position loop

    // Update cache position
    cache->cur_seq_len = seq_len;

    // === Final LayerNorm for ALL positions ===
    uint16_t* final_norm_w = get_tensor(model, "model.norm.weight");
    uint16_t* all_normed_bf16 = malloc(seq_len * hidden_size * sizeof(uint16_t));

    for (int pos = 0; pos < seq_len; pos++) {
        rms_norm(&all_normed_bf16[pos * hidden_size],
                 &all_hidden_bf16[pos * hidden_size],
                 final_norm_w, hidden_size, rms_eps);
    }

    // Cache ALL final hidden states for LoRA backward
    if (model->act_cache) {
        activation_cache_t* ac = (activation_cache_t*)model->act_cache;
        for (int pos = 0; pos < seq_len; pos++) {
            float* dst = &ac->final_hidden[pos * hidden_size];
            for (int i = 0; i < hidden_size; i++) {
                dst[i] = bf16_to_f32(all_normed_bf16[pos * hidden_size + i]);
            }
        }
        ac->cur_seq_len = seq_len;
    }

    // === LM Head (compute logits for ALL positions) ===
    uint16_t* lm_head = get_tensor(model, "model.embed_tokens.weight");
    if (config->tie_word_embeddings == false) {
        uint16_t* lm_head_weight = get_tensor(model, "lm_head.weight");
        if (lm_head_weight) lm_head = lm_head_weight;
    }

    // Allocate for all positions [seq_len * vocab_size]
    float* logits = malloc(seq_len * vocab_size * sizeof(float));
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < vocab_size; i++) {
            logits[pos * vocab_size + i] = bf16_dot(&lm_head[i * hidden_size],
                                                            &all_normed_bf16[pos * hidden_size],
                                                            hidden_size);
        }
    }

    // Cleanup working buffers
    free(x_bf16);
    free(x_norm_bf16);
    free(q_proj_bf16);
    free(k_proj_bf16);
    free(v_proj_bf16);
    free(attn_out_bf16);
    free(mlp_gate);
    free(mlp_up);
    free(q_float);
    free(attn_head_out);
    free(attn_combined);
    free(attn_scores);
    free(lora_buf1);
    free(lora_buf2);
    free(all_hidden_bf16);
    free(all_normed_bf16);

    return logits;  // Now returns [seq_len * vocab_size]
}

// ═══════════════════════════════════════════════════════════════════════════
