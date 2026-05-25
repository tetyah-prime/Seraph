// init_random_weights.c - Seraph Random Weight Initializer
// Creates safetensor file with random weights for any transformer config
//
// Usage: seraph-init <config_dir> <output.safetensors>
//
// Native weight initialization - no pretrained weights needed!

#include "../include/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

// Xavier/Glorot initialization: stddev = sqrt(2 / (fan_in + fan_out))
static float xavier_init(int fan_in, int fan_out) {
    float stddev = sqrtf(2.0f / (fan_in + fan_out));
    // Box-Muller transform for normal distribution
    float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1);
    float u2 = (float)(rand() + 1) / ((float)RAND_MAX + 1);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z * stddev;
}

// Float32 to BFloat16 conversion
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

typedef struct {
    char* name;
    int64_t* shape;
    int n_dims;
    size_t num_elements;
    uint16_t* data;  // BF16 data
} weight_t;

// Build vocab.json and tokenizer.json from a vocab.txt file
// Returns total token count (including specials), or -1 on error
static int build_tokenizer(const char* vocab_path, const char* output_dir) {
    FILE* f = fopen(vocab_path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", vocab_path);
        return -1;
    }

    // Read words from vocab.txt
    char** words = NULL;
    int word_count = 0;
    int word_cap = 1024;
    words = malloc(word_cap * sizeof(char*));

    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        // Strip newline/whitespace
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r' || line[len-1] == ' '))
            line[--len] = '\0';
        if (len == 0) continue;

        if (word_count >= word_cap) {
            word_cap *= 2;
            words = realloc(words, word_cap * sizeof(char*));
        }
        words[word_count++] = strdup(line);
    }
    fclose(f);

    printf("  Read %d words from %s\n", word_count, vocab_path);

    // Build vocab.json: special tokens + words
    cJSON* vocab = cJSON_CreateObject();
    cJSON_AddNumberToObject(vocab, "<|pad|>", 0);
    cJSON_AddNumberToObject(vocab, "<|bos|>", 1);
    cJSON_AddNumberToObject(vocab, "<|eos|>", 2);
    for (int i = 0; i < word_count; i++) {
        cJSON_AddNumberToObject(vocab, words[i], i + 3);
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/vocab.json", output_dir);
    char* vocab_str = cJSON_Print(vocab);
    FILE* out = fopen(path, "w");
    if (!out) { fprintf(stderr, "ERROR: Cannot write %s\n", path); return -1; }
    fputs(vocab_str, out);
    fclose(out);
    free(vocab_str);
    printf("  ✅ Wrote %s (%d tokens: 3 special + %d words)\n", path, word_count + 3, word_count);

    // Build tokenizer.json (HuggingFace WordLevel format)
    cJSON* tok = cJSON_CreateObject();
    cJSON_AddStringToObject(tok, "version", "1.0");
    cJSON_AddNullToObject(tok, "truncation");
    cJSON_AddNullToObject(tok, "padding");

    // added_tokens
    cJSON* added = cJSON_CreateArray();
    const char* special_names[] = {"<|pad|>", "<|bos|>", "<|eos|>"};
    for (int i = 0; i < 3; i++) {
        cJSON* t = cJSON_CreateObject();
        cJSON_AddNumberToObject(t, "id", i);
        cJSON_AddStringToObject(t, "content", special_names[i]);
        cJSON_AddBoolToObject(t, "single_word", 0);
        cJSON_AddBoolToObject(t, "lstrip", 0);
        cJSON_AddBoolToObject(t, "rstrip", 0);
        cJSON_AddBoolToObject(t, "normalized", 0);
        cJSON_AddBoolToObject(t, "special", 1);
        cJSON_AddItemToArray(added, t);
    }
    cJSON_AddItemToObject(tok, "added_tokens", added);

    cJSON_AddNullToObject(tok, "normalizer");

    // pre_tokenizer
    cJSON* pre_tok = cJSON_CreateObject();
    cJSON_AddStringToObject(pre_tok, "type", "WhitespaceSplit");
    cJSON_AddItemToObject(tok, "pre_tokenizer", pre_tok);

    // post_processor
    cJSON* post = cJSON_CreateObject();
    cJSON_AddStringToObject(post, "type", "TemplateProcessing");
    cJSON* single = cJSON_CreateArray();
    cJSON* s1 = cJSON_CreateObject(); cJSON* s1inner = cJSON_CreateObject();
    cJSON_AddStringToObject(s1inner, "id", "<|bos|>"); cJSON_AddNumberToObject(s1inner, "type_id", 0);
    cJSON_AddItemToObject(s1, "SpecialToken", s1inner); cJSON_AddItemToArray(single, s1);
    cJSON* s2 = cJSON_CreateObject(); cJSON* s2inner = cJSON_CreateObject();
    cJSON_AddStringToObject(s2inner, "id", "A"); cJSON_AddNumberToObject(s2inner, "type_id", 0);
    cJSON_AddItemToObject(s2, "Sequence", s2inner); cJSON_AddItemToArray(single, s2);
    cJSON_AddItemToObject(post, "single", single);
    cJSON* pair = cJSON_CreateArray();
    cJSON* p1 = cJSON_CreateObject(); cJSON* p1inner = cJSON_CreateObject();
    cJSON_AddStringToObject(p1inner, "id", "<|bos|>"); cJSON_AddNumberToObject(p1inner, "type_id", 0);
    cJSON_AddItemToObject(p1, "SpecialToken", p1inner); cJSON_AddItemToArray(pair, p1);
    cJSON* p2 = cJSON_CreateObject(); cJSON* p2inner = cJSON_CreateObject();
    cJSON_AddStringToObject(p2inner, "id", "A"); cJSON_AddNumberToObject(p2inner, "type_id", 0);
    cJSON_AddItemToObject(p2, "Sequence", p2inner); cJSON_AddItemToArray(pair, p2);
    cJSON* p3 = cJSON_CreateObject(); cJSON* p3inner = cJSON_CreateObject();
    cJSON_AddStringToObject(p3inner, "id", "B"); cJSON_AddNumberToObject(p3inner, "type_id", 1);
    cJSON_AddItemToObject(p3, "Sequence", p3inner); cJSON_AddItemToArray(pair, p3);
    cJSON_AddItemToObject(post, "pair", pair);
    cJSON* sp_tokens = cJSON_CreateObject();
    cJSON* bos_entry = cJSON_CreateObject();
    cJSON_AddStringToObject(bos_entry, "id", "<|bos|>");
    cJSON* bos_ids = cJSON_CreateArray(); cJSON_AddItemToArray(bos_ids, cJSON_CreateNumber(1));
    cJSON_AddItemToObject(bos_entry, "ids", bos_ids);
    cJSON* bos_toks = cJSON_CreateArray(); cJSON_AddItemToArray(bos_toks, cJSON_CreateString("<|bos|>"));
    cJSON_AddItemToObject(bos_entry, "tokens", bos_toks);
    cJSON_AddItemToObject(sp_tokens, "<|bos|>", bos_entry);
    cJSON_AddItemToObject(post, "special_tokens", sp_tokens);
    cJSON_AddItemToObject(tok, "post_processor", post);

    cJSON_AddNullToObject(tok, "decoder");

    // model (WordLevel with vocab embedded)
    cJSON* model = cJSON_CreateObject();
    cJSON_AddStringToObject(model, "type", "WordLevel");
    cJSON_AddItemToObject(model, "vocab", cJSON_Duplicate(vocab, 1));
    cJSON_AddNullToObject(model, "unk_token");
    cJSON_AddItemToObject(tok, "model", model);

    snprintf(path, sizeof(path), "%s/tokenizer.json", output_dir);
    char* tok_str = cJSON_Print(tok);
    out = fopen(path, "w");
    if (!out) { fprintf(stderr, "ERROR: Cannot write %s\n", path); return -1; }
    fputs(tok_str, out);
    fclose(out);
    free(tok_str);
    printf("  ✅ Wrote %s (WordLevel, %d tokens)\n", path, word_count + 3);

    int total_tokens = word_count + 3;
    cJSON_Delete(vocab);
    cJSON_Delete(tok);
    for (int i = 0; i < word_count; i++) free(words[i]);
    free(words);
    return total_tokens;
}

int main(int argc, char** argv) {
    // Check for --optimal flag FIRST (before argc check)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--optimal") == 0) {
            // Fork to GPU query tool which generates config.json
            // Then continue with normal weight init
            const char* dir = NULL;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                dir = argv[i + 1];
            }
            if (!dir) {
                fprintf(stderr, "Usage: %s --optimal <output_dir> [--vocab N]\n", argv[0]);
                fprintf(stderr, "\nQueries GPU hardware, solves topology equations,\n");
                fprintf(stderr, "generates config.json, then initializes weights.\n");
                fprintf(stderr, "\nRequires: tests/gpu-profile-query (build with: make cuda-tools)\n");
                return 1;
            }

            // Build the command — pass through any extra flags
            char cmd[2048];
            int pos = snprintf(cmd, sizeof(cmd), "tests/gpu-profile-query --optimal %s", dir);

            // Pass --vocab N if provided
            for (int j = 1; j < argc; j++) {
                if (strcmp(argv[j], "--vocab") == 0 && j + 1 < argc) {
                    pos += snprintf(cmd + pos, sizeof(cmd) - pos, " --vocab %s", argv[j + 1]);
                    break;
                }
            }

            printf("\n  Running GPU topology optimizer...\n");
            printf("  Command: %s\n\n", cmd);
            int ret = system(cmd);
            if (ret != 0) {
                fprintf(stderr, "ERROR: GPU topology optimizer failed (exit %d)\n", ret);
                fprintf(stderr, "Make sure tests/gpu-profile-query is built: make cuda-tools\n");
                return 1;
            }

            // Now generate the output path and continue with normal init
            char output_path_buf[512];
            snprintf(output_path_buf, sizeof(output_path_buf), "%s/model.safetensors", dir);

            // Reconstruct argv for the normal init flow
            argv[1] = (char*)dir;
            argv[2] = output_path_buf;
            argc = 3;
            break;
        }
    }

    if (argc < 3) {
        printf("\n");
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("  Seraph Random Weight Initializer\n");
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("\n");
        printf("Usage: %s <config_dir> <output.safetensors> [--vocab]\n", argv[0]);
        printf("\n");
        printf("Creates random weights for training from scratch.\n");
        printf("No pretrained weights - pure emergence from noise\n");
        printf("\n");
        printf("If <config_dir>/vocab.txt exists, vocab.json and tokenizer.json\n");
        printf("are built automatically into <config_dir>/\n");
        printf("\n");
        printf("Options:\n");
        printf("  --vocab [FILE]  Force tokenizer build. If FILE omitted, uses\n");
        printf("                  <config_dir>/vocab.txt (one word per line)\n");
        printf("  --optimal <dir> Query GPU, solve topology equations, generate\n");
        printf("                  config.json + weights. Requires CUDA GPU tool.\n");
        printf("\n");
        return 1;
    }

    const char* config_dir = argv[1];
    const char* output_path = argv[2];

    // Check for --vocab flag or auto-detect vocab.txt in config_dir
    const char* vocab_path = NULL;
    int vocab_forced = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--vocab") == 0) {
            vocab_forced = 1;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                vocab_path = argv[++i];
            }
        }
    }

    // Auto-detect vocab.txt in config_dir if no explicit path
    char auto_vocab[512];
    if (!vocab_path) {
        snprintf(auto_vocab, sizeof(auto_vocab), "%s/vocab.txt", config_dir);
        FILE* test = fopen(auto_vocab, "r");
        if (test) {
            fclose(test);
            vocab_path = auto_vocab;
            if (!vocab_forced)
                printf("\n  Found %s - building tokenizer automatically\n", auto_vocab);
        } else if (vocab_forced) {
            fprintf(stderr, "ERROR: --vocab specified but no vocab.txt found in %s\n", config_dir);
            return 1;
        }
    }

    int vocab_tok_count = 0;

    // Seed random with time
    srand(time(NULL));

    // Load config.json
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.json", config_dir);

    FILE* f = fopen(config_path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", config_path);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long config_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* config_json = malloc(config_size + 1);
    fread(config_json, 1, config_size, f);
    config_json[config_size] = '\0';
    fclose(f);

    cJSON* config = cJSON_Parse(config_json);
    free(config_json);

    if (!config) {
        fprintf(stderr, "ERROR: Failed to parse config.json\n");
        return 1;
    }

    // Extract config values
    cJSON* vocab_item = cJSON_GetObjectItem(config, "vocab_size");
    int vocab_size = vocab_item ? vocab_item->valueint : 0;
    int hidden_size = cJSON_GetObjectItem(config, "hidden_size")->valueint;
    int num_layers = cJSON_GetObjectItem(config, "num_hidden_layers")->valueint;
    int intermediate_size = cJSON_GetObjectItem(config, "intermediate_size")->valueint;
    int num_heads = cJSON_GetObjectItem(config, "num_attention_heads")->valueint;
    int num_kv_heads = cJSON_GetObjectItem(config, "num_key_value_heads")->valueint;
    int head_dim = cJSON_GetObjectItem(config, "head_dim")->valueint;
    int tie_embeddings = 1;
    cJSON* tie = cJSON_GetObjectItem(config, "tie_word_embeddings");
    if (tie) tie_embeddings = cJSON_IsTrue(tie);

    // Video model parameters
    cJSON* num_queries_item = cJSON_GetObjectItem(config, "num_queries");
    int num_queries = num_queries_item ? num_queries_item->valueint : 0;
    int is_video_model = num_queries > 0;

    int patch_size[3] = {1, 1, 1};
    int in_channels = 3;
    int query_dim = hidden_size;
    int query_fusion_dim = hidden_size;

    if (is_video_model) {
        cJSON* patch_arr = cJSON_GetObjectItem(config, "patch_size");
        if (patch_arr && cJSON_IsArray(patch_arr)) {
            patch_size[0] = cJSON_GetArrayItem(patch_arr, 0)->valueint;
            patch_size[1] = cJSON_GetArrayItem(patch_arr, 1)->valueint;
            patch_size[2] = cJSON_GetArrayItem(patch_arr, 2)->valueint;
        }
        cJSON* in_ch = cJSON_GetObjectItem(config, "in_channels");
        if (in_ch) in_channels = in_ch->valueint;
        cJSON* qd = cJSON_GetObjectItem(config, "query_dim");
        if (qd) query_dim = qd->valueint;
        cJSON* qfd = cJSON_GetObjectItem(config, "query_fusion_dim");
        if (qfd) query_fusion_dim = qfd->valueint;
    }

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Seraph Random Weight Initialization\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    if (is_video_model) {
        printf("  Model type:      VIDEO\n");
        printf("  Patch size:      [%d, %d, %d]\n", patch_size[0], patch_size[1], patch_size[2]);
        printf("  In channels:     %d\n", in_channels);
        printf("  Num queries:     %d\n", num_queries);
        printf("  Query dim:       %d\n", query_dim);
    } else {
        printf("  Model type:      LLM\n");
        printf("  Vocab size:      %d\n", vocab_size);
        printf("  Tie embeddings:  %s\n", tie_embeddings ? "yes" : "no");
    }
    printf("  Hidden size:     %d\n", hidden_size);
    printf("  Num layers:      %d\n", num_layers);
    printf("  Intermediate:    %d\n", intermediate_size);
    printf("  Attention heads: %d\n", num_heads);
    printf("  KV heads:        %d\n", num_kv_heads);
    printf("  Head dim:        %d\n", head_dim);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Build tokenizer if vocab found (after config banner so output flows naturally)
    if (vocab_path) {
        printf("[1/5] Building tokenizer from vocabulary...\n");
        vocab_tok_count = build_tokenizer(vocab_path, config_dir);
        if (vocab_tok_count < 0) {
            return 1;
        }
        if (vocab_tok_count != vocab_size) {
            printf("\n  MISMATCH: vocab.txt has %d tokens but config.json says vocab_size=%d\n", vocab_tok_count, vocab_size);
        }
        printf("\n");
    }

    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    // Calculate total tensors
    // LLM: embed + (per-layer weights) + final norm + lm_head
    // Video: patch_embed + queries + query_fusion + (per-layer) + final norm + output_head
    int num_tensors;
    if (is_video_model) {
        // patch_embed + queries + query_fusion + (layers * 9) + norm + class_head + mask_head
        num_tensors = 1 + 1 + 1 + (num_layers * 9) + 1 + 1 + 1;
    } else {
        num_tensors = 1 + (num_layers * 9) + 1 + (tie_embeddings ? 0 : 1);
    }

    weight_t* weights = calloc(num_tensors, sizeof(weight_t));
    int w_idx = 0;
    size_t total_params = 0;
    size_t data_offset = 0;
    int step = vocab_path ? 1 : 0;
    int total_steps = 4 + step;

    printf("[%d/%d] Generating %s weights...\n", step + 1, total_steps,
           is_video_model ? "patch embedding" : "embedding");

    if (is_video_model) {
        // Patch embedding: [hidden_size, in_channels * patch_t * patch_h * patch_w]
        int patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2];
        {
            weight_t* w = &weights[w_idx++];
            w->name = strdup("model.patch_embed.weight");
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = hidden_size;
            w->shape[1] = patch_dim;
            w->num_elements = hidden_size * patch_dim;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(patch_dim, hidden_size));
            }
            total_params += w->num_elements;
            printf("    patch_embed: [%d, %d] = %zu params\n", hidden_size, patch_dim, w->num_elements);
        }

        // Learnable queries: [num_queries, query_dim]
        {
            weight_t* w = &weights[w_idx++];
            w->name = strdup("model.queries.weight");
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = num_queries;
            w->shape[1] = query_dim;
            w->num_elements = num_queries * query_dim;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(query_dim, num_queries));
            }
            total_params += w->num_elements;
            printf("    queries: [%d, %d] = %zu params\n", num_queries, query_dim, w->num_elements);
        }

        // Query fusion linear: [query_dim, query_fusion_dim]
        {
            weight_t* w = &weights[w_idx++];
            w->name = strdup("model.query_fusion.weight");
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = query_dim;
            w->shape[1] = query_fusion_dim;
            w->num_elements = query_dim * query_fusion_dim;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(query_fusion_dim, query_dim));
            }
            total_params += w->num_elements;
            printf("    query_fusion: [%d, %d] = %zu params\n", query_dim, query_fusion_dim, w->num_elements);
        }
    } else {
        // LLM: Token embedding [vocab_size, hidden_size]
        weight_t* w = &weights[w_idx++];
        w->name = strdup("model.embed_tokens.weight");
        w->n_dims = 2;
        w->shape = malloc(2 * sizeof(int64_t));
        w->shape[0] = vocab_size;
        w->shape[1] = hidden_size;
        w->num_elements = vocab_size * hidden_size;
        w->data = malloc(w->num_elements * sizeof(uint16_t));
        for (size_t i = 0; i < w->num_elements; i++) {
            w->data[i] = f32_to_bf16(xavier_init(vocab_size, hidden_size));
        }
        total_params += w->num_elements;
        printf("    embed_tokens: [%d, %d] = %zu params\n", vocab_size, hidden_size, w->num_elements);
    }

    printf("[%d/%d] Generating layer weights...\n", step + 2, total_steps);

    // Per-layer weights
    for (int layer = 0; layer < num_layers; layer++) {
        char name[256];

        // Q projection: [q_dim, hidden_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer);
            w->name = strdup(name);
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = q_dim;
            w->shape[1] = hidden_size;
            w->num_elements = q_dim * hidden_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(hidden_size, q_dim));
            }
            total_params += w->num_elements;
        }

        // K projection: [kv_dim, hidden_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer);
            w->name = strdup(name);
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = kv_dim;
            w->shape[1] = hidden_size;
            w->num_elements = kv_dim * hidden_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(hidden_size, kv_dim));
            }
            total_params += w->num_elements;
        }

        // V projection: [kv_dim, hidden_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer);
            w->name = strdup(name);
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = kv_dim;
            w->shape[1] = hidden_size;
            w->num_elements = kv_dim * hidden_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(hidden_size, kv_dim));
            }
            total_params += w->num_elements;
        }

        // O projection: [hidden_size, q_dim]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer);
            w->name = strdup(name);
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = hidden_size;
            w->shape[1] = q_dim;
            w->num_elements = hidden_size * q_dim;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(q_dim, hidden_size));
            }
            total_params += w->num_elements;
        }

        // Gate projection: [intermediate_size, hidden_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer);
            w->name = strdup(name);
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = intermediate_size;
            w->shape[1] = hidden_size;
            w->num_elements = intermediate_size * hidden_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(hidden_size, intermediate_size));
            }
            total_params += w->num_elements;
        }

        // Up projection: [intermediate_size, hidden_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer);
            w->name = strdup(name);
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = intermediate_size;
            w->shape[1] = hidden_size;
            w->num_elements = intermediate_size * hidden_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(hidden_size, intermediate_size));
            }
            total_params += w->num_elements;
        }

        // Down projection: [hidden_size, intermediate_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer);
            w->name = strdup(name);
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = hidden_size;
            w->shape[1] = intermediate_size;
            w->num_elements = hidden_size * intermediate_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(intermediate_size, hidden_size));
            }
            total_params += w->num_elements;
        }

        // Input layernorm: [hidden_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer);
            w->name = strdup(name);
            w->n_dims = 1;
            w->shape = malloc(1 * sizeof(int64_t));
            w->shape[0] = hidden_size;
            w->num_elements = hidden_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(1.0f);  // Initialize to 1 for RMSNorm
            }
            total_params += w->num_elements;
        }

        // Post-attention layernorm: [hidden_size]
        {
            weight_t* w = &weights[w_idx++];
            snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer);
            w->name = strdup(name);
            w->n_dims = 1;
            w->shape = malloc(1 * sizeof(int64_t));
            w->shape[0] = hidden_size;
            w->num_elements = hidden_size;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(1.0f);  // Initialize to 1 for RMSNorm
            }
            total_params += w->num_elements;
        }

        if ((layer + 1) % 2 == 0 || layer == num_layers - 1) {
            printf("    Layer %d/%d complete\n", layer + 1, num_layers);
        }
    }

    printf("[%d/%d] Generating final norm...\n", step + 3, total_steps);

    // Final norm: [hidden_size]
    {
        weight_t* w = &weights[w_idx++];
        w->name = strdup("model.norm.weight");
        w->n_dims = 1;
        w->shape = malloc(1 * sizeof(int64_t));
        w->shape[0] = hidden_size;
        w->num_elements = hidden_size;
        w->data = malloc(w->num_elements * sizeof(uint16_t));
        for (size_t i = 0; i < w->num_elements; i++) {
            w->data[i] = f32_to_bf16(1.0f);
        }
        total_params += w->num_elements;
    }

    // Output heads
    if (is_video_model) {
        // Class head: [num_queries, num_classes] - for now use query_dim as proxy
        {
            weight_t* w = &weights[w_idx++];
            w->name = strdup("model.class_head.weight");
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = 256;  // num_classes placeholder
            w->shape[1] = query_dim;
            w->num_elements = 256 * query_dim;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(query_dim, 256));
            }
            total_params += w->num_elements;
            printf("    class_head: [256, %d] = %zu params\n", query_dim, w->num_elements);
        }

        // Mask head MLP: query_dim -> hidden -> 1 (per-pixel mask logit)
        {
            weight_t* w = &weights[w_idx++];
            w->name = strdup("model.mask_head.weight");
            w->n_dims = 2;
            w->shape = malloc(2 * sizeof(int64_t));
            w->shape[0] = hidden_size;
            w->shape[1] = query_dim;
            w->num_elements = hidden_size * query_dim;
            w->data = malloc(w->num_elements * sizeof(uint16_t));
            for (size_t i = 0; i < w->num_elements; i++) {
                w->data[i] = f32_to_bf16(xavier_init(query_dim, hidden_size));
            }
            total_params += w->num_elements;
            printf("    mask_head: [%d, %d] = %zu params\n", hidden_size, query_dim, w->num_elements);
        }
    } else if (!tie_embeddings) {
        // LM head (only if not tied)
        weight_t* w = &weights[w_idx++];
        w->name = strdup("lm_head.weight");
        w->n_dims = 2;
        w->shape = malloc(2 * sizeof(int64_t));
        w->shape[0] = vocab_size;
        w->shape[1] = hidden_size;
        w->num_elements = vocab_size * hidden_size;
        w->data = malloc(w->num_elements * sizeof(uint16_t));
        for (size_t i = 0; i < w->num_elements; i++) {
            w->data[i] = f32_to_bf16(xavier_init(hidden_size, vocab_size));
        }
        total_params += w->num_elements;
    }

    int actual_tensors = w_idx;
    printf("\n  Total tensors: %d\n", actual_tensors);
    printf("  Total parameters: %zu (%.2f M)\n", total_params, total_params / 1e6);

    printf("\n[%d/%d] Writing safetensors file...\n", step + 4, total_steps);

    // Build JSON header
    cJSON* header = cJSON_CreateObject();

    // Add metadata
    cJSON* metadata = cJSON_CreateObject();
    cJSON_AddStringToObject(metadata, "format", "pt");
    cJSON_AddStringToObject(metadata, "framework", "tetyah");
    cJSON_AddStringToObject(metadata, "initialization", "xavier_random");
    cJSON_AddItemToObject(header, "__metadata__", metadata);

    // Calculate offsets and add tensor entries
    data_offset = 0;
    for (int i = 0; i < actual_tensors; i++) {
        weight_t* w = &weights[i];
        size_t data_size = w->num_elements * sizeof(uint16_t);

        cJSON* tensor = cJSON_CreateObject();
        cJSON_AddStringToObject(tensor, "dtype", "BF16");

        cJSON* shape = cJSON_CreateArray();
        for (int j = 0; j < w->n_dims; j++) {
            cJSON_AddItemToArray(shape, cJSON_CreateNumber((double)w->shape[j]));
        }
        cJSON_AddItemToObject(tensor, "shape", shape);

        cJSON* offsets = cJSON_CreateArray();
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)data_offset));
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)(data_offset + data_size)));
        cJSON_AddItemToObject(tensor, "data_offsets", offsets);

        cJSON_AddItemToObject(header, w->name, tensor);

        data_offset += data_size;
    }

    char* header_json = cJSON_PrintUnformatted(header);
    uint64_t header_size = strlen(header_json);

    // Write file
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot create %s\n", output_path);
        return 1;
    }

    // Header size (8 bytes)
    fwrite(&header_size, sizeof(uint64_t), 1, out);

    // JSON header
    fwrite(header_json, 1, header_size, out);

    // Tensor data
    for (int i = 0; i < actual_tensors; i++) {
        fwrite(weights[i].data, sizeof(uint16_t), weights[i].num_elements, out);
    }

    fclose(out);
    free(header_json);
    cJSON_Delete(header);
    cJSON_Delete(config);

    // Get file size
    f = fopen(output_path, "rb");
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  Seraph weights initialized!\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Output: %s\n", output_path);
    printf("  Size: %.2f MB\n", file_size / (1024.0 * 1024.0));
    printf("  Parameters: %.2f M\n", total_params / 1e6);
    printf("  Initialization: Xavier/Glorot (normal)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("  Ready for consciousness training!\n");
    printf("\n");

    // Cleanup
    for (int i = 0; i < actual_tensors; i++) {
        free(weights[i].name);
        free(weights[i].shape);
        free(weights[i].data);
    }
    free(weights);

    return 0;
}
