// model_config.c - Parse model configuration from config.json
// Part of TETYAH-PRIME's native training and inference engine

#include "../include/model_config.h"
#include "../include/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper: Read entire file into string
static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buffer = malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    return buffer;
}

// Helper: Get string from JSON with default
static void json_get_string(cJSON* json, const char* key, char* dest, size_t dest_size, const char* default_val) {
    cJSON* item = cJSON_GetObjectItem(json, key);
    if (item && cJSON_IsString(item)) {
        strncpy(dest, item->valuestring, dest_size - 1);
        dest[dest_size - 1] = '\0';
    } else {
        strncpy(dest, default_val, dest_size - 1);
        dest[dest_size - 1] = '\0';
    }
}

// Helper: Get int from JSON with default
static int json_get_int(cJSON* json, const char* key, int default_val) {
    cJSON* item = cJSON_GetObjectItem(json, key);
    if (item && cJSON_IsNumber(item)) {
        return item->valueint;
    }
    return default_val;
}

// Helper: Get float from JSON with default
static float json_get_float(cJSON* json, const char* key, float default_val) {
    cJSON* item = cJSON_GetObjectItem(json, key);
    if (item && cJSON_IsNumber(item)) {
        return (float)item->valuedouble;
    }
    return default_val;
}

// Helper: Get bool from JSON with default
static bool json_get_bool(cJSON* json, const char* key, bool default_val) {
    cJSON* item = cJSON_GetObjectItem(json, key);
    if (item) {
        if (cJSON_IsBool(item)) {
            return cJSON_IsTrue(item);
        }
    }
    return default_val;
}

// Detect architecture from model_type string
model_arch_t model_config_detect_arch(const char* model_type) {
    if (!model_type) return ARCH_UNKNOWN;

    if (strstr(model_type, "qwen") || strstr(model_type, "Qwen")) {
        return ARCH_QWEN;
    } else if (strstr(model_type, "mistral") || strstr(model_type, "Mistral") ||
               strstr(model_type, "ministral") || strstr(model_type, "Ministral")) {
        return ARCH_MISTRAL;
    } else if (strstr(model_type, "phi") || strstr(model_type, "Phi")) {
        return ARCH_PHI;
    } else if (strstr(model_type, "granite") || strstr(model_type, "Granite")) {
        return ARCH_GRANITE;
    } else if (strstr(model_type, "llama") || strstr(model_type, "Llama")) {
        return ARCH_LLAMA;
    } else if (strstr(model_type, "deepseek") || strstr(model_type, "DeepSeek")) {
        return ARCH_DEEPSEEK;
    }

    return ARCH_UNKNOWN;
}

// Detect chat format from architecture and model type
chat_format_t model_config_detect_chat_format(model_arch_t arch, const char* model_type) {
    // Check for specific model indicators
    bool is_instruct = model_type && (strstr(model_type, "instruct") ||
                                       strstr(model_type, "Instruct") ||
                                       strstr(model_type, "chat") ||
                                       strstr(model_type, "Chat"));

    switch (arch) {
        case ARCH_QWEN:
            // Qwen uses ChatML format
            return CHAT_FORMAT_CHATML;

        case ARCH_MISTRAL:
            return CHAT_FORMAT_MISTRAL;

        case ARCH_LLAMA:
            // Llama3 uses different format than Llama2
            if (model_type && strstr(model_type, "3")) {
                return CHAT_FORMAT_LLAMA3;
            }
            return CHAT_FORMAT_LLAMA2;

        case ARCH_PHI:
            return CHAT_FORMAT_CHATML;  // Phi-3+ uses ChatML

        case ARCH_GRANITE:
            return CHAT_FORMAT_CHATML;  // Granite typically uses ChatML

        default:
            // Fall back to raw for base models, TETYAH for unknown
            return is_instruct ? CHAT_FORMAT_CHATML : CHAT_FORMAT_TETYAH;
    }
}

// Get chat format name for debugging
const char* model_config_chat_format_name(chat_format_t format) {
    switch (format) {
        case CHAT_FORMAT_CHATML:  return "ChatML";
        case CHAT_FORMAT_LLAMA2:  return "Llama2";
        case CHAT_FORMAT_LLAMA3:  return "Llama3";
        case CHAT_FORMAT_MISTRAL: return "Mistral";
        case CHAT_FORMAT_ALPACA:  return "Alpaca";
        case CHAT_FORMAT_TETYAH:  return "TETYAH";
        case CHAT_FORMAT_RAW:     return "Raw";
        default:                  return "Unknown";
    }
}

// Setup chat template tokens based on format
static void setup_chat_tokens(model_config_t* config) {
    switch (config->chat_format) {
        case CHAT_FORMAT_CHATML:
            strcpy(config->im_start_token, "<|im_start|>");
            strcpy(config->im_end_token, "<|im_end|>");
            strcpy(config->system_token, "system");
            strcpy(config->user_token, "user");
            strcpy(config->assistant_token, "assistant");
            break;

        case CHAT_FORMAT_LLAMA2:
            strcpy(config->im_start_token, "[INST]");
            strcpy(config->im_end_token, "[/INST]");
            strcpy(config->system_token, "<<SYS>>");
            strcpy(config->user_token, "");
            strcpy(config->assistant_token, "");
            break;

        case CHAT_FORMAT_LLAMA3:
            strcpy(config->im_start_token, "<|start_header_id|>");
            strcpy(config->im_end_token, "<|eot_id|>");
            strcpy(config->system_token, "system");
            strcpy(config->user_token, "user");
            strcpy(config->assistant_token, "assistant");
            break;

        case CHAT_FORMAT_MISTRAL:
            strcpy(config->im_start_token, "[INST]");
            strcpy(config->im_end_token, "[/INST]");
            strcpy(config->system_token, "");
            strcpy(config->user_token, "");
            strcpy(config->assistant_token, "");
            break;

        case CHAT_FORMAT_ALPACA:
            strcpy(config->im_start_token, "### ");
            strcpy(config->im_end_token, "\n\n");
            strcpy(config->system_token, "Instruction:");
            strcpy(config->user_token, "Input:");
            strcpy(config->assistant_token, "Response:");
            break;

        case CHAT_FORMAT_TETYAH:
        default:
            strcpy(config->im_start_token, "");
            strcpy(config->im_end_token, "\n\n");
            strcpy(config->system_token, "SystemKernel:");
            strcpy(config->user_token, "User:");
            strcpy(config->assistant_token, "TETYAH-PRIME:");
            break;
    }
}

// Load configuration from model directory
model_config_t* model_config_load(const char* model_dir) {
    // Build path to config.json
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

    // Read config file
    char* json_str = read_file(config_path);
    if (!json_str) {
        fprintf(stderr, "ERROR: Cannot read %s\n", config_path);
        return NULL;
    }

    // Parse JSON
    cJSON* json = cJSON_Parse(json_str);
    free(json_str);

    if (!json) {
        fprintf(stderr, "ERROR: Failed to parse config.json\n");
        return NULL;
    }

    // Allocate config struct
    model_config_t* config = calloc(1, sizeof(model_config_t));
    if (!config) {
        cJSON_Delete(json);
        return NULL;
    }

    // Store model directory
    strncpy(config->model_dir, model_dir, sizeof(config->model_dir) - 1);

    // Check for multimodal models that nest text config under "text_config"
    // (e.g., Mistral3ForConditionalGeneration, LlavaForConditionalGeneration)
    cJSON* text_cfg = cJSON_GetObjectItem(json, "text_config");
    cJSON* src = text_cfg ? text_cfg : json;

    // Parse model identity (check text_config first, then top-level)
    json_get_string(src, "model_type", config->model_type, sizeof(config->model_type), "");
    if (config->model_type[0] == '\0') {
        // Fallback to top-level model_type
        json_get_string(json, "model_type", config->model_type, sizeof(config->model_type), "unknown");
    }
    config->arch = model_config_detect_arch(config->model_type);

    if (text_cfg) {
        printf("  📦 Multimodal model detected, using text_config\n");
    }

    // Parse core dimensions (from text config source)
    config->vocab_size = json_get_int(src, "vocab_size", 32000);
    config->hidden_size = json_get_int(src, "hidden_size", 4096);
    config->num_hidden_layers = json_get_int(src, "num_hidden_layers", 32);
    config->intermediate_size = json_get_int(src, "intermediate_size", 11008);

    // Parse attention configuration
    config->num_attention_heads = json_get_int(src, "num_attention_heads", 32);
    config->num_key_value_heads = json_get_int(src, "num_key_value_heads", config->num_attention_heads);

    // Get head_dim - explicit from config, or compute from hidden_size/num_heads
    int explicit_head_dim = json_get_int(src, "head_dim", 0);
    if (explicit_head_dim > 0) {
        config->head_dim = explicit_head_dim;  // Use explicit value (Qwen3-4B has 128)
    } else {
        config->head_dim = config->hidden_size / config->num_attention_heads;  // Compute
    }
    config->kv_dim = config->num_key_value_heads * config->head_dim;

    // Parse normalization
    config->rms_norm_eps = json_get_float(src, "rms_norm_eps", 1e-6f);

    // Parse RoPE (check rope_parameters sub-object as fallback)
    config->rope_theta = json_get_float(src, "rope_theta", 0.0f);
    if (config->rope_theta == 0.0f) {
        cJSON* rope_params = cJSON_GetObjectItem(src, "rope_parameters");
        if (rope_params) {
            config->rope_theta = json_get_float(rope_params, "rope_theta", 10000.0f);
        } else {
            config->rope_theta = 10000.0f;
        }
    }
    config->max_position_embeddings = json_get_int(src, "max_position_embeddings", 4096);

    // Video/ViT mode
    config->use_qk_norm = json_get_int(src, "use_qk_norm", 0);

    // Video model parameters
    cJSON* patch_arr = cJSON_GetObjectItem(src, "patch_size");
    if (patch_arr && cJSON_IsArray(patch_arr) && cJSON_GetArraySize(patch_arr) == 3) {
        config->patch_size[0] = cJSON_GetArrayItem(patch_arr, 0)->valueint;
        config->patch_size[1] = cJSON_GetArrayItem(patch_arr, 1)->valueint;
        config->patch_size[2] = cJSON_GetArrayItem(patch_arr, 2)->valueint;
    }
    cJSON* res_arr = cJSON_GetObjectItem(src, "input_resolution");
    if (res_arr && cJSON_IsArray(res_arr) && cJSON_GetArraySize(res_arr) == 2) {
        config->input_resolution[0] = cJSON_GetArrayItem(res_arr, 0)->valueint;
        config->input_resolution[1] = cJSON_GetArrayItem(res_arr, 1)->valueint;
    }
    config->temporal_window = json_get_int(src, "temporal_window", 1);
    config->in_channels = json_get_int(src, "in_channels", 3);
    config->num_queries = json_get_int(src, "num_queries", 0);
    config->query_dim = json_get_int(src, "query_dim", 0);
    config->query_propagation = json_get_bool(src, "query_propagation", false);
    config->query_fusion_dim = json_get_int(src, "query_fusion_dim", 0);
    config->num_classes = json_get_int(src, "num_classes", 0);  // 0 = auto from data
    config->mask_dim = json_get_int(src, "mask_dim", 0);  // 0 = use query_dim
    // Video models (num_queries > 0) default to mask output enabled
    int default_mask_output = config->num_queries > 0 ? 1 : 0;
    config->mask_output = json_get_int(src, "mask_output", default_mask_output);

    // CLIP alignment configuration
    config->clip_embed_dim = json_get_int(src, "clip_embed_dim", 0);  // 0 = use query_dim, else from CLIP model
    json_get_string(src, "clip_embeddings_path", config->clip_embeddings_path,
                    sizeof(config->clip_embeddings_path), "");
    config->clip_loss_weight = json_get_float(src, "clip_loss_weight", 0.0f);  // 0 = disabled

    // Parse activation
    json_get_string(src, "hidden_act", config->hidden_act, sizeof(config->hidden_act), "silu");

    // Parse special tokens
    config->bos_token_id = json_get_int(src, "bos_token_id", 1);
    config->eos_token_id = json_get_int(src, "eos_token_id", 2);
    config->pad_token_id = json_get_int(src, "pad_token_id", 0);

    // Parse weight configuration
    config->tie_word_embeddings = json_get_bool(src, "tie_word_embeddings", false);
    json_get_string(src, "torch_dtype", config->torch_dtype, sizeof(config->torch_dtype), "bfloat16");

    // Detect and setup chat format
    config->chat_format = model_config_detect_chat_format(config->arch, config->model_type);
    setup_chat_tokens(config);

    // Build model name
    snprintf(config->model_name, sizeof(config->model_name), "%s-%dB",
             config->model_type,
             (int)((float)config->vocab_size * config->hidden_size * 2 / 1e9));  // Rough param estimate

    // Count model files
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    FILE* f = fopen(path, "rb");
    if (f) {
        config->num_model_files = 1;
        fclose(f);
    } else {
        // Try sharded format - detect total from first file that exists
        config->num_model_files = 0;
        for (int total = 2; total <= 10; total++) {
            snprintf(path, sizeof(path), "%s/model-00001-of-%05d.safetensors", model_dir, total);
            f = fopen(path, "rb");
            if (f) {
                fclose(f);
                // Found the pattern, now count all shards
                for (int i = 1; i <= total; i++) {
                    snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.safetensors", model_dir, i, total);
                    f = fopen(path, "rb");
                    if (f) {
                        config->num_model_files++;
                        fclose(f);
                    }
                }
                break;
            }
        }
    }

    cJSON_Delete(json);
    return config;
}

// Free configuration
void model_config_free(model_config_t* config) {
    if (config) {
        free(config);
    }
}

// ANSI Colors
#define RESET_CLR   "\033[0m"
#define BOLD_CLR    "\033[1m"
#define GOLD_CLR    "\033[93m"

// Print configuration summary
void model_config_print(const model_config_t* config) {
    if (!config) return;

    const char* arch_names[] = {"LLAMA", "QWEN", "MISTRAL", "PHI", "GRANITE", "DEEPSEEK", "UNKNOWN"};

    printf("\n");
    printf(GOLD_CLR "═══════════════════════════════════════════════════════════════════" RESET_CLR "\n");
    printf("  " BOLD_CLR "MODEL CONFIGURATION" RESET_CLR "\n");
    printf(GOLD_CLR "═══════════════════════════════════════════════════════════════════" RESET_CLR "\n");
    printf("  Model type:        %s\n", config->model_type);
    printf("  Architecture:      %s\n", arch_names[config->arch]);
    printf("  Model files:       %d\n", config->num_model_files);
    printf(GOLD_CLR "───────────────────────────────────────────────────────────────────" RESET_CLR "\n");
    printf("  Vocab size:        %d\n", config->vocab_size);
    printf("  Hidden size:       %d\n", config->hidden_size);
    printf("  Num layers:        %d\n", config->num_hidden_layers);
    printf("  Intermediate:      %d\n", config->intermediate_size);
    printf(GOLD_CLR "───────────────────────────────────────────────────────────────────" RESET_CLR "\n");
    printf("  Attention heads:   %d\n", config->num_attention_heads);
    printf("  KV heads (GQA):    %d\n", config->num_key_value_heads);
    printf("  Head dim:          %d\n", config->head_dim);
    printf("  KV dim:            %d\n", config->kv_dim);
    printf(GOLD_CLR "───────────────────────────────────────────────────────────────────" RESET_CLR "\n");
    printf("  Activation:        %s\n", config->hidden_act);
    printf("  RMS norm eps:      %.1e\n", config->rms_norm_eps);
    printf("  RoPE theta:        %.1f\n", config->rope_theta);
    printf("  Max positions:     %d\n", config->max_position_embeddings);
    printf(GOLD_CLR "───────────────────────────────────────────────────────────────────" RESET_CLR "\n");
    printf("  Dtype:             %s\n", config->torch_dtype);
    printf("  Tie embeddings:    %s\n", config->tie_word_embeddings ? "yes" : "no");
    printf("  Chat format:       %s\n", model_config_chat_format_name(config->chat_format));
    if (config->use_qk_norm || config->num_queries > 0) {
        printf(GOLD_CLR "───────────────────────────────────────────────────────────────────" RESET_CLR "\n");
        printf("  QK norm:           %s\n", config->use_qk_norm ? "yes" : "no");
        if (config->num_queries > 0) {
            printf("  Patch size:        [%d, %d, %d]\n", config->patch_size[0], config->patch_size[1], config->patch_size[2]);
            printf("  Input resolution:  %dx%d\n", config->input_resolution[0], config->input_resolution[1]);
            printf("  Temporal window:   %d\n", config->temporal_window);
            printf("  Num queries:       %d\n", config->num_queries);
            printf("  Query propagation: %s\n", config->query_propagation ? "yes" : "no");
            printf("  Num classes:       %d\n", config->num_classes);
            if (config->clip_loss_weight > 0.0f) {
                printf("  CLIP embed dim:    %d\n", config->clip_embed_dim);
                printf("  CLIP loss weight:  %.2f\n", config->clip_loss_weight);
                if (config->clip_embeddings_path[0]) {
                    printf("  CLIP embeddings:   %s\n", config->clip_embeddings_path);
                }
            }
        }
    }
    printf(GOLD_CLR "═══════════════════════════════════════════════════════════════════" RESET_CLR "\n");
    printf("\n");
}
