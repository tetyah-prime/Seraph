// model_config.h - Model configuration and architecture detection
// Supports Llama, Mistral, Qwen, Granite, DeepSeek, and compatible architectures
// Part of TETYAH-PRIME's native training and inference engine

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

// Model architecture type
typedef enum {
    ARCH_LLAMA,      // Llama, Llama2, Llama3, Nemotron
    ARCH_QWEN,       // Qwen, Qwen2, Qwen3
    ARCH_MISTRAL,    // Mistral, Ministral
    ARCH_PHI,        // Microsoft Phi (fused qkv_proj/gate_up_proj - partial support)
    ARCH_GRANITE,    // IBM Granite
    ARCH_DEEPSEEK,   // DeepSeek (MoE - not yet supported)
    ARCH_UNKNOWN
} model_arch_t;

// Chat template format
typedef enum {
    CHAT_FORMAT_CHATML,      // <|im_start|>role\ncontent<|im_end|> (Qwen, many others)
    CHAT_FORMAT_LLAMA2,      // [INST] <<SYS>>system<</SYS>> user [/INST]
    CHAT_FORMAT_LLAMA3,      // <|start_header_id|>role<|end_header_id|>content<|eot_id|>
    CHAT_FORMAT_MISTRAL,     // [INST] user [/INST]
    CHAT_FORMAT_ALPACA,      // ### Instruction:\n### Response:
    CHAT_FORMAT_TETYAH,      // User:\nTETYAH-PRIME:\n (our custom format)
    CHAT_FORMAT_RAW          // No formatting, just concatenate
} chat_format_t;

// Universal model configuration
typedef struct {
    // Model identity
    char model_name[256];
    char model_type[64];
    model_arch_t arch;

    // Core dimensions
    int vocab_size;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;

    // Attention configuration
    int num_attention_heads;
    int num_key_value_heads;  // For GQA (Grouped Query Attention)
    int head_dim;             // Computed: hidden_size / num_attention_heads
    int kv_dim;               // Computed: num_key_value_heads * head_dim

    // Normalization
    float rms_norm_eps;

    // RoPE (Rotary Position Embedding)
    float rope_theta;
    int max_position_embeddings;

    // Video/ViT mode: QK normalization across heads before attention
    int use_qk_norm;  // 0 = LLM mode (disabled), 1 = Video/ViT mode (enabled)

    // Video model parameters (only used when model_type contains "video")
    int patch_size[3];           // [temporal, height, width] patch dimensions
    int temporal_window;         // Number of frames processed together
    int input_resolution[2];     // [H, W] input resolution
    int in_channels;             // Input channels (3 for RGB)
    int num_queries;             // Learnable object queries
    int query_dim;               // Query embedding dimension
    int query_propagation;       // 1 = enable Q[t-1] propagation (VidEoMT style)
    int query_fusion_dim;        // Dimension for query fusion linear layer

    // Video output heads
    int num_classes;             // Number of output classes (classification/detection)
    int mask_output;             // 1 = output masks (segmentation), 0 = classification only
    int mask_dim;                // Hidden dim for mask MLP (default: query_dim)

    // CLIP alignment (semantic embeddings)
    int clip_embed_dim;          // CLIP embedding dimension (512 for ViT-B, 768 for ViT-L)
    char clip_embeddings_path[512]; // Path to pre-computed class embeddings (.safetensors)
    float clip_loss_weight;      // Weight for contrastive loss (0.0 = disabled)

    // Activation
    char hidden_act[32];      // "silu", "gelu", etc.

    // Special tokens
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;

    // Chat template configuration
    chat_format_t chat_format;
    char im_start_token[32];      // "<|im_start|>" for ChatML
    char im_end_token[32];        // "<|im_end|>" for ChatML
    char system_token[64];        // For system prompts
    char user_token[64];          // User role marker
    char assistant_token[64];     // Assistant role marker

    // Weight configuration
    bool tie_word_embeddings;
    char torch_dtype[16];     // "bfloat16", "float16", etc.

    // File info
    int num_model_files;      // 1 for small models, 4 for 7B+
    char model_dir[512];

} model_config_t;

// Load configuration from model directory (reads config.json)
model_config_t* model_config_load(const char* model_dir);

// Free configuration
void model_config_free(model_config_t* config);

// Print configuration summary
void model_config_print(const model_config_t* config);

// Get architecture from model_type string
model_arch_t model_config_detect_arch(const char* model_type);

// Detect chat format from architecture and model type
chat_format_t model_config_detect_chat_format(model_arch_t arch, const char* model_type);

// Get chat format name for debugging
const char* model_config_chat_format_name(chat_format_t format);

#endif // MODEL_CONFIG_H
