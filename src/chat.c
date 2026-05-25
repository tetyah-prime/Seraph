// chat.c - Seraph native chat/response template
// Part of TETYAH-PRIME's native training and inference engine

#include "../include/chat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Debug flag
#define TETYAH_DEBUG 0

// Role name mapping
const char* tetyah_role_name(tetyah_role_t role) {
    switch (role) {
        case ROLE_USER:       return "User";
        case ROLE_HOST_USER:  return "HostUser";
        case ROLE_ADMIN:      return "SecurityAdmin";
        case ROLE_SYSTEM:     return "SystemKernel";
        case ROLE_ASSISTANT:  return "TETYAH-PRIME";
        default:              return "Unknown";
    }
}

// Create chat context
tetyah_chat_t* tetyah_chat_new(void) {
    tetyah_chat_t* chat = calloc(1, sizeof(tetyah_chat_t));
    chat->capacity = 16;
    chat->messages = calloc(chat->capacity, sizeof(tetyah_message_t));
    chat->num_messages = 0;
    return chat;
}

// Free chat context
void tetyah_chat_free(tetyah_chat_t* chat) {
    if (!chat) return;
    for (int i = 0; i < chat->num_messages; i++) {
        free(chat->messages[i].content);
    }
    free(chat->messages);
    if (chat->system_prompt) free(chat->system_prompt);
    free(chat);
}

// Set system prompt
void tetyah_chat_set_system(tetyah_chat_t* chat, const char* system_prompt) {
    if (chat->system_prompt) free(chat->system_prompt);
    chat->system_prompt = system_prompt ? strdup(system_prompt) : NULL;
}

// Add message to chat
void tetyah_chat_add(tetyah_chat_t* chat, tetyah_role_t role, const char* content) {
    if (chat->num_messages >= chat->capacity) {
        chat->capacity *= 2;
        chat->messages = realloc(chat->messages, chat->capacity * sizeof(tetyah_message_t));
    }
    chat->messages[chat->num_messages].role = role;
    chat->messages[chat->num_messages].content = strdup(content);
    chat->num_messages++;
}

// Format chat into prompt string (Seraph format)
// Format:
// SystemKernel:
// system prompt
//
// RoleName:
// content
//
// TETYAH-PRIME:
char* tetyah_chat_format(tetyah_chat_t* chat) {
    // Calculate total size needed
    size_t total = 0;
    if (chat->system_prompt) {
        total += strlen("SystemKernel:\n") + strlen(chat->system_prompt) + 2;
    }
    for (int i = 0; i < chat->num_messages; i++) {
        total += strlen(tetyah_role_name(chat->messages[i].role)) + 2;  // "Role:\n"
        total += strlen(chat->messages[i].content) + 2;                  // "content\n\n"
    }
    total += strlen("TETYAH-PRIME:\n") + 1;  // Final prompt for response

    char* prompt = malloc(total);
    prompt[0] = '\0';

    // Add system prompt first
    if (chat->system_prompt) {
        strcat(prompt, "SystemKernel:\n");
        strcat(prompt, chat->system_prompt);
        strcat(prompt, "\n\n");
    }

    for (int i = 0; i < chat->num_messages; i++) {
        strcat(prompt, tetyah_role_name(chat->messages[i].role));
        strcat(prompt, ":\n");
        strcat(prompt, chat->messages[i].content);
        strcat(prompt, "\n\n");
    }
    strcat(prompt, "TETYAH-PRIME:\n");

    return prompt;
}

// Format chat using model's native template
char* tetyah_chat_format_native(tetyah_chat_t* chat, const model_config_t* config) {
    // Calculate buffer size (generous estimate)
    size_t total = 4096;  // Base
    if (chat->system_prompt) total += strlen(chat->system_prompt) + 100;
    for (int i = 0; i < chat->num_messages; i++) {
        total += strlen(chat->messages[i].content) + 100;
    }

    char* prompt = malloc(total);
    prompt[0] = '\0';

    switch (config->chat_format) {
        case CHAT_FORMAT_CHATML:
            // <|im_start|>system\ncontent<|im_end|>\n<|im_start|>user\ncontent<|im_end|>\n<|im_start|>assistant\n
            if (chat->system_prompt) {
                strcat(prompt, config->im_start_token);
                strcat(prompt, config->system_token);
                strcat(prompt, "\n");
                strcat(prompt, chat->system_prompt);
                strcat(prompt, config->im_end_token);
                strcat(prompt, "\n");
            }
            for (int i = 0; i < chat->num_messages; i++) {
                strcat(prompt, config->im_start_token);
                if (chat->messages[i].role == ROLE_ASSISTANT) {
                    strcat(prompt, config->assistant_token);
                } else {
                    strcat(prompt, config->user_token);
                }
                strcat(prompt, "\n");
                strcat(prompt, chat->messages[i].content);
                strcat(prompt, config->im_end_token);
                strcat(prompt, "\n");
            }
            // Add assistant prompt
            strcat(prompt, config->im_start_token);
            strcat(prompt, config->assistant_token);
            strcat(prompt, "\n");
            break;

        case CHAT_FORMAT_LLAMA2:
            // [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST]
            strcat(prompt, "[INST] ");
            if (chat->system_prompt) {
                strcat(prompt, "<<SYS>>\n");
                strcat(prompt, chat->system_prompt);
                strcat(prompt, "\n<</SYS>>\n\n");
            }
            for (int i = 0; i < chat->num_messages; i++) {
                if (chat->messages[i].role == ROLE_ASSISTANT) {
                    strcat(prompt, " [/INST] ");
                    strcat(prompt, chat->messages[i].content);
                    strcat(prompt, " </s><s>[INST] ");
                } else {
                    strcat(prompt, chat->messages[i].content);
                }
            }
            strcat(prompt, " [/INST]");
            break;

        case CHAT_FORMAT_LLAMA3:
            // <|start_header_id|>system<|end_header_id|>\n\ncontent<|eot_id|>
            strcat(prompt, "<|begin_of_text|>");
            if (chat->system_prompt) {
                strcat(prompt, "<|start_header_id|>system<|end_header_id|>\n\n");
                strcat(prompt, chat->system_prompt);
                strcat(prompt, "<|eot_id|>");
            }
            for (int i = 0; i < chat->num_messages; i++) {
                strcat(prompt, "<|start_header_id|>");
                if (chat->messages[i].role == ROLE_ASSISTANT) {
                    strcat(prompt, "assistant");
                } else {
                    strcat(prompt, "user");
                }
                strcat(prompt, "<|end_header_id|>\n\n");
                strcat(prompt, chat->messages[i].content);
                strcat(prompt, "<|eot_id|>");
            }
            strcat(prompt, "<|start_header_id|>assistant<|end_header_id|>\n\n");
            break;

        case CHAT_FORMAT_MISTRAL:
            // [INST] user [/INST]
            strcat(prompt, "[INST] ");
            if (chat->system_prompt) {
                strcat(prompt, chat->system_prompt);
                strcat(prompt, "\n\n");
            }
            for (int i = 0; i < chat->num_messages; i++) {
                if (chat->messages[i].role == ROLE_ASSISTANT) {
                    strcat(prompt, " [/INST] ");
                    strcat(prompt, chat->messages[i].content);
                    strcat(prompt, "</s> [INST] ");
                } else {
                    strcat(prompt, chat->messages[i].content);
                }
            }
            strcat(prompt, " [/INST]");
            break;

        case CHAT_FORMAT_TETYAH:
        default:
            // Fall back to TETYAH format
            free(prompt);
            return tetyah_chat_format(chat);
    }

    return prompt;
}

// Sample from logits with temperature and repetition penalty
static int sample_token_with_penalty(float* logits, int vocab_size, float temperature,
                                      int* recent_tokens, int num_recent, float rep_penalty) {
    // Apply repetition penalty to recent tokens
    if (rep_penalty != 1.0f && recent_tokens && num_recent > 0) {
        for (int i = 0; i < num_recent; i++) {
            int tok = recent_tokens[i];
            if (tok >= 0 && tok < vocab_size) {
                if (logits[tok] > 0) {
                    logits[tok] /= rep_penalty;
                } else {
                    logits[tok] *= rep_penalty;
                }
            }
        }
    }

    if (temperature <= 0.0f) {
        // Greedy - just pick max
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Apply temperature
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Softmax with temperature
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum += logits[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }

    // Sample
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return vocab_size - 1;
}


static int is_nl_special_token(const char* raw) {
    return raw && (
        strcmp(raw, "<|nl|>") == 0 ||
        strcmp(raw, "<|newline|>") == 0
    );
}

static int is_newline_token(tokenizer_t* tokenizer, int token_id) {
    const char* raw = tokenizer_id_to_token(tokenizer, token_id);
    if (is_nl_special_token(raw)) {
        return 1;
    }

    const char* clean = tokenizer_id_to_token_clean(tokenizer, token_id);
    return clean && strcmp(clean, "\n") == 0;
}

// Generate response (autoregressive)
char* tetyah_generate_with_format(
    seraph_model_t* model,
    const model_config_t* config,
    tokenizer_t* tokenizer,
    const char* prompt,
    int max_tokens,
    float temperature,
    chat_format_t format
) {
    // Reset KV cache for new generation
    seraph_reset_cache(model);

    // Build stop token list from tokenizer's special tokens
    int stop_ids[16];
    int num_stop_ids = 0;

    // Add format-specific stop tokens
    switch (format) {
        case CHAT_FORMAT_CHATML:
            if (tokenizer->im_end_token_id >= 0) {
                stop_ids[num_stop_ids++] = tokenizer->im_end_token_id;
            }
            break;
        case CHAT_FORMAT_LLAMA3:
            if (tokenizer->eot_token_id >= 0) {
                stop_ids[num_stop_ids++] = tokenizer->eot_token_id;
            }
            break;
        default:
            break;
    }

    // Always add EOS token as a stop
    if (tokenizer->eos_token_id >= 0 && num_stop_ids < 16) {
        stop_ids[num_stop_ids++] = tokenizer->eos_token_id;
    }

    #if TETYAH_DEBUG
    printf("  Stop tokens: ");
    for (int i = 0; i < num_stop_ids; i++) {
        printf("%d ", stop_ids[i]);
    }
    printf("\n");
    #endif

    // Tokenize prompt
    int num_tokens;
    int* tokens = tokenizer_encode(tokenizer, prompt, &num_tokens);

    // Allocate space for generated tokens
    int* all_tokens = malloc((num_tokens + max_tokens) * sizeof(int));
    memcpy(all_tokens, tokens, num_tokens * sizeof(int));
    free(tokens);

    int total_tokens = num_tokens;

    #if TETYAH_DEBUG
    printf("  Generating (prompt: %d tokens, max new: %d, stop tokens: %d)...\n",
           num_tokens, max_tokens, num_stop_ids);
    #endif

    // Autoregressive generation
    int started_output = 0;  // Skip leading whitespace in streaming
    float rep_penalty = 1.3f;  // Repetition penalty (1.0 = none, 1.1-1.2 = light, 1.5 = heavy)
    int penalty_window = 64;   // Look back this many tokens for penalty
    int consecutive_newlines = 0;

    for (int i = 0; i < max_tokens; i++) {
        // Forward pass — returns logits for ALL positions [seq_len * vocab_size]
        float* logits = seraph_forward(model, config, all_tokens, total_tokens);
        if (!logits) break;

        // Sample from LAST position's logits (not position 0)
        float* last_logits = &logits[(total_tokens - 1) * config->vocab_size];
        int recent_start = (total_tokens > penalty_window) ? total_tokens - penalty_window : 0;
        int next_token = sample_token_with_penalty(last_logits, config->vocab_size, temperature,
                                                    &all_tokens[recent_start], total_tokens - recent_start,
                                                    rep_penalty);
        free(logits);

        // Check stop tokens (EOS + format-specific)
        int is_stop = 0;
        for (int s = 0; s < num_stop_ids; s++) {
            if (next_token == stop_ids[s]) {
                is_stop = 1;
                #if TETYAH_DEBUG
                printf("  [STOP TOKEN after %d tokens]\n", i + 1);
                #endif
                break;
            }
        }
        if (is_stop) break;

        if (is_newline_token(tokenizer, next_token)) {
            consecutive_newlines++;
            // Guardrail for degenerate "<|nl|>" loops in early checkpoints
            if (started_output && consecutive_newlines >= 8) {
                #if TETYAH_DEBUG
                printf("  [NEWLINE LOOP STOP after %d tokens]\n", i + 1);
                #endif
                break;
            }
        } else {
            consecutive_newlines = 0;
        }

        // Add to sequence
        all_tokens[total_tokens++] = next_token;

        // Print token as we generate (cleaned for display)
        const char* token_str;
        if (is_newline_token(tokenizer, next_token)) {
            token_str = "\n";
        } else {
            token_str = tokenizer_id_to_token_clean(tokenizer, next_token);
        }
        if (token_str) {
            // Skip leading whitespace
            if (!started_output) {
                const char* p = token_str;
                while (*p == ' ' || *p == '\n' || *p == '\t') p++;
                if (*p) {
                    started_output = 1;
                    printf("%s", p);
                }
            } else {
                // WordLevel tokenizer: add space before each token (spaces are implicit)
                if (tokenizer->is_word_level && token_str[0] != '\n') {
                    printf(" %s", token_str);
                } else {
                    printf("%s", token_str);
                }
            }
        }
        fflush(stdout);
    }
    printf("\n");  // End the streaming line

    // Decode generated portion only
    char* response = tokenizer_decode(tokenizer, &all_tokens[num_tokens], total_tokens - num_tokens);

    free(all_tokens);
    return response;
}


// High-level: chat and get response
char* tetyah_chat_respond(
    seraph_model_t* model,
    const model_config_t* config,
    tokenizer_t* tokenizer,
    tetyah_chat_t* chat,
    int max_tokens,
    float temperature
) {
    // Format chat to prompt - use model's native format for best results
    char* prompt = tetyah_chat_format_native(chat, config);

    #if TETYAH_DEBUG
    printf("\n--- PROMPT (%s format) ---\n%s--- END PROMPT ---\n\n",
           model_config_chat_format_name(config->chat_format), prompt);
    #endif

    // Generate response with format-aware stop tokens
    char* response = tetyah_generate_with_format(model, config, tokenizer, prompt,
                                                  max_tokens, temperature, config->chat_format);

    free(prompt);

    // Clean response before adding to history (strip partial special tokens)
    if (response) {
        // Remove any <| fragments that might confuse next turn
        char* clean = response;
        char* write = response;
        while (*clean) {
            if (strncmp(clean, "<|nl|>", 6) == 0) {
                *write++ = '\n';
                clean += 6;
                continue;
            }
            if (strncmp(clean, "<|newline|>", 11) == 0) {
                *write++ = '\n';
                clean += 11;
                continue;
            }
            if (clean[0] == '<' && clean[1] == '|') {
                // Skip until we find |> or end
                while (*clean && !(clean[0] == '|' && clean[1] == '>')) clean++;
                if (*clean) clean += 2;  // Skip |>
            } else {
                *write++ = *clean++;
            }
        }
        *write = '\0';

        // Only add if there's content left
        if (strlen(response) > 0) {
            tetyah_chat_add(chat, ROLE_ASSISTANT, response);
        }
    }

    return response;
}
