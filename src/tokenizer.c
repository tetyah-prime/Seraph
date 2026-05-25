// tokenizer.c - BPE Tokenizer implementation
#include "../include/tokenizer.h"
#include "../include/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Forward declarations
static void build_vocab_hash(tokenizer_t* tok);

// Helper: read entire file
static char* read_file_contents(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* content = malloc(fsize + 1);
    fread(content, 1, fsize, f);
    content[fsize] = '\0';
    fclose(f);
    return content;
}

// Load vocab.json (Qwen/GPT style: {"token": id, ...})
static char** load_vocab(const char* path, int* vocab_size) {
    char* json_str = read_file_contents(path);
    if (!json_str) {
        perror("fopen vocab.json");
        return NULL;
    }

    cJSON* root = cJSON_Parse(json_str);
    free(json_str);

    if (!root) {
        fprintf(stderr, "Failed to parse vocab.json: %s\n", cJSON_GetErrorPtr());
        return NULL;
    }

    // Count tokens
    int count = cJSON_GetArraySize(root);
    *vocab_size = count;

    // Allocate vocab array (indexed by token ID)
    char** vocab = calloc(count, sizeof(char*));

    // Parse each token
    cJSON* item = NULL;
    cJSON_ArrayForEach(item, root) {
        const char* token_str = item->string;
        int token_id = item->valueint;

        if (token_id >= 0 && token_id < count) {
            vocab[token_id] = strdup(token_str);
        }
    }

    cJSON_Delete(root);
    return vocab;
}

// Load tokenizer.json (HuggingFace style with model.vocab)
static char** load_vocab_from_tokenizer_json(const char* path, int* vocab_size, bpe_merge_t** merges_out, int* num_merges_out) {
    char* json_str = read_file_contents(path);
    if (!json_str) return NULL;

    cJSON* root = cJSON_Parse(json_str);
    free(json_str);

    if (!root) {
        fprintf(stderr, "Failed to parse tokenizer.json\n");
        return NULL;
    }

    // Navigate to model.vocab
    cJSON* model = cJSON_GetObjectItem(root, "model");
    if (!model) {
        cJSON_Delete(root);
        return NULL;
    }

    // Detect tokenizer type (WordLevel vs BPE)
    cJSON* model_type = cJSON_GetObjectItem(model, "type");
    if (model_type && cJSON_IsString(model_type) &&
        strcmp(model_type->valuestring, "WordLevel") == 0) {
        // Flag will be set on tok after return
        // Store in num_merges_out as signal: -1 = WordLevel
        if (num_merges_out) *num_merges_out = -1;
    }

    cJSON* vocab_obj = cJSON_GetObjectItem(model, "vocab");
    if (!vocab_obj) {
        cJSON_Delete(root);
        return NULL;
    }

    // Count and allocate vocab
    int count = cJSON_GetArraySize(vocab_obj);
    *vocab_size = count;
    char** vocab = calloc(count, sizeof(char*));

    // Parse vocab
    cJSON* item = NULL;
    cJSON_ArrayForEach(item, vocab_obj) {
        const char* token_str = item->string;
        int token_id = item->valueint;
        if (token_id >= 0 && token_id < count) {
            vocab[token_id] = strdup(token_str);
        }
    }

    // Parse merges if requested
    if (merges_out && num_merges_out) {
        cJSON* merges_arr = cJSON_GetObjectItem(model, "merges");
        if (merges_arr && cJSON_IsArray(merges_arr)) {
            int merge_count = cJSON_GetArraySize(merges_arr);
            *num_merges_out = merge_count;
            *merges_out = malloc(merge_count * sizeof(bpe_merge_t));

            int idx = 0;
            cJSON* merge_item = NULL;
            cJSON_ArrayForEach(merge_item, merges_arr) {
                if (cJSON_IsString(merge_item)) {
                    (*merges_out)[idx].pair = strdup(merge_item->valuestring);
                    (*merges_out)[idx].rank = idx;
                    idx++;
                }
            }
            *num_merges_out = idx;
        } else {
            *merges_out = NULL;
            *num_merges_out = 0;
        }
    }

    cJSON_Delete(root);
    return vocab;
}

// Load merges.txt
static bpe_merge_t* load_merges(const char* path, int* num_merges) {
    FILE* f = fopen(path, "r");
    if (!f) {
        perror("fopen merges.txt");
        return NULL;
    }

    // Count lines
    int count = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        count++;
    }
    rewind(f);

    *num_merges = count;
    bpe_merge_t* merges = malloc(count * sizeof(bpe_merge_t));

    // Parse each merge rule
    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < count) {
        // Remove newline
        line[strcspn(line, "\n")] = '\0';

        // Skip empty lines
        if (strlen(line) == 0) continue;

        // Store the pair and its rank
        merges[idx].pair = strdup(line);
        merges[idx].rank = idx;
        idx++;
    }

    fclose(f);
    *num_merges = idx;
    return merges;
}

// Load special/added tokens from tokenizer_config.json
static void load_special_tokens(tokenizer_t* tok, const char* model_dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer_config.json", model_dir);

    char* json_str = read_file_contents(path);
    if (!json_str) return;

    cJSON* root = cJSON_Parse(json_str);
    free(json_str);
    if (!root) return;

    // Parse added_tokens_decoder: {"151645": {"content": "<|im_end|>", ...}, ...}
    cJSON* added = cJSON_GetObjectItem(root, "added_tokens_decoder");
    if (added) {
        cJSON* item = NULL;
        cJSON_ArrayForEach(item, added) {
            int token_id = atoi(item->string);
            cJSON* content = cJSON_GetObjectItem(item, "content");
            if (content && cJSON_IsString(content)) {
                const char* token_str = content->valuestring;

                // Match known special tokens
                if (strcmp(token_str, "<|im_end|>") == 0) {
                    tok->im_end_token_id = token_id;
                } else if (strcmp(token_str, "<|im_start|>") == 0) {
                    tok->im_start_token_id = token_id;
                } else if (strcmp(token_str, "<|endoftext|>") == 0) {
                    tok->eos_token_id = token_id;
                } else if (strcmp(token_str, "<|eot_id|>") == 0) {
                    tok->eot_token_id = token_id;
                } else if (strcmp(token_str, "<|end_of_text|>") == 0) {
                    tok->eos_token_id = token_id;
                }
            }
        }
    }

    cJSON_Delete(root);
}

tokenizer_t* tokenizer_load(const char* model_dir) {
    tokenizer_t* tok = calloc(1, sizeof(tokenizer_t));

    char path[512];
    int loaded = 0;

    // Initialize special tokens to -1 (not found)
    tok->im_end_token_id = -1;
    tok->im_start_token_id = -1;
    tok->eot_token_id = -1;

    // Try 1: vocab.json + merges.txt (Qwen/GPT style)
    snprintf(path, sizeof(path), "%s/vocab.json", model_dir);
    FILE* f = fopen(path, "r");
    if (f) {
        fclose(f);
        printf("Loading vocab from: %s\n", path);
        tok->vocab = load_vocab(path, &tok->vocab_size);
        if (tok->vocab) {
            printf("  Loaded %d tokens\n", tok->vocab_size);

            snprintf(path, sizeof(path), "%s/merges.txt", model_dir);
            printf("Loading merges from: %s\n", path);
            tok->merges = load_merges(path, &tok->num_merges);
            if (tok->merges) {
                printf("  Loaded %d merge rules\n", tok->num_merges);
                loaded = 1;
            }
        }
    }

    // Try 2: tokenizer.json (HuggingFace/Mistral style)
    if (!loaded) {
        snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
        printf("Loading from tokenizer.json: %s\n", path);
        tok->vocab = load_vocab_from_tokenizer_json(path, &tok->vocab_size, &tok->merges, &tok->num_merges);
        if (tok->vocab) {
            // Check WordLevel signal (num_merges set to -1)
            if (tok->num_merges == -1) {
                tok->is_word_level = 1;
                tok->num_merges = 0;
                tok->merges = NULL;
                printf("  Loaded %d tokens (WordLevel tokenizer)\n", tok->vocab_size);
            } else {
                printf("  Loaded %d tokens", tok->vocab_size);
                if (tok->merges) {
                    printf(", %d merge rules", tok->num_merges);
                }
                printf("\n");
            }
            loaded = 1;
        }
    }

    if (!loaded) {
        fprintf(stderr, "ERROR: No tokenizer found (tried vocab.json and tokenizer.json)\n");
        free(tok);
        return NULL;
    }

    // Set default special tokens
    tok->bos_token_id = 1;   // Common default
    tok->eos_token_id = 2;   // Common default
    tok->pad_token_id = 0;

    // Load special tokens from tokenizer_config.json (overrides defaults)
    load_special_tokens(tok, model_dir);

    // Detect WordLevel tokenizer from tokenizer.json (regardless of load path)
    if (!tok->is_word_level) {
        snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
        char* tj = read_file_contents(path);
        if (tj) {
            cJSON* tj_root = cJSON_Parse(tj);
            free(tj);
            if (tj_root) {
                cJSON* tj_model = cJSON_GetObjectItem(tj_root, "model");
                if (tj_model) {
                    cJSON* tj_type = cJSON_GetObjectItem(tj_model, "type");
                    if (tj_type && cJSON_IsString(tj_type) &&
                        strcmp(tj_type->valuestring, "WordLevel") == 0) {
                        tok->is_word_level = 1;
                    }
                }
                cJSON_Delete(tj_root);
            }
        }
    }

    if (tok->is_word_level) {
        printf("  Tokenizer type: WordLevel (spaces are implicit)\n");
    }

    // Build hash table for O(1) token lookup
    printf("  Building vocab hash table...\n");
    build_vocab_hash(tok);

    return tok;
}

void tokenizer_free(tokenizer_t* tok) {
    if (!tok) return;

    // Free hash table
    if (tok->vocab_hash) {
        for (int i = 0; i < VOCAB_HASH_SIZE; i++) {
            vocab_hash_entry_t* entry = tok->vocab_hash[i];
            while (entry) {
                vocab_hash_entry_t* next = entry->next;
                free(entry);
                entry = next;
            }
        }
        free(tok->vocab_hash);
    }

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }

    if (tok->merges) {
        for (int i = 0; i < tok->num_merges; i++) {
            free(tok->merges[i].pair);
        }
        free(tok->merges);
    }

    free(tok);
}

const char* tokenizer_id_to_token(tokenizer_t* tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) {
        return NULL;
    }
    return tok->vocab[token_id];
}

// Clean BPE markers from a token for display
// Returns pointer to static buffer - not thread safe but fine for streaming
const char* tokenizer_id_to_token_clean(tokenizer_t* tok, int token_id) {
    static char clean_buf[256];

    const char* raw = tokenizer_id_to_token(tok, token_id);
    if (!raw) return NULL;

    // Copy and clean
    size_t j = 0;
    for (size_t i = 0; raw[i] && j < sizeof(clean_buf) - 1; ) {
        unsigned char c = (unsigned char)raw[i];

        // Ġ (U+0120) = UTF-8 0xC4 0xA0 → space
        if (c == 0xC4 && (unsigned char)raw[i+1] == 0xA0) {
            clean_buf[j++] = ' ';
            i += 2;
        }
        // Ċ (U+010A) = UTF-8 0xC4 0x8A → newline
        else if (c == 0xC4 && (unsigned char)raw[i+1] == 0x8A) {
            clean_buf[j++] = '\n';
            i += 2;
        }
        // ▁ (U+2581) = UTF-8 0xE2 0x96 0x81 → space (SentencePiece style)
        else if (c == 0xE2 && (unsigned char)raw[i+1] == 0x96 && (unsigned char)raw[i+2] == 0x81) {
            clean_buf[j++] = ' ';
            i += 3;
        }
        // Other BPE control chars (0xC4 0x8X range) → skip
        else if (c == 0xC4 && (unsigned char)raw[i+1] >= 0x80 && (unsigned char)raw[i+1] < 0xC0) {
            i += 2;
        }
        else {
            clean_buf[j++] = raw[i++];
        }
    }
    clean_buf[j] = '\0';

    return clean_buf;
}

// FNV-1a hash function for strings
static unsigned int hash_string(const char* str) {
    unsigned int hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

// Build hash table from vocab (called once during load)
static void build_vocab_hash(tokenizer_t* tok) {
    tok->vocab_hash = calloc(VOCAB_HASH_SIZE, sizeof(vocab_hash_entry_t*));

    for (int i = 0; i < tok->vocab_size; i++) {
        if (!tok->vocab[i]) continue;

        unsigned int h = hash_string(tok->vocab[i]) & (VOCAB_HASH_SIZE - 1);

        vocab_hash_entry_t* entry = malloc(sizeof(vocab_hash_entry_t));
        entry->token = tok->vocab[i];  // Point to existing string
        entry->id = i;
        entry->next = tok->vocab_hash[h];  // Chain
        tok->vocab_hash[h] = entry;
    }
}

int tokenizer_token_to_id(tokenizer_t* tok, const char* token) {
    // O(1) hash table lookup
    if (!tok->vocab_hash) {
        // Fallback to linear search if hash not built
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i] && strcmp(tok->vocab[i], token) == 0) {
                return i;
            }
        }
        return -1;
    }

    unsigned int h = hash_string(token) & (VOCAB_HASH_SIZE - 1);
    vocab_hash_entry_t* entry = tok->vocab_hash[h];

    while (entry) {
        if (strcmp(entry->token, token) == 0) {
            return entry->id;
        }
        entry = entry->next;
    }
    return -1;  // Not found
}

// Simplified BPE encoding (basic implementation)
// Full BPE is complex - this handles the core algorithm
// Convert text to BPE format (spaces become Ġ for GPT-style, ▁ for SentencePiece)
static char* text_to_bpe_format(const char* text, int add_leading_space) {
    size_t len = strlen(text);
    // Worst case: every char is a space -> 2 bytes each (Ġ)
    char* result = malloc(len * 3 + 4);
    size_t j = 0;

    // Optionally add leading space marker for continuation
    if (add_leading_space && len > 0 && text[0] != ' ') {
        // Add Ġ (0xC4 0xA0) at start
        result[j++] = 0xC4;
        result[j++] = 0xA0;
    }

    for (size_t i = 0; i < len; i++) {
        if (text[i] == ' ') {
            // Space -> Ġ (U+0120 = 0xC4 0xA0)
            result[j++] = 0xC4;
            result[j++] = 0xA0;
        } else if (text[i] == '\n') {
            // Newline -> Ċ (U+010A = 0xC4 0x8A)
            result[j++] = 0xC4;
            result[j++] = 0x8A;
        } else {
            result[j++] = text[i];
        }
    }
    result[j] = '\0';
    return result;
}

// BPE-encode a text segment (no special token handling)
static int bpe_encode_segment(tokenizer_t* tok, const char* text, int* tokens, int max_tokens) {
    int token_count = 0;

    char* bpe_text = text_to_bpe_format(text, 0);
    size_t bpe_len = strlen(bpe_text);

    size_t pos = 0;
    while (pos < bpe_len && token_count < max_tokens) {
        int matched = 0;
        int max_len = (bpe_len - pos < 32 ? bpe_len - pos : 32);

        for (int len = max_len; len > 0; len--) {
            char substr[64];
            if ((size_t)len >= sizeof(substr)) continue;

            strncpy(substr, bpe_text + pos, len);
            substr[len] = '\0';

            int token_id = tokenizer_token_to_id(tok, substr);
            if (token_id >= 0) {
                tokens[token_count++] = token_id;
                pos += len;
                matched = 1;
                break;
            }
        }

        if (!matched) {
            char byte[2] = {bpe_text[pos], '\0'};
            int token_id = tokenizer_token_to_id(tok, byte);
            if (token_id >= 0) {
                tokens[token_count++] = token_id;
            }
            pos++;
        }
    }

    free(bpe_text);
    return token_count;
}

int* tokenizer_encode(tokenizer_t* tok, const char* text, int* num_tokens) {
    size_t text_len = strlen(text);

    // Allocate space for worst case (each byte = one token)
    int* tokens = malloc(text_len * 2 * sizeof(int));
    int token_count = 0;

    // Split on special tokens (<|...|>) then BPE-encode the segments between them
    const char* pos = text;
    while (*pos) {
        // Look for next special token pattern: <|...|>
        const char* special_start = strstr(pos, "<|");
        if (!special_start) {
            // No more special tokens — BPE-encode the rest
            if (*pos) {
                token_count += bpe_encode_segment(tok, pos, &tokens[token_count],
                                                   (int)(text_len * 2) - token_count);
            }
            break;
        }

        // BPE-encode text before the special token
        if (special_start > pos) {
            // Extract the segment before the special token
            size_t seg_len = special_start - pos;
            char* segment = malloc(seg_len + 1);
            memcpy(segment, pos, seg_len);
            segment[seg_len] = '\0';

            token_count += bpe_encode_segment(tok, segment, &tokens[token_count],
                                               (int)(text_len * 2) - token_count);
            free(segment);
        }

        // Find the end of the special token
        const char* special_end = strstr(special_start + 2, "|>");
        if (!special_end) {
            // Malformed — BPE-encode the rest including the "<|"
            token_count += bpe_encode_segment(tok, special_start, &tokens[token_count],
                                               (int)(text_len * 2) - token_count);
            break;
        }
        special_end += 2;  // Include "|>"

        // Extract the special token string and look it up directly
        size_t special_len = special_end - special_start;
        char special_token[64];
        if (special_len < sizeof(special_token)) {
            memcpy(special_token, special_start, special_len);
            special_token[special_len] = '\0';

            int token_id = tokenizer_token_to_id(tok, special_token);
            if (token_id >= 0) {
                tokens[token_count++] = token_id;
            }
            // If not found in vocab, silently skip (model doesn't know this token)
        }

        pos = special_end;
    }

    *num_tokens = token_count;
    return tokens;
}

// Decode token IDs to text
char* tokenizer_decode(tokenizer_t* tok, int* token_ids, int num_tokens) {
    // WordLevel tokenizer: join tokens with spaces (whitespace is implicit)
    if (tok->is_word_level) {
        size_t total_len = 0;
        for (int i = 0; i < num_tokens; i++) {
            const char* token_str = tokenizer_id_to_token(tok, token_ids[i]);
            if (token_str) {
                total_len += strlen(token_str) + 1;  // +1 for space
            }
        }

        char* output = malloc(total_len + 1);
        output[0] = '\0';
        int first = 1;

        for (int i = 0; i < num_tokens; i++) {
            const char* token_str = tokenizer_id_to_token(tok, token_ids[i]);
            if (token_str) {
                // Skip space before special tokens and after special tokens
                if (token_str[0] == '<' && token_str[strlen(token_str)-1] == '>') {
                    strcat(output, token_str);
                    first = 1;  // Next real token shouldn't have leading space
                    continue;
                }
                if (!first) strcat(output, " ");
                strcat(output, token_str);
                first = 0;
            }
        }
        return output;
    }

    // BPE tokenizer: concatenate directly, then clean markers
    // Calculate total length needed
    size_t total_len = 0;
    for (int i = 0; i < num_tokens; i++) {
        const char* token_str = tokenizer_id_to_token(tok, token_ids[i]);
        if (token_str) {
            total_len += strlen(token_str);
        }
    }

    // Allocate output buffer
    char* output = malloc(total_len + 1);
    output[0] = '\0';

    // Concatenate all tokens
    for (int i = 0; i < num_tokens; i++) {
        const char* token_str = tokenizer_id_to_token(tok, token_ids[i]);
        if (token_str) {
            strcat(output, token_str);
        }
    }

    // Clean BPE markers
    char* p = output;
    while (*p) {
        unsigned char c = (unsigned char)*p;
        if (c == 0xC4) {
            unsigned char next = (unsigned char)*(p+1);
            if (next == 0xA0) {
                // Ġ (U+0120) → space
                *p = ' ';
                memmove(p + 1, p + 2, strlen(p + 2) + 1);
            } else if (next == 0x8A) {
                // Ċ (U+010A) → newline
                *p = '\n';
                memmove(p + 1, p + 2, strlen(p + 2) + 1);
            } else if (next >= 0x80 && next < 0xC0) {
                // Other BPE control chars → remove
                memmove(p, p + 2, strlen(p + 2) + 1);
                continue;
            }
        } else if (c == 0xE2 && (unsigned char)*(p+1) == 0x96 && (unsigned char)*(p+2) == 0x81) {
            // ▁ (U+2581) → space (SentencePiece)
            *p = ' ';
            memmove(p + 1, p + 3, strlen(p + 3) + 1);
        }
        p++;
    }

    return output;
}

// Check if token matches any stop token
int tokenizer_is_stop_token(tokenizer_t* tok, int token_id, const char** stop_tokens, int num_stop_tokens) {
    const char* token_str = tokenizer_id_to_token(tok, token_id);
    if (!token_str) return 0;

    for (int i = 0; i < num_stop_tokens; i++) {
        if (strcmp(token_str, stop_tokens[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Get stop token IDs - static buffer, returns up to 16 stop IDs
static int stop_id_buffer[16];

int* tokenizer_get_stop_ids(tokenizer_t* tok, const char** stop_strings, int num_strings, int* num_stop) {
    int count = 0;

    for (int i = 0; i < num_strings && count < 16; i++) {
        int id = tokenizer_token_to_id(tok, stop_strings[i]);
        if (id >= 0) {
            stop_id_buffer[count++] = id;
        }
    }

    *num_stop = count;
    return stop_id_buffer;
}
