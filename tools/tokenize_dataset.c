// tokenize_dataset.c - Convert JSONL chat datasets to tokenized binary format
// Part of TETYAH-PRIME's native training and inference engine
//
// Usage: seraph-tokenize <model_dir> <input.jsonl> <output.ttok>
//
// Input format (JSONL - one JSON object per line):
//   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
//
// Output format (.ttok binary):
//   Header: magic (4 bytes) + version (4) + num_samples (4) + vocab_size (4)
//   Per sample: num_tokens (4 bytes) + token_ids (num_tokens * 4 bytes)

#include "../include/tokenizer.h"
#include "../include/model_config.h"
#include "../include/cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TTOK_MAGIC 0x544F4B54  // "TTOK"
#define TTOK_VERSION 1

// Map Seraph roles to ChatML roles
static const char* map_role(const char* role) {
    if (strcmp(role, "yaakov") == 0 || strcmp(role, "user") == 0 || strcmp(role, "human") == 0) {
        return "user";
    } else if (strcmp(role, "tetyah") == 0 || strcmp(role, "assistant") == 0 || strcmp(role, "bot") == 0) {
        return "assistant";
    } else if (strcmp(role, "system") == 0) {
        return "system";
    }
    return role;  // Pass through unknown roles
}

// Format a single conversation to ChatML format
static char* format_conversation_chatml(cJSON* messages) {
    size_t total_len = 4096;
    char* result = malloc(total_len);
    result[0] = '\0';

    cJSON* msg;
    cJSON_ArrayForEach(msg, messages) {
        cJSON* role = cJSON_GetObjectItem(msg, "role");
        cJSON* content = cJSON_GetObjectItem(msg, "content");

        if (!role || !content || !cJSON_IsString(role) || !cJSON_IsString(content)) {
            continue;
        }

        // Grow buffer if needed
        size_t needed = strlen(result) + strlen(content->valuestring) + 100;
        if (needed > total_len) {
            total_len = needed * 2;
            result = realloc(result, total_len);
        }

        // ChatML format with role mapping (yaakov->user, tetyah->assistant)
        strcat(result, "<|im_start|>");
        strcat(result, map_role(role->valuestring));
        strcat(result, "\n");
        strcat(result, content->valuestring);
        strcat(result, "<|im_end|>\n");
    }

    return result;
}

int main(int argc, char** argv) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Seraph Dataset Tokenizer\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    if (argc < 4) {
        printf("Usage: %s <model_dir> <input.jsonl> <output.ttok>\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s models/qwen3-8b dataset.jsonl train.ttok\n", argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];

    // Load tokenizer
    printf("[1/3] Loading tokenizer from %s...\n", model_dir);
    tokenizer_t* tokenizer = tokenizer_load(model_dir);
    if (!tokenizer) {
        fprintf(stderr, "ERROR: Failed to load tokenizer\n");
        return 1;
    }
    printf("  Vocab size: %d\n\n", tokenizer->vocab_size);

    // Open input file
    printf("[2/3] Processing %s...\n", input_path);
    FILE* fin = fopen(input_path, "r");
    if (!fin) {
        fprintf(stderr, "ERROR: Cannot open %s\n", input_path);
        tokenizer_free(tokenizer);
        return 1;
    }

    // Open output file
    FILE* fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "ERROR: Cannot create %s\n", output_path);
        fclose(fin);
        tokenizer_free(tokenizer);
        return 1;
    }

    // Write header placeholder (will update num_samples later)
    uint32_t header[4] = {TTOK_MAGIC, TTOK_VERSION, 0, tokenizer->vocab_size};
    fwrite(header, sizeof(uint32_t), 4, fout);

    // Process each line
    char line[1024 * 1024];  // 1MB max line
    int num_samples = 0;
    long total_tokens = 0;

    while (fgets(line, sizeof(line), fin)) {
        // Skip empty lines
        if (strlen(line) < 2) continue;

        // Parse JSON
        cJSON* root = cJSON_Parse(line);
        if (!root) {
            fprintf(stderr, "  Warning: Failed to parse line %d\n", num_samples + 1);
            continue;
        }

        // Get messages array
        cJSON* messages = cJSON_GetObjectItem(root, "messages");
        if (!messages || !cJSON_IsArray(messages)) {
            cJSON_Delete(root);
            continue;
        }

        // Format to ChatML
        char* formatted = format_conversation_chatml(messages);
        cJSON_Delete(root);

        // Tokenize
        int num_tokens;
        int* tokens = tokenizer_encode(tokenizer, formatted, &num_tokens);
        free(formatted);

        if (num_tokens > 0) {
            // Write sample: num_tokens + token_ids
            uint32_t n = num_tokens;
            fwrite(&n, sizeof(uint32_t), 1, fout);
            fwrite(tokens, sizeof(int), num_tokens, fout);

            num_samples++;
            total_tokens += num_tokens;

            if (num_samples % 1000 == 0) {
                printf("  Processed %d samples (%ld tokens)...\n", num_samples, total_tokens);
            }
        }

        free(tokens);
    }

    fclose(fin);

    // Update header with final sample count
    fseek(fout, 2 * sizeof(uint32_t), SEEK_SET);
    uint32_t final_count = num_samples;
    fwrite(&final_count, sizeof(uint32_t), 1, fout);
    fclose(fout);

    printf("\n[3/3] Complete!\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Samples:      %d\n", num_samples);
    printf("  Total tokens: %ld\n", total_tokens);
    printf("  Avg tokens:   %.1f per sample\n", (float)total_tokens / num_samples);
    printf("  Output:       %s\n", output_path);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    tokenizer_free(tokenizer);
    return 0;
}
