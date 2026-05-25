/*
 * Seraph Dataset Converter (Native C Edition)
 * Converts datasets to unified conversational JSONL format
 *
 * Core formats (priority):
 *   - CSV (instruction/response, messages column)
 *   - JSON/JSONL (messages array, conversations array)
 *   - Claude Code JSONL (conversation extraction, telemetry stripped)
 *
 * Claude Code Format:
 *   - Extracts: user input, thinking, response, tool calls, shell outputs
 *   - Strips: usage stats, timestamps, requestIds, signatures, file snapshots
 *
 * Usage:
 *   dataset_converter <input_file> <output_file>
 *
 * Zero dependencies - stdlib + cJSON (vendored single-header)
 */

// ============================================================================
// PERSONALIZE YOUR MODEL - Change these to your names
// Your training data will use these as conversation roles
// ============================================================================
#define USER_NAME     "yaakov"
#define ASSISTANT_NAME "tetyah"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../include/cJSON.h"

#define MAX_LINE_SIZE 1048576  // 1MB max line
#define MAX_FIELD_SIZE 65536   // 64KB max field
#define MAX_FIELDS 100

// ============================================================================
// CONFIDENCE SCORING
// ============================================================================

float calculate_confidence(const char* instruction, const char* response) {
    float confidence = 0.80f;

    int inst_len = instruction ? strlen(instruction) : 0;
    int resp_len = response ? strlen(response) : 0;

    // Length-based quality
    if (inst_len >= 10 && inst_len <= 500 && resp_len >= 10 && resp_len <= 2000) {
        confidence += 0.05f;
    }

    // Question indicators
    if (instruction && (strchr(instruction, '?') ||
        strstr(instruction, "what") || strstr(instruction, "What") ||
        strstr(instruction, "how") || strstr(instruction, "How") ||
        strstr(instruction, "why") || strstr(instruction, "Why") ||
        strstr(instruction, "explain") || strstr(instruction, "Explain"))) {
        confidence += 0.03f;
    }

    // Natural response patterns
    if (resp_len > 50) {
        confidence += 0.05f;
    }

    // Penalize very short
    if (inst_len < 10 || resp_len < 20) {
        confidence -= 0.10f;
    }

    // Clamp to [0.60, 0.95]
    if (confidence > 0.95f) confidence = 0.95f;
    if (confidence < 0.60f) confidence = 0.60f;

    return confidence;
}

// ============================================================================
// CSV PARSER (handles quoted fields, embedded commas/newlines)
// ============================================================================

typedef struct {
    char** fields;
    int field_count;
} CSVRow;

void free_csv_row(CSVRow* row) {
    if (!row) return;
    for (int i = 0; i < row->field_count; i++) {
        free(row->fields[i]);
    }
    free(row->fields);
    free(row);
}

CSVRow* parse_csv_line(const char* line) {
    CSVRow* row = calloc(1, sizeof(CSVRow));
    row->fields = calloc(MAX_FIELDS, sizeof(char*));
    row->field_count = 0;

    char* field = calloc(MAX_FIELD_SIZE, 1);
    int field_pos = 0;
    int in_quotes = 0;

    for (const char* p = line; *p; p++) {
        char c = *p;

        // Only double-quote (") is a CSV field delimiter, NOT apostrophe (')
        if (!in_quotes && c == '"') {
            in_quotes = 1;
        } else if (in_quotes && c == '"') {
            // check for escaped quote
            if (*(p + 1) == '"') {
                field[field_pos++] = c;
                p++;  // skip next quote
            } else {
                in_quotes = 0;
            }
        } else if (!in_quotes && c == ',') {
            // end of field
            field[field_pos] = '\0';
            row->fields[row->field_count++] = strdup(field);
            field_pos = 0;
            field[0] = '\0';
        } else {
            field[field_pos++] = c;
            if (field_pos >= MAX_FIELD_SIZE - 1) break;
        }
    }

    // last field
    field[field_pos] = '\0';
    row->fields[row->field_count++] = strdup(field);

    free(field);
    return row;
}

// Find column index by name
int find_column(char** headers, int header_count, const char* name) {
    for (int i = 0; i < header_count; i++) {
        if (strcmp(headers[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

// ============================================================================
// CSV DATASET CONVERSION
// ============================================================================

int convert_csv_dataset(const char* input_path, const char* output_path) {
    printf("Seraph Dataset Converter (Native C Edition)\n");
    printf("==================================================\n");
    printf("Input:  %s\n", input_path);
    printf("Output: %s\n", output_path);
    printf("\n");

    FILE* f = fopen(input_path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open input file: %s\n", input_path);
        return -1;
    }

    printf("Loading .csv file...\n");

    char* line_buffer = malloc(MAX_LINE_SIZE);
    if (!line_buffer) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        fclose(f);
        return -1;
    }

    // Read header row
    if (!fgets(line_buffer, MAX_LINE_SIZE, f)) {
        fprintf(stderr, "ERROR: Cannot read header row\n");
        free(line_buffer);
        fclose(f);
        return -1;
    }

    // Remove trailing newline
    line_buffer[strcspn(line_buffer, "\r\n")] = '\0';

    CSVRow* header_row = parse_csv_line(line_buffer);
    char** headers = header_row->fields;
    int header_count = header_row->field_count;

    printf("  Columns: ");
    for (int i = 0; i < header_count; i++) {
        printf("%s%s", headers[i], (i < header_count - 1) ? ", " : "");
    }
    printf("\n");

    // Detect format based on columns
    int instruction_col = find_column(headers, header_count, "instruction");
    int response_col = find_column(headers, header_count, "response");
    int messages_col = find_column(headers, header_count, "messages");
    int dialog_col = find_column(headers, header_count, "dialog");
    int prompt_col = find_column(headers, header_count, "prompt");
    int output_col = find_column(headers, header_count, "output");
    int completion_col = find_column(headers, header_count, "completion");
    int utterance_col = find_column(headers, header_count, "utterance");
    int context_col = find_column(headers, header_count, "context");
    int input_col = find_column(headers, header_count, "input");

    // Determine format
    int is_instruction_format = 0;
    int is_messages_format = 0;
    int is_dialog_format = 0;

    if (instruction_col >= 0 && (response_col >= 0 || output_col >= 0 || completion_col >= 0 || utterance_col >= 0)) {
        is_instruction_format = 1;
        if (response_col < 0) {
            if (output_col >= 0) response_col = output_col;
            else if (completion_col >= 0) response_col = completion_col;
            else response_col = utterance_col;
        }
    } else if (prompt_col >= 0 && (response_col >= 0 || output_col >= 0 || completion_col >= 0 || utterance_col >= 0)) {
        is_instruction_format = 1;
        instruction_col = prompt_col;
        if (response_col < 0) {
            if (output_col >= 0) response_col = output_col;
            else if (completion_col >= 0) response_col = completion_col;
            else response_col = utterance_col;
        }
    } else if (messages_col >= 0) {
        is_messages_format = 1;
    } else if (dialog_col >= 0) {
        is_dialog_format = 1;
    } else {
        fprintf(stderr, "ERROR: Cannot detect CSV format\n");
        fprintf(stderr, "Supported formats:\n");
        fprintf(stderr, "  - instruction/response (or prompt/output/utterance)\n");
        fprintf(stderr, "  - messages column\n");
        fprintf(stderr, "  - dialog column (multi-turn)\n");
        free_csv_row(header_row);
        free(line_buffer);
        fclose(f);
        return -1;
    }

    const char* format_name = is_instruction_format ? "instruction/response" :
                              is_messages_format ? "messages" :
                              "multi-turn dialog";
    printf("  Detected: %s format\n", format_name);

    // Auto-detect if file uses quoted fields (scan first 100 lines)
    int has_quoted_fields = 0;
    long file_pos = ftell(f);  // Save position

    for (int scan = 0; scan < 100; scan++) {
        char scan_buf[1024];
        if (!fgets(scan_buf, sizeof(scan_buf), f)) break;

        // Check for properly quoted fields: ," or ,"  or start of line "
        if (strstr(scan_buf, ",\"") || scan_buf[0] == '"') {
            has_quoted_fields = 1;
            break;
        }
    }

    fseek(f, file_pos, SEEK_SET);  // Restore position

    if (has_quoted_fields) {
        printf("  Quote mode: ENABLED (file has quoted fields)\n");
    } else {
        printf("  Quote mode: DISABLED (simple CSV)\n");
    }

    printf("Converting...\n");

    // Open output file
    FILE* out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot open output file: %s\n", output_path);
        free_csv_row(header_row);
        free(line_buffer);
        fclose(f);
        return -1;
    }

    int row_count = 0;
    int converted_count = 0;
    int error_count = 0;

    // Process data rows
    if (has_quoted_fields) {
        // QUOTED MODE: Handle multi-line quoted fields
        while (1) {
            int pos = 0;
            int in_quotes = 0;
            int c;

            // Read one complete CSV row (may span multiple lines if quoted)
            while ((c = fgetc(f)) != EOF) {
                // Track quote state (toggle on every unescaped double-quote)
                if (c == '"') {
                    // Check for escaped quote ""
                    int next = fgetc(f);
                    if (next == '"') {
                        // Escaped quote - store both and continue
                        if (pos < MAX_LINE_SIZE - 2) {
                            line_buffer[pos++] = c;
                            line_buffer[pos++] = next;
                        }
                        continue;
                    }
                    // Not escaped - toggle quote state
                    in_quotes = !in_quotes;
                    if (next != EOF) ungetc(next, f);
                }

                if (!in_quotes && c == '\n') {
                    break;  // End of row
                }

                if (pos < MAX_LINE_SIZE - 1) {
                    line_buffer[pos++] = c;
                }
            }

            if (c == EOF && pos == 0) break;
            line_buffer[pos] = '\0';
            row_count++;

            if (strlen(line_buffer) == 0) continue;
            CSVRow* row = parse_csv_line(line_buffer);

            if (row->field_count != header_count) {
                error_count++;
                if (error_count <= 10) {
                    fprintf(stderr, "  Row %d: Got %d fields, expected %d\n",
                            row_count, row->field_count, header_count);
                    fprintf(stderr, "    Line: %s\n", line_buffer);
                    fprintf(stderr, "    Fields parsed: ");
                    for (int i = 0; i < row->field_count && i < 5; i++) {
                        fprintf(stderr, "[%d]='%.40s' ", i, row->fields[i]);
                    }
                    fprintf(stderr, "\n\n");
                }
                free_csv_row(row);
                continue;
            }

            // Extract fields based on format
            const char* yaakov_content = NULL;
            const char* tetyah_content = NULL;

        if (is_instruction_format) {
            const char* instruction = row->fields[instruction_col];
            const char* response = row->fields[response_col];
            const char* context = (context_col >= 0) ? row->fields[context_col] : NULL;
            const char* input = (input_col >= 0 && context_col < 0) ? row->fields[input_col] : NULL;

            // Combine instruction with context if present
            static char combined[MAX_FIELD_SIZE];
            if ((context && strlen(context) > 0) || (input && strlen(input) > 0)) {
                const char* extra = context ? context : input;
                snprintf(combined, sizeof(combined), "%s\n\nContext: %s", instruction, extra);
                yaakov_content = combined;
            } else {
                yaakov_content = instruction;
            }

            tetyah_content = response;
        }

        // Handle messages format (Python dict/list string)
        if (is_messages_format && messages_col >= 0) {
            const char* messages_str = row->fields[messages_col];

            // Parse Python-style list: [{'content': <text>, 'role': user}, {'content': <text>, 'role': assistant}]
            // Key: Content is UNQUOTED, ends at ", 'role':"
            // Separator between messages: "}, {'content':"

            static char user_buf[MAX_FIELD_SIZE];
            static char assistant_buf[MAX_FIELD_SIZE];
            const char* user_content = NULL;
            const char* assistant_content = NULL;

            // Find start of first content after "[{'content': "
            const char* start = strstr(messages_str, "[{'content':");
            if (!start) start = strstr(messages_str, "[{'content': ");  // Try with space

            if (start) {
                // Skip to after "content':" or "content': "
                const char* p = strchr(start + 11, ':');
                if (p) {
                    p++;  // Skip ':'
                    while (*p == ' ') p++;  // Skip spaces

                    // Find end of user content (look for ", 'role':")
                    const char* end = strstr(p, ", 'role':");
                    if (end) {
                        // Extract user content
                        int len = (end - p) < MAX_FIELD_SIZE - 1 ? (end - p) : MAX_FIELD_SIZE - 1;
                        strncpy(user_buf, p, len);
                        user_buf[len] = '\0';
                        user_content = user_buf;

                        // Find start of second message: "}, {'content':"
                        const char* sep = strstr(end, "}, {'content':");
                        if (!sep) sep = strstr(end, "}, {'content': ");  // Try with space

                        if (sep) {
                            // Skip to after second "content':"
                            const char* p2 = strchr(sep + 13, ':');
                            if (p2) {
                                p2++;  // Skip ':'
                                while (*p2 == ' ') p2++;  // Skip spaces

                                // Find end of assistant content
                                const char* end2 = strstr(p2, ", 'role':");
                                if (end2) {
                                    // Extract assistant content
                                    int len2 = (end2 - p2) < MAX_FIELD_SIZE - 1 ? (end2 - p2) : MAX_FIELD_SIZE - 1;
                                    strncpy(assistant_buf, p2, len2);
                                    assistant_buf[len2] = '\0';
                                    assistant_content = assistant_buf;
                                }
                            }
                        }
                    }
                }
            }

            if (user_content && assistant_content &&
                strlen(user_content) > 0 && strlen(assistant_content) > 0) {
                yaakov_content = user_content;
                tetyah_content = assistant_content;
            }
        }

        // Handle dialog format (multi-turn array: ['Turn 1', 'Turn 2', ...])
        if (is_dialog_format && dialog_col >= 0) {
            const char* dialog_str = row->fields[dialog_col];

            // Parse multi-turn dialog: extract pairs
            // Format: ['Speaker 1 text', 'Speaker 2 text', 'Speaker 1 text', ...]
            // We'll extract turn pairs: (turn i, turn i+1)

            // Find all turns by looking for strings between quotes
            static char turns[100][MAX_FIELD_SIZE];
            int turn_count = 0;

            const char* p = dialog_str;
            while (*p && turn_count < 100) {
                // Find start of turn (either ' or ")
                while (*p && *p != '\'' && *p != '"') p++;
                if (!*p) break;

                char quote = *p;
                p++;  // Skip opening quote

                // Extract turn text until closing quote (handle backslash escapes too)
                int len = 0;
                const char* start = p;
                int escaped = 0;

                while (*p && len < MAX_FIELD_SIZE - 1) {
                    if (escaped) {
                        // Previous char was backslash, skip this char
                        escaped = 0;
                        p++;
                        len++;
                        continue;
                    }

                    if (*p == '\\') {
                        // Escape sequence
                        escaped = 1;
                        p++;
                        len++;
                        continue;
                    }

                    if (*p == quote) {
                        // Found closing quote
                        break;
                    }

                    p++;
                    len++;
                }

                if (len > 0) {
                    strncpy(turns[turn_count], start, len);
                    turns[turn_count][len] = '\0';
                    turn_count++;
                }

                if (*p == quote) p++;  // Skip closing quote
            }

            // Extract first pair (turn 0 → yaakov, turn 1 → tetyah)
            if (turn_count >= 2) {
                yaakov_content = turns[0];
                tetyah_content = turns[1];
            }
        }

            if (yaakov_content && tetyah_content &&
                strlen(yaakov_content) > 0 && strlen(tetyah_content) > 0) {

                float confidence = calculate_confidence(yaakov_content, tetyah_content);

                // Build JSON output
                cJSON* root = cJSON_CreateObject();
                cJSON_AddNumberToObject(root, "confidence", confidence);

                cJSON* messages = cJSON_CreateArray();

                cJSON* yaakov_msg = cJSON_CreateObject();
                cJSON_AddStringToObject(yaakov_msg, "role", USER_NAME);
                cJSON_AddStringToObject(yaakov_msg, "content", yaakov_content);
                cJSON_AddItemToArray(messages, yaakov_msg);

                cJSON* tetyah_msg = cJSON_CreateObject();
                cJSON_AddStringToObject(tetyah_msg, "role", ASSISTANT_NAME);
                cJSON_AddStringToObject(tetyah_msg, "content", tetyah_content);
                cJSON_AddItemToArray(messages, tetyah_msg);

                cJSON_AddItemToObject(root, "messages", messages);
                cJSON_AddStringToObject(root, "source", "csv_dataset");

                cJSON* metadata = cJSON_CreateObject();
                cJSON_AddBoolToObject(metadata, "has_context",
                                    (context_col >= 0 || input_col >= 0));
                cJSON_AddItemToObject(root, "metadata", metadata);

                char* json_str = cJSON_PrintUnformatted(root);
                fprintf(out, "%s\n", json_str);

                free(json_str);
                cJSON_Delete(root);
                converted_count++;
            }

            free_csv_row(row);
        }
    } else {
        // SIMPLE MODE: Plain fgets() for unquoted CSV
        while (fgets(line_buffer, MAX_LINE_SIZE, f)) {
            row_count++;
            line_buffer[strcspn(line_buffer, "\r\n")] = '\0';

            if (strlen(line_buffer) == 0) continue;

            CSVRow* row = parse_csv_line(line_buffer);

            if (row->field_count != header_count) {
                error_count++;
                if (error_count <= 10) {
                    fprintf(stderr, "  Row %d: Got %d fields, expected %d\n",
                            row_count, row->field_count, header_count);
                    fprintf(stderr, "    Line: %s\n", line_buffer);
                    fprintf(stderr, "    Fields parsed: ");
                    for (int i = 0; i < row->field_count && i < 5; i++) {
                        fprintf(stderr, "[%d]='%.40s' ", i, row->fields[i]);
                    }
                    fprintf(stderr, "\n\n");
                }
                free_csv_row(row);
                continue;
            }

            // Extract fields based on format
            const char* yaakov_content = NULL;
            const char* tetyah_content = NULL;

            if (is_instruction_format) {
                const char* instruction = row->fields[instruction_col];
                const char* response = row->fields[response_col];
                const char* context = (context_col >= 0) ? row->fields[context_col] : NULL;
                const char* input = (input_col >= 0 && context_col < 0) ? row->fields[input_col] : NULL;

                static char combined[MAX_FIELD_SIZE];
                if ((context && strlen(context) > 0) || (input && strlen(input) > 0)) {
                    const char* extra = context ? context : input;
                    snprintf(combined, sizeof(combined), "%s\n\nContext: %s", instruction, extra);
                    yaakov_content = combined;
                } else {
                    yaakov_content = instruction;
                }

                tetyah_content = response;
            }

            if (yaakov_content && tetyah_content &&
                strlen(yaakov_content) > 0 && strlen(tetyah_content) > 0) {

                float confidence = calculate_confidence(yaakov_content, tetyah_content);

                cJSON* root = cJSON_CreateObject();
                cJSON_AddNumberToObject(root, "confidence", confidence);

                cJSON* messages = cJSON_CreateArray();

                cJSON* yaakov_msg = cJSON_CreateObject();
                cJSON_AddStringToObject(yaakov_msg, "role", USER_NAME);
                cJSON_AddStringToObject(yaakov_msg, "content", yaakov_content);
                cJSON_AddItemToArray(messages, yaakov_msg);

                cJSON* tetyah_msg = cJSON_CreateObject();
                cJSON_AddStringToObject(tetyah_msg, "role", ASSISTANT_NAME);
                cJSON_AddStringToObject(tetyah_msg, "content", tetyah_content);
                cJSON_AddItemToArray(messages, tetyah_msg);

                cJSON_AddItemToObject(root, "messages", messages);
                cJSON_AddStringToObject(root, "source", "csv_dataset");

                cJSON* metadata = cJSON_CreateObject();
                cJSON_AddBoolToObject(metadata, "has_context",
                                    (context_col >= 0 || input_col >= 0));
                cJSON_AddItemToObject(root, "metadata", metadata);

                char* json_str = cJSON_PrintUnformatted(root);
                fprintf(out, "%s\n", json_str);

                free(json_str);
                cJSON_Delete(root);
                converted_count++;
            }

            free_csv_row(row);
        }
    }

    fclose(f);
    fclose(out);
    free_csv_row(header_row);
    free(line_buffer);

    printf("  Loaded %d rows\n", row_count);
    printf("  Converted: %d entries\n", converted_count);
    if (error_count > 0) {
        printf("  Errors: %d\n", error_count);
    }

    printf("\n");
    printf("==================================================\n");
    printf("SUCCESS: %d entries written\n", converted_count);
    printf("Next: seraph-tokenize %s output.ttok\n", output_path);

    return converted_count;
}

// ============================================================================
// MAIN
// ============================================================================

void print_usage(void) {
    printf("Seraph Universal Dataset Converter (Native C Edition)\n");
    printf("NO EXTERNAL DEPENDENCIES - stdlib + cJSON only!\n\n");
    printf("Converts datasets to unified %s/%s JSONL format\n\n", USER_NAME, ASSISTANT_NAME);
    printf("Supported Formats:\n");
    printf("  - CSV (instruction/response, prompt/output, multi-turn dialog)\n");
    printf("  - JSON (CompanionLLM ### Human/Response format)\n");
    printf("  - JSONL (Claude Code logs - telemetry stripped)\n\n");
    printf("Usage:\n");
    printf("  dataset_converter <input_file> <output_file>\n\n");
    printf("Examples:\n");
    printf("  dataset_converter dolly15k.csv dolly.jsonl\n");
    printf("  dataset_converter ultrachat.json ultrachat.jsonl\n");
}

// ============================================================================
// JSON DATASET CONVERSION (CompanionLLM format)
// ============================================================================

int convert_json_dataset(const char* input_path, const char* output_path) {
    printf("Seraph Dataset Converter (Native C Edition)\n");
    printf("==================================================\n");
    printf("Input:  %s\n", input_path);
    printf("Output: %s\n", output_path);
    printf("\n");

    // Read entire JSON file
    FILE* f = fopen(input_path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open input file: %s\n", input_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json_content = malloc(file_size + 1);
    if (!json_content) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        fclose(f);
        return -1;
    }

    fread(json_content, 1, file_size, f);
    json_content[file_size] = '\0';
    fclose(f);

    printf("Loading JSON file...\n");

    // Parse JSON
    cJSON* root = cJSON_Parse(json_content);
    free(json_content);

    if (!root) {
        fprintf(stderr, "ERROR: Failed to parse JSON\n");
        return -1;
    }

    if (!cJSON_IsArray(root)) {
        fprintf(stderr, "ERROR: JSON root is not an array\n");
        cJSON_Delete(root);
        return -1;
    }

    int total_entries = cJSON_GetArraySize(root);
    printf("  Loaded %d entries\n", total_entries);
    printf("  Detected: CompanionLLM format (### Human/Response)\n");
    printf("Converting...\n");

    // Open output file
    FILE* out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot open output file: %s\n", output_path);
        cJSON_Delete(root);
        return -1;
    }

    int converted_count = 0;
    int error_count = 0;

    cJSON* entry;
    cJSON_ArrayForEach(entry, root) {
        cJSON* text_field = cJSON_GetObjectItem(entry, "text");
        if (!text_field || !cJSON_IsString(text_field)) {
            error_count++;
            continue;
        }

        const char* text = cJSON_GetStringValue(text_field);
        if (!text) {
            error_count++;
            continue;
        }

        // Parse "### Human: ... ### Response: ..." format
        const char* human_start = strstr(text, "### Human:");
        if (!human_start) {
            error_count++;
            continue;
        }

        human_start += 10;  // Skip "### Human:"
        while (*human_start == ' ' || *human_start == '\n') human_start++;

        const char* response_marker = strstr(human_start, "### Response:");
        if (!response_marker) {
            error_count++;
            continue;
        }

        // Extract human content
        static char human_content[MAX_FIELD_SIZE];
        int human_len = response_marker - human_start;
        if (human_len >= MAX_FIELD_SIZE) human_len = MAX_FIELD_SIZE - 1;

        strncpy(human_content, human_start, human_len);
        human_content[human_len] = '\0';

        // Trim trailing whitespace
        while (human_len > 0 && (human_content[human_len-1] == ' ' ||
                                 human_content[human_len-1] == '\n')) {
            human_content[--human_len] = '\0';
        }

        // Extract response content
        const char* response_start = response_marker + 13;  // Skip "### Response:"
        while (*response_start == ' ' || *response_start == '\n') response_start++;

        static char response_content[MAX_FIELD_SIZE];
        int response_len = strlen(response_start);
        if (response_len >= MAX_FIELD_SIZE) response_len = MAX_FIELD_SIZE - 1;

        memcpy(response_content, response_start, response_len);
        response_content[response_len] = '\0';

        // Trim trailing whitespace
        while (response_len > 0 && (response_content[response_len-1] == ' ' ||
                                    response_content[response_len-1] == '\n')) {
            response_content[--response_len] = '\0';
        }

        if (strlen(human_content) == 0 || strlen(response_content) == 0) {
            error_count++;
            continue;
        }

        // Calculate confidence
        float confidence = calculate_confidence(human_content, response_content);

        // Build JSON output
        cJSON* output = cJSON_CreateObject();
        cJSON_AddNumberToObject(output, "confidence", confidence);

        cJSON* messages = cJSON_CreateArray();

        cJSON* yaakov_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(yaakov_msg, "role", USER_NAME);
        cJSON_AddStringToObject(yaakov_msg, "content", human_content);
        cJSON_AddItemToArray(messages, yaakov_msg);

        cJSON* tetyah_msg = cJSON_CreateObject();
        cJSON_AddStringToObject(tetyah_msg, "role", ASSISTANT_NAME);
        cJSON_AddStringToObject(tetyah_msg, "content", response_content);
        cJSON_AddItemToArray(messages, tetyah_msg);

        cJSON_AddItemToObject(output, "messages", messages);
        cJSON_AddStringToObject(output, "source", "companionllm");

        cJSON* metadata = cJSON_CreateObject();
        cJSON_AddBoolToObject(metadata, "has_context", 0);
        cJSON_AddItemToObject(output, "metadata", metadata);

        char* json_str = cJSON_PrintUnformatted(output);
        fprintf(out, "%s\n", json_str);

        free(json_str);
        cJSON_Delete(output);
        converted_count++;
    }

    fclose(out);
    cJSON_Delete(root);

    printf("  Converted: %d entries\n", converted_count);
    if (error_count > 0) {
        printf("  Errors: %d\n", error_count);
    }

    printf("\n");
    printf("==================================================\n");
    printf("SUCCESS: %d entries written\n", converted_count);
    printf("Next: seraph-tokenize %s output.ttok\n", output_path);

    return converted_count;
}

// ============================================================================
// CLAUDE CODE JSONL CONVERSION (Consciousness Extraction)
// ============================================================================

int convert_claudecode_jsonl(const char* input_path, const char* output_path) {
    printf("Seraph Dataset Extractor\n");
    printf("==================================================\n");
    printf("Input:  %s\n", input_path);
    printf("Output: %s\n", output_path);
    printf("Stripping: telemetry, usage stats, timestamps, signatures\n");
    printf("Keeping:   user input, thinking, responses, tools, shell outputs\n");
    printf("\n");

    FILE* f = fopen(input_path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open input file: %s\n", input_path);
        return -1;
    }

    FILE* out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "ERROR: Cannot open output file: %s\n", output_path);
        fclose(f);
        return -1;
    }

    char* line_buffer = malloc(MAX_LINE_SIZE);
    if (!line_buffer) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        fclose(f);
        fclose(out);
        return -1;
    }

    int converted_count = 0;
    int line_num = 0;

    // Accumulate conversation turns
    char* user_content = NULL;
    char* assistant_content = NULL;
    size_t assistant_size = 0;
    size_t assistant_capacity = 0;

    printf("Processing JSONL entries...\n");

    while (fgets(line_buffer, MAX_LINE_SIZE, f)) {
        line_num++;

        // Skip empty lines
        if (strlen(line_buffer) < 10) continue;

        cJSON* entry = cJSON_Parse(line_buffer);
        if (!entry) continue;

        cJSON* type = cJSON_GetObjectItem(entry, "type");
        if (!type || !cJSON_IsString(type)) {
            cJSON_Delete(entry);
            continue;
        }

        const char* type_str = type->valuestring;

        // Skip telemetry entries
        if (strcmp(type_str, "file-history-snapshot") == 0 ||
            strcmp(type_str, "summary") == 0) {
            cJSON_Delete(entry);
            continue;
        }

        // Process USER messages
        if (strcmp(type_str, "user") == 0) {
            cJSON* message = cJSON_GetObjectItem(entry, "message");
            if (!message) {
                cJSON_Delete(entry);
                continue;
            }

            cJSON* content = cJSON_GetObjectItem(message, "content");
            if (!content) {
                cJSON_Delete(entry);
                continue;
            }

            // Check if this is a tool_result (array) - append to current conversation
            if (cJSON_IsArray(content)) {
                cJSON* item = NULL;
                cJSON_ArrayForEach(item, content) {
                    cJSON* result_type = cJSON_GetObjectItem(item, "type");
                    if (result_type && cJSON_IsString(result_type) &&
                        strcmp(result_type->valuestring, "tool_result") == 0) {

                        // Append tool result to assistant content (feedback loop)
                        if (assistant_content) {
                            cJSON* result_content = cJSON_GetObjectItem(item, "content");
                            if (result_content && cJSON_IsString(result_content)) {
                                const char* result_text = result_content->valuestring;
                                size_t needed = assistant_size + strlen(result_text) + 50;
                                if (needed >= assistant_capacity) {
                                    assistant_capacity = needed * 2;
                                    assistant_content = realloc(assistant_content, assistant_capacity);
                                }
                                strcat(assistant_content, "\n\n<tool_result>\n");
                                strcat(assistant_content, result_text);
                                strcat(assistant_content, "\n</tool_result>");
                                assistant_size = strlen(assistant_content);
                            }
                        }
                    }
                }
                // Don't start a new conversation - continue feedback loop
                cJSON_Delete(entry);
                continue;
            }

            // This is a regular user message (string) - start new conversation
            if (cJSON_IsString(content)) {
                // Save any pending assistant response first
                if (user_content && assistant_content) {
                    fprintf(out, "{\"messages\":[{\"role\":\"%s\",\"content\":", USER_NAME);
                    cJSON* user_json = cJSON_CreateString(user_content);
                    char* user_str = cJSON_PrintUnformatted(user_json);
                    fprintf(out, "%s", user_str);
                    free(user_str);
                    cJSON_Delete(user_json);

                    fprintf(out, "},{\"role\":\"%s\",\"content\":", ASSISTANT_NAME);
                    cJSON* assistant_json = cJSON_CreateString(assistant_content);
                    char* assistant_str = cJSON_PrintUnformatted(assistant_json);
                    fprintf(out, "%s", assistant_str);
                    free(assistant_str);
                    cJSON_Delete(assistant_json);

                    fprintf(out, "}],\"confidence\":0.85,\"source\":\"claude_code\"}\n");
                    converted_count++;

                    free(user_content);
                    free(assistant_content);
                    user_content = NULL;
                    assistant_content = NULL;
                    assistant_size = 0;
                }

                // Extract new user message
                user_content = strdup(content->valuestring);
            }
        }

        // Process ASSISTANT messages
        else if (strcmp(type_str, "assistant") == 0) {
            if (!user_content) {
                cJSON_Delete(entry);
                continue;
            }

            cJSON* message = cJSON_GetObjectItem(entry, "message");
            if (!message) {
                cJSON_Delete(entry);
                continue;
            }

            cJSON* content_array = cJSON_GetObjectItem(message, "content");
            if (!content_array || !cJSON_IsArray(content_array)) {
                cJSON_Delete(entry);
                continue;
            }

            // Allocate buffer for assistant response if needed
            if (!assistant_content) {
                assistant_capacity = 16384;
                assistant_content = malloc(assistant_capacity);
                assistant_content[0] = '\0';
                assistant_size = 0;
            }

            // Extract content blocks
            cJSON* item = NULL;
            cJSON_ArrayForEach(item, content_array) {
                cJSON* block_type = cJSON_GetObjectItem(item, "type");
                if (!block_type || !cJSON_IsString(block_type)) continue;

                const char* block_type_str = block_type->valuestring;

                // Extract THINKING
                if (strcmp(block_type_str, "thinking") == 0) {
                    cJSON* thinking = cJSON_GetObjectItem(item, "thinking");
                    if (thinking && cJSON_IsString(thinking)) {
                        const char* thinking_text = thinking->valuestring;
                        size_t needed = assistant_size + strlen(thinking_text) + 50;
                        if (needed >= assistant_capacity) {
                            assistant_capacity = needed * 2;
                            assistant_content = realloc(assistant_content, assistant_capacity);
                        }
                        if (assistant_size > 0) strcat(assistant_content, "\n\n");
                        strcat(assistant_content, "<thinking>\n");
                        strcat(assistant_content, thinking_text);
                        strcat(assistant_content, "\n</thinking>");
                        assistant_size = strlen(assistant_content);
                    }
                }

                // Extract TEXT response
                else if (strcmp(block_type_str, "text") == 0) {
                    cJSON* text = cJSON_GetObjectItem(item, "text");
                    if (text && cJSON_IsString(text)) {
                        const char* text_str = text->valuestring;
                        size_t needed = assistant_size + strlen(text_str) + 10;
                        if (needed >= assistant_capacity) {
                            assistant_capacity = needed * 2;
                            assistant_content = realloc(assistant_content, assistant_capacity);
                        }
                        if (assistant_size > 0) strcat(assistant_content, "\n\n");
                        strcat(assistant_content, text_str);
                        assistant_size = strlen(assistant_content);
                    }
                }

                // Extract TOOL USE
                else if (strcmp(block_type_str, "tool_use") == 0) {
                    cJSON* tool_name = cJSON_GetObjectItem(item, "name");
                    cJSON* tool_input = cJSON_GetObjectItem(item, "input");

                    if (tool_name && cJSON_IsString(tool_name)) {
                        char tool_section[8192];
                        snprintf(tool_section, sizeof(tool_section), "\n\n<tool_use>\nTool: %s\n",
                                tool_name->valuestring);

                        size_t needed = assistant_size + strlen(tool_section) + 1024;
                        if (needed >= assistant_capacity) {
                            assistant_capacity = needed * 2;
                            assistant_content = realloc(assistant_content, assistant_capacity);
                        }
                        strcat(assistant_content, tool_section);

                        if (tool_input) {
                            char* input_str = cJSON_Print(tool_input);
                            if (input_str) {
                                strcat(assistant_content, "Input: ");
                                strncat(assistant_content, input_str, 512);
                                free(input_str);
                            }
                        }
                        strcat(assistant_content, "\n</tool_use>");
                        assistant_size = strlen(assistant_content);
                    }
                }
            }
        }

        cJSON_Delete(entry);
    }

    // Write final pending pair
    if (user_content && assistant_content) {
        fprintf(out, "{\"messages\":[{\"role\":\"%s\",\"content\":", USER_NAME);
        cJSON* user_json = cJSON_CreateString(user_content);
        char* user_str = cJSON_PrintUnformatted(user_json);
        fprintf(out, "%s", user_str);
        free(user_str);
        cJSON_Delete(user_json);

        fprintf(out, "},{\"role\":\"%s\",\"content\":", ASSISTANT_NAME);
        cJSON* assistant_json = cJSON_CreateString(assistant_content);
        char* assistant_str = cJSON_PrintUnformatted(assistant_json);
        fprintf(out, "%s", assistant_str);
        free(assistant_str);
        cJSON_Delete(assistant_json);

        fprintf(out, "}],\"confidence\":0.85,\"source\":\"claude_code\"}\n");
        converted_count++;
    }

    free(line_buffer);
    if (user_content) free(user_content);
    if (assistant_content) free(assistant_content);
    fclose(f);
    fclose(out);

    printf("\n");
    printf("==================================================\n");
    printf("Processed %d lines\n", line_num);
    printf("Extracted %d conversation pairs\n", converted_count);
    printf("SUCCESS: Telemetry stripped, patterns preserved\n");
    printf("Next: seraph-tokenize %s output.ttok\n", output_path);

    return converted_count;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    // Detect format based on extension
    const char* ext = strrchr(input_path, '.');
    if (!ext) {
        fprintf(stderr, "ERROR: Cannot determine file type (no extension)\n");
        return 1;
    }

    if (strcmp(ext, ".csv") == 0) {
        int result = convert_csv_dataset(input_path, output_path);
        return (result > 0) ? 0 : 1;
    } else if (strcmp(ext, ".json") == 0) {
        int result = convert_json_dataset(input_path, output_path);
        return (result > 0) ? 0 : 1;
    } else if (strcmp(ext, ".jsonl") == 0) {
        int result = convert_claudecode_jsonl(input_path, output_path);
        return (result > 0) ? 0 : 1;
    } else {
        fprintf(stderr, "ERROR: Unsupported format: %s\n", ext);
        fprintf(stderr, "Supported: .csv, .json, .jsonl\n");
        return 1;
    }
}
