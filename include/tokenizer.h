// tokenizer.h - BPE and WordLevel tokenizer
// Byte-Pair Encoding with merge rules and word-level whitespace tokenization
// Part of TETYAH-PRIME's native training and inference engine

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

// BPE merge rule
typedef struct {
    char* pair;     // e.g., "th" or "Ġthe"
    int rank;       // Merge priority (lower = merge first)
} bpe_merge_t;

// Hash table entry for fast token→id lookup
#define VOCAB_HASH_SIZE 262144  // Power of 2, ~2x vocab size
typedef struct vocab_hash_entry {
    char* token;
    int id;
    struct vocab_hash_entry* next;  // Chain for collisions
} vocab_hash_entry_t;

// Tokenizer state
typedef struct {
    // Vocabulary
    char** vocab;           // Token ID → String mapping [vocab_size]
    int vocab_size;         // 151,643 for Qwen

    // Reverse vocab hash table (String → Token ID) - O(1) lookup!
    vocab_hash_entry_t** vocab_hash;

    // BPE merges
    bpe_merge_t* merges;    // Merge rules [num_merges]
    int num_merges;         // 151,387 for Qwen

    // Special tokens
    int bos_token_id;       // Beginning of sequence
    int eos_token_id;       // End of sequence
    int pad_token_id;       // Padding token

    // Chat stop tokens (ChatML, Llama, etc.)
    int im_end_token_id;    // <|im_end|> for ChatML
    int im_start_token_id;  // <|im_start|> for ChatML
    int eot_token_id;       // <|eot_id|> for Llama3

    // Tokenizer type
    int is_word_level;      // 1 = WordLevel (join with spaces), 0 = BPE (use markers)
} tokenizer_t;

// Load tokenizer from model directory
// Expects: vocab.json, merges.txt in the directory
tokenizer_t* tokenizer_load(const char* model_dir);

// Free tokenizer resources
void tokenizer_free(tokenizer_t* tok);

// Encode text to token IDs
// Returns: array of token IDs, sets *num_tokens to length
int* tokenizer_encode(tokenizer_t* tok, const char* text, int* num_tokens);

// Decode token IDs to text
// Returns: malloc'd string (caller must free)
char* tokenizer_decode(tokenizer_t* tok, int* token_ids, int num_tokens);

// Get token string by ID (raw - may contain BPE markers)
const char* tokenizer_id_to_token(tokenizer_t* tok, int token_id);

// Get token string by ID (BPE markers converted to spaces/newlines)
// Returns static buffer - use immediately or copy
const char* tokenizer_id_to_token_clean(tokenizer_t* tok, int token_id);

// Get token ID by string 
int tokenizer_token_to_id(tokenizer_t* tok, const char* token);

// Check if token is a stop/end token (for generation termination)
// Returns 1 if token should stop generation, 0 otherwise
int tokenizer_is_stop_token(tokenizer_t* tok, int token_id, const char** stop_tokens, int num_stop_tokens);

// Get stop token IDs for a chat format
// Returns array of token IDs (caller should not free), sets *num_stop to count
int* tokenizer_get_stop_ids(tokenizer_t* tok, const char** stop_strings, int num_strings, int* num_stop);

#endif // TOKENIZER_H
