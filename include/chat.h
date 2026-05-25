// chat.h - Chat interface and response formatting
// Role-aware message templating with security-aware source tags
// Part of TETYAH-PRIME's native training and inference engine

#ifndef TETYAH_CHAT_H
#define TETYAH_CHAT_H

#include "transformer.h"
#include "tokenizer.h"
#include "model_config.h"

// Source/role types for security-aware prompting
typedef enum {
    ROLE_USER,           // Standard user input
    ROLE_HOST_USER,      // Local host user (higher trust)
    ROLE_ADMIN,          // Security/admin context
    ROLE_SYSTEM,         // System/kernel messages
    ROLE_ASSISTANT       // TETYAH-PRIME responses
} tetyah_role_t;

// Chat message
typedef struct {
    tetyah_role_t role;
    char* content;
} tetyah_message_t;

// Chat context
typedef struct {
    tetyah_message_t* messages;
    int num_messages;
    int capacity;
    char* system_prompt;     // System prompt (optional)
} tetyah_chat_t;

// Get role name string
const char* tetyah_role_name(tetyah_role_t role);

// Create/free chat context
tetyah_chat_t* tetyah_chat_new(void);
void tetyah_chat_free(tetyah_chat_t* chat);

// Add message to chat
void tetyah_chat_add(tetyah_chat_t* chat, tetyah_role_t role, const char* content);

// Set system prompt
void tetyah_chat_set_system(tetyah_chat_t* chat, const char* system_prompt);

// Format chat into prompt string (TETYAH format)
char* tetyah_chat_format(tetyah_chat_t* chat);

// Format chat using model's native template (ChatML, Llama, etc.)
char* tetyah_chat_format_native(tetyah_chat_t* chat, const model_config_t* config);

// High-level: chat and get response
char* tetyah_chat_respond(
    seraph_model_t* model,
    const model_config_t* config,
    tokenizer_t* tokenizer,
    tetyah_chat_t* chat,
    int max_tokens,
    float temperature
);

#endif // TETYAH_CHAT_H
