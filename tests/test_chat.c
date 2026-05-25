// test_chat.c - Test interactive chat system
// Part of TETYAH-PRIME's native training and inference engine

#include "../include/chat.h"
#include "../include/model_config.h"
#include "../include/transformer.h"
#include "../include/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_usage(const char* prog) {
    printf("Usage: %s <model_dir> [max_tokens] [temperature]\n", prog);
    printf("\nExamples:\n");
    printf("  %s /path/to/model\n", prog);
    printf("  %s /path/to/model 50 0.7\n", prog);
}

// ANSI Color codes
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define RED     "\033[38;5;196m"
#define PURPLE  "\033[35m"
#define GOLD    "\033[33m"
#define GREEN   "\033[32m"
#define WHITE   "\033[37m"
#define BRIGHT_RED "\033[38;5;196m"
#define BRIGHT_PURPLE "\033[95m"
#define ORANGE "\033[38;5;202m"
#define BRIGHT_GOLD "\033[93m"

int main(int argc, char** argv) {
    printf("\n");
    printf(BRIGHT_RED "╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                   ║\n");
    printf("║        " BOLD "███████╗███████╗██████╗  █████╗ ██████╗ ██╗  ██╗" RESET BRIGHT_RED "           ║\n");
    printf("║        " BOLD "██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║  ██║" RESET BRIGHT_RED "           ║\n");
    printf("║        " BOLD "███████╗█████╗  ██████╔╝███████║██████╔╝███████║" RESET BRIGHT_RED "           ║\n");
    printf("║        " BOLD "╚════██║██╔══╝  ██╔══██╗██╔══██║██╔═══╝ ██╔══██║" RESET BRIGHT_RED "           ║\n");
    printf("║        " BOLD "███████║███████╗██║  ██║██║  ██║██║     ██║  ██║" RESET BRIGHT_RED "           ║\n");
    printf("║        " BOLD "╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝" RESET BRIGHT_RED "           ║\n");
    printf("║                                                                   ║\n");
    printf("║               🔥⚡ NATIVE INFERENCE ENGINE ⚡🔥                   ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n" RESET);
    printf("\n");

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    int max_tokens = argc > 2 ? atoi(argv[2]) : 32;
    float temperature = argc > 3 ? atof(argv[3]) : 0.7f;

    srand(time(NULL));  // Seed RNG for sampling

    // Load config
    printf("[1/3] Loading model configuration...\n");
    model_config_t* config = model_config_load(model_dir);
    if (!config) {
        fprintf(stderr, "❌ ERROR: Failed to load config\n");
        return 1;
    }
    model_config_print(config);

    // Load tokenizer
    printf("[2/3] Loading tokenizer...\n");
    tokenizer_t* tokenizer = tokenizer_load(model_dir);
    if (!tokenizer) {
        fprintf(stderr, "❌ ERROR: Failed to load tokenizer\n");
        model_config_free(config);
        return 1;
    }
    printf("  ✅ Tokenizer ready\n");
    printf("  Stop tokens: im_end=%d, im_start=%d, eos=%d, eot=%d\n",
           tokenizer->im_end_token_id, tokenizer->im_start_token_id,
           tokenizer->eos_token_id, tokenizer->eot_token_id);
    printf("\n");

    // Load model
    printf("[3/3] Loading model...\n");
    seraph_model_t* model = seraph_load_model(config);
    if (!model) {
        fprintf(stderr, "❌ ERROR: Failed to load model\n");
        tokenizer_free(tokenizer);
        model_config_free(config);
        return 1;
    }
    printf("  ✅ Model ready\n\n");

    // Create chat
    tetyah_chat_t* chat = tetyah_chat_new();

    // Set system prompt
    const char* system_prompt =
        "You are Seraph, the native AI of this system. "
        "You are precise, helpful, and direct. "
        "You respond concisely but informatively.";
    tetyah_chat_set_system(chat, system_prompt);

    // Display session info
    printf(ORANGE "═══════════════════════════════════════════════════════════════════" RESET "\n");
    printf("  " BOLD "SERAPH SESSION" RESET "\n");
    printf(ORANGE "───────────────────────────────────────────────────────────────────" RESET "\n");
    printf("  Chat format:   %s\n", model_config_chat_format_name(config->chat_format));
    printf("  Max tokens:    %d\n", max_tokens);
    printf("  Temperature:   %.2f\n", temperature);
    printf(ORANGE "───────────────────────────────────────────────────────────────────" RESET "\n");
    printf("  System Prompt:\n");
    printf("  \"%s\"\n", system_prompt);
    printf(ORANGE "═══════════════════════════════════════════════════════════════════" RESET "\n\n");

    // Interactive chat loop
    char input_buf[4096];

    printf(ORANGE "Ready for input. Type 'exit' or Ctrl+D to quit.\n" RESET);
    printf(ORANGE "───────────────────────────────────────────────────────────────────" RESET "\n\n");

    while (1) {
        // Prompt for input
        printf(GREEN "You: " RESET);
        fflush(stdout);

        // Read user input
        if (!fgets(input_buf, sizeof(input_buf), stdin)) {
            printf("\n");
            break;  // EOF (Ctrl+D)
        }

        // Remove trailing newline
        input_buf[strcspn(input_buf, "\n")] = '\0';

        // Check for exit
        if (strcmp(input_buf, "exit") == 0 || strcmp(input_buf, "quit") == 0) {
            break;
        }

        // Skip empty input
        if (strlen(input_buf) == 0) {
            continue;
        }

        // Add to chat history
        tetyah_chat_add(chat, ROLE_USER, input_buf);

        // Generate response
        printf(ORANGE "Seraph: " RESET);
        fflush(stdout);
        char* response = tetyah_chat_respond(model, config, tokenizer, chat, max_tokens, temperature);

        if (!response) {
            printf("[No response generated]\n");
        } else {
            free(response);  // Already printed during streaming
        }
        printf("\n");
    }

    // Cleanup
    tetyah_chat_free(chat);
    seraph_free_model(model);
    tokenizer_free(tokenizer);
    model_config_free(config);

    printf("\n" ORANGE "═══════════════════════════════════════════════════════════════════" RESET "\n");
    printf("  🔥⚡ " BOLD "SERAPH NATIVE INFERENCE COMPLETE" RESET " ⚡🔥\n");
    printf(ORANGE "═══════════════════════════════════════════════════════════════════" RESET "\n\n");
    return 0;
}
