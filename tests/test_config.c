// test_config.c - Test model configuration loading
// Part of TETYAH-PRIME's native training and inference engine

#include "../include/model_config.h"
#include <stdio.h>

void print_usage(const char* prog) {
    printf("Usage: %s <model_dir> [model_dir2] ...\n", prog);
    printf("\nLoads and prints model configuration from each directory.\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    printf("🔥⚡ SERAPH MODEL CONFIG LOADER ⚡🔥\n\n");

    int failures = 0;
    for (int i = 1; i < argc; i++) {
        printf("Loading: %s\n", argv[i]);
        model_config_t* config = model_config_load(argv[i]);

        if (config) {
            model_config_print(config);
            model_config_free(config);
        } else {
            printf("  ❌ Failed to load\n\n");
            failures++;
        }
    }

    printf("✅ Config loader test complete (%d/%d loaded)\n", argc - 1 - failures, argc - 1);
    return failures > 0 ? 1 : 0;
}
