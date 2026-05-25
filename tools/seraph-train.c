// train_lora.c - Seraph LoRA Training
// Usage: seraph-train <model_dir> <train.ttok> [options]
//
// Part of TETYAH-PRIME's native training and inference engine

#include "../include/train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_usage(const char* prog) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Seraph Training\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("Usage: %s <model_dir> <train.ttok> [options]\n", prog);
    printf("\n");
    printf("Training Mode:\n");
    printf("  --full-training   Train all weights (FP32, for random init)\n");
    printf("  (default)         LoRA mode - train adapter weights only\n");
    printf("\n");
    printf("Options:\n");
    printf("  --resume PATH     Resume from checkpoint (e.g., checkpoints/checkpoint_5000.bin)\n");
    printf("  --epochs N        Number of epochs (default: 1)\n");
    printf("  --lr RATE         Learning rate (default: 1e-4)\n");
    printf("  --accum N         Gradient accumulation steps (default: 1)\n");
    printf("  --lora-rank N     LoRA rank (default: 8, LoRA mode only)\n");
    printf("  --lora-alpha F    LoRA alpha (default: 16.0, LoRA mode only)\n");
    printf("  --max-seq N       Max sequence length (default: 512)\n");
    printf("  --fixed-seq       Fixed-length windows only (prevents gradient scale variance)\n");
    printf("  --save-every N    Save checkpoint every N steps (default: 1000)\n");
    printf("  --log-every N     Log every N steps (default: 100)\n");
    printf("  --snapshot-every N  Capture gradients every N training steps (for visualization)\n");
    printf("  --checkpoint DIR  Checkpoint directory (default: ./checkpoints)\n");
    printf("  --gpu-device N    Vulkan device index (-1 = auto; prefer discrete)\n");
    printf("  --cpu-only        Force CPU training (disable Vulkan even if available)\n");
    printf("  --debug           Enable extra debug output\n");
    printf("  --debug-weight-delta  Print w_embed[0] delta after optimizer steps (GPU full training)\n");
    printf("  --debug-nan-check Enable GPU NaN/Inf checks (stops before weights become NaN)\n");
    printf("  --video           [TESTING]Video/ViT mode: enable QK normalization across heads before attention\n");
    printf("  --max-frames N    Limit video frames loaded (saves memory, 0 = all)\n");
    printf("  --clip-weight F   CLIP contrastive loss weight (default: 1.0, 0 = disabled)\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s models/tetyah-prime-1.5b train.ttok --epochs 3 --lr 2e-4\n", prog);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* train_data = argv[2];
    const char* resume_path = NULL;
    float clip_weight_override = -1.0f;  // -1 = use config, >= 0 = override

    // Default config
    train_config_t config = {
        .train_data_path = (char*)train_data,
        .val_data_path = NULL,
        .epochs = 1,
        .batch_size = 1,
        .max_seq_len = 512,
        .fixed_seq_windows = 0,  // 0 = variable length, 1 = fixed max_seq_len windows
        .learning_rate = 1e-4f,
        .lr_warmup_ratio = 0.03f,
        .weight_decay = 0.01f,
        .use_full_training = 0,  // 0 = LoRA mode (default), 1 = Full weight training
        .lora_rank = 8,
        .lora_alpha = 16.0f,
        .lora_dropout = 0.0f,
        .save_every_n_steps = 1000,
        .log_every_n_steps = 100,
        .eval_every_n_steps = 0,
        .checkpoint_dir = "./checkpoints",
        .gradient_accumulation_steps = 1,
        .snapshot_every_n_steps = 0,  // 0 = disabled
        .debug = 0,
        .debug_weight_delta = 0,
        .debug_nan_check = 0,
        .cpu_only = 0,
        .gpu_device_index = -1,
        .log_path = NULL,
        .max_video_frames = 0  // 0 = load all
    };

    // Parse options
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            config.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            config.learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--accum") == 0 && i + 1 < argc) {
            config.gradient_accumulation_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lora-rank") == 0 && i + 1 < argc) {
            config.lora_rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lora-alpha") == 0 && i + 1 < argc) {
            config.lora_alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-seq") == 0 && i + 1 < argc) {
            config.max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fixed-seq") == 0) {
            config.fixed_seq_windows = 1;
        } else if (strcmp(argv[i], "--save-every") == 0 && i + 1 < argc) {
            config.save_every_n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-every") == 0 && i + 1 < argc) {
            config.log_every_n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            config.checkpoint_dir = argv[++i];
        } else if (strcmp(argv[i], "--full-training") == 0) {
            config.use_full_training = 1;  // Enable full weight training (be consciense of weight expansion)
        } else if (strcmp(argv[i], "--snapshot-every") == 0 && i + 1 < argc) {
            config.snapshot_every_n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) {
            config.gpu_device_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-only") == 0) {
            config.cpu_only = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            config.debug = 1;
            config.debug_weight_delta = 1;
            config.debug_nan_check = 1;
        } else if (strcmp(argv[i], "--debug-weight-delta") == 0) {
            config.debug_weight_delta = 1;
        } else if (strcmp(argv[i], "--debug-nan-check") == 0) {
            config.debug_nan_check = 1;
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            resume_path = argv[++i];
        } else if (strcmp(argv[i], "--video") == 0) {
            config.use_qk_norm = 1;  // Video/ViT mode: QK normalization across heads
        } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
            config.max_video_frames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--clip-weight") == 0 && i + 1 < argc) {
            clip_weight_override = atof(argv[++i]);
        }
    }

    // Seed RNG
    srand(time(NULL));

    // Banner
    printf("\n");
    printf("\033[38;5;196m╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                   ║\n");
    printf("║         \033[1m███████╗███████╗██████╗  █████╗ ██████╗ ██╗  ██╗\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║  ██║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m███████╗█████╗  ██████╔╝███████║██████╔╝███████║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m╚════██║██╔══╝  ██╔══██╗██╔══██║██╔═══╝ ██╔══██║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m███████║███████╗██║  ██║██║  ██║██║     ██║  ██║\033[0m\033[38;5;196m          ║\n");
    printf("║         \033[1m╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝\033[0m\033[38;5;196m          ║\n");
    printf("║                                                                   ║\n");
    printf("║                     🔥⚡ NATIVE TRAINING ⚡🔥                     ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\033[0m\n");
    printf("\n");

    // Initialize training
    printf("Initializing training...\n");
    train_state_t* state = train_init(model_dir, &config);
    if (!state) {
        fprintf(stderr, "ERROR: Failed to initialize training\n");
        return 1;
    }

    // Override CLIP weight if specified on command line
    if (clip_weight_override >= 0.0f && state->config) {
        state->config->clip_loss_weight = clip_weight_override;
        if (clip_weight_override > 0.0f) {
            printf("  CLIP contrastive loss weight: %.2f\n", clip_weight_override);
        } else {
            printf("  CLIP contrastive loss: DISABLED\n");
        }
    }

    // Resume from checkpoint if specified
    if (resume_path) {
        printf("\nResuming from checkpoint: %s\n", resume_path);
        if (load_checkpoint(state, resume_path) != 0) {
            fprintf(stderr, "WARNING: Could not load checkpoint, starting fresh\n");
        } else {
            printf("  Resuming from epoch %d, step %d, loss %.4f\n",
                   state->current_epoch, state->current_step, state->current_loss);
        }
    }

    // Create checkpoint directory
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", config.checkpoint_dir);
    system(cmd);

    // Run training loop
    train_loop(state, &config);

    // Save final checkpoint (uses save_checkpoint for consistent .bin + .state)
    char final_path[512];
    if (!config.use_full_training && state->lora) {
        // LoRA mode - save adapter weights + training state
        snprintf(final_path, sizeof(final_path), "%s/lora_final.bin", config.checkpoint_dir);
        save_checkpoint(state, final_path);
    } else if (config.use_full_training && state->full_weights) {
        // Full training mode - save all weights + training state
        // save_checkpoint handles video vs LLM mode and saves state automatically
        snprintf(final_path, sizeof(final_path), "%s/model_final.safetensors", config.checkpoint_dir);
        save_checkpoint(state, final_path);

        // For video models: also save to models/<name>-T/ for inference
        if (state->config && state->config->num_queries > 0) {
            char trained_dir[512];
            snprintf(trained_dir, sizeof(trained_dir), "%s-T", model_dir);

            // Create directory
            char mkdir_cmd[600];
            snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", trained_dir);
            system(mkdir_cmd);

            // Copy model.safetensors
            char model_dest[600];
            snprintf(model_dest, sizeof(model_dest), "%s/model.safetensors", trained_dir);
            FILE* src = fopen(final_path, "rb");
            FILE* dst = fopen(model_dest, "wb");
            if (src && dst) {
                char buf[65536];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
                    fwrite(buf, 1, n, dst);
                }
                printf("  Saved trained model: %s\n", model_dest);
            }
            if (src) fclose(src);
            if (dst) fclose(dst);

            // Copy config.json
            char config_src[600], config_dest[600];
            snprintf(config_src, sizeof(config_src), "%s/config.json", model_dir);
            snprintf(config_dest, sizeof(config_dest), "%s/config.json", trained_dir);
            src = fopen(config_src, "r");
            dst = fopen(config_dest, "w");
            if (src && dst) {
                char buf[4096];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
                    fwrite(buf, 1, n, dst);
                }
                printf("  Copied config: %s\n", config_dest);
            }
            if (src) fclose(src);
            if (dst) fclose(dst);
        }
    }

    // Cleanup
    train_free(state);

    printf("\n\033[38;5;196m═══════════════════════════════════════════════════════════════════\033[0m\n");
    printf("  🔥⚡ \033[1mSERAPH NATIVE TRAINING COMPLETE\033[0m ⚡🔥\n");
    printf("\033[38;5;196m═══════════════════════════════════════════════════════════════════\033[0m\n\n");

    return 0;
}
