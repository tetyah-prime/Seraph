// train_lora_cuda.c - Seraph CUDA Training
// Usage: seraph-train-cuda <model_dir> <train.ttok> [options]
//
// Part of TETYAH-PRIME's native training and inference engine
// CUDA backend for NVIDIA GPUs (cloud or local)

#include "../include/train.h"
#include "../include/train_cuda.h"
#include "../include/cuda_profiler.h"
#include "../include/lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

void print_usage(const char* prog) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Seraph CUDA Training\n");
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
    printf("  --gpu-device N    CUDA device index (default: 0)\n");
    printf("  --batch N         Batch size (N >= 2)\n");
    printf("  --bf16            BF16 mixed precision (tensor cores, requires Ampere+)\n");
    printf("  --flash           FlashAttention (tiled, no seq*seq memory)\n");
    printf("  --val PATH        Validation .ttok file (eval-only forward pass, no training)\n");
    printf("  --eval-every N [M] [S]  Evaluate every N steps, max M samples, S = val split ratio\n");
    printf("                          (e.g. --eval-every 500 20 0.1 = every 500 steps, 20 samples, 10%% split)\n");
    printf("  --debug           Enable extra debug output\n");
    printf("  --debug-weight-delta  Print w_embed[0] delta after optimizer steps (GPU full training)\n");
    printf("  --debug-nan-check Enable GPU NaN/Inf checks (stops before weights become NaN)\n");
    printf("  --profile N       Profile N training steps (print per-kernel GPU timing)\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s models/tetyah-prime-1.5b train.ttok --epochs 3 --lr 2e-4\n", prog);
    printf("\n");
}

// Timing helper
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* train_data = argv[2];
    const char* resume_path = NULL;
    int eval_max_samples = 32;
    float eval_split = 0.0f;

    // Default config
    train_config_t config = {
        .train_data_path = (char*)train_data,
        .val_data_path = NULL,
        .epochs = 1,
        .batch_size = 1,
        .max_seq_len = 512,
        .fixed_seq_windows = 0,
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
        .snapshot_every_n_steps = 0,
        .debug = 0,
        .debug_weight_delta = 0,
        .debug_nan_check = 0,
        .cpu_only = 0,
        .gpu_device_index = 0,
        .log_path = NULL
    };

    // Parse options
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            config.epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            config.learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--accum") == 0 && i + 1 < argc) {
            config.gradient_accumulation_steps = atoi(argv[++i]);
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
            config.use_full_training = 1;  // Enable full weight training
        } else if (strcmp(argv[i], "--lora-rank") == 0 && i + 1 < argc) {
            config.lora_rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lora-alpha") == 0 && i + 1 < argc) {
            config.lora_alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--snapshot-every") == 0 && i + 1 < argc) {
            config.snapshot_every_n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) {
            config.gpu_device_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            config.batch_size = atoi(argv[++i]);
            if (config.batch_size < 2) {
                fprintf(stderr, "ERROR: --batch must be >= 2\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--bf16") == 0) {
            config.use_bf16 = 1;
        } else if (strcmp(argv[i], "--flash") == 0) {
            // set after cuda init below
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
        } else if (strcmp(argv[i], "--val") == 0 && i + 1 < argc) {
            config.val_data_path = argv[++i];
        } else if (strcmp(argv[i], "--eval-every") == 0 && i + 1 < argc) {
            // --eval-every <steps> [max_samples] [split_ratio]
            config.eval_every_n_steps = atoi(argv[++i]);
            if (i + 1 < argc && argv[i+1][0] >= '0' && argv[i+1][0] <= '9') {
                eval_max_samples = atoi(argv[++i]);
            }
            if (i + 1 < argc && argv[i+1][0] >= '0' && argv[i+1][0] <= '9') {
                eval_split = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            // Handled below after CUDA init
        }
    }

    // Parse --profile separately (need value after CUDA init)
    int profile_steps = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            profile_steps = atoi(argv[i + 1]);
            break;
        }
    }

    // Seed RNG
    srand(time(NULL));

    // Banner
    printf("\n");
    printf("\033[38;5;208m╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                   ║\n");
    printf("║         \033[1m███████╗███████╗██████╗  █████╗ ██████╗ ██╗  ██╗\033[0m\033[38;5;208m          ║\n");
    printf("║         \033[1m██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║  ██║\033[0m\033[38;5;208m          ║\n");
    printf("║         \033[1m███████╗█████╗  ██████╔╝███████║██████╔╝███████║\033[0m\033[38;5;208m          ║\n");
    printf("║         \033[1m╚════██║██╔══╝  ██╔══██╗██╔══██║██╔═══╝ ██╔══██║\033[0m\033[38;5;208m          ║\n");
    printf("║         \033[1m███████║███████╗██║  ██║██║  ██║██║     ██║  ██║\033[0m\033[38;5;208m          ║\n");
    printf("║         \033[1m╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝\033[0m\033[38;5;208m          ║\n");
    printf("║                                                                   ║\n");
    printf("║                     🔥⚡ CUDA TRAINING ⚡🔥                       ║\n");
    printf("║                                                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\033[0m\n");
    printf("\n");

    // Initialize training (CUDA init happens at step 7 via function pointer)
    printf("Initializing training...\n");

    config.cuda_backend = 1;
    config.cuda_init_fn = (void* (*)(int))cuda_train_init;

    train_state_t* state = train_init(model_dir, &config);
    if (!state) {
        fprintf(stderr, "ERROR: Failed to initialize training\n");
        return 1;
    }

    // Get CUDA context from state (initialized at step 7)
    cuda_train_context_t* cuda_ctx = (cuda_train_context_t*)state->cuda_ctx;

    // Set max buffer size for CUDA context
    cuda_ctx->max_fwd_buffer_size = (size_t)config.max_seq_len *
                                    (size_t)state->config->intermediate_size * sizeof(float);

    // Set FlashAttention mode
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--flash") == 0) {
            cuda_ctx->use_flash_attn = 1;
            printf("  Attention:      FlashAttention\n");
            break;
        }
    }

    // Set BF16 mode on CUDA context
    if (config.use_bf16) {
        if (cuda_ctx->gpu_compute_major >= 8) {
            cuda_ctx->use_bf16 = 1;
            printf("  Precision:      BF16 mixed (tensor cores)\n");
        } else {
            fprintf(stderr, "WARNING: --bf16 requires compute capability >= 8.0 (Ampere+), got %d.%d\n",
                    cuda_ctx->gpu_compute_major, cuda_ctx->gpu_compute_minor);
            fprintf(stderr, "         Falling back to FP32\n");
            cuda_ctx->use_bf16 = 0;
        }
    }

    // Create checkpoint directory early (before profile/snapshot open files)
    mkdir(config.checkpoint_dir, 0755);

    // Setup profiler if requested
    cuda_profiler_t* profiler = NULL;
    FILE* profile_log = NULL;
    if (profile_steps > 0) {
        profiler = cuda_profiler_create();
        cuda_train_set_profiler(cuda_ctx, profiler);

        // Save profile output to checkpoint dir
        char profile_path[512];
        snprintf(profile_path, sizeof(profile_path), "%s/profile.log", config.checkpoint_dir);
        profile_log = fopen(profile_path, "w");
        if (profile_log) {
            cuda_profiler_set_log(profiler, profile_log);
            // Write hardware header so replay knows the GPU specs
            fprintf(profile_log, "# HARDWARE sm_count=%d warp_size=%d threads_sm=%d threads_block=%d "
                    "shared_sm=%d shared_block=%d shared_optin=%d regs_sm=%d compute=%d.%d\n",
                    cuda_ctx->gpu_sm_count, cuda_ctx->gpu_warp_size,
                    cuda_ctx->gpu_max_threads_per_sm, cuda_ctx->gpu_max_threads_per_block,
                    (int)cuda_ctx->gpu_shared_mem_per_sm, (int)cuda_ctx->gpu_shared_mem_per_block,
                    (int)cuda_ctx->gpu_max_shared_per_block_optin, cuda_ctx->gpu_regs_per_sm,
                    cuda_ctx->gpu_compute_major, cuda_ctx->gpu_compute_minor);
            printf("  Profiling:      %d steps → %s\n", profile_steps, profile_path);
        } else {
            printf("  Profiling:      %d steps (stdout only)\n", profile_steps);
        }
    }

    // Load validation data — either from --val file or split from training data
    int* val_tokens = NULL;
    int num_val_tokens = 0;
    int* val_offsets = NULL;
    int num_val_samples = 0;
    int val_from_split = 0;
    if (config.val_data_path) {
        // Separate val file
        train_state_t val_tmp = {0};
        val_tmp.config = state->config;
        if (train_load_data(&val_tmp, config.val_data_path) == 0) {
            val_tokens = val_tmp.train_tokens;
            num_val_tokens = val_tmp.num_train_tokens;
            val_offsets = val_tmp.sample_offsets;
            num_val_samples = val_tmp.num_samples;
            printf("  Validation:     %d samples, %d tokens (%s)\n",
                   num_val_samples, num_val_tokens, config.val_data_path);
        } else {
            fprintf(stderr, "WARNING: Could not load validation data: %s\n", config.val_data_path);
        }
    } else if (eval_split > 0.0f && eval_split < 1.0f && state->num_samples > 1) {
        // Split from training data — take last N% as validation
        int split_idx = (int)(state->num_samples * (1.0f - eval_split));
        if (split_idx < 1) split_idx = 1;
        if (split_idx >= state->num_samples) split_idx = state->num_samples - 1;

        num_val_samples = state->num_samples - split_idx;
        val_offsets = &state->sample_offsets[split_idx];
        val_tokens = state->train_tokens;  // same buffer, different offsets
        num_val_tokens = state->num_train_tokens;
        val_from_split = 1;

        // Shrink training set
        state->num_samples = split_idx;
        printf("  Val split:      %.0f%% → %d train / %d val samples\n",
               eval_split * 100.0f, state->num_samples, num_val_samples);
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

    // Create log file path
    char log_path[512];
    snprintf(log_path, sizeof(log_path), "%s/train.log", config.checkpoint_dir);
    FILE* log_file = fopen(log_path, "a");

    // Training info (matches train_loop output format)
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  SERAPH CUDA TRAINING\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Samples:        %d\n", state->num_samples);
    printf("  Total tokens:   %d\n", state->num_train_tokens);
    printf("  Epochs:         %d\n", config.epochs);
    printf("  Learning rate:  %.6f\n", config.learning_rate);
    if (!config.use_full_training) {
        // LoRA mode
        printf("  LoRA rank:      %d\n", config.lora_rank);
        printf("  LoRA alpha:     %.1f\n", config.lora_alpha);
    } else {
        // Full training mode
        printf("  Training:       Full weights\n");
    }
    if (config.batch_size > 1)
        printf("  Batch size:     %d\n", config.batch_size);
    printf("  Precision:      %s\n", cuda_ctx->use_bf16 ? "BF16 mixed (tensor cores)" : "FP32");
    if (num_val_samples > 0) {
        printf("  Validation:     %d samples", num_val_samples);
        if (config.eval_every_n_steps > 0)
            printf(" (every %d steps)", config.eval_every_n_steps);
        printf("\n");
    }
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("  Log file:       %s\n", log_path);

    // Training loop
    state->learning_rate = config.learning_rate;
    state->gradient_accumulation_steps = config.gradient_accumulation_steps;

    int total_steps = 0;
    float epoch_loss_sum = 0.0f;
    int epoch_loss_count = 0;

    // Rolling window loss
    int loss_window = config.log_every_n_steps > 0 ? config.log_every_n_steps : 100;
    float* loss_ring = calloc(loss_window, sizeof(float));
    float loss_ring_sum = 0.0f;
    int loss_ring_pos = 0;
    int loss_ring_filled = 0;

    // Pre-allocate packed batch buffers (reused each step)
    int batch_sz = config.batch_size;
    int max_batch_tokens = batch_sz * config.max_seq_len;
    int* batch_tokens_packed = malloc((size_t)max_batch_tokens * sizeof(int));
    int* batch_seq_starts = malloc((size_t)(batch_sz + 1) * sizeof(int));  // prefix sum

    for (int epoch = state->current_epoch; epoch < config.epochs; epoch++) {
        state->current_epoch = epoch;
        epoch_loss_sum = 0.0f;
        epoch_loss_count = 0;

        printf("Epoch %d/%d\n", epoch + 1, config.epochs);
        printf("───────────────────────────────────────────────────────────────────\n");

        int num_units = (state->num_samples + batch_sz - 1) / batch_sz;  // Steps per epoch

        // Iterate through training data
        int sample_idx = 0;
        while (sample_idx < state->num_samples) {

            // ── Gather packed batch (zero padding waste) ──
            int batch_count = 0;
            int total_valid_tokens = 0;
            batch_seq_starts[0] = 0;

            int src_idx = sample_idx;
            while (batch_count < batch_sz && src_idx < state->num_samples) {
                int start = state->sample_offsets[src_idx];
                int end = (src_idx + 1 < state->num_samples)
                        ? state->sample_offsets[src_idx + 1]
                        : state->num_train_tokens;
                int seq_len = end - start;
                if (seq_len > config.max_seq_len) seq_len = config.max_seq_len;
                if (seq_len < 2) { src_idx++; continue; }

                // Pack contiguously — no padding
                memcpy(&batch_tokens_packed[total_valid_tokens],
                       &state->train_tokens[start],
                       (size_t)seq_len * sizeof(int));
                total_valid_tokens += seq_len;
                batch_count++;
                batch_seq_starts[batch_count] = total_valid_tokens;
                src_idx++;
            }

            if (batch_count == 0) {
                sample_idx = src_idx;
                continue;
            }
            sample_idx = src_idx;

            double step_start = now_ms();

            // Enable profiler for this step if within profile window
            if (profiler && total_steps < profile_steps) {
                cuda_profiler_reset(profiler);
                cuda_profiler_enable(profiler);
            }

            // CUDA training step
            float loss;
            if (batch_count == 1) {
                // Single sequence: original unbatched path
                if (config.use_full_training) {
                    loss = cuda_train_step_gpu_full(state, batch_tokens_packed,
                                                     batch_seq_starts[1], cuda_ctx);
                } else {
                    loss = cuda_train_step_gpu_lora(state, batch_tokens_packed,
                                                     batch_seq_starts[1], cuda_ctx);
                }
            } else {
                // Packed batch: multiple sequences
                if (config.use_full_training) {
                    loss = cuda_train_step_gpu_full_batch(state, batch_tokens_packed,
                                                          total_valid_tokens, batch_seq_starts,
                                                          batch_count, cuda_ctx);
                } else {
                    loss = cuda_train_step_gpu_lora_batch(state, batch_tokens_packed,
                                                          total_valid_tokens, batch_seq_starts,
                                                          batch_count, cuda_ctx);
                }
            }

            // Print profiler results if this was a profiled step
            if (profiler && total_steps < profile_steps) {
                cuda_profiler_sync_and_analyze(profiler, cuda_ctx->stream);
                if (batch_count > 1) {
                    printf("\n  ── PROFILE STEP %d (batch=%d, packed_tokens=%d, loss=%.4f) ──\n",
                           total_steps + 1, batch_count, total_valid_tokens, loss);
                    if (profile_log) fprintf(profile_log, "\n  ── PROFILE STEP %d (batch=%d, packed_tokens=%d, loss=%.4f) ──\n",
                           total_steps + 1, batch_count, total_valid_tokens, loss);
                } else {
                    printf("\n  ── PROFILE STEP %d (seq_len=%d, loss=%.4f) ──\n",
                           total_steps + 1, total_valid_tokens, loss);
                    if (profile_log) fprintf(profile_log, "\n  ── PROFILE STEP %d (seq_len=%d, loss=%.4f) ──\n",
                           total_steps + 1, total_valid_tokens, loss);
                }
                cuda_profiler_print(profiler);
                cuda_profiler_disable(profiler);
                if (profile_log) fflush(profile_log);
            }

            // Check for NaN and stop training if detected
            if (state->nan_detected) {
                fprintf(stderr, "\n[FATAL] NaN detected at step %d - stopping training\n", state->current_step + 1);
                fprintf(stderr, "        Check /tmp/seraph_nan_step_%d_* for buffer dumps\n", state->current_step + 1);
                if (log_file) {
                    fprintf(log_file, "[FATAL] NaN detected at step %d - stopping training\n", state->current_step + 1);
                    fflush(log_file);
                }
                free(loss_ring);
                free(batch_tokens_packed);
                free(batch_seq_starts);
                goto cleanup;
            }

            double step_end = now_ms();
            double step_ms = step_end - step_start;
            float tok_per_sec = (float)total_valid_tokens / (step_ms / 1000.0f);

            // Update loss tracking
            epoch_loss_sum += loss;
            epoch_loss_count++;
            state->current_loss = loss;
            state->current_step++;
            total_steps++;

            // Rolling window loss
            loss_ring_sum -= loss_ring[loss_ring_pos];
            loss_ring[loss_ring_pos] = loss;
            loss_ring_sum += loss;
            loss_ring_pos = (loss_ring_pos + 1) % loss_window;
            if (!loss_ring_filled && loss_ring_pos == 0) loss_ring_filled = 1;
            int ring_count = loss_ring_filled ? loss_window : loss_ring_pos;
            float rolling_loss = ring_count > 0 ? loss_ring_sum / ring_count : loss;

            // Log progress
            if (config.log_every_n_steps > 0 && total_steps % config.log_every_n_steps == 0) {
                float avg_loss = epoch_loss_sum / epoch_loss_count;
                float ppl = expf(rolling_loss);
                if (batch_count > 1) {
                    printf("  Step %d/%d | Loss: %.4f | Avg: %.4f | PPL: %.1f | %.0f tok/s | batch=%d\n",
                           total_steps, num_units * config.epochs, rolling_loss, avg_loss,
                           ppl, tok_per_sec, batch_count);
                } else {
                    printf("  Step %d/%d | Loss: %.4f | Avg: %.4f | PPL: %.1f | %.0f tok/s\n",
                           total_steps, num_units * config.epochs, rolling_loss, avg_loss,
                           ppl, tok_per_sec);
                }

                if (log_file) {
                    fprintf(log_file, "  Step %d/%d | Loss: %.4f | Avg: %.4f | PPL: %.1f | %.0f tok/s\n",
                            total_steps, num_units * config.epochs, rolling_loss, avg_loss,
                            ppl, tok_per_sec);
                    fflush(log_file);
                }
            }

            // Validation eval
            if (config.eval_every_n_steps > 0 && num_val_samples > 0 &&
                total_steps % config.eval_every_n_steps == 0) {
                float val_ppl = 0.0f;
                float val_loss = cuda_evaluate(state, val_tokens, num_val_tokens,
                                                val_offsets, num_val_samples,
                                                config.max_seq_len, eval_max_samples,
                                                cuda_ctx, &val_ppl);
                printf("  [VAL] Step %d | Val Loss: %.4f | Val PPL: %.1f\n",
                       total_steps, val_loss, val_ppl);
                if (log_file) {
                    fprintf(log_file, "  [VAL] Step %d | Val Loss: %.4f | Val PPL: %.1f\n",
                            total_steps, val_loss, val_ppl);
                    fflush(log_file);
                }
                // Track best val loss for early stopping
                if (val_loss > 0.0f && (state->best_val_loss <= 0.0f || val_loss < state->best_val_loss)) {
                    state->best_val_loss = val_loss;
                }
            }

            // Save checkpoint (matches Vulkan save_checkpoint output)
            if (config.save_every_n_steps > 0 && state->current_step % config.save_every_n_steps == 0) {
                char ckpt_path[512];

                if (config.use_full_training) {
                    cuda_download_full_weights(cuda_ctx, state);
                    snprintf(ckpt_path, sizeof(ckpt_path), "%s/checkpoint_%d.safetensors",
                             config.checkpoint_dir, state->current_step);
                    trainable_weights_save_to_safetensors(state->full_weights, ckpt_path);
                } else {
                    cuda_download_lora_weights(cuda_ctx, state->lora);
                    snprintf(ckpt_path, sizeof(ckpt_path), "%s/checkpoint_%d.bin",
                             config.checkpoint_dir, state->current_step);
                    lora_save(state->lora, ckpt_path);  // prints "Saved LoRA weights: ..."
                }

                // Save training state
                char state_file[512];
                snprintf(state_file, sizeof(state_file), "%s.state", ckpt_path);
                FILE* sf = fopen(state_file, "wb");
                if (sf) {
                    fwrite(&state->current_epoch, sizeof(int), 1, sf);
                    fwrite(&state->current_step, sizeof(int), 1, sf);
                    fwrite(&state->current_loss, sizeof(float), 1, sf);
                    fwrite(&state->optimizer_step, sizeof(int), 1, sf);
                    fclose(sf);
                    printf("  Saved training state: %s\n", state_file);
                }
                printf("  ✅ Saved checkpoint: %s\n", ckpt_path);
            }

            // Gradient snapshot for field visualization (full training only)
            if (config.snapshot_every_n_steps > 0 &&
                state->current_step % config.snapshot_every_n_steps == 0 &&
                state->snapshot_writer && state->full_weights) {
                cuda_download_gpu_gradients(cuda_ctx, state);
                float lr = config.learning_rate;
                gradient_snapshot_write(state->snapshot_writer, state->full_weights,
                                       state->current_step, state->current_loss, lr);
            }

            // Exit after profile steps
            if (profiler && total_steps >= profile_steps) {
                printf("\n  Profiling complete (%d steps). Exiting.\n\n", profile_steps);
                if (profile_log) fprintf(profile_log, "\n  Profiling complete (%d steps).\n\n", profile_steps);
                cuda_profiler_print_top(profiler, 20);
                if (profile_log) { fflush(profile_log); fclose(profile_log); profile_log = NULL; }
                cuda_profiler_destroy(profiler);
                free(loss_ring);
                free(batch_tokens_packed);
                free(batch_seq_starts);
                goto cleanup;
            }
        }

        // Epoch summary
        float avg_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0f;
        printf("  Epoch %d complete | Avg Loss: %.4f\n\n", epoch + 1, avg_loss);
    }

    free(loss_ring);
    free(batch_tokens_packed);
    free(batch_seq_starts);

    // Save final checkpoint
    char final_path[512];

    if (config.use_full_training) {
        cuda_download_full_weights(cuda_ctx, state);
        snprintf(final_path, sizeof(final_path), "%s/checkpoint_final.safetensors", config.checkpoint_dir);
        trainable_weights_save_to_safetensors(state->full_weights, final_path);
    } else {
        cuda_download_lora_weights(cuda_ctx, state->lora);
        snprintf(final_path, sizeof(final_path), "%s/checkpoint_final.bin", config.checkpoint_dir);
        lora_save(state->lora, final_path);  // prints "Saved LoRA weights: ..."
    }

    // Save final training state
    char state_file[512];
    snprintf(state_file, sizeof(state_file), "%s.state", final_path);
    FILE* sf = fopen(state_file, "wb");
    if (sf) {
        fwrite(&state->current_epoch, sizeof(int), 1, sf);
        fwrite(&state->current_step, sizeof(int), 1, sf);
        fwrite(&state->current_loss, sizeof(float), 1, sf);
        fwrite(&state->optimizer_step, sizeof(int), 1, sf);
        fclose(sf);
        printf("  Saved training state: %s\n", state_file);
    }
    printf("  ✅ Saved checkpoint: %s\n", final_path);

    printf("  Final loss: %.4f\n", state->current_loss);

cleanup:
    // Cleanup
    if (profile_log) fclose(profile_log);
    if (log_file) fclose(log_file);
    if (!val_from_split) {
        free(val_tokens);
        free(val_offsets);
    }
    cuda_train_free(cuda_ctx);
    train_free(state);

    printf("\n\033[38;5;208m═══════════════════════════════════════════════════════════════════\033[0m\n");
    printf("  🔥⚡ \033[1mSERAPH CUDA TRAINING COMPLETE\033[0m ⚡🔥\n");
    printf("\033[38;5;208m═══════════════════════════════════════════════════════════════════\033[0m\n\n");

    return 0;
}
