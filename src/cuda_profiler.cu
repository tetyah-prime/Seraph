// cuda_profiler.cu - Per-operation CUDA event profiler implementation
// Part of TETYAH-PRIME's native training and inference engine
//
// Uses cudaEvent pairs to measure GPU-side kernel execution time
// Zero-overhead when disabled (just a branch on prof->enabled)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "../include/cuda_profiler.h"
}

// Print to stdout and optionally to log file
#define PROF_PRINTF(prof, fmt, ...) do { \
    printf(fmt, ##__VA_ARGS__); \
    if ((prof)->log_file) fprintf((FILE*)(prof)->log_file, fmt, ##__VA_ARGS__); \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════

extern "C" cuda_profiler_t* cuda_profiler_create(void) {
    cuda_profiler_t* prof = (cuda_profiler_t*)calloc(1, sizeof(cuda_profiler_t));
    if (!prof) return NULL;

    // Pre-create all event pairs (creating events is cheap, doing it once avoids alloc during profiling)
    for (int i = 0; i < PROFILER_MAX_OPS; i++) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        prof->ops[i].event_start = start;
        prof->ops[i].event_end = end;
    }

    return prof;
}

extern "C" void cuda_profiler_destroy(cuda_profiler_t* prof) {
    if (!prof) return;
    for (int i = 0; i < PROFILER_MAX_OPS; i++) {
        if (prof->ops[i].event_start) cudaEventDestroy((cudaEvent_t)prof->ops[i].event_start);
        if (prof->ops[i].event_end)   cudaEventDestroy((cudaEvent_t)prof->ops[i].event_end);
    }
    free(prof);
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTROL
// ═══════════════════════════════════════════════════════════════════════════

extern "C" void cuda_profiler_enable(cuda_profiler_t* prof) {
    if (!prof) return;
    prof->enabled = 1;
    prof->num_ops = 0;
}

extern "C" void cuda_profiler_disable(cuda_profiler_t* prof) {
    if (!prof) return;
    prof->enabled = 0;
}

extern "C" void cuda_profiler_reset(cuda_profiler_t* prof) {
    if (!prof) return;
    prof->num_ops = 0;
    prof->total_step_ms = 0;
    prof->total_matmul_ms = 0;
    prof->total_attention_ms = 0;
    prof->total_attention_bwd_ms = 0;
    prof->total_rmsnorm_ms = 0;
    prof->total_rmsnorm_bwd_ms = 0;
    prof->total_cross_entropy_ms = 0;
    prof->total_adamw_ms = 0;
    prof->total_copy_ms = 0;
    prof->total_other_ms = 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// RECORDING
// ═══════════════════════════════════════════════════════════════════════════

extern "C" int cuda_profiler_begin_op(cuda_profiler_t* prof, const char* name, void* stream) {
    if (!prof || !prof->enabled) return -1;
    if (prof->num_ops >= PROFILER_MAX_OPS) return -1;

    int idx = prof->num_ops++;
    profiler_op_t* op = &prof->ops[idx];

    strncpy(op->name, name, sizeof(op->name) - 1);
    op->name[sizeof(op->name) - 1] = '\0';
    op->elapsed_ms = 0;
    op->grid_x = op->grid_y = op->grid_z = 0;
    op->block_x = op->block_y = op->block_z = 0;
    op->shared_mem = 0;
    op->flops = 0;
    op->bytes = 0;

    cudaEventRecord((cudaEvent_t)op->event_start, (cudaStream_t)stream);
    return idx;
}

extern "C" void cuda_profiler_end_op(cuda_profiler_t* prof, int op_idx, void* stream) {
    if (!prof || !prof->enabled || op_idx < 0 || op_idx >= prof->num_ops) return;
    cudaEventRecord((cudaEvent_t)prof->ops[op_idx].event_end, (cudaStream_t)stream);
}

extern "C" void cuda_profiler_set_launch(cuda_profiler_t* prof, int op_idx,
                                          int gx, int gy, int gz,
                                          int bx, int by, int bz,
                                          size_t shared_mem) {
    if (!prof || op_idx < 0 || op_idx >= prof->num_ops) return;
    profiler_op_t* op = &prof->ops[op_idx];
    op->grid_x = gx; op->grid_y = gy; op->grid_z = gz;
    op->block_x = bx; op->block_y = by; op->block_z = bz;
    op->shared_mem = shared_mem;
}

extern "C" void cuda_profiler_set_flops(cuda_profiler_t* prof, int op_idx, size_t flops) {
    if (!prof || op_idx < 0 || op_idx >= prof->num_ops) return;
    prof->ops[op_idx].flops = flops;
}

extern "C" void cuda_profiler_set_bytes(cuda_profiler_t* prof, int op_idx, size_t bytes) {
    if (!prof || op_idx < 0 || op_idx >= prof->num_ops) return;
    prof->ops[op_idx].bytes = bytes;
}

// ═══════════════════════════════════════════════════════════════════════════
// ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

extern "C" void cuda_profiler_sync_and_analyze(cuda_profiler_t* prof, void* stream) {
    if (!prof || prof->num_ops == 0) return;

    cudaStreamSynchronize((cudaStream_t)stream);

    // Reset aggregates
    prof->total_step_ms = 0;
    prof->total_matmul_ms = 0;
    prof->total_attention_ms = 0;
    prof->total_attention_bwd_ms = 0;
    prof->total_rmsnorm_ms = 0;
    prof->total_rmsnorm_bwd_ms = 0;
    prof->total_cross_entropy_ms = 0;
    prof->total_adamw_ms = 0;
    prof->total_copy_ms = 0;
    prof->total_other_ms = 0;

    for (int i = 0; i < prof->num_ops; i++) {
        profiler_op_t* op = &prof->ops[i];
        cudaEventElapsedTime(&op->elapsed_ms,
                             (cudaEvent_t)op->event_start,
                             (cudaEvent_t)op->event_end);
        prof->total_step_ms += op->elapsed_ms;

        // Categorize by name prefix
        if (strstr(op->name, "matmul") || strstr(op->name, "Matmul")) {
            prof->total_matmul_ms += op->elapsed_ms;
        } else if (strstr(op->name, "attn_bwd") || strstr(op->name, "attention_bwd")) {
            prof->total_attention_bwd_ms += op->elapsed_ms;
        } else if (strstr(op->name, "attn") || strstr(op->name, "attention")) {
            prof->total_attention_ms += op->elapsed_ms;
        } else if (strstr(op->name, "rmsnorm_bwd")) {
            prof->total_rmsnorm_bwd_ms += op->elapsed_ms;
        } else if (strstr(op->name, "rmsnorm")) {
            prof->total_rmsnorm_ms += op->elapsed_ms;
        } else if (strstr(op->name, "cross_entropy")) {
            prof->total_cross_entropy_ms += op->elapsed_ms;
        } else if (strstr(op->name, "adamw")) {
            prof->total_adamw_ms += op->elapsed_ms;
        } else if (strstr(op->name, "copy") || strstr(op->name, "memset") || strstr(op->name, "memcpy")) {
            prof->total_copy_ms += op->elapsed_ms;
        } else {
            prof->total_other_ms += op->elapsed_ms;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PRINTING
// ═══════════════════════════════════════════════════════════════════════════

extern "C" void cuda_profiler_print(cuda_profiler_t* prof) {
    if (!prof || prof->num_ops == 0) {
        PROF_PRINTF(prof, "  [profiler] No operations recorded\n");
        return;
    }

    float total = prof->total_step_ms;
    if (total <= 0) total = 1e-6f;

    PROF_PRINTF(prof, "\n");
    PROF_PRINTF(prof, "  ══════════════════════════════════════════════════════════════════\n");
    PROF_PRINTF(prof, "  SERAPH STEP PROFILER (%d operations, %.2f ms total)\n", prof->num_ops, prof->total_step_ms);
    PROF_PRINTF(prof, "  ══════════════════════════════════════════════════════════════════\n");

    // Category summary
    PROF_PRINTF(prof, "\n  ── CATEGORY BREAKDOWN ──\n\n");
    PROF_PRINTF(prof, "  %-24s %10s  %6s  %s\n", "Category", "Time (ms)", "%", "Bar");
    PROF_PRINTF(prof, "  %-24s %10s  %6s  %s\n", "────────────────────────", "──────────", "──────", "────────────────────────────────────────");

    struct { const char* name; float ms; } cats[] = {
        {"matmul (all)",         prof->total_matmul_ms},
        {"attention fwd",        prof->total_attention_ms},
        {"attention bwd",        prof->total_attention_bwd_ms},
        {"rmsnorm fwd",          prof->total_rmsnorm_ms},
        {"rmsnorm bwd",          prof->total_rmsnorm_bwd_ms},
        {"cross_entropy",        prof->total_cross_entropy_ms},
        {"adamw",                prof->total_adamw_ms},
        {"copy/memset",          prof->total_copy_ms},
        {"other",                prof->total_other_ms},
    };
    int ncats = sizeof(cats) / sizeof(cats[0]);

    for (int i = 0; i < ncats; i++) {
        float pct = 100.0f * cats[i].ms / total;
        int bar_len = (int)(pct * 0.4f);
        if (bar_len > 40) bar_len = 40;
        char bar[41];
        for (int b = 0; b < bar_len; b++) bar[b] = '#';
        bar[bar_len] = '\0';
        PROF_PRINTF(prof, "  %-24s %10.3f  %5.1f%%  %s\n", cats[i].name, cats[i].ms, pct, bar);
    }

    // Per-operation detail
    PROF_PRINTF(prof, "\n  ── PER-OPERATION DETAIL ──\n\n");
    PROF_PRINTF(prof, "  %4s  %-60s %10s  %5s", "#", "Operation", "Time (ms)", "%");
    PROF_PRINTF(prof, "  %s\n", "Grid / Block");
    PROF_PRINTF(prof, "  %4s  %-60s %10s  %5s  %s\n",
           "────", "────────────────────────────────────────────────────────────",
           "──────────", "─────", "──────────────────");

    for (int i = 0; i < prof->num_ops; i++) {
        profiler_op_t* op = &prof->ops[i];
        float pct = 100.0f * op->elapsed_ms / total;
        char launch[40] = "";
        if (op->block_x > 0) {
            if (op->grid_z > 1 || op->block_z > 1) {
                snprintf(launch, sizeof(launch), "(%d,%d,%d)/(%d,%d,%d)",
                         op->grid_x, op->grid_y, op->grid_z,
                         op->block_x, op->block_y, op->block_z);
            } else if (op->grid_y > 1 || op->block_y > 1) {
                snprintf(launch, sizeof(launch), "(%d,%d)/(%d,%d)",
                         op->grid_x, op->grid_y, op->block_x, op->block_y);
            } else {
                snprintf(launch, sizeof(launch), "%d/%d",
                         op->grid_x, op->block_x);
            }
        }

        // Highlight ops > 1% of step time
        const char* marker = (pct > 5.0f) ? " <<<" : (pct > 1.0f) ? " <<" : "";
        PROF_PRINTF(prof, "  %4d  %-60s %10.3f  %4.1f%%  %-20s%s\n",
               i, op->name, op->elapsed_ms, pct, launch, marker);
    }

    // Bandwidth/FLOPS for ops that have it
    int has_perf = 0;
    for (int i = 0; i < prof->num_ops; i++) {
        if (prof->ops[i].flops > 0 || prof->ops[i].bytes > 0) { has_perf = 1; break; }
    }
    if (has_perf) {
        PROF_PRINTF(prof, "\n  ── PERFORMANCE METRICS ──\n\n");
        PROF_PRINTF(prof, "  %4s  %-40s %10s  %10s  %10s\n", "#", "Operation", "GFLOPS", "GB/s", "Time (ms)");
        PROF_PRINTF(prof, "  %4s  %-40s %10s  %10s  %10s\n", "────", "────────────────────────────────────────",
               "──────────", "──────────", "──────────");
        for (int i = 0; i < prof->num_ops; i++) {
            profiler_op_t* op = &prof->ops[i];
            if (op->flops == 0 && op->bytes == 0) continue;
            float gflops = (op->flops > 0 && op->elapsed_ms > 0)
                ? (float)op->flops / (op->elapsed_ms * 1e6f) : 0;
            float gbps = (op->bytes > 0 && op->elapsed_ms > 0)
                ? (float)op->bytes / (op->elapsed_ms * 1e6f) : 0;
            PROF_PRINTF(prof, "  %4d  %-40.40s %10.2f  %10.2f  %10.3f\n",
                   i, op->name, gflops, gbps, op->elapsed_ms);
        }
    }

    PROF_PRINTF(prof, "\n");
}

extern "C" void cuda_profiler_print_top(cuda_profiler_t* prof, int top_n) {
    if (!prof || prof->num_ops == 0) return;

    // Simple selection sort for top_n
    int* sorted = (int*)malloc(prof->num_ops * sizeof(int));
    for (int i = 0; i < prof->num_ops; i++) sorted[i] = i;
    for (int i = 0; i < prof->num_ops - 1 && i < top_n; i++) {
        int max_idx = i;
        for (int j = i + 1; j < prof->num_ops; j++) {
            if (prof->ops[sorted[j]].elapsed_ms > prof->ops[sorted[max_idx]].elapsed_ms) {
                max_idx = j;
            }
        }
        int tmp = sorted[i]; sorted[i] = sorted[max_idx]; sorted[max_idx] = tmp;
    }

    float total = prof->total_step_ms > 0 ? prof->total_step_ms : 1e-6f;

    PROF_PRINTF(prof, "\n  ── TOP %d SLOWEST OPERATIONS ──\n\n", top_n);
    PROF_PRINTF(prof, "  %4s  %-55s %10s  %5s\n", "Rank", "Operation", "Time (ms)", "%");
    PROF_PRINTF(prof, "  %4s  %-55s %10s  %5s\n", "────",
           "───────────────────────────────────────────────────────",
           "──────────", "─────");

    int n = top_n < prof->num_ops ? top_n : prof->num_ops;
    for (int i = 0; i < n; i++) {
        profiler_op_t* op = &prof->ops[sorted[i]];
        float pct = 100.0f * op->elapsed_ms / total;
        PROF_PRINTF(prof, "  %4d  %-55.55s %10.3f  %4.1f%%\n", i + 1, op->name, op->elapsed_ms, pct);
    }

    free(sorted);
}

extern "C" void cuda_profiler_set_log(cuda_profiler_t* prof, void* file_ptr) {
    if (prof) prof->log_file = file_ptr;
}
