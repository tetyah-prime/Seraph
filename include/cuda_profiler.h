// cuda_profiler.h - Per-operation CUDA event profiler for Seraph training
// Records GPU timestamps around every kernel dispatch in a training step
// Part of TETYAH-PRIME's native training and inference engine
//
// Usage: call cuda_profiler_enable() before a step, cuda_profiler_print() after
// Each wrapper function in cuda_train_context.cu auto-records when profiling is active

#ifndef CUDA_PROFILER_H
#define CUDA_PROFILER_H

#include <stdint.h>

// Maximum number of profiled operations per step
// A full GPT-2 (12 layers) forward+backward+optimizer step has ~400+ kernel calls
#define PROFILER_MAX_OPS 1024

// Single profiled operation
typedef struct {
    char name[80];          // Human-readable: "L3 fwd matmul_Bt Q proj [seq=214, M=214, N=768, K=768]"
    void* event_start;      // cudaEvent_t (opaque in C)
    void* event_end;        // cudaEvent_t
    float elapsed_ms;       // Filled after sync
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    size_t shared_mem;
    size_t flops;           // Estimated FLOPs for this op (0 if unknown)
    size_t bytes;           // Estimated bytes moved (0 if unknown)
} profiler_op_t;

// Profiler state (lives on ctx or standalone)
struct cuda_profiler_t {
    int enabled;            // 1 = recording, 0 = off
    int num_ops;            // How many ops recorded this step
    profiler_op_t ops[PROFILER_MAX_OPS];

    // Aggregated stats (filled by cuda_profiler_analyze)
    float total_step_ms;
    float total_matmul_ms;
    float total_attention_ms;
    float total_attention_bwd_ms;
    float total_rmsnorm_ms;
    float total_rmsnorm_bwd_ms;
    float total_cross_entropy_ms;
    float total_adamw_ms;
    float total_copy_ms;
    float total_other_ms;

    // Optional log file — profiler writes to this alongside stdout
    void* log_file;         // FILE* (NULL = stdout only)
};

// Create/destroy
cuda_profiler_t* cuda_profiler_create(void);
void cuda_profiler_destroy(cuda_profiler_t* prof);

// Control
void cuda_profiler_enable(cuda_profiler_t* prof);
void cuda_profiler_disable(cuda_profiler_t* prof);
void cuda_profiler_reset(cuda_profiler_t* prof);

// Record an operation (called by kernel wrappers)
// Returns the op index, or -1 if profiling disabled / full
int cuda_profiler_begin_op(cuda_profiler_t* prof, const char* name, void* stream);
void cuda_profiler_end_op(cuda_profiler_t* prof, int op_idx, void* stream);

// Set launch config on last op (for display)
void cuda_profiler_set_launch(cuda_profiler_t* prof, int op_idx,
                               int gx, int gy, int gz,
                               int bx, int by, int bz,
                               size_t shared_mem);
void cuda_profiler_set_flops(cuda_profiler_t* prof, int op_idx, size_t flops);
void cuda_profiler_set_bytes(cuda_profiler_t* prof, int op_idx, size_t bytes);

// Sync + compute elapsed times + aggregate
void cuda_profiler_sync_and_analyze(cuda_profiler_t* prof, void* stream);

// Print results (writes to stdout + log_file if set)
void cuda_profiler_print(cuda_profiler_t* prof);
void cuda_profiler_print_top(cuda_profiler_t* prof, int top_n);

// Set log file — profiler output is tee'd to this file alongside stdout
void cuda_profiler_set_log(cuda_profiler_t* prof, void* file_ptr);

#endif // CUDA_PROFILER_H
