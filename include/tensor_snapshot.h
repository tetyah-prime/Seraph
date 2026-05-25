// tensor_snapshot.h - Binary tensor snapshot dumper for visual profiling
// Dumps GPU tensors to files for the Seraph web tensor viewer
// Part of TETYAH-PRIME's native training and inference engine
//
// Binary format per file:
//   4 bytes: magic "SRPH"
//   4 bytes: version (1)
//   4 bytes: step number
//   4 bytes: layer index (-1 for global)
//   4 bytes: rows
//   4 bytes: cols
//   4 bytes: name_len
//   name_len bytes: tensor name (UTF-8, no null)
//   rows * cols * 4 bytes: float32 data (row-major)

#ifndef TENSOR_SNAPSHOT_H
#define TENSOR_SNAPSHOT_H

#include <stdint.h>
#include <stddef.h>

// Snapshot configuration
typedef struct {
    char output_dir[256];   // Directory to write snapshots
    int enabled;            // 1 = active
    int step_interval;      // Dump every N steps (e.g., 10)
    int current_step;       // Updated by training loop

    // Which tensors to capture (bitfield)
    uint32_t capture_flags;
} tensor_snapshot_config_t;

// Capture flags
#define SNAP_ATTENTION_SCORES   (1u << 0)   // Q@K^T per head (before softmax)
#define SNAP_ATTENTION_WEIGHTS  (1u << 1)   // Softmax output per head
#define SNAP_HIDDEN_STATES      (1u << 2)   // Hidden state after each layer
#define SNAP_GRADIENTS          (1u << 3)   // Gradient magnitudes per layer
#define SNAP_WEIGHT_NORMS       (1u << 4)   // Per-layer weight matrix norms
#define SNAP_LOGITS             (1u << 5)   // Final logits (before softmax)
#define SNAP_LOSS_PER_TOKEN     (1u << 6)   // Loss per token position
#define SNAP_FFN_ACTIVATIONS    (1u << 7)   // Gate/up/down activations
#define SNAP_ALL                (0xFFFFFFFFu)

// File header
#pragma pack(push, 1)
typedef struct {
    char magic[4];          // "SRPH"
    uint32_t version;       // 1
    uint32_t step;
    int32_t layer;          // -1 for global tensors
    uint32_t rows;
    uint32_t cols;
    uint32_t name_len;
    // followed by: name[name_len] + data[rows*cols*4]
} snapshot_header_t;
#pragma pack(pop)

// Initialize snapshot config
void snapshot_init(tensor_snapshot_config_t* cfg, const char* output_dir,
                   int step_interval, uint32_t flags);

// Check if we should snapshot this step
int snapshot_should_capture(tensor_snapshot_config_t* cfg, int step);

// Write a tensor snapshot to disk (downloads from GPU, writes file)
// ctx is cuda_train_context_t*, buf is cuda_buffer_t*
void snapshot_write_gpu_tensor(tensor_snapshot_config_t* cfg, void* ctx, void* buf,
                               int step, int layer, const char* name,
                               int rows, int cols);

// Write a CPU tensor snapshot to disk
void snapshot_write_cpu_tensor(tensor_snapshot_config_t* cfg, const float* data,
                               int step, int layer, const char* name,
                               int rows, int cols);

// Write a summary manifest (JSON index of all snapshots for this step)
void snapshot_write_manifest(tensor_snapshot_config_t* cfg, int step,
                              int num_layers, int hidden_size, int num_heads,
                              int head_dim, int vocab_size, float loss);

#endif // TENSOR_SNAPSHOT_H
