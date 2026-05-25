// tensor_snapshot.cu - Binary tensor snapshot dumper implementation
// Part of TETYAH-PRIME's native training and inference engine

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>

extern "C" {
#include "../include/tensor_snapshot.h"
#include "../include/cuda_train_context.h"
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

extern "C" void snapshot_init(tensor_snapshot_config_t* cfg, const char* output_dir,
                               int step_interval, uint32_t flags) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    strncpy(cfg->output_dir, output_dir, sizeof(cfg->output_dir) - 1);
    cfg->step_interval = step_interval;
    cfg->capture_flags = flags;
    cfg->enabled = 1;

    // Create output directory
    mkdir(cfg->output_dir, 0755);

    printf("  Tensor snapshots: %s (every %d steps, flags=0x%x)\n",
           cfg->output_dir, step_interval, flags);
}

extern "C" int snapshot_should_capture(tensor_snapshot_config_t* cfg, int step) {
    if (!cfg || !cfg->enabled) return 0;
    return (step % cfg->step_interval == 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU TENSOR DOWNLOAD + WRITE
// ═══════════════════════════════════════════════════════════════════════════

extern "C" void snapshot_write_gpu_tensor(tensor_snapshot_config_t* cfg, void* ctx_ptr, void* buf_ptr,
                                           int step, int layer, const char* name,
                                           int rows, int cols) {
    if (!cfg || !cfg->enabled || !ctx_ptr || !buf_ptr) return;

    (void)ctx_ptr; /* available for cudaMemcpyAsync with ctx->stream if needed */
    cuda_buffer_t* buf = (cuda_buffer_t*)buf_ptr;

    size_t count = (size_t)rows * (size_t)cols;
    float* host_data = (float*)malloc(count * sizeof(float));
    if (!host_data) return;

    // Download from GPU
    cudaMemcpy(host_data, buf->data, count * sizeof(float), cudaMemcpyDeviceToHost);

    // Write to file
    snapshot_write_cpu_tensor(cfg, host_data, step, layer, name, rows, cols);
    free(host_data);
}

extern "C" void snapshot_write_cpu_tensor(tensor_snapshot_config_t* cfg, const float* data,
                                           int step, int layer, const char* name,
                                           int rows, int cols) {
    if (!cfg || !cfg->enabled || !data || !name) return;

    // Build filename: output_dir/step_NNNNNN_layer_NN_name.bin
    char filename[512];
    if (layer >= 0) {
        snprintf(filename, sizeof(filename), "%s/step_%06d_L%02d_%s.bin",
                 cfg->output_dir, step, layer, name);
    } else {
        snprintf(filename, sizeof(filename), "%s/step_%06d_%s.bin",
                 cfg->output_dir, step, name);
    }

    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "[snapshot] failed to open %s\n", filename);
        return;
    }

    // Write header
    snapshot_header_t hdr;
    memcpy(hdr.magic, "SRPH", 4);
    hdr.version = 1;
    hdr.step = (uint32_t)step;
    hdr.layer = (int32_t)layer;
    hdr.rows = (uint32_t)rows;
    hdr.cols = (uint32_t)cols;
    int name_len = (int)strlen(name);
    hdr.name_len = (uint32_t)name_len;

    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(name, 1, name_len, f);
    fwrite(data, sizeof(float), (size_t)rows * (size_t)cols, f);
    fclose(f);
}

// ═══════════════════════════════════════════════════════════════════════════
// MANIFEST (JSON index for web viewer)
// ═══════════════════════════════════════════════════════════════════════════

extern "C" void snapshot_write_manifest(tensor_snapshot_config_t* cfg, int step,
                                         int num_layers, int hidden_size, int num_heads,
                                         int head_dim, int vocab_size, float loss) {
    if (!cfg || !cfg->enabled) return;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/step_%06d_manifest.json", cfg->output_dir, step);

    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"step\": %d,\n", step);
    fprintf(f, "  \"loss\": %.6f,\n", loss);
    fprintf(f, "  \"num_layers\": %d,\n", num_layers);
    fprintf(f, "  \"hidden_size\": %d,\n", hidden_size);
    fprintf(f, "  \"num_heads\": %d,\n", num_heads);
    fprintf(f, "  \"head_dim\": %d,\n", head_dim);
    fprintf(f, "  \"vocab_size\": %d,\n", vocab_size);
    fprintf(f, "  \"capture_flags\": %u,\n", cfg->capture_flags);
    fprintf(f, "  \"tensors\": [\n");

    int first = 1;
    // List global tensors
    const char* global_tensors[] = {"logits", "loss_per_token", "hidden_final", NULL};
    for (int i = 0; global_tensors[i]; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/step_%06d_%s.bin", cfg->output_dir, step, global_tensors[i]);
        FILE* check = fopen(path, "r");
        if (check) {
            fclose(check);
            if (!first) fprintf(f, ",\n");
            fprintf(f, "    {\"name\": \"%s\", \"layer\": -1, \"file\": \"step_%06d_%s.bin\"}",
                    global_tensors[i], step, global_tensors[i]);
            first = 0;
        }
    }

    // List per-layer tensors
    const char* layer_tensors[] = {"attn_scores", "attn_weights", "hidden", "grad_hidden",
                                    "ffn_gate", "ffn_up", NULL};
    for (int l = 0; l < num_layers; l++) {
        for (int i = 0; layer_tensors[i]; i++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/step_%06d_L%02d_%s.bin",
                     cfg->output_dir, step, l, layer_tensors[i]);
            FILE* check = fopen(path, "r");
            if (check) {
                fclose(check);
                if (!first) fprintf(f, ",\n");
                fprintf(f, "    {\"name\": \"%s\", \"layer\": %d, \"file\": \"step_%06d_L%02d_%s.bin\"}",
                        layer_tensors[i], l, step, l, layer_tensors[i]);
                first = 0;
            }
        }
    }

    fprintf(f, "\n  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}
