// nan_dump_inspect.c - Inspect NaN dump files produced by seraph-train --debug-nan-check
//
// Usage:
//   ./seraph-nan-inspect --model-dir ../models/your-model --prefix /tmp/seraph_nan_step_61
//
// Prints shape info, basic stats, and first NaN/Inf index for each known dump file.

#include "../include/model_config.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t count;
    size_t nan_count;
    size_t inf_count;
    float min_v;
    float max_v;
    float max_abs;
    size_t first_nan_idx;
    size_t first_inf_idx;
    int has_nan;
    int has_inf;
} stats_f32_t;

static int file_size_bytes(const char* path, size_t* out) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    long n = ftell(f);
    fclose(f);
    if (n < 0) return 0;
    *out = (size_t)n;
    return 1;
}

static void* read_all(const char* path, size_t* out_bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    void* buf = malloc((size_t)n);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        free(buf);
        return NULL;
    }
    if (out_bytes) *out_bytes = (size_t)n;
    return buf;
}

static stats_f32_t stats_f32(const float* x, size_t n) {
    stats_f32_t st;
    memset(&st, 0, sizeof(st));
    st.count = n;
    st.min_v = INFINITY;
    st.max_v = -INFINITY;
    st.max_abs = 0.0f;
    st.first_nan_idx = (size_t)-1;
    st.first_inf_idx = (size_t)-1;

    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        if (isnan(v)) {
            st.nan_count++;
            if (!st.has_nan) {
                st.has_nan = 1;
                st.first_nan_idx = i;
            }
            continue;
        }
        if (isinf(v)) {
            st.inf_count++;
            if (!st.has_inf) {
                st.has_inf = 1;
                st.first_inf_idx = i;
            }
            continue;
        }
        if (v < st.min_v) st.min_v = v;
        if (v > st.max_v) st.max_v = v;
        float av = fabsf(v);
        if (av > st.max_abs) st.max_abs = av;
    }

    if (!isfinite(st.min_v)) st.min_v = 0.0f;
    if (!isfinite(st.max_v)) st.max_v = 0.0f;
    return st;
}

static void print_stats_line(const char* name, const char* path, stats_f32_t st) {
    printf("%-18s  n=%zu  nan=%zu  inf=%zu  min=%g  max=%g  maxabs=%g",
           name, st.count, st.nan_count, st.inf_count,
           (double)st.min_v, (double)st.max_v, (double)st.max_abs);
    if (st.has_nan) printf("  first_nan=%zu", st.first_nan_idx);
    if (st.has_inf) printf("  first_inf=%zu", st.first_inf_idx);
    printf("\n  %s\n", path);
}

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --model-dir <dir> --prefix <dump_prefix>\n"
            "\n"
            "Example:\n"
            "  %s --model-dir ../models/your-model --prefix /tmp/seraph_nan_step_61\n",
            prog, prog);
}

int main(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* prefix = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            prefix = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (!model_dir || !prefix) {
        usage(argv[0]);
        return 2;
    }

    model_config_t* cfg = model_config_load(model_dir);
    if (!cfg) {
        fprintf(stderr, "ERROR: failed to load model config from %s\n", model_dir);
        return 1;
    }

    char path_tokens[512];
    snprintf(path_tokens, sizeof(path_tokens), "%s_tokens.u32", prefix);

    size_t tok_bytes = 0;
    if (!file_size_bytes(path_tokens, &tok_bytes) || (tok_bytes % sizeof(uint32_t)) != 0) {
        fprintf(stderr, "ERROR: could not size tokens file: %s (%s)\n", path_tokens, strerror(errno));
        model_config_free(cfg);
        return 1;
    }
    size_t seq = tok_bytes / sizeof(uint32_t);
    size_t num_predictions = (seq > 0) ? (seq - 1) : 0;

    int hidden = cfg->hidden_size;
    int intermediate = cfg->intermediate_size;
    int vocab = cfg->vocab_size;
    int q_dim = cfg->num_attention_heads * cfg->head_dim;
    int kv_dim = cfg->kv_dim; // already computed in config loader

    printf("Dump prefix: %s\n", prefix);
    printf("Model dir:   %s\n", model_dir);
    printf("seq=%zu hidden=%d inter=%d vocab=%d q_dim=%d kv_dim=%d\n\n", seq, hidden, intermediate, vocab, q_dim, kv_dim);

    struct item {
        const char* name;
        const char* suffix;
        size_t expected_count;
        int required;
    } items[] = {
        {"hidden", "hidden.f32", seq * (size_t)hidden, 1},
        {"hidden_norm", "hidden_norm.f32", seq * (size_t)hidden, 1},
        {"logits", "logits.f32", seq * (size_t)vocab, 1},
        {"grad_logits", "grad_logits.f32", num_predictions * (size_t)vocab, 1},
        {"grad_hidden", "grad_hidden.f32", seq * (size_t)hidden, 1},
        {"grad_tmp_hidden", "grad_tmp_hidden.f32", seq * (size_t)hidden, 0},
        {"grad_x_norm", "grad_x_norm.f32", seq * (size_t)hidden, 0},
        {"grad_ffn", "grad_ffn.f32", seq * (size_t)intermediate, 0},
        {"grad_gate", "grad_gate.f32", seq * (size_t)intermediate, 0},
        {"grad_up", "grad_up.f32", seq * (size_t)intermediate, 0},
        {"L0_layer_in", "L0_layer_in.f32", seq * (size_t)hidden, 0},
        {"L0_x_norm_attn", "L0_x_norm_attn.f32", seq * (size_t)hidden, 0},
        {"L0_post_attn_in", "L0_post_attn_in.f32", seq * (size_t)hidden, 0},
        {"L0_x_norm_ffn", "L0_x_norm_ffn.f32", seq * (size_t)hidden, 0},
        {"L0_ffn_gate", "L0_ffn_gate.f32", seq * (size_t)intermediate, 0},
        {"L0_ffn_up", "L0_ffn_up.f32", seq * (size_t)intermediate, 0},
        {"L0_silu", "L0_silu.f32", seq * (size_t)intermediate, 0},
        {"L0_ffn_hidden", "L0_ffn_hidden.f32", seq * (size_t)intermediate, 0},
        {"L0_q", "L0_q.f32", seq * (size_t)q_dim, 0},
        {"L0_k", "L0_k.f32", seq * (size_t)kv_dim, 0},
        {"L0_v", "L0_v.f32", seq * (size_t)kv_dim, 0},
        {"L0_attn_out", "L0_attn_out.f32", seq * (size_t)q_dim, 0},
        {"grad_attn", "grad_attn.f32", seq * (size_t)q_dim, 0},
        {"grad_q", "grad_q.f32", seq * (size_t)q_dim, 0},
        {"grad_k", "grad_k.f32", seq * (size_t)kv_dim, 0},
        {"grad_v", "grad_v.f32", seq * (size_t)kv_dim, 0},
    };

    for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s_%s", prefix, items[i].suffix);
        size_t bytes = 0;
        if (!file_size_bytes(path, &bytes)) {
            if (items[i].required) {
                fprintf(stderr, "MISSING: %s\n  %s\n", items[i].name, path);
            }
            continue;
        }
        if ((bytes % sizeof(float)) != 0) {
            fprintf(stderr, "BAD SIZE: %s (bytes=%zu not multiple of 4)\n  %s\n", items[i].name, bytes, path);
            continue;
        }
        size_t count = bytes / sizeof(float);
        if (items[i].expected_count != 0 && count != items[i].expected_count) {
            fprintf(stderr, "WARN: %s count=%zu expected=%zu\n  %s\n", items[i].name, count, items[i].expected_count, path);
        }

        size_t read_bytes = 0;
        float* data = (float*)read_all(path, &read_bytes);
        if (!data || read_bytes != bytes) {
            fprintf(stderr, "ERROR: failed reading %s\n  %s\n", items[i].name, path);
            free(data);
            continue;
        }
        stats_f32_t st = stats_f32(data, count);
        print_stats_line(items[i].name, path, st);
        free(data);
        printf("\n");
    }

    model_config_free(cfg);
    return 0;
}
