// Seraph Band Monitor
// Standalone band-dynamics monitor with equation-derived topology walls
//
// Usage:
//   tail -f checkpoints/train.log | ./seraph-band-monitor models

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include "../include/cJSON.h"

#define MAX_WALLS 256
#define MAX_BANDS (MAX_WALLS + 2)
#define MAX_RECENT 64
#define LINE_BUF 4096
#define WALL_CLUSTER_EPS 0.02f
#define EPS 1e-9f
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ANSI color codes (matching topology monitor palette)
#define C_RESET   "\033[0m"
#define C_RED     "\033[38;5;196m"
#define C_GREEN   "\033[38;5;46m"
#define C_YELLOW  "\033[38;5;226m"
#define C_CYAN    "\033[38;5;51m"
#define C_MAGENTA "\033[38;5;201m"
#define C_WHITE   "\033[1;37m"
#define C_DIM     "\033[38;5;248m"
#define C_BOLD    "\033[1m"

#define W 78

typedef struct {
    float walls[MAX_WALLS];
    int num_walls;
    int num_bands;
} band_map_t;

typedef struct {
    int hidden_size;
    int num_heads;
    int head_dim;
    int intermediate_size;
} model_topology_t;

typedef struct {
    int band_id;
    int entry_step;
    int exit_step;
    float entry_loss;
    float exit_loss;
    int dwell_steps;
    float loss_delta;
    float absorption_rate;
    float band_resistance;
} band_segment_t;

typedef struct {
    int visits;
    long long dwell_steps;
    double loss_delta;
} band_aggregate_t;

typedef struct {
    int active;
    int band_id;
    int entry_step;
    float entry_loss;
    int last_step;
    float last_loss;
} live_segment_t;

typedef struct {
    float loss;
    float x;
} crossing_root_t;

static int cmp_root_loss_desc(const void* a, const void* b) {
    const crossing_root_t* ra = (const crossing_root_t*)a;
    const crossing_root_t* rb = (const crossing_root_t*)b;
    if (rb->loss > ra->loss) return 1;
    if (rb->loss < ra->loss) return -1;
    return 0;
}

static int load_model_topology(const char* path, model_topology_t* topo) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return -2;
    }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return -3;
    }
    buf[len] = '\0';
    fclose(f);

    cJSON* root = cJSON_Parse(buf);
    free(buf);
    if (!root) return -4;

    cJSON* item = cJSON_GetObjectItem(root, "hidden_size");
    topo->hidden_size = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "num_attention_heads");
    topo->num_heads = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "head_dim");
    topo->head_dim = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "intermediate_size");
    topo->intermediate_size = item ? item->valueint : 0;

    cJSON_Delete(root);

    if (topo->hidden_size <= 0 || topo->num_heads <= 0) return -5;
    if (topo->head_dim <= 0) topo->head_dim = topo->hidden_size / topo->num_heads;
    return 0;
}

static int compute_walls_from_equation(const model_topology_t* topo, band_map_t* map) {
    float z = (float)topo->hidden_size;
    float h = (float)topo->num_heads;
    float max_loss = logf(z);

    int raw_cap = 512;
    int raw_count = 0;
    crossing_root_t* raw = (crossing_root_t*)malloc((size_t)raw_cap * sizeof(crossing_root_t));
    if (!raw) return -1;

    typedef struct { float freq; int is_cos; float theta_max; } curve_t;
    curve_t curves[4] = {
        {h,         0, 2.0f * (float)M_PI},
        {h,         1, 2.0f * (float)M_PI},
        {h / 2.0f,  0, (float)M_PI},
        {h / 4.0f,  0, 4.0f * (float)M_PI},
    };
    int num_curves = (h >= 4.0f) ? 4 : (h >= 2.0f) ? 3 : 2;
    float dtheta = 0.00001f;

    for (int ci = 0; ci < num_curves; ci++) {
        float freq = curves[ci].freq;
        int is_cos = curves[ci].is_cos;
        float theta_max = curves[ci].theta_max;
        float prev_f = 0.0f;
        float prev_theta = 0.0f;
        int prev_valid = 0;

        for (float theta = 0.001f; theta < theta_max; theta += dtheta) {
            float R = is_cos ? cosf(freq * theta) : sinf(freq * theta);
            float log_arg = R * cosf(theta);
            float y_rose = z * R * sinf(theta);

            if (log_arg <= 0.0f || log_arg > 1.0f || y_rose < 0.0f) {
                prev_valid = 0;
                continue;
            }

            float f = z * R * sinf(theta) + logf(log_arg);
            if (prev_valid && ((prev_f > 0.0f && f < 0.0f) || (prev_f < 0.0f && f > 0.0f))) {
                float t = prev_f / (prev_f - f);
                float theta_root = prev_theta + t * dtheta;
                float R_root = is_cos ? cosf(freq * theta_root) : sinf(freq * theta_root);
                float x_root = (z * R_root) * cosf(theta_root);
                float loss_root = -logf(x_root / z);

                if (x_root > 0.0f && x_root < z && loss_root > 0.0f && loss_root <= max_loss) {
                    if (raw_count >= raw_cap) {
                        raw_cap *= 2;
                        crossing_root_t* grown =
                            (crossing_root_t*)realloc(raw, (size_t)raw_cap * sizeof(crossing_root_t));
                        if (!grown) {
                            free(raw);
                            return -2;
                        }
                        raw = grown;
                    }
                    raw[raw_count].loss = loss_root;
                    raw[raw_count].x = x_root;
                    raw_count++;
                }
            }

            prev_f = f;
            prev_theta = theta;
            prev_valid = 1;
        }
    }

    if (raw_count <= 0) {
        free(raw);
        return -3;
    }

    qsort(raw, (size_t)raw_count, sizeof(crossing_root_t), cmp_root_loss_desc);

    map->num_walls = 0;
    map->walls[map->num_walls++] = raw[0].loss;
    for (int i = 1; i < raw_count && map->num_walls < MAX_WALLS; i++) {
        if ((raw[i - 1].loss - raw[i].loss) > WALL_CLUSTER_EPS) {
            map->walls[map->num_walls++] = raw[i].loss;
        }
    }

    free(raw);

    if (map->num_walls <= 0) return -4;
    map->num_bands = map->num_walls;
    return 0;
}

static int parse_train_line(const char* line, int* out_step, float* out_loss,
                            float* out_avg, float* out_tok_s) {
    int step = 0, total = 0, epoch = 0;
    float loss = 0.0f, avg = 0.0f, tok_s = 0.0f;

    int n = sscanf(line, "  Step %d/%d | Loss: %f | Avg: %f | %f tok/s",
                   &step, &total, &loss, &avg, &tok_s);
    if (n >= 3) {
        *out_step = step;
        *out_loss = loss;
        *out_avg = (n >= 4) ? avg : loss;
        *out_tok_s = (n >= 5) ? tok_s : 0.0f;
        return 1;
    }

    n = sscanf(line, "Step %d/%d | Loss: %f | Avg: %f | %f tok/s",
               &step, &total, &loss, &avg, &tok_s);
    if (n >= 3) {
        *out_step = step;
        *out_loss = loss;
        *out_avg = (n >= 4) ? avg : loss;
        *out_tok_s = (n >= 5) ? tok_s : 0.0f;
        return 1;
    }

    n = sscanf(line, "  Step %d/%d | Loss: %f | %f tok/s",
               &step, &total, &loss, &tok_s);
    if (n >= 3) {
        *out_step = step;
        *out_loss = loss;
        *out_avg = loss;
        *out_tok_s = (n >= 4) ? tok_s : 0.0f;
        return 1;
    }

    n = sscanf(line, "epoch=%d step=%d loss=%f avg=%f tok_s=%f",
               &epoch, &step, &loss, &avg, &tok_s);
    if (n >= 3) {
        *out_step = step;
        *out_loss = loss;
        *out_avg = (n >= 4) ? avg : loss;
        *out_tok_s = (n >= 5) ? tok_s : 0.0f;
        return 1;
    }

    return 0;
}

static int locate_band(const band_map_t* map, float loss) {
    // Band: crossing wall N moves from band N-1 to band N.
    // While loss is still above wall N, we remain in band N-1.
    if (map->num_walls <= 0) return 1;
    if (loss >= map->walls[0]) return 1;
    for (int i = 1; i < map->num_walls; i++) {
        if (loss >= map->walls[i]) return i;
    }
    return map->num_walls;
}

static void clear_screen(void) {
    printf("\033[2J\033[H");
}

static void print_box_top(void) {
    printf(C_RED "╔");
    for (int i = 0; i < W; i++) printf("═");
    printf("╗\n" C_RESET);
}

static void print_box_mid(void) {
    printf(C_WHITE "╠");
    for (int i = 0; i < W; i++) printf("═");
    printf("╣\n" C_RESET);
}

static void print_box_bottom(void) {
    printf(C_RED "╚");
    for (int i = 0; i < W; i++) printf("═");
    printf("╝" C_RESET "\n");
}

static void pline(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    int visible = 0;
    for (int i = 0; buf[i]; i++) {
        if (buf[i] == '\033') {
            while (buf[i] && buf[i] != 'm') i++;
        } else if ((unsigned char)buf[i] >= 0x80 && (unsigned char)buf[i] < 0xC0) {
            // UTF-8 continuation byte
        } else {
            visible++;
        }
    }

    printf(C_RED "║" C_RESET "%s", buf);
    for (int i = visible; i < W; i++) printf(" ");
    printf(C_RED "║" C_RESET "\n");
}

static int bar_fill(float value, float max_value, int width) {
    if (max_value < EPS) max_value = 1.0f;
    if (value < 0.0f) value = -value;
    float ratio = value / max_value;
    if (ratio > 1.0f) ratio = 1.0f;
    int fill = (int)lroundf(ratio * width);
    if (fill < 0) fill = 0;
    if (fill > width) fill = width;
    return fill;
}

static void fill_chars(char* out, size_t out_size, int count, char ch) {
    if (count < 0) count = 0;
    if ((size_t)count >= out_size) count = (int)out_size - 1;
    for (int i = 0; i < count; i++) out[i] = ch;
    out[count] = '\0';
}

static void append_recent_segment(band_segment_t* recent, int* recent_count,
                                  const band_segment_t* seg) {
    if (*recent_count < MAX_RECENT) {
        recent[*recent_count] = *seg;
        (*recent_count)++;
        return;
    }

    memmove(&recent[0], &recent[1], sizeof(band_segment_t) * (MAX_RECENT - 1));
    recent[MAX_RECENT - 1] = *seg;
}

static void close_live_segment(live_segment_t* live, int exit_step, float exit_loss,
                               band_segment_t* recent, int* recent_count,
                               band_aggregate_t* agg, int agg_count) {
    if (!live->active) return;

    int dwell = exit_step - live->entry_step;
    if (dwell < 1) dwell = 1;

    float delta = live->entry_loss - exit_loss;

    float absorption = delta / (float)dwell;
    float resistance = (fabsf(delta) > EPS) ? ((float)dwell / delta) : 0.0f;

    band_segment_t seg;
    seg.band_id = live->band_id;
    seg.entry_step = live->entry_step;
    seg.exit_step = exit_step;
    seg.entry_loss = live->entry_loss;
    seg.exit_loss = exit_loss;
    seg.dwell_steps = dwell;
    seg.loss_delta = delta;
    seg.absorption_rate = absorption;
    seg.band_resistance = resistance;

    append_recent_segment(recent, recent_count, &seg);

    if (live->band_id >= 0 && live->band_id < agg_count) {
        agg[live->band_id].visits += 1;
        agg[live->band_id].dwell_steps += dwell;
        agg[live->band_id].loss_delta += delta;
    }

    live->active = 0;
}

static void start_live_segment(live_segment_t* live, int band_id, int step, float loss) {
    live->active = 1;
    live->band_id = band_id;
    live->entry_step = step;
    live->entry_loss = loss;
    live->last_step = step;
    live->last_loss = loss;
}

static void display_monitor(const band_map_t* map, const live_segment_t* live,
                            const band_segment_t* recent, int recent_count,
                            const band_aggregate_t* agg,
                            int step, float loss, float avg, float tok_s,
                            const char* source_label) {
    (void)step;
    (void)avg;
    (void)tok_s;
    (void)source_label;
    clear_screen();

    int dwell = step - live->entry_step;
    if (dwell < 0) dwell = 0;
    float delta = live->entry_loss - loss;
    float absorption = (dwell > 0) ? (delta / (float)dwell) : 0.0f;
    float resistance = (fabsf(delta) > EPS) ? ((float)dwell / delta) : 0.0f;

    float max_dwell = (float)dwell;
    float max_absorption = fabsf(absorption);
    float max_resistance = fabsf(resistance);
    for (int i = 0; i < recent_count; i++) {
        if (recent[i].dwell_steps > max_dwell) max_dwell = (float)recent[i].dwell_steps;
        if (fabsf(recent[i].absorption_rate) > max_absorption) {
            max_absorption = fabsf(recent[i].absorption_rate);
        }
        if (fabsf(recent[i].band_resistance) > max_resistance) {
            max_resistance = fabsf(recent[i].band_resistance);
        }
    }
    if (max_dwell < 1.0f) max_dwell = 1.0f;
    if (max_absorption < 1e-4f) max_absorption = 1e-4f;
    if (max_resistance < 1e-3f) max_resistance = 1e-3f;

    // Smooth, persistent scales for bars
    static float ui_scale_dwell = 20.0f;
    static float ui_scale_absorption = 0.02f;
    static float ui_scale_resistance = 10.0f;

    if (max_dwell > ui_scale_dwell) ui_scale_dwell = max_dwell;
    else ui_scale_dwell = fmaxf(max_dwell * 1.10f, ui_scale_dwell * 0.985f);

    if (max_absorption > ui_scale_absorption) ui_scale_absorption = max_absorption;
    else ui_scale_absorption = fmaxf(max_absorption * 1.15f, ui_scale_absorption * 0.985f);

    if (max_resistance > ui_scale_resistance) ui_scale_resistance = max_resistance;
    else ui_scale_resistance = fmaxf(max_resistance * 1.15f, ui_scale_resistance * 0.985f);

    int recent_rows = 7;

    int bar_w = 44;
    char fill[64], empty[64];
    int current_band = live->active ? live->band_id : 0;

    print_box_top();

    int f = bar_fill((float)dwell, ui_scale_dwell, bar_w);
    fill_chars(fill, sizeof(fill), f, '#');
    fill_chars(empty, sizeof(empty), bar_w - f, '.');
    pline("  " C_CYAN "%-16s" C_RESET " [" C_GREEN "%s" C_DIM "%s" C_RESET "] %.6f",
          "dwell_steps", fill, empty, (float)dwell);

    f = bar_fill(absorption, ui_scale_absorption, bar_w);
    fill_chars(fill, sizeof(fill), f, '#');
    fill_chars(empty, sizeof(empty), bar_w - f, '.');
    pline("  " C_CYAN "%-16s" C_RESET " [" C_YELLOW "%s" C_DIM "%s" C_RESET "] %.6f",
          "absorption_rate", fill, empty, fabsf(absorption));

    f = bar_fill(resistance, ui_scale_resistance, bar_w);
    fill_chars(fill, sizeof(fill), f, '#');
    fill_chars(empty, sizeof(empty), bar_w - f, '.');
    pline("  " C_CYAN "%-16s" C_RESET " [" C_MAGENTA "%s" C_DIM "%s" C_RESET "] %.6f",
          "band_resistance", fill, empty, fabsf(resistance));

    print_box_mid();
    pline("  " C_BOLD "Recent closed segments (latest first)" C_RESET);
    pline("  " C_DIM "band  steps(entry->exit)  loss(entry->exit)  dwell   absorb     resist" C_RESET);
    pline("  " C_DIM "----  ------------------  -----------------  ------  ---------  ---------" C_RESET);

    int shown = 0;
    for (int i = recent_count - 1; i >= 0 && shown < recent_rows; i--, shown++) {
        const band_segment_t* s = &recent[i];
        const char* row_color = C_DIM;
        if (current_band > 0) {
            if (s->band_id < current_band) row_color = C_GREEN;
            else if (s->band_id == current_band) row_color = C_RED C_BOLD;
        }
        pline(" " C_RESET "%s%4d  %7d -> %-7d  %7.3f -> %-7.3f  %6d  %9.6f  %9.4f" C_RESET,
              row_color, s->band_id, s->entry_step, s->exit_step,
              s->entry_loss, s->exit_loss, s->dwell_steps,
              s->absorption_rate, s->band_resistance);
    }
    if (shown == 0) {
        pline(" " C_DIM "(no closed segments yet)" C_RESET);
    }

    print_box_mid();
    pline("  " C_BOLD "Band aggregates (cumulative)" C_RESET);
    pline("  " C_DIM "band  visits   dwell_steps   delta_loss   absorption_rate   resistance" C_RESET);
    pline("  " C_DIM "----  ------   -----------   ----------   ---------------   ----------" C_RESET);
    for (int b = 1; b <= map->num_bands; b++) {
        double dloss = agg[b].loss_delta;
        long long ds = agg[b].dwell_steps;
        double a_rate = (ds > 0) ? (dloss / (double)ds) : 0.0;
        double b_res = (fabs(dloss) > EPS) ? ((double)ds / dloss) : 0.0;
        const char* row_color = C_DIM;
        if (current_band > 0) {
            if (b < current_band) row_color = C_GREEN;
            else if (b == current_band) row_color = C_RED C_BOLD;
        }
        pline(" " C_RESET "%s%4d  %6d   %11lld   %10.4f   %15.8f   %10.4f" C_RESET,
              row_color, b, agg[b].visits, ds, dloss, a_rate, b_res);
    }

    print_box_bottom();
    fflush(stdout);
}

static void print_usage(const char* argv0) {
    printf("\n");
    printf("Seraph Band Monitor\n");
    printf("\n");
    printf("Usage:\n");
    printf("  tail -f checkpoints/train.log | %s <model_dir>\n", argv0);
    printf("\n");
    printf("Inputs:\n");
    printf("  model_dir      model directory containing config.json\n");
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    const char* model_path = argv[1];
    char config_path[512];
    char source_label[256];
    source_label[0] = '\0';
    config_path[0] = '\0';

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    snprintf(config_path, sizeof(config_path), "%s/config.json", model_path);
    FILE* cfg = fopen(config_path, "r");
    if (!cfg) {
        fprintf(stderr, "Could not find config file: %s\n", config_path);
        return 1;
    }
    fclose(cfg);

    model_topology_t topo;
    memset(&topo, 0, sizeof(topo));
    int lrc = load_model_topology(config_path, &topo);
    if (lrc != 0) {
        fprintf(stderr, "Failed to load topology config from %s (rc=%d)\n", config_path, lrc);
        return 1;
    }
    snprintf(source_label, sizeof(source_label), "equation (%s)", config_path);

    band_map_t map;
    memset(&map, 0, sizeof(map));
    int rc = compute_walls_from_equation(&topo, &map);
    if (rc != 0) {
        fprintf(stderr, "Failed to compute walls from equations (rc=%d)\n", rc);
        return 1;
    }

    if (isatty(STDIN_FILENO)) {
        fprintf(stderr, "No pipe detected. Use: tail -f checkpoints/train.log | %s <model_dir>\n",
                argv[0]);
        return 1;
    }
    FILE* in = stdin;

    live_segment_t live;
    memset(&live, 0, sizeof(live));

    band_segment_t recent[MAX_RECENT];
    int recent_count = 0;

    band_aggregate_t agg[MAX_BANDS];
    memset(agg, 0, sizeof(agg));

    char line[LINE_BUF];
    int last_step = -1;
    while (fgets(line, sizeof(line), in)) {
        int step = 0;
        float loss = 0.0f;
        float avg = 0.0f;
        float tok_s = 0.0f;

        if (!parse_train_line(line, &step, &loss, &avg, &tok_s)) {
            continue;
        }
        if (step <= 0) continue;

        int band_id = locate_band(&map, loss);

        if (!live.active) {
            start_live_segment(&live, band_id, step, loss);
        } else if (band_id != live.band_id) {
            close_live_segment(&live, step, loss, recent, &recent_count, agg, MAX_BANDS);
            start_live_segment(&live, band_id, step, loss);
        } else {
            live.last_step = step;
            live.last_loss = loss;
        }

        if (step != last_step) {
            display_monitor(&map, &live, recent, recent_count, agg, step, loss, avg, tok_s,
                            source_label);
            last_step = step;
        }
    }
    return 0;
}
