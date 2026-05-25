// Seraph Topology Monitor — Electromagnetic Topology Visualizer
// Reads training log from stdin and maps loss to petal topology
// Usage: tail -f checkpoints/train.log | ./seraph-topology-monitor <model_dir>
//   or:  ./seraph-topology-monitor --state checkpoint.state <model_dir>
//
// Zero GPU overhead — pure math from precomputed topology

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>
#include "../include/cJSON.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Petal intersection computation
// ═══════════════════════════════════════════════════════════════════════════

typedef struct {
    float x;        // x coordinate on loss curve
    float loss;     // y = -ln(x/z)
    int ring;       // which petal ring (0 = innermost)
} petal_crossing_t;

typedef struct {
    int hidden_size;        // z
    int num_heads;          // h
    int head_dim;
    int intermediate_size;
    float max_loss;         // ln(z)
    float scale;            // 1/sqrt(head_dim)

    petal_crossing_t* crossings;
    int num_crossings;
} topology_t;

// Find all points where RoPE rose curves intersect the loss curve y = -ln(x/z)
//
// RoPE equations (polar):
//   r = z·sin(h·θ)       (eq 46)
//   r = z·cos(h·θ)       (eq 47)
//   r = z·sin((h/2)·θ)   (eq 48)
//   r = z·sin((h/4)·θ)   (eq 49)
//
// Polar to Cartesian: x = r·cos(θ), y = r·sin(θ)
// Loss curve: y = -ln(x/z)
//
// At intersection: r·sin(θ) = -ln((r·cos(θ))/z)
//
// Substituting each RoPE form:
//   1. z·sin(h·θ)·sin(θ) = -ln(sin(h·θ)·cos(θ))
//   2. z·cos(h·θ)·sin(θ) = -ln(cos(h·θ)·cos(θ))
//   3. z·sin((h/2)·θ)·sin(θ) = -ln(sin((h/2)·θ)·cos(θ))
//   4. z·sin((h/4)·θ)·sin(θ) = -ln(sin((h/4)·θ)·cos(θ))
//
// Constraints:
//   - log input positive: R(θ)·cos(θ) > 0
//   - x ≤ z: R(θ)·cos(θ) ≤ 1
//   - positive loss: r·sin(θ) ≥ 0
//
// Solve f(θ) = z·R(θ)·sin(θ) + ln(R(θ)·cos(θ)) = 0
// Then: x = r·cos(θ), loss = -ln(x/z)

static void compute_petal_crossings(topology_t* topo) {
    float z = (float)topo->hidden_size;
    float h = (float)topo->num_heads;
    int capacity = 256;
    topo->crossings = malloc(capacity * sizeof(petal_crossing_t));
    topo->num_crossings = 0;

    // Curve definitions: {freq, is_cosine, theta_max}
    typedef struct { float freq; int is_cos; float theta_max; } curve_t;
    curve_t curves[4] = {
        {h,       0, 2.0f * M_PI},  // r = z·sin(h·θ)
        {h,       1, 2.0f * M_PI},  // r = z·cos(h·θ)
        {h / 2.0f, 0, M_PI},         // r = z·sin((h/2)·θ)
        {h / 4.0f, 0, 4.0f * M_PI},  // r = z·sin((h/4)·θ)
    };
    int num_curves = (h >= 4) ? 4 : (h >= 2) ? 3 : 2;

    float step = 0.00001f;  // Fine step for accuracy

    for (int ci = 0; ci < num_curves; ci++) {
        float freq = curves[ci].freq;
        int is_cos = curves[ci].is_cos;
        float theta_max = curves[ci].theta_max;
        float prev_f = 0;
        float prev_theta = 0;
        int prev_valid = 0;

        for (float theta = 0.001f; theta < theta_max; theta += step) {
            // R(θ) = sin(freq·θ) or cos(freq·θ)
            float R = is_cos ? cosf(freq * theta) : sinf(freq * theta);

            // Check constraints
            float log_arg = R * cosf(theta);
            float y_rose = z * R * sinf(theta);

            // Skip invalid: log_arg must be in (0, 1], y must be >= 0
            if (log_arg <= 0.0f || log_arg > 1.0f || y_rose < 0.0f) {
                prev_valid = 0;
                continue;
            }

            // f(θ) = z·R·sin(θ) + ln(R·cos(θ))
            // Root when f(θ) = 0, i.e., z·R·sin(θ) = -ln(R·cos(θ))
            float f = z * R * sinf(theta) + logf(log_arg);

            // Find sign change (root crossing)
            if (prev_valid && ((prev_f > 0 && f < 0) || (prev_f < 0 && f > 0))) {
                // Linear interpolation for θ at root
                float t = prev_f / (prev_f - f);
                float theta_root = prev_theta + t * step;

                // Compute exact values at root
                float R_root = is_cos ? cosf(freq * theta_root) : sinf(freq * theta_root);
                float r_root = z * R_root;
                float x_root = r_root * cosf(theta_root);
                float loss_root = -logf(x_root / z);

                // Validate
                if (x_root > 0.0f && x_root < z && loss_root > 0.0f && loss_root <= topo->max_loss) {
                    if (topo->num_crossings >= capacity) {
                        capacity *= 2;
                        topo->crossings = realloc(topo->crossings, capacity * sizeof(petal_crossing_t));
                    }
                    petal_crossing_t* c = &topo->crossings[topo->num_crossings];
                    c->x = x_root;
                    c->loss = loss_root;
                    c->ring = 0;
                    topo->num_crossings++;
                }
            }

            prev_f = f;
            prev_theta = theta;
            prev_valid = 1;
        }
    }

    // Sort by loss descending (highest loss = innermost)
    for (int i = 0; i < topo->num_crossings - 1; i++) {
        for (int j = i + 1; j < topo->num_crossings; j++) {
            if (topo->crossings[j].loss > topo->crossings[i].loss) {
                petal_crossing_t tmp = topo->crossings[i];
                topo->crossings[i] = topo->crossings[j];
                topo->crossings[j] = tmp;
            }
        }
    }

    // Assign wall numbers (crossings within 0.02 = same wall)
    int wall = 0;
    for (int i = 0; i < topo->num_crossings; i++) {
        if (i > 0 && (topo->crossings[i - 1].loss - topo->crossings[i].loss) > 0.02f) {
            wall++;
        }
        topo->crossings[i].ring = wall;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// State file reader
// ═══════════════════════════════════════════════════════════════════════════

static int read_state_file(const char* path, int* epoch, int* step, float* loss, int* opt_step) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fread(epoch, sizeof(int), 1, f);
    fread(step, sizeof(int), 1, f);
    fread(loss, sizeof(float), 1, f);
    fread(opt_step, sizeof(int), 1, f);
    fclose(f);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// Config reader (uses cJSON — same parser as rest of Seraph)
// ═══════════════════════════════════════════════════════════════════════════

static int load_config(const char* path, topology_t* topo) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);

    cJSON* root = cJSON_Parse(buf);
    free(buf);
    if (!root) return -1;

    cJSON* item;

    item = cJSON_GetObjectItem(root, "hidden_size");
    topo->hidden_size = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "num_attention_heads");
    topo->num_heads = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "intermediate_size");
    topo->intermediate_size = item ? item->valueint : 0;

    item = cJSON_GetObjectItem(root, "head_dim");
    topo->head_dim = item ? item->valueint : 0;

    cJSON_Delete(root);

    if (topo->hidden_size == 0 || topo->num_heads == 0) {
        fprintf(stderr, "ERROR: config missing hidden_size or num_attention_heads\n");
        return -1;
    }

    if (topo->head_dim == 0)
        topo->head_dim = topo->hidden_size / topo->num_heads;

    topo->max_loss = logf((float)topo->hidden_size);
    topo->scale = 1.0f / sqrtf((float)topo->head_dim);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// TUI Display
// ═══════════════════════════════════════════════════════════════════════════

static void clear_screen(void) {
    printf("\033[2J\033[H");
}

// ANSI color codes
#define C_RESET   "\033[0m"
#define C_RED     "\033[38;5;196m"
#define C_GREEN   "\033[38;5;46m"
#define C_YELLOW  "\033[38;5;226m"
#define C_CYAN    "\033[38;5;51m"
#define C_MAGENTA "\033[38;5;201m"
#define C_WHITE   "\033[1;37m"
#define C_DIM     "\033[38;5;248m"
#define C_BOLD    "\033[1m"

// Print a line padded to exactly W visible chars between ║ borders
#define W 70
static void pline(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    // Count visible columns (skip ANSI escapes, count UTF-8 codepoints not bytes)
    int visible = 0;
    for (int i = 0; buf[i]; i++) {
        if (buf[i] == '\033') {
            while (buf[i] && buf[i] != 'm') i++;
        } else if ((unsigned char)buf[i] >= 0x80 && (unsigned char)buf[i] < 0xC0) {
            // UTF-8 continuation byte — skip (already counted the leading byte)
        } else {
            visible++;
        }
    }

    printf(C_RED "║" C_RESET "%s", buf);
    for (int i = visible; i < W; i++) printf(" ");
    printf(C_RED "║" C_RESET "\n");
}

static void display_topology(topology_t* topo, int step, float loss, float avg_loss, float rate, float tok_s) {
    clear_screen();

    float z = (float)topo->hidden_size;

    // Find current ring
    int current_ring = -1;
    float next_wall = 0;
    int total_rings = 0;

    if (topo->num_crossings > 0) {
        total_rings = topo->crossings[topo->num_crossings - 1].ring + 1;
    }

    for (int i = 0; i < topo->num_crossings; i++) {
        if (loss >= topo->crossings[i].loss) {
            current_ring = (i > 0) ? topo->crossings[i - 1].ring : -1;
            next_wall = topo->crossings[i].loss;
            break;
        }
    }

    if (current_ring == -1 && topo->num_crossings > 0) {
        if (loss < topo->crossings[topo->num_crossings - 1].loss) {
            current_ring = total_rings;
            next_wall = 0;
        }
    }

    float dist_to_wall = loss - next_wall;
    float steps_to_wall = (rate != 0) ? (dist_to_wall / fabsf(rate)) * 100.0f : 0;
    float hours_to_wall = (tok_s > 0 && rate != 0) ? steps_to_wall / (tok_s * 3.6f) : 0;

    float x_pos = z * expf(-loss);
    float x_pct = (x_pos / z) * 100.0f;

    printf(C_RED "╔══════════════════════════════════════════════════════════════════════╗\n" C_RESET);
    pline("  " C_BOLD C_RED "SERAPH TOPOLOGY MONITOR" C_RESET);
    pline("  " C_DIM "Electromagnetic Topology Visualizer" C_RESET);
    printf(C_WHITE "╠══════════════════════════════════════════════════════════════════════╣\n" C_RESET);
    pline("");
    pline("  " C_CYAN "Config:" C_RESET " z=%d  h=%d  head_dim=%d  k=%.3f",
          topo->hidden_size, topo->num_heads, topo->head_dim,
          (float)topo->intermediate_size / (float)topo->hidden_size);
    pline("  " C_CYAN "Scale:" C_RESET "  s=%.4f  max_loss=%.3f (ln(z))",
          topo->scale, topo->max_loss);
    pline("");
    printf(C_WHITE "╠══════════════════════════════════════════════════════════════════════╣\n" C_RESET);
    pline("");
    pline("  " C_WHITE "Step:" C_RESET " %-8d  " C_YELLOW "Loss:" C_RESET " %-8.4f  " C_DIM "Avg:" C_RESET " %.4f",
          step, loss, avg_loss);
    pline("  " C_GREEN "Rate:" C_RESET " %+.5f/100", -fabsf(rate));
    pline("  " C_WHITE "Tok/s:" C_RESET " %-6.0f   " C_DIM "Position: x=%.1f (%.1f%% to convergence)" C_RESET,
          tok_s, x_pos, x_pct);
    pline("");

    // Ring status
    if (dist_to_wall > 0.01f) {
        pline("  " C_MAGENTA "Ring:" C_RESET " %d/%d   " C_GREEN "▼ Open space" C_RESET " (%.3f to next wall)",
              current_ring + 1, total_rings, dist_to_wall);
    } else {
        pline("  " C_MAGENTA "Ring:" C_RESET " %d/%d   " C_RED "◆ AT PETAL BOUNDARY" C_RESET " (absorbing)",
              current_ring + 1, total_rings);
    }

    if (next_wall > 0) {
        if (rate != 0) {
            pline("  " C_YELLOW "Next wall:" C_RESET " %.3f  (~%.0f steps, ~%.1f hrs)",
                  next_wall, steps_to_wall, hours_to_wall);
        } else {
            pline("  " C_YELLOW "Next wall:" C_RESET " %.3f", next_wall);
        }
    } else {
        pline("  " C_GREEN "Beyond all mapped petals" C_RESET " -- deep convergence");
    }
    pline("");

    // Progress bar
    printf(C_WHITE "╠══════════════════════════════════════════════════════════════════════╣\n" C_RESET);

    int bar_width = 66;
    float loss_range = topo->max_loss + 0.5f;

    printf(C_RED "║" C_RESET "  ");
    for (int i = 0; i < bar_width; i++) {
        float bar_loss = loss_range * (1.0f - (float)i / bar_width);
        int is_wall = 0;
        // Show wall marker only at first crossing of each wall cluster
        for (int j = 0; j < topo->num_crossings; j++) {
            // First crossing of this wall? (j==0 or different ring from previous)
            int is_first = (j == 0) || (topo->crossings[j].ring != topo->crossings[j-1].ring);
            if (is_first) {
                int ppos = (int)((1.0f - topo->crossings[j].loss / loss_range) * bar_width);
                if (ppos == i) { is_wall = 1; break; }
            }
        }
        int cur_pos = (int)((1.0f - loss / loss_range) * bar_width);

        if (i == cur_pos) {
            printf(C_RED C_BOLD "@" C_RESET);
        } else if (is_wall) {
            printf(C_YELLOW "|" C_RESET);
        } else if (bar_loss > loss) {
            printf(C_GREEN "-" C_RESET);
        } else {
            printf(C_DIM "-" C_RESET);
        }
    }
    printf("  " C_RED "║" C_RESET "\n");

    // Loss labels
    {
        char lbuf[16];
        int llen = snprintf(lbuf, sizeof(lbuf), "%.1f", topo->max_loss);
        int pad = bar_width - llen - 3;  // 3 for "0.0"
        printf(C_RED "║" C_RESET "  " C_DIM "%s", lbuf);
        for (int i = 0; i < pad; i++) printf(" ");
        printf("0.0" C_RESET "  " C_RED "║" C_RESET "\n");
    }

    pline("");

    // Waveform — scrolling seismograph of loss history
    printf(C_WHITE "╠══════════════════════════════════════════════════════════════════════╣\n" C_RESET);
    {
        #define WAVE_COLS 62  // waveform data columns
        #define WAVE_H 8
        #define WAVE_CAP (WAVE_COLS + 1)  // +1 for prev-point lookback
        static float wave_history[WAVE_CAP];
        static int wave_count = 0;

        // Push new loss value
        if (wave_count < WAVE_CAP) {
            wave_history[wave_count++] = loss;
        } else {
            for (int i = 0; i < WAVE_CAP - 1; i++)
                wave_history[i] = wave_history[i + 1];
            wave_history[WAVE_CAP - 1] = loss;
        }

        // Auto-zoom to visible range
        float wmin = wave_history[0], wmax = wave_history[0];
        for (int i = 1; i < wave_count; i++) {
            if (wave_history[i] < wmin) wmin = wave_history[i];
            if (wave_history[i] > wmax) wmax = wave_history[i];
        }
        if (wmax - wmin < 0.1f) {
            float mid = (wmax + wmin) / 2.0f;
            wmin = mid - 0.05f;
            wmax = mid + 0.05f;
        }
        float wrange = wmax - wmin;

        // Precompute Y positions for each history point
        int hist_start = (wave_count <= WAVE_COLS) ? 0 : wave_count - WAVE_COLS;
        int num_vis = (wave_count < WAVE_COLS) ? wave_count : WAVE_COLS;
        int col_offset = WAVE_COLS - num_vis;  // left padding

        int ypos[WAVE_CAP];
        for (int i = 0; i < wave_count; i++)
            ypos[i] = (int)((wmax - wave_history[i]) / wrange * (WAVE_H - 1) + 0.5f);

        // Y-axis labels
        char ytop[8], ybot[8];
        snprintf(ytop, sizeof(ytop), "%.2f", wmax);
        snprintf(ybot, sizeof(ybot), "%.2f", wmin);

        for (int row = 0; row < WAVE_H; row++) {
            // Left border + label (6 visible chars: space + 5-char label)
            printf(C_RED "║" C_RESET);
            if (row == 0)
                printf(" " C_DIM "%-5s" C_RESET, ytop);
            else if (row == WAVE_H - 1)
                printf(" " C_DIM "%-5s" C_RESET, ybot);
            else
                printf("      ");

            int visible = 6;  // label width

            for (int col = 0; col < WAVE_COLS; col++) {
                if (col < col_offset) {
                    printf(" ");
                    visible++;
                    continue;
                }

                int hi = hist_start + (col - col_offset);
                int cy = ypos[hi];
                int py = (hi > 0) ? ypos[hi - 1] : cy;  // previous point
                int is_last = (col - col_offset == num_vis - 1);

                // Check if this row is between prev and current (connected wave)
                int y_lo = (py < cy) ? py : cy;
                int y_hi = (py > cy) ? py : cy;
                int in_wave = (row >= y_lo && row <= y_hi);

                if (in_wave) {
                    // Check if at petal wall
                    float val = wave_history[hi];
                    int at_wall = 0;
                    for (int p = 0; p < topo->num_crossings; p++) {
                        if (fabsf(val - topo->crossings[p].loss) < 0.08f) {
                            at_wall = 1;
                            break;
                        }
                    }

                    // Check if next point is also at same Y (run of 2+)
                    int ny = (hi + 1 < wave_count) ? ypos[hi + 1] : -1;
                    int flat_run = (py == cy || ny == cy);

                    if (is_last && row == cy)
                        printf(C_RED C_BOLD "●" C_RESET);
                    else if (at_wall && row == cy)
                        printf(C_YELLOW "█" C_RESET);
                    else if (row == cy && flat_run)
                        printf(C_CYAN "─" C_RESET);
                    else if (row == cy)
                        printf(C_CYAN "│" C_RESET);
                    else
                        printf(C_CYAN "│" C_RESET);
                } else {
                    // Petal wall dotted line
                    int is_wall = 0;
                    for (int p = 0; p < topo->num_crossings; p++) {
                        int wy = (int)((wmax - topo->crossings[p].loss) / wrange * (WAVE_H - 1) + 0.5f);
                        if (wy == row && topo->crossings[p].loss >= wmin && topo->crossings[p].loss <= wmax) {
                            is_wall = 1;
                            break;
                        }
                    }
                    if (is_wall)
                        printf(C_DIM "·" C_RESET);
                    else
                        printf(" ");
                }
                visible++;
            }

            // Pad to W and close
            for (int i = visible; i < W; i++) printf(" ");
            printf(C_RED "║" C_RESET "\n");
        }
    }

    pline("");

    // Petal crossing table
    printf(C_WHITE "╠══════════════════════════════════════════════════════════════════════╣\n" C_RESET);
    pline("  " C_BOLD "PETAL CROSSINGS" C_RESET " (loss y=-ln(x/z) x RoPE curves)");
    pline("  " C_DIM "Ring   x        Loss     Status" C_RESET);
    pline("  " C_DIM "----   ------   ------   ------" C_RESET);

    int shown = 0;
    for (int i = 0; i < topo->num_crossings && shown < 20; i++) {
        petal_crossing_t* c = &topo->crossings[i];

        if (loss < c->loss) {
            pline("  " C_GREEN "%2d     %7.3f   %6.3f   ✓ passed" C_RESET,
                  c->ring + 1, c->x, c->loss);
        } else if (fabsf(loss - c->loss) < 0.15f) {
            pline("  " C_RED C_BOLD "%2d     %7.3f   %6.3f   ◆ HERE" C_RESET,
                  c->ring + 1, c->x, c->loss);
        } else {
            pline("  " C_DIM "%2d     %7.3f   %6.3f     ahead" C_RESET,
                  c->ring + 1, c->x, c->loss);
        }
        shown++;
    }

    pline("");
    printf(C_RED "╚══════════════════════════════════════════════════════════════════════╝" C_RESET);
    fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    topology_t topo = {0};
    const char* config_path = NULL;
    const char* state_path = NULL;

    if (argc < 2) {
        printf("\n");
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("  Seraph Topology Monitor\n");
        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("\n");
        printf("Usage: tail -f checkpoints/train.log | %s <model_dir>\n", argv[0]);
        printf("       %s --state checkpoint.state <model_dir>\n", argv[0]);
        printf("\n");
        printf("Maps training loss to electromagnetic topology petal crossings.\n");
        printf("Reads model config for z, h, head_dim and computes rose curve\n");
        printf("intersections to predict loss walls and convergence rings.\n");
        printf("\n");
        printf("Options:\n");
        printf("  --state PATH    Read loss from checkpoint .state file\n");
        printf("\n");
        return 1;
    }

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--state") == 0 && i + 1 < argc) {
            state_path = argv[++i];
        } else if (strstr(argv[i], ".json")) {
            config_path = argv[i];
        } else if (!config_path && argv[i][0] != '-') {
            // Assume it's a model directory — look for config.json inside
            char buf[512];
            snprintf(buf, sizeof(buf), "%s/config.json", argv[i]);
            FILE* test = fopen(buf, "r");
            if (test) {
                fclose(test);
                config_path = strdup(buf);
            } else {
                config_path = argv[i];
            }
        }
    }

    // Load config
    if (!config_path) {
        fprintf(stderr, "No config found. Pass a model directory or config.json\n");
        return 1;
    }
    if (load_config(config_path, &topo) != 0) {
        fprintf(stderr, "Failed to load config: %s\n", config_path);
        return 1;
    }

    compute_petal_crossings(&topo);

    // State file mode — single display
    if (state_path) {
        int epoch, step, opt_step;
        float loss;
        if (read_state_file(state_path, &epoch, &step, &loss, &opt_step) == 0) {
            display_topology(&topo, step, loss, loss, 0, 0);
        } else {
            fprintf(stderr, "Failed to read state: %s\n", state_path);
        }
        free(topo.crossings);
        return 0;
    }

    // Pipe mode — need stdin from pipe, not terminal
    if (isatty(STDIN_FILENO)) {
        printf("\nNo pipe detected. Pipe training log to monitor:\n");
        printf("  tail -f checkpoints/train.log | %s <model_dir>\n\n", argv[0]);
        free(topo.crossings);
        return 1;
    }

    char line[4096];
    float last_loss = 0;
    float rate = 0;
    int last_step = 0;
    float tok_s = 0;
    int first = 1;

    while (fgets(line, sizeof(line), stdin)) {
        // Parse: "  Step 10/297096 | Loss: 9.4213 | Avg: 9.4213 | PPL: 12348.9 | 2729 tok/s"
        int step = 0;
        int total = 0;
        float loss = 0;
        float avg = 0;
        float ppl = 0;
        float ts = 0;

        int parsed = sscanf(line, "  Step %d/%d | Loss: %f | Avg: %f | PPL: %f | %f tok/s",
                            &step, &total, &loss, &avg, &ppl, &ts);
        if (parsed < 3) {
            parsed = sscanf(line, "  Step %d/%d | Loss: %f | Avg: %f | %f tok/s",
                            &step, &total, &loss, &avg, &ts);
        }
        if (parsed < 3) {
            parsed = sscanf(line, "  Step %d/%d | Loss: %f | %f tok/s",
                            &step, &total, &loss, &ts);
        }
        if (parsed >= 3) {
            if (!first && step > last_step) {
                int step_diff = step - last_step;
                float loss_diff = last_loss - loss;
                rate = (loss_diff / step_diff) * 100.0f;  // per 100 steps
            }
            if (ts > 0) tok_s = ts;

            display_topology(&topo, step, loss, avg, rate, tok_s);

            last_loss = loss;
            last_step = step;
            first = 0;
        }
    }

    free(topo.crossings);
    return 0;
}
