/*
 * ============================================
 * SERAPH GRADIENT FIELD MONITOR
 * ============================================
 * Electromagnetic field visualization for neural network training
 * Renders gradient vectors sampled at solfeggio-harmonic intervals
 *
 * Theory: Jefimenko's retarded time field observation
 * - Weights = stationary lattice points in space
 * - Gradients = field vectors at those points
 * - CPU updates = observer changing the field
 *
 * Compile: via Makefile (links SDL2, GL, GLU, m)
 * ============================================
 */

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

// ============================================
// CONSTANTS
// ============================================

#define WINDOW_WIDTH  1280
#define WINDOW_HEIGHT 720
#define WINDOW_TITLE  "Seraph Gradient Field Monitor"

#define GRADIENT_SNAPSHOT_MAGIC 0x47524144  // "GRAD"
#define GRADIENT_SAMPLES_PER_TENSOR 528     // Solfeggio frequency

// Model architecture
#define TENSORS_PER_LAYER 9  // q, k, v, o, gate, up, down, input_norm, post_norm

// Tensor names for labeling
static const char* TENSOR_NAMES[TENSORS_PER_LAYER] = {
    "q_proj",      // Query projection (attention)
    "k_proj",      // Key projection (attention)
    "v_proj",      // Value projection (attention)
    "o_proj",      // Output projection (attention)
    "gate_proj",   // Gate projection (FFN)
    "up_proj",     // Up projection (FFN)
    "down_proj",   // Down projection (FFN)
    "input_norm",  // Input layer normalization
    "post_norm"    // Post-attention normalization
};

// Spatial layout
#define GRID_COLS 3          // 3x3 grid per layer
#define GRID_ROWS 3
#define LAYER_SPACING 2.0f   // Z-axis spacing between layers
#define TENSOR_SPACING 1.5f  // X-Y spacing between tensors
#define VECTOR_SCALE 0.3f    // Arrow length scale factor

// Camera defaults
#define CAMERA_DISTANCE 15.0f
#define CAMERA_ROTATION_SPEED 2.0f
#define CAMERA_ZOOM_SPEED 0.5f

// Playback defaults
#define PLAYBACK_FPS 10.0f

// ============================================
// DATA STRUCTURES
// ============================================

// Snapshot file header
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t samples_per_tensor;
    uint32_t num_layers;
} snapshot_header_t;

// Per-snapshot metadata
typedef struct {
    uint32_t step;
    uint32_t timestamp;
    float loss;
    float lr;
} snapshot_metadata_t;

// Gradient vector (3D position + magnitude + direction)
typedef struct {
    float x, y, z;          // Position in 3D space
    float magnitude;        // Gradient magnitude (for coloring)
    float dx, dy, dz;       // Direction (normalized)
} gradient_vector_t;

// Complete snapshot (metadata + all vectors)
typedef struct {
    snapshot_metadata_t meta;
    gradient_vector_t* vectors;
    int num_vectors;
    int valid;  // 1 if loaded successfully
} gradient_snapshot_t;

// Camera state
typedef struct {
    float distance;         // Distance from origin
    float rotation_y;       // Y-axis rotation (degrees)
    float rotation_x;       // X-axis rotation (degrees)
    float target_x, target_y, target_z;  // Look-at point
} camera_t;

// Playback state
typedef struct {
    gradient_snapshot_t* snapshots;  // Array of loaded snapshots
    int num_snapshots;
    int current_index;
    int playing;
    float playback_speed;   // Snapshots per second
    double last_frame_time;
    int num_layers;         // Model architecture (from snapshot header)
    int total_tensors;      // Calculated: 1 + num_layers * 9 + 1
    int total_vectors;      // Calculated: total_tensors * 528
} playback_state_t;

// Bitmap font system
typedef struct {
    GLuint texture;         // Font atlas texture
    int char_width;         // Width of each character
    int char_height;        // Height of each character
    int atlas_cols;         // Characters per row in atlas
    int atlas_rows;         // Rows in atlas
} bitmap_font_t;

// Application state
typedef struct {
    SDL_Window* window;
    SDL_GLContext gl_context;
    TTF_Font* font;         // Font for generating bitmap atlas
    bitmap_font_t bitmap_font;  // Bitmap font for rendering
    camera_t camera;
    playback_state_t playback;
    int running;
    int mode_realtime;      // 0 = replay, 1 = real-time
    char* watch_path;       // Directory to watch (real-time mode)
    FILE* snapshot_file;    // Open file handle (real-time mode)
    long last_file_pos;     // Last read position (real-time mode)
    int last_printed_step;  // Last step printed to console
    int show_labels;        // 1 = show tensor labels, 0 = hide
} app_state_t;

// ============================================
// MATH HELPERS
// ============================================

static inline float clamp(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

// EMF spectrum color mapping (magnitude -> RGB)
// Blue (low) -> Cyan -> Green -> Yellow -> Red (high)
static void emf_spectrum_color(float magnitude, float* r, float* g, float* b) {
    // Normalize magnitude to [0, 1] range
    // Log scale for better visualization of small gradients
    float t = log10f(1.0f + fabsf(magnitude) * 100.0f) / 3.0f;  // Range ~0-1
    t = clamp(t, 0.0f, 1.0f);

    if (t < 0.25f) {
        // Blue -> Cyan
        float s = t / 0.25f;
        *r = 0.0f;
        *g = s;
        *b = 1.0f;
    } else if (t < 0.5f) {
        // Cyan -> Green
        float s = (t - 0.25f) / 0.25f;
        *r = 0.0f;
        *g = 1.0f;
        *b = 1.0f - s;
    } else if (t < 0.75f) {
        // Green -> Yellow
        float s = (t - 0.5f) / 0.25f;
        *r = s;
        *g = 1.0f;
        *b = 0.0f;
    } else {
        // Yellow -> Red
        float s = (t - 0.75f) / 0.25f;
        *r = 1.0f;
        *g = 1.0f - s;
        *b = 0.0f;
    }
}

// ============================================
// 3D SPATIAL LAYOUT
// ============================================

// Map tensor index to 3D position
static void tensor_index_to_position(int tensor_idx, int num_layers, float* x, float* y, float* z) {
    // Layout:
    // - Z-axis: Layer depth (embed at z=-1, layers 0 to N-1, final_norm at z=N)
    // - X-Y plane: 3x3 grid per layer

    int total_tensors = 1 + num_layers * TENSORS_PER_LAYER + 1;

    if (tensor_idx == 0) {
        // embed_tokens: center of front plane
        *x = 0.0f;
        *y = 0.0f;
        *z = -LAYER_SPACING;
    } else if (tensor_idx == total_tensors - 1) {
        // final_norm: center of back plane
        *x = 0.0f;
        *y = 0.0f;
        *z = num_layers * LAYER_SPACING;
    } else {
        // Layer tensors: 3x3 grid
        int layer_tensor_idx = tensor_idx - 1;  // Skip embed_tokens
        int layer = layer_tensor_idx / TENSORS_PER_LAYER;
        int tensor_in_layer = layer_tensor_idx % TENSORS_PER_LAYER;

        int grid_x = tensor_in_layer % GRID_COLS;
        int grid_y = tensor_in_layer / GRID_COLS;

        *x = (1 - grid_x) * TENSOR_SPACING;  // Inverted: grid_x=0 → right, grid_x=2 → left
        *y = (grid_y - 1) * TENSOR_SPACING;  // grid_y=0 → bottom, grid_y=2 → top
        *z = layer * LAYER_SPACING;
    }
}

// Get tensor name for labeling
static void get_tensor_name(int tensor_idx, int num_layers, char* name_buf, size_t buf_size) {
    int total_tensors = 1 + num_layers * TENSORS_PER_LAYER + 1;

    if (tensor_idx == 0) {
        snprintf(name_buf, buf_size, "embed_tokens");
    } else if (tensor_idx == total_tensors - 1) {
        snprintf(name_buf, buf_size, "final_norm");
    } else {
        int layer_tensor_idx = tensor_idx - 1;
        int layer = layer_tensor_idx / TENSORS_PER_LAYER;
        int tensor_in_layer = layer_tensor_idx % TENSORS_PER_LAYER;
        snprintf(name_buf, buf_size, "layer_%d.%s", layer, TENSOR_NAMES[tensor_in_layer]);
    }
}

// ============================================
// SNAPSHOT FILE I/O
// ============================================

// Load snapshot header from file
static int load_snapshot_header(FILE* fp, snapshot_header_t* header) {
    rewind(fp);

    if (fread(&header->magic, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header->version, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header->samples_per_tensor, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header->num_layers, sizeof(uint32_t), 1, fp) != 1) {
        return -1;
    }

    if (header->magic != GRADIENT_SNAPSHOT_MAGIC) {
        fprintf(stderr, "ERROR: Invalid snapshot file magic: 0x%08X\n", header->magic);
        return -1;
    }

    if (header->samples_per_tensor != GRADIENT_SAMPLES_PER_TENSOR) {
        fprintf(stderr, "ERROR: Unexpected samples per tensor: %u (expected %d)\n",
                header->samples_per_tensor, GRADIENT_SAMPLES_PER_TENSOR);
        return -1;
    }

    return 0;
}

// Load a single snapshot from current file position
static int load_snapshot(FILE* fp, gradient_snapshot_t* snapshot, int num_layers) {
    int total_tensors = 1 + num_layers * TENSORS_PER_LAYER + 1;
    int total_vectors = total_tensors * GRADIENT_SAMPLES_PER_TENSOR;

    // Allocate vector array if not already allocated
    if (!snapshot->vectors) {
        snapshot->vectors = calloc(total_vectors, sizeof(gradient_vector_t));
        snapshot->num_vectors = total_vectors;
    }

    // Read metadata
    if (fread(&snapshot->meta.step, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&snapshot->meta.timestamp, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&snapshot->meta.loss, sizeof(float), 1, fp) != 1 ||
        fread(&snapshot->meta.lr, sizeof(float), 1, fp) != 1) {
        return -1;  // EOF or read error
    }

    // Read gradient samples for all tensors
    float grad_samples[GRADIENT_SAMPLES_PER_TENSOR];
    int vector_idx = 0;

    for (int tensor_idx = 0; tensor_idx < total_tensors; tensor_idx++) {
        // Read 528 gradient values for this tensor
        if (fread(grad_samples, sizeof(float), GRADIENT_SAMPLES_PER_TENSOR, fp) != GRADIENT_SAMPLES_PER_TENSOR) {
            return -1;
        }

        // Get base position for this tensor
        float base_x, base_y, base_z;
        tensor_index_to_position(tensor_idx, num_layers, &base_x, &base_y, &base_z);

        // Map gradient samples to 3D space around the tensor
        // Arrange in a small grid around the tensor position
        int samples_per_side = (int)ceil(sqrt(GRADIENT_SAMPLES_PER_TENSOR));  // ~23x23 grid
        float sample_spacing = 0.05f;  // Tight clustering around tensor point

        for (int s = 0; s < GRADIENT_SAMPLES_PER_TENSOR; s++) {
            gradient_vector_t* vec = &snapshot->vectors[vector_idx++];

            // Position: offset from tensor base position
            int gx = s % samples_per_side;
            int gy = s / samples_per_side;
            float offset_x = (gx - samples_per_side/2) * sample_spacing;
            float offset_y = (gy - samples_per_side/2) * sample_spacing;
            vec->x = base_x + offset_x;
            vec->y = base_y + offset_y;
            vec->z = base_z;

            // Magnitude (absolute value - used for length and color)
            vec->magnitude = fabsf(grad_samples[s]);

            // Direction: Radial spread pointing toward high-energy area
            // Magnitude only affects LENGTH and COLOR, not direction

            // Calculate radial direction from center (in x-y plane only)
            float radial_dist = sqrtf(offset_x * offset_x + offset_y * offset_y);
            if (radial_dist < 0.001f) {
                // At center, point toward high z (where high energy is observed)
                vec->dx = 0.0f;
                vec->dy = 0.0f;
                vec->dz = 1.0f;  // Toward high z
            } else {
                // Angle outward in x-y plane, toward high z
                // 40% radial (x-y spread), 60% toward high z
                vec->dx = (offset_x / radial_dist) * 0.4f;
                vec->dy = (offset_y / radial_dist) * 0.4f;
                vec->dz = 0.6f;  // Toward high z (where high energy is)

                // Normalize
                float len = sqrtf(vec->dx * vec->dx + vec->dy * vec->dy + vec->dz * vec->dz);
                vec->dx /= len;
                vec->dy /= len;
                vec->dz /= len;
            }
        }
    }

    snapshot->valid = 1;
    return 0;
}

// Load all snapshots from file (replay mode)
static int load_all_snapshots(const char* path, playback_state_t* playback) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open snapshot file: %s\n", path);
        return -1;
    }

    // Load header
    snapshot_header_t header;
    if (load_snapshot_header(fp, &header) < 0) {
        fclose(fp);
        return -1;
    }

    printf("[FIELD] Snapshot header: version=%u, samples=%u, layers=%u\n",
           header.version, header.samples_per_tensor, header.num_layers);

    // Store architecture info in playback state
    playback->num_layers = header.num_layers;
    playback->total_tensors = 1 + header.num_layers * TENSORS_PER_LAYER + 1;
    playback->total_vectors = playback->total_tensors * GRADIENT_SAMPLES_PER_TENSOR;

    // Count snapshots (estimate from file size)
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    long header_size = sizeof(uint32_t) * 4;
    long snapshot_size = sizeof(snapshot_metadata_t) +
                        (playback->total_tensors * GRADIENT_SAMPLES_PER_TENSOR * sizeof(float));
    int estimated_snapshots = (file_size - header_size) / snapshot_size;

    printf("[FIELD] File size: %ld bytes, estimated snapshots: %d\n", file_size, estimated_snapshots);

    // Allocate snapshot array
    playback->snapshots = calloc(estimated_snapshots + 10, sizeof(gradient_snapshot_t));
    playback->num_snapshots = 0;

    // Load all snapshots
    fseek(fp, header_size, SEEK_SET);
    while (playback->num_snapshots < estimated_snapshots + 10) {
        gradient_snapshot_t* snapshot = &playback->snapshots[playback->num_snapshots];
        if (load_snapshot(fp, snapshot, playback->num_layers) < 0) {
            break;  // EOF or error
        }
        playback->num_snapshots++;
    }

    fclose(fp);

    printf("[FIELD] Loaded %d snapshots\n", playback->num_snapshots);

    if (playback->num_snapshots == 0) {
        free(playback->snapshots);
        playback->snapshots = NULL;
        return -1;
    }

    // Initialize playback state
    playback->current_index = 0;
    playback->playing = 0;
    playback->playback_speed = PLAYBACK_FPS;
    playback->last_frame_time = SDL_GetTicks() / 1000.0;

    return 0;
}

// ============================================
// RENDERING
// ============================================

// Forward declarations
static void render_bitmap_text(bitmap_font_t* bfont, const char* text, int x, int y, SDL_Color color);

// Draw a 3D arrow (vector visualization)
static void draw_arrow(float x, float y, float z, float dx, float dy, float dz,
                      float magnitude, float scale) {
    float r, g, b;
    emf_spectrum_color(magnitude, &r, &g, &b);
    glColor3f(r, g, b);

    // Arrow length proportional to magnitude
    float length = fabsf(magnitude) * scale;

    // Draw ALL vectors with minimum visible length
    if (length < 0.02f) length = 0.02f;

    float ex = x + dx * length;
    float ey = y + dy * length;
    float ez = z + dz * length;

    // Draw line (thicker for visibility)
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glVertex3f(x, y, z);
    glVertex3f(ex, ey, ez);
    glEnd();

    // Draw arrowhead 
    glPushMatrix();
    glTranslatef(ex, ey, ez);

    // Cone size based on arrow length
    float cone_height = fmaxf(length * 0.15f, 0.01f);
    float cone_radius = cone_height * 0.5f;

    // Calculate rotation to align cone with direction vector
    // (Simplified: assume pointing along +z for now, proper rotation later)
    // FML spent 2 hrs on this
    // Draw cone as 8 triangular segments for smooth appearance
    int segments = 8;
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < segments; i++) {
        float angle1 = (i * 2.0f * M_PI) / segments;
        float angle2 = ((i + 1) * 2.0f * M_PI) / segments;

        float x1 = cone_radius * cosf(angle1);
        float y1 = cone_radius * sinf(angle1);
        float x2 = cone_radius * cosf(angle2);
        float y2 = cone_radius * sinf(angle2);

        // Triangle from tip to base edge
        glVertex3f(dx * cone_height, dy * cone_height, dz * cone_height);  // Tip
        glVertex3f(x1, y1, 0.0f);  // Base point 1
        glVertex3f(x2, y2, 0.0f);  // Base point 2
    }
    glEnd();

    glPopMatrix();
    glLineWidth(1.5f);  // Reset line width
}

// Draw bounding box around a tensor's vector field
static void draw_tensor_bbox(float x, float y, float z, float size) {
    glColor3f(0.3f, 0.3f, 0.3f);  // Dark gray
    glLineWidth(1.0f);

    float hs = size / 2.0f;  // Half size

    glBegin(GL_LINE_LOOP);
    // Bottom face
    glVertex3f(x - hs, y - hs, z - hs);
    glVertex3f(x + hs, y - hs, z - hs);
    glVertex3f(x + hs, y - hs, z + hs);
    glVertex3f(x - hs, y - hs, z + hs);
    glEnd();

    glBegin(GL_LINE_LOOP);
    // Top face
    glVertex3f(x - hs, y + hs, z - hs);
    glVertex3f(x + hs, y + hs, z - hs);
    glVertex3f(x + hs, y + hs, z + hs);
    glVertex3f(x - hs, y + hs, z + hs);
    glEnd();

    glBegin(GL_LINES);
    // Vertical edges
    glVertex3f(x - hs, y - hs, z - hs); glVertex3f(x - hs, y + hs, z - hs);
    glVertex3f(x + hs, y - hs, z - hs); glVertex3f(x + hs, y + hs, z - hs);
    glVertex3f(x + hs, y - hs, z + hs); glVertex3f(x + hs, y + hs, z + hs);
    glVertex3f(x - hs, y - hs, z + hs); glVertex3f(x - hs, y + hs, z + hs);
    glEnd();

    glLineWidth(2.0f);  // Restore
}

// Tensor label rendering using bitmap font
typedef struct {
    float world_x, world_y, world_z;
    char text[64];
} label_info_t;

static label_info_t g_label_buffer[256];
static int g_num_labels = 0;

// Queue label for rendering
static void queue_tensor_label(float x, float y, float z, float box_size, const char* label) {
    if (g_num_labels >= 256) return;

    label_info_t* info = &g_label_buffer[g_num_labels++];
    info->world_x = x;
    info->world_y = y + (box_size / 2.0f) + 0.3f;  // Top of box
    info->world_z = z;

    // Safe string copy
    size_t label_len = strlen(label);
    size_t copy_len = (label_len < sizeof(info->text) - 1) ? label_len : sizeof(info->text) - 1;
    memcpy(info->text, label, copy_len);
    info->text[copy_len] = '\0';
}

// Render all queued labels using bitmap font
static void render_queued_labels(bitmap_font_t* bfont) {
    if (!bfont || !bfont->texture || g_num_labels == 0) return;

    // Get current matrices
    GLdouble modelview[16], projection[16];
    GLint viewport[4];

    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetIntegerv(GL_VIEWPORT, viewport);

    // Switch to 2D overlay mode
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);

    // Render all labels
    SDL_Color white = {255, 255, 255, 255};

    for (int i = 0; i < g_num_labels; i++) {
        label_info_t* info = &g_label_buffer[i];

        // Project 3D world position to 2D screen
        GLdouble screen_x, screen_y, screen_z;
        gluProject(info->world_x, info->world_y, info->world_z,
                  modelview, projection, viewport,
                  &screen_x, &screen_y, &screen_z);

        // Only render if in front of camera
        if (screen_z < 1.0 && screen_z > 0.0) {
            // Calculate text width for centering
            int text_len = strlen(info->text);
            int text_width = text_len * bfont->char_width;

            int text_x = (int)screen_x - (text_width / 2);  // Center horizontally
            int text_y = WINDOW_HEIGHT - (int)screen_y - 5;  // Flip Y

            render_bitmap_text(bfont, info->text, text_x, text_y, white);
        }
    }

    glEnable(GL_DEPTH_TEST);

    // Restore matrices
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Clear buffer for next frame
    g_num_labels = 0;
}

// Render a single snapshot
static void render_snapshot(gradient_snapshot_t* snapshot, playback_state_t* playback, int show_labels) {
    if (!snapshot || !snapshot->valid) return;

    // Draw all gradient vectors
    for (int i = 0; i < snapshot->num_vectors; i++) {
        gradient_vector_t* vec = &snapshot->vectors[i];
        draw_arrow(vec->x, vec->y, vec->z,
                  vec->dx, vec->dy, vec->dz,
                  vec->magnitude, VECTOR_SCALE);
    }

    // Draw tensor bounding boxes and queue labels
    float box_size = TENSOR_SPACING * 0.9f;
    for (int t = 0; t < playback->total_tensors; t++) {
        float x, y, z;
        tensor_index_to_position(t, playback->num_layers, &x, &y, &z);

        // Draw bounding box
        draw_tensor_bbox(x, y, z, box_size);

        // Queue label for rendering if enabled
        if (show_labels) {
            char label[64];
            get_tensor_name(t, playback->num_layers, label, sizeof(label));
            queue_tensor_label(x, y, z, box_size, label);
        }

        // Draw center point
        glColor3f(0.5f, 0.5f, 0.5f);
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        glVertex3f(x, y, z);
        glEnd();
    }
}

// Generate bitmap font atlas from TTF font
static int generate_bitmap_font(TTF_Font* ttf_font, bitmap_font_t* bfont) {
    if (!ttf_font) return -1;

    // Character dimensions
    int char_w = 12;  // Character width
    int char_h = 18;  // Character height
    int cols = 16;    // 16 characters per row
    int rows = 6;     // 6 rows (96 printable ASCII chars)

    int atlas_w = char_w * cols;
    int atlas_h = char_h * rows;

    // Create atlas surface
    SDL_Surface* atlas = SDL_CreateRGBSurfaceWithFormat(0, atlas_w, atlas_h, 32, SDL_PIXELFORMAT_RGBA32);
    if (!atlas) return -1;

    SDL_FillRect(atlas, NULL, SDL_MapRGBA(atlas->format, 0, 0, 0, 0));  // Transparent background

    // Render each printable ASCII character (32-127)
    SDL_Color white = {255, 255, 255, 255};
    for (int i = 0; i < 96; i++) {
        char str[2] = {(char)(32 + i), '\0'};
        SDL_Surface* char_surf = TTF_RenderText_Blended(ttf_font, str, white);

        if (char_surf) {
            SDL_Rect dest;
            dest.x = (i % cols) * char_w;
            dest.y = (i / cols) * char_h;
            dest.w = char_surf->w;
            dest.h = char_surf->h;

            SDL_BlitSurface(char_surf, NULL, atlas, &dest);
            SDL_FreeSurface(char_surf);
        }
    }

    // Upload to OpenGL
    glGenTextures(1, &bfont->texture);
    glBindTexture(GL_TEXTURE_2D, bfont->texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlas_w, atlas_h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, atlas->pixels);

    SDL_FreeSurface(atlas);

    bfont->char_width = char_w;
    bfont->char_height = char_h;
    bfont->atlas_cols = cols;
    bfont->atlas_rows = rows;

    printf("[FIELD] Bitmap font atlas generated: %dx%d (%d chars)\n", atlas_w, atlas_h, cols * rows);

    return 0;
}

// Render single character from bitmap font
static void render_bitmap_char(bitmap_font_t* bfont, char c, int x, int y) {
    unsigned char uc = (unsigned char)c;
    if (uc < 32 || uc > 127) return;  // Only printable ASCII

    int char_index = uc - 32;
    int col = char_index % bfont->atlas_cols;
    int row = char_index / bfont->atlas_cols;

    // UV coordinates in atlas
    float u0 = (float)col / bfont->atlas_cols;
    float v0 = (float)row / bfont->atlas_rows;
    float u1 = (float)(col + 1) / bfont->atlas_cols;
    float v1 = (float)(row + 1) / bfont->atlas_rows;

    // Render quad
    glBegin(GL_QUADS);
    glTexCoord2f(u0, v0); glVertex2f(x, y);
    glTexCoord2f(u1, v0); glVertex2f(x + bfont->char_width, y);
    glTexCoord2f(u1, v1); glVertex2f(x + bfont->char_width, y + bfont->char_height);
    glTexCoord2f(u0, v1); glVertex2f(x, y + bfont->char_height);
    glEnd();
}

// Render text string using bitmap font
static void render_bitmap_text(bitmap_font_t* bfont, const char* text, int x, int y, SDL_Color color) {
    if (!bfont || !bfont->texture || !text) return;

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, bfont->texture);
    glColor4ub(color.r, color.g, color.b, color.a);

    int cursor_x = x;
    for (const char* p = text; *p; p++) {
        render_bitmap_char(bfont, *p, cursor_x, y);
        cursor_x += bfont->char_width;
    }

    glDisable(GL_TEXTURE_2D);
}

// Removed legacy render_text - no longer used

// Render HUD overlay using bitmap font
static void render_hud(app_state_t* app) {
    // Switch to 2D orthographic projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);

    // Get current snapshot
    gradient_snapshot_t* snapshot = NULL;
    if (app->playback.num_snapshots > 0 && app->playback.current_index >= 0) {
        snapshot = &app->playback.snapshots[app->playback.current_index];
    }

    // Draw semi-transparent background
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.0f, 0.0f, 0.0f, 0.8f);
    glBegin(GL_QUADS);
    glVertex2f(10, 10);
    glVertex2f(450, 10);
    glVertex2f(450, 170);
    glVertex2f(10, 170);
    glEnd();

    // Render text overlay using bitmap font
    if (app->bitmap_font.texture && snapshot && snapshot->valid) {
        SDL_Color white = {255, 255, 255, 255};
        SDL_Color cyan = {0, 255, 255, 255};
        SDL_Color yellow = {255, 255, 0, 255};
        SDL_Color green = {0, 255, 0, 255};

        char buf[256];

        // Title
        render_bitmap_text(&app->bitmap_font, "SERAPH GRADIENT FIELD", 20, 15, cyan);

        // Step
        snprintf(buf, sizeof(buf), "Step: %u", snapshot->meta.step);
        render_bitmap_text(&app->bitmap_font, buf, 20, 45, white);

        // Snapshot progress
        snprintf(buf, sizeof(buf), "Snap: %d/%d", app->playback.current_index + 1, app->playback.num_snapshots);
        render_bitmap_text(&app->bitmap_font, buf, 20, 70, white);

        // Loss
        snprintf(buf, sizeof(buf), "Loss: %.6f", snapshot->meta.loss);
        render_bitmap_text(&app->bitmap_font, buf, 20, 95, yellow);

        // Learning rate
        snprintf(buf, sizeof(buf), "LR: %.8f", snapshot->meta.lr);
        render_bitmap_text(&app->bitmap_font, buf, 20, 120, green);

        // Playback status
        const char* status = app->playback.playing ? "PLAY" : "PAUSE";
        SDL_Color status_color = app->playback.playing ? green : yellow;
        render_bitmap_text(&app->bitmap_font, status, 280, 30, status_color);

        // Speed
        snprintf(buf, sizeof(buf), "Speed: %.1fx", app->playback.playback_speed);
        render_bitmap_text(&app->bitmap_font, buf, 280, 50, white);
    }

    if (snapshot && snapshot->valid) {
        // Print metadata to console when snapshot changes
        if (app->last_printed_step != app->playback.current_index) {
            printf("\r[FIELD] Snapshot %d/%d | Step: %u | Loss: %.6f | LR: %.8f | Speed: %.1fx     ",
                   app->playback.current_index + 1, app->playback.num_snapshots,
                   snapshot->meta.step, snapshot->meta.loss, snapshot->meta.lr,
                   app->playback.playback_speed);
            fflush(stdout);
            app->last_printed_step = app->playback.current_index;
        }
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // Restore matrices
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// ============================================
// INPUT HANDLING
// ============================================

static void handle_keyboard(app_state_t* app, SDL_KeyboardEvent* event) {
    if (event->type != SDL_KEYDOWN) return;

    switch (event->keysym.sym) {
        // Camera panning (WASD - move around the scene)
        case SDLK_w:  // Move forward
            app->camera.target_z -= 0.5f;
            break;
        case SDLK_s:  // Move backward
            app->camera.target_z += 0.5f;
            break;
        case SDLK_a:  // Move left
            app->camera.target_x -= 0.5f;
            break;
        case SDLK_d:  // Move right
            app->camera.target_x += 0.5f;
            break;
	case SDLK_u:
	    app->camera.target_y += 0.5f;
	    break;
	case SDLK_o:
	    app->camera.target_y -= 0.5f;
	    break;


        // Camera rotation (I, J, K, L - vim style)
        case SDLK_i:  // Rotate camera up
            app->camera.rotation_x += CAMERA_ROTATION_SPEED;
            break;
        case SDLK_k:  // Rotate camera down
            app->camera.rotation_x -= CAMERA_ROTATION_SPEED;
            break;
        case SDLK_j:  // Rotate camera left
            app->camera.rotation_y -= CAMERA_ROTATION_SPEED;
            break;
        case SDLK_l:  // Rotate camera right
            app->camera.rotation_y += CAMERA_ROTATION_SPEED;
            break;

        // Zoom
        case SDLK_q:  // Zoom out
            app->camera.distance += CAMERA_ZOOM_SPEED;
            break;
        case SDLK_e:  // Zoom in
            app->camera.distance -= CAMERA_ZOOM_SPEED;
            if (app->camera.distance < 1.0f) app->camera.distance = 1.0f;
            break;

        // Playback controls
        case SDLK_SPACE:  // Play/pause
            app->playback.playing = !app->playback.playing;
            printf("[FIELD] Playback %s\n", app->playback.playing ? "playing" : "paused");
            break;
        case SDLK_LEFT:  // Step backward
            if (app->playback.current_index > 0) {
                app->playback.current_index--;
                printf("[FIELD] Step %d/%d\n", app->playback.current_index, app->playback.num_snapshots);
            }
            break;
        case SDLK_RIGHT:  // Step forward
            if (app->playback.current_index < app->playback.num_snapshots - 1) {
                app->playback.current_index++;
                printf("[FIELD] Step %d/%d\n", app->playback.current_index, app->playback.num_snapshots);
            }
            break;
        case SDLK_DOWN:  // Slow down
            app->playback.playback_speed /= 1.5f;
            if (app->playback.playback_speed < 0.1f) app->playback.playback_speed = 0.1f;
            printf("[FIELD] Speed: %.1fx\n", app->playback.playback_speed);
            break;
        case SDLK_UP:  // Speed up
            app->playback.playback_speed *= 1.5f;
            printf("[FIELD] Speed: %.1fx\n", app->playback.playback_speed);
            break;
        case SDLK_HOME:  // Jump to start
            app->playback.current_index = 0;
            printf("[FIELD] Jump to start\n");
            break;
        case SDLK_END:  // Jump to end
            app->playback.current_index = app->playback.num_snapshots - 1;
            printf("[FIELD] Jump to end\n");
            break;

        // Application controls
        case SDLK_ESCAPE:  // Quit
            app->running = 0;
            break;
        case SDLK_r:  // Reset camera
            app->camera.distance = CAMERA_DISTANCE;
            app->camera.rotation_y = 45.0f;
            app->camera.rotation_x = 30.0f;
            app->camera.target_x = 0.0f;
            app->camera.target_y = 0.0f;
            app->camera.target_z = 0.0f;
            printf("[FIELD] Camera reset\n");
            break;
        case SDLK_t:  // Toggle labels
            app->show_labels = !app->show_labels;
            printf("[FIELD] Tensor labels: %s\n", app->show_labels ? "ON" : "OFF");
            break;
    }
}

// ============================================
// MAIN LOOP
// ============================================

// Check for new snapshots in live mode
static void update_live_snapshots(app_state_t* app) {
    if (!app->mode_realtime || !app->snapshot_file || !app->watch_path) return;

    // Check file size
    struct stat st;
    if (stat(app->watch_path, &st) != 0) return;

    long current_size = st.st_size;
    if (current_size <= app->last_file_pos) return;  // No new data

    // Seek to last read position
    fseek(app->snapshot_file, app->last_file_pos, SEEK_SET);

    // Load new snapshots
    int loaded = 0;
    while (app->playback.num_snapshots < 1000) {  // Safety limit
        gradient_snapshot_t* snapshot = &app->playback.snapshots[app->playback.num_snapshots];
        if (load_snapshot(app->snapshot_file, snapshot, app->playback.num_layers) < 0) {
            break;  // EOF or error
        }
        app->playback.num_snapshots++;
        loaded++;
    }

    if (loaded > 0) {
        printf("[FIELD] Loaded %d new snapshots (total: %d)\n", loaded, app->playback.num_snapshots);
        // Auto-advance to latest snapshot in live mode
        app->playback.current_index = app->playback.num_snapshots - 1;
    }

    // Update file position
    app->last_file_pos = ftell(app->snapshot_file);
}

static void update_playback(app_state_t* app) {
    // Check for new snapshots in live mode
    if (app->mode_realtime) {
        static double last_check = 0.0;
        double now = SDL_GetTicks() / 1000.0;
        if (now - last_check >= 0.5) {  // Check every 0.5 seconds
            update_live_snapshots(app);
            last_check = now;
        }
    }

    if (!app->playback.playing || app->playback.num_snapshots == 0) return;

    double now = SDL_GetTicks() / 1000.0;
    double dt = now - app->playback.last_frame_time;

    // Advance frame based on playback speed (only in replay mode)
    if (!app->mode_realtime && dt >= 1.0 / app->playback.playback_speed) {
        app->playback.current_index++;
        if (app->playback.current_index >= app->playback.num_snapshots) {
            app->playback.current_index = 0;  // Loop
        }
        app->playback.last_frame_time = now;
    }
}

static void render_frame(app_state_t* app) {
    // Clear
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);  // Dark blue background
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup 3D camera
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (double)WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Camera positioning (orbit around target point)
    float cam_x = app->camera.target_x + app->camera.distance * sinf(app->camera.rotation_y * M_PI / 180.0f) *
                  cosf(app->camera.rotation_x * M_PI / 180.0f);
    float cam_y = app->camera.target_y + app->camera.distance * sinf(app->camera.rotation_x * M_PI / 180.0f);
    float cam_z = app->camera.target_z + app->camera.distance * cosf(app->camera.rotation_y * M_PI / 180.0f) *
                  cosf(app->camera.rotation_x * M_PI / 180.0f);

    gluLookAt(cam_x, cam_y, cam_z,                                     // Eye
              app->camera.target_x, app->camera.target_y, app->camera.target_z,  // Center (look at target)
              0.0, 1.0, 0.0);                                          // Up

    // Render current snapshot (3D pass)
    if (app->playback.num_snapshots > 0 && app->playback.current_index >= 0) {
        render_snapshot(&app->playback.snapshots[app->playback.current_index],
                       &app->playback, app->show_labels);
    }

    // Render queued tensor labels (2D overlay after 3D rendering)
    if (app->show_labels && app->bitmap_font.texture) {
        render_queued_labels(&app->bitmap_font);
    }

    // Render HUD
    render_hud(app);

    // Swap buffers
    SDL_GL_SwapWindow(app->window);
}

static int main_loop(app_state_t* app) {
    SDL_Event event;

    while (app->running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    app->running = 0;
                    break;
                case SDL_KEYDOWN:
                case SDL_KEYUP:
                    handle_keyboard(app, &event.key);
                    break;
            }
        }

        // Update playback
        update_playback(app);

        // Render
        render_frame(app);

        // Cap framerate
        SDL_Delay(16);  // ~60 FPS
    }

    return 0;
}

// ============================================
// INITIALIZATION
// ============================================

static int init_sdl(app_state_t* app) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "ERROR: SDL_Init failed: %s\n", SDL_GetError());
        return -1;
    }

    // Initialize SDL_ttf
    if (TTF_Init() < 0) {
        fprintf(stderr, "WARNING: TTF_Init failed: %s (text rendering disabled)\n", TTF_GetError());
    }

    // OpenGL attributes
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    // Create window
    app->window = SDL_CreateWindow(
        WINDOW_TITLE,
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
    );

    if (!app->window) {
        fprintf(stderr, "ERROR: SDL_CreateWindow failed: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        return -1;
    }

    // Create OpenGL context
    app->gl_context = SDL_GL_CreateContext(app->window);
    if (!app->gl_context) {
        fprintf(stderr, "ERROR: SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(app->window);
        TTF_Quit();
        SDL_Quit();
        return -1;
    }

    // Load font
    const char* font_paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        NULL
    };

    app->font = NULL;
    for (int i = 0; font_paths[i] != NULL; i++) {
        app->font = TTF_OpenFont(font_paths[i], 18);
        if (app->font) {
            printf("[FIELD] Loaded font: %s\n", font_paths[i]);
            break;
        }
    }

    if (!app->font) {
        fprintf(stderr, "WARNING: Could not load font, text rendering disabled\n");
    } else {
        // Generate bitmap font atlas
        if (generate_bitmap_font(app->font, &app->bitmap_font) < 0) {
            fprintf(stderr, "WARNING: Could not generate bitmap font\n");
        }
    }

    // Enable VSync
    SDL_GL_SetSwapInterval(1);

    // OpenGL setup 
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Enable anti-aliasing
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    // Higher quality rendering
    glLineWidth(2.0f);
    glPointSize(4.0f);

    // Better depth testing
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);

    printf("[FIELD] SDL2 + OpenGL initialized (HIGH QUALITY)\n");
    printf("[FIELD] Video driver: %s\n", SDL_GetCurrentVideoDriver());

    return 0;
}

// ============================================
// ENTRY POINT
// ============================================

static void print_usage(const char* prog) {
    printf("Seraph Gradient Field Monitor\n");
    printf("Usage:\n");
    printf("  Replay mode:    %s --replay <snapshot_file.bin>\n", prog);
    printf("  Real-time mode: %s --live <checkpoint_dir>\n", prog);
    printf("\n");
    printf("Camera Controls:\n");
    printf("  WASD       - Pan camera (move around scene)\n");
    printf("  I/J/K/L    - Rotate camera view (vim style)\n");
    printf("  Q/E        - Zoom out/in\n");
    printf("  R          - Reset camera\n");
    printf("\n");
    printf("Playback Controls:\n");
    printf("  Space      - Play/pause\n");
    printf("  Left/Right - Step backward/forward\n");
    printf("  Up/Down    - Speed up/slow down\n");
    printf("  Home/End   - Jump to start/end\n");
    printf("\n");
    printf("Visualization:\n");
    printf("  T          - Toggle tensor labels\n");
    printf("\n");
    printf("Snapshot Timing:\n");
    printf("  --snapshot-every N captures gradients every N TRAINING STEPS\n");
    printf("  (not per optimizer update - you see gradient ACCUMULATION in action!)\n");
    printf("  \n");
    printf("  If using --gradient-accumulation-steps 4:\n");
    printf("    Step 0,4,8,12...  → Fresh gradients (just zeroed)\n");
    printf("    Step 1,5,9,13...  → 1/4 accumulated\n");
    printf("    Step 2,6,10,14... → 2/4 accumulated\n");
    printf("    Step 3,7,11,15... → 3/4 accumulated (about to apply!)\n");
    printf("\n");
    printf("  ESC        - Quit\n");
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    int mode_realtime = 0;
    char* path = NULL;

    if (strcmp(argv[1], "--replay") == 0) {
        mode_realtime = 0;
        path = argv[2];
    } else if (strcmp(argv[1], "--live") == 0) {
        mode_realtime = 1;
        path = argv[2];
    } else {
        fprintf(stderr, "ERROR: Unknown mode: %s\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  SERAPH GRADIENT FIELD MONITOR                                  ║\n");
    printf("║  Electromagnetic Field Visualization (Jefimenko's Retarded Time)║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    // Initialize application state
    app_state_t app = {0};
    app.running = 1;
    app.mode_realtime = mode_realtime;
    app.last_printed_step = -1;
    app.show_labels = 1;  // Labels ON by default

    // Initialize camera
    app.camera.distance = CAMERA_DISTANCE;
    app.camera.rotation_y = 45.0f;
    app.camera.rotation_x = 30.0f;
    app.camera.target_x = 0.0f;
    app.camera.target_y = 0.0f;
    app.camera.target_z = 0.0f;

    // Load snapshots
    if (mode_realtime) {
        // Live mode: watch directory for new snapshots
        char snapshot_path[1024];
        snprintf(snapshot_path, sizeof(snapshot_path), "%s/gradient_snapshots.bin", path);

        printf("[FIELD] Live mode: watching %s\n", snapshot_path);

        // Open file for reading (will be updated periodically)
        app.snapshot_file = fopen(snapshot_path, "rb");
        if (!app.snapshot_file) {
            fprintf(stderr, "ERROR: Cannot open snapshot file: %s\n", snapshot_path);
            return 1;
        }

        // Load initial snapshots
        snapshot_header_t header;
        if (load_snapshot_header(app.snapshot_file, &header) < 0) {
            fprintf(stderr, "ERROR: Cannot read snapshot header\n");
            fclose(app.snapshot_file);
            return 1;
        }

        app.playback.num_layers = header.num_layers;
        app.playback.total_tensors = 1 + header.num_layers * TENSORS_PER_LAYER + 1;
        app.playback.total_vectors = app.playback.total_tensors * GRADIENT_SAMPLES_PER_TENSOR;

        printf("[FIELD] Live mode initialized: %d layers, %d tensors\n",
               app.playback.num_layers, app.playback.total_tensors);

        // Allocate initial snapshot array (will grow)
        app.playback.snapshots = calloc(1000, sizeof(gradient_snapshot_t));
        app.playback.num_snapshots = 0;
        app.playback.playing = 1;  // Auto-play in live mode

        long header_size = sizeof(uint32_t) * 4;
        app.last_file_pos = header_size;

        // Store path for periodic checking
        app.watch_path = strdup(snapshot_path);

    } else {
        // Replay mode: load all snapshots
        printf("[FIELD] Loading snapshots from: %s\n", path);
        if (load_all_snapshots(path, &app.playback) < 0) {
            fprintf(stderr, "ERROR: Failed to load snapshots\n");
            return 1;
        }
    }

    // Initialize SDL
    if (init_sdl(&app) < 0) {
        return 1;
    }

    printf("\n[FIELD] Initialization complete\n");
    printf("[FIELD] Model: %d layers, %d tensors, %d vectors per snapshot\n",
           app.playback.num_layers, app.playback.total_tensors, app.playback.total_vectors);
    printf("[FIELD] Snapshots loaded: %d\n", app.playback.num_snapshots);
    printf("[FIELD] Tensor labels: %s (press T to toggle)\n", app.show_labels ? "ON" : "OFF");
    printf("[FIELD] Controls: WASD (pan), IJKL (rotate), Q/E (zoom), T (labels), ESC (quit)\n\n");

    // Main loop
    int ret = main_loop(&app);

    // Cleanup
    if (app.watch_path) {
        free(app.watch_path);
    }

    if (app.snapshot_file) {
        fclose(app.snapshot_file);
    }

    if (app.playback.snapshots) {
        // Free individual snapshot vectors
        for (int i = 0; i < app.playback.num_snapshots; i++) {
            if (app.playback.snapshots[i].vectors) {
                free(app.playback.snapshots[i].vectors);
            }
        }
        free(app.playback.snapshots);
    }

    // Cleanup bitmap font
    if (app.bitmap_font.texture) {
        glDeleteTextures(1, &app.bitmap_font.texture);
    }

    if (app.font) {
        TTF_CloseFont(app.font);
    }

    if (app.gl_context) {
        SDL_GL_DeleteContext(app.gl_context);
    }
    if (app.window) {
        SDL_DestroyWindow(app.window);
    }

    TTF_Quit();
    SDL_Quit();

    printf("\n[FIELD] Shutdown complete\n");

    return ret;
}
