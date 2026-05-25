// vulkan_profile_query.c — Vulkan hardware profile + topology optimizer
// Queries Vulkan device limits, runs the matmul tile×reg×bk sweep and adamw
// block search, prints glslc -D flags to stdout for Makefile shader compiles.
//
// Usage:
//   ./vulkan-profile-query                     # print hardware info (stderr)
//   ./vulkan-profile-query --defines <dir>     # print -D flags (stdout)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

typedef struct {
    char name[256];

    // Sweep inputs
    uint32_t compute_units;            // sm_count — for blocks/SM calc
    uint32_t warps_per_compute_unit;   // → threads/sm = warps × subgroup
    uint32_t subgroup_size;            // sweep step for tile, workgroup alignment
    uint32_t max_workgroup_invocations;// TPB upper bound in sweep
    uint32_t max_workgroup_size_x;     // workgroup dim bound
    uint32_t max_workgroup_size_y;     // workgroup dim bound
    uint32_t max_shared_memory;        // smem fit check for tile+bk

    // Reduction shader gating
    int sg_arith;                      // subgroupAdd — tree reduction
    int sg_shuffle;                    // subgroupShuffle — cross-lane transpose

    // Reference only
    uint64_t vram_bytes;
    int has_sm_builtins;
} vk_hw_t;

typedef struct {
    int M;
    int N;
    int K;
    int count;
} matmul_op_t;

static int query_vk(vk_hw_t* hw) {
    VkInstance instance;
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vulkan-profile-query",
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    if (vkCreateInstance(&instance_info, NULL, &instance) != VK_SUCCESS) {
        fprintf(stderr, "ERROR: vkCreateInstance failed\n");
        return -1;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (device_count == 0) {
        fprintf(stderr, "ERROR: no Vulkan devices\n");
        vkDestroyInstance(instance, NULL);
        return -1;
    }
    VkPhysicalDevice* devices = malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &device_count, devices);

    // Prefer discrete GPU
    VkPhysicalDevice pdev = VK_NULL_HANDLE;
    int best_score = -1;
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        int score = 0;
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 500;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) score += 100;
        if (score > best_score) { best_score = score; pdev = devices[i]; }
    }

    // Only VK_NV_shader_sm_builtins is queried here — it gates sm_count/warps_per_sm.
    // Shader-selection extensions are detected at runtime in vulkan_backend.c.
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, NULL);
    VkExtensionProperties* exts = malloc(ext_count * sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(pdev, NULL, &ext_count, exts);
    int has_sm_builtins = 0;
    for (uint32_t i = 0; i < ext_count; i++) {
        if (strcmp(exts[i].extensionName, "VK_NV_shader_sm_builtins") == 0) {
            has_sm_builtins = 1;
            break;
        }
    }
    free(exts);

    // Chain property queries
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV,
    };
    VkPhysicalDeviceSubgroupProperties sg_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        .pNext = has_sm_builtins ? &sm_props : NULL,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &sg_props,
    };
    vkGetPhysicalDeviceProperties2(pdev, &props2);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);

    VkPhysicalDeviceProperties* p = &props2.properties;
    strncpy(hw->name, p->deviceName, sizeof(hw->name) - 1);
    hw->subgroup_size = sg_props.subgroupSize;
    hw->max_workgroup_invocations = p->limits.maxComputeWorkGroupInvocations;
    hw->max_workgroup_size_x = p->limits.maxComputeWorkGroupSize[0];
    hw->max_workgroup_size_y = p->limits.maxComputeWorkGroupSize[1];
    hw->max_shared_memory = p->limits.maxComputeSharedMemorySize;
    hw->has_sm_builtins = has_sm_builtins;
    hw->compute_units = has_sm_builtins ? sm_props.shaderSMCount : 0;
    hw->warps_per_compute_unit = has_sm_builtins ? sm_props.shaderWarpsPerSM : 0;
    hw->sg_arith = !!(sg_props.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    hw->sg_shuffle = !!(sg_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT);

    hw->vram_bytes = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (mem_props.memoryHeaps[i].size > hw->vram_bytes)
                hw->vram_bytes = mem_props.memoryHeaps[i].size;
        }
    }

    free(devices);
    vkDestroyInstance(instance, NULL);
    return 0;
}

static void print_hw(const vk_hw_t* hw) {
    uint32_t threads_per_sm = hw->warps_per_compute_unit * hw->subgroup_size;

    fprintf(stderr, "\n%s\n", hw->name);
    fprintf(stderr, "  sm_count       %u\n", hw->compute_units);
    fprintf(stderr, "  subgroup       %u\n", hw->subgroup_size);
    fprintf(stderr, "  threads/sm     %u\n", threads_per_sm);
    fprintf(stderr, "  threads/block  %u\n", hw->max_workgroup_invocations);
    fprintf(stderr, "  warps/sm       %u\n", hw->warps_per_compute_unit);
    fprintf(stderr, "  max_wg_size    [%u, %u]\n", hw->max_workgroup_size_x, hw->max_workgroup_size_y);
    fprintf(stderr, "  shared/block   %u\n", hw->max_shared_memory);
    fprintf(stderr, "  vram           %.2f GiB\n", (double)hw->vram_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  sg_arith       %d\n", hw->sg_arith);
    fprintf(stderr, "  sg_shuffle     %d\n", hw->sg_shuffle);
    fprintf(stderr, "\n");
}

static int read_config_int(const char* json, const char* key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    return atoi(p);
}

static int load_model_config(const char* dir, int* z, int* inter, int* vocab,
                              int* L, int* h, int* kv_h, int* hd, int* max_seq) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json = malloc(len + 1);
    fread(json, 1, len, f);
    json[len] = 0;
    fclose(f);

    *z = read_config_int(json, "hidden_size");
    *inter = read_config_int(json, "intermediate_size");
    *vocab = read_config_int(json, "vocab_size");
    *L = read_config_int(json, "num_hidden_layers");
    *h = read_config_int(json, "num_attention_heads");
    *kv_h = read_config_int(json, "num_key_value_heads");
    *hd = read_config_int(json, "head_dim");
    *max_seq = read_config_int(json, "max_position_embeddings");

    if (*kv_h == 0) *kv_h = *h;
    if (*hd == 0 && *h > 0) *hd = *z / *h;
    if (*max_seq == 0) *max_seq = 2048;

    free(json);
    return 0;
}

static void print_defines(const vk_hw_t* hw, const char* model_dir) {
    int z, inter, vocab, L, h, kv_h, hd, max_seq;
    if (load_model_config(model_dir, &z, &inter, &vocab, &L, &h, &kv_h, &hd, &max_seq) != 0)
        return;

    int kv_dim = kv_h * hd;
    int seq = max_seq;

    fprintf(stderr, "  model z=%d inter=%d vocab=%d L=%d h=%d kv_h=%d hd=%d seq=%d\n",
            z, inter, vocab, L, h, kv_h, hd, seq);

    // Model matmul ops
    matmul_op_t ops[32];
    int nops = 0;
    ops[nops++] = (matmul_op_t){seq, z, z, L};           // Q/O proj
    ops[nops++] = (matmul_op_t){seq, kv_dim, z, L};      // K proj
    ops[nops++] = (matmul_op_t){seq, kv_dim, z, L};      // V proj
    ops[nops++] = (matmul_op_t){seq, z, z, L};           // O proj
    ops[nops++] = (matmul_op_t){seq, inter, z, L};       // gate
    ops[nops++] = (matmul_op_t){seq, inter, z, L};       // up
    ops[nops++] = (matmul_op_t){seq, z, inter, L};       // down
    ops[nops++] = (matmul_op_t){seq, vocab, z, 1};       // logits fwd
    ops[nops++] = (matmul_op_t){seq, z, vocab, 1};       // logits bwd
    ops[nops++] = (matmul_op_t){vocab, z, seq, 1};       // embed grad
    ops[nops++] = (matmul_op_t){seq, z, z, L};           // Q/K/V/O input grads
    ops[nops++] = (matmul_op_t){seq, inter, inter, L};   // down input grad
    ops[nops++] = (matmul_op_t){seq, z, inter, L};       // gate/up input grad
    ops[nops++] = (matmul_op_t){z, z, seq, L};           // Q/K/V/O weight grads
    ops[nops++] = (matmul_op_t){inter, z, seq, L};       // gate/up weight grads
    ops[nops++] = (matmul_op_t){z, inter, seq, L};       // down weight grads

    uint32_t warps_per_sm = hw->warps_per_compute_unit > 0
        ? hw->warps_per_compute_unit
        : hw->max_workgroup_invocations / hw->subgroup_size;
    uint32_t max_threads_per_sm = warps_per_sm * hw->subgroup_size;
    uint32_t sm_count = hw->compute_units > 0 ? hw->compute_units : 1;

    // ── Matmul tile × reg × bk sweep ──
    int best_tile = 0, best_reg = 0, best_bk = 0;
    long long best_util = 0;

    for (uint32_t tile = hw->subgroup_size; tile <= 256; tile += hw->subgroup_size) {
        for (uint32_t reg = 2; reg <= 8; reg *= 2) {
            if (tile % reg != 0) continue;
            uint32_t tpb = (tile / reg) * (tile / reg);
            if (tpb > hw->max_workgroup_invocations) continue;
            if (tpb < hw->subgroup_size) continue;
            uint32_t wg_dim = tile / reg;
            if (wg_dim > hw->max_workgroup_size_x || wg_dim > hw->max_workgroup_size_y) continue;

            uint32_t sm_cap_by_threads = max_threads_per_sm / tpb;

            for (uint32_t bk = 64; bk >= 8; bk /= 2) {
                uint32_t smem = 4 * (tile * (bk + 1) + bk * (tile + 1));
                if (smem > hw->max_shared_memory) continue;

                // Score across real model ops
                long long total_util = 0;
                for (int i = 0; i < nops; i++) {
                    uint32_t blocks = ((ops[i].M + tile - 1) / tile) * ((ops[i].N + tile - 1) / tile);
                    uint32_t bpsm = blocks / sm_count;
                    if (bpsm > sm_cap_by_threads) bpsm = sm_cap_by_threads;
                    uint32_t active = bpsm * tpb;
                    uint32_t util = (100 * active) / max_threads_per_sm;
                    total_util += (long long)util * ops[i].count;
                }

                if (total_util > best_util) {
                    best_util = total_util;
                    best_tile = tile;
                    best_reg = reg;
                    best_bk = bk;
                }
                break;  // largest bk that fits
            }
        }
    }

    if (best_tile == 0) {
        fprintf(stderr, "ERROR: matmul sweep found no valid config\n");
        return;
    }

    int thr = (best_tile / best_reg) * (best_tile / best_reg);
    uint32_t bpsm = max_threads_per_sm / thr;
    uint32_t occ = (100 * bpsm * thr) / max_threads_per_sm;
    uint32_t smem = 4 * (best_tile * (best_bk + 1) + best_bk * (best_tile + 1));

    fprintf(stderr, "\n  matmul\n");
    fprintf(stderr, "  tile    %d\n", best_tile);
    fprintf(stderr, "  bk      %d\n", best_bk);
    fprintf(stderr, "  reg     %d\n", best_reg);
    fprintf(stderr, "  threads %d\n", thr);
    fprintf(stderr, "  shared  %u\n", smem);
    fprintf(stderr, "  blk/sm  %u\n", bpsm);
    fprintf(stderr, "  occup   %u%%\n", occ);

    // ── AdamW block search ──
    // Largest block where warps_per_block divides evenly into warps_per_sm.
    uint32_t adamw_block = hw->subgroup_size;
    for (uint32_t t = hw->max_workgroup_invocations; t >= hw->subgroup_size; t -= hw->subgroup_size) {
        uint32_t w = t / hw->subgroup_size;
        if (w == 0) continue;
        uint32_t concurrent = warps_per_sm / w;
        if (concurrent * w == warps_per_sm) { adamw_block = t; break; }
    }
    uint32_t aw_warps = adamw_block / hw->subgroup_size;
    uint32_t aw_bpsm = warps_per_sm / aw_warps;
    uint32_t aw_occ = (100 * aw_bpsm * aw_warps) / warps_per_sm;

    fprintf(stderr, "\n  adamw\n");
    fprintf(stderr, "  block   %u\n", adamw_block);
    fprintf(stderr, "  warps   %u\n", aw_warps);
    fprintf(stderr, "  blk/sm  %u\n", aw_bpsm);
    fprintf(stderr, "  occup   %u%%\n", aw_occ);

    // Emit defines — only what's actually consumed by a shader compile.
    printf("-DMATMUL_TILE=%d -DMATMUL_REG=%d -DMATMUL_BK=%d",
           best_tile, best_reg, best_bk);
    printf(" -DADAMW_BLOCK=%u", adamw_block);
}

int main(int argc, char** argv) {
    vk_hw_t hw = {0};
    if (query_vk(&hw) != 0) return 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--defines") == 0) {
            const char* model_dir = NULL;
            if (i + 1 < argc && argv[i+1][0] != '-') model_dir = argv[++i];
            if (!model_dir) {
                fprintf(stderr, "usage: %s --defines <model_dir>\n", argv[0]);
                return 1;
            }
            print_hw(&hw);
            print_defines(&hw, model_dir);
            return 0;
        }
    }

    print_hw(&hw);
    return 0;
}
