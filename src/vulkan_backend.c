// vulkan_backend.c - Seraph Vulkan Training Backend
// Pure compute backend for LLM training on any GPU

#include "../include/vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <math.h>
#include <time.h>

// ═══════════════════════════════════════════════════════════════════════════
// PROFILER — GPU-side timestamps via VkQueryPool
// ═══════════════════════════════════════════════════════════════════════════

#define VULKAN_PROFILER_MAX_OPS 1024

typedef struct {
    char name[80];
    float elapsed_ms;       // filled from GPU timestamps after sync
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
} vulkan_profiler_op_t;

#define PROF_LOG(ctx, fmt, ...) do { \
    printf(fmt, ##__VA_ARGS__); \
    if ((ctx)->prof_log) { fprintf((FILE*)(ctx)->prof_log, fmt, ##__VA_ARGS__); fflush((FILE*)(ctx)->prof_log); } \
} while(0)

// Register op name + allocate index (call at function entry, before cmd exists)
#define PROF_OP_BEGIN(ctx, name_str) \
    int _prof_idx = -1; \
    (void)_prof_idx; \
    if ((ctx) && (ctx)->profiling && (ctx)->prof_ops && \
        (ctx)->prof_num_ops < VULKAN_PROFILER_MAX_OPS) { \
        _prof_idx = (ctx)->prof_num_ops; \
        vulkan_profiler_op_t* _op = &((vulkan_profiler_op_t*)(ctx)->prof_ops)[_prof_idx]; \
        strncpy(_op->name, name_str, sizeof(_op->name) - 1); \
        _op->name[sizeof(_op->name) - 1] = '\0'; \
        _op->elapsed_ms = 0; \
        _op->grid_x = _op->grid_y = _op->grid_z = 0; \
        _op->block_x = _op->block_y = _op->block_z = 0; \
    }

// Store grid/block metadata (call before dispatch)
#define PROF_OP_LAUNCH(gx, gy, gz, bx, by, bz) \
    if (_prof_idx >= 0) { \
        vulkan_profiler_op_t* _op = &((vulkan_profiler_op_t*)(ctx)->prof_ops)[_prof_idx]; \
        _op->grid_x = gx; _op->grid_y = gy; _op->grid_z = gz; \
        _op->block_x = bx; _op->block_y = by; _op->block_z = bz; \
    }

// Write start timestamp into command buffer (call right before vkCmdDispatch)
#define PROF_GPU_BEGIN(ctx, cmd) \
    if (_prof_idx >= 0 && (ctx)->prof_query_pool) { \
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, \
                           (VkQueryPool)(ctx)->prof_query_pool, (uint32_t)(_prof_idx * 2)); \
    }

// Write end timestamp + increment op count (call after barrier, before submit)
#define PROF_GPU_END(ctx, cmd) \
    if (_prof_idx >= 0 && (ctx)->prof_query_pool) { \
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, \
                           (VkQueryPool)(ctx)->prof_query_pool, (uint32_t)(_prof_idx * 2 + 1)); \
        (ctx)->prof_num_ops++; \
    }

void vulkan_profiler_enable(vulkan_context_t* ctx) {
    if (!ctx) return;
    ctx->profiling = 1;
    if (!ctx->prof_ops) {
        ctx->prof_ops = calloc(VULKAN_PROFILER_MAX_OPS, sizeof(vulkan_profiler_op_t));
    }
    if (!ctx->prof_query_pool) {
        VkQueryPoolCreateInfo qp_info = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = VULKAN_PROFILER_MAX_OPS * 2,
        };
        VkQueryPool qp;
        vkCreateQueryPool((VkDevice)ctx->device, &qp_info, NULL, &qp);
        ctx->prof_query_pool = qp;
    }
}

void vulkan_profiler_disable(vulkan_context_t* ctx) {
    if (!ctx) return;
    ctx->profiling = 0;
}

void vulkan_profiler_set_log(vulkan_context_t* ctx, void* file_ptr) {
    if (ctx) ctx->prof_log = file_ptr;
}

void vulkan_profiler_begin_step(vulkan_context_t* ctx) {
    if (!ctx) return;
    ctx->prof_num_ops = 0;
    if (ctx->prof_query_pool) {
        vkResetQueryPool((VkDevice)ctx->device,
                         (VkQueryPool)ctx->prof_query_pool,
                         0, VULKAN_PROFILER_MAX_OPS * 2);
    }
}

void vulkan_profiler_end_step(vulkan_context_t* ctx, int step, int seq_len, float loss) {
    if (!ctx || !ctx->profiling || !ctx->prof_ops || !ctx->prof_query_pool) return;
    int nops = ctx->prof_num_ops;
    if (nops == 0) return;

    vulkan_profiler_op_t* ops = (vulkan_profiler_op_t*)ctx->prof_ops;

    // Read GPU timestamps (wait for all queries to be available)
    uint64_t* timestamps = (uint64_t*)malloc(nops * 2 * sizeof(uint64_t));
    vkGetQueryPoolResults((VkDevice)ctx->device,
                          (VkQueryPool)ctx->prof_query_pool,
                          0, (uint32_t)(nops * 2),
                          nops * 2 * sizeof(uint64_t), timestamps,
                          sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    // Convert to ms using timestamp_period (ns per tick)
    double ns_per_tick = (double)ctx->timestamp_period;
    double total_ms = 0;
    double cat_matmul = 0, cat_attn = 0, cat_attn_bwd = 0;
    double cat_rmsnorm = 0, cat_rmsnorm_bwd = 0;
    double cat_cross_entropy = 0, cat_adamw = 0, cat_copy = 0, cat_other = 0;

    for (int i = 0; i < nops; i++) {
        uint64_t t0 = timestamps[i * 2];
        uint64_t t1 = timestamps[i * 2 + 1];
        double elapsed_ns = (double)(t1 - t0) * ns_per_tick;
        ops[i].elapsed_ms = (float)(elapsed_ns / 1e6);
        total_ms += ops[i].elapsed_ms;

        const char* n = ops[i].name;
        double ms = ops[i].elapsed_ms;
        if (strstr(n, "matmul"))           cat_matmul += ms;
        else if (strstr(n, "attn_bwd"))    cat_attn_bwd += ms;
        else if (strstr(n, "attn"))        cat_attn += ms;
        else if (strstr(n, "rmsnorm_bwd")) cat_rmsnorm_bwd += ms;
        else if (strstr(n, "rmsnorm"))     cat_rmsnorm += ms;
        else if (strstr(n, "cross_ent"))   cat_cross_entropy += ms;
        else if (strstr(n, "adamw"))       cat_adamw += ms;
        else if (strstr(n, "copy") || strstr(n, "fill") || strstr(n, "barrier"))
                                           cat_copy += ms;
        else                               cat_other += ms;
    }
    free(timestamps);

    if (total_ms <= 0) total_ms = 1e-6;

    PROF_LOG(ctx, "\n");
    PROF_LOG(ctx, "  ══════════════════════════════════════════════════════════════════\n");
    PROF_LOG(ctx, "  SERAPH VULKAN STEP PROFILER (step %d, seq=%d, loss=%.4f)\n", step, seq_len, loss);
    PROF_LOG(ctx, "  %d operations, %.2f ms total\n", nops, total_ms);
    PROF_LOG(ctx, "  ══════════════════════════════════════════════════════════════════\n");

    PROF_LOG(ctx, "\n  ── CATEGORY BREAKDOWN ──\n\n");
    PROF_LOG(ctx, "  %-24s %10s  %6s  %s\n", "Category", "Time (ms)", "%", "Bar");
    PROF_LOG(ctx, "  %-24s %10s  %6s  %s\n", "------------------------", "----------", "------",
             "----------------------------------------");

    struct { const char* name; double ms; } cats[] = {
        {"matmul (all)",   cat_matmul},
        {"attention fwd",  cat_attn},
        {"attention bwd",  cat_attn_bwd},
        {"rmsnorm fwd",    cat_rmsnorm},
        {"rmsnorm bwd",    cat_rmsnorm_bwd},
        {"cross_entropy",  cat_cross_entropy},
        {"adamw",          cat_adamw},
        {"copy/fill",      cat_copy},
        {"other",          cat_other},
    };
    int ncats = sizeof(cats) / sizeof(cats[0]);

    for (int i = 0; i < ncats; i++) {
        double pct = 100.0 * cats[i].ms / total_ms;
        int bar_len = (int)(pct * 0.4);
        if (bar_len > 40) bar_len = 40;
        char bar[41];
        for (int b = 0; b < bar_len; b++) bar[b] = '#';
        bar[bar_len] = '\0';
        PROF_LOG(ctx, "  %-24s %10.3f  %5.1f%%  %s\n", cats[i].name, cats[i].ms, pct, bar);
    }

    PROF_LOG(ctx, "\n  ── PER-OPERATION DETAIL ──\n\n");
    PROF_LOG(ctx, "  %4s  %-60s %10s  %5s  %s\n", "#", "Operation", "Time (ms)", "%", "Grid / Block");
    PROF_LOG(ctx, "  %4s  %-60s %10s  %5s  %s\n",
             "----", "------------------------------------------------------------",
             "----------", "-----", "------------------");

    for (int i = 0; i < nops; i++) {
        vulkan_profiler_op_t* op = &ops[i];
        double pct = 100.0 * op->elapsed_ms / total_ms;
        char launch[40] = "";
        if (op->block_x > 0) {
            if (op->grid_y > 1 || op->block_y > 1) {
                snprintf(launch, sizeof(launch), "(%d,%d)/(%d,%d)",
                         op->grid_x, op->grid_y, op->block_x, op->block_y);
            } else {
                snprintf(launch, sizeof(launch), "%d/%d", op->grid_x, op->block_x);
            }
        }
        const char* marker = (pct > 5.0) ? " <<<" : (pct > 1.0) ? " <<" : "";
        PROF_LOG(ctx, "  %4d  %-60s %10.3f  %4.1f%%  %-20s%s\n",
                 i, op->name, op->elapsed_ms, pct, launch, marker);
    }
    PROF_LOG(ctx, "\n");
    fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════════════
// VULKAN INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

vulkan_context_t* vulkan_init(int pool_size_hint) {
    return vulkan_init_ex(pool_size_hint, -1);
}

static int device_has_compute_queue(VkPhysicalDevice physical_device, uint32_t* out_queue_family) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, NULL);
    if (queue_family_count == 0) return 0;
    VkQueueFamilyProperties* queue_families = malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    if (!queue_families) return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families);

    uint32_t compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    free(queue_families);
    if (compute_queue_family == UINT32_MAX) return 0;
    if (out_queue_family) *out_queue_family = compute_queue_family;
    return 1;
}

static int score_device(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);

    int score = 0;
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score += 1000; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 500; break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score += 250; break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: score += 100; break;
        default: break;
    }
    // Prefer higher Vulkan API versions when possible.
    score += (int)VK_VERSION_MAJOR(props.apiVersion) * 10;
    score += (int)VK_VERSION_MINOR(props.apiVersion);

    return score;
}

vulkan_context_t* vulkan_init_ex(int pool_size_hint, int device_index) {
    vulkan_context_t* ctx = calloc(1, sizeof(vulkan_context_t));

    printf("🔥 Initializing Seraph Vulkan backend...\n");

    // Create instance
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Seraph",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Seraph",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };

    VkInstance instance;
    VkResult result = vkCreateInstance(&instance_info, NULL, &instance);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to create Vulkan instance: %d\n", result);
        free(ctx);
        return NULL;
    }
    ctx->instance = instance;

    // Find compute-capable physical device
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (device_count == 0) {
        fprintf(stderr, "ERROR: No Vulkan devices found\n");
        vkDestroyInstance(instance, NULL);
        free(ctx);
        return NULL;
    }

    VkPhysicalDevice* devices = malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &device_count, devices);

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t compute_queue_family = UINT32_MAX;

    if (device_index >= 0 && (uint32_t)device_index < device_count) {
        physical_device = devices[(uint32_t)device_index];
        if (!device_has_compute_queue(physical_device, &compute_queue_family)) {
            fprintf(stderr, "ERROR: Selected Vulkan device %d has no compute queue\n", device_index);
            free(devices);
            vkDestroyInstance(instance, NULL);
            free(ctx);
            return NULL;
        }
    } else {
        int best_score = -1;
        for (uint32_t i = 0; i < device_count; i++) {
            uint32_t qfam = UINT32_MAX;
            if (!device_has_compute_queue(devices[i], &qfam)) continue;
            int s = score_device(devices[i]);
            if (s > best_score) {
                best_score = s;
                physical_device = devices[i];
                compute_queue_family = qfam;
            }
        }
        if (physical_device == VK_NULL_HANDLE) {
            fprintf(stderr, "ERROR: No compute-capable Vulkan devices found\n");
            free(devices);
            vkDestroyInstance(instance, NULL);
            free(ctx);
            return NULL;
        }
    }
    ctx->physical_device = physical_device;

    // Get device properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);
    printf("  GPU: %s\n", props.deviceName);
    printf("  Vulkan: %d.%d.%d\n",
           VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));

    // ── Populate GPU hardware capabilities ──
    strncpy(ctx->gpu_name, props.deviceName, sizeof(ctx->gpu_name) - 1);
    ctx->api_version = props.apiVersion;
    ctx->driver_version = props.driverVersion;
    ctx->vendor_id = props.vendorID;
    ctx->device_id = props.deviceID;
    ctx->device_type = props.deviceType;

    // Compute limits
    ctx->max_workgroup_size[0] = props.limits.maxComputeWorkGroupSize[0];
    ctx->max_workgroup_size[1] = props.limits.maxComputeWorkGroupSize[1];
    ctx->max_workgroup_size[2] = props.limits.maxComputeWorkGroupSize[2];
    ctx->max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    ctx->max_workgroup_count[0] = props.limits.maxComputeWorkGroupCount[0];
    ctx->max_workgroup_count[1] = props.limits.maxComputeWorkGroupCount[1];
    ctx->max_workgroup_count[2] = props.limits.maxComputeWorkGroupCount[2];
    ctx->max_shared_memory = props.limits.maxComputeSharedMemorySize;
    ctx->max_push_constant_size = props.limits.maxPushConstantsSize;
    ctx->max_bound_descriptor_sets = props.limits.maxBoundDescriptorSets;
    ctx->max_storage_buffer_range = props.limits.maxStorageBufferRange;
    ctx->min_storage_buffer_offset = props.limits.minStorageBufferOffsetAlignment;
    ctx->min_uniform_buffer_offset = props.limits.minUniformBufferOffsetAlignment;
    ctx->max_storage_buffers_per_stage = props.limits.maxPerStageDescriptorStorageBuffers;
    ctx->timestamp_period = props.limits.timestampPeriod;
    ctx->timestamp_valid_bits = 0;  // filled per queue family below

    // Timestamp valid bits from queue family
    {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, NULL);
        VkQueueFamilyProperties* qf_props = malloc(qf_count * sizeof(VkQueueFamilyProperties));
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, qf_props);
        if (compute_queue_family < qf_count) {
            ctx->timestamp_valid_bits = qf_props[compute_queue_family].timestampValidBits;
        }
        free(qf_props);
    }

    // Check for vendor-specific extensions
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, NULL, &ext_count, NULL);
    VkExtensionProperties* exts = malloc(ext_count * sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(physical_device, NULL, &ext_count, exts);

    int has_sm_builtins = 0;
    int has_subgroup_size_control = 0;
    int has_shader_atomic_float = 0;
    int has_cooperative_matrix = 0;
    int has_shader_clock = 0;
    int has_memory_budget = 0;
    for (uint32_t i = 0; i < ext_count; i++) {
        if (strcmp(exts[i].extensionName, "VK_NV_shader_sm_builtins") == 0) has_sm_builtins = 1;
        if (strcmp(exts[i].extensionName, "VK_EXT_subgroup_size_control") == 0) has_subgroup_size_control = 1;
        if (strcmp(exts[i].extensionName, "VK_EXT_shader_atomic_float") == 0) has_shader_atomic_float = 1;
        if (strcmp(exts[i].extensionName, "VK_KHR_cooperative_matrix") == 0) has_cooperative_matrix = 1;
        if (strcmp(exts[i].extensionName, "VK_KHR_shader_clock") == 0) has_shader_clock = 1;
        if (strcmp(exts[i].extensionName, "VK_EXT_memory_budget") == 0) has_memory_budget = 1;
    }
    free(exts);

    // Subgroup + SM builtins + subgroup size control (chained pNext)
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV sm_builtin_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV,
    };
    VkPhysicalDeviceSubgroupSizeControlProperties sg_ctrl_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
        .pNext = has_sm_builtins ? &sm_builtin_props : NULL,
    };
    VkPhysicalDeviceSubgroupProperties subgroup_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        .pNext = has_subgroup_size_control ? &sg_ctrl_props : (has_sm_builtins ? (void*)&sm_builtin_props : NULL),
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_props,
    };
    vkGetPhysicalDeviceProperties2(physical_device, &props2);

    ctx->subgroup_size = subgroup_props.subgroupSize;
    ctx->subgroup_supported_stages = subgroup_props.supportedStages;
    ctx->subgroup_supported_ops = subgroup_props.supportedOperations;
    ctx->subgroup_quad_ops = subgroup_props.quadOperationsInAllStages;
    ctx->max_subgroups_per_workgroup = ctx->subgroup_size > 0 ?
        ctx->max_workgroup_invocations / ctx->subgroup_size : 0;

    // Subgroup size control
    if (has_subgroup_size_control) {
        ctx->min_subgroup_size = sg_ctrl_props.minSubgroupSize;
        ctx->max_subgroup_size = sg_ctrl_props.maxSubgroupSize;
    }

    // SM builtins (NVIDIA: sm_count + warps_per_sm)
    if (has_sm_builtins) {
        ctx->max_compute_units = sm_builtin_props.shaderSMCount;
        ctx->warps_per_compute_unit = sm_builtin_props.shaderWarpsPerSM;
        printf("  SM count: %u | Warps/SM: %u\n",
               sm_builtin_props.shaderSMCount, sm_builtin_props.shaderWarpsPerSM);
    }

    // Store extension flags
    ctx->has_sm_builtins = has_sm_builtins;
    ctx->has_shader_atomic_float = has_shader_atomic_float;
    ctx->has_cooperative_matrix = has_cooperative_matrix;
    ctx->has_shader_clock = has_shader_clock;

    // Memory heaps — find VRAM and shared system memory
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    ctx->vram_bytes = 0;
    ctx->shared_system_bytes = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            ctx->vram_bytes += mem_props.memoryHeaps[i].size;
        } else {
            ctx->shared_system_bytes += mem_props.memoryHeaps[i].size;
        }
    }

    // Print hardware summary
    printf("  Subgroup: %u (stages=0x%x ops=0x%x)\n",
           ctx->subgroup_size, ctx->subgroup_supported_stages, ctx->subgroup_supported_ops);
    printf("  Workgroup: %u max invocations, [%u,%u,%u] max size\n",
           ctx->max_workgroup_invocations,
           ctx->max_workgroup_size[0], ctx->max_workgroup_size[1], ctx->max_workgroup_size[2]);
    printf("  Shared mem: %u bytes | Push constants: %u bytes\n",
           ctx->max_shared_memory, ctx->max_push_constant_size);
    printf("  VRAM: %.1f MB | Timestamp: %.1f ns/tick (%u bits)\n",
           (double)ctx->vram_bytes / (1024.0 * 1024.0),
           ctx->timestamp_period, ctx->timestamp_valid_bits);

    // ── Derive workgroup sizes from hardware ──
    ctx->wg_size_attn = ctx->subgroup_size;

    // 1D elementwise (AdamW search): largest block where subgroups_per_block
    // divides evenly into max_subgroups_per_CU = 100% occupancy
    {
        uint32_t max_sg_per_cu = ctx->warps_per_compute_unit > 0
            ? ctx->warps_per_compute_unit
            : ctx->max_workgroup_invocations / ctx->subgroup_size;
        ctx->wg_size_1d = ctx->subgroup_size;
        for (uint32_t t = ctx->max_workgroup_invocations; t >= ctx->subgroup_size; t -= ctx->subgroup_size) {
            uint32_t sg = t / ctx->subgroup_size;
            if ((max_sg_per_cu / sg) * sg == max_sg_per_cu) { ctx->wg_size_1d = t; break; }
        }
    }

    // Matmul tile search: sweep tile×reg×bk scored by SM occupancy
    {
        uint32_t warps_per_sm = ctx->warps_per_compute_unit > 0
            ? ctx->warps_per_compute_unit
            : ctx->max_workgroup_invocations / ctx->subgroup_size;
        uint32_t max_threads_per_sm = warps_per_sm * ctx->subgroup_size;

        uint32_t best_tile = 0, best_reg = 0, best_bk = 0;
        uint32_t best_occ = 0;

        for (uint32_t tile = ctx->subgroup_size; tile <= 256; tile += ctx->subgroup_size) {
            for (uint32_t reg = 2; reg <= 8; reg *= 2) {
                if (tile % reg != 0) continue;
                uint32_t tpb = (tile / reg) * (tile / reg);
                if (tpb > ctx->max_workgroup_invocations) continue;
                if (tpb < ctx->subgroup_size) continue;

                uint32_t wg_dim = tile / reg;
                if (wg_dim > ctx->max_workgroup_size[0] || wg_dim > ctx->max_workgroup_size[1]) continue;

                uint32_t blocks_per_sm = max_threads_per_sm / tpb;

                for (uint32_t bk = 64; bk >= 8; bk /= 2) {
                    uint32_t smem = 4 * (tile * (bk + 1) + bk * (tile + 1));
                    if (smem > ctx->max_shared_memory) continue;

                    uint32_t active = blocks_per_sm * tpb;
                    uint32_t occ = (100 * active) / max_threads_per_sm;

                    if (occ > best_occ || (occ == best_occ && bk > best_bk)) {
                        best_occ = occ;
                        best_tile = tile;
                        best_reg = reg;
                        best_bk = bk;
                    }
                    break;
                }
            }
        }

        if (best_tile == 0) {
            fprintf(stderr, "ERROR: matmul tile search found no valid config (subgroup=%u max_inv=%u shared=%u wg=[%u,%u])\n",
                    ctx->subgroup_size, ctx->max_workgroup_invocations,
                    ctx->max_shared_memory, ctx->max_workgroup_size[0], ctx->max_workgroup_size[1]);
            free(ctx);
            return NULL;
        }

        ctx->wg_size_matmul = best_tile;
        ctx->matmul_reg = best_reg;
        ctx->matmul_bk = best_bk;

        uint32_t thr = (best_tile / best_reg) * (best_tile / best_reg);
        uint32_t bpsm = max_threads_per_sm / thr;

        printf("  Derived workgroups: 1D=%u attn=%u matmul: tile=%u reg=%u bk=%u threads=%u blk/sm=%u occup=%u%%\n",
               ctx->wg_size_1d, ctx->wg_size_attn, best_tile, best_reg, best_bk, thr, bpsm, best_occ);
    }

    ctx->queue_family = compute_queue_family;

    // Create logical device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    // Enable available extensions
    const char* enabled_exts[16];
    uint32_t num_enabled = 0;
    if (has_sm_builtins) enabled_exts[num_enabled++] = "VK_NV_shader_sm_builtins";
    if (has_subgroup_size_control) enabled_exts[num_enabled++] = "VK_EXT_subgroup_size_control";
    if (has_shader_atomic_float) enabled_exts[num_enabled++] = "VK_EXT_shader_atomic_float";
    if (has_cooperative_matrix) enabled_exts[num_enabled++] = "VK_KHR_cooperative_matrix";
    if (has_shader_clock) enabled_exts[num_enabled++] = "VK_KHR_shader_clock";
    if (has_memory_budget) enabled_exts[num_enabled++] = "VK_EXT_memory_budget";

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = num_enabled,
        .ppEnabledExtensionNames = num_enabled > 0 ? enabled_exts : NULL,
    };

    VkDevice device;
    result = vkCreateDevice(physical_device, &device_info, NULL, &device);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to create logical device: %d\n", result);
        free(devices);
        vkDestroyInstance(instance, NULL);
        free(ctx);
        return NULL;
    }
    ctx->device = device;
    free(devices);

    // Get compute queue
    VkQueue queue;
    vkGetDeviceQueue(device, compute_queue_family, 0, &queue);
    ctx->queue = queue;

    // Create command pool
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };

    VkCommandPool command_pool;
    result = vkCreateCommandPool(device, &pool_info, NULL, &command_pool);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to create command pool: %d\n", result);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        free(ctx);
        return NULL;
    }
    ctx->command_pool = command_pool;

    // Create shared descriptor pool (sized for actual model)
    // Each tensor update needs 1 descriptor set
    // Default to 500 if no hint provided
    int pool_size = (pool_size_hint > 0) ? pool_size_hint : 500;

    // Double it for safety (descriptor sets freed per call, but async ops)
    pool_size *= 2;

    VkDescriptorPoolSize desc_pool_sizes[] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = pool_size * 8},  // up to ~7 buffers per set
    };

    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = 0,  // No individual frees - use pool reset (bump allocator, much faster)
        .maxSets = pool_size,
        .poolSizeCount = 1,
        .pPoolSizes = desc_pool_sizes,
    };

    printf("  Descriptor pool: %d sets (model-adaptive)\n", pool_size);

    VkDescriptorPool desc_pool;
    result = vkCreateDescriptorPool(device, &desc_pool_info, NULL, &desc_pool);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to create descriptor pool: %d\n", result);
        vkDestroyCommandPool(device, command_pool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        free(ctx);
        return NULL;
    }
    ctx->descriptor_pool = desc_pool;

    ctx->initialized = 1;
    printf("  ✅ Vulkan backend ready\n\n");

    return ctx;
}

void vulkan_free(vulkan_context_t* ctx) {
    if (!ctx) return;

    VkDevice device = (VkDevice)ctx->device;
    VkInstance instance = (VkInstance)ctx->instance;
    VkCommandPool command_pool = (VkCommandPool)ctx->command_pool;

    if (device) {
        vkDeviceWaitIdle(device);

        // Cleanup cached pipelines
        if (ctx->matmul_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->matmul_pipeline, NULL);
        }
        if (ctx->matmul_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->matmul_layout, NULL);
        }
        if (ctx->matmul_t_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->matmul_t_pipeline, NULL);
        }
        if (ctx->matmul_t_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->matmul_t_layout, NULL);
        }
        if (ctx->matmul_t_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->matmul_t_desc_layout, NULL);
        }
        if (ctx->matmul_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->matmul_desc_layout, NULL);
        }

        if (ctx->adamw_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->adamw_pipeline, NULL);
        }
        if (ctx->adamw_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->adamw_layout, NULL);
        }
        if (ctx->adamw_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->adamw_desc_layout, NULL);
        }

        if (ctx->softmax_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->softmax_pipeline, NULL);
        }
        if (ctx->softmax_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->softmax_layout, NULL);
        }
        if (ctx->softmax_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->softmax_desc_layout, NULL);
        }

        if (ctx->add_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->add_pipeline, NULL);
        }
        if (ctx->add_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->add_layout, NULL);
        }
        if (ctx->add_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->add_desc_layout, NULL);
        }

        if (ctx->mul_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->mul_pipeline, NULL);
        }
        if (ctx->mul_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->mul_layout, NULL);
        }
        if (ctx->mul_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->mul_desc_layout, NULL);
        }

        if (ctx->silu_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->silu_pipeline, NULL);
        }
        if (ctx->silu_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->silu_layout, NULL);
        }
        if (ctx->silu_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->silu_desc_layout, NULL);
        }

        if (ctx->batch_attn_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->batch_attn_pipeline, NULL);
        }
        if (ctx->batch_attn_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->batch_attn_layout, NULL);
        }
        if (ctx->batch_attn_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->batch_attn_desc_layout, NULL);
        }
        if (ctx->batch_attn_stream_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->batch_attn_stream_pipeline, NULL);
        }
        if (ctx->batch_attn_stream_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->batch_attn_stream_layout, NULL);
        }
        if (ctx->batch_attn_stream_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->batch_attn_stream_desc_layout, NULL);
        }

        if (ctx->rope_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->rope_pipeline, NULL);
        }
        if (ctx->rope_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->rope_layout, NULL);
        }
        if (ctx->rope_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->rope_desc_layout, NULL);
        }

        if (ctx->rmsnorm_pipeline) {
            vkDestroyPipeline(device, (VkPipeline)ctx->rmsnorm_pipeline, NULL);
        }
        if (ctx->rmsnorm_layout) {
            vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->rmsnorm_layout, NULL);
        }
        if (ctx->rmsnorm_desc_layout) {
            vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->rmsnorm_desc_layout, NULL);
        }

        // Backward/Training pipeline cleanup
        if (ctx->silu_backward_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->silu_backward_pipeline, NULL);
        if (ctx->silu_backward_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->silu_backward_layout, NULL);
        if (ctx->silu_backward_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->silu_backward_desc_layout, NULL);

        if (ctx->mul_backward_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->mul_backward_pipeline, NULL);
        if (ctx->mul_backward_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->mul_backward_layout, NULL);
        if (ctx->mul_backward_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->mul_backward_desc_layout, NULL);

        if (ctx->rmsnorm_bwd_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->rmsnorm_bwd_pipeline, NULL);
        if (ctx->rmsnorm_bwd_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->rmsnorm_bwd_layout, NULL);
        if (ctx->rmsnorm_bwd_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->rmsnorm_bwd_desc_layout, NULL);

        if (ctx->embed_lookup_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->embed_lookup_pipeline, NULL);
        if (ctx->embed_lookup_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->embed_lookup_layout, NULL);
        if (ctx->embed_lookup_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->embed_lookup_desc_layout, NULL);

        if (ctx->cross_entropy_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->cross_entropy_pipeline, NULL);
        if (ctx->cross_entropy_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->cross_entropy_layout, NULL);
        if (ctx->cross_entropy_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->cross_entropy_desc_layout, NULL);

        if (ctx->reduce_sum_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->reduce_sum_pipeline, NULL);
        if (ctx->reduce_sum_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->reduce_sum_layout, NULL);
        if (ctx->reduce_sum_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->reduce_sum_desc_layout, NULL);

        if (ctx->rope_backward_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->rope_backward_pipeline, NULL);
        if (ctx->rope_backward_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->rope_backward_layout, NULL);
        if (ctx->rope_backward_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->rope_backward_desc_layout, NULL);

        if (ctx->batch_attn_backward_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->batch_attn_backward_pipeline, NULL);
        if (ctx->batch_attn_backward_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->batch_attn_backward_layout, NULL);
        if (ctx->batch_attn_backward_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->batch_attn_backward_desc_layout, NULL);

        if (ctx->embedding_backward_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->embedding_backward_pipeline, NULL);
        if (ctx->embedding_backward_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->embedding_backward_layout, NULL);
        if (ctx->embedding_backward_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->embedding_backward_desc_layout, NULL);

        if (ctx->batch_attn_backward_stream_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->batch_attn_backward_stream_pipeline, NULL);
        if (ctx->batch_attn_backward_stream_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->batch_attn_backward_stream_layout, NULL);
        if (ctx->batch_attn_backward_stream_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->batch_attn_backward_stream_desc_layout, NULL);

        // Debug NaN/Inf checker
        if (ctx->debug_nan_flags) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->debug_nan_flags);
            ctx->debug_nan_flags = NULL;
        }
        if (ctx->nan_check_pipeline) vkDestroyPipeline(device, (VkPipeline)ctx->nan_check_pipeline, NULL);
        if (ctx->nan_check_layout) vkDestroyPipelineLayout(device, (VkPipelineLayout)ctx->nan_check_layout, NULL);
        if (ctx->nan_check_desc_layout) vkDestroyDescriptorSetLayout(device, (VkDescriptorSetLayout)ctx->nan_check_desc_layout, NULL);
        ctx->nan_check_pipeline = NULL;
        ctx->nan_check_layout = NULL;
        ctx->nan_check_desc_layout = NULL;

        if (ctx->descriptor_pool) {
            vkDestroyDescriptorPool(device, (VkDescriptorPool)ctx->descriptor_pool, NULL);
        }

        if (ctx->deferred_desc_sets) {
            free(ctx->deferred_desc_sets);
            ctx->deferred_desc_sets = NULL;
            ctx->deferred_desc_count = 0;
            ctx->deferred_desc_cap = 0;
        }

        // Cleanup persistent optimizer buffers
        if (ctx->weight_buffer) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->weight_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->grad_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->m_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->v_buffer);
        }

        // Cleanup persistent forward pass buffers
        if (ctx->fwd_input_buffer) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_input_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_weight_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_output_buffer);
        }

        // Cleanup persistent attention buffers
        if (ctx->attn_q_buffer) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_q_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_k_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_v_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->attn_out_buffer);
        }

        // Cleanup persistent GPU-forward buffers (full training)
        if (ctx->fwd_hidden_buffer) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_hidden_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_hidden_norm_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_tmp_hidden_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_ffn_gate_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_ffn_up_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_ffn_hidden_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_logits_buffer);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->fwd_norm_weight_buffer);
        }

        // Cleanup per-layer activation cache buffers
        if (ctx->fwd_cache_x_norm_attn) {
            for (int l = 0; l < ctx->fwd_cache_num_layers; l++) {
                vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_x_norm_attn)[l]);
                vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_x_norm_ffn)[l]);
                vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_attn_out)[l]);
                vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_ffn_gate_out)[l]);
                vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_ffn_up_out)[l]);
                vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_ffn_hidden)[l]);
                if (ctx->fwd_cache_layer_input) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_layer_input)[l]);
                if (ctx->fwd_cache_post_attn_in) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_post_attn_in)[l]);
                if (ctx->fwd_cache_silu_out) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_silu_out)[l]);
                if (ctx->fwd_cache_q) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_q)[l]);
                if (ctx->fwd_cache_k) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_k)[l]);
                if (ctx->fwd_cache_v) vulkan_buffer_free(ctx, ((vulkan_buffer_t**)ctx->fwd_cache_v)[l]);
            }
            free(ctx->fwd_cache_x_norm_attn);
            free(ctx->fwd_cache_x_norm_ffn);
            free(ctx->fwd_cache_attn_out);
            free(ctx->fwd_cache_ffn_gate_out);
            free(ctx->fwd_cache_ffn_up_out);
            free(ctx->fwd_cache_ffn_hidden);
            if (ctx->fwd_cache_layer_input) free(ctx->fwd_cache_layer_input);
            if (ctx->fwd_cache_post_attn_in) free(ctx->fwd_cache_post_attn_in);
            if (ctx->fwd_cache_silu_out) free(ctx->fwd_cache_silu_out);
            if (ctx->fwd_cache_q) free(ctx->fwd_cache_q);
            if (ctx->fwd_cache_k) free(ctx->fwd_cache_k);
            if (ctx->fwd_cache_v) free(ctx->fwd_cache_v);
            ctx->fwd_cache_x_norm_attn = NULL;
            ctx->fwd_cache_x_norm_ffn = NULL;
            ctx->fwd_cache_attn_out = NULL;
            ctx->fwd_cache_ffn_gate_out = NULL;
            ctx->fwd_cache_ffn_up_out = NULL;
            ctx->fwd_cache_ffn_hidden = NULL;
            ctx->fwd_cache_layer_input = NULL;
            ctx->fwd_cache_post_attn_in = NULL;
            ctx->fwd_cache_silu_out = NULL;
            ctx->fwd_cache_q = NULL;
            ctx->fwd_cache_k = NULL;
            ctx->fwd_cache_v = NULL;
            ctx->fwd_cache_num_layers = 0;
        }

        // Cleanup GPU-resident weights (if allocated)
        if (ctx->gpu_w_embed) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_w_embed);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_g_embed);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_m_embed);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_v_embed);
            ctx->gpu_w_embed = NULL;
            ctx->gpu_g_embed = NULL;
            ctx->gpu_m_embed = NULL;
            ctx->gpu_v_embed = NULL;
        }
        if (ctx->gpu_w_final_norm) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_w_final_norm);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_g_final_norm);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_m_final_norm);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->gpu_v_final_norm);
            ctx->gpu_w_final_norm = NULL;
            ctx->gpu_g_final_norm = NULL;
            ctx->gpu_m_final_norm = NULL;
            ctx->gpu_v_final_norm = NULL;
        }

        if (ctx->gpu_w_q) {
            int n = ctx->gpu_weights_num_layers;
            vulkan_buffer_t** arrays[] = {
                (vulkan_buffer_t**)ctx->gpu_w_q, (vulkan_buffer_t**)ctx->gpu_g_q, (vulkan_buffer_t**)ctx->gpu_m_q, (vulkan_buffer_t**)ctx->gpu_v_q,
                (vulkan_buffer_t**)ctx->gpu_w_k, (vulkan_buffer_t**)ctx->gpu_g_k, (vulkan_buffer_t**)ctx->gpu_m_k, (vulkan_buffer_t**)ctx->gpu_v_k,
                (vulkan_buffer_t**)ctx->gpu_w_v, (vulkan_buffer_t**)ctx->gpu_g_v, (vulkan_buffer_t**)ctx->gpu_m_v, (vulkan_buffer_t**)ctx->gpu_v_v,
                (vulkan_buffer_t**)ctx->gpu_w_o, (vulkan_buffer_t**)ctx->gpu_g_o, (vulkan_buffer_t**)ctx->gpu_m_o, (vulkan_buffer_t**)ctx->gpu_v_o,
                (vulkan_buffer_t**)ctx->gpu_w_gate, (vulkan_buffer_t**)ctx->gpu_g_gate, (vulkan_buffer_t**)ctx->gpu_m_gate, (vulkan_buffer_t**)ctx->gpu_v_gate,
                (vulkan_buffer_t**)ctx->gpu_w_up, (vulkan_buffer_t**)ctx->gpu_g_up, (vulkan_buffer_t**)ctx->gpu_m_up, (vulkan_buffer_t**)ctx->gpu_v_up,
                (vulkan_buffer_t**)ctx->gpu_w_down, (vulkan_buffer_t**)ctx->gpu_g_down, (vulkan_buffer_t**)ctx->gpu_m_down, (vulkan_buffer_t**)ctx->gpu_v_down,
                (vulkan_buffer_t**)ctx->gpu_w_in_norm, (vulkan_buffer_t**)ctx->gpu_g_in_norm, (vulkan_buffer_t**)ctx->gpu_m_in_norm, (vulkan_buffer_t**)ctx->gpu_v_in_norm,
                (vulkan_buffer_t**)ctx->gpu_w_post_norm, (vulkan_buffer_t**)ctx->gpu_g_post_norm, (vulkan_buffer_t**)ctx->gpu_m_post_norm, (vulkan_buffer_t**)ctx->gpu_v_post_norm,
            };

            for (size_t ai = 0; ai < sizeof(arrays) / sizeof(arrays[0]); ai++) {
                if (!arrays[ai]) continue;
                for (int l = 0; l < n; l++) {
                    if (arrays[ai][l]) vulkan_buffer_free(ctx, arrays[ai][l]);
                }
                free(arrays[ai]);
            }

            ctx->gpu_w_q = ctx->gpu_g_q = ctx->gpu_m_q = ctx->gpu_v_q = NULL;
            ctx->gpu_w_k = ctx->gpu_g_k = ctx->gpu_m_k = ctx->gpu_v_k = NULL;
            ctx->gpu_w_v = ctx->gpu_g_v = ctx->gpu_m_v = ctx->gpu_v_v = NULL;
            ctx->gpu_w_o = ctx->gpu_g_o = ctx->gpu_m_o = ctx->gpu_v_o = NULL;
            ctx->gpu_w_gate = ctx->gpu_g_gate = ctx->gpu_m_gate = ctx->gpu_v_gate = NULL;
            ctx->gpu_w_up = ctx->gpu_g_up = ctx->gpu_m_up = ctx->gpu_v_up = NULL;
            ctx->gpu_w_down = ctx->gpu_g_down = ctx->gpu_m_down = ctx->gpu_v_down = NULL;
            ctx->gpu_w_in_norm = ctx->gpu_g_in_norm = ctx->gpu_m_in_norm = ctx->gpu_v_in_norm = NULL;
            ctx->gpu_w_post_norm = ctx->gpu_g_post_norm = ctx->gpu_m_post_norm = ctx->gpu_v_post_norm = NULL;

            ctx->gpu_weights_num_layers = 0;
            ctx->gpu_weights_hidden_size = 0;
            ctx->gpu_weights_intermediate_size = 0;
            ctx->gpu_weights_q_dim = 0;
            ctx->gpu_weights_kv_dim = 0;
            ctx->gpu_weights_vocab_size = 0;
        }

        // Cleanup GPU-only training scratch buffers
        if (ctx->train_tokens_u32) {
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_tokens_u32);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_targets_u32);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_logits);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_loss_rows);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_reduce_tmp_a);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_reduce_tmp_b);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_hidden);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_tmp_hidden);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_x_norm);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_q);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_k);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_v);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_attn);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_ffn);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_gate);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_grad_up);
            vulkan_buffer_free(ctx, (vulkan_buffer_t*)ctx->train_final_input);

            ctx->train_tokens_u32 = NULL;
            ctx->train_targets_u32 = NULL;
            ctx->train_grad_logits = NULL;
            ctx->train_loss_rows = NULL;
            ctx->train_reduce_tmp_a = NULL;
            ctx->train_reduce_tmp_b = NULL;
            ctx->train_grad_hidden = NULL;
            ctx->train_grad_tmp_hidden = NULL;
            ctx->train_grad_x_norm = NULL;
            ctx->train_grad_q = NULL;
            ctx->train_grad_k = NULL;
            ctx->train_grad_v = NULL;
            ctx->train_grad_attn = NULL;
            ctx->train_grad_ffn = NULL;
            ctx->train_grad_gate = NULL;
            ctx->train_grad_up = NULL;
            ctx->train_final_input = NULL;
        }

        if (command_pool) vkDestroyCommandPool(device, command_pool, NULL);
        vkDestroyDevice(device, NULL);
    }
    if (instance) vkDestroyInstance(instance, NULL);

    free(ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// BUFFER MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

// Helper: Find suitable memory type
static uint32_t find_memory_type(VkPhysicalDevice physical_device,
                                  uint32_t type_filter,
                                  VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

vulkan_buffer_t* vulkan_buffer_create(vulkan_context_t* ctx, size_t size) {
    if (!ctx || !ctx->initialized) return NULL;

    VkDevice device = (VkDevice)ctx->device;
    VkPhysicalDevice physical_device = (VkPhysicalDevice)ctx->physical_device;

    vulkan_buffer_t* buf = calloc(1, sizeof(vulkan_buffer_t));
    buf->size = size;

    // Create buffer
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device, &buffer_info, NULL, &buffer);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to create buffer: %d\n", result);
        free(buf);
        return NULL;
    }
    buf->buffer = buffer;

    // Get memory requirements
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

    // Allocate memory (host-visible for easy CPU ↔ GPU transfer)
    uint32_t memory_type = find_memory_type(physical_device,
                                             mem_reqs.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = memory_type,
    };

    VkDeviceMemory memory;
    result = vkAllocateMemory(device, &alloc_info, NULL, &memory);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ERROR: Failed to allocate memory: %d\n", result);
        vkDestroyBuffer(device, buffer, NULL);
        free(buf);
        return NULL;
    }
    buf->memory = memory;

    // Bind buffer to memory
    vkBindBufferMemory(device, buffer, memory, 0);

    // Persistent map (host visible + coherent)
    buf->mapped = NULL;
    if (vkMapMemory(device, memory, 0, mem_reqs.size, 0, &buf->mapped) != VK_SUCCESS) {
        buf->mapped = NULL;
    }

    return buf;
}

void vulkan_buffer_upload(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                          const float* data, size_t count) {
    if (!ctx || !buf) return;
    if (buf->mapped) {
        memcpy(buf->mapped, data, count * sizeof(float));
        return;
    }

    VkDevice device = (VkDevice)ctx->device;
    VkDeviceMemory memory = (VkDeviceMemory)buf->memory;

    void* mapped;
    if (vkMapMemory(device, memory, 0, count * sizeof(float), 0, &mapped) != VK_SUCCESS) return;
    memcpy(mapped, data, count * sizeof(float));
    vkUnmapMemory(device, memory);
}

void vulkan_buffer_upload_bytes(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                                const void* data, size_t bytes) {
    if (!ctx || !buf || !data) return;
    if (bytes > buf->size) bytes = buf->size;
    if (buf->mapped) {
        memcpy(buf->mapped, data, bytes);
        return;
    }

    VkDevice device = (VkDevice)ctx->device;
    VkDeviceMemory memory = (VkDeviceMemory)buf->memory;

    void* mapped;
    if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return;
    memcpy(mapped, data, bytes);
    vkUnmapMemory(device, memory);
}

void vulkan_buffer_upload_u32(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                              const uint32_t* data, size_t count) {
    vulkan_buffer_upload_bytes(ctx, buf, data, count * sizeof(uint32_t));
}

void vulkan_buffer_download(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                            float* data, size_t count) {
    if (!ctx || !buf) return;
    if (buf->mapped) {
        memcpy(data, buf->mapped, count * sizeof(float));
        return;
    }

    VkDevice device = (VkDevice)ctx->device;
    VkDeviceMemory memory = (VkDeviceMemory)buf->memory;

    void* mapped;
    if (vkMapMemory(device, memory, 0, count * sizeof(float), 0, &mapped) != VK_SUCCESS) return;
    memcpy(data, mapped, count * sizeof(float));
    vkUnmapMemory(device, memory);
}

void vulkan_buffer_download_bytes(vulkan_context_t* ctx, vulkan_buffer_t* buf,
                                  void* data, size_t bytes) {
    if (!ctx || !buf || !data) return;
    if (bytes > buf->size) bytes = buf->size;
    if (buf->mapped) {
        memcpy(data, buf->mapped, bytes);
        return;
    }

    VkDevice device = (VkDevice)ctx->device;
    VkDeviceMemory memory = (VkDeviceMemory)buf->memory;

    void* mapped;
    if (vkMapMemory(device, memory, 0, bytes, 0, &mapped) != VK_SUCCESS) return;
    memcpy(data, mapped, bytes);
    vkUnmapMemory(device, memory);
}

void vulkan_buffer_free(vulkan_context_t* ctx, vulkan_buffer_t* buf) {
    if (!ctx || !buf) return;

    VkDevice device = (VkDevice)ctx->device;
    VkBuffer buffer = (VkBuffer)buf->buffer;
    VkDeviceMemory memory = (VkDeviceMemory)buf->memory;

    if (buf->mapped) {
        vkUnmapMemory(device, memory);
        buf->mapped = NULL;
    }

    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    free(buf);
}

// Forward decl (used by transfer helpers below)
static inline VkCommandBuffer get_cmd_for_op(vulkan_context_t* ctx, VkDevice device, VkCommandPool pool, int* owns_cmd);

void vulkan_fill_buffer(vulkan_context_t* ctx,
                        vulkan_buffer_t* buf,
                        uint32_t value,
                        size_t bytes) {
    vulkan_fill_buffer_range(ctx, buf, 0, value, bytes);
}

void vulkan_fill_buffer_range(vulkan_context_t* ctx,
                              vulkan_buffer_t* buf,
                              size_t offset_bytes,
                              uint32_t value,
                              size_t bytes) {
    if (!ctx || !buf || bytes == 0) return;
    if (offset_bytes >= buf->size) return;
    if (bytes > (buf->size - offset_bytes)) bytes = buf->size - offset_bytes;

    // vkCmdFillBuffer requires offset/size to be multiples of 4.
    offset_bytes &= ~((size_t)3);
    bytes &= ~((size_t)3);
    if (bytes == 0) return;

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    // Ensure previous compute writes visible to transfer fill, and fill visible to subsequent compute.
    VkBufferMemoryBarrier pre = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = (VkBuffer)buf->buffer,
        .offset = offset_bytes,
        .size = bytes,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 1, &pre, 0, NULL);

    vkCmdFillBuffer(cmd, (VkBuffer)buf->buffer, offset_bytes, bytes, value);

    VkBufferMemoryBarrier post = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = (VkBuffer)buf->buffer,
        .offset = offset_bytes,
        .size = bytes,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 1, &post, 0, NULL);

    if (owns_cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
        };
        vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(device, pool, 1, &cmd);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIONAL COMMAND BATCHING (API surface only for now)
// ═══════════════════════════════════════════════════════════════════════════

int vulkan_cmd_begin(vulkan_context_t* ctx) {
    if (!ctx || !ctx->initialized) return 0;
    if (ctx->recording) return 1;

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;

    VkCommandBufferAllocateInfo cmd_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &cmd_alloc_info, &cmd) != VK_SUCCESS) {
        return 0;
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, pool, 1, &cmd);
        return 0;
    }

    // Reset descriptor pool - recycles all sets instantly (bump allocator reset)
    // Safe: previous cmd_end_submit already called vkQueueWaitIdle()
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    vkResetDescriptorPool(device, desc_pool, 0);

    ctx->recording_cmd = cmd;
    ctx->recording = 1;
    ctx->deferred_desc_count = 0;
    return 1;
}

int vulkan_cmd_end_submit(vulkan_context_t* ctx) {
    if (!ctx || !ctx->initialized) return 0;
    if (!ctx->recording) return 1;

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    VkCommandBuffer cmd = (VkCommandBuffer)ctx->recording_cmd;
    if (!cmd) {
        ctx->recording = 0;
        return 1;
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return 0;

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) return 0;
    vkQueueWaitIdle(queue);

    // Descriptor sets freed by pool reset in next vulkan_cmd_begin()
    ctx->deferred_desc_count = 0;

    vkFreeCommandBuffers(device, pool, 1, &cmd);
    ctx->recording_cmd = NULL;
    ctx->recording = 0;
    return 1;
}

static void defer_desc_set(vulkan_context_t* ctx, VkDescriptorSet set) {
    // No-op: descriptor sets freed by pool reset in vulkan_cmd_begin()
    // Kept as stub so callsites don't need changing
    (void)ctx; (void)set;
}

static inline void cmd_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1, &barrier,
                         0, NULL,
                         0, NULL);
}

static inline VkCommandBuffer get_cmd_for_op(vulkan_context_t* ctx, VkDevice device, VkCommandPool pool, int* owns_cmd) {
    if (ctx && ctx->recording && ctx->recording_cmd) {
        if (owns_cmd) *owns_cmd = 0;
        return (VkCommandBuffer)ctx->recording_cmd;
    }
    if (owns_cmd) *owns_cmd = 1;
    VkCommandBufferAllocateInfo cmd_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmd_alloc_info, &cmd);
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);
    return cmd;
}

static inline void submit_and_cleanup_if_owned(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd, int owns_cmd) {
    if (!owns_cmd) return;
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

// ═══════════════════════════════════════════════════════════════════════════
// SHADER LOADING
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// SPECIALIZATION CONSTANT HELPERS
// All workgroup sizes are derived from hardware query at init time.
// These helpers create VkSpecializationInfo structs for pipeline creation.
// ═══════════════════════════════════════════════════════════════════════════

// 1D shader specialization: constant_id=0 → local_size_x
typedef struct {
    VkSpecializationMapEntry entries[1];
    VkSpecializationInfo info;
    uint32_t data[1];
} spec_1d_t;

static spec_1d_t make_spec_1d(uint32_t local_size_x) {
    spec_1d_t s;
    s.data[0] = local_size_x;
    s.entries[0] = (VkSpecializationMapEntry){.constantID = 0, .offset = 0, .size = sizeof(uint32_t)};
    s.info = (VkSpecializationInfo){
        .mapEntryCount = 1,
        .pMapEntries = s.entries,
        .dataSize = sizeof(uint32_t),
        .pData = s.data,
    };
    return s;
}

// 2D shader specialization: constant_id=0 → local_size_x, constant_id=1 → local_size_y
typedef struct {
    VkSpecializationMapEntry entries[2];
    VkSpecializationInfo info;
    uint32_t data[2];
} spec_2d_t;

static spec_2d_t make_spec_2d(uint32_t local_size_x, uint32_t local_size_y) {
    spec_2d_t s;
    s.data[0] = local_size_x;
    s.data[1] = local_size_y;
    s.entries[0] = (VkSpecializationMapEntry){.constantID = 0, .offset = 0, .size = sizeof(uint32_t)};
    s.entries[1] = (VkSpecializationMapEntry){.constantID = 1, .offset = sizeof(uint32_t), .size = sizeof(uint32_t)};
    s.info = (VkSpecializationInfo){
        .mapEntryCount = 2,
        .pMapEntries = s.entries,
        .dataSize = 2 * sizeof(uint32_t),
        .pData = s.data,
    };
    return s;
}

// Helper: Load SPIR-V shader from file (tries multiple locations)
uint32_t* load_shader_spirv(const char* shader_name, size_t* size) {
    // Try multiple locations in priority order:
    const char* search_paths[] = {
        NULL,  // SERAPH_SHADER_PATH env var (checked below)
        "shaders/%s",                           // Current dir (build tree)
        "../shaders/%s",                        // Parent dir (installed layout)
        "%s/.local/share/seraph/shaders/%s",   // User install
        "/usr/share/seraph/shaders/%s",        // System install
        "/usr/local/share/seraph/shaders/%s",  // Local install
        NULL
    };

    char path_buf[512];
    FILE* f = NULL;

    // Check environment variable first
    const char* env_path = getenv("SERAPH_SHADER_PATH");
    if (env_path) {
        snprintf(path_buf, sizeof(path_buf), "%s/%s", env_path, shader_name);
        f = fopen(path_buf, "rb");
        if (f) goto found;
    }

    // Try default locations
    for (int i = 1; search_paths[i] != NULL; i++) {
        if (strstr(search_paths[i], "%s/.local")) {
            // Expand $HOME
            const char* home = getenv("HOME");
            if (home) {
                snprintf(path_buf, sizeof(path_buf), search_paths[i], home, shader_name);
            } else {
                continue;
            }
        } else {
            snprintf(path_buf, sizeof(path_buf), search_paths[i], shader_name);
        }

        f = fopen(path_buf, "rb");
        if (f) goto found;
    }

    fprintf(stderr, "ERROR: Cannot find shader: %s\n", shader_name);
    fprintf(stderr, "  Searched: shaders/, ../shaders/, ~/.local/share/seraph/shaders/, /usr/share/seraph/shaders/\n");
    fprintf(stderr, "  Set SERAPH_SHADER_PATH env var to override shader location\n");
    return NULL;

found:
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint32_t* code = malloc(*size);
    fread(code, 1, *size, f);
    fclose(f);

    return code;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPUTE KERNELS
// ═══════════════════════════════════════════════════════════════════════════

void vulkan_nan_check(vulkan_context_t* ctx,
                      vulkan_buffer_t* input,
                      size_t count_floats,
                      vulkan_buffer_t* flags_u32,
                      uint32_t flag_index) {
    if (!ctx || !input || !flags_u32 || count_floats == 0) return;
    PROF_OP_BEGIN(ctx, "nan_check");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->nan_check_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("nan_check.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load nan_check shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        VkDescriptorSetLayoutBinding bindings[2] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->nan_check_desc_layout = desc_layout;

        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 2,
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->nan_check_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->nan_check_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->nan_check_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_infos[2] = {
        {.buffer = (VkBuffer)input->buffer, .offset = 0, .range = input->size},
        {.buffer = (VkBuffer)flags_u32->buffer, .offset = 0, .range = flags_u32->size},
    };

    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->nan_check_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->nan_check_layout,
                             0, 1, &desc_set, 0, NULL);

    struct {
        uint32_t count;
        uint32_t flag_index;
    } push_data = {(uint32_t)count_floats, flag_index};

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->nan_check_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    uint32_t workgroups = ((uint32_t)count_floats + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_matmul(vulkan_context_t* ctx,
                   vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                   int M, int N, int K) {
    if (!ctx || !A || !B || !C) return;
    char _pname[128]; snprintf(_pname, sizeof(_pname), "matmul [%dx%d K=%d]", M, N, K);
    PROF_OP_BEGIN(ctx, _pname);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Create pipeline if not cached
    // ═══════════════════════════════════════════════════════════════════════

    if (!ctx->matmul_pipeline) {
        // Shader pre-compiled with hardware-derived MATMUL_* defines by
        // Makefile via vulkan-profile-query.
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("matmul.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load matmul shader\n");
            return;
        }

        // Create shader module
        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor set layout (3 storage buffers)
        VkDescriptorSetLayoutBinding bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->matmul_desc_layout = desc_layout;

        // Pipeline layout with push constants (M, N, K)
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 3,
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->matmul_layout = layout;

        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->matmul_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Allocate descriptor set and bind buffers
    // ═══════════════════════════════════════════════════════════════════════

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->matmul_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    // Bind buffers to descriptor set
    VkDescriptorBufferInfo buffer_infos[3] = {
        {.buffer = (VkBuffer)A->buffer, .offset = 0, .range = A->size},
        {.buffer = (VkBuffer)B->buffer, .offset = 0, .range = B->size},
        {.buffer = (VkBuffer)C->buffer, .offset = 0, .range = C->size},
    };

    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[2]},
    };

    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Record (optionally batched) dispatch
    // ═══════════════════════════════════════════════════════════════════════

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->matmul_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->matmul_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants (M, N, K)
    uint32_t push_data[3] = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->matmul_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

    // Dispatch: each workgroup covers TILE output rows/cols
    uint32_t tile = ctx->wg_size_matmul;
    uint32_t wg_dim = tile / ctx->matmul_reg;
    uint32_t workgroup_x = (N + tile - 1) / tile;
    uint32_t workgroup_y = (M + tile - 1) / tile;
    PROF_OP_LAUNCH(workgroup_x, workgroup_y, 1, wg_dim, wg_dim, 1);
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroup_x, workgroup_y, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

// ═══════════════════════════════════════════════════════════════════════════
// MATRIX MULTIPLY WITH TRANSPOSE - For batch forward pass
// ═══════════════════════════════════════════════════════════════════════════

void vulkan_matmul_transpose(vulkan_context_t* ctx,
                             vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                             int M, int N, int K,
                             int transpose_A, int transpose_B, int accumulate) {
    if (!ctx || !A || !B || !C) return;
    char _pname[128]; snprintf(_pname, sizeof(_pname), "matmul_t [%dx%d K=%d]", M, N, K);
    PROF_OP_BEGIN(ctx, _pname);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create pipeline on first use (cached)
    if (!ctx->matmul_t_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("matmul_transpose.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load matmul_transpose shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor set layout (3 storage buffers: A, B, C)
        VkDescriptorSetLayoutBinding bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->matmul_t_desc_layout = desc_layout;

        // Push constants: M, N, K, transpose_A, transpose_B, accumulate
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 6,
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->matmul_t_layout = layout;

        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->matmul_t_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // Allocate descriptor set
    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->matmul_t_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_infos[3] = {
        {.buffer = (VkBuffer)A->buffer, .offset = 0, .range = A->size},
        {.buffer = (VkBuffer)B->buffer, .offset = 0, .range = B->size},
        {.buffer = (VkBuffer)C->buffer, .offset = 0, .range = C->size},
    };

    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[2]},
    };

    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->matmul_t_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->matmul_t_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants
    struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t transpose_A;
        uint32_t transpose_B;
        uint32_t accumulate;
    } push_data = {
        (uint32_t)M, (uint32_t)N, (uint32_t)K,
        (uint32_t)transpose_A, (uint32_t)transpose_B, (uint32_t)accumulate
    };

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->matmul_t_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    // Dispatch: each workgroup covers TILE output rows/cols
    uint32_t tile = ctx->wg_size_matmul;
    uint32_t wg_dim = tile / ctx->matmul_reg;
    uint32_t workgroups_x = (N + tile - 1) / tile;
    uint32_t workgroups_y = (M + tile - 1) / tile;
    PROF_OP_LAUNCH(workgroups_x, workgroups_y, 1, wg_dim, wg_dim, 1);
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups_x, workgroups_y, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_adamw_update(vulkan_context_t* ctx,
                         vulkan_buffer_t* weight,
                         vulkan_buffer_t* grad,
                         vulkan_buffer_t* m,
                         vulkan_buffer_t* v,
                         float lr, float beta1, float beta2,
                         float weight_decay, float eps,
                         int step, size_t size) {
    if (!ctx || !weight || !grad || !m || !v) return;
    char _pname_adamw[80];
    snprintf(_pname_adamw, sizeof(_pname_adamw), "adamw [%zu] %u-thr", size, ctx->wg_size_1d);
    PROF_OP_BEGIN(ctx, _pname_adamw);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create pipeline if not cached
    if (!ctx->adamw_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("adamw.spv", &spirv_size);
        if (!spirv) return;

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor layout (4 buffers: weight, grad, m, v)
        VkDescriptorSetLayoutBinding bindings[4] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->adamw_desc_layout = desc_layout;

        // Push constants (lr, betas, etc.)
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(float) * 7 + sizeof(uint32_t),  // 7 floats + 1 uint
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->adamw_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->adamw_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // Allocate descriptor set
    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->adamw_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    // Bind buffers
    VkDescriptorBufferInfo buffer_infos[4] = {
        {.buffer = (VkBuffer)weight->buffer, .offset = 0, .range = weight->size},
        {.buffer = (VkBuffer)grad->buffer, .offset = 0, .range = grad->size},
        {.buffer = (VkBuffer)m->buffer, .offset = 0, .range = m->size},
        {.buffer = (VkBuffer)v->buffer, .offset = 0, .range = v->size},
    };

    VkWriteDescriptorSet writes[4] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[2]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[3]},
    };

    vkUpdateDescriptorSets(device, 4, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->adamw_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->adamw_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants
    float bias_correction1 = 1.0f - powf(beta1, step);
    float bias_correction2 = 1.0f - powf(beta2, step);

    struct {
        float lr;
        float beta1;
        float beta2;
        float weight_decay;
        float eps;
        float bias_correction1;
        float bias_correction2;
        uint32_t size;
    } push_data = {lr, beta1, beta2, weight_decay, eps, bias_correction1, bias_correction2, (uint32_t)size};

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->adamw_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    uint32_t workgroups = (size + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_OP_LAUNCH(workgroups, 1, 1, ctx->wg_size_1d, 1, 1);
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_softmax(vulkan_context_t* ctx,
                    vulkan_buffer_t* input,
                    vulkan_buffer_t* output,
                    int rows, int cols) {
    if (!ctx || !input || !output) return;
    PROF_OP_BEGIN(ctx, "softmax");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create pipeline if not cached
    if (!ctx->softmax_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("softmax.spv", &spirv_size);
        if (!spirv) return;

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor layout (2 buffers: input, output)
        VkDescriptorSetLayoutBinding bindings[2] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->softmax_desc_layout = desc_layout;

        // Push constants (rows, cols)
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 2,
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->softmax_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->softmax_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // Allocate descriptor set
    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->softmax_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    // Bind buffers
    VkDescriptorBufferInfo buffer_infos[2] = {
        {.buffer = (VkBuffer)input->buffer, .offset = 0, .range = input->size},
        {.buffer = (VkBuffer)output->buffer, .offset = 0, .range = output->size},
    };

    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
    };

    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    // Create command buffer
    VkCommandBufferAllocateInfo cmd_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmd_alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->softmax_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->softmax_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants
    uint32_t push_data[2] = {(uint32_t)rows, (uint32_t)cols};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->softmax_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

    // Dispatch (one workgroup per row)
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, rows, 1, 1);

    vkEndCommandBuffer(cmd);

    // Submit
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
    // desc set freed by pool reset
    PROF_GPU_END(ctx, cmd);
}

void vulkan_add(vulkan_context_t* ctx,
                vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                size_t count) {
    PROF_OP_BEGIN(ctx, "add");
    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Create pipeline on first use (cached)
    // ═══════════════════════════════════════════════════════════════════════

    if (!ctx->add_pipeline) {
        // Load shader
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("add.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load add shader\n");
            return;
        }

        // Create shader module
        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor set layout (3 storage buffers: A, B, C)
        VkDescriptorSetLayoutBinding bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->add_desc_layout = desc_layout;

        // Pipeline layout with push constants (size)
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t),
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->add_layout = layout;

        // Compute pipeline
        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->add_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Allocate descriptor set and bind buffers
    // ═══════════════════════════════════════════════════════════════════════

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->add_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    // Bind buffers to descriptor set
    VkDescriptorBufferInfo buffer_infos[3] = {
        {.buffer = (VkBuffer)A->buffer, .offset = 0, .range = A->size},
        {.buffer = (VkBuffer)B->buffer, .offset = 0, .range = B->size},
        {.buffer = (VkBuffer)C->buffer, .offset = 0, .range = C->size},
    };

    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[2]},
    };

    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Record (optionally batched) dispatch
    // ═══════════════════════════════════════════════════════════════════════

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->add_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->add_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants
    uint32_t size = (uint32_t)count;
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->add_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), &size);

    // Dispatch
    uint32_t workgroups = (count + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_mul(vulkan_context_t* ctx,
                vulkan_buffer_t* A, vulkan_buffer_t* B, vulkan_buffer_t* C,
                size_t count) {
    PROF_OP_BEGIN(ctx, "mul");
    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Create pipeline on first use (cached)
    // ═══════════════════════════════════════════════════════════════════════

    if (!ctx->mul_pipeline) {
        // Load shader
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("mul.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load mul shader\n");
            return;
        }

        // Create shader module
        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor set layout (3 storage buffers: A, B, C)
        VkDescriptorSetLayoutBinding bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->mul_desc_layout = desc_layout;

        // Pipeline layout with push constants (size)
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t),
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->mul_layout = layout;

        // Compute pipeline
        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->mul_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Allocate descriptor set and bind buffers
    // ═══════════════════════════════════════════════════════════════════════

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->mul_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    // Bind buffers to descriptor set
    VkDescriptorBufferInfo buffer_infos[3] = {
        {.buffer = (VkBuffer)A->buffer, .offset = 0, .range = A->size},
        {.buffer = (VkBuffer)B->buffer, .offset = 0, .range = B->size},
        {.buffer = (VkBuffer)C->buffer, .offset = 0, .range = C->size},
    };

    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[2]},
    };

    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Record (optionally batched) dispatch
    // ═══════════════════════════════════════════════════════════════════════

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->mul_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->mul_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants
    uint32_t size = (uint32_t)count;
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->mul_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), &size);

    // Dispatch
    uint32_t workgroups = (count + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_silu(vulkan_context_t* ctx,
                 vulkan_buffer_t* input, vulkan_buffer_t* output,
                 size_t count) {
    PROF_OP_BEGIN(ctx, "silu");
    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Create pipeline on first use (cached)
    // ═══════════════════════════════════════════════════════════════════════

    if (!ctx->silu_pipeline) {
        // Load shader
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("silu.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load silu shader\n");
            return;
        }

        // Create shader module
        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor set layout (2 storage buffers: input, output)
        VkDescriptorSetLayoutBinding bindings[2] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->silu_desc_layout = desc_layout;

        // Pipeline layout with push constants (size)
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t),
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->silu_layout = layout;

        // Compute pipeline
        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->silu_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Allocate descriptor set and bind buffers
    // ═══════════════════════════════════════════════════════════════════════

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->silu_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    // Bind buffers to descriptor set
    VkDescriptorBufferInfo buffer_infos[2] = {
        {.buffer = (VkBuffer)input->buffer, .offset = 0, .range = input->size},
        {.buffer = (VkBuffer)output->buffer, .offset = 0, .range = output->size},
    };

    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
    };

    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Record (optionally batched) dispatch
    // ═══════════════════════════════════════════════════════════════════════

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->silu_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->silu_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants
    uint32_t size = (uint32_t)count;
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->silu_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), &size);

    // Dispatch
    uint32_t workgroups = (count + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

// ═══════════════════════════════════════════════════════════════════════════
// IN-PLACE SCALE - data[i] *= alpha
// ═══════════════════════════════════════════════════════════════════════════

void vulkan_scale(vulkan_context_t* ctx,
                  vulkan_buffer_t* A,
                  size_t count, float alpha) {
    PROF_OP_BEGIN(ctx, "scale");
    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->scale_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("scale.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load scale shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        VkDescriptorSetLayoutBinding bindings[1] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->scale_desc_layout = desc_layout;

        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(float) + sizeof(uint32_t),
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->scale_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->scale_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->scale_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_info = {
        .buffer = (VkBuffer)A->buffer, .offset = 0, .range = A->size,
    };

    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
        .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info,
    };

    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->scale_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->scale_layout,
                             0, 1, &desc_set, 0, NULL);

    struct { float alpha; uint32_t count; } push = { alpha, (uint32_t)count };
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->scale_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t workgroups = ((uint32_t)count + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    }
    PROF_GPU_END(ctx, cmd);
}

// ═══════════════════════════════════════════════════════════════════════════
// BATCH ATTENTION - Fused multi-head attention with causal mask
// ═══════════════════════════════════════════════════════════════════════════

void vulkan_batch_attention(vulkan_context_t* ctx,
                            vulkan_buffer_t* Q, vulkan_buffer_t* K,
                            vulkan_buffer_t* V, vulkan_buffer_t* out,
                            int seq_len, int num_heads, int kv_heads, int head_dim) {
    if (!ctx || !Q || !K || !V || !out) return;
    char _pname[128]; snprintf(_pname, sizeof(_pname), "attn_fwd [seq=%d h=%d hd=%d]", seq_len, num_heads, head_dim);
    PROF_OP_BEGIN(ctx, _pname);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create streaming attention pipeline on first use (cached).
    // This kernel does not assume a fixed maximum sequence length.
    if (!ctx->batch_attn_stream_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("batch_attention_stream.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load batch_attention_stream shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor set layout (4 storage buffers: Q, K, V, out)
        VkDescriptorSetLayoutBinding bindings[4] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->batch_attn_stream_desc_layout = desc_layout;

        // Push constants: seq_len, num_heads, kv_heads, head_dim, scale
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 4 + sizeof(float),
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->batch_attn_stream_layout = layout;

        // Specialization constants: id=0 = local_size_x (subgroup), id=1 = local_size_y
        VkPhysicalDeviceProperties dev_props;
        vkGetPhysicalDeviceProperties((VkPhysicalDevice)ctx->physical_device, &dev_props);
        uint32_t max_inv = dev_props.limits.maxComputeWorkGroupInvocations;
        uint32_t max_y = dev_props.limits.maxComputeWorkGroupSize[1];
        uint32_t local_x = ctx->wg_size_attn;
        uint32_t local_y = 8;
        if (local_x * local_y > max_inv) local_y = max_inv / local_x;
        if (local_y < 1u) local_y = 1u;
        if (local_y > max_y) local_y = max_y;
        if (local_y > 32u) local_y = 32u;

        spec_2d_t spec = make_spec_2d(local_x, local_y);

        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->batch_attn_stream_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // Allocate descriptor set and bind buffers
    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->batch_attn_stream_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_infos[4] = {
        {.buffer = (VkBuffer)Q->buffer, .offset = 0, .range = Q->size},
        {.buffer = (VkBuffer)K->buffer, .offset = 0, .range = K->size},
        {.buffer = (VkBuffer)V->buffer, .offset = 0, .range = V->size},
        {.buffer = (VkBuffer)out->buffer, .offset = 0, .range = out->size},
    };

    VkWriteDescriptorSet writes[4] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[2]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[3]},
    };

    vkUpdateDescriptorSets(device, 4, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->batch_attn_stream_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->batch_attn_stream_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants: seq_len, num_heads, kv_heads, head_dim, scale
    struct {
        uint32_t seq_len;
        uint32_t num_heads;
        uint32_t kv_heads;
        uint32_t head_dim;
        float scale;
    } push_data = {
        (uint32_t)seq_len,
        (uint32_t)num_heads,
        (uint32_t)kv_heads,
        (uint32_t)head_dim,
        1.0f / sqrtf((float)head_dim)
    };

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->batch_attn_stream_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    // Dispatch: (num_heads, seq_len, 1) workgroups
    PROF_OP_LAUNCH(num_heads, seq_len, 1, ctx->wg_size_attn, 1, 1);
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, num_heads, seq_len, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

// ═══════════════════════════════════════════════════════════════════════════
// ROTARY POSITION EMBEDDING (RoPE)
// ═══════════════════════════════════════════════════════════════════════════

void vulkan_rope(vulkan_context_t* ctx,
                 vulkan_buffer_t* x_in, vulkan_buffer_t* x_out,
                 int seq_len, int num_heads, int head_dim, float rope_theta) {
    if (!ctx || !x_in || !x_out) return;
    PROF_OP_BEGIN(ctx, "rope_fwd");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create pipeline on first use (cached)
    if (!ctx->rope_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("rope.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load rope shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // Descriptor set layout (2 storage buffers: in, out)
        VkDescriptorSetLayoutBinding bindings[2] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->rope_desc_layout = desc_layout;

        // Push constants: seq_len, num_heads, head_dim, rope_theta
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 3 + sizeof(float),
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->rope_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->rope_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // Allocate descriptor set and bind buffers
    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->rope_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_infos[2] = {
        {.buffer = (VkBuffer)x_in->buffer, .offset = 0, .range = x_in->size},
        {.buffer = (VkBuffer)x_out->buffer, .offset = 0, .range = x_out->size},
    };

    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
    };

    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rope_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rope_layout,
                             0, 1, &desc_set, 0, NULL);

    // Push constants: seq_len, num_heads, head_dim, rope_theta
    struct {
        uint32_t seq_len;
        uint32_t num_heads;
        uint32_t head_dim;
        float rope_theta;
    } push_data_rope = {
        (uint32_t)seq_len,
        (uint32_t)num_heads,
        (uint32_t)head_dim,
        rope_theta
    };

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->rope_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data_rope), &push_data_rope);

    uint32_t total_pairs = seq_len * num_heads * (head_dim / 2);
    uint32_t workgroups_rope = (total_pairs + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups_rope, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}

// ═══════════════════════════════════════════════════════════════════════════
// RMSNORM FORWARD
// ═══════════════════════════════════════════════════════════════════════════

void vulkan_rmsnorm(vulkan_context_t* ctx,
                    vulkan_buffer_t* input,
                    vulkan_buffer_t* scale,
                    vulkan_buffer_t* output,
                    int rows, int cols, float eps) {
    if (!ctx || !input || !scale || !output) return;
    char _pname_rms[80];
    snprintf(_pname_rms, sizeof(_pname_rms), "rmsnorm [%dx%d] %u-thr", rows, cols, ctx->wg_size_1d);
    PROF_OP_BEGIN(ctx, _pname_rms);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Lazy pipeline init
    if (!ctx->rmsnorm_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("rmsnorm.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load rmsnorm shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // 3 buffers: input, scale, output
        VkDescriptorSetLayoutBinding bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->rmsnorm_desc_layout = desc_layout;

        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 2 + sizeof(float),  // rows, cols, eps
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->rmsnorm_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->rmsnorm_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->rmsnorm_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_infos[3] = {
        {.buffer = (VkBuffer)input->buffer, .offset = 0, .range = input->size},
        {.buffer = (VkBuffer)scale->buffer, .offset = 0, .range = scale->size},
        {.buffer = (VkBuffer)output->buffer, .offset = 0, .range = output->size},
    };

    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[2]},
    };

    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rmsnorm_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->rmsnorm_layout,
                             0, 1, &desc_set, 0, NULL);

    struct {
        uint32_t rows;
        uint32_t cols;
        float eps;
    } push_data = {(uint32_t)rows, (uint32_t)cols, eps};

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->rmsnorm_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    // One workgroup per row
    PROF_OP_LAUNCH(rows, 1, 1, ctx->wg_size_1d, 1, 1);
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, (uint32_t)rows, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    } else {
        // freed by pool reset
    }
    PROF_GPU_END(ctx, cmd);
}


void vulkan_qk_norm(vulkan_context_t* ctx,
                    vulkan_buffer_t* qk,
                    int seq_len, int num_heads, int head_dim,
                    int use_qk_norm, float eps) {
    // Early exit if disabled (LLM mode)
    if (!use_qk_norm || !ctx || !qk) return;
    PROF_OP_BEGIN(ctx, "qk_norm");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Lazy pipeline init
    if (!ctx->qk_norm_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("qk_norm.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load qk_norm shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        // 1 buffer: qk (in-place)
        VkDescriptorSetLayoutBinding bindings[1] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->qk_norm_desc_layout = desc_layout;

        // Push constants: seq_len, num_heads, head_dim, use_qk_norm, eps
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 4 + sizeof(float),
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->qk_norm_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->qk_norm_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->qk_norm_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_info = {
        .buffer = (VkBuffer)qk->buffer, .offset = 0, .range = qk->size
    };

    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = desc_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info
    };

    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->qk_norm_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->qk_norm_layout,
                             0, 1, &desc_set, 0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t num_heads;
        uint32_t head_dim;
        uint32_t use_qk_norm;
        float eps;
    } push_data = {
        (uint32_t)seq_len,
        (uint32_t)num_heads,
        (uint32_t)head_dim,
        (uint32_t)use_qk_norm,
        eps
    };

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->qk_norm_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    // One workgroup per sequence position
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, (uint32_t)seq_len, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_patch_extract(vulkan_context_t* ctx,
                          vulkan_buffer_t* frame,
                          vulkan_buffer_t* patches,
                          int frame_h, int frame_w, int channels,
                          int patch_t, int patch_h, int patch_w) {
    if (!ctx || !frame || !patches) return;
    PROF_OP_BEGIN(ctx, "patch_extract");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->patch_extract_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("patch_extract.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load patch_extract shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };

        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        VkDescriptorSetLayoutBinding bindings[2] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->patch_extract_desc_layout = desc_layout;

        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 8,
        };

        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };

        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->patch_extract_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };

        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->patch_extract_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->patch_extract_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_infos[2] = {
        {.buffer = (VkBuffer)frame->buffer, .offset = 0, .range = frame->size},
        {.buffer = (VkBuffer)patches->buffer, .offset = 0, .range = patches->size},
    };

    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[1]},
    };

    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->patch_extract_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->patch_extract_layout,
                             0, 1, &desc_set, 0, NULL);

    uint32_t num_patches_h = (uint32_t)(frame_h / patch_h);
    uint32_t num_patches_w = (uint32_t)(frame_w / patch_w);
    uint32_t total_patches = num_patches_h * num_patches_w;

    struct {
        uint32_t frame_h;
        uint32_t frame_w;
        uint32_t channels;
        uint32_t patch_t;
        uint32_t patch_h;
        uint32_t patch_w;
        uint32_t num_patches_h;
        uint32_t num_patches_w;
    } push_data = {
        (uint32_t)frame_h,
        (uint32_t)frame_w,
        (uint32_t)channels,
        (uint32_t)patch_t,
        (uint32_t)patch_h,
        (uint32_t)patch_w,
        num_patches_h,
        num_patches_w
    };

    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->patch_extract_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    uint32_t num_workgroups = (total_patches + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, num_workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_add_bias(vulkan_context_t* ctx,
                     vulkan_buffer_t* data,
                     vulkan_buffer_t* bias,
                     int rows, int cols) {
    if (!ctx || !data || !bias || rows <= 0 || cols <= 0) return;
    PROF_OP_BEGIN(ctx, "add_bias");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create pipeline on first use
    if (!ctx->add_bias_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("add_bias.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load add_bias shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };
        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        VkDescriptorSetLayoutBinding bindings[] = {
            { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->add_bias_desc_layout = desc_layout;

        VkPushConstantRange push_range = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 8,  // rows, cols
        };
        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->add_bias_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };
        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->add_bias_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->add_bias_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buf_info[] = {
        { .buffer = (VkBuffer)data->buffer, .offset = 0, .range = data->size },
        { .buffer = (VkBuffer)bias->buffer, .offset = 0, .range = bias->size },
    };
    VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[1] },
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->add_bias_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            (VkPipelineLayout)ctx->add_bias_layout,
                            0, 1, &desc_set, 0, NULL);

    uint32_t push_data[2] = { (uint32_t)rows, (uint32_t)cols };
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->add_bias_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    uint32_t total = (uint32_t)rows * (uint32_t)cols;
    uint32_t num_workgroups = (total + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, num_workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_bce_loss_grad(vulkan_context_t* ctx,
                          vulkan_buffer_t* logits,
                          vulkan_buffer_t* targets,
                          vulkan_buffer_t* grad_logits,
                          vulkan_buffer_t* loss_out,
                          size_t count) {
    if (!ctx || !logits || !targets || !grad_logits || !loss_out || count == 0) return;
    PROF_OP_BEGIN(ctx, "bce_loss_grad");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create pipeline on first use
    if (!ctx->bce_loss_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("bce_loss_grad.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load bce_loss_grad shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };
        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        VkDescriptorSetLayoutBinding bindings[] = {
            { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->bce_loss_desc_layout = desc_layout;

        VkPushConstantRange push_range = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 4,  // count
        };
        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->bce_loss_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };
        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->bce_loss_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->bce_loss_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buf_info[] = {
        { .buffer = (VkBuffer)logits->buffer, .offset = 0, .range = logits->size },
        { .buffer = (VkBuffer)targets->buffer, .offset = 0, .range = targets->size },
        { .buffer = (VkBuffer)grad_logits->buffer, .offset = 0, .range = grad_logits->size },
        { .buffer = (VkBuffer)loss_out->buffer, .offset = 0, .range = loss_out->size },
    };
    VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[2] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[3] },
    };
    vkUpdateDescriptorSets(device, 4, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->bce_loss_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            (VkPipelineLayout)ctx->bce_loss_layout,
                            0, 1, &desc_set, 0, NULL);

    uint32_t push_data = (uint32_t)count;
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->bce_loss_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), &push_data);

    uint32_t num_workgroups = (count + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, num_workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    }
    PROF_GPU_END(ctx, cmd);
}

void vulkan_sum_cols(vulkan_context_t* ctx,
                     vulkan_buffer_t* data,
                     vulkan_buffer_t* out,
                     int rows, int cols) {
    if (!ctx || !data || !out || rows <= 0 || cols <= 0) return;
    PROF_OP_BEGIN(ctx, "sum_cols");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Create pipeline on first use
    if (!ctx->sum_cols_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("sum_cols.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load sum_cols shader\n");
            return;
        }

        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = spirv,
        };
        VkShaderModule shader_module;
        vkCreateShaderModule(device, &module_info, NULL, &shader_module);
        free(spirv);

        VkDescriptorSetLayoutBinding bindings[] = {
            { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
            { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->sum_cols_desc_layout = desc_layout;

        VkPushConstantRange push_range = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 8,  // rows, cols
        };
        VkPipelineLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        VkPipelineLayout layout;
        vkCreatePipelineLayout(device, &layout_info, NULL, &layout);
        ctx->sum_cols_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_1d);
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pSpecializationInfo = &spec.info,
            },
            .layout = layout,
        };
        VkPipeline pipeline;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);
        ctx->sum_cols_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->sum_cols_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buf_info[] = {
        { .buffer = (VkBuffer)data->buffer, .offset = 0, .range = data->size },
        { .buffer = (VkBuffer)out->buffer, .offset = 0, .range = out->size },
    };
    VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_info[1] },
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->sum_cols_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            (VkPipelineLayout)ctx->sum_cols_layout,
                            0, 1, &desc_set, 0, NULL);

    uint32_t push_data[2] = { (uint32_t)rows, (uint32_t)cols };
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->sum_cols_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

    uint32_t num_workgroups = (cols + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, num_workgroups, 1, 1);

    cmd_compute_barrier(cmd);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
    if (ctx->recording) {
        defer_desc_set(ctx, desc_set);
    }
    PROF_GPU_END(ctx, cmd);
}
void vulkan_copy_buffer(vulkan_context_t* ctx,
                        vulkan_buffer_t* src,
                        vulkan_buffer_t* dst,
                        size_t bytes) {
    if (!ctx || !src || !dst || bytes == 0) return;

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    VkBufferCopy region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bytes,
    };

    // Ensure prior compute writes are visible to transfer reads
    VkMemoryBarrier to_transfer = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         1, &to_transfer,
                         0, NULL,
                         0, NULL);

    vkCmdCopyBuffer(cmd, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &region);

    // Make transfer writes visible to subsequent compute reads
    VkMemoryBarrier to_compute = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1, &to_compute,
                         0, NULL,
                         0, NULL);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
}

void vulkan_copy_buffer_cache(vulkan_context_t* ctx,
                              vulkan_buffer_t* src,
                              vulkan_buffer_t* dst,
                              size_t bytes) {
    // Cache copy: pre-barrier only (no post-barrier)
    // Destination is only read in backward pass, so transfer→compute barrier is deferred
    if (!ctx || !src || !dst || bytes == 0) return;

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    VkBufferCopy region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bytes,
    };

    // Ensure prior compute writes are visible to transfer reads
    VkMemoryBarrier to_transfer = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         1, &to_transfer,
                         0, NULL,
                         0, NULL);

    vkCmdCopyBuffer(cmd, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &region);

    // NO transfer→compute barrier here - deferred to vulkan_barrier_transfer_to_compute()

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
}

void vulkan_barrier_transfer_to_compute(vulkan_context_t* ctx) {
    // Make ALL pending transfer writes visible to compute shaders.
    // Call once before backward pass begins.
    if (!ctx) return;

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op(ctx, device, pool, &owns_cmd);

    VkMemoryBarrier to_compute = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         1, &to_compute,
                         0, NULL,
                         0, NULL);

    submit_and_cleanup_if_owned(device, pool, queue, cmd, owns_cmd);
}
