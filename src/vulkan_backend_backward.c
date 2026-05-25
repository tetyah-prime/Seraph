// vulkan_backend_backward.c - Backward pass GPU kernels
// This file contains the gradient computation kernels

#include "../include/vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include <time.h>

// Profiler op struct + macros (must match vulkan_backend.c)
typedef struct {
    char name[80];
    float elapsed_ms;
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
} vulkan_profiler_op_t;
#define VULKAN_PROFILER_MAX_OPS 1024

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

#define PROF_OP_LAUNCH(gx, gy, gz, bx, by, bz) \
    if (_prof_idx >= 0) { \
        vulkan_profiler_op_t* _op = &((vulkan_profiler_op_t*)(ctx)->prof_ops)[_prof_idx]; \
        _op->grid_x = gx; _op->grid_y = gy; _op->grid_z = gz; \
        _op->block_x = bx; _op->block_y = by; _op->block_z = bz; \
    }

#define PROF_GPU_BEGIN(ctx, cmd) \
    if (_prof_idx >= 0 && (ctx)->prof_query_pool) { \
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, \
                           (VkQueryPool)(ctx)->prof_query_pool, (uint32_t)(_prof_idx * 2)); \
    }

#define PROF_GPU_END(ctx, cmd) \
    if (_prof_idx >= 0 && (ctx)->prof_query_pool) { \
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, \
                           (VkQueryPool)(ctx)->prof_query_pool, (uint32_t)(_prof_idx * 2 + 1)); \
        (ctx)->prof_num_ops++; \
    }

// External shader loader from vulkan_backend.c
extern uint32_t* load_shader_spirv(const char* shader_name, size_t* size);

// ═══════════════════════════════════════════════════════════════════════════
// SPECIALIZATION CONSTANT HELPERS (duplicated from vulkan_backend.c)
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


static void defer_desc_set(vulkan_context_t* ctx, VkDescriptorSet set) {
    if (!ctx || !ctx->recording || !set) return;

    if (!ctx->deferred_desc_sets || ctx->deferred_desc_cap == 0) {
        ctx->deferred_desc_cap = 256;
        ctx->deferred_desc_sets = calloc(ctx->deferred_desc_cap, sizeof(VkDescriptorSet));
        ctx->deferred_desc_count = 0;
    }
    if (ctx->deferred_desc_count >= ctx->deferred_desc_cap) {
        ctx->deferred_desc_cap *= 2;
        ctx->deferred_desc_sets = realloc(ctx->deferred_desc_sets,
                                          ctx->deferred_desc_cap * sizeof(VkDescriptorSet));
    }
    ((VkDescriptorSet*)ctx->deferred_desc_sets)[ctx->deferred_desc_count++] = set;
}

static VkCommandBuffer get_cmd_for_op_local(vulkan_context_t* ctx, VkDevice device, VkCommandPool pool, int* owns_cmd) {
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

static void cmd_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
}

static void submit_if_owned(vulkan_context_t* ctx, VkDevice device, VkCommandPool pool, VkQueue queue,
                            VkCommandBuffer cmd, int owns_cmd, VkDescriptorSet desc_set) {
    if (ctx && ctx->recording && !owns_cmd) {
        defer_desc_set(ctx, desc_set);
        cmd_compute_barrier(cmd);
        return;
    }

    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
    // desc set freed by pool reset (no individual frees needed)
}

// SiLU backward: grad_input = grad_output * silu'(input)
void vulkan_silu_backward(vulkan_context_t* ctx,
                          vulkan_buffer_t* input,
                          vulkan_buffer_t* grad_output,
                          vulkan_buffer_t* grad_input,
                          size_t count) {
    PROF_OP_BEGIN(ctx, "silu_bwd");
    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Lazy pipeline init
    if (!ctx->silu_backward_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("silu_backward.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load silu_backward shader\n");
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

        // 3 buffers: input, grad_output, grad_input
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
        ctx->silu_backward_desc_layout = desc_layout;

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
        ctx->silu_backward_layout = layout;

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
        ctx->silu_backward_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // Allocate descriptor set
    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->silu_backward_desc_layout;
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
        {.buffer = (VkBuffer)grad_output->buffer, .offset = 0, .range = grad_output->size},
        {.buffer = (VkBuffer)grad_input->buffer, .offset = 0, .range = grad_input->size},
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
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->silu_backward_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->silu_backward_layout,
                             0, 1, &desc_set, 0, NULL);

    uint32_t size = (uint32_t)count;
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->silu_backward_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), &size);

    uint32_t workgroups = (count + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

// Elementwise multiply backward
void vulkan_elementwise_mul_backward(vulkan_context_t* ctx,
                                     vulkan_buffer_t* A,
                                     vulkan_buffer_t* B,
                                     vulkan_buffer_t* grad_C,
                                     vulkan_buffer_t* grad_A,
                                     vulkan_buffer_t* grad_B,
                                     size_t count) {
    PROF_OP_BEGIN(ctx, "mul_bwd");
    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    // Lazy pipeline init
    if (!ctx->mul_backward_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("elementwise_mul_backward.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load elementwise_mul_backward shader\n");
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

        // 5 buffers: A, B, grad_C, grad_A, grad_B
        VkDescriptorSetLayoutBinding bindings[5] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 5,
            .pBindings = bindings,
        };

        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->mul_backward_desc_layout = desc_layout;

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
        ctx->mul_backward_layout = layout;

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
        ctx->mul_backward_pipeline = pipeline;

        vkDestroyShaderModule(device, shader_module, NULL);
    }

    // Allocate descriptor set
    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->mul_backward_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo buffer_infos[5] = {
        {.buffer = (VkBuffer)A->buffer, .offset = 0, .range = A->size},
        {.buffer = (VkBuffer)B->buffer, .offset = 0, .range = B->size},
        {.buffer = (VkBuffer)grad_C->buffer, .offset = 0, .range = grad_C->size},
        {.buffer = (VkBuffer)grad_A->buffer, .offset = 0, .range = grad_A->size},
        {.buffer = (VkBuffer)grad_B->buffer, .offset = 0, .range = grad_B->size},
    };

    VkWriteDescriptorSet writes[5] = {
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
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 4, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &buffer_infos[4]},
    };

    vkUpdateDescriptorSets(device, 5, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->mul_backward_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             (VkPipelineLayout)ctx->mul_backward_layout,
                             0, 1, &desc_set, 0, NULL);

    uint32_t size = (uint32_t)count;
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->mul_backward_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), &size);

    uint32_t workgroups = (count + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

void vulkan_embed_lookup(vulkan_context_t* ctx,
                         vulkan_buffer_t* tokens_u32,
                         vulkan_buffer_t* embed_weight,
                         vulkan_buffer_t* out_hidden,
                         int seq_len, int hidden_size, int vocab_size) {
    if (!ctx || !tokens_u32 || !embed_weight || !out_hidden) return;
    PROF_OP_BEGIN(ctx, "embed_lookup");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->embed_lookup_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("embed_lookup.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load embed_lookup shader\n");
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

        VkDescriptorSetLayoutBinding bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->embed_lookup_desc_layout = desc_layout;

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
        ctx->embed_lookup_layout = layout;

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
        ctx->embed_lookup_pipeline = pipeline;
        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->embed_lookup_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo infos[3] = {
        {.buffer = (VkBuffer)tokens_u32->buffer, .offset = 0, .range = tokens_u32->size},
        {.buffer = (VkBuffer)embed_weight->buffer, .offset = 0, .range = embed_weight->size},
        {.buffer = (VkBuffer)out_hidden->buffer, .offset = 0, .range = out_hidden->size},
    };
    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[2]},
    };
    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->embed_lookup_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->embed_lookup_layout, 0, 1, &desc_set, 0, NULL);

    uint32_t push[3] = {(uint32_t)seq_len, (uint32_t)hidden_size, (uint32_t)vocab_size};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->embed_lookup_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);

    uint32_t total = (uint32_t)seq_len * (uint32_t)hidden_size;
    uint32_t workgroups = (total + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

void vulkan_cross_entropy_grad(vulkan_context_t* ctx,
                               vulkan_buffer_t* logits,
                               vulkan_buffer_t* targets_u32,
                               vulkan_buffer_t* grad_logits,
                               vulkan_buffer_t* loss_rows,
                               int rows, int cols, int valid_rows) {
    if (!ctx || !logits || !targets_u32 || !grad_logits || !loss_rows) return;
    char _pname_ce[80];
    snprintf(_pname_ce, sizeof(_pname_ce), "cross_entropy_grad [%dx%d] %u-thr", rows, cols, ctx->wg_size_1d);
    PROF_OP_BEGIN(ctx, _pname_ce);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->cross_entropy_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("cross_entropy_grad.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load cross_entropy_grad shader\n");
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

        VkDescriptorSetLayoutBinding bindings[4] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->cross_entropy_desc_layout = desc_layout;

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
        ctx->cross_entropy_layout = layout;

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
        ctx->cross_entropy_pipeline = pipeline;
        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->cross_entropy_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo infos[4] = {
        {.buffer = (VkBuffer)logits->buffer, .offset = 0, .range = logits->size},
        {.buffer = (VkBuffer)targets_u32->buffer, .offset = 0, .range = targets_u32->size},
        {.buffer = (VkBuffer)grad_logits->buffer, .offset = 0, .range = grad_logits->size},
        {.buffer = (VkBuffer)loss_rows->buffer, .offset = 0, .range = loss_rows->size},
    };
    VkWriteDescriptorSet writes[4] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[2]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[3]},
    };
    vkUpdateDescriptorSets(device, 4, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->cross_entropy_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->cross_entropy_layout, 0, 1, &desc_set, 0, NULL);

    uint32_t push[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)valid_rows};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->cross_entropy_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);

    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, (uint32_t)rows, 1, 1);
    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

void vulkan_reduce_sum(vulkan_context_t* ctx,
                       vulkan_buffer_t* in_buf,
                       vulkan_buffer_t* out_buf,
                       size_t count) {
    if (!ctx || !in_buf || !out_buf || count == 0) return;
    PROF_OP_BEGIN(ctx, "reduce_sum");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->reduce_sum_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("reduce_sum.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load reduce_sum shader\n");
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
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->reduce_sum_desc_layout = desc_layout;

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
        ctx->reduce_sum_layout = layout;

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
        ctx->reduce_sum_pipeline = pipeline;
        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->reduce_sum_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo infos[2] = {
        {.buffer = (VkBuffer)in_buf->buffer, .offset = 0, .range = in_buf->size},
        {.buffer = (VkBuffer)out_buf->buffer, .offset = 0, .range = out_buf->size},
    };
    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1]},
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->reduce_sum_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->reduce_sum_layout, 0, 1, &desc_set, 0, NULL);

    uint32_t c = (uint32_t)count;
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->reduce_sum_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(c), &c);

    uint32_t workgroups = (uint32_t)((count + 2*ctx->wg_size_1d - 1) / (2*ctx->wg_size_1d));
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

void vulkan_rmsnorm_backward_batch(vulkan_context_t* ctx,
                                   vulkan_buffer_t* input,
                                   vulkan_buffer_t* scale,
                                   vulkan_buffer_t* grad_output,
                                   vulkan_buffer_t* grad_input,
                                   vulkan_buffer_t* grad_scale,
                                   int rows, int cols, float eps) {
    if (!ctx || !input || !scale || !grad_output || !grad_input || !grad_scale) return;
    char _pname_rms[80];
    snprintf(_pname_rms, sizeof(_pname_rms), "rmsnorm_bwd [%dx%d] %u-thr", rows, cols, ctx->wg_size_1d);
    PROF_OP_BEGIN(ctx, _pname_rms);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->rmsnorm_bwd_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("rmsnorm_backward_batch.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load rmsnorm_backward_batch shader\n");
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

        VkDescriptorSetLayoutBinding bindings[5] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 5,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->rmsnorm_bwd_desc_layout = desc_layout;

        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t) * 2 + sizeof(float),
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
        ctx->rmsnorm_bwd_layout = layout;

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
        ctx->rmsnorm_bwd_pipeline = pipeline;
        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->rmsnorm_bwd_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo infos[5] = {
        {.buffer = (VkBuffer)input->buffer, .offset = 0, .range = input->size},
        {.buffer = (VkBuffer)scale->buffer, .offset = 0, .range = scale->size},
        {.buffer = (VkBuffer)grad_output->buffer, .offset = 0, .range = grad_output->size},
        {.buffer = (VkBuffer)grad_input->buffer, .offset = 0, .range = grad_input->size},
        {.buffer = (VkBuffer)grad_scale->buffer, .offset = 0, .range = grad_scale->size},
    };
    VkWriteDescriptorSet writes[5] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[2]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[3]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 4, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[4]},
    };
    vkUpdateDescriptorSets(device, 5, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rmsnorm_bwd_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->rmsnorm_bwd_layout, 0, 1, &desc_set, 0, NULL);

    struct {
        uint32_t rows;
        uint32_t cols;
        float eps;
    } push = {(uint32_t)rows, (uint32_t)cols, eps};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->rmsnorm_bwd_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    PROF_OP_LAUNCH(rows, 1, 1, ctx->wg_size_1d, 1, 1);
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, (uint32_t)rows, 1, 1);
    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

void vulkan_rope_backward(vulkan_context_t* ctx,
                          vulkan_buffer_t* x_in, vulkan_buffer_t* x_out,
                          int seq_len, int num_heads, int head_dim, float rope_theta) {
    if (!ctx || !x_in || !x_out) return;
    PROF_OP_BEGIN(ctx, "rope_bwd");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->rope_backward_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("rope_backward.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load rope_backward shader\n");
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
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->rope_backward_desc_layout = desc_layout;

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
        ctx->rope_backward_layout = layout;

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
        ctx->rope_backward_pipeline = pipeline;
        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->rope_backward_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo infos[2] = {
        {.buffer = (VkBuffer)x_in->buffer, .offset = 0, .range = x_in->size},
        {.buffer = (VkBuffer)x_out->buffer, .offset = 0, .range = x_out->size},
    };
    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1]},
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->rope_backward_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->rope_backward_layout, 0, 1, &desc_set, 0, NULL);

    struct {
        uint32_t seq_len;
        uint32_t num_heads;
        uint32_t head_dim;
        float theta;
    } push = {(uint32_t)seq_len, (uint32_t)num_heads, (uint32_t)head_dim, rope_theta};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->rope_backward_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t total_pairs = (uint32_t)seq_len * (uint32_t)num_heads * ((uint32_t)head_dim / 2);
    uint32_t workgroups = (total_pairs + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);
    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

void vulkan_batch_attention_backward(vulkan_context_t* ctx,
                                     vulkan_buffer_t* Q, vulkan_buffer_t* K, vulkan_buffer_t* V,
                                     vulkan_buffer_t* grad_out,
                                     vulkan_buffer_t* grad_Q,
                                     vulkan_buffer_t* grad_K,
                                     vulkan_buffer_t* grad_V,
                                     int seq_len, int num_heads, int kv_heads, int head_dim) {
    if (!ctx || !Q || !K || !V || !grad_out || !grad_Q || !grad_K || !grad_V) return;
    char _pname[128]; snprintf(_pname, sizeof(_pname), "attn_bwd [seq=%d h=%d hd=%d]", seq_len, num_heads, head_dim);
    PROF_OP_BEGIN(ctx, _pname);

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->batch_attn_backward_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("batch_attention_backward.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load batch_attention_backward shader\n");
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

        VkDescriptorSetLayoutBinding bindings[7] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 6, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 7,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->batch_attn_backward_desc_layout = desc_layout;

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
        ctx->batch_attn_backward_layout = layout;

        spec_1d_t spec = make_spec_1d(ctx->wg_size_attn);
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
        ctx->batch_attn_backward_pipeline = pipeline;
        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->batch_attn_backward_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo infos[7] = {
        {.buffer = (VkBuffer)Q->buffer, .offset = 0, .range = Q->size},
        {.buffer = (VkBuffer)K->buffer, .offset = 0, .range = K->size},
        {.buffer = (VkBuffer)V->buffer, .offset = 0, .range = V->size},
        {.buffer = (VkBuffer)grad_out->buffer, .offset = 0, .range = grad_out->size},
        {.buffer = (VkBuffer)grad_Q->buffer, .offset = 0, .range = grad_Q->size},
        {.buffer = (VkBuffer)grad_K->buffer, .offset = 0, .range = grad_K->size},
        {.buffer = (VkBuffer)grad_V->buffer, .offset = 0, .range = grad_V->size},
    };
    VkWriteDescriptorSet writes[7];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 7; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, 7, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->batch_attn_backward_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->batch_attn_backward_layout, 0, 1, &desc_set, 0, NULL);

    float scale = 1.0f / sqrtf((float)head_dim);
    struct {
        uint32_t seq_len;
        uint32_t num_heads;
        uint32_t kv_heads;
        uint32_t head_dim;
        float scale;
    } push = {(uint32_t)seq_len, (uint32_t)num_heads, (uint32_t)kv_heads, (uint32_t)head_dim, scale};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->batch_attn_backward_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    PROF_OP_LAUNCH(num_heads, seq_len, 1, ctx->wg_size_attn, 1, 1);
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, (uint32_t)num_heads, (uint32_t)seq_len, 1);
    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}

void vulkan_embedding_backward(vulkan_context_t* ctx,
                               vulkan_buffer_t* tokens_u32,
                               vulkan_buffer_t* grad_hidden,
                               vulkan_buffer_t* grad_embed,
                               int seq_len, int hidden_size, int vocab_size) {
    if (!ctx || !tokens_u32 || !grad_hidden || !grad_embed) return;
    PROF_OP_BEGIN(ctx, "embed_bwd");

    VkDevice device = (VkDevice)ctx->device;
    VkCommandPool pool = (VkCommandPool)ctx->command_pool;
    VkQueue queue = (VkQueue)ctx->queue;

    if (!ctx->embedding_backward_pipeline) {
        size_t spirv_size;
        uint32_t* spirv = load_shader_spirv("embedding_backward.spv", &spirv_size);
        if (!spirv) {
            fprintf(stderr, "ERROR: Failed to load embedding_backward shader\n");
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

        VkDescriptorSetLayoutBinding bindings[3] = {
            {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };
        VkDescriptorSetLayoutCreateInfo desc_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };
        VkDescriptorSetLayout desc_layout;
        vkCreateDescriptorSetLayout(device, &desc_layout_info, NULL, &desc_layout);
        ctx->embedding_backward_desc_layout = desc_layout;

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
        ctx->embedding_backward_layout = layout;

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
        ctx->embedding_backward_pipeline = pipeline;
        vkDestroyShaderModule(device, shader_module, NULL);
    }

    VkDescriptorSetLayout desc_layout = (VkDescriptorSetLayout)ctx->embedding_backward_desc_layout;
    VkDescriptorPool desc_pool = (VkDescriptorPool)ctx->descriptor_pool;
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &desc_layout,
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &alloc_info, &desc_set);

    VkDescriptorBufferInfo infos[3] = {
        {.buffer = (VkBuffer)tokens_u32->buffer, .offset = 0, .range = tokens_u32->size},
        {.buffer = (VkBuffer)grad_hidden->buffer, .offset = 0, .range = grad_hidden->size},
        {.buffer = (VkBuffer)grad_embed->buffer, .offset = 0, .range = grad_embed->size},
    };
    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[2]},
    };
    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    int owns_cmd = 0;
    VkCommandBuffer cmd = get_cmd_for_op_local(ctx, device, pool, &owns_cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->embedding_backward_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->embedding_backward_layout, 0, 1, &desc_set, 0, NULL);

    uint32_t push[3] = {(uint32_t)seq_len, (uint32_t)hidden_size, (uint32_t)vocab_size};
    vkCmdPushConstants(cmd, (VkPipelineLayout)ctx->embedding_backward_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);

    uint32_t total = (uint32_t)seq_len * (uint32_t)hidden_size;
    uint32_t workgroups = (total + ctx->wg_size_1d - 1) / ctx->wg_size_1d;
    PROF_GPU_BEGIN(ctx, cmd);
    vkCmdDispatch(cmd, workgroups, 1, 1);

    submit_if_owned(ctx, device, pool, queue, cmd, owns_cmd, desc_set);
    PROF_GPU_END(ctx, cmd);
}
