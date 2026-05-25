# Seraph Native Training and Inference Engine
# Makefile for native transformer Training and Inference
# Supports x86_64 (AVX-512) and ARM64 (NEON)

CC = gcc

# Detect architecture and set appropriate flags
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    # x86_64: Use AVX-512 if available, fallback to AVX2
    AVX512_SUPPORT := $(shell grep -q avx512f /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
    ifeq ($(AVX512_SUPPORT),1)
        ARCH_FLAGS = -march=native -mfma -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mavx512vnni
    else
        ARCH_FLAGS = -march=native -mfma -mavx2
    endif
else ifeq ($(ARCH),aarch64)
    # ARM64 (Termux, Pixel, Apple Silicon, etc)
    # Check for SVE2 support (Pixel 6+, modern ARM servers)
    SVE2_SUPPORT := $(shell grep -q sve2 /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
    BF16_SUPPORT := $(shell grep -q bf16 /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
    ifeq ($(SVE2_SUPPORT),1)
        ifeq ($(BF16_SUPPORT),1)
            # Full Pixel 6+ / ARMv9: SVE2 + BF16
            ARCH_FLAGS = -march=armv9-a+sve2+bf16
        else
            # SVE2 without BF16
            ARCH_FLAGS = -march=armv9-a+sve2
        endif
    else
        # Fallback to NEON (Apple Silicon, older ARM)
        ARCH_FLAGS = -march=armv8-a+simd
    endif
else
    # Generic fallback
    ARCH_FLAGS = -march=native
endif

# Check for OpenMP support
OPENMP_SUPPORT := $(shell echo 'int main(){}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
ifeq ($(OPENMP_SUPPORT),1)
    OMP_FLAGS = -fopenmp
    OMP_LINK = -lgomp
else
    OMP_FLAGS =
    OMP_LINK =
endif

CFLAGS = -O3 $(ARCH_FLAGS) $(OMP_FLAGS) -Wall -Wextra -ffast-math -funroll-loops
INCLUDES = -I./include
LIBS = -lm $(OMP_FLAGS)

# Vulkan configuration
VULKAN_CFLAGS = -O3 $(ARCH_FLAGS) -Wall -Wextra
VULKAN_LIBS = -lvulkan -lm
GLSLC := $(shell which glslc 2>/dev/null || echo "glslc")

# Directories
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build
INCLUDE_DIR = include

# Source files
SRCS = $(SRC_DIR)/chat.c \
       $(SRC_DIR)/transformer.c \
       $(SRC_DIR)/model_config.c \
       $(SRC_DIR)/safetensors_cjson.c \
       $(SRC_DIR)/tokenizer.c \
       $(SRC_DIR)/lora.c

# Object files
OBJS = $(BUILD_DIR)/chat.o \
       $(BUILD_DIR)/transformer.o \
       $(BUILD_DIR)/model_config.o \
       $(BUILD_DIR)/safetensors_cjson.o \
       $(BUILD_DIR)/tokenizer.o \
       $(BUILD_DIR)/lora.o \
       $(BUILD_DIR)/cJSON.o

# Main binaries (project root)
SERAPH_CHAT = seraph-chat
SERAPH_TRAIN = seraph-train
SERAPH_MONITOR = seraph-monitor
SERAPH_TOPOLOGY = seraph-topology-monitor
SERAPH_FIELD_MONITOR = seraph-field-monitor
SERAPH_BAND_MONITOR = seraph-band-monitor

# Tool binaries (tools/)
SERAPH_TOKENIZE = tools/seraph-tokenize
SERAPH_CONVERTER = tools/seraph-dataset-converter
SERAPH_MERGE = tools/seraph-merge
SERAPH_INIT = tools/seraph-init
SERAPH_GPU_VIZ = seraph-gpu-viz
SERAPH_VK_QUERY = tests/vulkan-profile-query

# Default target - main binaries only
main: $(BUILD_DIR) $(SERAPH_CHAT) $(SERAPH_TRAIN) $(SERAPH_MONITOR) $(SERAPH_TOPOLOGY) $(SERAPH_FIELD_MONITOR) $(SERAPH_BAND_MONITOR) $(SERAPH_GPU_VIZ)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Main chat binary
$(SERAPH_CHAT): $(OBJS) $(TEST_DIR)/test_chat.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(TEST_DIR)/test_chat.c $(OBJS) $(LIBS)
	@echo "✅ Built $(SERAPH_CHAT)"
	@echo ""


# Object files
$(BUILD_DIR)/chat.o: $(SRC_DIR)/chat.c $(INCLUDE_DIR)/chat.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/transformer.o: $(SRC_DIR)/transformer.c $(INCLUDE_DIR)/transformer.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/model_config.o: $(SRC_DIR)/model_config.c $(INCLUDE_DIR)/model_config.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/safetensors_cjson.o: $(SRC_DIR)/safetensors_cjson.c $(INCLUDE_DIR)/safetensors.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/tokenizer.o: $(SRC_DIR)/tokenizer.c $(INCLUDE_DIR)/tokenizer.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/lora.o: $(SRC_DIR)/lora.c $(INCLUDE_DIR)/lora.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/cJSON.o: $(SRC_DIR)/cJSON.c $(INCLUDE_DIR)/cJSON.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<


# Dataset tokenizer
$(SERAPH_TOKENIZE): $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/model_config.o $(BUILD_DIR)/cJSON.o tools/tokenize_dataset.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/tokenize_dataset.c $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/model_config.o $(BUILD_DIR)/cJSON.o $(LIBS)
	@echo "✅ Built $(SERAPH_TOKENIZE)"
	@echo ""

# Dataset converter (CSV/JSON → chat JSONL)
$(SERAPH_CONVERTER): $(BUILD_DIR)/cJSON.o tools/dataset_converter.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/dataset_converter.c $(BUILD_DIR)/cJSON.o $(LIBS)
	@echo "✅ Built $(SERAPH_CONVERTER) (Native C Edition)"
	@echo ""

# Training binary (with Vulkan GPU acceleration)
$(SERAPH_TRAIN): $(OBJS) $(BUILD_DIR)/vulkan_backend.o $(BUILD_DIR)/vulkan_backend_backward.o $(SRC_DIR)/train.c tools/seraph-train.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/seraph-train.c $(SRC_DIR)/train.c $(OBJS) $(BUILD_DIR)/vulkan_backend.o $(BUILD_DIR)/vulkan_backend_backward.o $(LIBS) $(VULKAN_LIBS)
	@echo "✅ Built $(SERAPH_TRAIN) with GPU acceleration"
	@echo ""

# NaN dump inspector
SERAPH_NAN_INSPECT = tools/seraph-nan-inspect
$(SERAPH_NAN_INSPECT): $(BUILD_DIR)/model_config.o $(BUILD_DIR)/cJSON.o tools/nan_dump_inspect.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/nan_dump_inspect.c $(BUILD_DIR)/model_config.o $(BUILD_DIR)/cJSON.o -lm
	@echo "✅ Built $(SERAPH_NAN_INSPECT)"
	@echo ""

# LoRA merge binary
$(SERAPH_MERGE): $(BUILD_DIR)/model_config.o $(BUILD_DIR)/safetensors_cjson.o $(BUILD_DIR)/lora.o $(BUILD_DIR)/cJSON.o tools/merge_lora.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/merge_lora.c $(BUILD_DIR)/model_config.o $(BUILD_DIR)/safetensors_cjson.o $(BUILD_DIR)/lora.o $(BUILD_DIR)/cJSON.o $(LIBS)
	@echo "✅ Built $(SERAPH_MERGE)"
	@echo ""


# Random weight initializer
$(SERAPH_INIT): $(BUILD_DIR)/cJSON.o tools/init_random_weights.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/init_random_weights.c $(BUILD_DIR)/cJSON.o -lm
	@echo "✅ Built $(SERAPH_INIT) (Native Genesis)"
	@echo ""


# LoRA Weight monitor
$(SERAPH_MONITOR): $(BUILD_DIR)/lora.o $(BUILD_DIR)/safetensors_cjson.o $(BUILD_DIR)/cJSON.o tools/weight_monitor.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/weight_monitor.c $(BUILD_DIR)/lora.o $(BUILD_DIR)/safetensors_cjson.o $(BUILD_DIR)/cJSON.o -lm
	@echo "✅ Built $(SERAPH_MONITOR) (Neural Oscilloscope)"
	@echo ""

# Topology monitor (EM field training visualizer)
$(SERAPH_TOPOLOGY): $(BUILD_DIR)/cJSON.o tools/topology_monitor.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/topology_monitor.c $(BUILD_DIR)/cJSON.o -lm
	@echo "✅ Built $(SERAPH_TOPOLOGY) (Electromagnetic Topology Visualizer)"
	@echo ""

# Gradient field monitor (SDL2 + OpenGL visualization)
$(SERAPH_FIELD_MONITOR): tools/gradient_field_monitor.c
	$(CC) -O3 -march=native -Wall -Wextra $(INCLUDES) -o $@ tools/gradient_field_monitor.c -lSDL2 -lSDL2_ttf -lGL -lGLU -lm
	@echo "✅ Built $(SERAPH_FIELD_MONITOR) (Electromagnetic Field Visualizer)"
	@echo ""

# Band monitor (standalone band dynamics visualizer)
$(SERAPH_BAND_MONITOR): $(BUILD_DIR)/cJSON.o tools/band_monitor.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tools/band_monitor.c $(BUILD_DIR)/cJSON.o -lm
	@echo "✅ Built $(SERAPH_BAND_MONITOR) (Band Dynamics Visualizer)"
	@echo ""

# Build tool binaries
tools: $(BUILD_DIR) $(OBJS) $(SERAPH_TOKENIZE) $(SERAPH_CONVERTER) $(SERAPH_NAN_INSPECT) $(SERAPH_MERGE) $(SERAPH_BENCH) $(SERAPH_INIT) $(SERAPH_GPU_VIZ) $(SERAPH_VIDEO_INFERENCE) $(SERAPH_CLIP_ENCODE) $(SERAPH_VDAT_CONVERT)

# Build everything (main + tools)
all: $(BUILD_DIR) $(SERAPH_CHAT) $(SERAPH_TRAIN) $(SERAPH_MONITOR) $(SERAPH_TOPOLOGY) $(SERAPH_FIELD_MONITOR) $(SERAPH_BAND_MONITOR) $(SERAPH_TOKENIZE) $(SERAPH_CONVERTER) $(SERAPH_NAN_INSPECT) $(SERAPH_MERGE) $(SERAPH_INIT) $(SERAPH_GPU_VIZ)

# Clean build artifacts
clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(SERAPH_CHAT) $(SERAPH_TRAIN) $(SERAPH_TRAIN_CUDA) $(SERAPH_MONITOR) $(SERAPH_TOPOLOGY) $(SERAPH_FIELD_MONITOR) $(SERAPH_BAND_MONITOR)
	rm -f $(SERAPH_TOKENIZE) $(SERAPH_CONVERTER) $(SERAPH_NAN_INSPECT) $(SERAPH_MERGE) $(SERAPH_INIT)
	rm -f $(ALL_TESTS)
	rm -f $(SERAPH_GPU_QUERY) $(SERAPH_GPU_VIZ) $(SERAPH_VK_QUERY)
	@echo "🧹 Cleaned build artifacts"

# Full rebuild
rebuild: clean all

# Install to /usr/local/bin
install: $(SERAPH_CHAT)
	sudo cp $(SERAPH_CHAT) $(SERAPH_TRAIN) /usr/local/bin/
	@echo "📦 Installed to /usr/local/bin/"

# ═══════════════════════════════════════════════════════════════════════════
# VULKAN BACKEND
# ═══════════════════════════════════════════════════════════════════════════

# Vulkan backend object
$(BUILD_DIR)/vulkan_backend.o: $(SRC_DIR)/vulkan_backend.c $(INCLUDE_DIR)/vulkan_backend.h
	$(CC) $(VULKAN_CFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/vulkan_backend_backward.o: $(SRC_DIR)/vulkan_backend_backward.c $(INCLUDE_DIR)/vulkan_backend.h
	$(CC) $(VULKAN_CFLAGS) $(INCLUDES) -c -o $@ $<

# Compile shaders to SPIR-V
shaders: shaders/matmul.spv shaders/matmul_transpose.spv shaders/adamw.spv shaders/softmax.spv shaders/add.spv shaders/mul.spv shaders/scale.spv shaders/silu.spv shaders/rmsnorm.spv shaders/batch_attention.spv shaders/batch_attention_stream.spv shaders/rope.spv shaders/silu_backward.spv shaders/elementwise_mul_backward.spv shaders/rmsnorm_backward.spv shaders/embed_lookup.spv shaders/cross_entropy_grad.spv shaders/reduce_sum.spv shaders/reduce_maxabs.spv shaders/rmsnorm_backward_batch.spv shaders/rope_backward.spv shaders/batch_attention_backward.spv shaders/batch_attention_backward_stream.spv shaders/embedding_backward.spv shaders/nan_check.spv shaders/bce_loss_grad.spv shaders/patch_extract.spv shaders/qk_norm.spv shaders/add_bias.spv shaders/sum_cols.spv

shaders/matmul.spv: shaders/matmul.comp $(SERAPH_GPU_QUERY)
	$(eval VK_DEFINES := $(shell ./$(SERAPH_GPU_QUERY) --defines $(or $(MODEL_DIR),models) 2>/dev/null))
	$(GLSLC) -O $(VK_DEFINES) -o $@ $<
	@echo "✅ Compiled matmul shader ($(VK_DEFINES))"

shaders/matmul_transpose.spv: shaders/matmul_transpose.comp $(SERAPH_GPU_QUERY)
	$(eval VK_DEFINES := $(shell ./$(SERAPH_GPU_QUERY) --defines $(or $(MODEL_DIR),models) 2>/dev/null))
	$(GLSLC) -O $(VK_DEFINES) -o $@ $<
	@echo "✅ Compiled matmul_transpose shader ($(VK_DEFINES))"

shaders/adamw.spv: shaders/adamw.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled adamw shader"

shaders/softmax.spv: shaders/softmax.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled softmax shader"

shaders/add.spv: shaders/add.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled add shader"

shaders/mul.spv: shaders/mul.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled mul shader"

shaders/scale.spv: shaders/scale.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled scale shader"

shaders/silu.spv: shaders/silu.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled silu shader"

shaders/rmsnorm.spv: shaders/rmsnorm.comp
	$(GLSLC) --target-env=vulkan1.1 -o $@ $<
	@echo "✅ Compiled rmsnorm shader"

shaders/batch_attention.spv: shaders/batch_attention.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled batch_attention shader"

shaders/batch_attention_stream.spv: shaders/batch_attention_stream.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled batch_attention_stream shader"

shaders/rope.spv: shaders/rope.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled rope shader"

shaders/silu_backward.spv: shaders/silu_backward.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled silu_backward shader"

shaders/elementwise_mul_backward.spv: shaders/elementwise_mul_backward.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled elementwise_mul_backward shader"

shaders/rmsnorm_backward.spv: shaders/rmsnorm_backward.comp
	$(GLSLC) --target-env=vulkan1.1 -o $@ $<
	@echo "✅ Compiled rmsnorm_backward shader"

shaders/embed_lookup.spv: shaders/embed_lookup.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled embed_lookup shader"

shaders/cross_entropy_grad.spv: shaders/cross_entropy_grad.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled cross_entropy_grad shader"

shaders/reduce_sum.spv: shaders/reduce_sum.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled reduce_sum shader"

shaders/reduce_maxabs.spv: shaders/reduce_maxabs.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled reduce_maxabs shader"

shaders/rmsnorm_backward_batch.spv: shaders/rmsnorm_backward_batch.comp
	$(GLSLC) --target-env=vulkan1.1 -o $@ $<
	@echo "✅ Compiled rmsnorm_backward_batch shader"

shaders/rope_backward.spv: shaders/rope_backward.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled rope_backward shader"

shaders/batch_attention_backward.spv: shaders/batch_attention_backward.comp
	$(GLSLC) --target-env=vulkan1.1 -o $@ $<
	@echo "✅ Compiled batch_attention_backward shader"

shaders/batch_attention_backward_stream.spv: shaders/batch_attention_backward_stream.comp
	$(GLSLC) --target-env=vulkan1.1 -o $@ $<
	@echo "✅ Compiled batch_attention_backward_stream shader"

shaders/embedding_backward.spv: shaders/embedding_backward.comp
	$(GLSLC) --target-env=vulkan1.1 -o $@ $<
	@echo "✅ Compiled embedding_backward shader"

shaders/nan_check.spv: shaders/nan_check.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled nan_check shader"

shaders/bce_loss_grad.spv: shaders/bce_loss_grad.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled bce_loss_grad shader"

shaders/patch_extract.spv: shaders/patch_extract.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled patch_extract shader"

shaders/qk_norm.spv: shaders/qk_norm.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled qk_norm shader"

shaders/add_bias.spv: shaders/add_bias.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled add_bias shader"

shaders/sum_cols.spv: shaders/sum_cols.comp
	$(GLSLC) -o $@ $<
	@echo "✅ Compiled sum_cols shader"

# ═══════════════════════════════════════════════════════════════════════════
# GPU BACKEND ABSTRACTION (Runtime selection between Vulkan/CUDA)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# CUDA TRAINING
# ═══════════════════════════════════════════════════════════════════════════

NVCC = nvcc
# Auto-detect GPU architecture, fallback to sm_52 (Maxwell) for broad compatibility
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(CUDA_ARCH),)
    CUDA_ARCH = 52
endif
NVCC_FLAGS = -O3 -arch=sm_$(CUDA_ARCH) --compiler-options -fPIC -DCUDA_ARCH_NUM=$(CUDA_ARCH)
CUDA_LIBS = -lcudart

# CUDA training context (kernels + buffer management)
$(BUILD_DIR)/cuda_train_context.o: $(SRC_DIR)/cuda_train_context.cu $(INCLUDE_DIR)/cuda_train_context.h $(INCLUDE_DIR)/cuda_profiler.h $(SERAPH_GPU_QUERY)
	$(eval GPU_DEFINES := $(shell ./$(SERAPH_GPU_QUERY) --defines $(or $(MODEL_DIR),models)))
	$(NVCC) $(NVCC_FLAGS) $(GPU_DEFINES) -I$(INCLUDE_DIR) -c -o $@ $<
	@echo "✅ Compiled CUDA training context ($(GPU_DEFINES))"

# CUDA profiler (per-operation event timing)
$(BUILD_DIR)/cuda_profiler.o: $(SRC_DIR)/cuda_profiler.cu $(INCLUDE_DIR)/cuda_profiler.h
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c -o $@ $<
	@echo "✅ Compiled CUDA profiler"

# Tensor snapshot dumper (for web tensor viewer)
$(BUILD_DIR)/tensor_snapshot.o: $(SRC_DIR)/tensor_snapshot.cu $(INCLUDE_DIR)/tensor_snapshot.h
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c -o $@ $<
	@echo "✅ Compiled tensor snapshot"

# CUDA training pipeline (C code that uses CUDA context)
$(BUILD_DIR)/train_cuda.o: $(SRC_DIR)/train_cuda.c $(INCLUDE_DIR)/train_cuda.h $(INCLUDE_DIR)/cuda_train_context.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
	@echo "✅ Compiled CUDA training pipeline"

# CUDA object collection
CUDA_OBJS = $(BUILD_DIR)/cuda_train_context.o $(BUILD_DIR)/cuda_profiler.o $(BUILD_DIR)/tensor_snapshot.o

# CUDA-only training binary (includes Vulkan backend for train.c compatibility, but uses CUDA for actual training)
SERAPH_TRAIN_CUDA = seraph-train-cuda
$(SERAPH_TRAIN_CUDA): $(OBJS) $(BUILD_DIR)/vulkan_backend.o $(BUILD_DIR)/vulkan_backend_backward.o $(CUDA_OBJS) $(BUILD_DIR)/train_cuda.o tools/seraph_train_cuda.c $(SRC_DIR)/train.c
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(OMP_FLAGS)" -I$(INCLUDE_DIR) -o $@ tools/seraph_train_cuda.c $(SRC_DIR)/train.c $(SRC_DIR)/train_cuda.c $(OBJS) $(BUILD_DIR)/vulkan_backend.o $(BUILD_DIR)/vulkan_backend_backward.o $(CUDA_OBJS) -L/usr/local/cuda/lib64 $(CUDA_LIBS) $(VULKAN_LIBS) -lm $(OMP_LINK)
	@echo "✅ Built $(SERAPH_TRAIN_CUDA) with CUDA backend"
	@echo ""

# GPU profile query + optimal topology generator
SERAPH_GPU_QUERY = tests/gpu-profile-query
$(SERAPH_GPU_QUERY): $(TEST_DIR)/gpu_profile_query.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -o $@ $< -lm
	@echo "Built $(SERAPH_GPU_QUERY) (GPU Hardware Profiler + Topology Optimizer)"
	@echo ""

# GPU Visualizer (pure C HTTP server, no CUDA needed)
$(SERAPH_GPU_VIZ): tools/gpu-viz/server.c
	$(CC) -O2 -Wall -o $@ $<
	@echo "Built $(SERAPH_GPU_VIZ) (GPU Visualizer Server)"
	@echo ""

# Vulkan hardware profile query — queries device via Vulkan API, runs matmul
# tile×reg×bk sweep and adamw block search, outputs -D flags for shader compile.
$(SERAPH_VK_QUERY): $(TEST_DIR)/vulkan_profile_query.c
	$(CC) -O2 -Wall $(INCLUDES) -o $@ $< $(VULKAN_LIBS)
	@echo "Built $(SERAPH_VK_QUERY) (Vulkan Hardware Profiler)"
	@echo ""


# Build with CUDA support (requires nvcc)
cuda: $(BUILD_DIR) $(SERAPH_GPU_QUERY) $(OBJS) $(BUILD_DIR)/vulkan_backend.o $(BUILD_DIR)/vulkan_backend_backward.o $(CUDA_OBJS) $(BUILD_DIR)/train_cuda.o $(SERAPH_TRAIN_CUDA)

# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════

ALL_TESTS = $(TEST_DIR)/test-vulkan $(TEST_DIR)/test-matmul $(TEST_DIR)/test-config $(TEST_DIR)/test-universal $(TEST_DIR)/test-optimized-inference $(TEST_DIR)/test-chat

tests: $(BUILD_DIR) $(OBJS) $(BUILD_DIR)/vulkan_backend.o $(SERAPH_VK_QUERY) shaders
	$(CC) $(VULKAN_CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test-vulkan $(TEST_DIR)/test_vulkan.c $(BUILD_DIR)/vulkan_backend.o $(VULKAN_LIBS)
	@echo "✅ Built $(TEST_DIR)/test-vulkan"
	@echo ""
	$(CC) $(VULKAN_CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test-matmul $(TEST_DIR)/test_matmul.c $(BUILD_DIR)/vulkan_backend.o $(VULKAN_LIBS)
	@echo "✅ Built $(TEST_DIR)/test-matmul"
	@echo ""
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test-config $(TEST_DIR)/test_config.c $(BUILD_DIR)/model_config.o $(BUILD_DIR)/cJSON.o $(LIBS)
	@echo "✅ Built $(TEST_DIR)/test-config"
	@echo ""
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test-chat $(TEST_DIR)/test_chat.c $(OBJS) $(LIBS)
	@echo "✅ Built $(TEST_DIR)/test-chat"
	@echo ""

# Show help
help:
	@echo ""
	@echo "Seraph Native Training and Inference Engine - Build Targets"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  make              Build main binaries (chat, train, inference, monitors)"
	@echo "  make tools        Build tool binaries (tokenize, converter, merge, etc.)"
	@echo "  make all          Build everything (main + tools)"
	@echo "  make tests        Build test binaries"
	@echo "  make shaders      Compile GLSL shaders to SPIR-V"
	@echo "  make cuda         Build with CUDA backend (NVIDIA GPUs)"

	@echo "  make clean        Remove build artifacts"
	@echo "  make rebuild      Clean and rebuild"
	@echo "  make install      Install to /usr/local/bin"
	@echo "  make help         Show this help"
	@echo ""

.PHONY: main all tools clean rebuild install help tests shaders cuda
