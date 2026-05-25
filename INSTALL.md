# Seraph - Installation Guide

## Minimum Requirements

- **C compiler**: gcc or clang (C99 or later)
- **make**: GNU Make
- **Linux**: glibc or musl (both tested)

That's it for CPU-only inference. For GPU training, you'll also need Vulkan.

---

## Step 1: Core Build Dependencies

### Void Linux
```bash
sudo xbps-install -S base-devel
```

### Ubuntu / Debian
```bash
sudo apt install build-essential
```

### Fedora / RHEL
```bash
sudo dnf groupinstall "Development Tools"
```

### Arch Linux
```bash
sudo pacman -S base-devel
```

### Alpine (musl)
```bash
apk add build-base
```

### Termux (Android)
```bash
pkg install clang make
```

---

## Step 2: Vulkan (Required for GPU Training)

### Vulkan Loader + Headers

**Void Linux:**
```bash
sudo xbps-install -S vulkan-loader vulkan-loader-devel vulkan-headers
```

**Ubuntu / Debian:**
```bash
sudo apt install libvulkan-dev vulkan-tools
```

**Fedora:**
```bash
sudo dnf install vulkan-loader-devel vulkan-headers
```

**Arch Linux:**
```bash
sudo pacman -S vulkan-headers vulkan-icd-loader
```

**Termux (Android):**
```bash
pkg install vulkan-headers vulkan-loader-android
```

### GPU Drivers

**AMD (RADV - recommended open-source driver):**
```bash
# Void Linux
sudo xbps-install -S mesa-vulkan-radeon

# Ubuntu / Debian
sudo apt install mesa-vulkan-drivers

# Arch Linux
sudo pacman -S vulkan-radeon
```

**NVIDIA:**
```bash
# Void Linux
sudo xbps-install -S nvidia

# Ubuntu / Debian
sudo apt install nvidia-driver

# Arch Linux
sudo pacman -S nvidia-utils
```

**Intel:**
```bash
# Void Linux
sudo xbps-install -S mesa-vulkan-intel

# Ubuntu / Debian
sudo apt install mesa-vulkan-drivers

# Arch Linux
sudo pacman -S vulkan-intel
```

### Verify Vulkan is Working
```bash
vulkaninfo --summary
# Should show your GPU name and Vulkan version
```

---

## Step 3: Shader Compiler (Optional)

Pre-compiled SPIR-V shaders (`.spv`) are included. You only need the shader compiler if you want to modify the `.comp` source files.

```bash
# Void Linux
sudo xbps-install -S glslang

# Ubuntu / Debian
sudo apt install glslang-tools

# Fedora
sudo dnf install glslang

# Arch Linux
sudo pacman -S glslang
```

Rebuild shaders with:
```bash
make shaders
```

---

## Step 4: SDL2 + OpenGL (Optional - Gradient Field Visualizer Only)

The `seraph-field-monitor` tool provides real-time gradient visualization during training. It requires SDL2 and OpenGL but does not require a GUI. It can be ran in a teletype shell via kmsdrm. This is completely optional but highly valuable - training works without it.

**Note:** SDL2 can conflict with other versions on some systems. If installation fails, you can include the library once copied from source.

```bash
# Void Linux
sudo xbps-install -S SDL2-devel SDL2_ttf-devel mesa-devel glu-devel

# Ubuntu / Debian
sudo apt install libsdl2-dev libsdl2-ttf-dev libgl-dev libglu1-mesa-dev

# Fedora
sudo dnf install SDL2-devel SDL2_ttf-devel mesa-libGL-devel mesa-libGLU-devel

# Arch Linux
sudo pacman -S sdl2 sdl2_ttf mesa glu

# Termux (Android)
pkg install sdl2 sdl2-ttf glu
```

---

## Step 5: Build

```bash
# Build all binaries
make all

# Or build specific targets
make seraph-chat          # Inference only (no Vulkan needed)
make seraph-train         # GPU training (requires Vulkan)
make seraph-field-monitor # Gradient visualizer (requires SDL2+OpenGL)

# Rebuild shaders from source (requires glslc/glslang)
make shaders

# Clean build
make clean && make all
```

### Build Output

All binaries are placed in the project root. Total size: ~2MB.

```bash
# Verify build
./seraph-train --help
./seraph-init  --help
```

---

## Step 6: Get a Model

Seraph loads HuggingFace-format models (SafeTensors). You need a model directory with:

```
model-dir/
├── config.json          # Model architecture config
├── model.safetensors    # Weights (or model-00001-of-XXXXX.safetensors for sharded)
├── tokenizer.json       # Tokenizer vocabulary and config
└── vocab.json           # (optional) Word-level vocabulary
```

Download models from [HuggingFace Hub](https://huggingface.co/models):
```bash
# Example: download a small model
git lfs install
git lfs clone https://huggingface.co/Qwen/Qwen2.5-1.5B (git lfs assures the files are downloading)
```

---

## Troubleshooting

### "Cannot find shader: *.spv"
The engine searches for shaders in this order:
1. `SERAPH_SHADER_PATH` environment variable
2. `./shaders/` (current directory)
3. `../shaders/` (parent directory)
4. `~/.local/share/seraph/shaders/`
5. `/usr/share/seraph/shaders/`

Either run binaries from the project root, or set the env var:
```bash
export SERAPH_SHADER_PATH=/path/to/seraph/shaders
```

### "Failed to create Vulkan instance"
- Check `vulkaninfo --summary` works
- Ensure your GPU driver is installed (see Step 2)
- Integrated GPUs: make sure Vulkan is enabled in BIOS (some systems disable iGPU when discrete is present, also configure your memory split here)

### SDL2 version conflicts
SDL2 is only needed for `seraph-field-monitor`. If installation fails, get a copy from source and link it like cJson is.

### OpenMP not found
OpenMP is optional but recommended for CPU inference speed. The build auto-detects it. To install:
```bash
# Void Linux
sudo xbps-install -S libgomp-devel

# Ubuntu / Debian
sudo apt install libgomp1

# Fedora
sudo dnf install libgomp
```

---

## Platform Notes

### Void Linux (musl)/(glibc)
Fully tested. Primary development platform.
AMD iGPU, Intel iGPU, Nvidia discreet GPU 3050

### Ubuntu / Debian
Fully supported. Most straightforward dependency installation.

### ARM64 (Termux / Pixel / Apple Silicon)
CPU inference supported with NEON/SVE2 auto-detection. Vulkan GPU training works on devices with Vulkan compute support.

### macOS
CPU inference only (no Vulkan). Use MoltenVK for experimental GPU support.

### Windows
Not tested. Should work with WSL2 + Vulkan passthrough. Any statically compiled binary should work.
