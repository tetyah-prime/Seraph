// cuda_train_context.cu - CUDA training context implementation
// Part of TETYAH-PRIME's native training and inference engine

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include the header (extern "C" compatible)
extern "C" {
#include "../include/cuda_train_context.h"
#include "../include/cuda_profiler.h"
#include "../include/lora.h"
}

// Global profiler pointer (set by cuda_train_set_profiler)
static cuda_profiler_t* g_profiler = NULL;

extern "C" void cuda_train_set_profiler(cuda_train_context_t* ctx, cuda_profiler_t* prof) {
    (void)ctx;
    g_profiler = prof;
}

// Helper macros for profiling kernel wrappers
#define PROF_BEGIN(name) \
    int _prof_idx = cuda_profiler_begin_op(g_profiler, name, ctx->stream)
#define PROF_END() \
    cuda_profiler_end_op(g_profiler, _prof_idx, ctx->stream)
#define PROF_LAUNCH(gx,gy,gz,bx,by,bz,smem) \
    cuda_profiler_set_launch(g_profiler, _prof_idx, gx,gy,gz,bx,by,bz,smem)
#define PROF_FLOPS(f) cuda_profiler_set_flops(g_profiler, _prof_idx, f)
#define PROF_BYTES(b) cuda_profiler_set_bytes(g_profiler, _prof_idx, b)

// ═══════════════════════════════════════════════════════════════════════════
// FLASH ATTENTION — tile sizes + reduction macros + buffer management
// ═══════════════════════════════════════════════════════════════════════════

#ifndef FA_BR
#error "FA_BR not defined — run gpu-profile-query --defines"
#endif
#ifndef FA_BC
#error "FA_BC not defined — run gpu-profile-query --defines"
#endif

#ifndef GPU_WARP_SIZE
#error "GPU_WARP_SIZE not defined — run gpu-profile-query --defines"
#endif

#define WARP_REDUCE_MAX(val) do { \
    for (int _o = GPU_WARP_SIZE / 2; _o > 0; _o /= 2) \
        val = fmaxf(val, __shfl_xor_sync(~0u, val, _o)); \
} while(0)

#define WARP_REDUCE_SUM(val) do { \
    for (int _o = GPU_WARP_SIZE / 2; _o > 0; _o /= 2) \
        val += __shfl_xor_sync(~0u, val, _o); \
} while(0)

#define BLOCK_REDUCE_SUM(val, smem_reduce) do { \
    int _lane = threadIdx.x % GPU_WARP_SIZE; \
    int _wid  = threadIdx.x / GPU_WARP_SIZE; \
    WARP_REDUCE_SUM(val); \
    if (_lane == 0) smem_reduce[_wid] = val; \
    __syncthreads(); \
    if (_wid == 0) { \
        val = (_lane < (blockDim.x / GPU_WARP_SIZE)) ? smem_reduce[_lane] : 0.0f; \
        WARP_REDUCE_SUM(val); \
    } \
} while(0)

#define BLOCK_REDUCE_MAX(val, smem_reduce) do { \
    int _lane = threadIdx.x % GPU_WARP_SIZE; \
    int _wid  = threadIdx.x / GPU_WARP_SIZE; \
    WARP_REDUCE_MAX(val); \
    if (_lane == 0) smem_reduce[_wid] = val; \
    __syncthreads(); \
    if (_wid == 0) { \
        val = (_lane < (blockDim.x / GPU_WARP_SIZE)) ? smem_reduce[_lane] : -INFINITY; \
        WARP_REDUCE_MAX(val); \
    } \
} while(0)

// (FlashAttention logsumexp is stored per-layer in fwd_cache_attn_lse)

// ═══════════════════════════════════════════════════════════════════════════
// CUDA ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RET(call, ret) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return ret; \
    } \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════
// KERNELS - FORWARD PASS (ordered by transformer dataflow)
// ═══════════════════════════════════════════════════════════════════════════
//
// Dataflow: tokens → embed → [rmsnorm → attn → add → rmsnorm → ffn → add]×N → softmax
//
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// 1. EMBEDDING LOOKUP - tokens → embeddings (ENTRY POINT)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void embed_lookup_kernel(const uint32_t* tokens, const float* embed_table,
                                     float* output, int seq_len, int hidden_size) {
    int pos = blockIdx.x;
    int dim = threadIdx.x;

    if (pos >= seq_len || dim >= hidden_size) return;

    uint32_t token = tokens[pos];
    output[pos * hidden_size + dim] = embed_table[token * hidden_size + dim];
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. RMSNORM - pre-attention and pre-FFN normalization
// ─────────────────────────────────────────────────────────────────────────────
// OG single-threaded rmsnorm (kept as reference, unused)
// __global__ void rmsnorm_kernel_single(const float* input, const float* scale, float* output,
//                                 int hidden_size, float eps) {
//     int row = blockIdx.x;
//     const float* row_in = input + row * hidden_size;
//     float* row_out = output + row * hidden_size;
//     float sum_sq = 0.0f;
//     for (int i = 0; i < hidden_size; i++) sum_sq += row_in[i] * row_in[i];
//     float inv_rms = 1.0f / sqrtf(sum_sq / hidden_size + eps);
//     for (int i = 0; i < hidden_size; i++) row_out[i] = row_in[i] * inv_rms * scale[i];
// }

// Parallel RMSNorm: blockDim.x threads cooperate on one row
// Each thread handles hidden_size/blockDim.x elements, then tree-reduce sum_sq
// Launch: <<<rows, min(hidden_size, parallel_block_size)>>>
__global__ void rmsnorm_kernel(const float* input, const float* scale, float* output,
                                int hidden_size, float eps) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const float* row_in = input + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Step 1: Each thread accumulates sum_sq for its slice of hidden_size
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += num_threads) {
        float val = row_in[i];
        local_sum_sq += val * val;
    }

    // Step 2: Warp-level reduction (no shared memory, no sync needed)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum_sq += __shfl_down_sync(~0u, local_sum_sq, offset);
    }

    // Step 3: Cross-warp reduction via shared memory
    // Warp leaders write to shared, then warp 0 reduces across warps
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    if (lane_id == 0) {
        shared[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Warp 0 reduces all warp sums
    float sum_sq = 0.0f;
    if (tid < num_warps) {
        sum_sq = shared[tid];
    }
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(~0u, sum_sq, offset);
        }
    }

    // Thread 0 writes final result to shared for all threads to read
    if (tid == 0) {
        shared[0] = 1.0f / sqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    float inv_rms = shared[0];

    // Step 4: Every thread normalizes and scales its slice
    for (int i = tid; i < hidden_size; i += num_threads) {
        row_out[i] = row_in[i] * inv_rms * scale[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. MATMUL - Tiled shared-memory matmul with register blocking
// ─────────────────────────────────────────────────────────────────────────────
//
// C[M×N] = A[M×K] @ B[K×N]  (with optional transposes and accumulate)
//
// Tiling strategy:
//   - Each block computes a TILE×TILE sub-matrix of C
//   - Iterates over K in chunks of BK
//   - Each iteration: load A[TILE×BK] and B[BK×TILE] into shared memory
//   - Each thread computes a REG×REG sub-tile of the output using registers
//   - Thread layout: (TILE/REG) × (TILE/REG) threads per block
//
// Shared memory layout (with +1 padding to avoid bank conflicts):
//   sA[TILE][BK+1]  — tile of A (or A^T)
//   sB[BK][TILE+1]  — tile of B (or B^T)
//
// Register blocking:
//   Each thread accumulates REG×REG outputs in registers
//   Per BK iteration: load REG values from sA column, REG from sB row, outer product
//
// Parameters passed via kernel args (from GPU discovery, not hardcoded):
//   TILE, BK, REG are compile-time constants for this build
//   The wrapper selects launch dims from ctx->matmul_tile_dim etc.
//

#ifndef MATMUL_TILE
#error "MATMUL_TILE not defined — run gpu-profile-query --defines"
#endif
#ifndef MATMUL_BK
#error "MATMUL_BK not defined — run gpu-profile-query --defines"
#endif
#ifndef MATMUL_REG
#error "MATMUL_REG not defined — run gpu-profile-query --defines"
#endif

// ── Shared compute + write macro (same for all 3 variants) ──
// Extracted so each specialized kernel is lean: just tile loads + this macro
#define MATMUL_COMPUTE_AND_STORE() \
    __syncthreads(); \
    for (int k = 0; k < MATMUL_BK; k++) { \
        float a[MATMUL_REG]; \
        _Pragma("unroll") \
        for (int i = 0; i < MATMUL_REG; i++) \
            a[i] = SA(ty * MATMUL_REG + i, k); \
        float b[MATMUL_REG]; \
        _Pragma("unroll") \
        for (int i = 0; i < MATMUL_REG; i++) \
            b[i] = SB(k, tx * MATMUL_REG + i); \
        _Pragma("unroll") \
        for (int i = 0; i < MATMUL_REG; i++) \
            _Pragma("unroll") \
            for (int j = 0; j < MATMUL_REG; j++) \
                acc[i][j] += a[i] * b[j]; \
    } \
    __syncthreads();

#define MATMUL_PREAMBLE() \
    extern __shared__ float shared_mem[]; \
    float* sA_flat = shared_mem; \
    float* sB_flat = shared_mem + MATMUL_TILE * (MATMUL_BK + 1); \
    const int tx = threadIdx.x; \
    const int ty = threadIdx.y; \
    const int threads_x = MATMUL_TILE / MATMUL_REG; \
    const int tid = ty * threads_x + tx; \
    const int num_threads = (MATMUL_TILE / MATMUL_REG) * (MATMUL_TILE / MATMUL_REG); \
    const int block_row = blockIdx.y * MATMUL_TILE; \
    const int block_col = blockIdx.x * MATMUL_TILE; \
    float acc[MATMUL_REG][MATMUL_REG]; \
    _Pragma("unroll") \
    for (int i = 0; i < MATMUL_REG; i++) \
        _Pragma("unroll") \
        for (int j = 0; j < MATMUL_REG; j++) \
            acc[i][j] = 0.0f;

#define MATMUL_STORE(accumulate) \
    _Pragma("unroll") \
    for (int i = 0; i < MATMUL_REG; i++) { \
        int grow = block_row + ty * MATMUL_REG + i; \
        if (grow >= M) continue; \
        _Pragma("unroll") \
        for (int j = 0; j < MATMUL_REG; j++) { \
            int gcol = block_col + tx * MATMUL_REG + j; \
            if (gcol >= N) continue; \
            if (accumulate) C[grow * N + gcol] += acc[i][j]; \
            else C[grow * N + gcol] = acc[i][j]; \
        } \
    }

#define SA(r, c) sA_flat[(r) * (MATMUL_BK + 1) + (c)]
#define SB(r, c) sB_flat[(r) * (MATMUL_TILE + 1) + (c)]

// ── C = A @ B (no transpose) ──
// No per-element bounds checks on M/N (buffers allocated at max_seq_len, overflow = stale reads).
// K bounds: one check per BK tile — last partial tile zero-fills remaining shared slots.
// Interior BK tiles (bk + BK <= K) load directly with zero branching.

// Tile load helpers — separate full-tile (no checks) from partial-tile (K edge only)
#define LOAD_A_NN(bk) \
    for (int idx = tid; idx < MATMUL_TILE * MATMUL_BK; idx += num_threads) { \
        int si = idx / MATMUL_BK, sj = idx % MATMUL_BK; \
        SA(si, sj) = (block_row + si < M) ? A[(block_row + si) * K + bk + sj] : 0.0f; \
    }
#define LOAD_A_NN_EDGE(bk) \
    for (int idx = tid; idx < MATMUL_TILE * MATMUL_BK; idx += num_threads) { \
        int si = idx / MATMUL_BK, sj = idx % MATMUL_BK; \
        int gk = bk + sj; \
        SA(si, sj) = (block_row + si < M && gk < K) ? A[(block_row + si) * K + gk] : 0.0f; \
    }
#define LOAD_B_NN(bk) \
    for (int idx = tid; idx < MATMUL_BK * MATMUL_TILE; idx += num_threads) { \
        int si = idx / MATMUL_TILE, sj = idx % MATMUL_TILE; \
        SB(si, sj) = (block_col + sj < N) ? B[(bk + si) * N + block_col + sj] : 0.0f; \
    }
#define LOAD_B_NN_EDGE(bk) \
    for (int idx = tid; idx < MATMUL_BK * MATMUL_TILE; idx += num_threads) { \
        int si = idx / MATMUL_TILE, sj = idx % MATMUL_TILE; \
        int gk = bk + si; \
        SB(si, sj) = (block_col + sj < N && gk < K) ? B[gk * N + block_col + sj] : 0.0f; \
    }

__global__ void matmul_tiled_nn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int accumulate)
{
    MATMUL_PREAMBLE();
    int bk = 0;
    // Interior tiles: no bounds checks at all
    for (; bk + MATMUL_BK <= K; bk += MATMUL_BK) {
        LOAD_A_NN(bk);
        LOAD_B_NN(bk);
        MATMUL_COMPUTE_AND_STORE();
    }
    // Last partial tile (if K not multiple of BK): check K only
    if (bk < K) {
        LOAD_A_NN_EDGE(bk);
        LOAD_B_NN_EDGE(bk);
        MATMUL_COMPUTE_AND_STORE();
    }
    MATMUL_STORE(accumulate);
}

// ── C = A^T @ B (transpose A) ──
__global__ void matmul_tiled_tn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int accumulate)
{
    MATMUL_PREAMBLE();
    int bk = 0;
    for (; bk + MATMUL_BK <= K; bk += MATMUL_BK) {
        for (int idx = tid; idx < MATMUL_TILE * MATMUL_BK; idx += num_threads) {
            int si = idx / MATMUL_BK, sj = idx % MATMUL_BK;
            SA(si, sj) = (block_row + si < M) ? A[(bk + sj) * M + block_row + si] : 0.0f;
        }
        LOAD_B_NN(bk);
        MATMUL_COMPUTE_AND_STORE();
    }
    if (bk < K) {
        for (int idx = tid; idx < MATMUL_TILE * MATMUL_BK; idx += num_threads) {
            int si = idx / MATMUL_BK, sj = idx % MATMUL_BK;
            int gk = bk + sj;
            SA(si, sj) = (block_row + si < M && gk < K) ? A[gk * M + block_row + si] : 0.0f;
        }
        LOAD_B_NN_EDGE(bk);
        MATMUL_COMPUTE_AND_STORE();
    }
    MATMUL_STORE(accumulate);
}

// ── C = A @ B^T (transpose B) ──
__global__ void matmul_tiled_nt_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int accumulate)
{
    MATMUL_PREAMBLE();
    int bk = 0;
    for (; bk + MATMUL_BK <= K; bk += MATMUL_BK) {
        LOAD_A_NN(bk);
        for (int idx = tid; idx < MATMUL_BK * MATMUL_TILE; idx += num_threads) {
            int si = idx / MATMUL_TILE, sj = idx % MATMUL_TILE;
            SB(si, sj) = (block_col + sj < N) ? B[(block_col + sj) * K + bk + si] : 0.0f;
        }
        MATMUL_COMPUTE_AND_STORE();
    }
    if (bk < K) {
        LOAD_A_NN_EDGE(bk);
        for (int idx = tid; idx < MATMUL_BK * MATMUL_TILE; idx += num_threads) {
            int si = idx / MATMUL_TILE, sj = idx % MATMUL_TILE;
            int gk = bk + si;
            SB(si, sj) = (block_col + sj < N && gk < K) ? B[(block_col + sj) * K + gk] : 0.0f;
        }
        MATMUL_COMPUTE_AND_STORE();
    }
    MATMUL_STORE(accumulate);
}

#undef LOAD_A_NN
#undef LOAD_A_NN_EDGE
#undef LOAD_B_NN
#undef LOAD_B_NN_EDGE

#undef SA
#undef SB
#undef MATMUL_COMPUTE_AND_STORE
#undef MATMUL_PREAMBLE
#undef MATMUL_STORE

// ─────────────────────────────────────────────────────────────────────────────
// 3b. MATMUL BF16 — Tensor core wmma path (Ampere+ sm_80+)
// ─────────────────────────────────────────────────────────────────────────────
//
// FP32 inputs → BF16 conversion in shared memory → wmma 16×16×16 → FP32 output
// Same precision semantics as PyTorch autocast BF16: operands BF16, accumulator FP32
// RMSNorm, softmax, attention scores, AdamW all stay FP32 — only matmul goes BF16
//
// Hardware wmma tile: 16×16×16 (fixed by ISA, not tunable)
// Block tiling: WMMA_BLOCK_M × WMMA_BLOCK_N output, WMMA_BLOCK_K along reduction
// Warp layout: WMMA_NUM_WARPS warps, each handles a sub-region of the block tile
//

#if defined(CUDA_ARCH_NUM) && CUDA_ARCH_NUM >= 80

#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

#ifndef WMMA_BLOCK_M
#error "WMMA_BLOCK_M not defined — run gpu-profile-query --defines"
#endif
#ifndef WMMA_BLOCK_N
#error "WMMA_BLOCK_N not defined — run gpu-profile-query --defines"
#endif
#ifndef WMMA_BLOCK_K
#error "WMMA_BLOCK_K not defined — run gpu-profile-query --defines"
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#ifndef WMMA_NUM_WARPS
#error "WMMA_NUM_WARPS not defined — run gpu-profile-query --defines"
#endif
#ifndef WARP_LAYOUT_M
#error "WARP_LAYOUT_M not defined — run gpu-profile-query --defines"
#endif

#define WMMA_THREADS (WMMA_NUM_WARPS * GPU_WARP_SIZE)
#define WARP_LAYOUT_N (WMMA_NUM_WARPS / WARP_LAYOUT_M)
#define WARP_TILES_M (WMMA_BLOCK_M / WMMA_M / WARP_LAYOUT_M)
#define WARP_TILES_N (WMMA_BLOCK_N / WMMA_N / WARP_LAYOUT_N)

// Bank-conflict padding: 32 banks × 4 bytes/bank, BF16 = 2 bytes/element
// Pad stride by bank_width / sizeof(bf16) × 4 = 8 BF16 elements (16 bytes = 4 bank offset)
#define WMMA_BANK_PAD (4 / (int)sizeof(__nv_bfloat16) * 4)
#define WMMA_SA_STRIDE (WMMA_BLOCK_K + WMMA_BANK_PAD)
#define WMMA_SB_STRIDE (WMMA_BLOCK_N + WMMA_BANK_PAD)

// Shared memory size for wmma compute phase (bytes)
#define WMMA_SMEM_BYTES ((WMMA_BLOCK_M * WMMA_SA_STRIDE + WMMA_BLOCK_K * WMMA_SB_STRIDE) * (int)sizeof(__nv_bfloat16))

// Preamble: shared memory pointers, warp identification, accumulator init
#define WMMA_PREAMBLE() \
    extern __shared__ char _wmma_smem_raw[]; \
    __nv_bfloat16* sA = (__nv_bfloat16*)_wmma_smem_raw; \
    __nv_bfloat16* sB = sA + WMMA_BLOCK_M * WMMA_SA_STRIDE; \
    const int warp_id = threadIdx.x / 32; \
    const int lane_id = threadIdx.x % 32; \
    const int warp_row = (warp_id / WARP_LAYOUT_N) * WARP_TILES_M; \
    const int warp_col = (warp_id % WARP_LAYOUT_N) * WARP_TILES_N; \
    const int block_row = blockIdx.y * WMMA_BLOCK_M; \
    const int block_col = blockIdx.x * WMMA_BLOCK_N; \
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILES_M][WARP_TILES_N]; \
    for (int _i = 0; _i < WARP_TILES_M; _i++) \
        for (int _j = 0; _j < WARP_TILES_N; _j++) \
            wmma::fill_fragment(acc[_i][_j], 0.0f);

// wmma compute: iterate K-tiles within one shared memory load
#define WMMA_COMPUTE() \
    __syncthreads(); \
    for (int kk = 0; kk < WMMA_BLOCK_K / WMMA_K; kk++) { \
        for (int wi = 0; wi < WARP_TILES_M; wi++) { \
            for (int wj = 0; wj < WARP_TILES_N; wj++) { \
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag; \
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag; \
                wmma::load_matrix_sync(a_frag, \
                    sA + (warp_row + wi) * WMMA_M * WMMA_SA_STRIDE + kk * WMMA_K, \
                    WMMA_SA_STRIDE); \
                wmma::load_matrix_sync(b_frag, \
                    sB + kk * WMMA_K * WMMA_SB_STRIDE + (warp_col + wj) * WMMA_N, \
                    WMMA_SB_STRIDE); \
                wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]); \
            } \
        } \
    } \
    __syncthreads();

// Store results to global memory with boundary handling
// After K-loop, shared memory is free — reuse for staging boundary tiles
#define WMMA_STORE(accumulate) \
    { \
        float* _staging = (float*)_wmma_smem_raw + warp_id * WMMA_M * WMMA_N; \
        for (int wi = 0; wi < WARP_TILES_M; wi++) { \
            for (int wj = 0; wj < WARP_TILES_N; wj++) { \
                int c_row = block_row + (warp_row + wi) * WMMA_M; \
                int c_col = block_col + (warp_col + wj) * WMMA_N; \
                if (c_row >= M || c_col >= N) continue; \
                wmma::store_matrix_sync(_staging, acc[wi][wj], WMMA_N, wmma::mem_row_major); \
                __syncwarp(); \
                for (int _idx = lane_id; _idx < WMMA_M * WMMA_N; _idx += 32) { \
                    int _r = _idx / WMMA_N; \
                    int _c = _idx % WMMA_N; \
                    if (c_row + _r < M && c_col + _c < N) { \
                        float _val = _staging[_idx]; \
                        if (accumulate) _val += C[(c_row + _r) * N + (c_col + _c)]; \
                        C[(c_row + _r) * N + (c_col + _c)] = _val; \
                    } \
                } \
                __syncwarp(); \
            } \
        } \
    }

// ── Load helpers: FP32 global → BF16 shared ──

// NN: A[M×K] row-major, B[K×N] row-major
#define WMMA_LOAD_A_NN(bk) \
    for (int idx = threadIdx.x; idx < WMMA_BLOCK_M * WMMA_BLOCK_K; idx += WMMA_THREADS) { \
        int si = idx / WMMA_BLOCK_K, sj = idx % WMMA_BLOCK_K; \
        int gr = block_row + si, gc = bk + sj; \
        sA[si * WMMA_SA_STRIDE + sj] = (gr < M && gc < K) ? __float2bfloat16_rn(A[gr * K + gc]) : __float2bfloat16_rn(0.0f); \
    }
#define WMMA_LOAD_B_NN(bk) \
    for (int idx = threadIdx.x; idx < WMMA_BLOCK_K * WMMA_BLOCK_N; idx += WMMA_THREADS) { \
        int si = idx / WMMA_BLOCK_N, sj = idx % WMMA_BLOCK_N; \
        int gr = bk + si, gc = block_col + sj; \
        sB[si * WMMA_SB_STRIDE + sj] = (gr < K && gc < N) ? __float2bfloat16_rn(B[gr * N + gc]) : __float2bfloat16_rn(0.0f); \
    }

// TN: A stored as [K×M] (transposed), load as sA[m][k] = A[k*M + m]
#define WMMA_LOAD_A_TN(bk) \
    for (int idx = threadIdx.x; idx < WMMA_BLOCK_M * WMMA_BLOCK_K; idx += WMMA_THREADS) { \
        int si = idx / WMMA_BLOCK_K, sj = idx % WMMA_BLOCK_K; \
        int gr = block_row + si, gc = bk + sj; \
        sA[si * WMMA_SA_STRIDE + sj] = (gr < M && gc < K) ? __float2bfloat16_rn(A[gc * M + gr]) : __float2bfloat16_rn(0.0f); \
    }

// NT: B stored as [N×K] (transposed), load as sB[k][n] = B[n*K + k]
#define WMMA_LOAD_B_NT(bk) \
    for (int idx = threadIdx.x; idx < WMMA_BLOCK_K * WMMA_BLOCK_N; idx += WMMA_THREADS) { \
        int si = idx / WMMA_BLOCK_N, sj = idx % WMMA_BLOCK_N; \
        int gr = bk + si, gc = block_col + sj; \
        sB[si * WMMA_SB_STRIDE + sj] = (gr < K && gc < N) ? __float2bfloat16_rn(B[gc * K + gr]) : __float2bfloat16_rn(0.0f); \
    }

// ── C = A @ B (no transpose, wmma) ──
__global__ void matmul_wmma_nn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int accumulate)
{
    WMMA_PREAMBLE();
    for (int bk = 0; bk < K; bk += WMMA_BLOCK_K) {
        WMMA_LOAD_A_NN(bk);
        WMMA_LOAD_B_NN(bk);
        WMMA_COMPUTE();
    }
    WMMA_STORE(accumulate);
}

// ── C = A^T @ B (transpose A, wmma) ──
__global__ void matmul_wmma_tn_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int accumulate)
{
    WMMA_PREAMBLE();
    for (int bk = 0; bk < K; bk += WMMA_BLOCK_K) {
        WMMA_LOAD_A_TN(bk);
        WMMA_LOAD_B_NN(bk);
        WMMA_COMPUTE();
    }
    WMMA_STORE(accumulate);
}

// ── C = A @ B^T (transpose B, wmma) ──
__global__ void matmul_wmma_nt_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K, int accumulate)
{
    WMMA_PREAMBLE();
    for (int bk = 0; bk < K; bk += WMMA_BLOCK_K) {
        WMMA_LOAD_A_NN(bk);
        WMMA_LOAD_B_NT(bk);
        WMMA_COMPUTE();
    }
    WMMA_STORE(accumulate);
}

#undef WMMA_PREAMBLE
#undef WMMA_COMPUTE
#undef WMMA_STORE
#undef WMMA_LOAD_A_NN
#undef WMMA_LOAD_B_NN
#undef WMMA_LOAD_A_TN
#undef WMMA_LOAD_B_NT

#endif // CUDA_ARCH_NUM >= 80

// ─────────────────────────────────────────────────────────────────────────────
// 4. ROPE - Rotary Position Embedding (applied to Q and K)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void rope_kernel(const float* x_in, float* x_out,
                            int seq_len, int num_heads, int head_dim, float theta_base) {
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int pair = threadIdx.x;  // pair index (0 to head_dim/2 - 1)

    if (pos >= seq_len || head >= num_heads || pair >= head_dim / 2) return;

    int offset = pos * num_heads * head_dim + head * head_dim + pair * 2;

    float freq = 1.0f / powf(theta_base, (float)(pair * 2) / head_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = x_in[offset];
    float x1 = x_in[offset + 1];

    x_out[offset]     = x0 * cos_a - x1 * sin_a;
    x_out[offset + 1] = x0 * sin_a + x1 * cos_a;
}

// Packed RoPE: grid = (total_tokens, num_heads), block = head_dim/2
// seq_starts[batch_size+1]: prefix sum of sequence lengths
// Position for angle = global_pos - seq_starts[seq_idx] (local within sequence)
__device__ int find_seq_idx(const uint32_t* seq_starts, int batch_size, int global_pos) {
    // Binary search for which sequence this token belongs to
    int lo = 0, hi = batch_size;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if ((int)seq_starts[mid + 1] <= global_pos) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void rope_packed_kernel(const float* x_in, float* x_out,
                                    const uint32_t* seq_starts, int total_tokens,
                                    int batch_size, int num_heads, int head_dim,
                                    float theta_base) {
    int global_pos = blockIdx.x;
    int head = blockIdx.y;
    int pair = threadIdx.x;

    if (global_pos >= total_tokens || head >= num_heads || pair >= head_dim / 2) return;

    int seq_idx = find_seq_idx(seq_starts, batch_size, global_pos);
    int local_pos = global_pos - (int)seq_starts[seq_idx];

    int offset = global_pos * num_heads * head_dim + head * head_dim + pair * 2;

    float freq = 1.0f / powf(theta_base, (float)(pair * 2) / head_dim);
    float angle = local_pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = x_in[offset];
    float x1 = x_in[offset + 1];

    x_out[offset]     = x0 * cos_a - x1 * sin_a;
    x_out[offset + 1] = x0 * sin_a + x1 * cos_a;
}

__global__ void rope_backward_packed_kernel(const float* d_in, float* d_out,
                                             const uint32_t* seq_starts, int total_tokens,
                                             int batch_size, int num_heads, int head_dim,
                                             float theta_base) {
    int global_pos = blockIdx.x;
    int head = blockIdx.y;
    int pair = threadIdx.x;

    if (global_pos >= total_tokens || head >= num_heads || pair >= head_dim / 2) return;

    int seq_idx = find_seq_idx(seq_starts, batch_size, global_pos);
    int local_pos = global_pos - (int)seq_starts[seq_idx];

    int offset = global_pos * num_heads * head_dim + head * head_dim + pair * 2;

    float freq = 1.0f / powf(theta_base, (float)(pair * 2) / head_dim);
    float angle = local_pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float d0 = d_in[offset];
    float d1 = d_in[offset + 1];

    d_out[offset]     =  d0 * cos_a + d1 * sin_a;
    d_out[offset + 1] = -d0 * sin_a + d1 * cos_a;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. FUSED ATTENTION - Q @ K^T → softmax → @ V (THE MAIN EVENT)
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// 5. STANDARD ATTENTION — full seq×seq scores in shared memory (default)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void batch_attention_kernel(const float* Q, const float* K, const float* V, float* out,
                                        int seq_len, int num_heads, int kv_heads, int head_dim) {
    extern __shared__ float shared[];
    float* s_scores = shared;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float* s_warp = shared + seq_len;

    int head = blockIdx.x;
    int query_pos = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    if (head >= num_heads || query_pos >= seq_len) return;

    int kv_head = head * kv_heads / num_heads;
    int attend_len = query_pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);
    int q_offset = query_pos * num_heads * head_dim + head * head_dim;

    for (int j = tid; j < attend_len; j += num_threads) {
        int k_offset = j * kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += Q[q_offset + d] * K[k_offset + d];
        s_scores[j] = score * scale;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int j = tid; j < attend_len; j += num_threads)
        local_max = fmaxf(local_max, s_scores[j]);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, offset));
    if (lane_id == 0) s_warp[warp_id] = local_max;
    __syncthreads();
    float max_val = -1e30f;
    if (tid < num_warps) max_val = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            max_val = fmaxf(max_val, __shfl_down_sync(~0u, max_val, offset));
    if (tid == 0) s_warp[0] = max_val;
    __syncthreads();
    max_val = s_warp[0];

    float local_sum = 0.0f;
    for (int j = tid; j < attend_len; j += num_threads) {
        float val = expf(s_scores[j] - max_val);
        s_scores[j] = val;
        local_sum += val;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(~0u, local_sum, offset);
    if (lane_id == 0) s_warp[warp_id] = local_sum;
    __syncthreads();
    float sum = 0.0f;
    if (tid < num_warps) sum = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(~0u, sum, offset);
    if (tid == 0) s_warp[0] = 1.0f / sum;
    __syncthreads();
    float inv_sum = s_warp[0];

    for (int j = tid; j < attend_len; j += num_threads)
        s_scores[j] *= inv_sum;
    __syncthreads();

    int out_offset = query_pos * num_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        float val = 0.0f;
        for (int j = 0; j < attend_len; j++) {
            int v_offset = j * kv_heads * head_dim + kv_head * head_dim;
            val += s_scores[j] * V[v_offset + d];
        }
        out[out_offset + d] = val;
    }
}

__global__ void batch_attention_backward_kernel(
    const float* Q, const float* K, const float* V,
    const float* grad_out,
    float* grad_Q, float* grad_K, float* grad_V,
    int seq_len, int num_heads, int kv_heads, int head_dim) {

    extern __shared__ float shared[];
    float* s_scores = shared;
    float* s_softmax = shared + seq_len;
    float* s_grad_scores = shared + 2 * seq_len;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float* s_warp = shared + 3 * seq_len;

    int head = blockIdx.x;
    int query_pos = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    if (head >= num_heads || query_pos >= seq_len) return;

    int kv_head = head * kv_heads / num_heads;
    int attend_len = query_pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);
    int q_offset = query_pos * num_heads * head_dim + head * head_dim;

    // Recompute scores
    for (int j = tid; j < attend_len; j += num_threads) {
        int k_offset = j * kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += Q[q_offset + d] * K[k_offset + d];
        s_scores[j] = score * scale;
    }
    __syncthreads();

    // Softmax
    float local_max = -1e30f;
    for (int j = tid; j < attend_len; j += num_threads)
        local_max = fmaxf(local_max, s_scores[j]);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, offset));
    if (lane_id == 0) s_warp[warp_id] = local_max;
    __syncthreads();
    float max_val = -1e30f;
    if (tid < num_warps) max_val = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            max_val = fmaxf(max_val, __shfl_down_sync(~0u, max_val, offset));
    if (tid == 0) s_warp[0] = max_val;
    __syncthreads();
    max_val = s_warp[0];

    float local_sum = 0.0f;
    for (int j = tid; j < attend_len; j += num_threads) {
        float e = expf(s_scores[j] - max_val);
        s_softmax[j] = e;
        local_sum += e;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(~0u, local_sum, offset);
    if (lane_id == 0) s_warp[warp_id] = local_sum;
    __syncthreads();
    float sum = 0.0f;
    if (tid < num_warps) sum = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(~0u, sum, offset);
    if (tid == 0) s_warp[0] = 1.0f / sum;
    __syncthreads();
    float inv_sum = s_warp[0];

    for (int j = tid; j < attend_len; j += num_threads)
        s_softmax[j] *= inv_sum;
    __syncthreads();

    // grad_V + grad_softmax
    for (int j = tid; j < attend_len; j += num_threads) {
        float grad_softmax_j = 0.0f;
        int v_offset = j * kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float g = grad_out[q_offset + d];
            atomicAdd(&grad_V[v_offset + d], s_softmax[j] * g);
            grad_softmax_j += g * V[v_offset + d];
        }
        s_grad_scores[j] = grad_softmax_j;
    }
    __syncthreads();

    // Softmax backward
    float local_dot = 0.0f;
    for (int j = tid; j < attend_len; j += num_threads)
        local_dot += s_softmax[j] * s_grad_scores[j];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_dot += __shfl_down_sync(~0u, local_dot, offset);
    if (lane_id == 0) s_warp[warp_id] = local_dot;
    __syncthreads();
    float dot_sum = 0.0f;
    if (tid < num_warps) dot_sum = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            dot_sum += __shfl_down_sync(~0u, dot_sum, offset);
    if (tid == 0) s_warp[0] = dot_sum;
    __syncthreads();
    dot_sum = s_warp[0];

    for (int j = tid; j < attend_len; j += num_threads)
        s_grad_scores[j] = s_softmax[j] * (s_grad_scores[j] - dot_sum) * scale;
    __syncthreads();

    // grad_Q + grad_K
    for (int d = tid; d < head_dim; d += num_threads) {
        float g_q = 0.0f;
        for (int j = 0; j < attend_len; j++) {
            int k_offset = j * kv_heads * head_dim + kv_head * head_dim;
            g_q += s_grad_scores[j] * K[k_offset + d];
            atomicAdd(&grad_K[k_offset + d], s_grad_scores[j] * Q[q_offset + d]);
        }
        grad_Q[q_offset + d] = g_q;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5a-packed. STANDARD ATTENTION — packed sequences (no padding)
// ─────────────────────────────────────────────────────────────────────────────
// Grid: (num_heads, total_tokens). Each block = one (head, query_pos) in the packed stream.
// seq_starts[batch_size+1]: prefix sum boundaries. Binary search to find sequence.
// Shared memory: max_seq_in_batch floats for scores + num_warps for reductions.
__global__ void batch_attention_packed_kernel(
    const float* Q, const float* K, const float* V, float* out,
    const uint32_t* seq_starts, int total_tokens, int batch_size,
    int num_heads, int kv_heads, int head_dim) {

    extern __shared__ float shared[];

    int head = blockIdx.x;
    int query_pos = blockIdx.y;  // global position in packed stream
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    if (head >= num_heads || query_pos >= total_tokens) return;

    // Find which sequence this query belongs to
    int seq_idx = find_seq_idx(seq_starts, batch_size, query_pos);
    int seq_start = (int)seq_starts[seq_idx];
    int local_pos = query_pos - seq_start;
    int attend_len = local_pos + 1;  // causal within this sequence

    float* s_scores = shared;
    float* s_warp = shared + attend_len;  // safe: attend_len <= seq_len <= max_seq

    int kv_head = head * kv_heads / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    int q_offset = query_pos * num_heads * head_dim + head * head_dim;

    for (int j = tid; j < attend_len; j += num_threads) {
        int k_offset = (seq_start + j) * kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += Q[q_offset + d] * K[k_offset + d];
        s_scores[j] = score * scale;
    }
    __syncthreads();

    // Max (tree reduction)
    float local_max = -1e30f;
    for (int j = tid; j < attend_len; j += num_threads)
        local_max = fmaxf(local_max, s_scores[j]);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, offset));
    if (lane_id == 0) s_warp[warp_id] = local_max;
    __syncthreads();
    float max_val = -1e30f;
    if (tid < num_warps) max_val = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            max_val = fmaxf(max_val, __shfl_down_sync(~0u, max_val, offset));
    if (tid == 0) s_warp[0] = max_val;
    __syncthreads();
    max_val = s_warp[0];

    // Exp + sum
    float local_sum = 0.0f;
    for (int j = tid; j < attend_len; j += num_threads) {
        float val = expf(s_scores[j] - max_val);
        s_scores[j] = val;
        local_sum += val;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(~0u, local_sum, offset);
    if (lane_id == 0) s_warp[warp_id] = local_sum;
    __syncthreads();
    float sum = 0.0f;
    if (tid < num_warps) sum = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(~0u, sum, offset);
    if (tid == 0) s_warp[0] = 1.0f / sum;
    __syncthreads();
    float inv_sum = s_warp[0];

    for (int j = tid; j < attend_len; j += num_threads)
        s_scores[j] *= inv_sum;
    __syncthreads();

    for (int d = tid; d < head_dim; d += num_threads) {
        float val = 0.0f;
        for (int j = 0; j < attend_len; j++) {
            int v_offset = (seq_start + j) * kv_heads * head_dim + kv_head * head_dim;
            val += s_scores[j] * V[v_offset + d];
        }
        out[q_offset + d] = val;
    }
}

// Standard attention backward — packed
// Grid: (num_heads, total_tokens)
__global__ void batch_attention_backward_packed_kernel(
    const float* Q, const float* K, const float* V,
    const float* grad_out,
    float* grad_Q, float* grad_K, float* grad_V,
    const uint32_t* seq_starts, int total_tokens, int batch_size,
    int num_heads, int kv_heads, int head_dim) {

    extern __shared__ float shared[];

    int head = blockIdx.x;
    int query_pos = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    if (head >= num_heads || query_pos >= total_tokens) return;

    int seq_idx = find_seq_idx(seq_starts, batch_size, query_pos);
    int seq_start = (int)seq_starts[seq_idx];
    int local_pos = query_pos - seq_start;
    int attend_len = local_pos + 1;

    float* s_scores = shared;
    float* s_softmax = shared + attend_len;
    float* s_grad_scores = shared + 2 * attend_len;
    float* s_warp = shared + 3 * attend_len;

    int kv_head = head * kv_heads / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    int q_offset = query_pos * num_heads * head_dim + head * head_dim;

    // Recompute scores
    for (int j = tid; j < attend_len; j += num_threads) {
        int k_offset = (seq_start + j) * kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++)
            score += Q[q_offset + d] * K[k_offset + d];
        s_scores[j] = score * scale;
    }
    __syncthreads();

    // Softmax
    float local_max = -1e30f;
    for (int j = tid; j < attend_len; j += num_threads)
        local_max = fmaxf(local_max, s_scores[j]);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, offset));
    if (lane_id == 0) s_warp[warp_id] = local_max;
    __syncthreads();
    float max_val = -1e30f;
    if (tid < num_warps) max_val = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            max_val = fmaxf(max_val, __shfl_down_sync(~0u, max_val, offset));
    if (tid == 0) s_warp[0] = max_val;
    __syncthreads();
    max_val = s_warp[0];

    float local_sum = 0.0f;
    for (int j = tid; j < attend_len; j += num_threads) {
        float e = expf(s_scores[j] - max_val);
        s_softmax[j] = e;
        local_sum += e;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(~0u, local_sum, offset);
    if (lane_id == 0) s_warp[warp_id] = local_sum;
    __syncthreads();
    float sum = 0.0f;
    if (tid < num_warps) sum = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(~0u, sum, offset);
    if (tid == 0) s_warp[0] = 1.0f / sum;
    __syncthreads();
    float inv_sum = s_warp[0];

    for (int j = tid; j < attend_len; j += num_threads)
        s_softmax[j] *= inv_sum;
    __syncthreads();

    // grad_V + grad_softmax
    for (int j = tid; j < attend_len; j += num_threads) {
        float grad_softmax_j = 0.0f;
        int v_offset = (seq_start + j) * kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float g = grad_out[q_offset + d];
            atomicAdd(&grad_V[v_offset + d], s_softmax[j] * g);
            grad_softmax_j += g * V[v_offset + d];
        }
        s_grad_scores[j] = grad_softmax_j;
    }
    __syncthreads();

    float local_dot = 0.0f;
    for (int j = tid; j < attend_len; j += num_threads)
        local_dot += s_softmax[j] * s_grad_scores[j];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_dot += __shfl_down_sync(~0u, local_dot, offset);
    if (lane_id == 0) s_warp[warp_id] = local_dot;
    __syncthreads();
    float dot_sum = 0.0f;
    if (tid < num_warps) dot_sum = s_warp[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            dot_sum += __shfl_down_sync(~0u, dot_sum, offset);
    if (tid == 0) s_warp[0] = dot_sum;
    __syncthreads();
    dot_sum = s_warp[0];

    for (int j = tid; j < attend_len; j += num_threads)
        s_grad_scores[j] = s_softmax[j] * (s_grad_scores[j] - dot_sum) * scale;
    __syncthreads();

    for (int d = tid; d < head_dim; d += num_threads) {
        float g_q = 0.0f;
        for (int j = 0; j < attend_len; j++) {
            int k_offset = (seq_start + j) * kv_heads * head_dim + kv_head * head_dim;
            g_q += s_grad_scores[j] * K[k_offset + d];
            atomicAdd(&grad_K[k_offset + d], s_grad_scores[j] * Q[q_offset + d]);
        }
        grad_Q[q_offset + d] = g_q;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5b. FLASH ATTENTION FORWARD — tiled, online softmax, no seq×seq storage
// ─────────────────────────────────────────────────────────────────────────────
// Grid: (num_heads, ceil(seq_len / FA_BR))
// Shared: sQ[BR×hd] + sK[BC×hd] + sV[BC×hd] + sO[BR×hd] + rowmax[BR] + rowsum[BR] + scores[BR×BC] + reduce[nwarps]
__global__ void flash_attention_fwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ out,
    float* __restrict__ lse,
    int seq_len, int num_heads, int kv_heads, int head_dim, float scale)
{
    int head = blockIdx.x;
    int tile_r = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int kv_head = head * kv_heads / num_heads;

    int q_row_start = tile_r * FA_BR;
    if (q_row_start >= seq_len) return;
    int q_rows = (q_row_start + FA_BR <= seq_len) ? FA_BR : (seq_len - q_row_start);

    extern __shared__ float smem[];
    float* sQ       = smem;
    float* sK       = sQ + FA_BR * head_dim;
    float* sV       = sK + FA_BC * head_dim;
    float* sO       = sV + FA_BC * head_dim;
    float* s_rowmax = sO + FA_BR * head_dim;
    float* s_rowsum = s_rowmax + FA_BR;
    float* s_scores = s_rowsum + FA_BR;
    float* s_reduce = s_scores + FA_BR * FA_BC;

    // Init output accumulator, rowmax, rowsum
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) sO[i] = 0.0f;
    for (int i = tid; i < FA_BR; i += nthreads) { s_rowmax[i] = -INFINITY; s_rowsum[i] = 0.0f; }

    // Load Q tile
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) {
        int r = i / head_dim, d = i % head_dim;
        int gr = q_row_start + r;
        sQ[i] = (gr < seq_len) ? Q[gr * num_heads * head_dim + head * head_dim + d] : 0.0f;
    }
    __syncthreads();

    // Causal: max key tile we need
    int max_key_tile = (q_row_start + q_rows - 1) / FA_BC;
    int num_kv_tiles = (seq_len + FA_BC - 1) / FA_BC;

    for (int tc = 0; tc <= max_key_tile && tc < num_kv_tiles; tc++) {
        int kc_start = tc * FA_BC;
        int kc_count = (kc_start + FA_BC <= seq_len) ? FA_BC : (seq_len - kc_start);

        // Load K, V tiles
        for (int i = tid; i < FA_BC * head_dim; i += nthreads) {
            int c = i / head_dim, d = i % head_dim;
            int gc = kc_start + c;
            float kval = (gc < seq_len) ? K[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
            float vval = (gc < seq_len) ? V[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
            sK[i] = kval;
            sV[i] = vval;
        }
        __syncthreads();

        // Compute scores S = Q @ K^T * scale with causal mask
        for (int idx = tid; idx < FA_BR * FA_BC; idx += nthreads) {
            int r = idx / FA_BC, c = idx % FA_BC;
            int gq = q_row_start + r, gk = kc_start + c;
            float score;
            if (gq >= seq_len || gk >= seq_len || gk > gq) {
                score = -INFINITY;
            } else {
                score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += sQ[r * head_dim + d] * sK[c * head_dim + d];
                score *= scale;
            }
            s_scores[idx] = score;
        }
        __syncthreads();

        // Online softmax + accumulate P @ V per query row
        for (int r = 0; r < q_rows; r++) {
            // Row max (tree reduction)
            float local_max = -INFINITY;
            for (int c = tid; c < kc_count; c += nthreads)
                local_max = fmaxf(local_max, s_scores[r * FA_BC + c]);
            BLOCK_REDUCE_MAX(local_max, s_reduce);
            __syncthreads();
            if (tid == 0) s_reduce[0] = local_max;
            __syncthreads();
            float tile_max = s_reduce[0];
            float m_old = s_rowmax[r];
            float m_new = fmaxf(m_old, tile_max);

            // Exp + row sum (tree reduction)
            float local_sum = 0.0f;
            for (int c = tid; c < kc_count; c += nthreads) {
                float s = s_scores[r * FA_BC + c];
                float p = (s == -INFINITY) ? 0.0f : expf(s - m_new);
                s_scores[r * FA_BC + c] = p;
                local_sum += p;
            }
            for (int c = kc_count + tid; c < FA_BC; c += nthreads)
                s_scores[r * FA_BC + c] = 0.0f;
            BLOCK_REDUCE_SUM(local_sum, s_reduce);
            __syncthreads();
            if (tid == 0) s_reduce[0] = local_sum;
            __syncthreads();
            float tile_sum = s_reduce[0];

            // Rescale running output + accumulate
            float correction = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            float l_new = correction * s_rowsum[r] + tile_sum;

            for (int d = tid; d < head_dim; d += nthreads)
                sO[r * head_dim + d] *= correction;
            __syncthreads();

            for (int d = tid; d < head_dim; d += nthreads) {
                float acc = 0.0f;
                for (int c = 0; c < kc_count; c++)
                    acc += s_scores[r * FA_BC + c] * sV[c * head_dim + d];
                sO[r * head_dim + d] += acc;
            }
            __syncthreads();

            if (tid == 0) { s_rowmax[r] = m_new; s_rowsum[r] = l_new; }
            __syncthreads();
        }
        __syncthreads();
    }

    // Final: O = O / l, store output + logsumexp
    for (int r = 0; r < q_rows; r++) {
        float l = s_rowsum[r];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        int gr = q_row_start + r;
        for (int d = tid; d < head_dim; d += nthreads)
            out[gr * num_heads * head_dim + head * head_dim + d] = sO[r * head_dim + d] * inv_l;
        if (tid == 0)
            lse[gr * num_heads + head] = s_rowmax[r] + logf(l > 0.0f ? l : 1.0f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5b-packed. FLASH ATTENTION FORWARD — packed sequences (no padding)
// ─────────────────────────────────────────────────────────────────────────────
// Grid: (num_heads, num_q_tiles) where num_q_tiles = ceil(total_tokens / FA_BR)
// Each Q tile row is a global position in the packed stream.
// Per-row: find_seq_idx → seq_start, local_pos. Causal mask: gk < seq_start || gk > gq.
__global__ void flash_attention_fwd_packed_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ out,
    float* __restrict__ lse,
    const uint32_t* seq_starts, int total_tokens, int batch_size,
    int num_heads, int kv_heads, int head_dim, float scale)
{
    int head = blockIdx.x;
    int tile_r = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int kv_head = head * kv_heads / num_heads;

    int q_row_start = tile_r * FA_BR;
    if (q_row_start >= total_tokens) return;
    int q_rows = (q_row_start + FA_BR <= total_tokens) ? FA_BR : (total_tokens - q_row_start);

    extern __shared__ float smem[];
    float* sQ       = smem;
    float* sK       = sQ + FA_BR * head_dim;
    float* sV       = sK + FA_BC * head_dim;
    float* sO       = sV + FA_BC * head_dim;
    float* s_rowmax = sO + FA_BR * head_dim;
    float* s_rowsum = s_rowmax + FA_BR;
    float* s_scores = s_rowsum + FA_BR;
    float* s_reduce = s_scores + FA_BR * FA_BC;
    // Per-row sequence info cached in shared memory
    float* s_seq_start = s_reduce + ((nthreads + 31) / 32);  // FA_BR entries
    float* s_seq_end   = s_seq_start + FA_BR;

    for (int i = tid; i < FA_BR * head_dim; i += nthreads) sO[i] = 0.0f;
    for (int i = tid; i < FA_BR; i += nthreads) { s_rowmax[i] = -INFINITY; s_rowsum[i] = 0.0f; }

    // Precompute per-row sequence boundaries
    for (int r = tid; r < FA_BR; r += nthreads) {
        int gq = q_row_start + r;
        if (gq < total_tokens) {
            int si = find_seq_idx(seq_starts, batch_size, gq);
            s_seq_start[r] = __int_as_float((int)seq_starts[si]);
            s_seq_end[r]   = __int_as_float((int)seq_starts[si + 1]);
        } else {
            s_seq_start[r] = __int_as_float(total_tokens);
            s_seq_end[r]   = __int_as_float(total_tokens);
        }
    }

    // Load Q tile (global positions directly)
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) {
        int r = i / head_dim, d = i % head_dim;
        int gr = q_row_start + r;
        sQ[i] = (gr < total_tokens) ? Q[gr * num_heads * head_dim + head * head_dim + d] : 0.0f;
    }
    __syncthreads();

    // For causal tiling: the max K position any row in this tile can attend to is q_row_start + q_rows - 1
    int max_key_tile = (q_row_start + q_rows - 1) / FA_BC;
    int num_kv_tiles = (total_tokens + FA_BC - 1) / FA_BC;

    for (int tc = 0; tc <= max_key_tile && tc < num_kv_tiles; tc++) {
        int kc_start = tc * FA_BC;
        int kc_count = (kc_start + FA_BC <= total_tokens) ? FA_BC : (total_tokens - kc_start);

        for (int i = tid; i < FA_BC * head_dim; i += nthreads) {
            int c = i / head_dim, d = i % head_dim;
            int gc = kc_start + c;
            float kval = (gc < total_tokens) ? K[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
            float vval = (gc < total_tokens) ? V[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
            sK[i] = kval;
            sV[i] = vval;
        }
        __syncthreads();

        // Compute scores with packed causal mask
        for (int idx = tid; idx < FA_BR * FA_BC; idx += nthreads) {
            int r = idx / FA_BC, c = idx % FA_BC;
            int gq = q_row_start + r, gk = kc_start + c;
            int row_seq_start = __float_as_int(s_seq_start[r]);
            float score;
            // K must be in same sequence AND causally before: gk >= seq_start && gk <= gq
            if (gq >= total_tokens || gk >= total_tokens || gk < row_seq_start || gk > gq) {
                score = -INFINITY;
            } else {
                score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += sQ[r * head_dim + d] * sK[c * head_dim + d];
                score *= scale;
            }
            s_scores[idx] = score;
        }
        __syncthreads();

        for (int r = 0; r < q_rows; r++) {
            float local_max = -INFINITY;
            for (int c = tid; c < kc_count; c += nthreads)
                local_max = fmaxf(local_max, s_scores[r * FA_BC + c]);
            BLOCK_REDUCE_MAX(local_max, s_reduce);
            __syncthreads();
            if (tid == 0) s_reduce[0] = local_max;
            __syncthreads();
            float tile_max = s_reduce[0];
            float m_old = s_rowmax[r];
            float m_new = fmaxf(m_old, tile_max);

            float local_sum = 0.0f;
            for (int c = tid; c < kc_count; c += nthreads) {
                float s = s_scores[r * FA_BC + c];
                float p = (s == -INFINITY) ? 0.0f : expf(s - m_new);
                s_scores[r * FA_BC + c] = p;
                local_sum += p;
            }
            for (int c = kc_count + tid; c < FA_BC; c += nthreads)
                s_scores[r * FA_BC + c] = 0.0f;
            BLOCK_REDUCE_SUM(local_sum, s_reduce);
            __syncthreads();
            if (tid == 0) s_reduce[0] = local_sum;
            __syncthreads();
            float tile_sum = s_reduce[0];

            float correction = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            float l_new = correction * s_rowsum[r] + tile_sum;

            for (int d = tid; d < head_dim; d += nthreads)
                sO[r * head_dim + d] *= correction;
            __syncthreads();

            for (int d = tid; d < head_dim; d += nthreads) {
                float acc = 0.0f;
                for (int c = 0; c < kc_count; c++)
                    acc += s_scores[r * FA_BC + c] * sV[c * head_dim + d];
                sO[r * head_dim + d] += acc;
            }
            __syncthreads();

            if (tid == 0) { s_rowmax[r] = m_new; s_rowsum[r] = l_new; }
            __syncthreads();
        }
        __syncthreads();
    }

    for (int r = 0; r < q_rows; r++) {
        float l = s_rowsum[r];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        int gr = q_row_start + r;
        for (int d = tid; d < head_dim; d += nthreads)
            out[gr * num_heads * head_dim + head * head_dim + d] = sO[r * head_dim + d] * inv_l;
        if (tid == 0)
            lse[gr * num_heads + head] = s_rowmax[r] + logf(l > 0.0f ? l : 1.0f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. ELEMENTWISE ADD - skip connections
// ─────────────────────────────────────────────────────────────────────────────
__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    c[idx] = a[idx] + b[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. SILU ACTIVATION - FFN gating: x * sigmoid(x)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void silu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x / (1.0f + expf(-x));
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. ELEMENTWISE MUL - gate * up in FFN
// ─────────────────────────────────────────────────────────────────────────────
__global__ void mul_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    c[idx] = a[idx] * b[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. SCALE - utility for gradient scaling etc
// ─────────────────────────────────────────────────────────────────────────────
__global__ void scale_kernel(float* x, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    x[idx] *= alpha;
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. SOFTMAX - final logits (standalone, not fused in attention)
// ─────────────────────────────────────────────────────────────────────────────
// OG single-threaded softmax (kept as reference, unused)
// __global__ void softmax_kernel_single(...) { ... }

// Parallel softmax: blockDim.x threads cooperate on one row
// Two reductions: max (stability), exp+sum (normalize)
// Launch: <<<rows, min(cols, attention_block_size)>>>
__global__ void softmax_kernel(const float* input, float* output, int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // ── Pass 1: Parallel max reduction ──
    float local_max = -1e30f;
    for (int i = tid; i < cols; i += num_threads) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, offset));
    }
    if (lane_id == 0) shared[warp_id] = local_max;
    __syncthreads();

    float max_val = -1e30f;
    if (tid < num_warps) max_val = shared[tid];
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(~0u, max_val, offset));
        }
    }
    if (tid == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // ── Pass 2: Compute exp, write to output, parallel sum ──
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += num_threads) {
        float e = expf(row_in[i] - max_val);
        row_out[i] = e;
        local_sum += e;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(~0u, local_sum, offset);
    }
    if (lane_id == 0) shared[warp_id] = local_sum;
    __syncthreads();

    float sum = 0.0f;
    if (tid < num_warps) sum = shared[tid];
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(~0u, sum, offset);
        }
    }
    if (tid == 0) shared[0] = 1.0f / sum;
    __syncthreads();
    float inv_sum = shared[0];

    // ── Pass 3: Normalize ──
    for (int i = tid; i < cols; i += num_threads) {
        row_out[i] *= inv_sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNELS - BACKWARD PASS (reverse order of forward)
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// CROSS ENTROPY GRAD - loss gradient (includes inline softmax)
// ─────────────────────────────────────────────────────────────────────────────
// OG single-threaded cross-entropy (kept as reference, unused)
// __global__ void cross_entropy_grad_kernel_single(...) { ... }

// Parallel cross-entropy: blockDim.x threads cooperate on one row of vocab_size elements
// Three reductions: max (for stability), sum (for softmax), then parallel gradient write
// Launch: <<<rows, min(vocab_size, parallel_block_size)>>>
__global__ void cross_entropy_grad_kernel(const float* logits, const uint32_t* targets,
                                           float* grad_logits, float* loss_rows,
                                           int rows, int cols) {
    extern __shared__ float shared[];
    // shared layout: [0..num_warps-1] for warp reduction temps

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    const float* row_logits = logits + row * cols;
    float* row_grad = grad_logits + row * cols;
    uint32_t target = targets[row];

    // ── Pass 1: Parallel max reduction ──
    float local_max = -1e30f;
    for (int i = tid; i < cols; i += num_threads) {
        local_max = fmaxf(local_max, row_logits[i]);
    }

    // Warp-level max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, offset));
    }
    if (lane_id == 0) shared[warp_id] = local_max;
    __syncthreads();

    // Warp 0 reduces across warps
    float max_val = -1e30f;
    if (tid < num_warps) max_val = shared[tid];
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(~0u, max_val, offset));
        }
    }
    if (tid == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // ── Pass 2: Compute exp, write to grad buffer, parallel sum ──
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += num_threads) {
        float e = expf(row_logits[i] - max_val);
        row_grad[i] = e;
        local_sum += e;
    }

    // Warp-level sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(~0u, local_sum, offset);
    }
    if (lane_id == 0) shared[warp_id] = local_sum;
    __syncthreads();

    // Warp 0 reduces across warps
    float sum = 0.0f;
    if (tid < num_warps) sum = shared[tid];
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(~0u, sum, offset);
        }
    }
    if (tid == 0) {
        float inv_sum = 1.0f / sum;
        // Compute loss for this row
        loss_rows[row] = -logf(row_grad[target] * inv_sum);
        shared[0] = inv_sum;
    }
    __syncthreads();
    float inv_sum = shared[0];

    // ── Pass 3: Compute gradient = (softmax - one_hot) / rows ──
    float grad_scale = 1.0f / (float)rows;
    for (int i = tid; i < cols; i += num_threads) {
        float g = row_grad[i] * inv_sum;
        if (i == (int)target) {
            g -= 1.0f;
        }
        row_grad[i] = g * grad_scale;
    }
}

// Cross-entropy gradient — packed sequences (no padding)
// seq_starts[batch_size+1]: prefix sum boundaries
// For each row: find its sequence, check if last token (no prediction). Otherwise normal cross-entropy.
// num_valid_predictions = total_tokens - batch_size (one fewer per sequence)
__global__ void cross_entropy_grad_packed_kernel(const float* logits, const uint32_t* targets,
                                                   float* grad_logits, float* loss_rows,
                                                   const uint32_t* seq_starts,
                                                   int total_tokens, int batch_size,
                                                   int cols, int num_valid_predictions) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    if (row >= total_tokens) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (num_threads + warpSize - 1) / warpSize;

    // Find which sequence this row belongs to
    int seq_idx = find_seq_idx(seq_starts, batch_size, row);
    int seq_start = (int)seq_starts[seq_idx];
    int seq_end = (int)seq_starts[seq_idx + 1];
    int local_pos = row - seq_start;
    int seq_len = seq_end - seq_start;

    const float* row_logits = logits + row * cols;
    float* row_grad = grad_logits + row * cols;

    // Last token of sequence: no next-token prediction → zero grad and loss
    if (local_pos >= seq_len - 1) {
        for (int i = tid; i < cols; i += num_threads)
            row_grad[i] = 0.0f;
        if (tid == 0) loss_rows[row] = 0.0f;
        return;
    }

    uint32_t target = targets[row];

    // Pass 1: max reduction
    float local_max = -1e30f;
    for (int i = tid; i < cols; i += num_threads)
        local_max = fmaxf(local_max, row_logits[i]);
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(~0u, local_max, offset));
    if (lane_id == 0) shared[warp_id] = local_max;
    __syncthreads();
    float max_val = -1e30f;
    if (tid < num_warps) max_val = shared[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            max_val = fmaxf(max_val, __shfl_down_sync(~0u, max_val, offset));
    if (tid == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // Pass 2: exp + sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += num_threads) {
        float e = expf(row_logits[i] - max_val);
        row_grad[i] = e;
        local_sum += e;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(~0u, local_sum, offset);
    if (lane_id == 0) shared[warp_id] = local_sum;
    __syncthreads();
    float sum = 0.0f;
    if (tid < num_warps) sum = shared[tid];
    if (warp_id == 0)
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(~0u, sum, offset);
    if (tid == 0) {
        float inv_sum = 1.0f / sum;
        loss_rows[row] = -logf(row_grad[target] * inv_sum);
        shared[0] = inv_sum;
    }
    __syncthreads();
    float inv_sum = shared[0];

    // Pass 3: gradient = (softmax - one_hot) / num_valid_predictions
    float grad_scale = 1.0f / (float)num_valid_predictions;
    for (int i = tid; i < cols; i += num_threads) {
        float g = row_grad[i] * inv_sum;
        if (i == (int)target) g -= 1.0f;
        row_grad[i] = g * grad_scale;
    }
}

// SiLU backward
__global__ void silu_backward_kernel(const float* input, const float* grad_output,
                                      float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + expf(-x));
    float silu_grad = sig + x * sig * (1.0f - sig); //distributed form
    grad_input[idx] = grad_output[idx] * silu_grad;
}

// Elementwise multiply backward: grad_A = grad_C * B, grad_B = grad_C * A
__global__ void elementwise_mul_backward_kernel(const float* A, const float* B,
                                                 const float* grad_C,
                                                 float* grad_A, float* grad_B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    grad_A[idx] = grad_C[idx] * B[idx];
    grad_B[idx] = grad_C[idx] * A[idx];
}

// RMSNorm backward
__global__ void rmsnorm_backward_kernel(const float* input, const float* scale,
                                         const float* grad_output,
                                         float* grad_input, float* grad_scale,
                                         int hidden_size, float eps) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_input = input + row * hidden_size;
    const float* row_grad_out = grad_output + row * hidden_size;
    float* row_grad_input = grad_input + row * hidden_size;

    // Step 1: Compute RMS
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum_sq += row_input[i] * row_input[i];
    }

    // Reduce sum_sq
    shared[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / hidden_size + eps);
    float inv_rms = 1.0f / rms;
    float inv_rms3 = inv_rms * inv_rms * inv_rms;

    // Step 2: Gradient for scale (accumulate atomically)
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = row_input[i];
        float g = row_grad_out[i];
        atomicAdd(&grad_scale[i], g * x * inv_rms);
    }

    // Step 3: Compute coupled term
    float coupled = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        coupled += row_grad_out[i] * scale[i] * row_input[i];
    }

    shared[tid] = coupled;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float sum_coupled = shared[0];

    // Step 4: Final gradient for input
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = row_input[i];
        float g = row_grad_out[i];
        float s = scale[i];
        float grad_direct = g * s * inv_rms;
        float grad_coupled = (x * inv_rms3 / hidden_size) * sum_coupled;
        row_grad_input[i] = grad_direct - grad_coupled;
    }
}

// RoPE backward (inverse rotation)
__global__ void rope_backward_kernel(const float* d_in, float* d_out,
                                      int seq_len, int num_heads, int head_dim, float theta_base) {
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int pair = threadIdx.x;

    if (pos >= seq_len || head >= num_heads || pair >= head_dim / 2) return;

    int offset = pos * num_heads * head_dim + head * head_dim + pair * 2;

    float freq = 1.0f / powf(theta_base, (float)(pair * 2) / head_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Reverse rotation: R^T
    float d0 = d_in[offset];
    float d1 = d_in[offset + 1];

    d_out[offset]     =  d0 * cos_a + d1 * sin_a;
    d_out[offset + 1] = -d0 * sin_a + d1 * cos_a;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5b. FLASH ATTENTION BACKWARD — tiled, recomputes P from logsumexp
// ─────────────────────────────────────────────────────────────────────────────
// Grid: (num_heads, ceil(seq_len / FA_BR))
// Shared: sQ[BR×hd] + sdO[BR×hd] + sK[BC×hd] + sV[BC×hd] + scores[BR×BC] + Di[BR] + lse[BR] + sdQ[BR×hd] + reduce[nwarps]
__global__ void flash_attention_bwd_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ O,
    const float* __restrict__ grad_out, const float* __restrict__ lse,
    float* __restrict__ grad_Q, float* __restrict__ grad_K, float* __restrict__ grad_V,
    int seq_len, int num_heads, int kv_heads, int head_dim, float scale)
{
    int head = blockIdx.x;
    int tile_r = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int kv_head = head * kv_heads / num_heads;

    int q_row_start = tile_r * FA_BR;
    if (q_row_start >= seq_len) return;
    int q_rows = (q_row_start + FA_BR <= seq_len) ? FA_BR : (seq_len - q_row_start);

    extern __shared__ float smem[];
    float* sQ       = smem;
    float* sdO      = sQ  + FA_BR * head_dim;
    float* sK       = sdO + FA_BR * head_dim;
    float* sV       = sK  + FA_BC * head_dim;
    float* s_scores = sV  + FA_BC * head_dim;
    float* s_Di     = s_scores + FA_BR * FA_BC;
    float* s_lse    = s_Di + FA_BR;
    float* sdQ      = s_lse + FA_BR;
    float* s_reduce = sdQ + FA_BR * head_dim;

    // Load Q, dO tiles
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) {
        int r = i / head_dim, d = i % head_dim;
        int gr = q_row_start + r;
        sQ[i]  = (gr < seq_len) ? Q[gr * num_heads * head_dim + head * head_dim + d] : 0.0f;
        sdO[i] = (gr < seq_len) ? grad_out[gr * num_heads * head_dim + head * head_dim + d] : 0.0f;
    }
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) sdQ[i] = 0.0f;
    for (int r = tid; r < FA_BR; r += nthreads) {
        int gr = q_row_start + r;
        s_lse[r] = (gr < seq_len) ? lse[gr * num_heads + head] : 0.0f;
    }
    __syncthreads();

    // Di[r] = sum_d dO[r,d] * O[r,d] (tree reduction per row)
    for (int r = 0; r < q_rows; r++) {
        float local_sum = 0.0f;
        int gr = q_row_start + r;
        for (int d = tid; d < head_dim; d += nthreads)
            local_sum += sdO[r * head_dim + d] * O[gr * num_heads * head_dim + head * head_dim + d];
        BLOCK_REDUCE_SUM(local_sum, s_reduce);
        __syncthreads();
        if (tid == 0) s_Di[r] = local_sum;
        __syncthreads();
    }

    // Iterate K/V tiles
    int max_key_tile = (q_row_start + q_rows - 1) / FA_BC;
    int num_kv_tiles = (seq_len + FA_BC - 1) / FA_BC;

    for (int tc = 0; tc <= max_key_tile && tc < num_kv_tiles; tc++) {
        int kc_start = tc * FA_BC;
        int kc_count = (kc_start + FA_BC <= seq_len) ? FA_BC : (seq_len - kc_start);

        // Load K, V tiles
        for (int i = tid; i < FA_BC * head_dim; i += nthreads) {
            int c = i / head_dim, d = i % head_dim;
            int gc = kc_start + c;
            sK[i] = (gc < seq_len) ? K[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
            sV[i] = (gc < seq_len) ? V[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
        }
        __syncthreads();

        // Recompute P[r,c] = exp(Q[r] @ K[c]^T * scale - lse[r])
        for (int idx = tid; idx < FA_BR * FA_BC; idx += nthreads) {
            int r = idx / FA_BC, c = idx % FA_BC;
            int gq = q_row_start + r, gk = kc_start + c;
            float p;
            if (gq >= seq_len || gk >= seq_len || gk > gq) {
                p = 0.0f;
            } else {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += sQ[r * head_dim + d] * sK[c * head_dim + d];
                p = expf(score * scale - s_lse[r]);
            }
            s_scores[idx] = p;
        }
        __syncthreads();

        // grad_V[c,d] += sum_r P[r,c] * dO[r,d]
        for (int idx = tid; idx < FA_BC * head_dim; idx += nthreads) {
            int c = idx / head_dim, d = idx % head_dim;
            int gk = kc_start + c;
            if (gk >= seq_len) continue;
            float acc = 0.0f;
            for (int r = 0; r < q_rows; r++)
                acc += s_scores[r * FA_BC + c] * sdO[r * head_dim + d];
            if (acc != 0.0f)
                atomicAdd(&grad_V[gk * kv_heads * head_dim + kv_head * head_dim + d], acc);
        }
        __syncthreads();

        // dS[r,c] = P[r,c] * (sum_d dO[r,d]*V[c,d] - Di[r]) * scale
        for (int idx = tid; idx < FA_BR * FA_BC; idx += nthreads) {
            int r = idx / FA_BC, c = idx % FA_BC;
            int gq = q_row_start + r, gk = kc_start + c;
            if (gq >= seq_len || gk >= seq_len || gk > gq) {
                s_scores[idx] = 0.0f;
                continue;
            }
            float p_val = s_scores[idx];
            float dp = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dp += sdO[r * head_dim + d] * sV[c * head_dim + d];
            s_scores[idx] = p_val * (dp - s_Di[r]) * scale;
        }
        __syncthreads();

        // dQ[r,d] += sum_c dS[r,c] * K[c,d]
        for (int idx = tid; idx < FA_BR * head_dim; idx += nthreads) {
            int r = idx / head_dim, d = idx % head_dim;
            float acc = 0.0f;
            for (int c = 0; c < kc_count; c++)
                acc += s_scores[r * FA_BC + c] * sK[c * head_dim + d];
            sdQ[idx] += acc;
        }

        // dK[c,d] += sum_r dS[r,c] * Q[r,d]
        for (int idx = tid; idx < FA_BC * head_dim; idx += nthreads) {
            int c = idx / head_dim, d = idx % head_dim;
            int gk = kc_start + c;
            if (gk >= seq_len) continue;
            float acc = 0.0f;
            for (int r = 0; r < q_rows; r++)
                acc += s_scores[r * FA_BC + c] * sQ[r * head_dim + d];
            if (acc != 0.0f)
                atomicAdd(&grad_K[gk * kv_heads * head_dim + kv_head * head_dim + d], acc);
        }
        __syncthreads();
    }

    // Write grad_Q
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) {
        int r = i / head_dim, d = i % head_dim;
        int gr = q_row_start + r;
        if (gr < seq_len)
            grad_Q[gr * num_heads * head_dim + head * head_dim + d] = sdQ[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FLASH ATTENTION BACKWARD — packed sequences (no padding)
// ─────────────────────────────────────────────────────────────────────────────
// Grid: (num_heads, num_q_tiles) where num_q_tiles = ceil(total_tokens / FA_BR)
__global__ void flash_attention_bwd_packed_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, const float* __restrict__ O,
    const float* __restrict__ grad_out, const float* __restrict__ lse,
    float* __restrict__ grad_Q, float* __restrict__ grad_K, float* __restrict__ grad_V,
    const uint32_t* seq_starts, int total_tokens, int batch_size,
    int num_heads, int kv_heads, int head_dim, float scale)
{
    int head = blockIdx.x;
    int tile_r = blockIdx.y;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int kv_head = head * kv_heads / num_heads;

    int q_row_start = tile_r * FA_BR;
    if (q_row_start >= total_tokens) return;
    int q_rows = (q_row_start + FA_BR <= total_tokens) ? FA_BR : (total_tokens - q_row_start);

    extern __shared__ float smem[];
    float* sQ       = smem;
    float* sdO      = sQ  + FA_BR * head_dim;
    float* sK       = sdO + FA_BR * head_dim;
    float* sV       = sK  + FA_BC * head_dim;
    float* s_scores = sV  + FA_BC * head_dim;
    float* s_Di     = s_scores + FA_BR * FA_BC;
    float* s_lse    = s_Di + FA_BR;
    float* sdQ      = s_lse + FA_BR;
    float* s_reduce = sdQ + FA_BR * head_dim;
    float* s_seq_start = s_reduce + ((nthreads + 31) / 32);

    // Precompute per-row sequence start
    for (int r = tid; r < FA_BR; r += nthreads) {
        int gq = q_row_start + r;
        if (gq < total_tokens) {
            int si = find_seq_idx(seq_starts, batch_size, gq);
            s_seq_start[r] = __int_as_float((int)seq_starts[si]);
        } else {
            s_seq_start[r] = __int_as_float(total_tokens);
        }
    }

    // Load Q, dO tiles (global positions directly)
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) {
        int r = i / head_dim, d = i % head_dim;
        int gr = q_row_start + r;
        sQ[i]  = (gr < total_tokens) ? Q[gr * num_heads * head_dim + head * head_dim + d] : 0.0f;
        sdO[i] = (gr < total_tokens) ? grad_out[gr * num_heads * head_dim + head * head_dim + d] : 0.0f;
    }
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) sdQ[i] = 0.0f;
    for (int r = tid; r < FA_BR; r += nthreads) {
        int gr = q_row_start + r;
        s_lse[r] = (gr < total_tokens) ? lse[gr * num_heads + head] : 0.0f;
    }
    __syncthreads();

    // Di[r] = sum_d dO[r,d] * O[r,d]
    for (int r = 0; r < q_rows; r++) {
        float local_sum = 0.0f;
        int gr = q_row_start + r;
        for (int d = tid; d < head_dim; d += nthreads)
            local_sum += sdO[r * head_dim + d] * O[gr * num_heads * head_dim + head * head_dim + d];
        BLOCK_REDUCE_SUM(local_sum, s_reduce);
        __syncthreads();
        if (tid == 0) s_Di[r] = local_sum;
        __syncthreads();
    }

    int max_key_tile = (q_row_start + q_rows - 1) / FA_BC;
    int num_kv_tiles = (total_tokens + FA_BC - 1) / FA_BC;

    for (int tc = 0; tc <= max_key_tile && tc < num_kv_tiles; tc++) {
        int kc_start = tc * FA_BC;
        int kc_count = (kc_start + FA_BC <= total_tokens) ? FA_BC : (total_tokens - kc_start);

        for (int i = tid; i < FA_BC * head_dim; i += nthreads) {
            int c = i / head_dim, d = i % head_dim;
            int gc = kc_start + c;
            sK[i] = (gc < total_tokens) ? K[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
            sV[i] = (gc < total_tokens) ? V[gc * kv_heads * head_dim + kv_head * head_dim + d] : 0.0f;
        }
        __syncthreads();

        // Recompute P[r,c] with packed causal mask
        for (int idx = tid; idx < FA_BR * FA_BC; idx += nthreads) {
            int r = idx / FA_BC, c = idx % FA_BC;
            int gq = q_row_start + r, gk = kc_start + c;
            int row_seq_start = __float_as_int(s_seq_start[r]);
            float p;
            if (gq >= total_tokens || gk >= total_tokens || gk < row_seq_start || gk > gq) {
                p = 0.0f;
            } else {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += sQ[r * head_dim + d] * sK[c * head_dim + d];
                p = expf(score * scale - s_lse[r]);
            }
            s_scores[idx] = p;
        }
        __syncthreads();

        // grad_V
        for (int idx = tid; idx < FA_BC * head_dim; idx += nthreads) {
            int c = idx / head_dim, d = idx % head_dim;
            int gk = kc_start + c;
            if (gk >= total_tokens) continue;
            float acc = 0.0f;
            for (int r = 0; r < q_rows; r++)
                acc += s_scores[r * FA_BC + c] * sdO[r * head_dim + d];
            if (acc != 0.0f)
                atomicAdd(&grad_V[gk * kv_heads * head_dim + kv_head * head_dim + d], acc);
        }
        __syncthreads();

        // dS
        for (int idx = tid; idx < FA_BR * FA_BC; idx += nthreads) {
            int r = idx / FA_BC, c = idx % FA_BC;
            int gq = q_row_start + r, gk = kc_start + c;
            int row_seq_start = __float_as_int(s_seq_start[r]);
            if (gq >= total_tokens || gk >= total_tokens || gk < row_seq_start || gk > gq) {
                s_scores[idx] = 0.0f;
                continue;
            }
            float p_val = s_scores[idx];
            float dp = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dp += sdO[r * head_dim + d] * sV[c * head_dim + d];
            s_scores[idx] = p_val * (dp - s_Di[r]) * scale;
        }
        __syncthreads();

        // dQ
        for (int idx = tid; idx < FA_BR * head_dim; idx += nthreads) {
            int r = idx / head_dim, d = idx % head_dim;
            float acc = 0.0f;
            for (int c = 0; c < kc_count; c++)
                acc += s_scores[r * FA_BC + c] * sK[c * head_dim + d];
            sdQ[idx] += acc;
        }

        // dK
        for (int idx = tid; idx < FA_BC * head_dim; idx += nthreads) {
            int c = idx / head_dim, d = idx % head_dim;
            int gk = kc_start + c;
            if (gk >= total_tokens) continue;
            float acc = 0.0f;
            for (int r = 0; r < q_rows; r++)
                acc += s_scores[r * FA_BC + c] * sQ[r * head_dim + d];
            if (acc != 0.0f)
                atomicAdd(&grad_K[gk * kv_heads * head_dim + kv_head * head_dim + d], acc);
        }
        __syncthreads();
    }

    // Write grad_Q
    for (int i = tid; i < FA_BR * head_dim; i += nthreads) {
        int r = i / head_dim, d = i % head_dim;
        int gr = q_row_start + r;
        if (gr < total_tokens)
            grad_Q[gr * num_heads * head_dim + head * head_dim + d] = sdQ[i];
    }
}

// Embedding backward (scatter-add)
__global__ void embedding_backward_kernel(const uint32_t* tokens, const float* grad_output,
                                           float* grad_embed, int seq_len, int hidden_size) {
    int pos = blockIdx.x;
    int dim = threadIdx.x;

    if (pos >= seq_len || dim >= hidden_size) return;

    uint32_t token = tokens[pos];
    atomicAdd(&grad_embed[token * hidden_size + dim], grad_output[pos * hidden_size + dim]);
}

// AdamW optimizer
__global__ void adamw_kernel(float* params, const float* grads, float* m, float* v,
                              float lr, float beta1, float beta2, float eps,
                              float weight_decay, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grads[idx];

    // Update biased moments
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    // Bias correction
    float m_hat = m[idx] / (1.0f - powf(beta1, t));
    float v_hat = v[idx] / (1.0f - powf(beta2, t));

    // Update with weight decay
    params[idx] = params[idx] * (1.0f - lr * weight_decay)
                  - lr * m_hat / (sqrtf(v_hat) + eps);
}

// Reduce sum
__global__ void reduce_sum_kernel(const float* input, float* output, int size) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, shared[0]);
}

// NaN check
__global__ void nan_check_kernel(const float* input, uint32_t* flags, int size, uint32_t flag_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = input[idx];
    if (isnan(val) || isinf(val)) {
        flags[flag_idx] = 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTEXT MANAGEMENT (extern "C")
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {

// Forward declaration for LoRA cleanup (defined at end of file)
void cuda_lora_gpu_state_free(cuda_train_context_t* ctx);

cuda_train_context_t* cuda_train_init(int device_id) {
    cuda_train_context_t* ctx = (cuda_train_context_t*)calloc(1, sizeof(cuda_train_context_t));
    if (!ctx) return NULL;

    if (device_id < 0) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        device_id = 0;
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to set device %d: %s\n", device_id, cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }

    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to create stream: %s\n", cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }

    ctx->device_id = device_id;
    ctx->stream = stream;
    ctx->initialized = 1;

    // ═══════════════════════════════════════════════════════════════════
    // GPU HARDWARE DISCOVERY — query once, drive all kernel launches
    // ═══════════════════════════════════════════════════════════════════
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // Raw hardware caps
    ctx->gpu_sm_count              = prop.multiProcessorCount;
    ctx->gpu_max_threads_per_block = prop.maxThreadsPerBlock;
    ctx->gpu_shared_mem_per_block  = (int)prop.sharedMemPerBlock;
    ctx->gpu_shared_mem_per_sm     = (int)prop.sharedMemPerMultiprocessor;
    ctx->gpu_warp_size             = prop.warpSize;
    ctx->gpu_compute_major         = prop.major;
    ctx->gpu_compute_minor         = prop.minor;
    ctx->gpu_l2_cache_bytes        = prop.l2CacheSize;
    ctx->gpu_max_shared_per_block_optin = (int)prop.sharedMemPerBlockOptin;
    ctx->gpu_max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    ctx->gpu_regs_per_sm = prop.regsPerMultiprocessor;

    // ── DERIVED LAUNCH PARAMETERS ──
    // Computed from hardware caps + data dimensions, never hardcoded.
    //
    // The goal: maximize occupancy per SM.
    //   Occupancy = active_warps / max_warps_per_SM
    //   Limited by: threads/block, shared_mem/block, registers/thread
    //
    // Key principle: block_size should match the DATA dimension being
    // parallelized, capped by hardware. Don't pick arbitrary numbers.

    ctx->matmul_tile_dim = MATMUL_TILE;
    ctx->matmul_block_k  = MATMUL_BK;

    // ── Reduction kernels (rmsnorm, softmax, cross_entropy) ──
    // These reduce across a dimension (hidden_size=768, vocab_size=50257).
    // Each thread handles (dim / block_size) elements, then warp-reduce.
    //
    // Optimal: block_size = min(max_threads_per_block, next_pow2(dim))
    // For hidden_size=768: 1024 threads → each does 1 element (768/1024, some idle)
    //   Actually round up to nearest power of 2 >= dim: 1024
    //   Or use dim directly if warp-aligned: 768 = 24 warps * 32 = 768 threads
    //   768 threads per block → 1 element per thread, zero waste
    //
    // For vocab=50257: 1024 threads → each does ~49 elements
    //   Can't launch 50257 threads per block (max 1024), so 1024 is the cap
    //
    // We store the MAX threads per block for reductions.
    // Each kernel picks min(this, dim) at launch time.
    ctx->parallel_block_size = ctx->gpu_max_threads_per_block;  // 1024 on 3050, 1024 on A100
    // Ensure it's a multiple of warp_size
    ctx->parallel_block_size = (ctx->parallel_block_size / ctx->gpu_warp_size) * ctx->gpu_warp_size;

    // ── Attention kernels ──
    // Grid: (num_heads, seq_len) — one block per (head, query_pos)
    // Threads parallelize across: attend_len (score compute) + head_dim (output compute)
    //
    // Shared memory per block: seq_len * sizeof(float) for fwd, 3*seq_len*4 for bwd
    //   seq=512: fwd=2KB, bwd=6KB → trivial, shared mem is NOT the limit
    //   seq=2048: fwd=8KB, bwd=24KB → still fits in 48KB
    //
    // Thread limit: max_threads_per_block, but also we want multiple blocks/SM
    //   for occupancy. Each block is one (head, query_pos).
    //   With 12 heads * 512 seq = 6144 blocks total across 16 SMs = 384 blocks/SM
    //   So even at 1024 threads/block we'd be occupancy-limited to 1 block/SM
    //   (1024 threads = 1536 max → only 1 block fits)
    //
    // Better: 512 threads/block → 3 blocks/SM → 1536 threads = 100% occupancy
    // Or: 256 threads/block → 6 blocks/SM → 1536 threads = 100% occupancy
    //   (but more blocks = more scheduling overhead)
    //
    // Sweet spot: largest block where floor(max_threads_SM / block) >= 2
    //   3050: 1536/block >= 2 → block <= 768 → 768 threads = 24 warps
    //   A100: 2048/block >= 2 → block <= 1024 → 1024 threads = 32 warps
    //
    // Also cap by shared memory: blocks_per_SM = shared_SM / shared_per_block
    //   At seq=512 bwd: 6KB/block → 100KB/6KB = 16 blocks/SM (not the limit)
    //   At seq=2048 bwd: 24KB/block → 100KB/24KB = 4 blocks/SM
    {
        int max_thr_sm = prop.maxThreadsPerMultiProcessor;
        // Largest block size that allows >= 2 blocks per SM
        int attn_block = (max_thr_sm / 2);
        // Round down to warp boundary
        attn_block = (attn_block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
        // Cap at max threads per block
        if (attn_block > ctx->gpu_max_threads_per_block)
            attn_block = ctx->gpu_max_threads_per_block;
        ctx->attention_block_size = attn_block;
    }
    ctx->attention_num_warps = ctx->attention_block_size / ctx->gpu_warp_size;

    // Print GPU info
    printf("  GPU: %s\n", prop.name);
    printf("  CUDA: %d.%d | %d SMs | %d cores | %.1f GB VRAM\n",
           prop.major, prop.minor,
           ctx->gpu_sm_count,
           ctx->gpu_sm_count * 128,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Threads: %d/SM, %d/block | Warps: %d/SM | Shared: %dKB/SM, %dKB/block (%dKB optin)\n",
           prop.maxThreadsPerMultiProcessor,
           ctx->gpu_max_threads_per_block,
           prop.maxThreadsPerMultiProcessor / ctx->gpu_warp_size,
           ctx->gpu_shared_mem_per_sm / 1024,
           ctx->gpu_shared_mem_per_block / 1024,
           ctx->gpu_max_shared_per_block_optin / 1024);
    printf("  L2: %dKB | Registers: %d/SM | Bus: %d-bit\n",
           ctx->gpu_l2_cache_bytes / 1024,
           prop.regsPerMultiprocessor,
           prop.memoryBusWidth);
    printf("  Matmul: TILE=%d BK=%d REG=%d\n", MATMUL_TILE, MATMUL_BK, MATMUL_REG);

    return ctx;
}

void cuda_train_get_gpu_info(cuda_train_context_t* ctx, char* name, int name_len, float* memory_gb) {
    if (!ctx || !ctx->initialized) {
        if (name && name_len > 0) name[0] = '\0';
        if (memory_gb) *memory_gb = 0.0f;
        return;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, ctx->device_id);
    if (name && name_len > 0) {
        strncpy(name, prop.name, name_len - 1);
        name[name_len - 1] = '\0';
    }
    if (memory_gb) {
        *memory_gb = (float)(prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
}

void cuda_train_free(cuda_train_context_t* ctx) {
    if (!ctx) return;

    // Helper macro for freeing buffers
    #define FREE_BUF(b) if (ctx->b) cuda_train_buffer_free(ctx, ctx->b)

    // Optimizer buffers
    FREE_BUF(weight_buffer);
    FREE_BUF(grad_buffer);
    FREE_BUF(m_buffer);
    FREE_BUF(v_buffer);

    // Forward pass buffers
    FREE_BUF(fwd_input_buffer);
    FREE_BUF(fwd_weight_buffer);
    FREE_BUF(fwd_output_buffer);

    // Attention buffers
    FREE_BUF(attn_q_buffer);
    FREE_BUF(attn_k_buffer);
    FREE_BUF(attn_v_buffer);
    FREE_BUF(attn_out_buffer);

    // Forward activation buffers
    FREE_BUF(fwd_hidden_buffer);
    FREE_BUF(fwd_hidden_norm_buffer);
    FREE_BUF(fwd_tmp_hidden_buffer);
    FREE_BUF(fwd_ffn_gate_buffer);
    FREE_BUF(fwd_ffn_up_buffer);
    FREE_BUF(fwd_ffn_hidden_buffer);
    FREE_BUF(fwd_logits_buffer);
    FREE_BUF(fwd_norm_weight_buffer);

    // Per-layer forward cache
    if (ctx->fwd_cache_x_norm_attn) {
        for (int l = 0; l < ctx->fwd_cache_num_layers; l++) {
            if (ctx->fwd_cache_x_norm_attn[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_x_norm_attn[l]);
            if (ctx->fwd_cache_x_norm_ffn[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_x_norm_ffn[l]);
            if (ctx->fwd_cache_attn_out[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_attn_out[l]);
            if (ctx->fwd_cache_ffn_gate_out[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_ffn_gate_out[l]);
            if (ctx->fwd_cache_ffn_up_out[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_ffn_up_out[l]);
            if (ctx->fwd_cache_ffn_hidden[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_ffn_hidden[l]);
            if (ctx->fwd_cache_layer_input && ctx->fwd_cache_layer_input[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_layer_input[l]);
            if (ctx->fwd_cache_post_attn_in && ctx->fwd_cache_post_attn_in[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_post_attn_in[l]);
            if (ctx->fwd_cache_silu_out && ctx->fwd_cache_silu_out[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_silu_out[l]);
            if (ctx->fwd_cache_q && ctx->fwd_cache_q[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_q[l]);
            if (ctx->fwd_cache_k && ctx->fwd_cache_k[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_k[l]);
            if (ctx->fwd_cache_v && ctx->fwd_cache_v[l]) cuda_train_buffer_free(ctx, ctx->fwd_cache_v[l]);
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
    }

    // Training buffers
    FREE_BUF(train_tokens_u32);
    FREE_BUF(train_targets_u32);
    FREE_BUF(train_grad_logits);
    FREE_BUF(train_loss_rows);
    FREE_BUF(train_reduce_tmp_a);
    FREE_BUF(train_reduce_tmp_b);
    FREE_BUF(train_seq_starts);
    FREE_BUF(train_grad_hidden);
    FREE_BUF(train_grad_tmp_hidden);
    FREE_BUF(train_grad_x_norm);
    FREE_BUF(train_grad_q);
    FREE_BUF(train_grad_k);
    FREE_BUF(train_grad_v);
    FREE_BUF(train_grad_attn);
    FREE_BUF(train_grad_ffn);
    FREE_BUF(train_grad_gate);
    FREE_BUF(train_grad_up);
    FREE_BUF(train_final_input);

    // GPU-resident weights
    FREE_BUF(gpu_w_embed);
    FREE_BUF(gpu_g_embed);
    FREE_BUF(gpu_m_embed);
    FREE_BUF(gpu_v_embed);
    FREE_BUF(gpu_w_final_norm);
    FREE_BUF(gpu_g_final_norm);
    FREE_BUF(gpu_m_final_norm);
    FREE_BUF(gpu_v_final_norm);

    // Per-layer GPU weights
    #define FREE_LAYER_BUFS(arr) \
        if (ctx->arr) { \
            for (int l = 0; l < ctx->gpu_weights_num_layers; l++) { \
                if (ctx->arr[l]) cuda_train_buffer_free(ctx, ctx->arr[l]); \
            } \
            free(ctx->arr); \
        }

    FREE_LAYER_BUFS(gpu_w_q);
    FREE_LAYER_BUFS(gpu_g_q);
    FREE_LAYER_BUFS(gpu_m_q);
    FREE_LAYER_BUFS(gpu_v_q);
    FREE_LAYER_BUFS(gpu_w_k);
    FREE_LAYER_BUFS(gpu_g_k);
    FREE_LAYER_BUFS(gpu_m_k);
    FREE_LAYER_BUFS(gpu_v_k);
    FREE_LAYER_BUFS(gpu_w_v);
    FREE_LAYER_BUFS(gpu_g_v);
    FREE_LAYER_BUFS(gpu_m_v);
    FREE_LAYER_BUFS(gpu_v_v);
    FREE_LAYER_BUFS(gpu_w_o);
    FREE_LAYER_BUFS(gpu_g_o);
    FREE_LAYER_BUFS(gpu_m_o);
    FREE_LAYER_BUFS(gpu_v_o);
    FREE_LAYER_BUFS(gpu_w_gate);
    FREE_LAYER_BUFS(gpu_g_gate);
    FREE_LAYER_BUFS(gpu_m_gate);
    FREE_LAYER_BUFS(gpu_v_gate);
    FREE_LAYER_BUFS(gpu_w_up);
    FREE_LAYER_BUFS(gpu_g_up);
    FREE_LAYER_BUFS(gpu_m_up);
    FREE_LAYER_BUFS(gpu_v_up);
    FREE_LAYER_BUFS(gpu_w_down);
    FREE_LAYER_BUFS(gpu_g_down);
    FREE_LAYER_BUFS(gpu_m_down);
    FREE_LAYER_BUFS(gpu_v_down);
    FREE_LAYER_BUFS(gpu_w_in_norm);
    FREE_LAYER_BUFS(gpu_g_in_norm);
    FREE_LAYER_BUFS(gpu_m_in_norm);
    FREE_LAYER_BUFS(gpu_v_in_norm);
    FREE_LAYER_BUFS(gpu_w_post_norm);
    FREE_LAYER_BUFS(gpu_g_post_norm);
    FREE_LAYER_BUFS(gpu_m_post_norm);
    FREE_LAYER_BUFS(gpu_v_post_norm);

    // Debug
    FREE_BUF(debug_nan_flags);

    // LoRA GPU state
    FREE_BUF(lora_hidden_tmp);
    if (ctx->lora_gpu) {
        cuda_lora_gpu_state_free(ctx);
    }

    #undef FREE_BUF
    #undef FREE_LAYER_BUFS

    if (ctx->stream) cudaStreamDestroy((cudaStream_t)ctx->stream);
    free(ctx);
}

void cuda_train_sync(cuda_train_context_t* ctx) {
    if (!ctx || !ctx->stream) return;
    cudaStreamSynchronize((cudaStream_t)ctx->stream);
}

// ═══════════════════════════════════════════════════════════════════════════
// BUFFER MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

cuda_buffer_t* cuda_train_buffer_create(cuda_train_context_t* ctx, size_t size) {
    (void)ctx;
    cuda_buffer_t* buf = (cuda_buffer_t*)malloc(sizeof(cuda_buffer_t));
    if (!buf) return NULL;

    cudaError_t err = cudaMalloc(&buf->data, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to allocate %zu bytes: %s\n", size, cudaGetErrorString(err));
        free(buf);
        return NULL;
    }

    buf->size = size;
    return buf;
}

void cuda_train_buffer_free(cuda_train_context_t* ctx, cuda_buffer_t* buf) {
    (void)ctx;
    if (!buf) return;
    if (buf->data) cudaFree(buf->data);
    free(buf);
}

void cuda_train_buffer_upload(cuda_train_context_t* ctx, cuda_buffer_t* buf,
                               const float* data, size_t count) {
    if (!ctx || !buf || !data) return;
    cudaMemcpyAsync(buf->data, data, count * sizeof(float),
                    cudaMemcpyHostToDevice, (cudaStream_t)ctx->stream);
}

void cuda_train_buffer_upload_u32(cuda_train_context_t* ctx, cuda_buffer_t* buf,
                                   const uint32_t* data, size_t count) {
    if (!ctx || !buf || !data) return;
    cudaMemcpyAsync(buf->data, data, count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, (cudaStream_t)ctx->stream);
}

void cuda_train_buffer_download(cuda_train_context_t* ctx, cuda_buffer_t* buf,
                                 float* data, size_t count) {
    if (!ctx || !buf || !data) return;
    cudaMemcpyAsync(data, buf->data, count * sizeof(float),
                    cudaMemcpyDeviceToHost, (cudaStream_t)ctx->stream);
    cudaStreamSynchronize((cudaStream_t)ctx->stream);
}

void cuda_train_buffer_download_bytes(cuda_train_context_t* ctx, cuda_buffer_t* buf,
                                       void* data, size_t bytes) {
    if (!ctx || !buf || !data) return;
    cudaMemcpyAsync(data, buf->data, bytes,
                    cudaMemcpyDeviceToHost, (cudaStream_t)ctx->stream);
    cudaStreamSynchronize((cudaStream_t)ctx->stream);
}

void cuda_train_fill_buffer(cuda_train_context_t* ctx, cuda_buffer_t* buf,
                             uint32_t value, size_t bytes) {
    if (!ctx || !buf) return;
    cudaMemsetAsync(buf->data, value, bytes, (cudaStream_t)ctx->stream);
}

void cuda_train_fill_buffer_range(cuda_train_context_t* ctx, cuda_buffer_t* buf,
                                   size_t offset, uint32_t value, size_t bytes) {
    if (!ctx || !buf) return;
    cudaMemsetAsync((char*)buf->data + offset, value, bytes, (cudaStream_t)ctx->stream);
}

void cuda_train_copy_buffer(cuda_train_context_t* ctx, cuda_buffer_t* src,
                             cuda_buffer_t* dst, size_t bytes) {
    if (!ctx || !src || !dst) return;
    cudaMemcpyAsync(dst->data, src->data, bytes,
                    cudaMemcpyDeviceToDevice, (cudaStream_t)ctx->stream);
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPUTE KERNEL WRAPPERS
// ═══════════════════════════════════════════════════════════════════════════

// Shared tiled matmul dispatch: used by both cuda_train_matmul and cuda_train_matmul_transpose
static void matmul_dispatch(cuda_train_context_t* ctx,
                             const float* A, const float* B, float* C,
                             int M, int N, int K,
                             int transpose_A, int transpose_B, int accumulate,
                             const char* label) {
    char pname[80];
    snprintf(pname, 80, "%s [%dx%d K=%d]%s%s", label, M, N, K,
             accumulate ? " +=" : "", ctx->use_bf16 ? " bf16" : "");
    PROF_BEGIN(pname);

#if defined(CUDA_ARCH_NUM) && CUDA_ARCH_NUM >= 80
    if (ctx->use_bf16) {
        dim3 block(WMMA_THREADS);
        dim3 grid((N + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N, (M + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M);
        size_t smem = WMMA_SMEM_BYTES;
        if (transpose_A && !transpose_B) {
            matmul_wmma_tn_kernel<<<grid, block, smem, (cudaStream_t)ctx->stream>>>(
                A, B, C, M, N, K, accumulate);
        } else if (!transpose_A && transpose_B) {
            matmul_wmma_nt_kernel<<<grid, block, smem, (cudaStream_t)ctx->stream>>>(
                A, B, C, M, N, K, accumulate);
        } else {
            matmul_wmma_nn_kernel<<<grid, block, smem, (cudaStream_t)ctx->stream>>>(
                A, B, C, M, N, K, accumulate);
        }
        PROF_END();
        PROF_LAUNCH(grid.x, grid.y, 1, WMMA_THREADS, 1, 1, smem);
        PROF_FLOPS((size_t)2 * M * N * K);
        return;
    }
#endif

    {
        dim3 block(MATMUL_TILE / MATMUL_REG, MATMUL_TILE / MATMUL_REG);
        dim3 grid((N + MATMUL_TILE - 1) / MATMUL_TILE, (M + MATMUL_TILE - 1) / MATMUL_TILE);
        size_t smem = sizeof(float) * (MATMUL_TILE * (MATMUL_BK + 1) + MATMUL_BK * (MATMUL_TILE + 1));
        if (transpose_A && !transpose_B) {
            matmul_tiled_tn_kernel<<<grid, block, smem, (cudaStream_t)ctx->stream>>>(
                A, B, C, M, N, K, accumulate);
        } else if (!transpose_A && transpose_B) {
            matmul_tiled_nt_kernel<<<grid, block, smem, (cudaStream_t)ctx->stream>>>(
                A, B, C, M, N, K, accumulate);
        } else {
            matmul_tiled_nn_kernel<<<grid, block, smem, (cudaStream_t)ctx->stream>>>(
                A, B, C, M, N, K, accumulate);
        }
        PROF_END();
        PROF_LAUNCH(grid.x, grid.y, 1, block.x, block.y, 1, smem);
    }
    PROF_FLOPS((size_t)2 * M * N * K);
}

void cuda_train_matmul(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B,
                        cuda_buffer_t* C, int M, int N, int K) {
    matmul_dispatch(ctx, A->data, B->data, C->data, M, N, K, 0, 0, 0, "matmul");
}

void cuda_train_matmul_transpose(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B,
                                  cuda_buffer_t* C, int M, int N, int K,
                                  int transpose_A, int transpose_B, int accumulate) {
    const char* variant = (transpose_A && !transpose_B) ? "matmul_At" :
                          (!transpose_A && transpose_B) ? "matmul_Bt" : "matmul";
    matmul_dispatch(ctx, A->data, B->data, C->data, M, N, K,
                    transpose_A, transpose_B, accumulate, variant);
}

void cuda_train_silu(cuda_train_context_t* ctx, cuda_buffer_t* input,
                      cuda_buffer_t* output, size_t count) {
    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count + block - 1) / block;
    PROF_BEGIN("silu");
    silu_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        input->data, output->data, count);
    PROF_END();
}

void cuda_train_rmsnorm(cuda_train_context_t* ctx, cuda_buffer_t* input,
                         cuda_buffer_t* scale, cuda_buffer_t* output,
                         int rows, int cols, float eps) {
    // Use attention_block_size for 2 blocks/SM = 100% occupancy
    int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : 256;
    if (block > cols) block = cols;
    block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
    if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

    // Shared memory: one float per warp for cross-warp reduction
    int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
    size_t shared_mem = num_warps * sizeof(float);

    char pname[80]; snprintf(pname, 80, "rmsnorm [%dx%d] %d-thr", rows, cols, block);
    PROF_BEGIN(pname);
    rmsnorm_kernel<<<rows, block, shared_mem, (cudaStream_t)ctx->stream>>>(
        input->data, scale->data, output->data, cols, eps);
    PROF_END();
    PROF_LAUNCH(rows,1,1,block,1,1,shared_mem);
    PROF_BYTES((size_t)rows*cols*4*3);
}

void cuda_train_softmax(cuda_train_context_t* ctx, cuda_buffer_t* input,
                         cuda_buffer_t* output, int rows, int cols) {
    int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : 256;
    if (block > cols) block = cols;
    block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
    if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

    int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
    size_t shared_mem = num_warps * sizeof(float);

    char pname[80]; snprintf(pname, 80, "softmax [%dx%d] %d-thr", rows, cols, block);
    PROF_BEGIN(pname);
    softmax_kernel<<<rows, block, shared_mem, (cudaStream_t)ctx->stream>>>(
        input->data, output->data, cols);
    PROF_END();
    PROF_LAUNCH(rows,1,1,block,1,1,shared_mem);
    PROF_BYTES((size_t)rows*cols*4*3);
}

void cuda_train_rope(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out,
                      int seq_len, int num_heads, int head_dim, float rope_theta) {
    dim3 grid(seq_len, num_heads);
    int block = head_dim / 2;
    PROF_BEGIN("rope_fwd");
    rope_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        x_in->data, x_out->data, seq_len, num_heads, head_dim, rope_theta);
    PROF_END();
}

void cuda_train_rope_packed(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out,
                             cuda_buffer_t* seq_starts, int total_tokens, int batch_size,
                             int num_heads, int head_dim, float rope_theta) {
    dim3 grid(total_tokens, num_heads);
    int block = head_dim / 2;
    PROF_BEGIN("rope_fwd_packed");
    rope_packed_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        x_in->data, x_out->data, (const uint32_t*)seq_starts->data,
        total_tokens, batch_size, num_heads, head_dim, rope_theta);
    PROF_END();
}

void cuda_train_add(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B,
                     cuda_buffer_t* C, size_t count) {
    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count + block - 1) / block;
    PROF_BEGIN("add");
    add_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        A->data, B->data, C->data, count);
    PROF_END();
}

void cuda_train_mul(cuda_train_context_t* ctx, cuda_buffer_t* A, cuda_buffer_t* B,
                     cuda_buffer_t* C, size_t count) {
    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count + block - 1) / block;
    PROF_BEGIN("mul");
    mul_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        A->data, B->data, C->data, count);
    PROF_END();
}

void cuda_train_scale(cuda_train_context_t* ctx, cuda_buffer_t* A, size_t count, float alpha) {
    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count + block - 1) / block;
    scale_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        A->data, alpha, count);
}

void cuda_train_batch_attention(cuda_train_context_t* ctx, cuda_buffer_t* Q, cuda_buffer_t* K,
                                 cuda_buffer_t* V, cuda_buffer_t* out, cuda_buffer_t* lse_out,
                                 int seq_len, int num_heads, int kv_heads, int head_dim) {
    if (ctx->use_flash_attn) {
        // FlashAttention path — derive block size from hardware
        float scale = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA_BR - 1) / FA_BR;
        dim3 grid(num_heads, num_q_tiles);

        // Shared memory as function of num_warps:
        // fwd: (2*FA_BR + 2*FA_BC) * hd + 2*FA_BR + FA_BR*FA_BC + num_warps
        // The num_warps term is tiny — smem is dominated by the tile sizes
        // Start from attention_block_size (largest block allowing ≥2 blocks/SM),
        // then cap by max_threads_per_block, round to warp boundary
        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : ctx->gpu_warp_size;
        if (block > ctx->gpu_max_threads_per_block)
            block = ctx->gpu_max_threads_per_block;
        block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
        if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

        // Verify shared memory fits, reduce block if needed
        int num_warps = block / ctx->gpu_warp_size;
        int smem_bytes = ((FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR + FA_BR + FA_BR * FA_BC + num_warps) * (int)sizeof(float);
        int smem_limit = ctx->gpu_max_shared_per_block_optin > 0 ?
                         ctx->gpu_max_shared_per_block_optin : ctx->gpu_shared_mem_per_block;
        while (smem_bytes > smem_limit && block > ctx->gpu_warp_size) {
            block -= ctx->gpu_warp_size;
            num_warps = block / ctx->gpu_warp_size;
            smem_bytes = ((FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR + FA_BR + FA_BR * FA_BC + num_warps) * (int)sizeof(float);
        }

        if (smem_bytes > 48 * 1024)
            cudaFuncSetAttribute(flash_attention_fwd_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        char pname[80]; snprintf(pname, 80, "flash_attn_fwd [seq=%d h=%d hd=%d] %d-thr", seq_len, num_heads, head_dim, block);
        PROF_BEGIN(pname);
        flash_attention_fwd_kernel<<<grid, block, smem_bytes, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, out->data, lse_out->data,
            seq_len, num_heads, kv_heads, head_dim, scale);
        PROF_END();
        PROF_LAUNCH(num_heads, num_q_tiles, 1, block, 1, 1, smem_bytes);
    } else {
        // Standard attention path (lse_out unused)
        dim3 grid(num_heads, seq_len);
        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : 32;
        if (block > seq_len && block > head_dim) {
            int useful = seq_len > head_dim ? seq_len : head_dim;
            useful = ((useful + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size) * ctx->gpu_warp_size;
            if (useful < block) block = useful;
        }
        int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
        size_t shared_mem = (seq_len + num_warps) * sizeof(float);
        char pname[80]; snprintf(pname, 80, "attn_fwd [seq=%d h=%d hd=%d] %d-thr", seq_len, num_heads, head_dim, block);
        PROF_BEGIN(pname);
        batch_attention_kernel<<<grid, block, shared_mem, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, out->data,
            seq_len, num_heads, kv_heads, head_dim);
        PROF_END();
        PROF_LAUNCH(num_heads, seq_len, 1, block, 1, 1, shared_mem);
    }
}

// Packed attention wrapper — dispatches to flash or standard packed kernels
void cuda_train_batch_attention_packed(cuda_train_context_t* ctx, cuda_buffer_t* Q, cuda_buffer_t* K,
                                        cuda_buffer_t* V, cuda_buffer_t* out, cuda_buffer_t* lse_out,
                                        cuda_buffer_t* seq_starts, int total_tokens, int batch_size,
                                        int num_heads, int kv_heads, int head_dim, int max_seq_in_batch) {
    if (ctx->use_flash_attn) {
        float scale = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (total_tokens + FA_BR - 1) / FA_BR;
        dim3 grid(num_heads, num_q_tiles);

        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : ctx->gpu_warp_size;
        if (block > ctx->gpu_max_threads_per_block)
            block = ctx->gpu_max_threads_per_block;
        block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
        if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

        int num_warps = block / ctx->gpu_warp_size;
        // Extra shared: num_warps (reduce) + 2*FA_BR (seq_start, seq_end per row)
        int smem_bytes = ((FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR + FA_BR + FA_BR * FA_BC + num_warps + 2 * FA_BR) * (int)sizeof(float);
        int smem_limit = ctx->gpu_max_shared_per_block_optin > 0 ?
                         ctx->gpu_max_shared_per_block_optin : ctx->gpu_shared_mem_per_block;
        while (smem_bytes > smem_limit && block > ctx->gpu_warp_size) {
            block -= ctx->gpu_warp_size;
            num_warps = block / ctx->gpu_warp_size;
            smem_bytes = ((FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR + FA_BR + FA_BR * FA_BC + num_warps + 2 * FA_BR) * (int)sizeof(float);
        }

        if (smem_bytes > 48 * 1024)
            cudaFuncSetAttribute(flash_attention_fwd_packed_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        PROF_BEGIN("flash_attn_fwd_packed");
        flash_attention_fwd_packed_kernel<<<grid, block, smem_bytes, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, out->data, lse_out->data,
            (const uint32_t*)seq_starts->data, total_tokens, batch_size,
            num_heads, kv_heads, head_dim, scale);
        PROF_END();
    } else {
        dim3 grid(num_heads, total_tokens);
        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : 32;
        if (block > max_seq_in_batch && block > head_dim) {
            int useful = max_seq_in_batch > head_dim ? max_seq_in_batch : head_dim;
            useful = ((useful + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size) * ctx->gpu_warp_size;
            if (useful < block) block = useful;
        }
        int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
        size_t shared_mem = (max_seq_in_batch + num_warps) * sizeof(float);
        PROF_BEGIN("attn_fwd_packed");
        batch_attention_packed_kernel<<<grid, block, shared_mem, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, out->data,
            (const uint32_t*)seq_starts->data, total_tokens, batch_size,
            num_heads, kv_heads, head_dim);
        PROF_END();
    }
}

void cuda_train_embed_lookup(cuda_train_context_t* ctx, cuda_buffer_t* tokens_u32,
                              cuda_buffer_t* embed_weight, cuda_buffer_t* out_hidden,
                              int seq_len, int hidden_size, int vocab_size) {
    (void)vocab_size;
    PROF_BEGIN("embed_lookup");
    embed_lookup_kernel<<<seq_len, hidden_size, 0, (cudaStream_t)ctx->stream>>>(
        (uint32_t*)tokens_u32->data, embed_weight->data, out_hidden->data,
        seq_len, hidden_size);
    PROF_END();
}

void cuda_train_cross_entropy_grad(cuda_train_context_t* ctx, cuda_buffer_t* logits,
                                    cuda_buffer_t* targets_u32, cuda_buffer_t* grad_logits,
                                    cuda_buffer_t* loss_rows, int rows, int cols) {
    // Use attention_block_size (designed for 2 blocks/SM = 100% occupancy)
    // rather than parallel_block_size (1024 = 1 block/SM = 66%)
    int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : 256;
    if (block > cols) block = cols;
    block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
    if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

    int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
    size_t shared_mem = num_warps * sizeof(float);

    char pname[80]; snprintf(pname, 80, "cross_entropy [%dx%d] %d-thr", rows, cols, block);
    PROF_BEGIN(pname);
    cross_entropy_grad_kernel<<<rows, block, shared_mem, (cudaStream_t)ctx->stream>>>(
        logits->data, (uint32_t*)targets_u32->data, grad_logits->data,
        loss_rows->data, rows, cols);
    PROF_END();
    PROF_LAUNCH(rows,1,1,block,1,1,shared_mem);
    PROF_BYTES((size_t)rows*cols*4*3);
}

void cuda_train_reduce_sum(cuda_train_context_t* ctx, cuda_buffer_t* in_buf,
                            cuda_buffer_t* out_buf, size_t count) {
    // Zero output first
    cudaMemsetAsync(out_buf->data, 0, sizeof(float), (cudaStream_t)ctx->stream);

    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count + block - 1) / block;
    size_t shared_mem = block * sizeof(float);
    reduce_sum_kernel<<<grid, block, shared_mem, (cudaStream_t)ctx->stream>>>(
        in_buf->data, out_buf->data, count);
}

void cuda_train_silu_backward(cuda_train_context_t* ctx, cuda_buffer_t* input,
                               cuda_buffer_t* grad_output, cuda_buffer_t* grad_input,
                               size_t count) {
    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count + block - 1) / block;
    PROF_BEGIN("silu_bwd");
    silu_backward_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        input->data, grad_output->data, grad_input->data, count);
    PROF_END();
}

void cuda_train_elementwise_mul_backward(cuda_train_context_t* ctx, cuda_buffer_t* A,
                                          cuda_buffer_t* B, cuda_buffer_t* grad_C,
                                          cuda_buffer_t* grad_A, cuda_buffer_t* grad_B,
                                          size_t count) {
    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count + block - 1) / block;
    PROF_BEGIN("mul_bwd");
    elementwise_mul_backward_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        A->data, B->data, grad_C->data, grad_A->data, grad_B->data, count);
    PROF_END();
}

void cuda_train_rmsnorm_backward_batch(cuda_train_context_t* ctx, cuda_buffer_t* input,
                                        cuda_buffer_t* scale, cuda_buffer_t* grad_output,
                                        cuda_buffer_t* grad_input, cuda_buffer_t* grad_scale,
                                        int rows, int cols, float eps) {
    int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : 256;
    if (block > cols) block = cols;
    block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
    if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;
    size_t shared_mem = block * sizeof(float);
    char pname[80]; snprintf(pname, 80, "rmsnorm_bwd [%dx%d] %d-thr", rows, cols, block);
    PROF_BEGIN(pname);
    rmsnorm_backward_kernel<<<rows, block, shared_mem, (cudaStream_t)ctx->stream>>>(
        input->data, scale->data, grad_output->data,
        grad_input->data, grad_scale->data, cols, eps);
    PROF_END();
    PROF_LAUNCH(rows,1,1,block,1,1,shared_mem);
}

void cuda_train_rope_backward(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out,
                               int seq_len, int num_heads, int head_dim, float rope_theta) {
    dim3 grid(seq_len, num_heads);
    int block = head_dim / 2;
    PROF_BEGIN("rope_bwd");
    rope_backward_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        x_in->data, x_out->data, seq_len, num_heads, head_dim, rope_theta);
    PROF_END();
}

void cuda_train_rope_backward_packed(cuda_train_context_t* ctx, cuda_buffer_t* x_in, cuda_buffer_t* x_out,
                                      cuda_buffer_t* seq_starts, int total_tokens, int batch_size,
                                      int num_heads, int head_dim, float rope_theta) {
    dim3 grid(total_tokens, num_heads);
    int block = head_dim / 2;
    PROF_BEGIN("rope_bwd_packed");
    rope_backward_packed_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        x_in->data, x_out->data, (const uint32_t*)seq_starts->data,
        total_tokens, batch_size, num_heads, head_dim, rope_theta);
    PROF_END();
}

void cuda_train_batch_attention_backward(cuda_train_context_t* ctx, cuda_buffer_t* Q,
                                          cuda_buffer_t* K, cuda_buffer_t* V,
                                          cuda_buffer_t* attn_out, cuda_buffer_t* lse,
                                          cuda_buffer_t* grad_out,
                                          cuda_buffer_t* grad_Q, cuda_buffer_t* grad_K,
                                          cuda_buffer_t* grad_V,
                                          int seq_len, int num_heads, int kv_heads, int head_dim) {
    if (ctx->use_flash_attn) {
        // FlashAttention backward — derive block size from hardware
        float scale = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (seq_len + FA_BR - 1) / FA_BR;
        dim3 grid(num_heads, num_q_tiles);

        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : ctx->gpu_warp_size;
        if (block > ctx->gpu_max_threads_per_block)
            block = ctx->gpu_max_threads_per_block;
        block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
        if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

        // bwd shared: (3*FA_BR + 2*FA_BC) * hd + FA_BR*FA_BC + 2*FA_BR + num_warps
        int num_warps = block / ctx->gpu_warp_size;
        int smem_bytes = ((FA_BR + FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR * FA_BC + FA_BR + FA_BR + num_warps) * (int)sizeof(float);
        int smem_limit = ctx->gpu_max_shared_per_block_optin > 0 ?
                         ctx->gpu_max_shared_per_block_optin : ctx->gpu_shared_mem_per_block;
        while (smem_bytes > smem_limit && block > ctx->gpu_warp_size) {
            block -= ctx->gpu_warp_size;
            num_warps = block / ctx->gpu_warp_size;
            smem_bytes = ((FA_BR + FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR * FA_BC + FA_BR + FA_BR + num_warps) * (int)sizeof(float);
        }

        if (smem_bytes > 48 * 1024)
            cudaFuncSetAttribute(flash_attention_bwd_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        char pname[80]; snprintf(pname, 80, "flash_attn_bwd [seq=%d h=%d hd=%d] %d-thr", seq_len, num_heads, head_dim, block);
        PROF_BEGIN(pname);
        flash_attention_bwd_kernel<<<grid, block, smem_bytes, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, attn_out->data,
            grad_out->data, lse->data,
            grad_Q->data, grad_K->data, grad_V->data,
            seq_len, num_heads, kv_heads, head_dim, scale);
        PROF_END();
        PROF_LAUNCH(num_heads, num_q_tiles, 1, block, 1, 1, smem_bytes);
    } else {
        // Standard attention backward (attn_out and lse unused)
        dim3 grid(num_heads, seq_len);
        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : ctx->gpu_warp_size;
        if (block > ctx->gpu_max_threads_per_block)
            block = ctx->gpu_max_threads_per_block;
        block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
        if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;
        int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
        size_t shared_mem = (3 * seq_len + num_warps) * sizeof(float);
        char pname[80]; snprintf(pname, 80, "attn_bwd [seq=%d h=%d hd=%d] %d-thr", seq_len, num_heads, head_dim, block);
        PROF_BEGIN(pname);
        batch_attention_backward_kernel<<<grid, block, shared_mem, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, grad_out->data,
            grad_Q->data, grad_K->data, grad_V->data,
            seq_len, num_heads, kv_heads, head_dim);
        PROF_END();
        PROF_LAUNCH(num_heads, seq_len, 1, block, 1, 1, shared_mem);
    }
}

// Packed attention backward wrapper
void cuda_train_batch_attention_backward_packed(cuda_train_context_t* ctx, cuda_buffer_t* Q,
                                                  cuda_buffer_t* K, cuda_buffer_t* V,
                                                  cuda_buffer_t* attn_out, cuda_buffer_t* lse,
                                                  cuda_buffer_t* grad_out,
                                                  cuda_buffer_t* grad_Q, cuda_buffer_t* grad_K,
                                                  cuda_buffer_t* grad_V,
                                                  cuda_buffer_t* seq_starts, int total_tokens,
                                                  int batch_size, int num_heads, int kv_heads,
                                                  int head_dim, int max_seq_in_batch) {
    if (ctx->use_flash_attn) {
        float scale = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (total_tokens + FA_BR - 1) / FA_BR;
        dim3 grid(num_heads, num_q_tiles);

        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : ctx->gpu_warp_size;
        if (block > ctx->gpu_max_threads_per_block)
            block = ctx->gpu_max_threads_per_block;
        block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
        if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

        int num_warps = block / ctx->gpu_warp_size;
        // Extra shared: num_warps (reduce) + FA_BR (seq_start per row)
        int smem_bytes = ((FA_BR + FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR * FA_BC + FA_BR + FA_BR + num_warps + FA_BR) * (int)sizeof(float);
        int smem_limit = ctx->gpu_max_shared_per_block_optin > 0 ?
                         ctx->gpu_max_shared_per_block_optin : ctx->gpu_shared_mem_per_block;
        while (smem_bytes > smem_limit && block > ctx->gpu_warp_size) {
            block -= ctx->gpu_warp_size;
            num_warps = block / ctx->gpu_warp_size;
            smem_bytes = ((FA_BR + FA_BR + FA_BC + FA_BC + FA_BR) * head_dim
                         + FA_BR * FA_BC + FA_BR + FA_BR + num_warps + FA_BR) * (int)sizeof(float);
        }

        if (smem_bytes > 48 * 1024)
            cudaFuncSetAttribute(flash_attention_bwd_packed_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        PROF_BEGIN("flash_attn_bwd_packed");
        flash_attention_bwd_packed_kernel<<<grid, block, smem_bytes, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, attn_out->data,
            grad_out->data, lse->data,
            grad_Q->data, grad_K->data, grad_V->data,
            (const uint32_t*)seq_starts->data, total_tokens, batch_size,
            num_heads, kv_heads, head_dim, scale);
        PROF_END();
    } else {
        dim3 grid(num_heads, total_tokens);
        int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : ctx->gpu_warp_size;
        if (block > ctx->gpu_max_threads_per_block)
            block = ctx->gpu_max_threads_per_block;
        block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
        if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;
        int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
        size_t shared_mem = (3 * max_seq_in_batch + num_warps) * sizeof(float);
        PROF_BEGIN("attn_bwd_packed");
        batch_attention_backward_packed_kernel<<<grid, block, shared_mem, (cudaStream_t)ctx->stream>>>(
            Q->data, K->data, V->data, grad_out->data,
            grad_Q->data, grad_K->data, grad_V->data,
            (const uint32_t*)seq_starts->data, total_tokens, batch_size,
            num_heads, kv_heads, head_dim);
        PROF_END();
    }
}

// Packed cross-entropy wrapper
void cuda_train_cross_entropy_grad_packed(cuda_train_context_t* ctx, cuda_buffer_t* logits,
                                            cuda_buffer_t* targets_u32, cuda_buffer_t* grad_logits,
                                            cuda_buffer_t* loss_rows, cuda_buffer_t* seq_starts,
                                            int total_tokens, int batch_size, int vocab_size,
                                            int num_valid_predictions) {
    int block = ctx->attention_block_size > 0 ? ctx->attention_block_size : 256;
    if (block > vocab_size) block = vocab_size;
    block = (block / ctx->gpu_warp_size) * ctx->gpu_warp_size;
    if (block < ctx->gpu_warp_size) block = ctx->gpu_warp_size;

    int num_warps = (block + ctx->gpu_warp_size - 1) / ctx->gpu_warp_size;
    size_t shared_mem = num_warps * sizeof(float);

    PROF_BEGIN("cross_entropy_packed");
    cross_entropy_grad_packed_kernel<<<total_tokens, block, shared_mem, (cudaStream_t)ctx->stream>>>(
        logits->data, (uint32_t*)targets_u32->data, grad_logits->data,
        loss_rows->data, (const uint32_t*)seq_starts->data,
        total_tokens, batch_size, vocab_size, num_valid_predictions);
    PROF_END();
}

void cuda_train_embedding_backward(cuda_train_context_t* ctx, cuda_buffer_t* tokens_u32,
                                    cuda_buffer_t* grad_hidden, cuda_buffer_t* grad_embed,
                                    int seq_len, int hidden_size, int vocab_size) {
    (void)vocab_size;
    PROF_BEGIN("embed_bwd");
    embedding_backward_kernel<<<seq_len, hidden_size, 0, (cudaStream_t)ctx->stream>>>(
        (uint32_t*)tokens_u32->data, grad_hidden->data, grad_embed->data,
        seq_len, hidden_size);
    PROF_END();
}

void cuda_train_adamw_update(cuda_train_context_t* ctx, cuda_buffer_t* weight,
                              cuda_buffer_t* grad, cuda_buffer_t* m, cuda_buffer_t* v,
                              float lr, float beta1, float beta2, float weight_decay,
                              float eps, int step, size_t size,
                              const char* tensor_name, int rows, int cols) {
#ifdef ADAMW_BLOCK
    int block = ADAMW_BLOCK;
#else
    int block = 512;
#endif
    int grid = (size + block - 1) / block;
    char pname[80];
    if (cols > 0)
        snprintf(pname, 80, "adamw %s [%dx%d]", tensor_name ? tensor_name : "", rows, cols);
    else
        snprintf(pname, 80, "adamw %s [%d]", tensor_name ? tensor_name : "", rows);
    PROF_BEGIN(pname);
    adamw_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        weight->data, grad->data, m->data, v->data,
        lr, beta1, beta2, eps, weight_decay, step, size);
    PROF_END();
    PROF_LAUNCH(grid,1,1,block,1,1,0);
    PROF_BYTES(size*4*4);
}

void cuda_train_nan_check(cuda_train_context_t* ctx, cuda_buffer_t* input,
                           size_t count_floats, cuda_buffer_t* flags_u32, uint32_t flag_index) {
    int block = ctx->gpu_max_threads_per_block > 0 ? ctx->gpu_max_threads_per_block : 256;
    int grid = (count_floats + block - 1) / block;
    nan_check_kernel<<<grid, block, 0, (cudaStream_t)ctx->stream>>>(
        input->data, (uint32_t*)flags_u32->data, count_floats, flag_index);
}

// ═══════════════════════════════════════════════════════════════════════════
// LORA OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

void cuda_train_lora_backward(cuda_train_context_t* ctx,
                               cuda_buffer_t* grad_output,
                               cuda_buffer_t* input,
                               cuda_buffer_t* A,
                               cuda_buffer_t* B,
                               cuda_buffer_t* gA,
                               cuda_buffer_t* gB,
                               cuda_buffer_t* hidden_tmp,
                               int seq_len, int in_dim, int out_dim, int rank,
                               float scale) {
    // LoRA backward pass (uses tiled/wmma matmul — same kernels as full training):
    // 1. Recompute hidden = input @ A^T  [seq, rank]
    // 2. grad_B += scale * grad_output^T @ hidden
    // 3. grad_hidden = grad_output @ B
    // 4. grad_A += scale * grad_hidden^T @ input

    // 1. hidden = input @ A^T  [seq, in_dim] @ [in_dim, rank]^T → [seq, rank]
    matmul_dispatch(ctx, input->data, A->data, hidden_tmp->data,
                    seq_len, rank, in_dim, 0, 1, 0, "lora_recompute");

    // Scale hidden_tmp for grad_B computation
    cuda_train_scale(ctx, hidden_tmp, (size_t)seq_len * (size_t)rank, scale);

    // 2. grad_B += grad_output^T @ hidden  [out_dim, seq] @ [seq, rank] → [out_dim, rank]
    matmul_dispatch(ctx, grad_output->data, hidden_tmp->data, gB->data,
                    out_dim, rank, seq_len, 1, 0, 1, "lora_gB");

    // 3. grad_hidden = grad_output @ B  [seq, out_dim] @ [out_dim, rank] → [seq, rank]
    matmul_dispatch(ctx, grad_output->data, B->data, hidden_tmp->data,
                    seq_len, rank, out_dim, 0, 0, 0, "lora_grad_hidden");

    // Scale for grad_A computation
    cuda_train_scale(ctx, hidden_tmp, (size_t)seq_len * (size_t)rank, scale);

    // 4. grad_A += grad_hidden^T @ input  [rank, seq] @ [seq, in_dim] → [rank, in_dim]
    matmul_dispatch(ctx, hidden_tmp->data, input->data, gA->data,
                    rank, in_dim, seq_len, 1, 0, 1, "lora_gA");
}

// Helper: zero gradients for one adapter type
static void cuda_lora_adapter_zero_grads(cuda_train_context_t* ctx, cuda_lora_gpu_adapter_t* gpu, int num_layers) {
    if (!gpu->active) return;
    for (int l = 0; l < num_layers; l++) {
        if (gpu->gA && gpu->gA[l]) {
            cudaMemsetAsync(gpu->gA[l]->data, 0, gpu->gA[l]->size, (cudaStream_t)ctx->stream);
        }
        if (gpu->gB && gpu->gB[l]) {
            cudaMemsetAsync(gpu->gB[l]->data, 0, gpu->gB[l]->size, (cudaStream_t)ctx->stream);
        }
    }
}

void cuda_train_zero_lora_grads(cuda_train_context_t* ctx) {
    if (!ctx || !ctx->lora_gpu) return;

    cuda_lora_gpu_state_t* lg = (cuda_lora_gpu_state_t*)ctx->lora_gpu;
    int num_layers = lg->num_layers;

    cuda_lora_adapter_zero_grads(ctx, &lg->q, num_layers);
    cuda_lora_adapter_zero_grads(ctx, &lg->k, num_layers);
    cuda_lora_adapter_zero_grads(ctx, &lg->v, num_layers);
    cuda_lora_adapter_zero_grads(ctx, &lg->o, num_layers);
    cuda_lora_adapter_zero_grads(ctx, &lg->gate, num_layers);
    cuda_lora_adapter_zero_grads(ctx, &lg->up, num_layers);
    cuda_lora_adapter_zero_grads(ctx, &lg->down, num_layers);
}

// Helper: download one adapter type's weights to CPU
static void cuda_lora_adapter_download(cuda_train_context_t* ctx, cuda_lora_gpu_adapter_t* gpu,
                                         lora_adapter_t** adapters, int num_layers) {
    if (!gpu->active || !adapters) return;
    for (int l = 0; l < num_layers; l++) {
        if (!adapters[l]) continue;
        lora_adapter_t* a = adapters[l];

        if (gpu->A[l]) {
            cuda_train_buffer_download(ctx, gpu->A[l], a->A, (size_t)a->rank * (size_t)a->in_dim);
        }
        if (gpu->B[l]) {
            cuda_train_buffer_download(ctx, gpu->B[l], a->B, (size_t)a->out_dim * (size_t)a->rank);
        }
        // Also download optimizer state for checkpoint resume
        if (gpu->mA[l]) {
            cuda_train_buffer_download(ctx, gpu->mA[l], a->m_A, (size_t)a->rank * (size_t)a->in_dim);
        }
        if (gpu->vA[l]) {
            cuda_train_buffer_download(ctx, gpu->vA[l], a->v_A, (size_t)a->rank * (size_t)a->in_dim);
        }
        if (gpu->mB[l]) {
            cuda_train_buffer_download(ctx, gpu->mB[l], a->m_B, (size_t)a->out_dim * (size_t)a->rank);
        }
        if (gpu->vB[l]) {
            cuda_train_buffer_download(ctx, gpu->vB[l], a->v_B, (size_t)a->out_dim * (size_t)a->rank);
        }
    }
}

void cuda_download_lora_weights(cuda_train_context_t* ctx, void* lora_ptr) {
    if (!ctx || !ctx->lora_gpu || !lora_ptr) return;

    lora_model_t* lora = (lora_model_t*)lora_ptr;
    cuda_lora_gpu_state_t* lg = (cuda_lora_gpu_state_t*)ctx->lora_gpu;
    int num_layers = lg->num_layers;

    cuda_lora_adapter_download(ctx, &lg->q, lora->q_adapters, num_layers);
    cuda_lora_adapter_download(ctx, &lg->k, lora->k_adapters, num_layers);
    cuda_lora_adapter_download(ctx, &lg->v, lora->v_adapters, num_layers);
    cuda_lora_adapter_download(ctx, &lg->o, lora->o_adapters, num_layers);
    cuda_lora_adapter_download(ctx, &lg->gate, lora->gate_adapters, num_layers);
    cuda_lora_adapter_download(ctx, &lg->up, lora->up_adapters, num_layers);
    cuda_lora_adapter_download(ctx, &lg->down, lora->down_adapters, num_layers);

    cudaStreamSynchronize((cudaStream_t)ctx->stream);
}

// Free one adapter type's GPU buffers
static void cuda_lora_adapter_free(cuda_train_context_t* ctx, cuda_lora_gpu_adapter_t* gpu, int num_layers) {
    if (!gpu->active) return;
    for (int l = 0; l < num_layers; l++) {
        if (gpu->A && gpu->A[l])   cuda_train_buffer_free(ctx, gpu->A[l]);
        if (gpu->B && gpu->B[l])   cuda_train_buffer_free(ctx, gpu->B[l]);
        if (gpu->gA && gpu->gA[l]) cuda_train_buffer_free(ctx, gpu->gA[l]);
        if (gpu->gB && gpu->gB[l]) cuda_train_buffer_free(ctx, gpu->gB[l]);
        if (gpu->mA && gpu->mA[l]) cuda_train_buffer_free(ctx, gpu->mA[l]);
        if (gpu->vA && gpu->vA[l]) cuda_train_buffer_free(ctx, gpu->vA[l]);
        if (gpu->mB && gpu->mB[l]) cuda_train_buffer_free(ctx, gpu->mB[l]);
        if (gpu->vB && gpu->vB[l]) cuda_train_buffer_free(ctx, gpu->vB[l]);
    }
    free(gpu->A);  free(gpu->B);
    free(gpu->gA); free(gpu->gB);
    free(gpu->mA); free(gpu->vA);
    free(gpu->mB); free(gpu->vB);
    memset(gpu, 0, sizeof(*gpu));
}

// Free all LoRA GPU state (called from cuda_train_free)
void cuda_lora_gpu_state_free(cuda_train_context_t* ctx) {
    if (!ctx || !ctx->lora_gpu) return;

    cuda_lora_gpu_state_t* lg = (cuda_lora_gpu_state_t*)ctx->lora_gpu;
    int num_layers = lg->num_layers;

    cuda_lora_adapter_free(ctx, &lg->q, num_layers);
    cuda_lora_adapter_free(ctx, &lg->k, num_layers);
    cuda_lora_adapter_free(ctx, &lg->v, num_layers);
    cuda_lora_adapter_free(ctx, &lg->o, num_layers);
    cuda_lora_adapter_free(ctx, &lg->gate, num_layers);
    cuda_lora_adapter_free(ctx, &lg->up, num_layers);
    cuda_lora_adapter_free(ctx, &lg->down, num_layers);

    free(lg);
    ctx->lora_gpu = NULL;
}

} // extern "C"
