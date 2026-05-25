#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    char name[256];
    int sm_count;
    int warp_size;
    int max_threads_per_block;
    int max_threads_per_sm;
    int regs_per_sm;
    int regs_per_block;
    size_t shared_per_block;
    size_t shared_per_sm;
    size_t shared_per_block_optin;
    size_t vram_bytes;
    int l2_cache_bytes;
    int mem_bus_width;
    int compute_major;
    int compute_minor;
    unsigned int clock_sm_mhz;
    unsigned int clock_mem_mhz;
} gpu_hw_t;

typedef struct {
    int M;
    int N;
    int K;
    int count;
} matmul_op_t;

static int query_gpu(gpu_hw_t* hw) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "no CUDA devices\n");
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    strncpy(hw->name, prop.name, sizeof(hw->name) - 1);
    hw->sm_count = prop.multiProcessorCount;
    hw->warp_size = prop.warpSize;
    hw->max_threads_per_block = prop.maxThreadsPerBlock;
    hw->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    hw->regs_per_sm = prop.regsPerMultiprocessor;
    hw->regs_per_block = prop.regsPerBlock;
    hw->shared_per_block = prop.sharedMemPerBlock;
    hw->shared_per_sm = prop.sharedMemPerMultiprocessor;
    hw->shared_per_block_optin = prop.sharedMemPerBlockOptin;
    hw->vram_bytes = prop.totalGlobalMem;
    hw->l2_cache_bytes = prop.l2CacheSize;
    hw->mem_bus_width = prop.memoryBusWidth;
    hw->compute_major = prop.major;
    hw->compute_minor = prop.minor;

    int clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    hw->clock_sm_mhz = (unsigned int)(clock_khz / 1000);
    hw->clock_mem_mhz = (unsigned int)(mem_clock_khz / 1000);

    return 0;
}

static void print_gpu(const gpu_hw_t* hw) {
    fprintf(stderr, "\n%s\n", hw->name);
    fprintf(stderr, "  compute       %d.%d\n", hw->compute_major, hw->compute_minor);
    fprintf(stderr, "  sm_count      %d\n", hw->sm_count);
    fprintf(stderr, "  warp_size     %d\n", hw->warp_size);
    fprintf(stderr, "  threads/sm    %d\n", hw->max_threads_per_sm);
    fprintf(stderr, "  threads/block %d\n", hw->max_threads_per_block);
    fprintf(stderr, "  regs/sm       %d\n", hw->regs_per_sm);
    fprintf(stderr, "  regs/block    %d\n", hw->regs_per_block);
    fprintf(stderr, "  shared/sm     %zu\n", hw->shared_per_sm);
    fprintf(stderr, "  shared/block  %zu\n", hw->shared_per_block);
    fprintf(stderr, "  shared/optin  %zu\n", hw->shared_per_block_optin);
    fprintf(stderr, "  vram          %zu\n", hw->vram_bytes);
    fprintf(stderr, "  l2            %d\n", hw->l2_cache_bytes);
    fprintf(stderr, "  bus_width     %d\n", hw->mem_bus_width);
    fprintf(stderr, "  clock_sm      %u\n", hw->clock_sm_mhz);
    fprintf(stderr, "  clock_mem     %u\n", hw->clock_mem_mhz);
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

static int load_model_config(const char* dir, int* z, int* inter, int* vocab, int* L, int* h, int* kv_h, int* hd, int* max_seq) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json = (char*)malloc(len + 1);
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

static void print_defines(const gpu_hw_t* hw, const char* model_dir) {
    int z, inter, vocab, L, h, kv_h, hd, max_seq;
    if (load_model_config(model_dir, &z, &inter, &vocab, &L, &h, &kv_h, &hd, &max_seq) != 0)
        return;

    int kv_dim = kv_h * hd;
    int seq = max_seq;

    fprintf(stderr, "  model z=%d inter=%d vocab=%d L=%d h=%d kv_h=%d hd=%d seq=%d\n",
            z, inter, vocab, L, h, kv_h, hd, seq);

    matmul_op_t ops[32];
    int nops = 0;

    // forward per layer: Q, K, V projections + O projection
    ops[nops++] = (matmul_op_t){seq, z, z, L};           // Q proj
    ops[nops++] = (matmul_op_t){seq, kv_dim, z, L};      // K proj
    ops[nops++] = (matmul_op_t){seq, kv_dim, z, L};      // V proj
    ops[nops++] = (matmul_op_t){seq, z, z, L};           // O proj
    // forward per layer: FFN gate, up, down
    ops[nops++] = (matmul_op_t){seq, inter, z, L};       // gate
    ops[nops++] = (matmul_op_t){seq, inter, z, L};       // up
    ops[nops++] = (matmul_op_t){seq, z, inter, L};       // down
    // vocab head forward + backward
    ops[nops++] = (matmul_op_t){seq, vocab, z, 1};       // logits fwd
    ops[nops++] = (matmul_op_t){seq, z, vocab, 1};       // logits bwd
    ops[nops++] = (matmul_op_t){vocab, z, seq, 1};       // embed grad
    // backward per layer: input grads (same shapes as forward)
    ops[nops++] = (matmul_op_t){seq, z, z, L};           // Q/K/V/O input grads
    ops[nops++] = (matmul_op_t){seq, inter, inter, L};   // down input grad
    ops[nops++] = (matmul_op_t){seq, z, inter, L};       // gate/up input grad
    // backward per layer: weight grads (transposed, M and K swap)
    ops[nops++] = (matmul_op_t){z, z, seq, L};           // Q/K/V/O weight grads
    ops[nops++] = (matmul_op_t){inter, z, seq, L};       // gate/up weight grads
    ops[nops++] = (matmul_op_t){z, inter, seq, L};       // down weight grads

    int best_tile = 0, best_bk = 0, best_reg = 0;
    int best_util = 0;

    for (int tile = hw->warp_size; tile <= 256; tile += hw->warp_size) {
        for (int reg = 2; reg <= 8; reg *= 2) {
            if (tile % reg != 0) continue;
            int tpb = (tile / reg) * (tile / reg);
            if (tpb > hw->max_threads_per_block) continue;
            if (tpb < hw->warp_size) continue;

            int max_rpt = hw->regs_per_block / tpb;
            if (max_rpt < reg * reg * 2) continue;

            int sm_capacity = hw->max_threads_per_sm / tpb;

            for (int bk = 64; bk >= 8; bk /= 2) {
                size_t smem = (size_t)4 * (tile * (bk + 1) + bk * (tile + 1));
                if (smem > hw->shared_per_block) continue;

                int sm_by_smem = (int)(hw->shared_per_sm / smem);
                if (sm_by_smem < sm_capacity) sm_capacity = sm_by_smem;

                int total_util = 0;
                for (int i = 0; i < nops; i++) {
                    int blocks = ((ops[i].M + tile - 1) / tile) * ((ops[i].N + tile - 1) / tile);
                    int bpsm = blocks / hw->sm_count;
                    if (bpsm > sm_capacity) bpsm = sm_capacity;
                    int active_threads = bpsm * tpb;
                    int util = (100 * active_threads) / hw->max_threads_per_sm;
                    total_util += util * ops[i].count;
                }

                if (total_util > best_util) {
                    best_util = total_util;
                    best_tile = tile;
                    best_bk = bk;
                    best_reg = reg;
                }
                break;
            }
        }
    }

    if (best_tile == 0) {
        best_tile = 64;
        best_bk = 16;
        best_reg = 4;
    }

    int thr = (best_tile / best_reg) * (best_tile / best_reg);
    size_t smem = (size_t)4 * (best_tile * (best_bk + 1) + best_bk * (best_tile + 1));
    int bpsm_s = (int)(hw->shared_per_sm / smem);
    int bpsm_t = hw->max_threads_per_sm / thr;
    int bpsm = bpsm_s < bpsm_t ? bpsm_s : bpsm_t;
    if (bpsm < 1) bpsm = 1;
    int occ = (100 * bpsm * thr) / hw->max_threads_per_sm;

    fprintf(stderr, "  tile    %d\n", best_tile);
    fprintf(stderr, "  bk      %d\n", best_bk);
    fprintf(stderr, "  reg     %d\n", best_reg);
    fprintf(stderr, "  threads %d\n", thr);
    fprintf(stderr, "  regs/t  %d\n", hw->regs_per_block / thr);
    fprintf(stderr, "  shared  %zu\n", smem);
    fprintf(stderr, "  blk/sm  %d\n", bpsm);
    fprintf(stderr, "  occup   %d%%\n", occ);

    for (int i = 0; i < nops; i++) {
        int gm = (ops[i].M + best_tile - 1) / best_tile;
        int gn = (ops[i].N + best_tile - 1) / best_tile;
        fprintf(stderr, "  op %2d: %5dx%-5d K=%-5d x%-2d  grid=%-4d blocks\n",
                i, ops[i].M, ops[i].N, ops[i].K, ops[i].count, gm * gn);
    }

    // ── WMMA (BF16 tensor core) parameter search ──
    // Fragment size 16×16×16 is ISA-fixed. We search outer block tiling.
    // WMMA_BLOCK_M/N: output tile per block (multiples of 16)
    // WMMA_BLOCK_K: K-tile loaded to shared (multiples of 16)
    // WMMA_NUM_WARPS: warps per block
    // WARP_LAYOUT_M: warp grid rows (WARP_LAYOUT_N = NUM_WARPS / WARP_LAYOUT_M)

    int wbm = 0, wbn = 0, wbk = 0, wnw = 0, wlm = 0;
    int wmma_best_util = 0;

    if (hw->compute_major >= 8) {
        int wmma_candidates_m[] = {32, 48, 64, 80, 96, 128};
        int wmma_candidates_k[] = {16, 32, 48, 64};
        int wmma_candidates_w[] = {2, 4, 8};
        int nm = sizeof(wmma_candidates_m) / sizeof(int);
        int nk = sizeof(wmma_candidates_k) / sizeof(int);
        int nw = sizeof(wmma_candidates_w) / sizeof(int);

        for (int im = 0; im < nm; im++) {
            for (int in = 0; in < nm; in++) {
                int bm = wmma_candidates_m[im];
                int bn = wmma_candidates_m[in];
                int tiles_m = bm / 16;
                int tiles_n = bn / 16;
                int total_tiles = tiles_m * tiles_n;

                for (int iw = 0; iw < nw; iw++) {
                    int nwarps = wmma_candidates_w[iw];
                    if (total_tiles % nwarps != 0) continue;
                    int threads = nwarps * hw->warp_size;
                    if (threads > hw->max_threads_per_block) continue;

                    // warp layout: try to make it as square as possible
                    int layout_m = 0;
                    for (int lm = nwarps; lm >= 1; lm--) {
                        if (nwarps % lm != 0) continue;
                        int ln = nwarps / lm;
                        if (tiles_m % lm != 0) continue;
                        if (tiles_n % ln != 0) continue;
                        // prefer squarer layout
                        if (layout_m == 0 || (lm - ln) * (lm - ln) < (layout_m - (nwarps / layout_m)) * (layout_m - (nwarps / layout_m)))
                            layout_m = lm;
                    }
                    if (layout_m == 0) continue;

                    int wt_m = tiles_m / layout_m;
                    int wt_n = tiles_n / (nwarps / layout_m);
                    // accumulator registers per thread: wt_m * wt_n * 8 (FP32 fragment = 8 regs)
                    int acc_regs = wt_m * wt_n * 8;
                    int est_regs = acc_regs + 32; // overhead for a/b frags + loop vars
                    if (est_regs * threads > hw->regs_per_block) continue;

                    int sm_capacity_t = hw->max_threads_per_sm / threads;

                    for (int ik = nk - 1; ik >= 0; ik--) {
                        int bk = wmma_candidates_k[ik];
                        // shared memory: BF16 with +8 padding
                        size_t smem = (size_t)(bm * (bk + 8) + bk * (bn + 8)) * 2;
                        if (smem > (size_t)hw->shared_per_block) continue;

                        int sm_by_smem = (int)(hw->shared_per_sm / smem);
                        int sm_cap = sm_capacity_t < sm_by_smem ? sm_capacity_t : sm_by_smem;

                        int total_util = 0;
                        for (int i = 0; i < nops; i++) {
                            int blocks = ((ops[i].M + bm - 1) / bm) * ((ops[i].N + bn - 1) / bn);
                            int bpsm = blocks / hw->sm_count;
                            if (bpsm > sm_cap) bpsm = sm_cap;
                            int active = bpsm * threads;
                            int util = (100 * active) / hw->max_threads_per_sm;
                            total_util += util * ops[i].count;
                        }

                        if (total_util > wmma_best_util) {
                            wmma_best_util = total_util;
                            wbm = bm; wbn = bn; wbk = bk;
                            wnw = nwarps; wlm = layout_m;
                        }
                        break; // largest K that fits
                    }
                }
            }
        }

        if (wbm == 0) { wbm = 64; wbn = 64; wbk = 32; wnw = 4; wlm = 2; }

        int wthreads = wnw * hw->warp_size;
        size_t wsmem = (size_t)(wbm * (wbk + 8) + wbk * (wbn + 8)) * 2;
        int wbpsm_s = (int)(hw->shared_per_sm / wsmem);
        int wbpsm_t = hw->max_threads_per_sm / wthreads;
        int wbpsm = wbpsm_s < wbpsm_t ? wbpsm_s : wbpsm_t;
        if (wbpsm < 1) wbpsm = 1;
        int wocc = (100 * wbpsm * wthreads) / hw->max_threads_per_sm;
        int wt_m = (wbm / 16) / wlm;
        int wt_n = (wbn / 16) / (wnw / wlm);

        fprintf(stderr, "\n  wmma\n");
        fprintf(stderr, "  block   %dx%d K=%d\n", wbm, wbn, wbk);
        fprintf(stderr, "  warps   %d (%dx%d)\n", wnw, wlm, wnw / wlm);
        fprintf(stderr, "  warp_t  %dx%d (16x16 each)\n", wt_m, wt_n);
        fprintf(stderr, "  threads %d\n", wthreads);
        fprintf(stderr, "  shared  %zu\n", wsmem);
        fprintf(stderr, "  blk/sm  %d\n", wbpsm);
        fprintf(stderr, "  occup   %d%%\n", wocc);
    }

    // ── Flash Attention tile search ──
    // Grid: h × ceil(seq/BR) blocks. Inner loop: ceil(seq/BC) iterations per block.
    // Per iteration: load BC×hd K + BC×hd V, compute BR×BC scores.
    // Per block: load BR×hd Q once + ceil(seq/BC) × 2×BC×hd K/V loads.
    // Shared mem (bwd packed): (3*BR+2*BC)*hd + BR*BC + 3*BR + nwarps floats.

    int fa_br = 32, fa_bc = 32;

    if (hd > 0) {
        int fa_cand[] = {16, 32, 48, 64, 96, 128};
        int nfc = sizeof(fa_cand) / sizeof(int);

        int attn_block = hw->max_threads_per_sm / 2;
        attn_block = (attn_block / hw->warp_size) * hw->warp_size;
        if (attn_block > hw->max_threads_per_block) attn_block = hw->max_threads_per_block;
        int nwarps = attn_block / hw->warp_size;

        size_t smem_limit = hw->shared_per_block_optin > 0 ?
                            hw->shared_per_block_optin : hw->shared_per_block;

        long long best_score = 0;

        for (int ibr = 0; ibr < nfc; ibr++) {
            for (int ibc = 0; ibc < nfc; ibc++) {
                int br = fa_cand[ibr];
                int bc = fa_cand[ibc];

                size_t smem = (size_t)((3 * br + 2 * bc) * hd
                              + br * bc + 3 * br + nwarps) * sizeof(float);
                if (smem > smem_limit) continue;

                int sm_cap_t = hw->max_threads_per_sm / attn_block;
                int sm_cap_s = (int)(hw->shared_per_sm / smem);
                int sm_cap = sm_cap_t < sm_cap_s ? sm_cap_t : sm_cap_s;
                if (sm_cap < 1) continue;

                int q_tiles = (seq + br - 1) / br;
                int blocks = h * q_tiles;
                int bpsm = blocks / hw->sm_count;
                if (bpsm > sm_cap) bpsm = sm_cap;

                // occupancy: active threads / max threads per SM
                int occ = (100 * bpsm * attn_block) / hw->max_threads_per_sm;

                // arithmetic intensity: compute per byte loaded per inner iteration
                // compute: BR × BC × hd FMAs (score dot products)
                // bytes loaded: 2 × BC × hd × 4 (K + V tiles)
                // intensity = (BR × BC × hd) / (2 × BC × hd × 4) = BR / 8
                // but BC determines iterations: ceil(seq/BC) per block
                // total compute per block: ceil(seq/BC) × BR × BC × hd = ~seq × BR × hd
                // total bytes per block: ceil(seq/BC) × 2 × BC × hd × 4 = ~seq × hd × 8
                // per-block intensity: (seq × BR × hd) / (seq × hd × 8) = BR / 8
                //
                // BUT: fewer iterations (larger BC) = fewer syncs, fewer softmax reductions
                // each iteration has fixed overhead independent of BC
                // so fewer iterations = less overhead
                // iterations = ceil(seq/BC)
                //
                // score = occupancy × BR × BC 
                // occupancy gates SM utilization
                // BR × BC = tile area = work per iteration = amortizes per-iteration overhead

                long long score = (long long)occ * br * bc;

                if (score > best_score) {
                    best_score = score;
                    fa_br = br;
                    fa_bc = bc;
                }
            }
        }

        size_t fa_smem = (size_t)((3 * fa_br + 2 * fa_bc) * hd
                         + fa_br * fa_bc + 3 * fa_br + nwarps) * sizeof(float);
        int fa_cap_s = (int)(hw->shared_per_sm / fa_smem);
        int fa_cap_t = hw->max_threads_per_sm / attn_block;
        int fa_bpsm = fa_cap_s < fa_cap_t ? fa_cap_s : fa_cap_t;
        if (fa_bpsm < 1) fa_bpsm = 1;
        int fa_occ = (100 * fa_bpsm * attn_block) / hw->max_threads_per_sm;
        int fa_q_tiles = (seq + fa_br - 1) / fa_br;

        fprintf(stderr, "\n  flash_attn\n");
        fprintf(stderr, "  FA_BR   %d\n", fa_br);
        fprintf(stderr, "  FA_BC   %d\n", fa_bc);
        fprintf(stderr, "  hd      %d\n", hd);
        fprintf(stderr, "  shared  %zu\n", fa_smem);
        fprintf(stderr, "  blk/sm  %d\n", fa_bpsm);
        fprintf(stderr, "  occup   %d%%\n", fa_occ);
        fprintf(stderr, "  grid    %d (%d heads × %d tiles)\n", h * fa_q_tiles, h, fa_q_tiles);
        fprintf(stderr, "  iters   %d per block\n", (seq + fa_bc - 1) / fa_bc);
    }

    // ── AdamW block size search ──
    // AdamW is elementwise (no shared mem, no tiling) — occupancy is purely warp-limited.
    // Find block size where warps_per_block divides evenly into max_warps_per_sm.
    // Target: 100% occupancy = all warp slots filled.
    int max_warps_sm = hw->max_threads_per_sm / hw->warp_size;
    int adamw_block = hw->max_threads_per_block;  // default: max (often 1024)

    // Search from max down — largest block with 100% occupancy wins (fewer blocks = less scheduling overhead)
    for (int threads = hw->max_threads_per_block; threads >= hw->warp_size; threads -= hw->warp_size) {
        int warps = threads / hw->warp_size;
        int concurrent = max_warps_sm / warps;         // blocks that fit per SM
        int used_warps = concurrent * warps;            // warp slots actually used
        if (used_warps == max_warps_sm) {
            adamw_block = threads;
            break;
        }
    }

    int aw_warps = adamw_block / hw->warp_size;
    int aw_concurrent = max_warps_sm / aw_warps;
    int aw_occ = (100 * aw_concurrent * aw_warps) / max_warps_sm;

    fprintf(stderr, "\n  adamw\n");
    fprintf(stderr, "  block   %d\n", adamw_block);
    fprintf(stderr, "  warps   %d\n", aw_warps);
    fprintf(stderr, "  blk/sm  %d\n", aw_concurrent);
    fprintf(stderr, "  occup   %d%%\n", aw_occ);

    printf("-DMATMUL_TILE=%d -DMATMUL_BK=%d -DMATMUL_REG=%d", best_tile, best_bk, best_reg);
    printf(" -DFA_BR=%d -DFA_BC=%d", fa_br, fa_bc);
    printf(" -DGPU_WARP_SIZE=%d", hw->warp_size);
    printf(" -DADAMW_BLOCK=%d", adamw_block);
    if (wbm > 0) {
        printf(" -DWMMA_BLOCK_M=%d -DWMMA_BLOCK_N=%d -DWMMA_BLOCK_K=%d -DWMMA_NUM_WARPS=%d -DWARP_LAYOUT_M=%d",
               wbm, wbn, wbk, wnw, wlm);
    }
}

static int run_optimal(const char* dir, int vocab) {
    gpu_hw_t hw;
    if (query_gpu(&hw) != 0) return 1;
    print_gpu(&hw);

    return 0;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--defines") == 0) {
            gpu_hw_t hw;
            if (query_gpu(&hw) != 0) return 1;
            const char* model_dir = NULL;
            if (i + 1 < argc && argv[i+1][0] != '-') model_dir = argv[++i];
            if (!model_dir) {
                fprintf(stderr, "usage: %s --defines <model_dir>\n", argv[0]);
                return 1;
            }
            print_gpu(&hw);
            print_defines(&hw, model_dir);
            return 0;
        }
        if (strcmp(argv[i], "--optimal") == 0) {
            const char* dir = NULL;
            int vocab = 12061;
            if (i + 1 < argc && argv[i+1][0] != '-') dir = argv[++i];
            for (int j = i+1; j < argc; j++)
                if (strcmp(argv[j], "--vocab") == 0 && j+1 < argc)
                    { vocab = atoi(argv[j+1]); break; }
            if (!dir) {
                fprintf(stderr, "usage: %s --optimal <dir> [--vocab N]\n", argv[0]);
                return 1;
            }
            return run_optimal(dir, vocab);
        }
    }

    gpu_hw_t hw;
    if (query_gpu(&hw) != 0) return 1;
    print_gpu(&hw);
    return 0;
}
