/*
 * parser.js — Parse gpu-profile-query output and profiler step dumps
 * Streaming parser: feed lines one at a time, get structured data back
 */

/* ═══════════════════════════════════════════════════════════════════════════ */
/* GPU INFO PARSER                                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */

function parseGPUInfo(text) {
    const info = { defines: {} };
    const lines = text.split('\n');

    /* GPU name — first line that looks like a GPU name */
    for (const line of lines) {
        const t = line.trim();
        if (t && /NVIDIA|GeForce|PowerVR|Tesla|Quadro|A100|H100|RTX|GTX/i.test(t)) {
            info.name = t;
            break;
        }
    }

    /* Key-value pairs: "  key_name    value" (integers or decimals) */
    const kvRe = /^\s{2}(\w[\w/]*)\s+([\d.]+)/;
    for (const line of lines) {
        const m = line.match(kvRe);
        if (m) {
            const key = m[1].replace(/\//g, '_');
            const val = m[2];
            if (key === 'compute' && val.includes('.')) {
                const parts = val.split('.');
                info.compute_major = parseInt(parts[0]);
                info.compute_minor = parseInt(parts[1]);
            }
            info[key] = val.includes('.') ? parseFloat(val) : parseInt(val);
        }
    }

    /* Model params: "model z=540 inter=2175 vocab=12061 L=4 h=6 kv_h=6 hd=90 seq=2048" */
    const modelRe = /model z=(\d+) inter=(\d+) vocab=(\d+) L=(\d+) h=(\d+) kv_h=(\d+) hd=(\d+) seq=(\d+)/;
    const mm = text.match(modelRe);
    if (mm) {
        info.model = {
            z: +mm[1], inter: +mm[2], vocab: +mm[3], L: +mm[4],
            h: +mm[5], kv_h: +mm[6], hd: +mm[7], seq: +mm[8]
        };
    }

    /* Compiler defines: "-DMATMUL_TILE=64 -DFA_BR=48 ..." */
    const defRe = /-D(\w+)=(\d+)/g;
    let dm;
    while ((dm = defRe.exec(text))) {
        info.defines[dm[1]] = parseInt(dm[2]);
    }

    /* Flash attention section */
    const faRe = /flash_attn[\s\S]*?FA_BR\s+(\d+)[\s\S]*?FA_BC\s+(\d+)[\s\S]*?shared\s+(\d+)[\s\S]*?blk\/sm\s+(\d+)[\s\S]*?occup\s+(\d+)%[\s\S]*?grid\s+(\d+)/;
    const fa = text.match(faRe);
    if (fa) {
        info.flash_attn = {
            BR: +fa[1], BC: +fa[2], shared: +fa[3],
            blk_per_sm: +fa[4], occupancy: +fa[5], grid: +fa[6]
        };
    }

    /* WMMA section */
    const wmmaRe = /wmma[\s\S]*?block\s+(\d+)x(\d+)\s+K=(\d+)[\s\S]*?warps\s+(\d+)[\s\S]*?shared\s+(\d+)[\s\S]*?blk\/sm\s+(\d+)[\s\S]*?occup\s+(\d+)%/;
    const wm = text.match(wmmaRe);
    if (wm) {
        info.wmma = {
            block_m: +wm[1], block_n: +wm[2], block_k: +wm[3],
            warps: +wm[4], shared: +wm[5], blk_per_sm: +wm[6], occupancy: +wm[7]
        };
    }

    /* Matmul tile section */
    const tileRe = /tile\s+(\d+)[\s\S]*?bk\s+(\d+)[\s\S]*?reg\s+(\d+)[\s\S]*?threads\s+(\d+)[\s\S]*?shared\s+(\d+)[\s\S]*?blk\/sm\s+(\d+)[\s\S]*?occup\s+(\d+)%/;
    const ti = text.match(tileRe);
    if (ti) {
        info.matmul = {
            tile: +ti[1], bk: +ti[2], reg: +ti[3],
            threads: +ti[4], shared: +ti[5], blk_per_sm: +ti[6], occupancy: +ti[7]
        };
    }

    return info;
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* OP CATEGORY DETECTION                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

function categorizeOp(name) {
    if (/matmul|Matmul/i.test(name))                     return 'matmul';
    if (/attn_bwd|attention_bwd|attn_backward/i.test(name)) return 'attention_bwd';
    if (/attn|attention|flash/i.test(name))               return 'attention';
    if (/rmsnorm_bwd|rms_bwd/i.test(name))               return 'rmsnorm_bwd';
    if (/rmsnorm|rms_norm/i.test(name))                   return 'rmsnorm';
    if (/cross_entropy|xent/i.test(name))                 return 'cross_entropy';
    if (/adamw|adam_w|optimizer/i.test(name))              return 'adamw';
    if (/copy|memset|memcpy|zero/i.test(name))            return 'copy';
    if (/embed|rope|silu|residual|add/i.test(name))       return 'elementwise';
    return 'other';
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* PROFILER OUTPUT STREAMING PARSER                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

class ProfilerParser {
    constructor() {
        this.state = 'idle';       /* idle | header | categories | ops | metrics */
        this.currentStep = null;
        this.stepCount = 0;
    }

    /**
     * Feed a single line from training output.
     * Returns null (nothing to emit) or an object:
     *   { type: 'step', data: { total_ms, num_ops, categories: [], ops: [] } }
     *   { type: 'status', text: '...' }
     *   { type: 'training', step, loss, lr, toks, time_ms }
     */
    parseLine(line) {
        /* ── Step header ── */
        const stepRe = /SERAPH STEP PROFILER \((\d+) operations?, ([\d.]+) ms total\)/;
        const sm = line.match(stepRe);
        if (sm) {
            const prev = this.currentStep;
            this.currentStep = {
                num_ops: parseInt(sm[1]),
                total_ms: parseFloat(sm[2]),
                categories: [],
                ops: []
            };
            this.state = 'header';
            this.stepCount++;
            return prev ? { type: 'step', data: prev } : null;
        }

        /* ── Section headers ── */
        if (/CATEGORY BREAKDOWN/.test(line))     { this.state = 'categories'; return null; }
        if (/PER-OPERATION DETAIL/.test(line))   { this.state = 'ops';        return null; }
        if (/PERFORMANCE METRICS/.test(line))    { this.state = 'metrics';    return null; }
        if (/TOP \d+ SLOWEST/.test(line))        { this.state = 'idle';       return null; }
        if (/Profiling complete/.test(line))     { this.state = 'idle';       return null; }

        /* ── Separator / header lines ── */
        if (/^\s*[═─]{4,}/.test(line)) return null;
        if (/^\s*#\s+Operation/.test(line)) return null;
        if (/^\s*Category\s/.test(line)) return null;
        if (/^\s*Rank\s+Operation/.test(line)) return null;
        if (line.trim() === '') return null;

        /* ── Training status lines (outside profiler) ── */
        /* "  Step 100/33500 | Loss: 2.3456 | Avg: 2.5000 | 5780 tok/s" */
        const trainRe = /Step\s+(\d+)\/(\d+)\s*\|\s*Loss:\s*([\d.]+)/;
        const tm = line.match(trainRe);
        if (tm) {
            const result = {
                type: 'training',
                step: parseInt(tm[1]),
                total_steps: parseInt(tm[2]),
                loss: parseFloat(tm[3])
            };
            const avgM = line.match(/Avg:\s*([\d.]+)/);
            if (avgM) result.avg_loss = parseFloat(avgM[1]);
            const tokM = line.match(/([\d.]+)\s*tok\/s/i);
            if (tokM) result.toks = parseFloat(tokM[1]);
            return result;
        }

        /* ── Category line ── */
        if (this.state === 'categories' && this.currentStep) {
            /* "  matmul (all)                  23.456  51.9%  ########" */
            const cm = line.match(/^\s{2}(\S.{20,}?)\s{2,}([\d.]+)\s+([\d.]+)%/);
            if (cm) {
                this.currentStep.categories.push({
                    name: cm[1].trim(),
                    ms: parseFloat(cm[2]),
                    pct: parseFloat(cm[3])
                });
                return null;
            }
        }

        /* ── Per-op line ── */
        if (this.state === 'ops' && this.currentStep) {
            /* "     0  L0 fwd matmul_Bt Q proj [seq=214]       0.123  0.3%  (4,1)/(256,1)  <<<" */
            const om = line.match(/^\s+(\d+)\s{2}(.+?)\s{2,}([\d.]+)\s+([\d.]+)%\s*(.*)/);
            if (om) {
                const op = {
                    index: parseInt(om[1]),
                    name: om[2].trim(),
                    ms: parseFloat(om[3]),
                    pct: parseFloat(om[4]),
                    launch: om[5].replace(/\s*<+\s*$/, '').trim(),
                    category: categorizeOp(om[2]),
                    grid: null,
                    block: null,
                    total_blocks: 0,
                    threads_per_block: 0
                };

                /* Parse grid/block: "(4,1)/(256,1)" or "(4,1,1)/(256,1,1)" or "4/256" */
                const lb = op.launch.match(
                    /\(?(\d+)(?:,(\d+))?(?:,(\d+))?\)?\s*\/\s*\(?(\d+)(?:,(\d+))?(?:,(\d+))?\)?/
                );
                if (lb) {
                    op.grid  = { x: +lb[1], y: +(lb[2]||1), z: +(lb[3]||1) };
                    op.block = { x: +lb[4], y: +(lb[5]||1), z: +(lb[6]||1) };
                    op.total_blocks = op.grid.x * op.grid.y * op.grid.z;
                    op.threads_per_block = op.block.x * op.block.y * op.block.z;
                }

                this.currentStep.ops.push(op);
                return null;
            }
        }

        /* ── Metrics line (GFLOPS / GB/s) ── */
        if (this.state === 'metrics' && this.currentStep) {
            const mm = line.match(/^\s+(\d+)\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/);
            if (mm) {
                const idx = parseInt(mm[1]);
                const op = this.currentStep.ops.find(o => o.index === idx);
                if (op) {
                    op.gflops = parseFloat(mm[3]);
                    op.gbps = parseFloat(mm[4]);
                }
                return null;
            }
        }

        /* ── Generic status ── */
        if (this.state === 'idle' || !this.currentStep) {
            return { type: 'status', text: line };
        }

        return null;
    }

    /** Flush any remaining step data */
    flush() {
        if (this.currentStep && this.currentStep.ops.length > 0) {
            const step = this.currentStep;
            this.currentStep = null;
            return { type: 'step', data: step };
        }
        return null;
    }
}
