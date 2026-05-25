/*
 * renderer.js — Canvas-based GPU grid and timeline renderer
 * Uses d3-zoom for pan/zoom on both canvases
 */

const OP_COLORS = {
    matmul:        '#4a9eff',
    attention:     '#4aff9e',
    attention_bwd: '#2abead',
    rmsnorm:       '#ffa94a',
    rmsnorm_bwd:   '#e08a30',
    cross_entropy: '#ff4a6a',
    adamw:         '#b44aff',
    copy:          '#555555',
    elementwise:   '#7a8899',
    other:         '#444444'
};

const OP_LABELS = {
    matmul: 'matmul', attention: 'attn fwd', attention_bwd: 'attn bwd',
    rmsnorm: 'rms fwd', rmsnorm_bwd: 'rms bwd', cross_entropy: 'xent',
    adamw: 'adamw', copy: 'copy', elementwise: 'elem', other: 'other'
};


/* ═══════════════════════════════════════════════════════════════════════════ */
/* GPU RENDERER                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

class GPURenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');
        this.gpu    = null;
        this.step   = null;
        this.selectedOp = -1;
        this.transform  = d3.zoomIdentity;
        this.gridOffX = 30;
        this.gridOffY = 30;

        this.zoom = d3.zoom()
            .scaleExtent([0.05, 30])
            .on('zoom', (e) => {
                this.transform = e.transform;
                document.getElementById('zoom-level').textContent = e.transform.k.toFixed(1) + 'x';
                this.render();
            });
        d3.select(canvas)
            .call(this.zoom)
            .on('dblclick.zoom', null);

        canvas.addEventListener('click', (e) => this._onClick(e));
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const c = this.canvas.parentElement;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width  = c.clientWidth  * dpr;
        this.canvas.height = c.clientHeight * dpr;
        this.canvas.style.width  = c.clientWidth  + 'px';
        this.canvas.style.height = c.clientHeight + 'px';
        this.dpr = dpr;
        this.render();
    }

    setGPU(info) {
        this.gpu = info;
        this.render();
    }

    setStep(step) {
        this.step = step;
        this.render();
    }

    selectOp(index) {
        this.selectedOp = index;
        this.transform = d3.zoomIdentity;
        d3.select(this.canvas).call(this.zoom.transform, d3.zoomIdentity);
        this.render();
    }

    render() {
        const ctx = this.ctx;
        const dpr = this.dpr;
        const w = this.canvas.width / dpr;
        const h = this.canvas.height / dpr;

        ctx.save();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        ctx.fillStyle = '#111820';
        for (let x = 0; x < w; x += 20)
            for (let y = 0; y < h; y += 20)
                ctx.fillRect(x, y, 1, 1);

        ctx.translate(this.transform.x, this.transform.y);
        ctx.scale(this.transform.k, this.transform.k);

        if (!this.gpu) {
            ctx.fillStyle = '#333';
            ctx.font = '14px monospace';
            ctx.fillText('Load GPU to begin', 30, 50);
            ctx.restore();
            return;
        }

        const op = (this.selectedOp >= 0 && this.step)
            ? this.step.ops[this.selectedOp] : null;

        if (op && op.grid && op.block) {
            const dims = this._parseMatmulDims(op.name);
            const attnDims = this._parseAttnDims(op.name);
            const rmsDims = this._parseRmsNormDims(op.name);
            const xentDims = this._parseCrossEntropyDims(op.name);
            const adamDims = this._parseAdamWDims(op.name);

            /* op name — always top left */
            const displayName = op.name.replace(/(\d+)-thr/, `$1/${this.gpu.threads_block} threads`);
            ctx.fillStyle = '#4a9eff';
            ctx.font = 'bold 11px monospace';
            ctx.fillText(displayName, 20, 16);

            /* LEFT: model view */
            const modelOx = 20, modelOy = 30;
            let modelW = 0;

            if (dims) {
                this._drawMatrix(op, dims);
                modelW = dims.N + 40;
            } else if (attnDims) {
                this._drawAttnModel(op, attnDims, modelOx, modelOy);
                modelW = this._attnModelWidth(attnDims) + 20;
            } else if (rmsDims) {
                this._drawRmsNorm(op, rmsDims);
                modelW = rmsDims.hidden + 20;
            } else if (xentDims) {
                this._drawCrossEntropy(op, xentDims);
                modelW = xentDims.vocab + 20;
            } else if (adamDims) {
                this._drawAdamW(op, adamDims);
                modelW = (adamDims.cols > 0 ? adamDims.cols : 8) + 30;
            } else {
                this._drawOpGrid(op, modelOx, modelOy);
                modelW = this._opGridWidth(op) + 20;
            }

            /* RIGHT: SM info + warp grid */
            const smOx = modelOx + modelW + 20;
            this._drawSmHeader(op, smOx, 16);
            this._drawSMs(op, smOx, 44);
        } else if (op) {
            /* op without grid/block */
            ctx.fillStyle = '#4a9eff';
            ctx.font = 'bold 11px monospace';
            ctx.fillText(op.name + `  ${op.ms.toFixed(3)}ms (${op.pct.toFixed(1)}%)`, 20, 20);
        } else {
            this._drawIdleSMs();
        }

        ctx.restore();
    }

    /* ── SM info line (above SM grid) ── */
    _drawSmHeader(op, ox, oy) {
        const ctx = this.ctx;
        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        ctx.fillText(
            `grid(${op.grid.x},${op.grid.y},${op.grid.z})  block(${op.block.x},${op.block.y},${op.block.z})  ` +
            `${op.total_blocks} blocks  ${Math.ceil(op.threads_per_block / this.gpu.warp_size)} warps/block  ` +
            `${Math.floor((this.gpu.threads_sm / this.gpu.warp_size) / Math.ceil(op.threads_per_block / this.gpu.warp_size))} blocks/SM  ` +
            `${op.threads_per_block}/${this.gpu.threads_sm} threads  ` +
            `${Math.ceil(op.total_blocks / (Math.floor((this.gpu.threads_sm / this.gpu.warp_size) / Math.ceil(op.threads_per_block / this.gpu.warp_size)) * this.gpu.sm_count))} waves  ` +
            `${op.ms.toFixed(3)}ms (${op.pct.toFixed(1)}%)`,
            ox, oy + 12
        );
    }

    /* ── Attention dimension parser ── */
    _parseAttnDims(name) {
        if (!/attn|attention|flash/i.test(name)) return null;
        const m = name.match(/seq=(\d+)\s+h=(\d+)\s+hd=(\d+)/);
        if (!m) return null;
        return { seq: +m[1], h: +m[2], hd: +m[3] };
    }

    /* ── Attention model view: actual seq×seq score matrix per head ── */
    _drawAttnModel(op, attn, ox, oy) {
        const ctx = this.ctx;
        const { seq, h, hd } = attn;
        const fa_br = this.gpu.defines?.FA_BR || (op.grid.y > 0 ? Math.ceil(seq / op.grid.y) : seq);
        const fa_bc = this.gpu.defines?.FA_BC || fa_br;
        const qTiles = Math.ceil(seq / fa_br);
        const kvIters = Math.ceil(seq / fa_bc);
        const smCount = this.gpu.sm_count;

        /* draw seq×seq attention score matrix with BR×BC tiling */
        /* each element = 1 unit, same as matmul. zoom handles scale. */
        const cellSize = 1;
        const matW = seq * cellSize;
        const matH = seq * cellSize;

        /* label */
        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        ctx.fillText(
            `${seq}x${seq} scores  ${h} heads  hd=${hd}  BR=${fa_br}  BC=${fa_bc}  ${qTiles}x${kvIters} tiles`,
            ox, oy - 6
        );

        /* draw tiles: BR rows × BC cols, colored by SM assignment */
        for (let qr = 0; qr < qTiles; qr++) {
            for (let kc = 0; kc < kvIters; kc++) {
                /* flash attn: grid.x = heads, grid.y = q_tiles. inner loop = kv iters.
                   but for visualization, show the full score matrix tiling */
                const blockIdx = qr * kvIters + kc;
                const smIdx = blockIdx % smCount;
                const hue = (smIdx / smCount) * 300;

                const x = ox + kc * fa_bc * cellSize;
                const y = oy + qr * fa_br * cellSize;
                const w = Math.min(fa_bc, seq - kc * fa_bc) * cellSize;
                const tw = Math.min(fa_br, seq - qr * fa_br) * cellSize;

                ctx.fillStyle = `hsl(${hue}, 60%, 25%)`;
                ctx.fillRect(x, y, w, tw);

                ctx.strokeStyle = '#2a3a4a';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(x, y, w, tw);
            }
        }

        /* matrix outline */
        ctx.strokeStyle = '#4aff9e';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, matW, matH);

        /* causal mask triangle (lower triangular) */
        ctx.strokeStyle = '#4aff9e44';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ox + matW, oy + matH);
        ctx.stroke();

        /* axis labels */
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.save();
        ctx.translate(ox - 8, oy + matH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Q seq=' + seq, 0, 0);
        ctx.restore();
        ctx.textAlign = 'center';
        ctx.fillText('K seq=' + seq, ox + matW / 2, oy + matH + 14);
        ctx.textAlign = 'left';

        /* heads bar: h heads × hd elements each, every element = 1 cell */
        const headOx = ox + matW + 10;
        const headW = 8;
        const headCellH = cellSize;

        for (let i = 0; i < h; i++) {
            const smIdx = i % smCount;
            const hue = (smIdx / smCount) * 300;
            const headY = oy + i * hd * headCellH;

            for (let d = 0; d < hd; d++) {
                ctx.fillStyle = `hsl(${hue}, 60%, 25%)`;
                ctx.fillRect(headOx, headY + d * headCellH, headW, headCellH);
                ctx.strokeStyle = '#2a3a4a';
                ctx.lineWidth = 0.3;
                ctx.strokeRect(headOx, headY + d * headCellH, headW, headCellH);
            }

            /* head border */
            ctx.strokeStyle = '#4aff9e44';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(headOx, headY, headW, hd * headCellH);
        }

        /* outer border */
        ctx.strokeStyle = '#4aff9e';
        ctx.lineWidth = 1;
        ctx.strokeRect(headOx, oy, headW, h * hd * headCellH);

        ctx.fillStyle = '#4aff9e';
        ctx.font = '8px monospace';
        ctx.save();
        ctx.translate(headOx + headW + 10, oy + h * hd * headCellH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText(`${h}x${hd}`, 0, 0);
        ctx.restore();

        /* zoom-in element grid */
        if (this.transform.k > 3) {
            ctx.strokeStyle = '#1a222e';
            ctx.lineWidth = 0.2;
            ctx.beginPath();
            const zk = this.transform.k;
            const vl = -this.transform.x / zk, vr = vl + (this.canvas.width / this.dpr) / zk;
            const vt = -this.transform.y / zk, vb = vt + (this.canvas.height / this.dpr) / zk;
            const c0 = Math.max(0, Math.floor((vl - ox) / cellSize));
            const c1 = Math.min(seq, Math.ceil((vr - ox) / cellSize));
            const r0 = Math.max(0, Math.floor((vt - oy) / cellSize));
            const r1 = Math.min(seq, Math.ceil((vb - oy) / cellSize));
            for (let c = c0; c <= c1; c++) { ctx.moveTo(ox + c, oy + r0); ctx.lineTo(ox + c, oy + r1); }
            for (let r = r0; r <= r1; r++) { ctx.moveTo(ox + c0, oy + r); ctx.lineTo(ox + c1, oy + r); }
            ctx.stroke();
        }
    }

    _attnModelWidth(attn) {
        return attn.seq + 10 + 8 + 30;
    }

    /* ── RMSNorm dimension parser ── */
    _parseRmsNormDims(name) {
        if (!/rmsnorm|rms_norm/i.test(name)) return null;
        const m = name.match(/\[(\d+)x(\d+)\]/);
        if (!m) return null;
        return { seq: +m[1], hidden: +m[2] };
    }

    /* ── RMSNorm model view: seq × hidden matrix ── */
    _drawRmsNorm(op, dims) {
        const ctx = this.ctx;
        const { seq, hidden } = dims;
        const ox = 20, oy = 30;
        const cellSize = 1;

        /* each row = one position's hidden vector being normalized */
        const smCount = this.gpu.sm_count;

        /* one block per row (seq), so tile = 1 × hidden */
        for (let s = 0; s < seq; s++) {
            const smIdx = s % smCount;
            const hue = (smIdx / smCount) * 300;

            ctx.fillStyle = `hsl(${hue}, 60%, 25%)`;
            ctx.fillRect(ox, oy + s * cellSize, hidden * cellSize, cellSize);
            ctx.strokeStyle = '#2a3a4a';
            ctx.lineWidth = 0.3;
            ctx.strokeRect(ox, oy + s * cellSize, hidden * cellSize, cellSize);
        }

        ctx.strokeStyle = '#ffa94a';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, hidden * cellSize, seq * cellSize);

        /* axis labels */
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.save();
        ctx.translate(ox - 8, oy + seq * cellSize / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('seq=' + seq, 0, 0);
        ctx.restore();
        ctx.textAlign = 'center';
        ctx.fillText('hidden=' + hidden, ox + hidden * cellSize / 2, oy + seq * cellSize + 14);
        ctx.textAlign = 'left';

        /* zoom-in element grid */
        if (this.transform.k > 3) {
            ctx.strokeStyle = '#1a222e';
            ctx.lineWidth = 0.2;
            ctx.beginPath();
            const zk = this.transform.k;
            const vl = -this.transform.x / zk, vr = vl + (this.canvas.width / this.dpr) / zk;
            const vt = -this.transform.y / zk, vb = vt + (this.canvas.height / this.dpr) / zk;
            const c0 = Math.max(0, Math.floor((vl - ox) / cellSize));
            const c1 = Math.min(hidden, Math.ceil((vr - ox) / cellSize));
            const r0 = Math.max(0, Math.floor((vt - oy) / cellSize));
            const r1 = Math.min(seq, Math.ceil((vb - oy) / cellSize));
            for (let c = c0; c <= c1; c++) { ctx.moveTo(ox + c, oy + r0); ctx.lineTo(ox + c, oy + r1); }
            for (let r = r0; r <= r1; r++) { ctx.moveTo(ox + c0, oy + r); ctx.lineTo(ox + c1, oy + r); }
            ctx.stroke();
        }
    }

    /* ── Cross entropy parser + view: seq × vocab ── */
    _parseCrossEntropyDims(name) {
        if (!/cross_entropy|xent/i.test(name)) return null;
        const m = name.match(/\[(\d+)x(\d+)\]/);
        if (!m) return null;
        return { seq: +m[1], vocab: +m[2] };
    }

    _drawCrossEntropy(op, dims) {
        const ctx = this.ctx;
        const { seq, vocab } = dims;
        const ox = 20, oy = 30;
        const cellSize = 1;
        const smCount = this.gpu.sm_count;

        /* seq × vocab matrix — each row = one position's loss over vocab */
        for (let s = 0; s < seq; s++) {
            const smIdx = s % smCount;
            const hue = (smIdx / smCount) * 300;
            ctx.fillStyle = `hsl(${hue}, 60%, 25%)`;
            ctx.fillRect(ox, oy + s * cellSize, vocab * cellSize, cellSize);
            ctx.strokeStyle = '#2a3a4a';
            ctx.lineWidth = 0.3;
            ctx.strokeRect(ox, oy + s * cellSize, vocab * cellSize, cellSize);
        }

        ctx.strokeStyle = '#ff4a6a';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, vocab * cellSize, seq * cellSize);

        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.save();
        ctx.translate(ox - 8, oy + seq * cellSize / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('seq=' + seq, 0, 0);
        ctx.restore();
        ctx.textAlign = 'center';
        ctx.fillText('vocab=' + vocab, ox + vocab * cellSize / 2, oy + seq * cellSize + 14);
        ctx.textAlign = 'left';

        if (this.transform.k > 3) {
            ctx.strokeStyle = '#1a222e';
            ctx.lineWidth = 0.2;
            ctx.beginPath();
            const zk = this.transform.k;
            const vl = -this.transform.x / zk, vr = vl + (this.canvas.width / this.dpr) / zk;
            const vt = -this.transform.y / zk, vb = vt + (this.canvas.height / this.dpr) / zk;
            const c0 = Math.max(0, Math.floor((vl - ox) / cellSize));
            const c1 = Math.min(vocab, Math.ceil((vr - ox) / cellSize));
            const r0 = Math.max(0, Math.floor((vt - oy) / cellSize));
            const r1 = Math.min(seq, Math.ceil((vb - oy) / cellSize));
            for (let c = c0; c <= c1; c++) { ctx.moveTo(ox + c, oy + r0); ctx.lineTo(ox + c, oy + r1); }
            for (let r = r0; r <= r1; r++) { ctx.moveTo(ox + c0, oy + r); ctx.lineTo(ox + c1, oy + r); }
            ctx.stroke();
        }
    }

    /* ── AdamW parser + view: weight tensor with real shape ── */
    _parseAdamWDims(name) {
        if (!/adamw/i.test(name)) return null;
        /* "adamw w_embed [50304x768]" or "adamw rms_final [768]" */
        const m2d = name.match(/adamw\s+(\S+)\s+\[(\d+)x(\d+)\]/);
        if (m2d) return { tensor: m2d[1], rows: +m2d[2], cols: +m2d[3] };
        const m1d = name.match(/adamw\s+(\S+)\s+\[(\d+)\]/);
        if (m1d) return { tensor: m1d[1], rows: +m1d[2], cols: 0 };
        return null;
    }

    _drawAdamW(op, dims) {
        const ctx = this.ctx;
        const { tensor, rows, cols } = dims;
        const ox = 20, oy = 44;
        const cellSize = 1;
        const smCount = this.gpu.sm_count;
        const matW = cols > 0 ? cols : 8;
        const matH = rows;

        if (cols > 0) {
            /* 2D: tile each block as a horizontal strip of rows */
            const totalParams = rows * cols;
            const paramsPerBlock = Math.ceil(totalParams / op.total_blocks);
            const rowsPerTile = Math.max(1, Math.floor(paramsPerBlock / cols));
            const numTiles = Math.ceil(rows / rowsPerTile);

            for (let t = 0; t < numTiles; t++) {
                const smIdx = t % smCount;
                const hue = (smIdx / smCount) * 300;
                const r0 = t * rowsPerTile;
                const r1 = Math.min(r0 + rowsPerTile, rows);

                ctx.fillStyle = `hsl(${hue}, 60%, 25%)`;
                ctx.fillRect(ox, oy + r0 * cellSize, cols * cellSize, (r1 - r0) * cellSize);
                ctx.strokeStyle = '#2a3a4a';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(ox, oy + r0 * cellSize, cols * cellSize, (r1 - r0) * cellSize);
            }
        } else {
            /* 1D vector */
            for (let r = 0; r < rows; r++) {
                const smIdx = r % smCount;
                const hue = (smIdx / smCount) * 300;
                ctx.fillStyle = `hsl(${hue}, 60%, 25%)`;
                ctx.fillRect(ox, oy + r * cellSize, matW, cellSize);
                ctx.strokeStyle = '#2a3a4a';
                ctx.lineWidth = 0.3;
                ctx.strokeRect(ox, oy + r * cellSize, matW, cellSize);
            }
        }

        /* outline */
        ctx.strokeStyle = '#b44aff';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, matW * cellSize, matH * cellSize);

        /* axis labels — same style as matmul */
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.save();
        ctx.translate(ox - 8, oy + matH * cellSize / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText(rows + '', 0, 0);
        ctx.restore();

        if (cols > 0) {
            ctx.textAlign = 'center';
            ctx.fillText(cols + '', ox + cols * cellSize / 2, oy + matH * cellSize + 14);
            ctx.textAlign = 'left';
        }

        /* zoom-in element grid */
        if (this.transform.k > 3) {
            ctx.strokeStyle = '#1a222e';
            ctx.lineWidth = 0.2;
            ctx.beginPath();
            const zk = this.transform.k;
            const vl = -this.transform.x / zk, vr = vl + (this.canvas.width / this.dpr) / zk;
            const vt = -this.transform.y / zk, vb = vt + (this.canvas.height / this.dpr) / zk;
            const c0 = Math.max(0, Math.floor((vl - ox) / cellSize));
            const c1 = Math.min(matW, Math.ceil((vr - ox) / cellSize));
            const r0 = Math.max(0, Math.floor((vt - oy) / cellSize));
            const r1 = Math.min(matH, Math.ceil((vb - oy) / cellSize));
            for (let c = c0; c <= c1; c++) { ctx.moveTo(ox + c, oy + r0); ctx.lineTo(ox + c, oy + r1); }
            for (let r = r0; r <= r1; r++) { ctx.moveTo(ox + c0, oy + r); ctx.lineTo(ox + c1, oy + r); }
            ctx.stroke();
        }
    }

    /* ── Non-matmul op grid (grid.x × grid.y blocks) ── */
    _opGridCellSize(op) {
        const maxDim = Math.max(op.grid.x, op.grid.y, 1);
        if (maxDim > 1000) return 0.5;
        if (maxDim > 500) return 1;
        if (maxDim > 100) return 2;
        if (maxDim > 50) return 4;
        if (maxDim > 20) return 8;
        return 12;
    }

    _opGridWidth(op) {
        const cs = this._opGridCellSize(op);
        const gap = cs > 2 ? 1 : 0;
        return op.grid.x * (cs + gap);
    }

    _drawOpGrid(op, ox, oy) {
        const ctx = this.ctx;
        const gx = op.grid.x, gy = op.grid.y;
        const smCount = this.gpu.sm_count;
        const cs = this._opGridCellSize(op);
        const gap = cs > 2 ? 1 : 0;
        const step = cs + gap;

        /* each cell = one block, colored by SM assignment */
        for (let by = 0; by < gy; by++) {
            for (let bx = 0; bx < gx; bx++) {
                const blockIdx = by * gx + bx;
                const smIdx = blockIdx % smCount;
                const hue = (smIdx / smCount) * 300;
                ctx.fillStyle = `hsl(${hue}, 50%, 30%)`;
                ctx.fillRect(ox + bx * step, oy + by * step, cs, cs);
            }
        }

        /* outline */
        ctx.strokeStyle = '#3a4a5a';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(ox, oy, gx * step, gy * step);

        /* axis labels */
        ctx.fillStyle = '#555';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(gx + '', ox + gx * step / 2, oy + gy * step + 10);
        ctx.textAlign = 'left';
        ctx.save();
        ctx.translate(ox - 8, oy + gy * step / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText(gy + '', 0, 0);
        ctx.restore();
        ctx.textAlign = 'left';
    }

    /* ── Idle SM view (no op selected) ── */
    _drawIdleSMs() {
        const ctx = this.ctx;
        const gpu = this.gpu;
        const smCount = gpu.sm_count;
        const totalWarps = gpu.threads_sm / gpu.warp_size;

        const cell = 5, gap = 1, step = cell + gap;
        const warpCols = Math.ceil(Math.sqrt(totalWarps));
        const warpRows = Math.ceil(totalWarps / warpCols);
        const smPad = 4, smTop = 12;
        const sw = warpCols * step + smPad * 2;
        const sh = warpRows * step + smTop + smPad;
        const sp = 6;
        const gridCols = Math.ceil(Math.sqrt(smCount * 1.5));
        const ox = this.gridOffX, oy = this.gridOffY;

        ctx.fillStyle = '#4a9eff';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(`${smCount} SMs  ${totalWarps} warps/SM  ${gpu.threads_sm} threads/SM`, ox, oy - 8);

        for (let i = 0; i < smCount; i++) {
            const col = i % gridCols;
            const row = Math.floor(i / gridCols);
            const x = ox + col * (sw + sp);
            const y = oy + row * (sh + sp);

            ctx.fillStyle = '#0c1015';
            ctx.strokeStyle = '#151a22';
            ctx.lineWidth = 0.5;
            this._roundRect(x, y, sw, sh, 3);

            ctx.fillStyle = '#2a2a3a';
            ctx.font = 'bold 7px monospace';
            ctx.fillText('SM' + i, x + 3, y + 9);

            const ax = x + smPad, ay = y + smTop;
            for (let w = 0; w < totalWarps; w++) {
                ctx.fillStyle = '#1a1d22';
                ctx.fillRect(ax + (w % warpCols) * step, ay + Math.floor(w / warpCols) * step, cell, cell);
            }
        }
    }

    /* ── SM occupancy for a selected op ── */
    _drawSMs(op, ox, oy) {
        const ctx = this.ctx;
        const gpu = this.gpu;
        const smCount    = gpu.sm_count;
        const warpSize   = gpu.warp_size;
        const totalWarps = gpu.threads_sm / warpSize;
        const wpb        = Math.ceil(op.threads_per_block / warpSize);
        const maxConcurrent = Math.floor(totalWarps / wpb);

        const bpsm = new Array(smCount).fill(0);
        for (let b = 0; b < op.total_blocks; b++)
            bpsm[b % smCount]++;

        const cell = 5, gap = 1, step = cell + gap;
        const warpCols = Math.ceil(Math.sqrt(totalWarps));
        const warpRows = Math.ceil(totalWarps / warpCols);
        const smPad = 4, smTop = 12;
        const sw = warpCols * step + smPad * 2;
        const sh = warpRows * step + smTop + smPad;
        const sp = 6;
        const gridCols = Math.ceil(Math.sqrt(smCount * 1.5));

        const color = OP_COLORS[op.category] || '#4a9eff';

        for (let i = 0; i < smCount; i++) {
            const col = i % gridCols;
            const row = Math.floor(i / gridCols);
            const x = ox + col * (sw + sp);
            const y = oy + row * (sh + sp);

            const concurrent = Math.min(bpsm[i], maxConcurrent);
            const usedWarps = concurrent * wpb;
            const active = concurrent > 0;

            ctx.fillStyle = '#0c1015';
            ctx.strokeStyle = active ? '#2a3a4a' : '#151a22';
            ctx.lineWidth = 0.5;
            this._roundRect(x, y, sw, sh, 3);

            ctx.fillStyle = active ? '#667' : '#2a2a3a';
            ctx.font = 'bold 7px monospace';
            ctx.fillText('SM' + i, x + 3, y + 9);

            const ax = x + smPad, ay = y + smTop;

            /* warp slots */
            for (let w = 0; w < totalWarps; w++) {
                ctx.fillStyle = (w < usedWarps) ? color : '#1a1d22';
                ctx.fillRect(ax + (w % warpCols) * step, ay + Math.floor(w / warpCols) * step, cell, cell);
            }

            /* block separator lines */
            if (concurrent > 1) {
                ctx.strokeStyle = '#ffffff60';
                ctx.lineWidth = 1;
                for (let b = 1; b < concurrent; b++) {
                    const sepRow = Math.floor((b * wpb) / warpCols);
                    const ly = ay + sepRow * step - 1;
                    ctx.beginPath();
                    ctx.moveTo(ax, ly);
                    ctx.lineTo(ax + warpCols * step - gap, ly);
                    ctx.stroke();
                }
            }
        }
    }

    /* ── Matmul matrix grid ── */
    _parseMatmulDims(name) {
        if (!/matmul|Matmul/i.test(name)) return null;
        let M, N, K;
        const mnk = name.match(/M=(\d+).*?N=(\d+).*?K=(\d+)/);
        if (mnk) { M = +mnk[1]; N = +mnk[2]; K = +mnk[3]; }
        else {
            const alt = name.match(/\[(\d+)x(\d+)\s+K=(\d+)\]/);
            if (alt) { M = +alt[1]; N = +alt[2]; K = +alt[3]; }
        }
        if (!M || !N) return null;
        return { M, N, K: K || 0 };
    }

    _drawMatrix(op, dims) {
        const ctx = this.ctx;
        const { M, N, K } = dims;
        const ox = 20, oy = 44;
        const cellSize = 1;
        const matW = N * cellSize;
        const matH = M * cellSize;

        const tile = this.gpu.defines?.WMMA_BLOCK_M || this.gpu.defines?.MATMUL_TILE || 0;
        const tileN = this.gpu.defines?.WMMA_BLOCK_N || tile;
        const tilesM = tile > 0 ? Math.ceil(M / tile) : 1;
        const tilesN = tileN > 0 ? Math.ceil(N / tileN) : 1;
        const smCount = this.gpu.sm_count;

        for (let tm = 0; tm < tilesM; tm++) {
            for (let tn = 0; tn < tilesN; tn++) {
                const blockIdx = tm * tilesN + tn;
                const smIdx = blockIdx % smCount;
                const x = ox + tn * tileN * cellSize;
                const y = oy + tm * tile * cellSize;
                const w = Math.min(tileN, N - tn * tileN) * cellSize;
                const h = Math.min(tile, M - tm * tile) * cellSize;
                const hue = (smIdx / smCount) * 300;
                ctx.fillStyle = `hsl(${hue}, 60%, 25%)`;
                ctx.fillRect(x, y, w, h);
                ctx.strokeStyle = '#2a3a4a';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(x, y, w, h);
            }
        }

        ctx.strokeStyle = '#4a9eff';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, matW, matH);

        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.save();
        ctx.translate(ox - 8, oy + matH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('M=' + M, 0, 0);
        ctx.restore();
        ctx.textAlign = 'center';
        ctx.fillText('N=' + N, ox + matW / 2, oy + matH + 14);
        ctx.textAlign = 'left';

        /* zoom-in element grid */
        if (this.transform.k > 3) {
            ctx.strokeStyle = '#1a222e';
            ctx.lineWidth = 0.2;
            ctx.beginPath();
            const zk = this.transform.k;
            const vl = -this.transform.x / zk, vr = vl + (this.canvas.width / this.dpr) / zk;
            const vt = -this.transform.y / zk, vb = vt + (this.canvas.height / this.dpr) / zk;
            const c0 = Math.max(0, Math.floor((vl - ox) / cellSize));
            const c1 = Math.min(N, Math.ceil((vr - ox) / cellSize));
            const r0 = Math.max(0, Math.floor((vt - oy) / cellSize));
            const r1 = Math.min(M, Math.ceil((vb - oy) / cellSize));
            for (let c = c0; c <= c1; c++) { ctx.moveTo(ox + c, oy + r0); ctx.lineTo(ox + c, oy + r1); }
            for (let r = r0; r <= r1; r++) { ctx.moveTo(ox + c0, oy + r); ctx.lineTo(ox + c1, oy + r); }
            ctx.stroke();
        }

        /* K bar — every element has its own cell */
        const kOx = ox + matW + 10;
        const kW = 8;
        for (let k = 0; k < K; k++) {
            ctx.fillStyle = '#1a2a1a';
            ctx.fillRect(kOx, oy + k * cellSize, kW, cellSize);
            ctx.strokeStyle = '#2a3a2a';
            ctx.lineWidth = 0.3;
            ctx.strokeRect(kOx, oy + k * cellSize, kW, cellSize);
        }
        ctx.strokeStyle = '#4aff9e';
        ctx.lineWidth = 1;
        ctx.strokeRect(kOx, oy, kW, K * cellSize);
        ctx.fillStyle = '#4aff9e';
        ctx.font = '8px monospace';
        ctx.save();
        ctx.translate(kOx + kW + 10, oy + Math.min(K, M) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('K=' + K, 0, 0);
        ctx.restore();
    }

    _roundRect(x, y, w, h, r) {
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }

    _darken(hex, amt) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgb(${(r*(1-amt))|0},${(g*(1-amt))|0},${(b*(1-amt))|0})`;
    }

    _onClick(e) { }
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* TIMELINE RENDERER                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

class TimelineRenderer {
    constructor(canvas, onSelect) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');
        this.step   = null;
        this.selectedOp = -1;
        this.onSelect   = onSelect;
        this.transform  = d3.zoomIdentity;
        this.barY = 12;
        this.barH = 32;

        this.zoom = d3.zoom()
            .scaleExtent([0.5, 200])
            .on('zoom', (e) => { this.transform = e.transform; this.render(); });
        d3.select(canvas).call(this.zoom);

        canvas.addEventListener('click', (e) => this._onClick(e));
        canvas.addEventListener('mousemove', (e) => this._onHover(e));
        this.hoveredOp = -1;
        this._opRects = [];

        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const c = this.canvas.parentElement;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width  = c.clientWidth * dpr;
        this.canvas.height = c.clientHeight * dpr;
        this.canvas.style.width  = c.clientWidth + 'px';
        this.canvas.style.height = c.clientHeight + 'px';
        this.dpr = dpr;
        this.render();
    }

    setStep(step) {
        this.step = step;
        this.transform = d3.zoomIdentity;
        d3.select(this.canvas).call(this.zoom.transform, d3.zoomIdentity);
        this.render();
    }

    selectOp(index) { this.selectedOp = index; this.render(); }

    render() {
        const ctx = this.ctx;
        const dpr = this.dpr;
        const w = this.canvas.width / dpr;
        const h = this.canvas.height / dpr;

        ctx.save();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        if (!this.step || !this.step.ops.length) {
            ctx.fillStyle = '#333';
            ctx.font = '11px monospace';
            ctx.fillText('Waiting for profiler data...', 12, h / 2);
            ctx.restore();
            return;
        }

        const ops = this.step.ops;
        const totalMs = this.step.total_ms || 1;
        const margin = 12;
        const timeW = w - margin * 2;
        const tx = this.transform.x, sx = this.transform.k;
        this._opRects = [];

        let xAcc = 0;
        for (let i = 0; i < ops.length; i++) {
            const op = ops[i];
            const frac = op.ms / totalMs;
            const barW = frac * timeW;
            const drawX = margin + xAcc * sx + tx;
            const drawW = Math.max(0.5, barW * sx);

            if (drawX + drawW > 0 && drawX < w) {
                const selected = (i === this.selectedOp);
                const hovered  = (i === this.hoveredOp);
                ctx.fillStyle = selected ? '#ffffff'
                    : hovered ? this._lighten(OP_COLORS[op.category] || '#666', 0.3)
                    : (OP_COLORS[op.category] || '#444');
                ctx.fillRect(drawX, this.barY, drawW - (drawW > 2 ? 0.5 : 0), this.barH);

                if (drawW > 50) {
                    ctx.fillStyle = selected ? '#000' : '#000a';
                    ctx.font = '8px monospace';
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(drawX, this.barY, drawW, this.barH);
                    ctx.clip();
                    ctx.fillText(op.name.substring(0, 40), drawX + 3, this.barY + this.barH / 2 + 3);
                    ctx.restore();
                }
                this._opRects.push({ x: drawX, w: drawW, index: i });
            }
            xAcc += barW;
        }

        /* legend */
        const legY = this.barY + this.barH + 12;
        ctx.font = '9px monospace';
        let lx = margin;
        for (const [cat, color] of Object.entries(OP_COLORS)) {
            ctx.fillStyle = color;
            ctx.fillRect(lx, legY, 8, 8);
            ctx.fillStyle = '#666';
            const label = OP_LABELS[cat] || cat;
            ctx.fillText(label, lx + 11, legY + 7);
            lx += ctx.measureText(label).width + 22;
            if (lx > w - 50) break;
        }

        /* time axis */
        const axY = legY + 16;
        ctx.strokeStyle = '#222'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(margin, axY); ctx.lineTo(w - margin, axY); ctx.stroke();
        ctx.fillStyle = '#444'; ctx.font = '8px monospace';
        const numTicks = Math.min(20, Math.floor(w / 60));
        for (let t = 0; t <= numTicks; t++) {
            const ms = (t / numTicks) * totalMs;
            const tickX = margin + (ms / totalMs) * timeW * sx + tx;
            if (tickX > margin - 10 && tickX < w - margin + 10) {
                ctx.beginPath(); ctx.moveTo(tickX, axY - 2); ctx.lineTo(tickX, axY + 2); ctx.stroke();
                ctx.fillText(ms.toFixed(1), tickX - 8, axY + 11);
            }
        }

        /* tooltip */
        if (this.hoveredOp >= 0 && this.hoveredOp < ops.length) {
            const op = ops[this.hoveredOp];
            const text = `${op.name}  ${op.ms.toFixed(3)}ms (${op.pct.toFixed(1)}%)`;
            const tw = ctx.measureText(text).width + 12;
            const rect = this._opRects.find(r => r.index === this.hoveredOp);
            if (rect) {
                ctx.fillStyle = '#0d1117ee';
                ctx.fillRect(Math.min(rect.x, w - tw - 4), this.barY - 18, tw, 16);
                ctx.fillStyle = '#ddd'; ctx.font = '9px monospace';
                ctx.fillText(text, Math.min(rect.x, w - tw - 4) + 6, this.barY - 7);
            }
        }
        ctx.restore();
    }

    _lighten(hex, amt) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgb(${Math.min(255,r+(255-r)*amt)|0},${Math.min(255,g+(255-g)*amt)|0},${Math.min(255,b+(255-b)*amt)|0})`;
    }

    _hitTest(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        for (let i = this._opRects.length - 1; i >= 0; i--) {
            const r = this._opRects[i];
            if (mx >= r.x && mx <= r.x + r.w) return r.index;
        }
        return -1;
    }

    _onClick(e) { const idx = this._hitTest(e); if (idx >= 0 && this.onSelect) this.onSelect(idx); }

    _onHover(e) {
        const idx = this._hitTest(e);
        if (idx !== this.hoveredOp) {
            this.hoveredOp = idx;
            this.canvas.style.cursor = idx >= 0 ? 'pointer' : 'crosshair';
            this.render();
        }
    }
}
