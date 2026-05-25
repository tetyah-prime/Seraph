/*
 * app.js — Main application: SSE connection, training controls, state management
 * Wires parser → renderer, handles user interaction
 */

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STATE                                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

const state = {
    gpu: null,
    step: null,        /* latest profiler step data */
    selectedOp: -1,
    training: false,
    logLines: [],
    maxLog: 500,
    stepHistory: [],   /* { step, loss, toks, time_ms } */
    maxHistory: 1000,
};


/* ═══════════════════════════════════════════════════════════════════════════ */
/* INIT                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

const gpuCanvas = document.getElementById('gpu-canvas');
const tlCanvas  = document.getElementById('timeline');

const gpuRenderer = new GPURenderer(gpuCanvas);
const tlRenderer  = new TimelineRenderer(tlCanvas, onOpSelected);
const parser      = new ProfilerParser();

/* DOM refs */
const $model  = document.getElementById('in-model');
const $data   = document.getElementById('in-data');
const $flags  = document.getElementById('in-flags');
const $start  = document.getElementById('btn-start');
const $stop   = document.getElementById('btn-stop');
const $load   = document.getElementById('btn-load');
const $badge  = document.getElementById('status-badge');
const $gpuName  = document.getElementById('gpu-name');
const $gpuSpecs = document.getElementById('gpu-specs');
const $logOut   = document.getElementById('log-output');
const $logCount = document.getElementById('log-count');
const $logToggle = document.getElementById('log-toggle');
const $logWrap   = document.getElementById('log-wrap');
const $flagBtns = document.querySelectorAll('.flag-btn');

/* All persistable inputs: id → localStorage key */
const inputs = {
    'in-model':      'viz-model',
    'in-data':       'viz-data',
    'in-flags':      'viz-flags',
    'in-lr':         'viz-lr',
    'in-epochs':     'viz-epochs',
    'in-maxseq':     'viz-maxseq',
    'in-batch':      'viz-batch',
    'in-accum':      'viz-accum',
    'in-save':       'viz-save',
    'in-log':        'viz-log',
    'in-snapshot':   'viz-snapshot',
    'in-checkpoint': 'viz-checkpoint',
    'in-gpu-device': 'viz-gpu-device',
    'in-lora-rank':  'viz-lora-rank',
    'in-lora-alpha': 'viz-lora-alpha',
    'in-resume':     'viz-resume',
};

/* Restore + auto-save all inputs */
for (const [id, key] of Object.entries(inputs)) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.value = localStorage.getItem(key) || '';
    el.addEventListener('input', () => localStorage.setItem(key, el.value));
}

/* Restore active flag toggles */
const savedFlags = JSON.parse(localStorage.getItem('viz-flag-toggles') || '[]');
$flagBtns.forEach(btn => {
    if (savedFlags.includes(btn.dataset.flag)) btn.classList.add('active');
});

/* Flag toggle buttons */
$flagBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        const active = [...document.querySelectorAll('.flag-btn.active')].map(b => b.dataset.flag);
        localStorage.setItem('viz-flag-toggles', JSON.stringify(active));
    });
});

/* Build complete flags string from all controls */
function buildFlags() {
    const parts = [];

    /* toggle flags */
    $flagBtns.forEach(btn => {
        if (btn.classList.contains('active')) parts.push(btn.dataset.flag);
    });

    /* value flags: input id → flag name */
    const valFlags = {
        'in-lr':         '--lr',
        'in-epochs':     '--epochs',
        'in-maxseq':     '--max-seq',
        'in-batch':      '--batch',
        'in-accum':      '--accum',
        'in-save':       '--save-every',
        'in-log':        '--log-every',
        'in-snapshot':   '--snapshot-every',
        'in-checkpoint': '--checkpoint',
        'in-gpu-device': '--gpu-device',
        'in-lora-rank':  '--lora-rank',
        'in-lora-alpha': '--lora-alpha',
        'in-resume':     '--resume',
    };
    for (const [id, flag] of Object.entries(valFlags)) {
        const v = (document.getElementById(id)?.value || '').trim();
        if (v) parts.push(flag + ' ' + v);
    }

    /* extra flags (freeform) */
    const extra = $flags.value.trim();
    if (extra) parts.push(extra);

    return parts.join(' ');
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* GPU INFO                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

$load.addEventListener('click', async () => {
    const model = $model.value.trim();
    localStorage.setItem('viz-model', model);

    $load.disabled = true;
    $load.textContent = 'Loading...';

    try {
        const url = model
            ? `/api/gpu-info?model=${encodeURIComponent(model)}`
            : '/api/gpu-info';
        const resp = await fetch(url);
        const text = await resp.text();

        state.gpu = parseGPUInfo(text);
        gpuRenderer.setGPU(state.gpu);

        /* Update header */
        $gpuName.textContent = state.gpu.name || 'Unknown GPU';
        const sm = state.gpu.sm_count || '?';
        const vram = state.gpu.vram
            ? (state.gpu.vram / (1024 * 1024 * 1024)).toFixed(1) + 'GB'
            : '?';
        $gpuSpecs.textContent = `${sm} SMs | ${vram} VRAM`;

        /* Hardware panel */
        const $hwSec = document.getElementById('hw-section');
        $hwSec.style.display = 'block';
        let hwHtml = '';
        const hwFields = [
            ['Compute', `${state.gpu.compute_major ?? '?'}.${state.gpu.compute_minor ?? '?'}`],
            ['SMs', state.gpu.sm_count],
            ['Warp Size', state.gpu.warp_size],
            ['Threads/SM', state.gpu.threads_sm],
            ['Threads/Block', state.gpu.threads_block],
            ['Regs/SM', fmtNum(state.gpu.regs_sm)],
            ['Regs/Block', fmtNum(state.gpu.regs_block)],
            ['Shared/SM', fmtBytes(state.gpu.shared_sm)],
            ['Shared/Block', fmtBytes(state.gpu.shared_block)],
            ['Shared/Optin', fmtBytes(state.gpu.shared_optin)],
            ['VRAM', fmtBytes(state.gpu.vram)],
            ['L2 Cache', fmtBytes(state.gpu.l2)],
            ['Bus Width', state.gpu.bus_width ? state.gpu.bus_width + '-bit' : null],
            ['Clock SM', state.gpu.clock_sm ? state.gpu.clock_sm + ' MHz' : null],
            ['Clock Mem', state.gpu.clock_mem ? state.gpu.clock_mem + ' MHz' : null],
        ];
        for (const [k, v] of hwFields) {
            if (v) hwHtml += `<div class="hw-row"><span class="hw-key">${k}</span><span class="hw-val">${v}</span></div>`;
        }
        document.getElementById('hw-info').innerHTML = hwHtml;

        /* Model panel */
        if (state.gpu.model) {
            const $mSec = document.getElementById('model-section');
            $mSec.style.display = 'block';
            const m = state.gpu.model;
            let mHtml = '';
            const mFields = [
                ['z (hidden)', m.z], ['L (layers)', m.L], ['h (heads)', m.h],
                ['hd (head_dim)', m.hd], ['inter (ffn)', m.inter],
                ['vocab', m.vocab], ['seq (max)', m.seq], ['kv_h', m.kv_h],
            ];
            for (const [k, v] of mFields) {
                if (v) mHtml += `<div class="hw-row"><span class="hw-key">${k}</span><span class="hw-val">${v}</span></div>`;
            }
            document.getElementById('model-info').innerHTML = mHtml;
        }

        /* Derived defines panel */
        if (state.gpu.defines && Object.keys(state.gpu.defines).length > 0) {
            const $dSec = document.getElementById('defines-section');
            $dSec.style.display = 'block';
            let dHtml = '';
            for (const [k, v] of Object.entries(state.gpu.defines)) {
                dHtml += `<div class="hw-row"><span class="hw-key">${k}</span><span class="hw-val">${v}</span></div>`;
            }
            document.getElementById('defines-info').innerHTML = dHtml;
        }

        appendLog('GPU info loaded: ' + (state.gpu.name || 'unknown'));
    } catch (e) {
        appendLog('ERROR: ' + e.message);
    } finally {
        $load.disabled = false;
        $load.textContent = 'Load GPU';
    }
});


/* ═══════════════════════════════════════════════════════════════════════════ */
/* TRAINING CONTROLS                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

$start.addEventListener('click', async () => {
    const model = $model.value.trim();
    const data  = $data.value.trim();
    const flags = buildFlags();

    if (!model || !data) {
        appendLog('ERROR: need model directory and data file');
        return;
    }

    $start.disabled = true;
    $start.textContent = 'Starting...';
    appendLog('Command: seraph-train-cuda ' + model + ' ' + data + ' ' + flags);

    try {
        const resp = await fetch('/api/train/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, data, flags })
        });
        const json = await resp.json();

        if (json.ok) {
            setTraining(true);
            startPolling();
            appendLog('Training started');
        } else {
            appendLog('ERROR: ' + (json.error || 'unknown'));
            $start.disabled = false;
            $start.textContent = 'Start';
        }
    } catch (e) {
        appendLog('ERROR: ' + e.message);
        $start.disabled = false;
        $start.textContent = 'Start';
    }
});

$stop.addEventListener('click', async () => {
    $stop.disabled = true;
    $stop.textContent = 'Stopping...';
    appendLog('Sending stop signal...');
    try {
        const resp = await fetch('/api/train/stop', { method: 'POST' });
        const json = await resp.json();
        if (json.ok) {
            appendLog('Training stopped');
        } else {
            appendLog('ERROR: ' + (json.error || 'unknown'));
        }
    } catch (e) {
        appendLog('ERROR: ' + e.message);
    }
    /* setTraining(false) will be called when SSE gets the stopped/finished event */
});

/* arrow keys to step through ops */
document.addEventListener('keydown', (e) => {
    if (!state.step || !state.step.ops.length) return;
    if (e.target.tagName === 'INPUT') return; /* don't hijack input fields */
    const max = state.step.ops.length - 1;
    if (e.key === 'ArrowRight') {
        e.preventDefault();
        onOpSelected(Math.min(state.selectedOp + 1, max));
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onOpSelected(Math.max(state.selectedOp - 1, 0));
    }
});

function setTraining(running) {
    state.training = running;
    $start.disabled = running;
    $start.textContent = 'Start';
    $stop.disabled  = !running;
    $stop.textContent = 'Stop';
    $badge.className = 'badge ' + (running ? 'running' : 'idle');
    $badge.textContent = running ? 'TRAINING' : 'IDLE';
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* OUTPUT POLLING                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

let pollTimer = null;
let lastOutputLen = 0;

function startPolling() {
    stopPolling();
    lastOutputLen = 0;
    pollTimer = setInterval(pollOutput, 500);
}

function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollOutput() {
    try {
        const resp = await fetch('/api/train/output');
        const text = await resp.text();

        /* only process new content */
        if (text.length <= lastOutputLen) {
            /* check if training ended */
            const status = await fetch('/api/status');
            const json = await status.json();
            if (!json.training && state.training) {
                setTraining(false);
                appendLog('Training finished');
                const last = parser.flush();
                if (last) handleParserResult(last);
                stopPolling();
            }
            return;
        }

        const newText = text.substring(lastOutputLen);
        lastOutputLen = text.length;

        /* show in terminal */
        $term.textContent = text;
        $term.scrollTop = $term.scrollHeight;

        /* parse new lines */
        const lines = newText.split('\n');
        for (const line of lines) {
            if (line.trim() === '') continue;
            const result = parser.parseLine(line);
            if (result) handleParserResult(result);
        }
    } catch (_) {}
}

function handleParserResult(result) {
    switch (result.type) {
        case 'step':
            state.step = result.data;
            updateStepUI(result.data);
            gpuRenderer.setStep(result.data);
            tlRenderer.setStep(result.data);
            break;

        case 'training':
            updateTrainingStatus(result);
            state.stepHistory.push(result);
            if (state.stepHistory.length > state.maxHistory)
                state.stepHistory.shift();
            break;

        case 'status':
            appendLog(result.text);
            break;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* UI UPDATES                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

function updateStepUI(step) {
    document.getElementById('stat-ops').textContent = step.num_ops;
    document.getElementById('stat-steptime').textContent = step.total_ms.toFixed(2) + ' ms';

    /* category breakdown */
    const $cats = document.getElementById('cat-bars');
    let html = '';
    const sorted = [...step.categories].sort((a, b) => b.ms - a.ms);
    for (const cat of sorted) {
        const catKey = categorizeOp(cat.name);
        const color = OP_COLORS[catKey] || '#444';
        html += `<div class="cat-row">
            <span class="cat-name">${cat.name}</span>
            <div class="cat-bar-wrap">
                <div class="cat-bar-fill" style="width:${cat.pct}%;background:${color}"></div>
            </div>
            <span class="cat-ms">${cat.ms.toFixed(1)}</span>
            <span class="cat-pct">${cat.pct.toFixed(1)}%</span>
        </div>`;
    }
    $cats.innerHTML = html;
}

function updateTrainingStatus(info) {
    if (info.step !== undefined) {
        const stepText = info.total_steps
            ? `${info.step} / ${info.total_steps}`
            : `${info.step}`;
        document.getElementById('stat-step').textContent = stepText;
    }
    if (info.loss !== undefined) {
        let lossText = info.loss.toFixed(4);
        if (info.avg_loss !== undefined) lossText += ` (avg ${info.avg_loss.toFixed(4)})`;
        document.getElementById('stat-loss').textContent = lossText;
    }
    if (info.toks !== undefined)
        document.getElementById('stat-toks').textContent = info.toks.toFixed(0);
}

function onOpSelected(index) {
    state.selectedOp = index;
    gpuRenderer.selectOp(index);
    tlRenderer.selectOp(index);

    /* update op detail panel */
    const $det = document.getElementById('op-detail');
    if (!state.step || index < 0 || index >= state.step.ops.length) {
        $det.innerHTML = '<span class="hint">Click an operation in the timeline</span>';
        return;
    }

    const op = state.step.ops[index];
    let html = `<div class="op-name">${op.name}</div>`;

    const rows = [
        ['Time', op.ms.toFixed(3) + ' ms'],
        ['% of step', op.pct.toFixed(1) + '%'],
        ['Category', op.category],
    ];

    if (op.grid) {
        rows.push(['Grid', `(${op.grid.x}, ${op.grid.y}, ${op.grid.z})`]);
        rows.push(['Block', `(${op.block.x}, ${op.block.y}, ${op.block.z})`]);
        rows.push(['Total Blocks', op.total_blocks]);
        rows.push(['Threads/Block', op.threads_per_block]);
    }

    if (op.gflops) rows.push(['GFLOPS', op.gflops.toFixed(2)]);
    if (op.gbps) rows.push(['GB/s', op.gbps.toFixed(2)]);

    if (state.gpu && op.total_blocks && op.threads_per_block) {
        const sm = state.gpu.sm_count;
        const warpSize = state.gpu.warp_size;
        const maxWarps = state.gpu.threads_sm / warpSize;
        const wpb = Math.ceil(op.threads_per_block / warpSize);
        const maxConcurrent = Math.floor(maxWarps / wpb);
        const assigned = Math.ceil(op.total_blocks / sm);
        const concurrent = Math.min(assigned, maxConcurrent);
        const waves = Math.ceil(assigned / maxConcurrent);
        const occ = (concurrent * wpb) / maxWarps;

        rows.push(['Blocks/SM', concurrent]);
        rows.push(['Warps/Block', wpb]);
        rows.push(['Occupancy', (occ * 100).toFixed(0) + '% (' + (concurrent * wpb) + '/' + maxWarps + ' warps)']);
        rows.push(['Waves', waves]);
    }

    for (const [k, v] of rows) {
        html += `<div class="op-row"><span class="op-key">${k}</span><span class="op-val">${v}</span></div>`;
    }

    $det.innerHTML = html;
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* LOG                                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

function appendLog(text) {
    state.logLines.push(text);
    if (state.logLines.length > state.maxLog) state.logLines.shift();

    $logOut.textContent = state.logLines.join('\n');
    $logOut.scrollTop = $logOut.scrollHeight;
    $logCount.textContent = state.logLines.length;
}

$logToggle.addEventListener('click', () => {
    $logWrap.classList.toggle('open');
});


/* ═══════════════════════════════════════════════════════════════════════════ */
/* UTILITIES                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

function fmtBytes(b) {
    if (b === undefined || b === null) return null;
    if (b >= 1024 * 1024 * 1024) return (b / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    if (b >= 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
    return b + ' B';
}

function fmtNum(n) {
    if (n === undefined || n === null) return null;
    return n.toLocaleString();
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* TERMINAL                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

const $term = document.getElementById('term-output');
const $termWrap = document.getElementById('term-wrap');
const termLines = [];
const TERM_MAX = 2000;

function termAppend(line) {
    termLines.push(line);
    if (termLines.length > TERM_MAX) termLines.shift();
    $term.textContent = termLines.join('\n');
    $term.scrollTop = $term.scrollHeight;
}

document.getElementById('term-toggle').addEventListener('click', () => {
    $termWrap.classList.toggle('term-closed');
    $termWrap.classList.toggle('term-open');
});

document.getElementById('term-header').addEventListener('click', (e) => {
    if (e.target.id !== 'term-toggle') {
        $termWrap.classList.toggle('term-closed');
        $termWrap.classList.toggle('term-open');
    }
});


/* ═══════════════════════════════════════════════════════════════════════════ */
/* FILE BROWSER                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

const browserState = {
    targetInput: null,  /* which input to fill */
    pickMode: 'dir',    /* 'dir' or 'file' */
    currentPath: '/',
    selected: null,     /* selected entry name */
};

const $overlay   = document.getElementById('browser-overlay');
const $bPath     = document.getElementById('browser-path');
const $bList     = document.getElementById('browser-list');
const $bSelect   = document.getElementById('browser-select');
const $bClose    = document.getElementById('browser-close');
const $bGo       = document.getElementById('browser-go');
const $bTitle    = document.getElementById('browser-title');

/* Open browser modal */
document.querySelectorAll('.path-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const targetId = btn.dataset.target;
        browserState.targetInput = document.getElementById(targetId);
        browserState.pickMode = btn.dataset.pick || 'dir';
        $bTitle.textContent = browserState.pickMode === 'dir' ? 'Select Directory' : 'Select File';

        /* start from current value or home */
        const cur = browserState.targetInput.value.trim();
        const startPath = cur || '/home';
        browserState.currentPath = startPath;
        browserState.selected = null;

        $overlay.classList.remove('hidden');
        browseDir(startPath);
    });
});

$bClose.addEventListener('click', () => $overlay.classList.add('hidden'));
$overlay.addEventListener('click', (e) => { if (e.target === $overlay) $overlay.classList.add('hidden'); });

$bGo.addEventListener('click', () => browseDir($bPath.value.trim()));
$bPath.addEventListener('keydown', (e) => { if (e.key === 'Enter') browseDir($bPath.value.trim()); });

$bSelect.addEventListener('click', () => {
    if (!browserState.targetInput) return;

    let path = browserState.currentPath;
    if (browserState.pickMode === 'file' && browserState.selected) {
        path = browserState.currentPath + '/' + browserState.selected;
        path = path.replace(/\/+/g, '/');
    }

    browserState.targetInput.value = path;
    browserState.targetInput.dispatchEvent(new Event('input')); /* trigger save */
    $overlay.classList.add('hidden');
});

async function browseDir(path) {
    $bPath.value = path;
    $bList.innerHTML = '<div style="padding:12px;color:#666">Loading...</div>';

    try {
        const resp = await fetch('/api/browse?path=' + encodeURIComponent(path));
        const data = await resp.json();

        if (data.error) {
            $bList.innerHTML = '<div style="padding:12px;color:#ff4a6a">' + data.error + '</div>';
            return;
        }

        browserState.currentPath = data.path;
        $bPath.value = data.path;
        browserState.selected = null;

        /* sort: dirs first, then files, alphabetical */
        const entries = data.entries.sort((a, b) => {
            if (a.dir !== b.dir) return a.dir ? -1 : 1;
            return a.name.localeCompare(b.name);
        });

        let html = '';
        for (const ent of entries) {
            const cls = ent.dir ? 'browser-item is-dir' : 'browser-item';
            const icon = ent.dir ? '/' : ' ';
            html += `<div class="${cls}" data-name="${ent.name}" data-dir="${ent.dir}">
                <span class="icon">${icon}</span><span class="name">${ent.name}</span>
            </div>`;
        }
        $bList.innerHTML = html;

        /* click handlers */
        $bList.querySelectorAll('.browser-item').forEach(item => {
            item.addEventListener('dblclick', () => {
                if (item.dataset.dir === 'true') {
                    const newPath = browserState.currentPath + '/' + item.dataset.name;
                    browseDir(newPath.replace(/\/+/g, '/'));
                }
            });
            item.addEventListener('click', () => {
                $bList.querySelectorAll('.browser-item').forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');
                browserState.selected = item.dataset.name;

                /* for dir pick mode, navigate on single click of dirs */
                if (browserState.pickMode === 'dir' && item.dataset.dir === 'true' && item.dataset.name !== '..') {
                    /* just select it, dblclick to enter */
                }
            });
        });

    } catch (e) {
        $bList.innerHTML = '<div style="padding:12px;color:#ff4a6a">Error: ' + e.message + '</div>';
    }
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* REPLAY                                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

const replay = {
    steps: [],       /* array of parsed step data (each = { ops, categories, ... }) */
    stepIndex: 0,    /* current step being shown */
    playing: false,
    timer: null,
};

document.getElementById('btn-replay-load').addEventListener('click', () => {
    document.getElementById('replay-file').click();
});

document.getElementById('replay-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
        const text = reader.result;
        loadReplay(text);
    };
    reader.readAsText(file);
});

function loadReplay(text) {
    const rParser = new ProfilerParser();
    replay.steps = [];

    const lines = text.split('\n');
    for (const line of lines) {
        if (line.trim() === '') continue;
        const result = rParser.parseLine(line);
        if (result && result.type === 'step') {
            replay.steps.push(result.data);
        }
    }
    const last = rParser.flush();
    if (last && last.type === 'step') {
        replay.steps.push(last.data);
    }

    replay.stepIndex = 0;
    replay.playing = false;

    /* show controls */
    document.getElementById('replay-controls').style.display = 'block';
    document.getElementById('replay-steps').textContent = replay.steps.length;
    document.getElementById('replay-pos').textContent = '1 / ' + replay.steps.length;

    /* load first step — keep whatever op is selected */
    if (replay.steps.length > 0) {
        replayShowStep(0);
    }

    appendLog('Loaded replay: ' + replay.steps.length + ' profiled steps');
}

function replayShowStep(idx) {
    const step = replay.steps[idx];
    state.step = step;
    updateStepUI(step);
    gpuRenderer.setStep(step);
    tlRenderer.setStep(step);

    /* re-select the same op index (watch it change across steps) */
    const opIdx = Math.min(state.selectedOp, step.ops.length - 1);
    if (opIdx >= 0) onOpSelected(opIdx);

    document.getElementById('replay-pos').textContent =
        (idx + 1) + ' / ' + replay.steps.length;
}

function replayTick() {
    if (replay.stepIndex >= replay.steps.length) {
        replayPause();
        return;
    }

    replayShowStep(replay.stepIndex);
    replay.stepIndex++;
}

function getReplayInterval() {
    const active = document.querySelector('.speed-btn.active');
    return active ? parseInt(active.dataset.ms) : 500;
}

function replayPlay() {
    if (replay.steps.length === 0) return;
    replay.playing = true;
    if (replay.timer) clearInterval(replay.timer);
    replay.timer = setInterval(replayTick, getReplayInterval());
}

function replayPause() {
    replay.playing = false;
    if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
}

document.getElementById('btn-replay-play').addEventListener('click', replayPlay);
document.getElementById('btn-replay-pause').addEventListener('click', replayPause);
document.getElementById('btn-replay-reset').addEventListener('click', () => {
    replayPause();
    replay.stepIndex = 0;
    if (replay.steps.length > 0) replayShowStep(0);
});

/* speed buttons — click to select, restarts playback at new speed */
document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (replay.playing) {
            replayPause();
            replayPlay();
        }
    });
});


/* ═══════════════════════════════════════════════════════════════════════════ */
/* WALLPAPER                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

const $canvasWrap = document.getElementById('canvas-wrap');

document.getElementById('btn-wallpaper').addEventListener('click', () => {
    document.getElementById('wallpaper-file').click();
});

document.getElementById('wallpaper-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        setWallpaper(reader.result);
        localStorage.setItem('viz-wallpaper', reader.result);
    };
    reader.readAsDataURL(file);
});

document.getElementById('btn-wallpaper-clear').addEventListener('click', () => {
    $canvasWrap.style.backgroundImage = '';
    localStorage.removeItem('viz-wallpaper');
});

function setWallpaper(dataUrl) {
    $canvasWrap.style.backgroundImage = `url(${dataUrl})`;
}

/* restore saved wallpaper */
const savedWallpaper = localStorage.getItem('viz-wallpaper');
if (savedWallpaper) setWallpaper(savedWallpaper);


/* ═══════════════════════════════════════════════════════════════════════════ */
/* AUTO-CONNECT                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Check if training is already running on page load */
(async function init() {
    try {
        const resp = await fetch('/api/status');
        const json = await resp.json();
        if (json.training) {
            setTraining(true);
            startPolling();
            appendLog('Reconnected to running training');
        }
    } catch (_) {}

    /* Auto-load GPU if model dir is set */
    if ($model.value.trim()) {
        $load.click();
    }
})();
