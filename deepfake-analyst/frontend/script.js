// script.js - Deepfake Content Analyzer

const $ = (id) => document.getElementById(id);
const fileInput = $('video-upload');
const dropZone = $('drop-zone');
const filePrompt = $('file-prompt');
const fileHint = $('file-hint');
const preview = $('preview');
const analyzeBtn = $('analyze-btn');
const clearBtn = $('clear-btn');
const resultBody = $('result-body');
const statusEl = $('backend-status');

// ---------- Config ----------
// Local (file:// or localhost) -> FastAPI on port 8000.
// Deployed -> same origin, or set window.API_BASE in index.html.
const isLocal = location.protocol === 'file:' || ['localhost', '127.0.0.1'].includes(location.hostname);
const API_BASE = window.API_BASE ?? (isLocal ? 'http://127.0.0.1:8000' : '');
const ANALYZE_URL = `${API_BASE}/api/v1/analyze`;
// Vercel functions reject request bodies over ~4.5 MB, so the hosted default is small.
const MAX_MB = window.MAX_UPLOAD_MB ?? (isLocal ? 200 : 4);
const TIMEOUT_MS = 5 * 60 * 1000;

const STEPS = [
  'Uploading video',
  'Extracting frames and audio',
  'Checking frames for manipulation',
  'Transcribing and scanning speech',
];
const STEP_DELAYS = [0, 4000, 10000, 20000]; // estimates; the backend sends no live progress

let previewUrl = null;
let stepTimers = [];
let lastResult = null;

// ---------- Helpers ----------
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const pct = (n) => `${Math.round((Number(n) || 0) * 100)}%`;
const toProb = (n) => { n = Number(n); if (isNaN(n)) return 0; return n > 1 ? n / 100 : n; };
const tone = (p) => (p >= 0.8 ? 'bad' : p >= 0.4 ? 'warn' : 'safe');

function setBody(html) { resultBody.innerHTML = html; }

// ---------- Backend status ----------
async function checkBackend() {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 3000);
    const r = await fetch(`${API_BASE}/`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!r.ok) throw new Error();
    statusEl.dataset.state = 'online';
    statusEl.textContent = 'Backend online';
  } catch {
    statusEl.dataset.state = 'offline';
    statusEl.textContent = `Backend not reachable at ${API_BASE || location.origin}`;
  }
}

// ---------- File selection ----------
function setFile(file) {
  if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
  if (!file) {
    fileInput.value = '';
    filePrompt.textContent = 'Choose a video or drag and drop it here';
    fileHint.textContent = 'MP4, MOV or WebM';
    preview.hidden = true; preview.removeAttribute('src');
    analyzeBtn.disabled = true; clearBtn.hidden = true;
    return;
  }
  const mb = file.size / 1024 / 1024;
  filePrompt.textContent = file.name;
  fileHint.textContent = `${mb.toFixed(1)} MB`;
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl; preview.hidden = false;
  clearBtn.hidden = false;
  analyzeBtn.disabled = false;
  if (mb > MAX_MB) {
    analyzeBtn.disabled = true;
    showError(new Error(`This file is ${mb.toFixed(1)} MB. The limit here is ${MAX_MB} MB.`), 'size');
  } else if (!file.type.startsWith('video/')) {
    analyzeBtn.disabled = true;
    showError(new Error('That file is not a video.'), 'type');
  } else {
    showEmpty();
  }
}

fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
clearBtn.addEventListener('click', () => { setFile(null); showEmpty(); });

['dragenter', 'dragover'].forEach((ev) => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((ev) => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  const dt = new DataTransfer(); dt.items.add(f);
  fileInput.files = dt.files;
  setFile(f);
});

// ---------- Views ----------
function showEmpty() {
  setBody('<p class="empty">No result yet. Choose a video and select Analyze video.</p>');
}

function showLoading() {
  setBody(`<ol class="steps">${STEPS.map((s, i) => `<li data-i="${i}">${esc(s)}</li>`).join('')}</ol>
    <p class="note">Long videos can take a few minutes. Keep this tab open.</p>`);
  const items = [...resultBody.querySelectorAll('.steps li')];
  stepTimers.forEach(clearTimeout);
  stepTimers = STEP_DELAYS.map((d, i) => setTimeout(() => {
    items.forEach((li, j) => { li.className = j < i ? 'done' : j === i ? 'active' : ''; });
  }, d));
}

function stopLoading() { stepTimers.forEach(clearTimeout); stepTimers = []; }

function verdictFor(data, prob) {
  const label = String(data.label ?? '');
  if (/harm/i.test(label)) return { tone: 'bad', text: label };
  if (/susp/i.test(label)) return { tone: 'warn', text: label };
  if (/safe/i.test(label)) return { tone: 'safe', text: label };
  return { tone: tone(prob), text: label || (prob >= 0.8 ? 'Likely deepfake' : prob >= 0.4 ? 'Suspicious' : 'Likely real') };
}

function meter(name, value) {
  const p = toProb(value);
  return `<div class="meter">
    <div class="meter-head"><span>${esc(name)}</span><strong>${pct(p)}</strong></div>
    <div class="track"><div class="fill" data-tone="${tone(p)}" data-w="${Math.round(p * 100)}"></div></div>
  </div>`;
}

function renderResults(data, { demo = false } = {}) {
  lastResult = data;
  const prob = toProb(data.deepfake_probability ?? data.probability);
  const v = verdictFor(data, prob);
  const transcript = (data.transcript ?? '').trim();

  setBody(`
    ${demo ? '<span class="badge-demo">Demo data, not from your video</span>' : ''}
    <div class="verdict" data-tone="${v.tone}"><span class="verdict-dot"></span><h3>${esc(v.text)}</h3></div>
    ${meter('Deepfake probability', prob)}
    ${data.harmful_text_probability != null ? meter('Harmful speech probability', data.harmful_text_probability) : ''}
    ${data.ethical_score != null ? meter('Risk score (higher means riskier)', data.ethical_score) : ''}
    <div class="block">
      <h4>Transcript</h4>
      <p class="transcript ${transcript ? '' : 'muted'}">${transcript ? esc(transcript) : 'No speech detected in this video.'}</p>
    </div>
    <details>
      <summary>Raw backend response</summary>
      <pre>${esc(JSON.stringify(data, null, 2))}</pre>
      <div class="controls"><button class="btn ghost" id="copy-json" type="button">Copy JSON</button></div>
    </details>
  `);

  // animate meters after paint
  requestAnimationFrame(() => resultBody.querySelectorAll('.fill').forEach((f) => { f.style.width = f.dataset.w + '%'; }));
  $('copy-json').addEventListener('click', async (e) => {
    try { await navigator.clipboard.writeText(JSON.stringify(data, null, 2)); e.target.textContent = 'Copied'; }
    catch { e.target.textContent = 'Copy failed'; }
  });
}

function fixesFor(err, kind) {
  if (kind === 'size' || kind === 'type') return ['Choose a different video file.'];
  if (kind === 'timeout') return ['Try a shorter or smaller video.', 'Check the backend terminal to see whether it is still processing.'];
  if (kind === 'network') {
    const fixes = [
      `The page tried <code>${esc(ANALYZE_URL)}</code> and got no answer.`,
      'Start the backend from the <code>deepfake-analyst</code> folder: <code>uvicorn backend.app.main:app --reload --port 8000</code>',
      'Make sure your virtual environment is activated first.',
      'Open <code>http://127.0.0.1:8000/docs</code> in a browser. If it does not load, the server is not running.',
    ];
    if (!isLocal) fixes.push('On a hosted site, set <code>window.API_BASE</code> in index.html to your backend URL and allow that origin in CORS.');
    return fixes;
  }
  return [
    'Read the error text above. It comes from the server.',
    'Check the backend terminal for a Python traceback.',
    'Confirm <code>cnn_model.h5</code> and <code>Meso4_DF.h5</code> are in <code>backend/models</code> and loaded at startup.',
    'Make sure the video is a normal MP4 with a readable video track.',
  ];
}

function showError(err, kind = 'server') {
  stopLoading();
  setBody(`
    <div class="verdict" data-tone="error"><span class="verdict-dot"></span><h3>${kind === 'size' || kind === 'type' ? 'Can’t analyze this file' : 'Analysis failed'}</h3></div>
    <p class="error-text">${esc(err.message)}</p>
    <ul class="fixes">${fixesFor(err, kind).map((f) => `<li>${f}</li>`).join('')}</ul>
    <div class="controls">
      ${kind === 'size' || kind === 'type' ? '' : '<button class="btn" id="retry-btn" type="button">Try again</button>'}
      <button class="btn ghost" id="demo-btn" type="button">Preview with demo result</button>
    </div>
  `);
  const retry = $('retry-btn');
  if (retry) retry.addEventListener('click', analyze);
  $('demo-btn').addEventListener('click', () => renderResults({
    filename: 'demo.mp4',
    label: 'Suspicious ⚠',
    ethical_score: 0.52,
    deepfake_probability: 0.83,
    harmful_text_probability: 0.12,
    transcript: 'This is a sample transcript used to preview the layout.',
  }, { demo: true }));
}

// ---------- Analyze ----------
async function analyze() {
  const file = fileInput.files && fileInput.files[0];
  if (!file) { showError(new Error('Choose a video first.'), 'type'); return; }

  const body = new FormData();
  body.append('video', file); // field name must match `video` in main.py

  analyzeBtn.disabled = true;
  showLoading();

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);

  try {
    const resp = await fetch(ANALYZE_URL, { method: 'POST', body, signal: ctrl.signal });
    if (!resp.ok) {
      let text = '';
      try { text = await resp.text(); } catch { text = resp.statusText; }
      const err = new Error(`Server responded with ${resp.status}${text ? `: ${text.slice(0, 400)}` : ''}`);
      err.kind = 'server';
      throw err;
    }
    const json = await resp.json();
    stopLoading();
    renderResults(json);
    checkBackend();
  } catch (err) {
    console.error('Analysis error:', err);
    const kind = err.name === 'AbortError' ? 'timeout' : err.kind || (err instanceof TypeError ? 'network' : 'server');
    if (kind === 'timeout') err = new Error('The analysis took longer than 5 minutes and was stopped.');
    if (kind === 'network') err = new Error('Could not reach the backend. It may be offline, or the browser blocked the request.');
    showError(err, kind);
    checkBackend();
  } finally {
    clearTimeout(timer);
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener('click', (e) => { e.preventDefault(); analyze(); });

// ---------- Init ----------
showEmpty();
checkBackend();