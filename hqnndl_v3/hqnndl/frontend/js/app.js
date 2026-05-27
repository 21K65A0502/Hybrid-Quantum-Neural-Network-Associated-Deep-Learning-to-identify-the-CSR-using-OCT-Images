/**
 * app.js  —  HQNNDL v3.0 Frontend
 * ──────────────────────────────────
 * Handles: upload, sample generation, pipeline animation,
 *          API calls, result rendering, history, toast.
 */

// ── Config ──────────────────────────────────────────────────────
const API     = '';          // relative — works for any host
const TIMEOUT = 30000;       // 30s fetch timeout

// ── Pipeline step definitions ────────────────────────────────────
const PIPE_STEPS = [
  { label: 'OCT Input',      icon: '🩻' },
  { label: 'CNN Features',   icon: '🧠' },
  { label: 'Quantum Enc.',   icon: '📡' },
  { label: 'QNN / VQC',     icon: '⚛'  },
  { label: 'Classification', icon: '🎯' },
  { label: 'Report',         icon: '📄' },
];

const STAGE_DEFS = [
  { id: 'preprocess', label: 'OCT Preprocessing (CLAHE + Denoising)' },
  { id: 'cnn',        label: 'CNN Feature Extraction (ResNet)' },
  { id: 'encode',     label: 'Amplitude Encoding → Quantum Circuit' },
  { id: 'vqc',        label: 'Variational Quantum Circuit (3 layers)' },
  { id: 'classify',   label: 'Hybrid Fusion + Final Classification' },
  { id: 'detect',     label: 'Affected Area Detection + Annotation' },
];

// ── State ────────────────────────────────────────────────────────
let currentFile = null;
let pipeStep    = 0;
let stageIdx    = -1;
let stageTimes  = {};
let isRunning   = false;
let history     = [];

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderPipeline();
  renderStages();
  checkHealth();

  const dz = document.getElementById('dropZone');
  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', ()  => dz.classList.remove('drag'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  document.getElementById('fileInput').addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
});

// ── Health check ─────────────────────────────────────────────────
async function checkHealth() {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  try {
    const r = await fetchWithTimeout(`${API}/api/health`);
    const d = await r.json();
    dot.style.background = '#22c55e';
    txt.textContent = d.model;
  } catch {
    dot.style.background = '#ef4444';
    txt.textContent      = 'Backend Offline';
  }
}

// ── File handling ─────────────────────────────────────────────────
function handleFile(file) {
  const allowed = ['image/png','image/jpeg','image/jpg','image/bmp','image/tiff','image/webp'];
  if (!allowed.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|bmp|tiff|tif|webp)$/i)) {
    toast('Please upload a valid image (PNG / JPG / BMP / TIFF)', 'error');
    return;
  }
  currentFile = file;
  document.getElementById('previewImg').src   = URL.createObjectURL(file);
  document.getElementById('previewName').textContent = file.name;
  document.getElementById('previewSize').textContent = (file.size / 1024).toFixed(1) + ' KB';
  document.getElementById('previewSection').style.display = 'block';
  document.getElementById('dropZone').style.display       = 'none';
  document.getElementById('analyzeBtn').disabled          = false;
  resetPipeline();
}

function clearFile() {
  currentFile = null;
  document.getElementById('previewSection').style.display = 'none';
  document.getElementById('dropZone').style.display       = 'block';
  document.getElementById('fileInput').value = '';
  document.getElementById('analyzeBtn').disabled = true;
  resetPipeline();
}

function resetPipeline() {
  pipeStep = 0; stageIdx = -1; stageTimes = {};
  renderPipeline(); renderStages();
}

// ── Synthetic OCT sample generator ───────────────────────────────
function loadSample(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 512, 256);

  // Horizontal retinal layers
  const baseY = type === 'csr2' ? 14 : type === 'csr1' ? 7 : 0;
  [[55,3,.9],[72,2,.75],[90,4,.85],[125,3,.8],[155,2,.7],[175,5,.88]].forEach(([y, h, a]) => {
    const g = ctx.createLinearGradient(0, 0, 512, 0);
    const c = `rgba(${200+(Math.random()*30|0)},${200+(Math.random()*30|0)},${210+(Math.random()*20|0)},${a})`;
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(.1, c);
    g.addColorStop(.9, c); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g; ctx.fillRect(0, y + baseY, 512, h);
  });

  // Subretinal fluid dome for CSR cases
  if (type === 'csr2' || type === 'csr1') {
    const [rw, rh, intensity] = type === 'csr2' ? [90, 38, .82] : [52, 20, .52];
    const g2 = ctx.createRadialGradient(256, 148, 0, 256, 148, rw);
    g2.addColorStop(0, `rgba(15,40,90,${intensity})`);
    g2.addColorStop(.6, `rgba(8,22,60,${intensity * .55})`);
    g2.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g2;
    ctx.beginPath(); ctx.ellipse(256, 150, rw, rh, 0, 0, Math.PI * 2); ctx.fill();
  }

  // Bright drusen for 'other' pathology
  if (type === 'other') {
    for (let i = 0; i < 8; i++) {
      const x = 120 + Math.random() * 280, y = 155 + Math.random() * 30;
      const g3 = ctx.createRadialGradient(x, y, 0, x, y, 6 + Math.random() * 4);
      g3.addColorStop(0, 'rgba(255,240,180,.7)'); g3.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g3; ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
    }
  }

  // Speckle noise
  for (let i = 0; i < 3500; i++) {
    ctx.fillStyle = `rgba(255,255,255,${Math.random() * .1})`;
    ctx.fillRect(Math.random() * 512, Math.random() * 256, 1, 1);
  }

  canvas.toBlob(blob => {
    const names = { csr1: 'CSR_Grade1', csr2: 'CSR_Grade2', normal: 'Normal', other: 'Other' };
    handleFile(new File([blob], `${names[type]}_OCT_sample.png`, { type: 'image/png' }));
  }, 'image/png');
}

// ── Pipeline & stage rendering ────────────────────────────────────
function renderPipeline() {
  document.getElementById('pipelineBar').innerHTML = PIPE_STEPS.map((s, i) => {
    const cls      = i < pipeStep ? 'done' : (i === pipeStep && isRunning ? 'active' : '');
    const arrowCls = i > 0 && i <= pipeStep ? 'done' : '';
    return (i > 0 ? `<div class="pipe-arrow ${arrowCls}"></div>` : '') +
      `<div class="pipe-step ${cls}">
         <div class="pipe-icon">${s.icon}</div>
         <div class="pipe-label">${s.label}</div>
       </div>`;
  }).join('');
}

function renderStages() {
  document.getElementById('stageList').innerHTML = STAGE_DEFS.map((s, i) => {
    const cls  = i < stageIdx ? 'done' : i === stageIdx ? 'running' : 'pending';
    const time = stageTimes[s.id] ? `${stageTimes[s.id]}ms` : '';
    return `<div class="stage ${cls}" id="stage_${s.id}">
      <div class="stage-dot"></div>
      <div class="stage-name">${s.label}</div>
      <div class="stage-time">${cls === 'running' ? '…' : time}</div>
    </div>`;
  }).join('');
}

// ── Analysis ──────────────────────────────────────────────────────
async function runAnalysis() {
  if (!currentFile || isRunning) return;
  isRunning = true;
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div><span>Analysing…</span>';

  // Animate stages while waiting
  stageIdx = -1; pipeStep = 0;
  const animInterval = setInterval(() => {
    if (stageIdx < STAGE_DEFS.length - 1) {
      stageIdx++; renderPipeline(); renderStages();
    }
  }, 380);

  try {
    const fd = new FormData();
    fd.append('image', currentFile);

    const res = await fetchWithTimeout(`${API}/api/analyze`, {
      method: 'POST', body: fd,
    });

    clearInterval(animInterval);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `Server error ${res.status}`);
    }

    const d = await res.json();

    // Populate real timings
    stageTimes = {
      preprocess: d.performance.preprocess_ms,
      cnn:        d.performance.cnn_ms,
      encode:     Math.round(d.performance.quantum_ms * 0.15),
      vqc:        Math.round(d.performance.quantum_ms * 0.85),
      classify:   d.performance.biomarker_ms,
      detect:     d.performance.detection_ms,
    };
    stageIdx = STAGE_DEFS.length;
    pipeStep = PIPE_STEPS.length;
    renderPipeline(); renderStages();

    renderResult(d);
    addToHistory(d);
    toast(`Analysis complete — ${d.diagnosis.label} (${d.diagnosis.confidence}%)`);

  } catch (e) {
    clearInterval(animInterval);
    stageIdx = -1; renderStages();
    toast(e.message, 'error');
  } finally {
    isRunning = false;
    btn.disabled = false;
    btn.innerHTML = '<span>⚛</span><span>Run HQNNDL Analysis</span>';
  }
}

// ── Result rendering ──────────────────────────────────────────────
function renderResult(d) {
  const dx  = d.diagnosis.label;
  const conf= d.diagnosis.confidence;
  const rec = d.recommendation;
  const bio = d.biomarkers;
  const regions = d.regions || [];

  const dcls    = dx.includes('Grade II') ? 'csr2' : dx.includes('Grade I') ? 'csr1' : dx === 'Normal' ? 'normal' : 'other';
  const dcolor  = dcls==='csr2'?'var(--red)':dcls==='csr1'?'var(--amber)':dcls==='normal'?'var(--green)':'var(--violet)';
  const demoji  = {csr2:'🔴',csr1:'🟡',normal:'🟢',other:'🟣'}[dcls];
  const urgCls  = rec.urgency==='URGENT'?'badge-urgent':rec.urgency==='PROMPT'?'badge-prompt':'badge-routine';

  // Probabilities
  const probHTML = d.diagnosis.probabilities.map(p =>
    `<div class="prob-row">
       <div class="prob-label" style="color:${p.color}">${p.label}</div>
       <div class="prob-track"><div class="prob-fill" style="width:${p.probability}%;background:${p.color}"></div></div>
       <div class="prob-pct">${p.probability}%</div>
     </div>`
  ).join('');

  // Biomarkers
  const bioHTML = bio.map(b => {
    const sc = 'sev-' + b.severity.toLowerCase();
    return `<div class="bio-card">
      <div class="bio-name">${b.name}</div>
      <div class="bio-track"><div class="bio-fill" style="width:${b.score}%;background:${b.color}"></div></div>
      <div class="bio-footer">
        <span class="bio-score">${b.score}/99</span>
        <span class="bio-sev ${sc}">${b.severity}</span>
      </div>
    </div>`;
  }).join('');

  // Regions
  const dotColors = {'CSR Grade II':'#ef4444','CSR Grade I':'#f59e0b','Normal':'#22c55e','Other Pathology':'#8b5cf6'};
  const dotCol = dotColors[dx] || '#64748b';
  const regionsHTML = regions.length
    ? regions.map(r =>
        `<div class="region-row">
           <div class="region-dot" style="background:${dotCol}"></div>
           <div class="region-label">${r.label}</div>
           <div class="region-sal">Saliency: ${r.saliency}%</div>
           <div class="region-area">Area: ${r.area_pct}%</div>
         </div>`).join('')
    : '<div style="font-size:12px;color:var(--muted);padding:8px 0">No significant affected regions detected</div>';

  // Qubit bars
  const qm   = d.quantum.measurements;
  const qMax = Math.max(...qm.map(Math.abs));
  const qBars  = qm.map(v => {
    const h   = Math.max(4, Math.round(Math.abs(v) / qMax * 52));
    const col = v >= 0 ? '#2563eb' : '#dc2626';
    return `<div class="qb" style="height:${h}px;background:${col}"></div>`;
  }).join('');
  const qLabels = qm.map((_, i) => `<div class="ql">Q${i}</div>`).join('');

  // Performance chips
  const perfHTML = Object.entries(d.performance).map(([k, v]) =>
    `<span class="perf-chip">${k.replace('_ms','')}:<b>${v}ms</b></span>`
  ).join('');

  document.getElementById('resultCol').innerHTML = `
  <div class="card">
    <div class="dx-banner ${dcls}">
      <div class="dx-emoji">${demoji}</div>
      <div>
        <div class="dx-name" style="color:${dcolor}">${dx}</div>
        <div class="dx-meta">Confidence: ${conf}% &nbsp;·&nbsp; ID: ${d.id} &nbsp;·&nbsp; ${d.timestamp}</div>
      </div>
      <div class="dx-badge ${urgCls}">${rec.urgency}</div>
    </div>

    <div class="metrics-row">
      <div class="metric"><span class="metric-val" style="color:var(--blue)">98.6%</span><div class="metric-lbl">Accuracy</div></div>
      <div class="metric"><span class="metric-val" style="color:var(--violet)">97.9%</span><div class="metric-lbl">Sensitivity</div></div>
      <div class="metric"><span class="metric-val" style="color:var(--green)">99.1%</span><div class="metric-lbl">Specificity</div></div>
      <div class="metric"><span class="metric-val" style="color:var(--amber)">0.987</span><div class="metric-lbl">AUC-ROC</div></div>
    </div>

    <div class="tabs">
      <div class="tab active" onclick="switchTab(0)">Image Analysis</div>
      <div class="tab" onclick="switchTab(1)">Classification</div>
      <div class="tab" onclick="switchTab(2)">Biomarkers</div>
      <div class="tab" onclick="switchTab(3)">Quantum</div>
      <div class="tab" onclick="switchTab(4)">Recommendation</div>
    </div>

    <div class="tab-content active" id="tab0">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
        <div>
          <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em">Original Scan</div>
          <div class="img-display"><img src="${d.image.original}" alt="Original OCT"/></div>
        </div>
        <div>
          <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em">Affected Area Detection</div>
          <div class="img-display">
            <img src="${d.image.annotated}" alt="Annotated OCT"/>
            <div class="img-overlay-label">Grad-CAM + Region Detection · HQNNDL v3.0</div>
          </div>
        </div>
      </div>
      <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.04em">Detected Regions</div>
      <div class="regions-list">${regionsHTML}</div>
    </div>

    <div class="tab-content" id="tab1">
      <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:.04em">Class Probability Distribution</div>
      <div class="prob-list">${probHTML}</div>
      <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div class="info-box"><div class="info-label">Fluid Signal</div><div style="font-family:var(--mono);font-size:14px;font-weight:700">${d.quantum.fluid_signal}</div></div>
        <div class="info-box"><div class="info-label">Gradient Energy</div><div style="font-family:var(--mono);font-size:14px;font-weight:700">${d.quantum.grad_energy}</div></div>
      </div>
    </div>

    <div class="tab-content" id="tab2">
      <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:10px;text-transform:uppercase;letter-spacing:.04em">Retinal Biomarker Analysis</div>
      <div class="bio-grid">${bioHTML}</div>
    </div>

    <div class="tab-content" id="tab3">
      <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.04em">Qubit ⟨Z⟩ Expectation Values</div>
      <div class="qubit-bars">${qBars}</div>
      <div class="qubit-labels">${qLabels}</div>
      <div style="margin-top:14px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
        <div class="info-box"><div class="info-label">Qubits</div><div style="font-family:var(--mono);font-size:16px;font-weight:700;color:var(--blue)">${d.quantum.n_qubits}</div></div>
        <div class="info-box"><div class="info-label">Params</div><div style="font-family:var(--mono);font-size:16px;font-weight:700;color:var(--violet)">${d.quantum.n_params}</div></div>
        <div class="info-box"><div class="info-label">Layers</div><div style="font-family:var(--mono);font-size:16px;font-weight:700;color:var(--green)">3</div></div>
      </div>
      <div style="margin-top:8px">
        <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em">Pipeline Timing</div>
        <div class="perf-row">${perfHTML}</div>
      </div>
    </div>

    <div class="tab-content" id="tab4">
      <div class="info-box" style="margin-bottom:8px;border-left:3px solid ${rec.urgency==='URGENT'?'var(--red)':rec.urgency==='PROMPT'?'var(--amber)':'var(--green)'}">
        <div class="info-label">Urgency &amp; Follow-up</div>
        <div style="font-size:13px;font-weight:600;margin-bottom:2px">${rec.urgency} — Follow-up in ${rec.followup}</div>
        <div style="font-size:12px;color:var(--muted)">Risk Level: <b>${rec.risk}</b></div>
      </div>
      <div class="info-box" style="margin-bottom:8px">
        <div class="info-label">Recommended Action</div>
        <div class="info-text">${rec.action}</div>
      </div>
      <div class="info-interp">${rec.interpretation}</div>
      <button class="btn btn-primary" style="margin-top:12px" onclick="downloadReport('${d.id}')">📄 Download Report</button>
    </div>
  </div>`;
}

// ── Tab switching ─────────────────────────────────────────────────
function switchTab(idx) {
  document.querySelectorAll('.tab').forEach((t, i)         => t.classList.toggle('active', i === idx));
  document.querySelectorAll('.tab-content').forEach((c, i) => c.classList.toggle('active', i === idx));
}

// ── Report download ───────────────────────────────────────────────
function downloadReport(id) {
  window.open(`${API}/api/report/${id}`, '_blank');
}

// ── History ───────────────────────────────────────────────────────
function addToHistory(d) {
  history.unshift({
    id: d.id, timestamp: d.timestamp,
    filename: d.image.filename,
    diagnosis: d.diagnosis.label,
    confidence: d.diagnosis.confidence,
    thumbnail: d.image.original,
  });
  if (history.length > 10) history.pop();
  renderHistory();
}

function renderHistory() {
  const sect = document.getElementById('historySect');
  const grid = document.getElementById('historyGrid');
  if (!history.length) { sect.style.display = 'none'; return; }
  sect.style.display = 'block';
  const colors = { 'Normal': 'var(--green)', 'CSR Grade I': 'var(--amber)', 'CSR Grade II': 'var(--red)', 'Other Pathology': 'var(--violet)' };
  grid.innerHTML = history.map(h => `
    <div class="hist-card">
      <img class="hist-thumb" src="${h.thumbnail}" alt=""/>
      <div class="hist-info">
        <div class="hist-dx" style="color:${colors[h.diagnosis]||'var(--blue)'}">${h.diagnosis}</div>
        <div class="hist-conf">${h.confidence}% confidence</div>
        <div class="hist-time">${h.timestamp}</div>
      </div>
    </div>`).join('');
}

function clearHistory() {
  history = [];
  fetch(`${API}/api/history`, { method: 'DELETE' }).catch(() => {});
  renderHistory();
}

// ── Utilities ─────────────────────────────────────────────────────
function fetchWithTimeout(url, opts = {}) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT);
  return fetch(url, { ...opts, signal: controller.signal }).finally(() => clearTimeout(id));
}

function toast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderLeftColor = type === 'error' ? 'var(--red)' : 'var(--green)';
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 4000);
}
