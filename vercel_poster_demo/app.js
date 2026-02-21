const DEMO_STEPS = [
  { label: "Clean", scenario: "clean" },
  { label: "Audio Degraded", scenario: "audio_degraded" },
  { label: "Vision Degraded", scenario: "vision_degraded" },
  { label: "Radar Clutter", scenario: "radar_clutter" },
  { label: "Missing Audio", scenario: "missing_audio" }
];

const COLORS = {
  vis: "#2563eb",
  aud: "#16a34a",
  rad: "#dc2626",
  fuse: "#0f766e",
  gray: "#4b5563"
};

const state = {
  scenario: "clean",
  frame: 0,
  judgeStep: 0,
  dropVision: false,
  dropAudio: false,
  dropRadar: false,
  bestFusionMethod: "F-GATE",
  demoTimer: null
};

let dataStore = null;

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return await res.json();
}

function number(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function format3(v) {
  return number(v).toFixed(3);
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function softmax(vals) {
  const m = Math.max(...vals);
  const exps = vals.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  if (s <= 1e-8) return vals.map(() => 0);
  return exps.map((e) => e / s);
}

function getScenarioRows() {
  const rows = dataStore.predictions.filter((r) => (r.scenario || "clean") === state.scenario);
  return rows.length ? rows : dataStore.predictions;
}

function clampFrame(rows) {
  if (state.frame >= rows.length) state.frame = rows.length - 1;
  if (state.frame < 0) state.frame = 0;
}

function maskWeights(rawWeights) {
  const keep = [state.dropVision ? 0 : 1, state.dropAudio ? 0 : 1, state.dropRadar ? 0 : 1];
  const masked = rawWeights.map((w, i) => (keep[i] ? number(w, 0) : 0));
  const sum = masked[0] + masked[1] + masked[2];
  if (sum <= 1e-8) return [0, 0, 0];
  return masked.map((x) => x / sum);
}

function fusionWeights(row, pVis, pAud, pRad) {
  if (state.bestFusionMethod === "F-AVG") return maskWeights([1, 1, 1]);
  if (state.bestFusionMethod === "F-STATIC") return maskWeights([0.2, 0.45, 0.35]);
  if (state.bestFusionMethod === "F-GATE-NOQ") {
    return maskWeights(softmax([3.5 * (pVis - 0.5), 3.5 * (pAud - 0.5), 3.5 * (pRad - 0.5)]));
  }
  return maskWeights([number(row.w_vis, 0.33), number(row.w_aud, 0.33), number(row.w_rad, 0.33)]);
}

function currentScores(row) {
  const pVis = state.dropVision ? 0.5 : number(row.p_vis, 0.5);
  const pAud = state.dropAudio ? 0.5 : number(row.p_aud, 0.5);
  const pRad = state.dropRadar ? 0.5 : number(row.p_rad, 0.5);
  const w = fusionWeights(row, pVis, pAud, pRad);
  const pFuse = w[0] * pVis + w[1] * pAud + w[2] * pRad;
  return { pVis, pAud, pRad, pFuse, w };
}

function clearSVG(svgId) {
  const svg = document.getElementById(svgId);
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  return svg;
}

function svgText(x, y, text, color = "#4b5563", size = 11) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
  el.setAttribute("x", String(x));
  el.setAttribute("y", String(y));
  el.setAttribute("font-size", String(size));
  el.setAttribute("fill", color);
  el.textContent = text;
  return el;
}

function svgLine(x1, y1, x2, y2, color = "#e5e7eb", width = 1, dash = "") {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", String(x1));
  line.setAttribute("y1", String(y1));
  line.setAttribute("x2", String(x2));
  line.setAttribute("y2", String(y2));
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", String(width));
  if (dash) line.setAttribute("stroke-dasharray", dash);
  return line;
}

function drawGrid(svg, w, h, pad) {
  for (let i = 0; i <= 4; i += 1) {
    const y = pad + ((h - 2 * pad) * i) / 4;
    svg.appendChild(svgLine(pad, y, w - pad, y));
  }
}

function drawLineChart(svgId, seriesDefs, yMin = 0, yMax = 1, titleNote = "") {
  const svg = clearSVG(svgId);
  const w = 760;
  const h = 260;
  const pad = 34;
  drawGrid(svg, w, h, pad);

  const maxLen = Math.max(...seriesDefs.map((s) => s.values.length), 1);
  const xScale = (idx) => pad + ((w - 2 * pad) * idx) / Math.max(1, maxLen - 1);
  const yScale = (v) => {
    const clamped = Math.min(yMax, Math.max(yMin, number(v, yMin)));
    const ratio = (clamped - yMin) / Math.max(1e-8, yMax - yMin);
    return h - pad - ratio * (h - 2 * pad);
  };

  svg.appendChild(svgText(8, 16, yMax.toFixed(1)));
  svg.appendChild(svgText(8, h - pad + 4, yMin.toFixed(1)));

  for (const s of seriesDefs) {
    const points = s.values.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", s.color);
    poly.setAttribute("stroke-width", "2");
    poly.setAttribute("points", points);
    svg.appendChild(poly);
  }

  seriesDefs.forEach((s, i) => {
    const y = 16 + i * 14;
    svg.appendChild(svgLine(w - 220, y, w - 198, y, s.color, 3));
    svg.appendChild(svgText(w - 194, y + 4, s.label));
  });
  if (titleNote) svg.appendChild(svgText(w - 220, h - 8, titleNote, "#6b7280", 10));
}

function drawBarChart(svgId, labels, values, color = COLORS.gray, yMax = 1) {
  const svg = clearSVG(svgId);
  const w = 760;
  const h = 260;
  const pad = 34;
  drawGrid(svg, w, h, pad);
  const n = labels.length;
  const band = (w - 2 * pad) / Math.max(1, n);
  for (let i = 0; i < n; i += 1) {
    const x = pad + i * band + band * 0.14;
    const bw = band * 0.72;
    const v = Math.min(yMax, Math.max(0, number(values[i], 0)));
    const bh = (h - 2 * pad) * (v / Math.max(1e-8, yMax));
    const y = h - pad - bh;

    const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bar.setAttribute("x", String(x));
    bar.setAttribute("y", String(y));
    bar.setAttribute("width", String(bw));
    bar.setAttribute("height", String(bh));
    bar.setAttribute("fill", color);
    svg.appendChild(bar);
    svg.appendChild(svgText(x, h - pad + 14, labels[i]));
    svg.appendChild(svgText(x + 2, y - 4, v.toFixed(3)));
  }
}

function drawSignalBars(svgId, values, color = COLORS.rad, yMin = 0, yMax = 1, markerIndex = -1) {
  const svg = clearSVG(svgId);
  const w = 760;
  const h = 260;
  const pad = 34;
  drawGrid(svg, w, h, pad);
  const n = values.length;
  const band = (w - 2 * pad) / Math.max(1, n);
  for (let i = 0; i < n; i += 1) {
    const x = pad + i * band + 1;
    const bw = Math.max(1, band - 2);
    const v = Math.min(yMax, Math.max(yMin, number(values[i], 0)));
    const ratio = (v - yMin) / Math.max(1e-8, yMax - yMin);
    const bh = ratio * (h - 2 * pad);
    const y = h - pad - bh;
    const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bar.setAttribute("x", String(x));
    bar.setAttribute("y", String(y));
    bar.setAttribute("width", String(bw));
    bar.setAttribute("height", String(Math.max(1, bh)));
    bar.setAttribute("fill", color);
    svg.appendChild(bar);
  }
  if (markerIndex >= 0) {
    const x = pad + ((w - 2 * pad) * markerIndex) / Math.max(1, n - 1);
    svg.appendChild(svgLine(x, pad, x, h - pad, "#111827", 1.5));
  }
}

function renderScenarioButtons() {
  const wrap = document.getElementById("scenarioButtons");
  wrap.innerHTML = "";
  const scenarios = [...new Set(dataStore.predictions.map((r) => r.scenario || "clean"))];
  scenarios.forEach((scenario) => {
    const btn = document.createElement("button");
    btn.textContent = scenario.replaceAll("_", " ");
    if (scenario === state.scenario) btn.classList.add("active");
    btn.onclick = () => {
      stopAutoDemo();
      state.scenario = scenario;
      state.frame = 0;
      state.judgeStep = Math.max(0, DEMO_STEPS.findIndex((s) => s.scenario === scenario));
      setText("demoStatus", "Scenario selected.");
      render();
    };
    wrap.appendChild(btn);
  });
}

function renderWalkthroughText() {
  const parts = DEMO_STEPS.map((s, i) => (i === state.judgeStep ? `[${s.label}]` : s.label));
  setText("walkthroughSteps", parts.join(" -> "));
}

function renderLatencyTable() {
  const tbody = document.querySelector("#latencyTable tbody");
  tbody.innerHTML = "";
  const methods = dataStore.latency.methods || {};
  Object.keys(methods)
    .sort()
    .forEach((name) => {
      const wall = methods[name].wall_clock || {};
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${name}</td><td>${number(wall.mean_ms).toFixed(4)}</td><td>${number(wall.p95_ms).toFixed(4)}</td>`;
      tbody.appendChild(tr);
    });
}

function renderStaticCharts() {
  const allMetrics = [
    ...dataStore.metricsUnimodal.map((r) => ({ id: r.run_id, pr: number(r.pr_auc) })),
    ...dataStore.metricsFusion.map((r) => ({ id: r.run_id, pr: number(r.pr_auc) }))
  ].sort((a, b) => b.pr - a.pr);
  drawBarChart(
    "metricsChart",
    allMetrics.map((m) => m.id),
    allMetrics.map((m) => m.pr),
    COLORS.gray,
    1
  );

  const stress = dataStore.stressAudio
    .filter((r) => r.run_id === state.bestFusionMethod)
    .sort((a, b) => number(a.snr_db) - number(b.snr_db));
  drawLineChart(
    "stressChart",
    [{ label: `${state.bestFusionMethod} PR-AUC`, color: COLORS.fuse, values: stress.map((r) => number(r.pr_auc)) }],
    0.5,
    1,
    "audio SNR sweep"
  );
  setText("stressTitle", `Robustness Snapshot (${state.bestFusionMethod})`);
}

function initBestFusionMethod() {
  const best = [...dataStore.metricsFusion]
    .filter((r) => (r.split || "test") === "test")
    .sort((a, b) => number(b.pr_auc) - number(a.pr_auc))[0];
  state.bestFusionMethod = best ? best.run_id : "F-GATE";
}

function updateFusionSummary() {
  const bestFusion = [...dataStore.metricsFusion]
    .filter((r) => (r.split || "test") === "test")
    .sort((a, b) => number(b.pr_auc) - number(a.pr_auc))[0];
  const bestUni = [...dataStore.metricsUnimodal]
    .filter((r) => (r.split || "test") === "test")
    .sort((a, b) => number(b.pr_auc) - number(a.pr_auc))[0];
  const gain = number(bestFusion.pr_auc) - number(bestUni.pr_auc);
  setText("fusionLabel", `${state.bestFusionMethod} (current frame)`);
  setText("fusionSummary", "This panel emphasizes the top fusion model selected by TEST PR-AUC.");
  setText("bestFusionStat", `${bestFusion.run_id} (${format3(bestFusion.pr_auc)})`);
  setText("bestUnimodalStat", `${bestUni.run_id} (${format3(bestUni.pr_auc)})`);
  setText("fusionGainStat", `+${format3(gain)}`);
}

function updateFrameImage(row) {
  const img = document.getElementById("frameImage");
  if (!img) return;
  const frameFiles = Array.isArray(dataStore?.manifest?.frame_files) ? dataStore.manifest.frame_files : [];
  if (!frameFiles.length) {
    img.src = "./assets/sample_frame.png";
    img.style.filter = "none";
    setText("frameCaption", "Static sample frame (no frame sequence found).");
    return;
  }
  const srcIdx = Math.max(0, Math.floor(number(row?.frame_idx, state.frame)));
  const file = frameFiles[srcIdx % frameFiles.length];
  img.src = `./assets/frames/${file}`;
  img.onerror = () => {
    img.onerror = null;
    img.src = "./assets/sample_frame.png";
  };

  if (state.dropVision) img.style.filter = "grayscale(1) blur(2.5px) opacity(0.45)";
  else if (state.scenario === "vision_degraded") img.style.filter = "blur(2px) brightness(0.65) contrast(0.85)";
  else img.style.filter = "none";

  setText("frameCaption", `Frame ${String(srcIdx).padStart(3, "0")} | scenario=${state.scenario.replaceAll("_", " ")}`);
}

function renderAudioPanel(row) {
  const base = dataStore.audioFeatures || [];
  const currentT = number(row.t_sec, number(row.frame_idx, 0) * 0.125);
  const snr = number(row.audio_snr_db, 30);
  const noiseScale = Math.max(0, Math.min(1, (30 - snr) / 40));

  const values = base.map((p, i) => {
    const t = number(p.t_sec, i / 10);
    let amp = number(p.amplitude, 0);
    amp = amp * (1 - noiseScale * 0.55) + noiseScale * 0.25 * Math.sin(i * 0.67 + 1.2);
    if (state.dropAudio) amp = 0;
    return { t, v: Math.max(-1, Math.min(1, amp)) };
  });

  drawLineChart(
    "audioChart",
    [{ label: "audio waveform", color: COLORS.aud, values: values.map((x) => x.v) }],
    -1,
    1,
    `SNR ${snr.toFixed(1)} dB`
  );

  // marker line at current time
  const svg = document.getElementById("audioChart");
  const w = 760;
  const h = 260;
  const pad = 34;
  const idx = Math.min(values.length - 1, Math.max(0, Math.floor((currentT / Math.max(0.1, values.at(-1)?.t || 1)) * values.length)));
  const x = pad + ((w - 2 * pad) * idx) / Math.max(1, values.length - 1);
  svg.appendChild(svgLine(x, pad, x, h - pad, "#111827", 1.5, "4,4"));
  setText(
    "audioCaption",
    state.dropAudio
      ? "Audio dropped -> waveform suppressed and gate reallocates weight."
      : `Audio waveform with replayed degradation tied to scenario controls.`
  );
}

function renderRadarPanel(row) {
  const radarRows = dataStore.radarFeatures || [];
  if (!radarRows.length) return;
  const idx = Math.max(0, Math.floor(number(row.frame_idx, state.frame))) % radarRows.length;
  const radarRow = radarRows[idx];
  const clutter = number(row.radar_clutter_level, 0);
  const binKeys = Object.keys(radarRow).filter((k) => k.startsWith("bin_")).sort();
  const values = binKeys.map((k, i) => {
    let v = number(radarRow[k], 0.5);
    const clutterBoost = (state.scenario === "radar_clutter" ? 1 : 0.35) * clutter;
    v = v * (1 - clutterBoost * 0.35) + clutterBoost * 0.45 * (0.5 + 0.5 * Math.sin(i * 1.3 + idx * 0.07));
    if (state.dropRadar) v = 0;
    return Math.max(0, Math.min(1, v));
  });
  drawSignalBars("radarChart", values, COLORS.rad, 0, 1);
  setText(
    "radarCaption",
    state.dropRadar
      ? "Radar dropped -> bins set to zero and gate reallocates weight."
      : `Radar bin intensities with clutter level ${clutter.toFixed(2)}.`
  );
}

function applyStep(stepIdx) {
  const step = DEMO_STEPS[stepIdx];
  state.judgeStep = stepIdx;
  state.scenario = step.scenario;
  state.frame = 0;
  state.dropVision = step.scenario === "missing_vision";
  state.dropAudio = step.scenario === "missing_audio";
  state.dropRadar = step.scenario === "missing_radar";
  document.getElementById("dropVision").checked = state.dropVision;
  document.getElementById("dropAudio").checked = state.dropAudio;
  document.getElementById("dropRadar").checked = state.dropRadar;
}

function stopAutoDemo() {
  if (state.demoTimer) {
    clearInterval(state.demoTimer);
    state.demoTimer = null;
  }
}

function startAutoDemo() {
  stopAutoDemo();
  let idx = 0;
  applyStep(idx);
  setText("demoStatus", "Walkthrough running...");
  render();
  state.demoTimer = setInterval(() => {
    idx += 1;
    if (idx >= DEMO_STEPS.length) {
      stopAutoDemo();
      setText("demoStatus", "Walkthrough complete.");
      return;
    }
    applyStep(idx);
    render();
  }, 2600);
}

function render() {
  const rows = getScenarioRows();
  clampFrame(rows);
  const row = rows[state.frame];
  const scores = currentScores(row);

  const frameSlider = document.getElementById("frameSlider");
  frameSlider.max = String(Math.max(0, rows.length - 1));
  frameSlider.value = String(state.frame);
  setText("frameLabel", `Frame ${state.frame + 1} / ${rows.length}`);
  setText("judgeStepLabel", `Step ${state.judgeStep + 1}/${DEMO_STEPS.length}: ${DEMO_STEPS[state.judgeStep].label}`);

  setText("scoreFusion", format3(scores.pFuse));
  setText("qSNR", number(row.audio_snr_db, 30).toFixed(1));
  setText("qVision", row.vision_condition || "normal");
  setText("qRadar", number(row.radar_clutter_level, 0).toFixed(2));

  updateFrameImage(row);
  renderAudioPanel(row);
  renderRadarPanel(row);

  const seriesRows = rows.map((r) => {
    const s = currentScores(r);
    return { w_vis: s.w[0], w_aud: s.w[1], w_rad: s.w[2], p_fuse: s.pFuse };
  });

  drawLineChart(
    "gateChart",
    [
      { label: "w_vis", color: COLORS.vis, values: seriesRows.map((r) => r.w_vis) },
      { label: "w_aud", color: COLORS.aud, values: seriesRows.map((r) => r.w_aud) },
      { label: "w_rad", color: COLORS.rad, values: seriesRows.map((r) => r.w_rad) }
    ],
    0,
    1
  );

  drawLineChart(
    "confChart",
    [{ label: `${state.bestFusionMethod} confidence`, color: COLORS.fuse, values: seriesRows.map((r) => r.p_fuse) }],
    0,
    1
  );

  renderScenarioButtons();
  renderWalkthroughText();
}

function bindUI() {
  document.getElementById("btnStart").onclick = () => startAutoDemo();
  document.getElementById("btnNext").onclick = () => {
    stopAutoDemo();
    state.judgeStep = (state.judgeStep + 1) % DEMO_STEPS.length;
    applyStep(state.judgeStep);
    setText("demoStatus", "Manual scenario step.");
    render();
  };
  document.getElementById("btnStop").onclick = () => {
    stopAutoDemo();
    setText("demoStatus", "Paused.");
  };

  document.getElementById("frameSlider").oninput = (e) => {
    stopAutoDemo();
    state.frame = Number(e.target.value || 0);
    setText("demoStatus", "Manual frame selection.");
    render();
  };
  document.getElementById("dropVision").onchange = (e) => {
    stopAutoDemo();
    state.dropVision = Boolean(e.target.checked);
    render();
  };
  document.getElementById("dropAudio").onchange = (e) => {
    stopAutoDemo();
    state.dropAudio = Boolean(e.target.checked);
    render();
  };
  document.getElementById("dropRadar").onchange = (e) => {
    stopAutoDemo();
    state.dropRadar = Boolean(e.target.checked);
    render();
  };
}

async function main() {
  dataStore = {
    predictions: await loadJSON("./data/predictions.json"),
    metricsFusion: await loadJSON("./data/metrics_fusion.json"),
    metricsUnimodal: await loadJSON("./data/metrics_unimodal.json"),
    missing: await loadJSON("./data/missing_modality.json"),
    stressAudio: await loadJSON("./data/stress_audio_snr.json"),
    stressVision: await loadJSON("./data/stress_vision_corruptions.json"),
    stressRadar: await loadJSON("./data/stress_radar_clutter.json"),
    latency: await loadJSON("./data/latency_summary.json"),
    audioFeatures: await loadJSON("./data/audio_features.json"),
    radarFeatures: await loadJSON("./data/radar_features.json"),
    manifest: await loadJSON("./data/manifest.json")
  };

  const scenarios = [...new Set(dataStore.predictions.map((r) => r.scenario || "clean"))];
  state.scenario = scenarios.includes("clean") ? "clean" : scenarios[0];
  initBestFusionMethod();
  bindUI();
  renderLatencyTable();
  renderStaticCharts();
  updateFusionSummary();
  render();
  setText("demoStatus", "Ready.");
  setText(
    "dataManifest",
    `data files: ${dataStore.manifest.data_files.length}, assets: ${dataStore.manifest.asset_files.length}, frames: ${number(dataStore.manifest.frame_count, 0)}`
  );
}

main().catch((err) => {
  console.error(err);
  document.body.innerHTML = `<pre style="padding:1rem;color:#b91c1c">Failed to load demo data:\n${String(err)}</pre>`;
});

