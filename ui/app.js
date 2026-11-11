// ===========================================================================
// Biometric Verification System - UI logic
//
// - Sends the claimed ID + face image + voice sample to the backend
//   (POST {API_BASE}/api/verify, multipart: claimed_id, face, voice).
// - If no backend is running, DEMO MODE replays a real Session-3 test trial
//   from results/hierarchical/hierarchical_trial_decisions.csv (clearly labeled).
// - Shows the real evaluation results from the results/ folder.
// ===========================================================================
const API_BASE = "http://localhost:8000";      // backend address
const API = API_BASE + "/api/verify";
const RESULTS = "../results/hierarchical/";      // works when served from the project root

const $ = (id) => document.getElementById(id);
// Small rates get 2 decimals so close values (e.g. 0.80% vs 0.83%) don't look identical.
const fmtPct = (x, d) => (x * 100).toFixed(d ?? (Math.abs(x) < 0.1 ? 2 : 1)) + "%";

// ---------- snapshot of results (used if the CSV/JSON files can't be loaded, e.g. file://) ----------
const SNAPSHOT = {
  thresholds: { low_face_threshold: 0.050224, high_face_threshold: 0.171650, voice_threshold: 0.977003 },
  stages: { total_trials: 3698, face_direct_accept: 124, face_direct_reject: 3430, voice_escalations: 144, voice_escalation_rate: 0.03894 },
  comparison: [
    { System: "Face SVM", Accuracy: 0.98594, Precision: 0.68085, Recall: 0.74419, F1: 0.71111, FAR: 0.008306, FRR: 0.255814, "ROC-AUC": 0.96565, TN: 3582, FP: 30, FN: 22, TP: 64 },
    { System: "Voice SVM", Accuracy: 0.97215, Precision: 0.29268, Recall: 0.13953, F1: 0.18898, FAR: 0.008029, FRR: 0.860465, "ROC-AUC": 0.80095, TN: 3583, FP: 29, FN: 74, TP: 12 },
    { System: "Hierarchical Face-Voice", Accuracy: 0.97999, Precision: 0.54839, Recall: 0.79070, F1: 0.64762, FAR: 0.015504, FRR: 0.209302, "ROC-AUC": null, TN: 3556, FP: 56, FN: 18, TP: 68 },
  ],
};

const state = {
  thresholds: SNAPSHOT.thresholds,
  stages: SNAPSHOT.stages,
  comparison: SNAPSHOT.comparison,
  trials: [],           // rows from hierarchical_trial_decisions.csv
  backend: false,       // true when the Python backend (backend/server.py) answers
  health: null,         // backend /api/health response
  face: null,           // File/Blob shown in the preview
  faceFrames: [],       // what gets sent: one photo, or a burst of camera frames
  voice: null,          // Blob
  enrollPhotos: [],     // photos for "Enroll a new person"
};

// ===========================================================================
// Small helpers
// ===========================================================================
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 4500);
}

function parseCSV(text) {
  const [head, ...lines] = text.trim().split(/\r?\n/);
  const cols = head.split(",");
  return lines.map((line) => {
    const cells = line.split(",");
    const row = {};
    cols.forEach((c, i) => {
      const v = cells[i];
      row[c] = v === "" || v === undefined ? null : isNaN(v) ? v : Number(v);
    });
    return row;
  });
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status);
  return r.text();
}

function withTimeout(ms) {
  const c = new AbortController();
  setTimeout(() => c.abort(), ms);
  return c.signal;
}

// Tooltip shared by all charts
const tip = $("tip");
function bindTip(el, html) {
  el.addEventListener("mousemove", (e) => {
    tip.innerHTML = html;
    tip.hidden = false;
    const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
    const y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  });
  el.addEventListener("mouseleave", () => (tip.hidden = true));
}

// ===========================================================================
// Load data + backend status
// ===========================================================================
async function loadData() {
  const jobs = [
    fetchText(RESULTS + "hierarchical_trial_decisions.csv").then((t) => (state.trials = parseCSV(t))),
    fetchText(RESULTS + "hierarchical_comparison.csv").then((t) => (state.comparison = parseCSV(t))),
    fetchText(RESULTS + "selected_thresholds.json").then((t) => (state.thresholds = JSON.parse(t))),
    fetchText(RESULTS + "hierarchical_stage_summary.json").then((t) => (state.stages = JSON.parse(t))),
  ];
  await Promise.allSettled(jobs);
}

async function checkBackend() {
  const el = $("status");
  try {
    const r = await fetch(API_BASE + "/api/health", { signal: withTimeout(2500) });
    if (!r.ok) throw new Error();
    state.health = await r.json();
    state.backend = true;
    el.className = "status online";
    $("statusText").textContent = "Face model online";
    el.title = `Connected to ${API_BASE} · ${state.health.model} · threshold ${state.health.threshold.toFixed(3)} (FAR ${state.health.far_target * 100}%)`;
    $("enrollPanel").hidden = false;
  } catch {
    state.backend = false;
    el.className = "status demo";
    $("statusText").textContent = "Demo mode";
    el.title = "No backend at " + API_BASE + " - results replay recorded test trials.";
  }
}

// ===========================================================================
// Identity picker
// ===========================================================================
let enrolledIds = new Set();

async function fillIdentities(selectId) {
  const sel = $("claimed");
  const previous = selectId || sel.value;
  let vid = [...new Set(state.trials.map((r) => r.claimed_speaker))].filter(Boolean).sort();
  let enrolled = [];
  if (state.backend) {
    try {
      const r = await fetch(API_BASE + "/api/identities");
      const data = await r.json();
      enrolled = data.enrolled;
      if (data.vidtimit.length) vid = data.vidtimit;
    } catch { /* keep local list */ }
  }
  if (!vid.length) vid = Array.from({ length: 43 }, (_, i) => "person" + String(i + 1).padStart(2, "0"));
  enrolledIds = new Set(enrolled.map((e) => e.id));

  sel.innerHTML = "";
  if (enrolled.length) {
    const g = document.createElement("optgroup");
    g.label = "Enrolled people";
    enrolled.forEach((e) => g.append(new Option(e.id, e.id)));
    sel.append(g);
  }
  const g2 = document.createElement("optgroup");
  g2.label = "VidTIMIT speakers";
  vid.forEach((id) => g2.append(new Option(id, id)));
  sel.append(g2);
  if ([...sel.options].some((o) => o.value === previous)) sel.value = previous;
  $("sSpeakers").textContent = vid.length;
  renderEnrolledList(enrolled);
  updateIdCard();
}

function updateIdCard() {
  const id = $("claimed").value;
  $("avatar").textContent = id.slice(0, 1).toUpperCase();
  $("idName").textContent = id;
  if (enrolledIds.has(id)) {
    $("idMeta").textContent = "Enrolled person";
  } else {
    const g = id[0] === "f" ? "female" : id[0] === "m" ? "male" : "";
    $("idMeta").textContent = "VidTIMIT speaker" + (g ? " · " + g : "");
  }
}
$("claimed").addEventListener("change", updateIdCard);

// ===========================================================================
// Face input (upload, drag & drop, camera)
// ===========================================================================
const drop = $("drop");
let camStream = null;

function setFace(file) {
  if (!file || !file.type.startsWith("image/")) return toast("Please choose an image file.");
  state.face = file;
  if (!setFace.keepFrames) state.faceFrames = [file];
  stopCamera();
  const img = $("faceImg");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  drop.classList.add("has-media", "scanning");
  setTimeout(() => drop.classList.remove("scanning"), 1400);
  refresh();
}

drop.addEventListener("click", () => { if (!camStream) $("faceFile").click(); });
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("faceFile").click(); }
});
$("faceFile").addEventListener("change", (e) => setFace(e.target.files[0]));
["dragover", "dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.toggle("over", ev === "dragover");
    if (ev === "drop" && e.dataTransfer.files[0]) setFace(e.dataTransfer.files[0]);
  })
);

$("camBtn").addEventListener("click", async () => {
  if (camStream) return stopCamera();
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
  } catch {
    return toast("Couldn't open the camera. Check that it's connected and allowed in the browser.");
  }
  const v = $("video");
  v.srcObject = camStream;
  v.hidden = false;
  $("faceImg").hidden = true;
  drop.classList.add("has-media", "scanning");
  $("snapBtn").hidden = false;
  $("camBtn").textContent = "✕ Close camera";
});

// Grab one JPEG frame from a <video>.
function grabFrame(v, name) {
  const c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  return new Promise((res) => c.toBlob((b) => res(new File([b], name, { type: "image/jpeg" })), "image/jpeg", 0.92));
}

// Capture a short burst (like the video frames the model was trained on).
$("snapBtn").addEventListener("click", async () => {
  const v = $("video");
  const btn = $("snapBtn");
  btn.disabled = true;
  const frames = [];
  for (let i = 0; i < 10; i++) {
    btn.textContent = `Hold still… ${10 - i}`;
    frames.push(await grabFrame(v, `frame${i}.jpg`));
    await new Promise((r) => setTimeout(r, 120));
  }
  btn.disabled = false;
  btn.textContent = "● Capture";
  state.faceFrames = frames;
  setFace.keepFrames = true;
  setFace(frames[0]);
  setFace.keepFrames = false;
});

function stopCamera() {
  if (camStream) camStream.getTracks().forEach((t) => t.stop());
  camStream = null;
  $("video").hidden = true;
  $("snapBtn").hidden = true;
  $("camBtn").textContent = "📷 Use camera";
  drop.classList.remove("scanning");
  if (!state.face) drop.classList.remove("has-media");
  else $("faceImg").hidden = false;
}

// ===========================================================================
// Voice input (record with live waveform, or upload)
// ===========================================================================
const canvas = $("wave");
const ctx2d = canvas.getContext("2d");
let recorder = null, analyser = null, audioCtx = null, rafId = 0, recStart = 0, staticPeaks = null;

function sizeCanvas() {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * devicePixelRatio));
  canvas.height = Math.round(110 * devicePixelRatio);
}

// Draw vertical bars from an array of 0..1 levels.
function drawBars(levels, color) {
  const w = canvas.width, h = canvas.height, n = levels.length;
  const gap = 3 * devicePixelRatio, bw = Math.max(2, (w - gap * (n - 1)) / n);
  ctx2d.clearRect(0, 0, w, h);
  levels.forEach((lv, i) => {
    const bh = Math.max(3 * devicePixelRatio, lv * h * 0.9);
    const x = i * (bw + gap), y = (h - bh) / 2;
    ctx2d.fillStyle = color;
    ctx2d.beginPath();
    ctx2d.roundRect ? ctx2d.roundRect(x, y, bw, bh, bw / 2) : ctx2d.rect(x, y, bw, bh);
    ctx2d.fill();
  });
}

// Waveform colors come from the CSS theme (--wave-idle / --wave-live / --wave-done).
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawIdle() {
  if (staticPeaks) return drawBars(staticPeaks, cssVar("--wave-done"));
  const n = 48, t = performance.now() / 900;
  drawBars(Array.from({ length: n }, (_, i) => 0.08 + 0.06 * Math.sin(i * 0.5 + t)), cssVar("--wave-idle"));
  rafId = requestAnimationFrame(drawIdle);
}

function drawLive() {
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  const n = 48, step = Math.floor(data.length / n / 1.6) || 1;
  const levels = Array.from({ length: n }, (_, i) => Math.min(1, (data[i * step] / 255) * 1.3));
  drawBars(levels, cssVar("--wave-live"));
  const s = Math.floor((performance.now() - recStart) / 1000);
  $("timer").textContent = `00:${String(s).padStart(2, "0")}`;
  rafId = requestAnimationFrame(drawLive);
}

// Show the real waveform of a recorded / uploaded clip.
async function drawStaticWave(blob) {
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ac.decodeAudioData(await blob.arrayBuffer());
    const ch = buf.getChannelData(0), n = 48, size = Math.floor(ch.length / n);
    const peaks = Array.from({ length: n }, (_, i) => {
      let m = 0;
      for (let j = i * size; j < (i + 1) * size; j += 16) m = Math.max(m, Math.abs(ch[j] || 0));
      return m;
    });
    const max = Math.max(...peaks, 0.01);
    staticPeaks = peaks.map((p) => Math.max(0.05, p / max));
    $("timer").textContent = `00:${String(Math.round(buf.duration)).padStart(2, "0")}`;
    ac.close();
  } catch {
    staticPeaks = Array.from({ length: 48 }, () => 0.15 + Math.random() * 0.5);
  }
  cancelAnimationFrame(rafId);
  drawIdle();
}

function setVoice(blob) {
  if (!blob) return;
  state.voice = blob;
  const p = $("player");
  p.src = URL.createObjectURL(blob);
  p.hidden = false;
  drawStaticWave(blob);
  refresh();
}

$("audBtn").addEventListener("click", () => $("audFile").click());
$("audFile").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f && !f.type.startsWith("audio/")) return toast("Please choose an audio file.");
  setVoice(f);
});

$("recBtn").addEventListener("click", async () => {
  if (recorder && recorder.state === "recording") return recorder.stop();
  let mic;
  try {
    mic = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    return toast("Couldn't use the microphone. Check that it's connected and allowed in the browser.");
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(mic).connect(analyser);

  const chunks = [];
  recorder = new MediaRecorder(mic);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    mic.getTracks().forEach((t) => t.stop());
    audioCtx.close();
    cancelAnimationFrame(rafId);
    $("recBtn").classList.remove("on");
    $("recBtn").textContent = "● Record again";
    canvas.parentElement.classList.remove("live");
    setVoice(new Blob(chunks, { type: recorder.mimeType }));
  };
  recorder.start();
  recStart = performance.now();
  staticPeaks = null;
  cancelAnimationFrame(rafId);
  drawLive();
  $("recBtn").classList.add("on");
  $("recBtn").textContent = "■ Stop";
  canvas.parentElement.classList.add("live");
});

// ===========================================================================
// Progress + verify
// ===========================================================================
function refresh() {
  $("stepFace").classList.toggle("done", !!state.face);
  $("stepVoice").classList.toggle("done", !!state.voice);
  // The real backend verifies faces; voice is optional there.
  const needVoice = !state.backend;
  const ready = !!(state.face && (state.voice || !needVoice));
  $("verify").disabled = !ready;
  const missing = [!state.face && "a face image", needVoice && !state.voice && "a voice sample"].filter(Boolean);
  $("needs").textContent = ready
    ? (state.backend && !state.voice ? "Ready - press Verify (face verification; voice is optional)." : "Ready - press Verify.")
    : `Add ${missing.join(" and ")} to continue.`;
}

$("verify").addEventListener("click", async () => {
  const btn = $("verify");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.querySelector(".go-label").textContent = "Analyzing…";
  drop.classList.add("scanning");
  const started = performance.now();

  let data = null;
  if (state.backend) {
    try {
      const fd = new FormData();
      fd.append("claimed_id", $("claimed").value);
      state.faceFrames.forEach((f, i) => fd.append("face", f, f.name || `face${i}.jpg`));
      if (state.voice) fd.append("voice", state.voice, "voice.webm");
      const r = await fetch(API, { method: "POST", body: fd, signal: withTimeout(30000) });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw Object.assign(new Error(body.detail || "HTTP " + r.status), { shown: r.status === 422 || r.status === 404 });
      data = normalize(body);
    } catch (err) {
      drop.classList.remove("scanning");
      btn.classList.remove("busy");
      btn.querySelector(".go-label").textContent = "Verify identity";
      refresh();
      // A real answer like "no face found" should not fall back to a fake demo result.
      return toast(err.shown ? err.message : "The backend didn't answer (" + err.message + ").");
    }
  }
  if (!data) data = demoResult($("claimed").value);

  // Keep the scan animation visible for a moment.
  await new Promise((r) => setTimeout(r, Math.max(0, 1400 - (performance.now() - started))));
  drop.classList.remove("scanning");
  btn.classList.remove("busy");
  btn.querySelector(".go-label").textContent = "Verify again";
  refresh();
  showResult(data);
});

// Turn a backend response into the shape the UI uses.
// Expected: { accepted, score, threshold, stage, face_score?, voice_score? }
function normalize(d) {
  const escalated = /voice/i.test(String(d.stage || ""));
  return {
    accepted: !!d.accepted,
    escalated,
    faceScore: d.face_score ?? (!escalated ? d.score : null),
    voiceScore: d.voice_score ?? (escalated ? d.score : null),
    score: Number(d.score ?? 0),
    threshold: d.threshold,
    farTarget: d.far_target,
    preview: d.face_preview,
    framesUsed: d.frames_used,
    demo: false,
  };
}

// Demo: replay a real Session-3 trial for this claimed identity.
function demoResult(claimed) {
  const rows = state.trials.filter((r) => r.claimed_speaker === claimed);
  if (!rows.length) {
    const s = Math.random();
    return { accepted: s > 0.5, escalated: false, faceScore: s, voiceScore: null, score: s, demo: true, trial: null };
  }
  const genuine = rows.filter((r) => r.y_true === 1);
  const impostor = rows.filter((r) => r.y_true === 0);
  const pool = Math.random() < 0.5 && genuine.length ? genuine : impostor.length ? impostor : rows;
  const t = pool[Math.floor(Math.random() * pool.length)];
  const escalated = String(t.stage).startsWith("voice");
  return {
    accepted: t.decision === 1,
    escalated,
    faceScore: t.face_score,
    voiceScore: escalated ? t.voice_score : null,
    score: escalated ? t.voice_score : t.face_score,
    demo: true,
    trial: t,
  };
}

// Face scores are tiny near the thresholds, so the face gauge uses a square-root scale.
const faceScale = (s) => Math.sqrt(Math.max(0, Math.min(1, s)));

// Hierarchical mode: reject | escalate | accept. Real-backend (face only): reject | accept at one threshold.
function setupFaceGauge(single) {
  const [z1, z2, z3] = $("gFace").querySelectorAll(".zone");
  const legend = $("gFace").nextElementSibling.children;
  if (single != null) {
    const t = faceScale(single);
    z1.style.flex = t; z2.style.flex = 0; z2.hidden = true; z3.style.flex = 1 - t;
    legend[1].textContent = `threshold ${single.toFixed(3)}`;
  } else {
    const lo = faceScale(state.thresholds.low_face_threshold);
    const hi = faceScale(state.thresholds.high_face_threshold);
    z1.style.flex = lo; z2.style.flex = hi - lo; z2.hidden = false; z3.style.flex = 1 - hi;
    legend[1].textContent = "escalate to voice";
  }
  $("gVoice").querySelector(".mark").style.left = state.thresholds.voice_threshold * 100 + "%";
}

function showResult(d) {
  const T = state.thresholds;
  const res = $("result");
  res.hidden = false;
  res.style.animation = "none";
  void res.offsetWidth;
  res.style.animation = "";
  res.style.setProperty("--c", d.accepted ? "var(--ok)" : "var(--bad)");
  const single = !d.demo && !d.escalated && d.threshold != null ? Number(d.threshold) : null;
  setupFaceGauge(single);
  $("cropFig").hidden = !d.preview;
  if (d.preview) $("cropImg").src = d.preview;

  // reset pipeline
  const needle = $("gFace").querySelector(".needle");
  const fill = $("gVoice").querySelector(".fill");
  needle.style.left = "0%";
  fill.style.width = "0%";
  ["stFace", "stVoice"].forEach((id) => $(id).classList.remove("active", "skipped"));
  $("conn").classList.remove("on");
  $("verdict").textContent = "";
  $("meta").innerHTML = "";
  $("pct").textContent = "0%";
  const arc = $("arc");
  arc.style.transition = "none";
  arc.style.strokeDashoffset = 540.35;

  // Stage 1: face
  const fs = d.faceScore;
  $("stFace").classList.add("active");
  $("stFaceOut").textContent = fs == null ? "—" : "score " + fs.toFixed(3);
  requestAnimationFrame(() => (needle.style.left = `calc(${faceScale(fs ?? 0) * 100}% - 2px)`));
  $("stFaceNote").textContent =
    fs == null ? "" :
    single != null ? (d.accepted
      ? `At or above ${single.toFixed(3)} (threshold for ${(d.farTarget ?? 0.01) * 100}% false accepts) → same person.`
      : `Below ${single.toFixed(3)} (threshold for ${(d.farTarget ?? 0.01) * 100}% false accepts) → not the same person.`) +
      (d.framesUsed > 1 ? ` Used ${d.framesUsed} camera frames.` : "") :
    d.escalated ? `Between ${T.low_face_threshold.toFixed(3)} and ${T.high_face_threshold.toFixed(3)} → not sure, so the voice is checked.` :
    d.accepted ? `Above ${T.high_face_threshold.toFixed(3)} → the face alone is enough to accept.` :
    `Below ${T.low_face_threshold.toFixed(3)} → the face alone is enough to reject.`;

  // Stage 2: voice (only if escalated)
  setTimeout(() => {
    if (d.escalated) {
      $("conn").classList.add("on");
      $("stVoice").classList.add("active");
      $("stVoiceOut").textContent = d.voiceScore == null ? "—" : "score " + d.voiceScore.toFixed(3);
      fill.style.width = (d.voiceScore ?? 0) * 100 + "%";
      $("stVoiceNote").textContent = `Needs ≥ ${T.voice_threshold.toFixed(3)} to accept → ${d.accepted ? "accepted" : "rejected"}.`;
    } else {
      $("stVoice").classList.add("skipped");
      $("stVoiceOut").textContent = "skipped";
      $("stVoiceNote").textContent = single != null
        ? "The live backend verifies faces only (voice stage not connected yet)."
        : "Not needed - the face stage was confident.";
    }
  }, 700);

  // Verdict + ring
  setTimeout(() => {
    const v = $("verdict");
    v.className = "verdict " + (d.accepted ? "ok" : "bad");
    v.textContent = d.accepted ? "✔ Identity verified" : "✖ Access denied";
    const pct = Math.round(Math.max(0, Math.min(1, d.score)) * 100);
    arc.style.transition = "";
    requestAnimationFrame(() => (arc.style.strokeDashoffset = 540.35 * (1 - pct / 100)));
    let n = 0;
    const timer = setInterval(() => {
      n = Math.min(pct, n + 2);
      $("pct").textContent = n + "%";
      if (n >= pct) clearInterval(timer);
    }, 16);

    const chips = [
      ["Claimed", $("claimed").value],
      ["Decided by", d.escalated ? "face + voice" : "face only"],
    ];
    if (d.faceScore != null) chips.push(["Face", d.faceScore.toFixed(3)]);
    if (d.voiceScore != null) chips.push(["Voice", d.voiceScore.toFixed(3)]);
    const meta = $("meta");
    chips.forEach(([k, val]) => {
      const c = document.createElement("span");
      c.className = "chip";
      c.textContent = k + " ";
      const em = document.createElement("em");
      em.textContent = val;
      c.append(em);
      meta.append(c);
    });
    if (d.demo) {
      const c = document.createElement("span");
      c.className = "chip warn";
      c.textContent = "⚠ Demo";
      meta.append(c);
    }
  }, d.escalated ? 1500 : 900);

  // Honest demo note
  const note = $("demoNote");
  if (d.demo && d.trial) {
    const t = d.trial;
    note.hidden = false;
    note.textContent =
      `Demo mode: no backend is running at ${API_BASE}, so your uploads were not analyzed. ` +
      `This replays a real Session-3 test trial: recording ${t.sequence} of speaker ${t.probe_speaker} ` +
      `claiming to be ${t.claimed_speaker} (${t.y_true ? "genuine attempt" : "impostor attempt"}).`;
  } else if (d.demo) {
    note.hidden = false;
    note.textContent = "Demo mode: no backend and no results data were found, so this is a random example.";
  } else {
    note.hidden = true;
  }
  // On wide screens the result panel is already visible on the right.
  if (innerWidth <= 1000) setTimeout(() => res.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
}

$("resetBtn").addEventListener("click", () => {
  state.face = null;
  state.faceFrames = [];
  state.voice = null;
  staticPeaks = null;
  stopCamera();
  $("faceImg").hidden = true;
  $("faceImg").removeAttribute("src");
  drop.classList.remove("has-media");
  $("player").hidden = true;
  $("player").removeAttribute("src");
  $("timer").textContent = "00:00";
  $("recBtn").textContent = "● Record";
  $("result").hidden = true;
  $("verify").querySelector(".go-label").textContent = "Verify identity";
  $("faceFile").value = "";
  $("audFile").value = "";
  cancelAnimationFrame(rafId);
  drawIdle();
  refresh();
});

// ===========================================================================
// Performance section (charts from results/)
// ===========================================================================
const SYSTEMS = [
  { key: "Face SVM", label: "Face only", short: "Face only", color: "var(--s-face)" },
  { key: "Voice SVM", label: "Voice only", short: "Voice only", color: "var(--s-voice)" },
  { key: "Hierarchical Face-Voice", label: "Hierarchical (face → voice)", short: "Hierarchical", color: "var(--s-hier)" },
];
const METRICS = [
  { key: "FAR", title: "False accept rate", note: "Impostors let in", better: "low" },
  { key: "FRR", title: "False reject rate", note: "Real users turned away", better: "low" },
  { key: "Recall", title: "Recall", note: "Real users correctly accepted", better: "high" },
  { key: "F1", title: "F1 score", note: "Balance of precision and recall", better: "high" },
];

function renderPerformance() {
  const rows = SYSTEMS.map((s) => ({ ...s, d: state.comparison.find((r) => r.System === s.key) })).filter((s) => s.d);

  // legend (always shown for multiple series)
  $("legend").innerHTML = "";
  rows.forEach((s) => {
    const span = document.createElement("span");
    span.innerHTML = `<i style="background:${s.color}"></i>`;
    span.append(s.label);
    $("legend").append(span);
  });

  // one small chart per metric
  const charts = $("charts");
  charts.innerHTML = "";
  METRICS.forEach((m) => {
    const vals = rows.map((s) => s.d[m.key]);
    const max = Math.max(...vals) || 1;
    const bestVal = m.better === "low" ? Math.min(...vals) : Math.max(...vals);
    const card = document.createElement("div");
    card.className = "chart";
    card.innerHTML = `<h4>${m.title}<small>${m.better === "low" ? "lower is better" : "higher is better"}</small></h4><p>${m.note}</p><div class="bars"></div>`;
    const bars = card.querySelector(".bars");
    rows.forEach((s) => {
      const v = s.d[m.key];
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `<div class="lab"><span></span><b>${fmtPct(v)}${v === bestVal ? '<span class="best">★ best</span>' : ""}</b></div>
        <div class="track"><div class="b" style="background:${s.color}"></div></div>`;
      row.querySelector(".lab span").textContent = s.short;
      bars.append(row);
      requestAnimationFrame(() => (row.querySelector(".b").style.width = (v / max) * 100 + "%"));
      bindTip(row, `${s.label}<br>${m.title}: <b>${fmtPct(v, 2)}</b><br>TP ${s.d.TP} · FP ${s.d.FP} · FN ${s.d.FN} · TN ${s.d.TN}`);
    });
    charts.append(card);
  });

  // table view
  const cols = ["Accuracy", "Precision", "Recall", "F1", "FAR", "FRR", "ROC-AUC"];
  let html = `<table><thead><tr><th>System</th>${cols.map((c) => `<th>${c}</th>`).join("")}<th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr></thead><tbody>`;
  rows.forEach((s) => {
    html += `<tr><td>${s.label}</td>${cols.map((c) => `<td>${s.d[c] == null ? "—" : fmtPct(s.d[c], 2)}</td>`).join("")}<td>${s.d.TP}</td><td>${s.d.FP}</td><td>${s.d.FN}</td><td>${s.d.TN}</td></tr>`;
  });
  $("tableWrap").innerHTML = html + "</tbody></table>";

  // where trials were decided (segmented bar)
  const st = state.stages;
  const segs = [
    { label: "Accepted by face", n: st.face_direct_accept, color: "var(--o-accept)" },
    { label: "Escalated to voice", n: st.voice_escalations, color: "var(--o-escalate)" },
    { label: "Rejected by face", n: st.face_direct_reject, color: "var(--o-reject)" },
  ];
  const total = st.total_trials;
  $("seg").innerHTML = "";
  $("segLegend").innerHTML = "";
  segs.forEach((s) => {
    const share = s.n / total;
    const d = document.createElement("div");
    d.style.flex = s.n;
    d.style.background = s.color;
    if (share > 0.12) d.textContent = `${s.label} · ${s.n.toLocaleString()}`;
    bindTip(d, `${s.label}<br><b>${s.n.toLocaleString()}</b> of ${total.toLocaleString()} trials (${fmtPct(share)})`);
    $("seg").append(d);
    const l = document.createElement("span");
    l.innerHTML = `<i style="background:${s.color}"></i>`;
    l.append(`${s.label} - ${s.n.toLocaleString()} (${fmtPct(share)})`);
    $("segLegend").append(l);
  });

  // hero stats
  const hier = rows.find((s) => s.key === "Hierarchical Face-Voice");
  $("sTrials").textContent = total.toLocaleString();
  if (hier) $("sAcc").textContent = fmtPct(hier.d.Accuracy);
  $("sEsc").textContent = fmtPct(st.voice_escalation_rate ?? st.voice_escalations / total);
}

$("tableToggle").addEventListener("click", () => {
  const open = $("tableWrap").hidden;
  $("tableWrap").hidden = !open;
  $("tableToggle").textContent = open ? "Hide table" : "Show table";
  $("tableToggle").setAttribute("aria-expanded", String(open));
});

// ===========================================================================
// Enroll a new person (real backend only)
// ===========================================================================
function renderEnrollThumbs() {
  const box = $("enrollThumbs");
  box.innerHTML = "";
  if (!state.enrollPhotos.length) {
    box.innerHTML = `<p class="hint">No photos yet.</p>`;
  }
  state.enrollPhotos.forEach((f) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    img.alt = "Enrollment photo";
    box.append(img);
  });
  const name = $("enrollName").value.trim();
  $("enrollBtn").disabled = !(name && state.enrollPhotos.length);
  $("enrollBtn").querySelector(".go-label").textContent =
    state.enrollPhotos.length ? `Enroll with ${state.enrollPhotos.length} photo${state.enrollPhotos.length > 1 ? "s" : ""}` : "Enroll";
}

function renderEnrolledList(list) {
  const box = $("enrolledList");
  box.innerHTML = "";
  if (!list.length) {
    box.innerHTML = `<p class="hint">Nobody enrolled yet.</p>`;
    return;
  }
  list.forEach((e) => {
    const row = document.createElement("div");
    row.className = "enrolled-item";
    row.innerHTML = `<b></b><small></small><button type="button">Remove</button>`;
    row.querySelector("b").textContent = e.id;
    row.querySelector("small").textContent = `${e.samples} photo${e.samples > 1 ? "s" : ""}`;
    row.querySelector("button").addEventListener("click", async () => {
      await fetch(API_BASE + "/api/enroll/" + encodeURIComponent(e.id), { method: "DELETE" });
      fillIdentities();
    });
    box.append(row);
  });
}

$("enrollName").addEventListener("input", renderEnrollThumbs);
$("enrollUpload").addEventListener("click", () => $("enrollFiles").click());
$("enrollFiles").addEventListener("change", (e) => {
  state.enrollPhotos.push(...[...e.target.files].filter((f) => f.type.startsWith("image/")));
  e.target.value = "";
  renderEnrollThumbs();
});

$("enrollCam").addEventListener("click", async () => {
  const v = $("enrollVideo");
  const btn = $("enrollCam");
  let s;
  try {
    s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
  } catch {
    return toast("Couldn't open the camera. Check that it's connected and allowed in the browser.");
  }
  v.srcObject = s;
  v.hidden = false;
  btn.disabled = true;
  await new Promise((r) => (v.onloadeddata = r));
  for (let i = 5; i >= 1; i--) {
    btn.textContent = `Photo in ${i}… move your head a little`;
    await new Promise((r) => setTimeout(r, 900));
    state.enrollPhotos.push(await grabFrame(v, `enroll${Date.now()}.jpg`));
    renderEnrollThumbs();
  }
  s.getTracks().forEach((t) => t.stop());
  v.hidden = true;
  btn.disabled = false;
  btn.textContent = "📷 Take 5 photos";
});

$("enrollBtn").addEventListener("click", async () => {
  const btn = $("enrollBtn");
  const name = $("enrollName").value.trim();
  btn.disabled = true;
  btn.classList.add("busy");
  try {
    const fd = new FormData();
    fd.append("name", name);
    state.enrollPhotos.forEach((f, i) => fd.append("faces", f, f.name || `enroll${i}.jpg`));
    const r = await fetch(API_BASE + "/api/enroll", { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || "HTTP " + r.status);
    toast(`Enrolled ${body.id} with ${body.samples} photo(s)` + (body.skipped ? ` - ${body.skipped} had no visible face and were skipped.` : "."));
    state.enrollPhotos = [];
    $("enrollName").value = "";
    await fillIdentities(body.id);
  } catch (err) {
    toast(err.message);
  } finally {
    btn.classList.remove("busy");
    renderEnrollThumbs();
  }
});

// ===========================================================================
// Start
// ===========================================================================
// ---------- motion: scroll reveal, card spotlight, floating particles ----------
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

document.querySelectorAll(".stats, .panel, .grid > .card, .chart").forEach((el, i) => {
  el.classList.add("reveal");
  el.style.transitionDelay = (i % 4) * 70 + "ms";
});
const io = new IntersectionObserver((entries) => {
  entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
// Charts are created later, so watch for them too.
new MutationObserver(() => {
  document.querySelectorAll(".chart:not(.reveal)").forEach((el, i) => {
    el.classList.add("reveal");
    el.style.transitionDelay = i * 80 + "ms";
    io.observe(el);
  });
}).observe($("charts"), { childList: true });

// Glow that follows the mouse inside cards
document.addEventListener("pointermove", (e) => {
  const card = e.target.closest(".card");
  if (!card) return;
  const r = card.getBoundingClientRect();
  card.style.setProperty("--mx", e.clientX - r.left + "px");
  card.style.setProperty("--my", e.clientY - r.top + "px");
});

if (!reduceMotion) {
  const box = $("particles");
  for (let i = 0; i < 22; i++) {
    const p = document.createElement("i");
    p.style.left = Math.random() * 100 + "%";
    p.style.animationDuration = 10 + Math.random() * 14 + "s";
    p.style.animationDelay = -Math.random() * 20 + "s";
    if (i % 3 === 1) { p.style.background = "#8b5cf6"; p.style.boxShadow = "0 0 8px #8b5cf6"; }
    if (i % 3 === 2) { p.style.background = "#f472b6"; p.style.boxShadow = "0 0 8px #f472b6"; }
    box.append(p);
  }
}

sizeCanvas();
addEventListener("resize", () => { sizeCanvas(); if (staticPeaks) drawBars(staticPeaks, cssVar("--wave-done")); });
drawIdle();
Promise.all([checkBackend(), loadData()]).then(async () => {
  refresh();
  await fillIdentities();
  renderPerformance();
});
refresh();
