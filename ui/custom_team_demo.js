(() => {
  "use strict";
  const API = "http://127.0.0.1:8000";
  const $ = (id) => document.getElementById(id);
  const state = { identity: "nour", face: null, faceLive: false, voice: null, results: {}, stream: null, recorder: null, chunks: [] };

  function setStatus(id, text, tone = "") {
    const el = $(id); el.textContent = text; el.classList.remove("ok", "bad");
    if (tone) el.classList.add(tone);
  }
  function pill(id, decision) {
    const el = $(id); el.textContent = decision || "Pending";
    el.classList.remove("accept", "reject");
    if (decision === "ACCEPT") el.classList.add("accept");
    if (decision === "REJECT") el.classList.add("reject");
  }
  function errorText(data, fallback) { return data?.detail || data?.message || fallback; }
  async function post(url, body) {
    const response = await fetch(API + url, { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorText(data, `Request failed (${response.status})`));
    return data;
  }
  function refreshFinal() {
    const names = ["face", "voice", "fingerprint"];
    for (const name of names) pill(`${name}Decision`, state.results[name]?.decision);
    const complete = names.every((name) => state.results[name]);
    const accepted = complete && names.every((name) => state.results[name].decision === "ACCEPT");
    const final = complete ? (accepted ? "ACCEPT" : "REJECT") : "Pending";
    pill("finalDecision", final);
    if (complete) $("resultHeadline").textContent = accepted
      ? "All three checks accepted this claim in this demonstration."
      : "At least one check rejected this claim.";
    $("resultIdentity").textContent = state.identity[0].toUpperCase() + state.identity.slice(1);
  }
  function showScore(name, data) {
    const threshold = Number(data.threshold);
    const score = Number(data.score);
    const decision = data.decision || (data.accepted ? "ACCEPT" : "REJECT");
    state.results[name] = { decision, score, threshold };
    refreshFinal();
    return `${decision} · score ${score.toFixed(6)} · threshold ${threshold.toFixed(6)}`;
  }
  function setStep(id, done) { $(id).classList.toggle("done", !!done); }
  function enableCard(cardId, ...ids) {
    $(cardId).setAttribute("aria-disabled", "false");
    for (const id of ids) $(id).disabled = false;
  }
  async function checkApi() {
    try {
      const response = await fetch(`${API}/api/health`, { cache: "no-store" });
      if (!response.ok) throw new Error();
      const data = await response.json();
      $("apiStatus").className = "status online";
      $("apiStatusText").textContent = `Backend online · ${data.mode}`;
    } catch {
      $("apiStatus").className = "status demo";
      $("apiStatusText").textContent = "Backend unavailable";
    }
  }
  checkApi();

  $("identity").addEventListener("change", () => {
    state.identity = $("identity").value;
    $("identityName").textContent = state.identity[0].toUpperCase() + state.identity.slice(1);
    $("avatar").textContent = state.identity[0].toUpperCase();
    $("resultIdentity").textContent = $("identityName").textContent;
    resetFlow(false);
  });

  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null; $("faceVideo").srcObject = null; $("faceVideo").hidden = true;
    $("cameraStart").disabled = false; $("cameraStop").disabled = true;
  }
  $("cameraStart").addEventListener("click", async () => {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      $("faceVideo").srcObject = state.stream; $("faceVideo").hidden = false;
      $("faceCapture").disabled = false; $("cameraStart").disabled = true; $("cameraStop").disabled = false;
      setStatus("faceStatus", "Camera ready. Center one face and capture.");
    } catch (error) { setStatus("faceStatus", `Camera unavailable: ${error.message || "permission denied"}`, "bad"); }
  });
  $("cameraStop").addEventListener("click", stopCamera);
  addEventListener("pagehide", stopCamera);
  $("faceCapture").addEventListener("click", async () => {
    const video = $("faceVideo");
    if (!video.videoWidth || !video.videoHeight) { setStatus("faceStatus", "Waiting for a camera frame.", "bad"); return; }
    const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) { setStatus("faceStatus", "Could not capture this frame. Try again.", "bad"); return; }
    state.face = new File([blob], "custom-live-face.jpg", { type: "image/jpeg" }); state.faceLive = true;
    $("facePreview").src = URL.createObjectURL(state.face); $("facePreview").hidden = false; $("faceEmpty").hidden = true;
    $("faceVerify").disabled = false; setStatus("faceStatus", "Photo captured. Verify Face to continue.");
  });
  $("faceUploadButton").addEventListener("click", () => $("faceUpload").click());
  $("faceUpload").addEventListener("change", () => {
    const file = $("faceUpload").files[0]; if (!file) return;
    if (!file.type.startsWith("image/")) { setStatus("faceStatus", "Choose an image file.", "bad"); return; }
    state.face = file; state.faceLive = false; $("facePreview").src = URL.createObjectURL(file); $("facePreview").hidden = false;
    $("faceEmpty").hidden = true; $("faceVerify").disabled = false; setStatus("faceStatus", "Image selected. Verify Face to continue.");
  });
  $("faceVerify").addEventListener("click", async () => {
    if (!state.face) return;
    const button = $("faceVerify"); button.disabled = true; button.classList.add("busy"); setStatus("faceStatus", "Verifying Face…");
    const body = new FormData(); body.append("claimed_identity", state.identity); body.append("model", $("faceModel").value); body.append("image", state.face, state.face.name);
    try {
      const route = state.faceLive ? "/api/custom/face/verify-camera" : "/api/custom/face/verify";
      const data = await post(route, body); setStep("stepFace", true);
      setStatus("faceStatus", showScore("face", data), data.decision === "ACCEPT" ? "ok" : "bad");
      enableCard("voiceCard", "recordVoice", "voiceUploadButton");
      setStatus("voiceStatus", "Face checked. Record or choose audio, then verify Voice.");
      $("resultHeadline").textContent = `Face check: ${data.decision}. Complete Voice and Fingerprint next.`;
    } catch (error) { setStatus("faceStatus", `Face verification failed: ${error.message}`, "bad"); button.disabled = false; }
    finally { button.classList.remove("busy"); }
  });

  async function toWav32k(blob) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    const decoded = await context.decodeAudioData(await blob.arrayBuffer()); await context.close();
    const rate = 32000, offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * rate), rate);
    const source = offline.createBufferSource(); source.buffer = decoded; source.connect(offline.destination); source.start();
    const pcm = (await offline.startRendering()).getChannelData(0), buffer = new ArrayBuffer(44 + pcm.length * 2), view = new DataView(buffer);
    const write = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
    write(0, "RIFF"); view.setUint32(4, 36 + pcm.length * 2, true); write(8, "WAVE"); write(12, "fmt ");
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, "data"); view.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) { const sample = Math.max(-1, Math.min(1, pcm[i])); view.setInt16(44 + i * 2, sample < 0 ? sample * 32768 : sample * 32767, true); }
    return new File([buffer], "custom-voice.wav", { type: "audio/wav" });
  }
  async function useVoiceBlob(blob, filename) {
    try {
      state.voice = await toWav32k(blob);
      $("voicePlayer").src = URL.createObjectURL(state.voice); $("voicePlayer").hidden = false;
      setStatus("voiceStatus", `${filename} ready as 32 kHz mono WAV. Verify Voice to continue.`);
      $("voiceVerify").disabled = false;
    } catch (error) { state.voice = null; setStatus("voiceStatus", `Could not prepare audio: ${error.message}`, "bad"); }
  }
  $("voiceUploadButton").addEventListener("click", () => $("voiceUpload").click());
  $("voiceUpload").addEventListener("change", async () => { const file = $("voiceUpload").files[0]; if (file) await useVoiceBlob(file, file.name); });
  $("recordVoice").addEventListener("click", async () => {
    if (state.recorder?.state === "recording") { state.recorder.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); state.chunks = [];
      state.recorder = new MediaRecorder(stream);
      state.recorder.ondataavailable = (event) => { if (event.data.size) state.chunks.push(event.data); };
      state.recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop()); $("voiceBox").classList.remove("live");
        $("recordVoice").classList.remove("on"); $("recordVoice").textContent = "● Record again";
        await useVoiceBlob(new Blob(state.chunks, { type: state.recorder.mimeType }), "Recording");
      };
      state.recorder.start(); $("voiceBox").classList.add("live"); $("recordVoice").classList.add("on");
      $("recordVoice").textContent = "■ Stop recording"; setStatus("voiceStatus", "Recording… press Stop recording when finished.");
    } catch (error) { setStatus("voiceStatus", `Microphone unavailable: ${error.message || "permission denied"}`, "bad"); }
  });
  $("voiceVerify").addEventListener("click", async () => {
    if (!state.voice) return;
    const button = $("voiceVerify"); button.disabled = true; button.classList.add("busy"); setStatus("voiceStatus", "Verifying Voice…");
    const body = new FormData(); body.append("claimed_identity", state.identity); body.append("audio", state.voice, state.voice.name);
    try {
      const data = await post("/api/custom/voice/verify", body); setStep("stepVoice", true);
      const voiceSummary = showScore("voice", data);
      const voiceGuidance = data.decision === "REJECT"
        ? `${voiceSummary}. Check the selected identity and recording quality; genuine live samples may reject with this experimental model.`
        : voiceSummary;
      setStatus("voiceStatus", voiceGuidance, data.decision === "ACCEPT" ? "ok" : "bad");
      enableCard("fingerprintCard", "fingerprintFile");
      setStatus("fingerprintStatus", "Voice checked. Upload a fingerprint image, then verify it.");
      $("resultHeadline").textContent = `Face ${state.results.face.decision}; Voice ${data.decision}. Complete Fingerprint next.`;
    } catch (error) { setStatus("voiceStatus", `Voice verification failed: ${error.message}`, "bad"); button.disabled = false; }
    finally { button.classList.remove("busy"); }
  });

  $("fingerprintFile").addEventListener("change", () => {
    const file = $("fingerprintFile").files[0]; if (!file) return;
    const allowed = /\.(jpe?g|png|tiff?)$|\.webp$/i.test(file.name);
    if (!allowed) { setStatus("fingerprintStatus", "Choose a JPG, PNG, TIFF, or WebP fingerprint image.", "bad"); $("fingerprintVerify").disabled = true; return; }
    if (file.type.startsWith("image/") && !/\.tiff?$/i.test(file.name)) $("fingerprintPreview").src = URL.createObjectURL(file);
    else $("fingerprintPreview").removeAttribute("src");
    $("fingerprintPreview").hidden = false; $("fingerprintVerify").disabled = false;
    setStatus("fingerprintStatus", `${file.name} selected. Verify Fingerprint to finish.`);
  });
  $("fingerprintVerify").addEventListener("click", async () => {
    const file = $("fingerprintFile").files[0]; if (!file) return;
    const button = $("fingerprintVerify"); button.disabled = true; button.classList.add("busy"); setStatus("fingerprintStatus", "Verifying Fingerprint…");
    const body = new FormData(); body.append("claimed_alias", state.identity); body.append("image", file, file.name);
    try {
      const data = await post("/api/custom-fingerprint/verify-upload", body); setStep("stepFingerprint", true);
      setStatus("fingerprintStatus", `${showScore("fingerprint", data)} · mapped claim ${data.claimed_identity}`, data.decision === "ACCEPT" ? "ok" : "bad");
      $("resultHeadline").textContent = `Face ${state.results.face.decision}; Voice ${state.results.voice.decision}; Fingerprint ${data.decision}.`;
    } catch (error) { setStatus("fingerprintStatus", `Fingerprint verification failed: ${error.message}`, "bad"); button.disabled = false; }
    finally { button.classList.remove("busy"); }
  });

  function resetFlow(clearIdentity = true) {
    stopCamera();
    if (clearIdentity) { $("identity").value = "nour"; state.identity = "nour"; }
    state.face = null; state.faceLive = false; state.voice = null; state.results = {};
    for (const step of ["stepFace", "stepVoice", "stepFingerprint"]) setStep(step, false);
    $("facePreview").removeAttribute("src"); $("facePreview").hidden = true; $("faceEmpty").hidden = false;
    $("faceUpload").value = ""; $("voiceUpload").value = ""; $("fingerprintFile").value = "";
    $("voicePlayer").removeAttribute("src"); $("voicePlayer").hidden = true;
    $("fingerprintPreview").removeAttribute("src"); $("fingerprintPreview").hidden = true;
    $("faceVerify").disabled = true; $("voiceVerify").disabled = true; $("fingerprintVerify").disabled = true;
    $("recordVoice").disabled = true; $("voiceUploadButton").disabled = true; $("fingerprintFile").disabled = true;
    $("voiceCard").setAttribute("aria-disabled", "true"); $("fingerprintCard").setAttribute("aria-disabled", "true");
    setStatus("faceStatus", "Waiting for a face image."); setStatus("voiceStatus", "Complete Face verification first.");
    setStatus("fingerprintStatus", "Complete Voice verification first."); $("resultHeadline").textContent = "Waiting for the first face check.";
    $("identityName").textContent = state.identity[0].toUpperCase() + state.identity.slice(1); $("avatar").textContent = state.identity[0].toUpperCase();
    refreshFinal();
  }
  $("resetFlow").addEventListener("click", () => resetFlow(true));
  resetFlow(false);

  $("classifyCommand").addEventListener("click", async () => {
    const file = $("commandAudio").files[0]; if (!file) { setStatus("commandResult", "Choose a WAV command sample first.", "bad"); return; }
    try {
      const body = new FormData(); body.append("audio", file);
      const data = await post("/api/custom/commands/classify", body);
      setStatus("commandResult", `Predicted ${data.command.toUpperCase()} · score ${Number(data.score).toFixed(6)}. Classification only; no authorization.`, "ok");
    } catch (error) { setStatus("commandResult", `Classification error: ${error.message}`, "bad"); }
  });
  $("commandExecute").addEventListener("click", async () => {
    const file = $("commandAudio").files[0]; if (!file) { setStatus("commandResult", "Protected execution remains disabled (HTTP 503).", "bad"); return; }
    const body = new FormData(); body.append("audio", file);
    try { await post("/api/custom/commands/execute", body); }
    catch (error) { setStatus("commandResult", `Protected execution disabled: ${error.message}`, "bad"); }
  });
})();
