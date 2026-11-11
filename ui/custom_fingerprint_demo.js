"use strict";

const API = "http://localhost:8000/api/custom-fingerprint";
const $ = (id) => document.getElementById(id);

async function getJson(path) {
  const response = await fetch(`${API}${path}`);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function fillSelect(select, values) {
  select.replaceChildren(...values.map((value) => new Option(value, value)));
}

async function loadDemo() {
  const [identityData, imageData] = await Promise.all([
    getJson("/identities"),
    getJson("/test-images"),
  ]);
  fillSelect($("claimed-alias"), identityData.identities);
  fillSelect($("probe-alias"), Object.keys(imageData.images));
  window.customFingerprintTestImages = imageData.images;
  updateImages();
  $("result").textContent = "Select a claim and one of the mapped test images.";
}

function updateImages() {
  const alias = $("probe-alias").value;
  fillSelect($("image-name"), window.customFingerprintTestImages?.[alias] || []);
}

$("probe-alias").addEventListener("change", updateImages);

$("verify-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("verify-button");
  const result = $("result");
  button.disabled = true;
  result.textContent = "Verifying…";
  try {
    const form = new FormData();
    form.set("claimed_alias", $("claimed-alias").value);
    form.set("probe_alias", $("probe-alias").value);
    form.set("image_name", $("image-name").value);
    const response = await fetch(`${API}/verify`, { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    result.textContent = [
      `${body.decision} (${body.model})`,
      `Custom claim: ${body.custom_identity} → ${body.claimed_identity}`,
      `Score: ${Number(body.score).toFixed(6)}`,
      `Threshold: ${Number(body.threshold).toFixed(8)}`,
      `Probe: ${$("probe-alias").value}/${$("image-name").value}`,
    ].join("\n");
  } catch (error) {
    result.textContent = `Verification failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
});

loadDemo().catch((error) => {
  $("result").textContent = `Could not load custom fingerprint demo: ${error.message}`;
});
