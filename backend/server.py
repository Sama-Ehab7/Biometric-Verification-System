"""
Biometric Verification System - backend (face verification)

Uses the project's own pipeline so real faces are processed exactly like the
training data:
  Haar-cascade face detection -> 15% padded crop -> 128x128 grayscale
  -> HOG + LBP per frame -> mean/std over frames (sequence features)
  -> |probe - enrolled reference| -> trained SVM (saved_models/face/svm_face_security.joblib)
  -> genuine probability, accepted if >= the threshold chosen for the target FAR.

Identities:
  * the 43 VidTIMIT speakers (references = mean of their Session-1 recordings)
  * new people you enroll through POST /api/enroll (stored in backend/enrolled/)

Run (from the project root):
  F:\\bvs\\venv\\Scripts\\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "features"))

# Reuse the project's exact preprocessing + feature code.
from face_preprocessing import detect_face  # noqa: E402
from face_features import IMAGE_SIZE, extract_hog_features, extract_lbp_features  # noqa: E402

MODEL_PATH = ROOT / "saved_models" / "face" / "svm_face_security.joblib"
TRAIN_SPLIT = ROOT / "data" / "processed" / "splits" / "face_train.csv"
ENROLL_DIR = Path(__file__).resolve().parent / "enrolled"
ENROLL_FILE = ENROLL_DIR / "enrolled.json"
FAR_TARGET = 0.01          # operating point: at most ~1% impostors accepted
MIN_FRAMES = 8             # single photos are augmented up to this many frames

# --------------------------------------------------------------------------
# Load model + enrollment references
# --------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise RuntimeError(f"Trained face model not found: {MODEL_PATH}. Run the training pipeline first.")
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
THRESHOLDS = {float(k): float(v) for k, v in artifact["thresholds"].items()}
THRESHOLD = THRESHOLDS.get(FAR_TARGET) or min(THRESHOLDS.values())
FEATURE_COLS = list(artifact["feature_columns"])
POS = list(MODEL.classes_).index(1)

train_df = pd.read_csv(TRAIN_SPLIT)
REFERENCES: dict[str, np.ndarray] = {
    str(spk): grp[FEATURE_COLS].to_numpy(np.float32).mean(axis=0)
    for spk, grp in train_df.groupby("speaker")
}
VIDTIMIT_IDS = sorted(REFERENCES)

ENROLL_DIR.mkdir(exist_ok=True)
ENROLLED: dict[str, dict] = json.loads(ENROLL_FILE.read_text()) if ENROLL_FILE.exists() else {}


def save_enrolled() -> None:
    ENROLL_FILE.write_text(json.dumps(ENROLLED))


# --------------------------------------------------------------------------
# Image -> face crop -> features
# --------------------------------------------------------------------------
def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read the image file.")
    # Very large photos make the Haar detector slow; VidTIMIT frames are 512x384.
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def crop_face(img: np.ndarray):
    """Same logic as src/preprocessing/face_preprocessing.preprocess_image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = detect_face(gray)
    if det is None:
        return None, None
    x, y, w, h = [int(v) for v in det]
    px, py = int(w * 0.15), int(h * 0.15)
    H, W = gray.shape
    x1, y1, x2, y2 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    face = gray[y1:y2, x1:x2]
    if face.size == 0:
        return None, None
    return cv2.resize(face, IMAGE_SIZE, interpolation=cv2.INTER_AREA), (x1, y1, x2 - x1, y2 - y1)


def augment(face: np.ndarray, n: int) -> list[np.ndarray]:
    """Make small natural variations of one face (shift, scale, rotation, light),
    standing in for the frame-to-frame variation of a video sequence."""
    rng = np.random.default_rng(0)
    out = [face]
    c = (IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2)
    while len(out) < n:
        M = cv2.getRotationMatrix2D(c, rng.uniform(-4, 4), rng.uniform(0.96, 1.04))
        M[:, 2] += rng.uniform(-3, 3, size=2)
        f = cv2.warpAffine(face, M, IMAGE_SIZE, borderMode=cv2.BORDER_REFLECT)
        f = np.clip(f.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8), 0, 255).astype(np.uint8)
        out.append(f)
    return out


def sequence_features(faces: list[np.ndarray]) -> np.ndarray:
    """Same aggregation as src/features/face_features.extract_sequence_features."""
    if len(faces) < MIN_FRAMES:
        extra = []
        per = int(np.ceil(MIN_FRAMES / len(faces)))
        for f in faces:
            extra.extend(augment(f, per))
        faces = extra[: max(MIN_FRAMES, len(faces))]
    hog = np.asarray([extract_hog_features(f) for f in faces])
    lbp = np.asarray([extract_lbp_features(f) for f in faces])
    return np.concatenate([hog.mean(0), hog.std(0), lbp.mean(0), lbp.std(0)]).astype(np.float32)


async def faces_from_uploads(files: list[UploadFile]):
    faces, preview, skipped = [], None, 0
    for f in files:
        img = decode_image(await f.read())
        face, box = crop_face(img)
        if face is None:
            skipped += 1
            continue
        faces.append(face)
        if preview is None:
            ok, png = cv2.imencode(".png", face)
            preview = "data:image/png;base64," + base64.b64encode(png).decode() if ok else None
    if not faces:
        raise HTTPException(422, "No face was found. Use a clear, front-facing photo with good light.")
    return faces, preview, skipped


def score(probe: np.ndarray, reference: np.ndarray) -> float:
    pair = np.abs(probe - reference).reshape(1, -1)
    return float(MODEL.predict_proba(pair)[0, POS])


def reference_for(identity: str) -> np.ndarray:
    if identity in ENROLLED:
        return np.asarray(ENROLLED[identity]["reference"], np.float32)
    if identity in REFERENCES:
        return REFERENCES[identity]
    raise HTTPException(404, f"Unknown identity: {identity}")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
app = FastAPI(title="Biometric Verification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model": MODEL_PATH.name,
        "far_target": FAR_TARGET,
        "threshold": THRESHOLD,
        "vidtimit_identities": len(VIDTIMIT_IDS),
        "enrolled_identities": len(ENROLLED),
    }


@app.get("/api/identities")
def identities():
    return {
        "enrolled": [{"id": k, "samples": v["samples"]} for k, v in sorted(ENROLLED.items())],
        "vidtimit": VIDTIMIT_IDS,
    }


@app.post("/api/enroll")
async def enroll(name: str = Form(...), faces: list[UploadFile] = File(...)):
    """Enroll a new person. Each uploaded photo (or each camera capture) is one sample;
    the reference is the mean of the samples, like Session-1 enrollment."""
    ident = re.sub(r"[^\w\- ]", "", name).strip()[:40]
    if not ident:
        raise HTTPException(400, "Please give a name.")
    if ident in REFERENCES:
        raise HTTPException(400, "That name is a VidTIMIT speaker ID. Choose another name.")
    samples, preview, skipped = [], None, 0
    for f in faces:
        img = decode_image(await f.read())
        face, _ = crop_face(img)
        if face is None:
            skipped += 1
            continue
        samples.append(sequence_features([face]))
        if preview is None:
            ok, png = cv2.imencode(".png", face)
            preview = "data:image/png;base64," + base64.b64encode(png).decode() if ok else None
    if not samples:
        raise HTTPException(422, "No face was found in any photo. Use clear, front-facing photos.")
    ENROLLED[ident] = {"reference": np.mean(samples, axis=0).tolist(), "samples": len(samples)}
    save_enrolled()
    return {"id": ident, "samples": len(samples), "skipped": skipped, "face_preview": preview}


@app.delete("/api/enroll/{ident}")
def unenroll(ident: str):
    if ENROLLED.pop(ident, None) is None:
        raise HTTPException(404, "Not enrolled.")
    save_enrolled()
    return {"deleted": ident}


@app.post("/api/verify")
async def verify(
    claimed_id: str = Form(...),
    face: list[UploadFile] = File(...),          # one photo, or several camera frames
    voice: UploadFile | None = File(None),       # accepted for compatibility; not used yet
):
    reference = reference_for(claimed_id)
    faces, preview, skipped = await faces_from_uploads(face)
    s = score(sequence_features(faces), reference)
    return {
        "accepted": s >= THRESHOLD,
        "score": s,
        "face_score": s,
        "threshold": THRESHOLD,
        "far_target": FAR_TARGET,
        "stage": "face only",
        "frames_used": len(faces),
        "frames_without_face": skipped,
        "face_preview": preview,
    }
