"""
Biometric Verification System - backend API (face + voice, hierarchical)

Reuses the project's own code so live inputs are processed exactly like the
training data:

  FACE  : Haar detection -> 15% padded crop -> 128x128 gray -> HOG + LBP per frame
          -> mean/std over frames -> |probe - enrolled reference| -> face SVM
  VOICE : 32 kHz mono WAV -> 13 MFCC + ZCR + RMS (mean/std)
          -> |probe - enrolled reference| -> voice SVM
  POLICY: the validated hierarchical policy (Session-2 thresholds):
            face >= high  -> accept by face
            face <= low   -> reject by face
            otherwise     -> voice decides (voice >= voice threshold -> accept)

Files it needs (all git-ignored, they live on the machine that trained the models):
  saved_models/face/svm_face_security.joblib          (required)
  data/processed/splits/face_train.csv                (required - Session-1 enrollment)
  saved_models/voice/svm_voice_security.joblib        (optional - enables voice)
  data/processed/splits/voice_train.csv               (optional - enables voice)
  saved_models/hierarchical/hierarchical_policy_audit.joblib
      (optional - otherwise results/hierarchical/selected_thresholds_audit.json is used)

Run from the project root:
  python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
import tempfile
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

if not hasattr(cv2, "CascadeClassifier"):
    raise RuntimeError(
        "This OpenCV build has no CascadeClassifier (OpenCV 5.x removed it). "
        "Install OpenCV 4.x:  pip install \"opencv-python<5\""
    )

# The project's exact preprocessing + feature code.
from face_preprocessing import detect_face  # noqa: E402
from face_features import IMAGE_SIZE, extract_hog_features, extract_lbp_features  # noqa: E402
from voice_features import EXPECTED_SAMPLE_RATE, extract_audio_features  # noqa: E402

FACE_MODEL = ROOT / "saved_models" / "face" / "svm_face_security.joblib"
FACE_TRAIN = ROOT / "data" / "processed" / "splits" / "face_train.csv"
VOICE_MODEL = ROOT / "saved_models" / "voice" / "svm_voice_security.joblib"
VOICE_TRAIN = ROOT / "data" / "processed" / "splits" / "voice_train.csv"
POLICY_FILE = ROOT / "saved_models" / "hierarchical" / "hierarchical_policy_audit.joblib"
POLICY_JSON = ROOT / "results" / "hierarchical" / "selected_thresholds_audit.json"
ENROLL_DIR = Path(__file__).resolve().parent / "enrolled"
ENROLL_FILE = ENROLL_DIR / "enrolled.json"

FACE_ONLY_FAR = 0.01   # face-only operating point when no hierarchical policy is available
MIN_FRAMES = 8         # a single photo is augmented to this many frames


# --------------------------------------------------------------------------
# Load models, enrollment references and the decision policy
# --------------------------------------------------------------------------
def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise RuntimeError(
            f"Missing {what}: {path}\n"
            "These files are git-ignored. Copy them from the machine that ran the "
            "pipeline, or re-run the pipeline (see backend/README.md)."
        )
    return path


def load_artifact(path: Path):
    art = joblib.load(path)
    model = art["model"]
    thresholds = {float(k): float(v) for k, v in art.get("thresholds", {}).items()}
    return model, thresholds, list(art["feature_columns"]), list(model.classes_).index(1)


FACE_SVM, FACE_THRESHOLDS, FACE_COLS, FACE_POS = load_artifact(require(FACE_MODEL, "face model"))
face_df = pd.read_csv(require(FACE_TRAIN, "face enrollment features"))
FACE_REFS = {str(s): g[FACE_COLS].to_numpy(np.float32).mean(0) for s, g in face_df.groupby("speaker")}
FACE_ONLY_THRESHOLD = FACE_THRESHOLDS.get(FACE_ONLY_FAR) or min(FACE_THRESHOLDS.values())

VOICE_SVM = None
VOICE_POS = 1
VOICE_REFS: dict[str, np.ndarray] = {}
VOICE_COLS: list[str] = []
if VOICE_MODEL.exists() and VOICE_TRAIN.exists():
    VOICE_SVM, _, VOICE_COLS, VOICE_POS = load_artifact(VOICE_MODEL)
    voice_df = pd.read_csv(VOICE_TRAIN)
    VOICE_REFS = {str(s): g[VOICE_COLS].to_numpy(np.float32).mean(0) for s, g in voice_df.groupby("speaker")}

POLICY = None
POLICY_SOURCE = None
if POLICY_FILE.exists():
    POLICY, POLICY_SOURCE = joblib.load(POLICY_FILE), POLICY_FILE.name
elif POLICY_JSON.exists():
    POLICY, POLICY_SOURCE = json.loads(POLICY_JSON.read_text(encoding="utf-8")), POLICY_JSON.name
if POLICY is not None:
    POLICY = {k: float(POLICY[k]) for k in ("low_face_threshold", "high_face_threshold", "voice_threshold")}

HIERARCHICAL = POLICY is not None and VOICE_SVM is not None
VIDTIMIT_IDS = sorted(FACE_REFS)

ENROLL_DIR.mkdir(exist_ok=True)
ENROLLED: dict[str, dict] = json.loads(ENROLL_FILE.read_text()) if ENROLL_FILE.exists() else {}


def save_enrolled() -> None:
    ENROLL_FILE.write_text(json.dumps(ENROLLED))


# --------------------------------------------------------------------------
# Face: image -> crop -> sequence features
# --------------------------------------------------------------------------
def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read the image file.")
    h, w = img.shape[:2]
    scale = 800 / max(h, w)  # large photos make the Haar detector slow
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def crop_face(img: np.ndarray):
    """Same as src/preprocessing/face_preprocessing.preprocess_image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = detect_face(gray)
    if det is None:
        return None
    x, y, w, h = [int(v) for v in det]
    px, py = int(w * 0.15), int(h * 0.15)
    H, W = gray.shape
    face = gray[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]
    if face.size == 0:
        return None
    return cv2.resize(face, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def augment(face: np.ndarray, n: int) -> list[np.ndarray]:
    """Small shifts / scale / rotation / light changes, standing in for the
    frame-to-frame variation of a video sequence when only one photo is given."""
    rng = np.random.default_rng(0)
    out = [face]
    c = (IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2)
    while len(out) < n:
        M = cv2.getRotationMatrix2D(c, rng.uniform(-4, 4), rng.uniform(0.96, 1.04))
        M[:, 2] += rng.uniform(-3, 3, size=2)
        f = cv2.warpAffine(face, M, IMAGE_SIZE, borderMode=cv2.BORDER_REFLECT)
        out.append(np.clip(f.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8), 0, 255).astype(np.uint8))
    return out


def face_sequence_features(faces: list[np.ndarray]) -> np.ndarray:
    """Same aggregation as src/features/face_features.extract_sequence_features."""
    if len(faces) < MIN_FRAMES:
        per = int(np.ceil(MIN_FRAMES / len(faces)))
        faces = [a for f in faces for a in augment(f, per)][:MIN_FRAMES]
    hog = np.asarray([extract_hog_features(f) for f in faces])
    lbp = np.asarray([extract_lbp_features(f) for f in faces])
    return np.concatenate([hog.mean(0), hog.std(0), lbp.mean(0), lbp.std(0)]).astype(np.float32)


def png_data_url(face: np.ndarray) -> str | None:
    ok, png = cv2.imencode(".png", face)
    return "data:image/png;base64," + base64.b64encode(png).decode() if ok else None


async def read_faces(files: list[UploadFile]):
    faces = []
    for f in files:
        face = crop_face(decode_image(await f.read()))
        if face is not None:
            faces.append(face)
    return faces, len(files) - len(faces)


# --------------------------------------------------------------------------
# Voice: audio -> 32 kHz WAV -> the project's extract_audio_features
# --------------------------------------------------------------------------
def voice_features_from_bytes(data: bytes) -> np.ndarray:
    import librosa
    import soundfile as sf

    try:
        signal, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(400, "Could not read the audio. Send a WAV file (the UI converts recordings to WAV).") from exc
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if len(signal) < sr * 0.5:
        raise HTTPException(422, "The voice sample is too short. Record at least 1-2 seconds.")
    if sr != EXPECTED_SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=EXPECTED_SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        sf.write(path, signal, EXPECTED_SAMPLE_RATE, subtype="PCM_16")
        feats, _, _ = extract_audio_features(path)  # project code, unchanged
    finally:
        path.unlink(missing_ok=True)
    return np.asarray([feats[c] for c in VOICE_COLS], np.float32)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def face_score(probe: np.ndarray, ref: np.ndarray) -> float:
    return float(FACE_SVM.predict_proba(np.abs(probe - ref).reshape(1, -1))[0, FACE_POS])


def voice_score(probe: np.ndarray, ref: np.ndarray) -> float:
    return float(VOICE_SVM.predict_proba(np.abs(probe - ref).reshape(1, -1))[0, VOICE_POS])


def references(identity: str):
    if identity in ENROLLED:
        e = ENROLLED[identity]
        v = np.asarray(e["voice"], np.float32) if e.get("voice") is not None else None
        return np.asarray(e["face"], np.float32), v
    if identity in FACE_REFS:
        return FACE_REFS[identity], VOICE_REFS.get(identity)
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
        "mode": "hierarchical" if HIERARCHICAL else "face_only",
        "face_model": FACE_MODEL.name,
        "voice_model": VOICE_MODEL.name if VOICE_SVM is not None else None,
        "policy_source": POLICY_SOURCE,
        "thresholds": {**(POLICY or {}), "face_only": FACE_ONLY_THRESHOLD, "face_only_far": FACE_ONLY_FAR},
        "vidtimit_identities": len(VIDTIMIT_IDS),
        "enrolled_identities": len(ENROLLED),
    }


@app.get("/api/identities")
def identities():
    return {
        "enrolled": [{"id": k, "samples": v["samples"], "voice": v.get("voice") is not None}
                     for k, v in sorted(ENROLLED.items())],
        "vidtimit": VIDTIMIT_IDS,
    }


@app.post("/api/enroll")
async def enroll(
    name: str = Form(...),
    faces: list[UploadFile] = File(...),
    voices: list[UploadFile] | None = File(None),
):
    """Enroll a new person: each photo is one sample, like Session-1 recordings.
    Optional voice samples give the person a voice reference too."""
    ident = re.sub(r"[^\w\- ]", "", name).strip()[:40]
    if not ident:
        raise HTTPException(400, "Please give a name.")
    if ident in FACE_REFS:
        raise HTTPException(400, "That name is a VidTIMIT speaker ID. Choose another name.")

    samples, preview = [], None
    for f in faces:
        face = crop_face(decode_image(await f.read()))
        if face is None:
            continue
        samples.append(face_sequence_features([face]))
        preview = preview or png_data_url(face)
    if not samples:
        raise HTTPException(422, "No face was found in any photo. Use clear, front-facing photos.")

    voice_ref = None
    if voices and VOICE_SVM is not None:
        vs = [voice_features_from_bytes(await v.read()) for v in voices]
        voice_ref = np.mean(vs, axis=0).tolist()

    ENROLLED[ident] = {"face": np.mean(samples, 0).tolist(), "voice": voice_ref, "samples": len(samples)}
    save_enrolled()
    return {"id": ident, "samples": len(samples), "skipped": len(faces) - len(samples),
            "voice_enrolled": voice_ref is not None, "face_preview": preview}


@app.delete("/api/enroll/{ident}")
def unenroll(ident: str):
    if ENROLLED.pop(ident, None) is None:
        raise HTTPException(404, "Not enrolled.")
    save_enrolled()
    return {"deleted": ident}


@app.post("/api/verify")
async def verify(
    claimed_id: str = Form(...),
    face: list[UploadFile] = File(...),     # one photo, or several camera frames
    voice: UploadFile | None = File(None),  # WAV; used only if the face stage is unsure
):
    face_ref, voice_ref = references(claimed_id)
    faces, skipped = await read_faces(face)
    if not faces:
        raise HTTPException(422, "No face was found. Use a clear, front-facing photo with good light.")
    fs = face_score(face_sequence_features(faces), face_ref)
    out = {
        "claimed_id": claimed_id,
        "face_score": fs,
        "voice_score": None,
        "frames_used": len(faces),
        "frames_without_face": skipped,
        "face_preview": png_data_url(faces[0]),
        "mode": "hierarchical" if HIERARCHICAL else "face_only",
    }

    if not HIERARCHICAL:
        accepted = fs >= FACE_ONLY_THRESHOLD
        return {**out, "accepted": accepted, "score": fs, "threshold": FACE_ONLY_THRESHOLD,
                "far_target": FACE_ONLY_FAR, "stage": "face_accept" if accepted else "face_reject"}

    lo, hi, vt = POLICY["low_face_threshold"], POLICY["high_face_threshold"], POLICY["voice_threshold"]
    out["thresholds"] = {"low_face": lo, "high_face": hi, "voice": vt}
    if fs >= hi:
        return {**out, "accepted": True, "score": fs, "threshold": hi, "stage": "face_accept"}
    if fs <= lo:
        return {**out, "accepted": False, "score": fs, "threshold": lo, "stage": "face_reject"}

    # Face is unsure -> the voice decides.
    if voice is None:
        return {**out, "accepted": False, "score": fs, "threshold": hi, "stage": "voice_required",
                "message": "The face score is in the uncertain zone. Add a voice sample and verify again."}
    if voice_ref is None:
        return {**out, "accepted": False, "score": fs, "threshold": hi, "stage": "voice_required",
                "message": "This person has no enrolled voice, so the uncertain face score can't be resolved."}
    vs = voice_score(voice_features_from_bytes(await voice.read()), voice_ref)
    accepted = vs >= vt
    return {**out, "accepted": accepted, "score": vs, "threshold": vt, "voice_score": vs,
            "stage": "voice_accept" if accepted else "voice_reject"}
