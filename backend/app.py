"""
FastAPI app: the HTTP routes only. The work happens in:
  face.py        image -> face crop -> features
  voice.py       audio -> features
  models.py      trained SVMs, enrollment references, policy
  policy.py      the hierarchical decision rules
  enrollment.py  people enrolled through the API

Run from the project root:
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, face, voice
from .custom_face_demo import router as custom_face_demo_router
from .custom_commands_demo import router as custom_commands_demo_router
from .custom_fingerprint_demo import router as custom_fingerprint_demo_router
from .custom_voice_demo import router as custom_voice_demo_router
from .custom_combined_demo import router as custom_combined_demo_router
from .enrollment import EnrollmentStore, clean_name
from .models import load_models
from .policy import VOICE_REQUIRED, face_only_stage, is_accepted

MODELS = load_models()
STORE = EnrollmentStore(config.ENROLL_FILE)

app = FastAPI(title="Biometric Verification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(custom_fingerprint_demo_router)
app.include_router(custom_face_demo_router)
app.include_router(custom_commands_demo_router)
app.include_router(custom_voice_demo_router)
app.include_router(custom_combined_demo_router)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def references(identity: str):
    """(face reference, voice reference or None) for an enrolled person or VidTIMIT speaker."""
    if identity in STORE:
        return STORE.references(identity)
    if identity in MODELS.face.references:
        voice_ref = MODELS.voice.references.get(identity) if MODELS.voice else None
        return MODELS.face.references[identity], voice_ref
    raise HTTPException(404, f"Unknown identity: {identity}")


async def read_all(files: list[UploadFile]) -> list[bytes]:
    return [await f.read() for f in files]


def voice_features(data: bytes):
    try:
        return voice.features_from_bytes(data, MODELS.voice.columns)
    except voice.BadAudioError as exc:
        raise HTTPException(422, str(exc)) from exc


def crop_all(images: list[bytes]):
    try:
        return face.faces_from_images(images)
    except face.NoFaceError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/")
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "mode": MODELS.mode,
        "face_model": config.FACE_MODEL.name,
        "voice_model": config.VOICE_MODEL.name if MODELS.voice else None,
        "policy_source": MODELS.policy_source,
        "thresholds": {
            **(MODELS.policy.as_dict() if MODELS.policy else {}),
            "face_only": MODELS.face_only_threshold,
            "face_only_far": config.FACE_ONLY_FAR,
        },
        "vidtimit_identities": len(MODELS.face.references),
        "enrolled_identities": len(STORE.people),
    }


@app.get("/api/identities")
def identities():
    return {"enrolled": STORE.summary(), "vidtimit": sorted(MODELS.face.references)}


@app.post("/api/enroll")
async def enroll(
    name: str = Form(...),
    faces: list[UploadFile] = File(...),
    voices: list[UploadFile] | None = File(None),
):
    """Each photo is one enrollment sample (like a Session-1 recording);
    the reference is their mean. Optional voice samples give a voice reference."""
    ident = clean_name(name)
    if not ident:
        raise HTTPException(400, "Please give a name.")
    if ident in MODELS.face.references:
        raise HTTPException(400, "That name is a VidTIMIT speaker ID. Choose another name.")

    crops, skipped = crop_all(await read_all(faces))
    if not crops:
        raise HTTPException(422, "No face was found in any photo. Use clear, front-facing photos.")
    samples = [face.sequence_features([c]) for c in crops]

    voice_ref = None
    if voices and MODELS.voice:
        voice_ref = sum(voice_features(b) for b in await read_all(voices)) / len(voices)

    STORE.add(ident, np.mean(samples, axis=0), voice_ref, len(samples))
    return {"id": ident, "samples": len(samples), "skipped": skipped,
            "voice_enrolled": voice_ref is not None, "face_preview": face.to_data_url(crops[0])}


@app.delete("/api/enroll/{ident}")
def unenroll(ident: str):
    if not STORE.remove(ident):
        raise HTTPException(404, "Not enrolled.")
    return {"deleted": ident}


@app.post("/api/verify")
async def verify(
    claimed_id: str = Form(...),
    face_files: list[UploadFile] = File(..., alias="face"),   # one photo or several camera frames
    voice_file: UploadFile | None = File(None, alias="voice"),  # WAV, used only when the face is unsure
):
    face_ref, voice_ref = references(claimed_id)
    crops, skipped = crop_all(await read_all(face_files))
    if not crops:
        raise HTTPException(422, "No face was found. Use a clear, front-facing photo with good light.")

    face_score = MODELS.face.score(face.sequence_features(crops), face_ref)
    result = {
        "claimed_id": claimed_id,
        "mode": MODELS.mode,
        "face_score": face_score,
        "voice_score": None,
        "frames_used": len(crops),
        "frames_without_face": skipped,
        "face_preview": face.to_data_url(crops[0]),
    }

    # Face-only mode: one threshold (FAR target from the face model).
    if not MODELS.hierarchical:
        stage = face_only_stage(face_score, MODELS.face_only_threshold)
        return {**result, "stage": stage, "accepted": is_accepted(stage), "score": face_score,
                "threshold": MODELS.face_only_threshold, "far_target": config.FACE_ONLY_FAR}

    # Hierarchical mode: face decides when confident, otherwise voice.
    policy = MODELS.policy
    result["thresholds"] = {"low_face": policy.low_face, "high_face": policy.high_face, "voice": policy.voice}
    stage = policy.face_stage(face_score)
    if stage is not None:
        threshold = policy.high_face if is_accepted(stage) else policy.low_face
        return {**result, "stage": stage, "accepted": is_accepted(stage), "score": face_score, "threshold": threshold}

    if voice_file is None or voice_ref is None:
        message = ("The face score is in the uncertain zone. Add a voice sample and verify again."
                   if voice_file is None else
                   "This person has no enrolled voice, so the uncertain face score can't be resolved.")
        return {**result, "stage": VOICE_REQUIRED, "accepted": False, "score": face_score,
                "threshold": policy.high_face, "message": message}

    voice_score = MODELS.voice.score(voice_features(await voice_file.read()), voice_ref)
    stage = policy.voice_stage(voice_score)
    return {**result, "stage": stage, "accepted": is_accepted(stage), "score": voice_score,
            "voice_score": voice_score, "threshold": policy.voice}
