"""API routes for experimental custom-team Voice verification/identification."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.verification.custom_voice import IDENTITIES, identify_custom_voice, verify_custom_voice

router = APIRouter(prefix="/api/custom/voice", tags=["custom-voice"])
MAX_AUDIO_BYTES = 8 * 1024 * 1024
WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}


@router.get("/status")
def status():
    return {"available": True, "identities": list(IDENTITIES), "model_scope": "experimental custom corpus",
            "unknown_rejection_validated": False}


async def _audio_bytes(audio: UploadFile) -> bytes:
    if audio.content_type not in WAV_TYPES:
        raise HTTPException(415, "Upload a WAV recording.")
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(400, "Audio upload is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio exceeds the 8 MiB limit.")
    return data


@router.post("/verify")
async def verify_route(claimed_identity: str = Form(...), audio: UploadFile = File(...)):
    try:
        return verify_custom_voice(await _audio_bytes(audio), claimed_identity.strip().lower())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Custom Voice model artifacts are unavailable.") from exc


@router.post("/identify")
async def identify_route(audio: UploadFile = File(...)):
    try:
        return identify_custom_voice(await _audio_bytes(audio))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Custom Voice model artifacts are unavailable.") from exc
