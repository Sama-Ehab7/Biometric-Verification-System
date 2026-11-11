"""Command classifier API with protected execution deliberately disabled.

The experimental Voice verifier now runs, but its corpus is too small and lacks
unknown-speaker validation, so it is not accepted as a security authenticator.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.verification.custom_commands import classify_custom_command

router = APIRouter(prefix="/api/custom/commands", tags=["custom-commands"])
MAX_AUDIO_BYTES = 5 * 1024 * 1024


@router.get("/status")
def status():
    return {"available": True, "mode": "classification_only", "execution_enabled": False,
            "reason": "Experimental speaker verification is not validated for protected authorization; command classification never authenticates a speaker."}


@router.post("/classify")
async def classify_route(audio: UploadFile = File(...)):
    if audio.content_type not in {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}:
        raise HTTPException(415, "Upload a WAV command recording.")
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(400, "Audio upload is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio exceeds the 5 MiB limit.")
    try:
        return classify_custom_command(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/execute")
async def execute_route(audio: UploadFile = File(...)):
    # Deny before command classification. Client-provided identity is never
    # trusted, and the experimental server verifier is not security-validated.
    raise HTTPException(503, "Protected command execution is disabled: the experimental speaker verifier is not validated for authorization.")

