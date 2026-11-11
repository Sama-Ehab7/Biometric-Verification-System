"""Fail-closed all-of-three demo policy for independent custom modalities."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.custom_face_demo import _image_bytes
from src.verification.custom_face import verify_custom_face
from src.verification.custom_voice import IDENTITIES, verify_custom_voice
from src.verification.custom_fingerprint_demo import verify_custom_fingerprint_test_image

router = APIRouter(prefix="/api/custom/combined", tags=["custom-combined"])
MAX_AUDIO_BYTES = 8 * 1024 * 1024
WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}
REQUIRED_MODALITIES = ("face", "voice", "fingerprint")


def all_modalities_accept(components: dict) -> bool:
    """Return true only when each required modality has an explicit ACCEPT."""
    if any(name not in components for name in REQUIRED_MODALITIES):
        return False
    for name in REQUIRED_MODALITIES:
        item = components[name]
        accepted = item.get("accepted")
        if accepted is None:
            accepted = item.get("decision") == "ACCEPT"
        if accepted is not True:
            return False
    return True


@router.get("/status")
def status():
    return {"available": True, "policy": "all three modalities must accept", "decision_source": "server",
            "fingerprint_mapping": "assumed demo alias only; not proof of real cross-modal identity",
            "scientifically_validated_fusion": False}


@router.post("/verify")
async def verify_route(claimed_identity: str = Form(...), face_model: str = Form("svm"),
                       face_image: UploadFile = File(...), voice_audio: UploadFile = File(...),
                       fingerprint_probe_alias: str = Form(...), fingerprint_image_name: str = Form(...)):
    alias = claimed_identity.strip().lower()
    if alias not in IDENTITIES:
        raise HTTPException(400, "Unknown custom identity.")
    if voice_audio.content_type not in WAV_TYPES:
        raise HTTPException(415, "Upload a WAV voice recording.")
    audio = await voice_audio.read(MAX_AUDIO_BYTES + 1)
    if not audio:
        raise HTTPException(400, "Voice upload is empty.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Voice upload exceeds the 8 MiB limit.")
    try:
        face_result = verify_custom_face(await _image_bytes(face_image), alias, face_model)
        voice_result = verify_custom_voice(audio, alias)
        fingerprint_result = verify_custom_fingerprint_test_image(
            alias, fingerprint_probe_alias.strip().lower(), fingerprint_image_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "A required custom modality is unavailable.") from exc
    components = {"face": face_result, "voice": voice_result,
                  "fingerprint": fingerprint_result}
    accepted = all_modalities_accept(components)
    return {"claimed_identity": alias, "decision": "ACCEPT" if accepted else "REJECT",
            "accepted": accepted, "policy": "all three modalities must accept",
            "components": components,
            "limitation": "Fingerprint association is an assumed alias mapping; this is not scientifically validated three-modality fusion."}
