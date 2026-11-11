"""API routes for the isolated Custom Team Fingerprint Demo."""

from pathlib import Path
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.verification.custom_fingerprint_demo import (
    list_custom_fingerprint_aliases,
    list_custom_fingerprint_test_images,
    verify_custom_fingerprint_test_image,
    verify_custom_fingerprint_upload,
)


router = APIRouter(prefix="/api/custom-fingerprint", tags=["custom-fingerprint-demo"])
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@router.get("/identities")
def custom_fingerprint_identities():
    try:
        return {"identities": list_custom_fingerprint_aliases()}
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/test-images")
def custom_fingerprint_test_images():
    try:
        return {"images": list_custom_fingerprint_test_images()}
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/verify")
def verify_custom_fingerprint_route(
    claimed_alias: str = Form(...),
    probe_alias: str = Form(...),
    image_name: str = Form(...),
):
    try:
        return verify_custom_fingerprint_test_image(claimed_alias, probe_alias, image_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/verify-upload")
async def verify_custom_fingerprint_upload_route(
    claimed_alias: str = Form(...), image: UploadFile = File(...)
):
    """Verify an uploaded fingerprint without writing into project artifacts."""
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, TIFF, or WebP fingerprint image.")
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The fingerprint image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Fingerprint image exceeds the 8 MiB limit.")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        return verify_custom_fingerprint_upload(temporary_path, claimed_alias)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Fingerprint model or enrollment templates are unavailable.") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
