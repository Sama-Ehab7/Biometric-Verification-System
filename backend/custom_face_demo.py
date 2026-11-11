"""Routes for the isolated custom Face demo."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.verification.custom_face import IDENTITIES, identify_custom_face, verify_custom_face

router = APIRouter(prefix="/api/custom/face", tags=["custom-face"])
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _image_bytes(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 8 MiB limit.")
    return data


@router.get("/status")
def status():
    return {"available": True, "identities": list(IDENTITIES),
            "models": ["svm", "random_forest"], "model_scope": "custom team dataset"}


@lru_cache(maxsize=1)
def _face_detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV face detector is unavailable")
    return detector


@router.post("/verify")
async def verify_route(claimed_identity: str = Form(...), model: str = Form("svm"),
                       image: UploadFile = File(...)):
    try:
        return verify_custom_face(await _image_bytes(image), claimed_identity.strip().lower(), model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Custom Face model artifacts are not available.") from exc


@router.post("/identify")
async def identify_route(model: str = Form("svm"), image: UploadFile = File(...)):
    try:
        return identify_custom_face(await _image_bytes(image), model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Custom Face model artifacts are not available.") from exc


@router.post("/verify-camera")
async def verify_camera_route(claimed_identity: str = Form(...), model: str = Form("svm"),
                              image: UploadFile = File(...)):
    """Live-camera guard: require exactly one detectable face, then use unchanged custom features."""
    data = await _image_bytes(image)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(400, "Unable to decode the camera frame.")
    gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    try:
        faces = _face_detector().detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(40, 40))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if len(faces) == 0:
        raise HTTPException(422, "No face detected in the camera frame. Center one face and try again.")
    if len(faces) > 1:
        raise HTTPException(422, "Multiple faces detected. Only one person should be in frame.")
    try:
        return {**verify_custom_face(data, claimed_identity.strip().lower(), model),
                "faces_detected": 1, "camera_guard": "exactly one face detected; existing custom feature extraction applied"}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Custom Face model artifacts are not available.") from exc

