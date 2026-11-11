"""Face: image -> face crop -> sequence features (same steps as the training data)."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from . import config

if not hasattr(cv2, "CascadeClassifier"):
    raise RuntimeError(
        "This OpenCV build has no CascadeClassifier (OpenCV 5.x removed it). "
        'Install OpenCV 4.x:  pip install "opencv-python<5"'
    )

# Project code (src/preprocessing, src/features).
from face_preprocessing import detect_face  # noqa: E402
from face_features import IMAGE_SIZE, extract_hog_features, extract_lbp_features  # noqa: E402


class NoFaceError(ValueError):
    """Raised when an image can't be read."""


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise NoFaceError("Could not read the image file.")
    h, w = img.shape[:2]
    scale = config.MAX_IMAGE_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def crop_face(img: np.ndarray) -> np.ndarray | None:
    """Same as src/preprocessing/face_preprocessing.preprocess_image:
    largest detection, 15% padding, 128x128 grayscale."""
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
    center = (IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2)
    while len(out) < n:
        M = cv2.getRotationMatrix2D(center, rng.uniform(-4, 4), rng.uniform(0.96, 1.04))
        M[:, 2] += rng.uniform(-3, 3, size=2)
        f = cv2.warpAffine(face, M, IMAGE_SIZE, borderMode=cv2.BORDER_REFLECT)
        f = np.clip(f.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8), 0, 255)
        out.append(f.astype(np.uint8))
    return out


def sequence_features(faces: list[np.ndarray]) -> np.ndarray:
    """Same aggregation as src/features/face_features.extract_sequence_features:
    [HOG mean, HOG std, LBP mean, LBP std] over the frames."""
    if len(faces) < config.MIN_FRAMES:
        per = int(np.ceil(config.MIN_FRAMES / len(faces)))
        faces = [a for f in faces for a in augment(f, per)][: config.MIN_FRAMES]
    hog = np.asarray([extract_hog_features(f) for f in faces])
    lbp = np.asarray([extract_lbp_features(f) for f in faces])
    return np.concatenate([hog.mean(0), hog.std(0), lbp.mean(0), lbp.std(0)]).astype(np.float32)


def faces_from_images(images: list[bytes]) -> tuple[list[np.ndarray], int]:
    """Crop the face from each image. Returns (faces, number of images without a face)."""
    faces = [f for f in (crop_face(decode_image(b)) for b in images) if f is not None]
    return faces, len(images) - len(faces)


def to_data_url(face: np.ndarray) -> str | None:
    """PNG data URL of a face crop, so the UI can show what was detected."""
    ok, png = cv2.imencode(".png", face)
    return "data:image/png;base64," + base64.b64encode(png).decode() if ok else None
