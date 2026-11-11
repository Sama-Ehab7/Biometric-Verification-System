"""Inference helpers for the saved real-data Fingerprint SVM verifier.

Input images may be raw grayscale/color TIFF, PNG, or another format supported
by OpenCV. They are passed through the existing preprocessing and HOG/LBP
functions. A claim is verified against database-qualified enrolled identities.
No model fitting or threshold selection occurs in this module.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_SIZE = 256

MODEL_PATH = PROJECT_ROOT / "saved_models/fingerprint/svm_fingerprint.joblib"
ENROLLMENT_SPLIT = (
    PROJECT_ROOT
    / "data/processed/fingerprint/splits/real/fingerprint_enrollment.csv"
)
TEMPLATE_ARTIFACT = (
    PROJECT_ROOT
    / "saved_models/fingerprint/fingerprint_enrollment_templates_v1.npz"
)


def create_enrollment_artifact(
    enrollment_csv: Path = ENROLLMENT_SPLIT,
    model_path: Path = MODEL_PATH,
    output_path: Path = TEMPLATE_ARTIFACT,
) -> Path:
    """Create a separate template artifact from the existing enrollment split.

    This only summarizes already extracted enrollment features. It does not
    train or modify a model. Existing output files are never overwritten.
    """
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    model_bundle = joblib.load(model_path)
    feature_columns = model_bundle.get("feature_columns")
    if not feature_columns or len(feature_columns) != 8110:
        raise ValueError("Saved model has missing or unexpected feature columns")

    grouped = {}
    seen_images = set()
    with enrollment_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"identity", "image_name", *feature_columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Enrollment file is missing columns: {sorted(missing)[:5]}")
        for row in reader:
            identity = row["identity"]
            key = (identity, row["image_name"])
            if key in seen_images:
                raise ValueError("Enrollment file contains duplicate identity/image rows")
            seen_images.add(key)
            vector = np.asarray([float(row[column]) for column in feature_columns], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Non-finite enrollment features for {identity}")
            grouped.setdefault(identity, []).append(vector)

    templates = {}
    for identity, rows in sorted(grouped.items()):
        if len(rows) != 4:
            raise ValueError(f"{identity} has {len(rows)} rows; expected 4")
        templates[str(identity)] = np.mean(np.stack(rows), axis=0)

    identities = np.asarray(sorted(templates), dtype=np.str_)
    template_matrix = np.stack([templates[identity] for identity in identities])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format=np.asarray("fingerprint_enrollment_templates_v1"),
        identities=identities,
        templates=template_matrix,
        feature_columns=np.asarray(feature_columns, dtype=np.str_),
        model_path=np.asarray(str(model_path.relative_to(PROJECT_ROOT))),
        images_per_identity=np.asarray(4),
        source_split=np.asarray(str(enrollment_csv.relative_to(PROJECT_ROOT))),
        template_method=np.asarray(
            "mean of four enrollment feature vectors"
        ),
    )
    return output_path


def _load_artifacts(model_path: Path, template_path: Path):
    bundle = joblib.load(model_path)
    with np.load(template_path, allow_pickle=False) as saved:
        if str(saved["format"]) != "fingerprint_enrollment_templates_v1":
            raise ValueError("Unsupported enrollment template artifact format")
        identities = saved["identities"].astype(str).tolist()
        artifact_columns = saved["feature_columns"].astype(str).tolist()
        matrix = np.asarray(saved["templates"], dtype=np.float32)
    if list(bundle.get("feature_columns", [])) != artifact_columns:
        raise ValueError("Model and enrollment feature ordering do not match")
    if matrix.shape != (len(identities), len(artifact_columns)) or not identities:
        raise ValueError("Enrollment template artifact is empty")
    artifact = {
        "templates": dict(zip(identities, matrix)),
        "feature_columns": artifact_columns,
    }
    return bundle, artifact


def extract_inference_features(image_path: str | Path) -> np.ndarray:
    """Apply the training preprocessing and return ordered 8,110D HOG+LBP."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Fingerprint image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read fingerprint image: {image_path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    denoised = cv2.medianBlur(enhanced, 3)
    height, width = denoised.shape
    scale = min(TARGET_SIZE / width, TARGET_SIZE / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_AREA)
    processed = cv2.copyMakeBorder(
        resized,
        top=(TARGET_SIZE - new_height) // 2,
        bottom=(TARGET_SIZE - new_height + 1) // 2,
        left=(TARGET_SIZE - new_width) // 2,
        right=(TARGET_SIZE - new_width + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )
    hog_values = hog(
        processed,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    lbp_image = local_binary_pattern(processed, P=8, R=1, method="uniform")
    lbp_values, _ = np.histogram(
        lbp_image.ravel(), bins=np.arange(0, 11), range=(0, 10)
    )
    lbp_values = lbp_values.astype(np.float32)
    lbp_values /= lbp_values.sum() + 1e-7
    features = np.concatenate([hog_values, lbp_values]).astype(np.float32, copy=False)
    if processed.shape != (256, 256):
        raise ValueError(f"Unexpected preprocessed image shape: {processed.shape}")
    if features.shape != (8110,) or not np.isfinite(features).all():
        raise ValueError("Inference feature vector is invalid or has wrong size")
    return features


def verify_fingerprint(
    image_path: str | Path,
    claimed_identity: str,
    *,
    model_path: Path = MODEL_PATH,
    template_path: Path = TEMPLATE_ARTIFACT,
) -> dict:
    """Verify one raw fingerprint image against a claimed enrolled identity.

    Returns identity, genuine-class probability score, selected threshold,
    and binary decision. `claimed_identity` must include its database prefix,
    for example ``DB1_B_101``.
    """
    bundle, artifact = _load_artifacts(model_path, template_path)
    templates = artifact["templates"]
    if claimed_identity not in templates:
        raise ValueError(f"Unknown enrolled claim: {claimed_identity}")

    probe = extract_inference_features(image_path)
    reference = np.asarray(templates[claimed_identity], dtype=np.float32)
    if reference.shape != probe.shape or not np.isfinite(reference).all():
        raise ValueError("Claimed enrollment template is invalid")

    pair = np.abs(probe - reference).reshape(1, -1)
    score = float(bundle["model"].predict_proba(pair)[0, 1])
    threshold = float(bundle["threshold"])
    return {
        "claimed_identity": claimed_identity,
        "score": score,
        "threshold": threshold,
        "decision": "ACCEPT" if score >= threshold else "REJECT",
        "model": "SVM",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Path to a raw fingerprint image")
    parser.add_argument("claimed_identity", nargs="?", help="For example DB1_B_101")
    parser.add_argument(
        "--create-enrollment-artifact",
        action="store_true",
        help="Create the new template artifact from the saved enrollment split",
    )
    args = parser.parse_args()
    if args.create_enrollment_artifact:
        print(create_enrollment_artifact())
    if not args.image or not args.claimed_identity:
        if args.create_enrollment_artifact:
            return
        parser.error("image and claimed_identity are required for verification")
    print(verify_fingerprint(args.image, args.claimed_identity))


if __name__ == "__main__":
    main()
