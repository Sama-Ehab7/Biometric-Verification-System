"""Isolated Traditional-ML pipeline for the collected custom Face images.

This module is intentionally separate from the validated VidTIMIT pipeline.
The custom corpus is stored at data/raw/custom/face and has train/validation/
test splits. Its preprocessing uses a center square because the existing Haar
detector misses some plainly visible custom faces; no original preprocessor is
changed.
"""
from __future__ import annotations

import json
import csv
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "raw" / "custom" / "face"
MANIFEST = ROOT / "data" / "processed" / "custom_face_v1" / "custom_face_split_manifest.csv"
MODEL_DIR = ROOT / "saved_models" / "custom" / "face_v4"
RESULT_DIR = ROOT / "results" / "custom" / "face_v4"
IDENTITIES = ("nour", "sama")
IMAGE_SIZE = (128, 128)


def extract_custom_face_features(image: np.ndarray) -> np.ndarray:
    """Center-crop and extract the project HOG/LBP layout for one image."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Expected a grayscale or BGR image")
    height, width = gray.shape
    side = min(height, width)
    x1, y1 = (width - side) // 2, (height - side) // 2
    crop = gray[y1:y1 + side, x1:x1 + side]
    crop = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    hog_vector = hog(crop, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), block_norm="L2-Hys",
                     visualize=False, feature_vector=True).astype(np.float32)
    lbp_image = local_binary_pattern(crop, P=8, R=1, method="uniform")
    lbp, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp = lbp.astype(np.float32)
    lbp /= lbp.sum() + 1e-8
    # Keep the same [HOG mean, HOG std, LBP mean, LBP std] ordering as the
    # original sequence extractor; a single still image has zero frame std.
    return np.concatenate((hog_vector, np.zeros_like(hog_vector), lbp,
                           np.zeros_like(lbp))).astype(np.float32)


def _read_split(split: str):
    features, labels, files = [], [], []
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing leakage-screened Face manifest: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["new_split"] == split]
    for row in sorted(rows, key=lambda item: (item["identity"], item["path"])):
        path = ROOT / Path(row["path"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable custom Face image: {path}")
        features.append(extract_custom_face_features(image))
        labels.append(row["identity"])
        files.append(path.relative_to(DATA_ROOT).as_posix())
    if not features:
        raise ValueError(f"No custom Face images in split {split}")
    return np.vstack(features), np.asarray(labels), files


def _pairs(model, x, y):
    classes = list(model.classes_)
    scores, truth, claims = [], [], []
    for sample, actual in zip(x, y):
        probabilities = model.predict_proba(sample.reshape(1, -1))[0]
        for index, claim in enumerate(classes):
            scores.append(float(probabilities[index]))
            truth.append(int(claim == actual))
            claims.append(claim)
    return np.asarray(scores), np.asarray(truth), claims


def _best_validation_threshold(scores, labels):
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best = None
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        # Prefer the larger threshold on equal F1 to be conservative.
        candidate = (float(f1), float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[1]


def _metrics(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "far": float(fp / (fp + tn)) if fp + tn else None,
        "frr": float(fn / (fn + tp)) if fn + tp else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "trials": int(len(labels)),
    }


def train_custom_face_models() -> dict:
    """Train SVM/RF on train only, select verification thresholds on validation.

    Outputs are isolated and this function refuses to overwrite any prior run.
    """
    if MODEL_DIR.exists() or RESULT_DIR.exists():
        raise FileExistsError("Custom Face v4 outputs already exist; refusing to overwrite")
    x_train, y_train, _ = _read_split("train")
    x_val, y_val, _ = _read_split("validation")
    estimators = {
        "svm": SVC(C=1.0, kernel="linear", probability=True,
                   class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=42, n_jobs=-1),
    }
    trained = {}
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        val_scores, val_truth, val_claims = _pairs(estimator, x_val, y_val)
        threshold = _best_validation_threshold(val_scores, val_truth)
        val_probabilities = estimator.predict_proba(x_val)
        class_index = {str(label): i for i, label in enumerate(estimator.classes_)}
        correct_validation_scores = np.asarray([
            val_probabilities[i, class_index[str(label)]] for i, label in enumerate(y_val)
        ])
        identification_threshold = float(correct_validation_scores.min())
        trained[name] = {
            "bundle": {"model": estimator, "identities": list(estimator.classes_),
                       "feature_dimension": int(x_train.shape[1]), "threshold": threshold,
                       "identification_threshold": identification_threshold,
                       "feature_method": "custom center-square 128x128 grayscale + HOG/LBP",
                       "threshold_split": "validation"},
            "threshold": threshold,
            "identification_threshold": identification_threshold,
            "validation": _metrics(val_scores, val_truth, threshold),
        }
    # Do not access the held-out split until model and threshold selection is frozen.
    x_test, y_test, test_files = _read_split("test")
    for name, data in trained.items():
        test_scores, test_truth, test_claims = _pairs(estimators[name], x_test, y_test)
        data["test"] = _metrics(test_scores, test_truth, data["threshold"])
        data["test_rows"] = [{"file": test_files[i // len(IDENTITIES)],
                              "actual_identity": str(y_test[i // len(IDENTITIES)]),
                              "claimed_identity": test_claims[i], "label": int(test_truth[i]),
                              "score": float(test_scores[i]),
                              "decision": int(test_scores[i] >= data["threshold"])}
                             for i in range(len(test_truth))]
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    for name, data in trained.items():
        joblib.dump(data["bundle"], MODEL_DIR / f"{name}.joblib")
        data.pop("bundle")
    report = {"dataset": str(DATA_ROOT.relative_to(ROOT)),
              "split_manifest": str(MANIFEST.relative_to(ROOT)),
              "split_policy": "perceptual near-duplicate groups are kept in one split; train=fitting; validation=selection; test is first read after selection is frozen, then used for final evaluation",
              "test_identity_counts": {i: int(np.sum(y_test == i)) for i in IDENTITIES},
              "validation_identity_counts": {i: int(np.sum(y_val == i)) for i in IDENTITIES},
              "train_identity_counts": {i: int(np.sum(y_train == i)) for i in IDENTITIES},
              "models": trained,
              "limitations": ["Only two identities and a small image dataset are available.",
                              "Validation-derived thresholds have high uncertainty.",
                              "No independent unknown-identity samples exist, so open-set rejection is not validated."]}
    (RESULT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@lru_cache(maxsize=2)
def load_custom_face_model(model_name: str):
    if model_name not in {"svm", "random_forest"}:
        raise ValueError("Model must be svm or random_forest")
    bundle = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    return bundle


def evaluate_custom_face_identification() -> dict:
    """Evaluate closed-set identity predictions and threshold rejection on test only."""
    output = RESULT_DIR / "identification_evaluation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    x_test, y_test, files = _read_split("test")
    results = {}
    for name in ("svm", "random_forest"):
        bundle = load_custom_face_model(name)
        model = bundle["model"]
        probabilities = model.predict_proba(x_test)
        max_indices = probabilities.argmax(axis=1)
        max_scores = probabilities.max(axis=1)
        threshold = float(bundle["identification_threshold"])
        predicted = np.asarray([str(model.classes_[i]) if score >= threshold else "unknown"
                                for i, score in zip(max_indices, max_scores)])
        results[name] = {
            "threshold": threshold,
            "threshold_split": "validation",
            "test_images": len(y_test),
            "accuracy_including_unknown_rejection": float(np.mean(predicted == y_test)),
            "unknown_predictions": int(np.sum(predicted == "unknown")),
            "per_image": [{"file": f, "actual": str(actual), "predicted": str(pred),
                           "score": float(score)}
                          for f, actual, pred, score in zip(files, y_test, predicted, max_scores)],
        }
    report = {"evaluation_split": "test", "models": results,
              "limitation": "Only known identities occur in the test split; unknown-identity rejection is not measured."}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def verify_custom_face(image_bytes: bytes, claimed_identity: str, model_name: str = "svm"):
    if claimed_identity not in IDENTITIES:
        raise ValueError("Unknown custom Face identity")
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to read the uploaded image")
    bundle = load_custom_face_model(model_name)
    feature = extract_custom_face_features(image)
    model = bundle["model"]
    class_index = list(model.classes_).index(claimed_identity)
    score = float(model.predict_proba(feature.reshape(1, -1))[0, class_index])
    threshold = float(bundle["threshold"])
    accepted = score >= threshold
    return {"claimed_identity": claimed_identity, "model": model_name,
            "score": score, "threshold": threshold,
            "decision": "ACCEPT" if accepted else "REJECT", "accepted": accepted}


def identify_custom_face(image_bytes: bytes, model_name: str = "svm"):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to read the uploaded image")
    bundle = load_custom_face_model(model_name)
    model = bundle["model"]
    probabilities = model.predict_proba(extract_custom_face_features(image).reshape(1, -1))[0]
    index = int(np.argmax(probabilities))
    score = float(probabilities[index])
    threshold = float(bundle["identification_threshold"])
    identity = str(model.classes_[index]) if score >= threshold else "unknown"
    return {"identity": identity, "model": model_name, "score": score,
            "threshold": threshold, "known": identity != "unknown"}

