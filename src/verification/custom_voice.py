"""Small, isolated Traditional-ML speaker verification experiment.

This uses only canonical clips from the duplicate-grouped custom audio
manifest. Its very small corpus makes all scores exploratory; unknown-speaker
rejection is not validated because no unknown-person recordings are available.
"""
from __future__ import annotations

import csv
import io
import json
from functools import lru_cache
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/custom_audio_v3/custom_voice_grouped_split_manifest.csv"
DATA_ROOT = ROOT / "data/processed/custom_audio_v3/converted/voice"
MODEL_DIR = ROOT / "saved_models/custom/voice_v1"
RESULT_DIR = ROOT / "results/custom/voice_v1"
IDENTITIES = ("nour", "sama")
SAMPLE_RATE = 32_000
FEATURE_DIMENSION = 30


def extract_voice_features_from_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Match the project's 13 MFCC mean/std + ZCR/RMS mean/std layout."""
    signal = np.asarray(signal, dtype=np.float32)
    if sample_rate <= 0 or signal.ndim != 1 or signal.size < int(sample_rate * 0.2):
        raise ValueError("Audio must be mono and at least 0.2 seconds long")
    if not np.isfinite(signal).all() or not np.any(signal):
        raise ValueError("Audio is silent or contains invalid samples")
    if sample_rate != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)
    values = []
    for row in mfcc:
        values.extend((float(row.mean()), float(row.std())))
    values.extend((float(zcr.mean()), float(zcr.std()), float(rms.mean()), float(rms.std())))
    return np.asarray(values, dtype=np.float32)


def extract_voice_features(path: Path) -> np.ndarray:
    signal, rate = librosa.load(str(path), sr=None, mono=True)
    return extract_voice_features_from_signal(signal, rate)


def _load_manifest_split(split: str):
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing custom Voice integrity manifest: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["new_split"] == split and row["use_as_split_sample"].lower() == "true"]
    if not rows:
        raise ValueError(f"No canonical custom Voice records in {split}")
    vectors, labels, paths = [], [], []
    for row in sorted(rows, key=lambda item: (item["identity"], item["source_path"])):
        # The manifest relative_output includes voice/...; resolve from converted/.
        path = ROOT / "data/processed/custom_audio_v3/converted" / Path(row["relative_output"])
        if not path.is_file():
            raise FileNotFoundError(path)
        vector = extract_voice_features(path)
        if vector.shape != (FEATURE_DIMENSION,):
            raise ValueError(f"Unexpected voice feature width {vector.shape}: {path}")
        vectors.append(vector)
        labels.append(row["identity"])
        paths.append(row["source_path"])
    return np.vstack(vectors), np.asarray(labels), paths


def _pair_features(references: dict[str, np.ndarray], probes: np.ndarray,
                   actual: np.ndarray):
    pairs, labels = [], []
    for index, (probe, truth) in enumerate(zip(probes, actual)):
        for claim in IDENTITIES:
            reference = references[claim]
            pairs.append(np.abs(probe - reference))
            labels.append(int(claim == truth))
    return np.asarray(pairs, np.float32), np.asarray(labels, np.int32)


def _validation_threshold(labels, scores):
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    ranked = []
    for threshold in candidates:
        pred = scores >= threshold
        ranked.append((float(f1_score(labels, pred, zero_division=0)),
                       float(threshold)))
    return max(ranked)[1]


def _metrics(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"trials": int(len(labels)), "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, pred)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "far": float(fp / (fp + tn)) if fp + tn else None,
            "frr": float(fn / (fn + tp)) if fn + tp else None,
            "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def train_custom_voice_models() -> dict:
    if MODEL_DIR.exists() or RESULT_DIR.exists():
        raise FileExistsError("Custom Voice v1 outputs exist; refusing to overwrite")
    x_train, y_train, train_paths = _load_manifest_split("train")
    x_val, y_val, val_paths = _load_manifest_split("validation")
    if set(y_train) != set(IDENTITIES) or set(y_val) != set(IDENTITIES):
        raise ValueError("Training and validation must each contain both custom identities")

    # Create leave-one-recording-out genuine trials and all-identity impostor trials.
    per_identity = {identity: x_train[y_train == identity] for identity in IDENTITIES}
    if any(len(per_identity[identity]) < 2 for identity in IDENTITIES):
        raise ValueError("Each identity needs at least two unique training recordings for leave-one-out trials")
    refs = {identity: np.mean(per_identity[identity], axis=0) for identity in IDENTITIES}
    train_pairs, train_labels = [], []
    for identity in IDENTITIES:
        for probe_index, probe in enumerate(per_identity[identity]):
            own_ref = np.mean(np.delete(per_identity[identity], probe_index, axis=0), axis=0)
            # A fallback is unnecessary with >=2 unique records, but guard it explicitly.
            if not np.isfinite(own_ref).all():
                raise ValueError("Could not form a leave-one-out enrollment reference")
            for claim in IDENTITIES:
                reference = own_ref if claim == identity else refs[claim]
                train_pairs.append(np.abs(probe - reference))
                train_labels.append(int(claim == identity))
    train_pairs = np.asarray(train_pairs, np.float32)
    train_labels = np.asarray(train_labels, np.int32)

    estimators = {
        "svm": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0,
                             probability=True, class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=42, n_jobs=-1),
    }
    val_refs = refs
    val_pairs, val_labels = _pair_features(val_refs, x_val, y_val)
    candidates = {}
    for name, estimator in estimators.items():
        estimator.fit(train_pairs, train_labels)
        positive = list(estimator.classes_).index(1)
        scores = estimator.predict_proba(val_pairs)[:, positive]
        threshold = _validation_threshold(val_labels, scores)
        candidates[name] = {"model": estimator, "scores": scores, "threshold": threshold,
                            "metrics": _metrics(val_labels, scores, threshold)}
    selected = max(candidates, key=lambda name: (candidates[name]["metrics"]["f1"],
                                                   candidates[name]["metrics"]["accuracy"],
                                                   name == "svm"))
    chosen = candidates[selected]

    # Identification confidence abstention is only calibrated on known validation
    # samples; it is explicitly not evidence of unknown-speaker rejection.
    identity_model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0,
                                  probability=True, class_weight="balanced", random_state=42))
    identity_model.fit(x_train, y_train)
    val_identity_scores = identity_model.predict_proba(x_val)
    class_index = {str(label): i for i, label in enumerate(identity_model.classes_)}
    confidence_threshold = float(min(val_identity_scores[i, class_index[str(label)]]
                                     for i, label in enumerate(y_val)))

    # Freeze model and thresholds before opening the test split.
    x_test, y_test, test_paths = _load_manifest_split("test")
    test_refs = refs
    test_pairs, test_labels = _pair_features(test_refs, x_test, y_test)
    positive = list(chosen["model"].classes_).index(1)
    test_scores = chosen["model"].predict_proba(test_pairs)[:, positive]
    test_metrics = _metrics(test_labels, test_scores, chosen["threshold"])
    id_prob = identity_model.predict_proba(x_test)
    id_conf = id_prob.max(axis=1)
    id_raw = identity_model.classes_[id_prob.argmax(axis=1)]
    id_pred = np.asarray([str(label) if confidence >= confidence_threshold else "unknown"
                          for label, confidence in zip(id_raw, id_conf)])

    report = {
        "scope": "experimental custom speaker verification and closed-set identification",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "split_protocol": "training pairs from canonical train recordings; model and verification threshold selected on validation; test split opened once after selection",
        "feature_method": "32 kHz mono; 13 MFCC mean/std + ZCR mean/std + RMS mean/std",
        "feature_dimension": FEATURE_DIMENSION,
        "counts": {split: {identity: int(np.sum(labels == identity)) for identity in IDENTITIES}
                   for split, labels in (("train", y_train), ("validation", y_val), ("test", y_test))},
        "source_paths": {"train": train_paths, "validation": val_paths, "test": test_paths},
        "training_trials": int(len(train_labels)),
        "validation_model_selection": {name: data["metrics"] for name, data in candidates.items()},
        "selected_verifier": selected,
        "threshold_split": "validation",
        "test_verification": test_metrics,
        "test_identification": {"samples": len(y_test),
                                "accuracy_with_abstention": float(accuracy_score(y_test, id_pred)),
                                "unknown_predictions": int(np.sum(id_pred == "unknown")),
                                "confidence_threshold": confidence_threshold,
                                "threshold_split": "validation",
                                "predictions": [{"source": path, "actual": str(actual),
                                                 "predicted": str(pred), "confidence": float(conf)}
                                                for path, actual, pred, conf in zip(test_paths, y_test, id_pred, id_conf)]},
        "limitations": ["Only 5 unique Nour and 10 unique Sama voice recordings exist; train has 3 and 6 clips, respectively.",
                        "Six held-out clips create only 12 claimed-speaker trials; estimates have high uncertainty.",
                        "No independent unknown speakers are available; unknown rejection is not validated.",
                        "Recordings may differ in channel, environment, language and content; recording/session metadata is unavailable."],
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    joblib.dump({"model": chosen["model"], "threshold": chosen["threshold"],
                 "threshold_split": "validation", "identities": list(IDENTITIES),
                 "enrollment_references": refs, "feature_dimension": FEATURE_DIMENSION,
                 "sample_rate": SAMPLE_RATE, "selected_model": selected}, MODEL_DIR / "verifier.joblib")
    joblib.dump({"model": identity_model, "confidence_threshold": confidence_threshold,
                 "threshold_split": "validation", "identities": list(IDENTITIES),
                 "feature_dimension": FEATURE_DIMENSION, "sample_rate": SAMPLE_RATE},
                MODEL_DIR / "identifier.joblib")
    (RESULT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@lru_cache(maxsize=2)
def load_custom_voice_bundle(task: str):
    if task not in {"verifier", "identifier"}:
        raise ValueError("Task must be verifier or identifier")
    return joblib.load(MODEL_DIR / f"{task}.joblib")


def _read_uploaded_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        signal, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("Upload a valid WAV recording") from exc
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Audio is empty or invalid")
    return extract_voice_features_from_signal(signal.mean(axis=1), int(rate))


def verify_custom_voice(audio_bytes: bytes, claimed_identity: str) -> dict:
    if claimed_identity not in IDENTITIES:
        raise ValueError("Unknown custom Voice identity")
    bundle = load_custom_voice_bundle("verifier")
    feature = _read_uploaded_audio(audio_bytes)
    model = bundle["model"]
    reference = bundle["enrollment_references"][claimed_identity]
    vector = np.abs(feature - reference).reshape(1, -1)
    score = float(model.predict_proba(vector)[0, list(model.classes_).index(1)])
    threshold = float(bundle["threshold"])
    accepted = score >= threshold
    return {"claimed_identity": claimed_identity, "model": bundle["selected_model"],
            "score": score, "threshold": threshold, "threshold_split": bundle["threshold_split"],
            "decision": "ACCEPT" if accepted else "REJECT", "accepted": accepted}


def identify_custom_voice(audio_bytes: bytes) -> dict:
    bundle = load_custom_voice_bundle("identifier")
    feature = _read_uploaded_audio(audio_bytes)
    probabilities = bundle["model"].predict_proba(feature.reshape(1, -1))[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    threshold = float(bundle["confidence_threshold"])
    identity = str(bundle["model"].classes_[index]) if confidence >= threshold else "unknown"
    return {"identity": identity, "confidence": confidence, "threshold": threshold,
            "threshold_split": bundle["threshold_split"], "known": identity != "unknown",
            "unknown_rejection_validated": False}
