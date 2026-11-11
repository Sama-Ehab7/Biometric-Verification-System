"""Traditional-ML command recognizer on the validated, converted custom dataset.

Speaker authentication and permission authorization remain independent systems;
this classifier cannot authorize actions.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/processed/custom_audio_v3/converted/commands"
MODEL_DIR = ROOT / "saved_models/custom/commands_v3"
RESULT_DIR = ROOT / "results/custom/commands_v3"
COMMANDS = ("open", "close", "start", "stop", "help")
SPEAKERS = ("nour", "sama")
TARGET_RATE = 32000


def extract_command_features(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".wav":
        raise ValueError("Only WAV command recordings can be decoded in this environment")
    info = sf.info(str(path))
    if info.channels < 1 or info.duration <= 0:
        raise ValueError("Audio is empty or has invalid metadata")
    signal, sample_rate = librosa.load(str(path), sr=None, mono=True)
    if signal.size == 0:
        raise ValueError("Audio is empty")
    if sample_rate != TARGET_RATE:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=TARGET_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=TARGET_RATE, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)
    values = []
    for row in mfcc:
        values.extend((float(row.mean()), float(row.std())))
    values.extend((float(zcr.mean()), float(zcr.std()), float(rms.mean()), float(rms.std())))
    return np.asarray(values, dtype=np.float32)


def _load_split(split: str):
    x, y, speakers, files = [], [], [], []
    for speaker in SPEAKERS:
        for command in COMMANDS:
            folder = DATA_ROOT / split / speaker / command
            if not folder.is_dir():
                continue
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                if path.suffix.lower() != ".wav":
                    continue
                x.append(extract_command_features(path))
                y.append(command)
                speakers.append(speaker)
                files.append(path.relative_to(DATA_ROOT).as_posix())
    if not x:
        raise ValueError(f"No decodable WAV command clips in split {split}")
    return np.vstack(x), np.asarray(y), speakers, files


def train_custom_command_classifier() -> dict:
    if MODEL_DIR.exists() or RESULT_DIR.exists():
        raise FileExistsError("Custom command v3 outputs exist; refusing to overwrite")
    x_train, y_train, train_speakers, _ = _load_split("train")
    x_val, y_val, val_speakers, _ = _load_split("validation")
    candidates = {
        "svm": SVC(C=1.0, kernel="linear", probability=True, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
    }
    fitted = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        fitted[name] = {"model": model, "validation_macro_f1": float(f1_score(y_val, pred, average="macro")),
                        "validation_accuracy": float(accuracy_score(y_val, pred))}
    selected = max(fitted, key=lambda n: (fitted[n]["validation_macro_f1"], fitted[n]["validation_accuracy"]))
    model = fitted[selected]["model"]
    val_probabilities = model.predict_proba(x_val)
    val_indices = {str(label): i for i, label in enumerate(model.classes_)}
    correct_confidence = np.asarray([val_probabilities[i, val_indices[str(label)]]
                                     for i, label in enumerate(y_val)])
    confidence_threshold = float(correct_confidence.min())
    # The held-out split is loaded only after validation model selection ends.
    x_test, y_test, test_speakers, test_files = _load_split("test")
    test_probabilities = model.predict_proba(x_test)
    test_indices = test_probabilities.argmax(axis=1)
    test_confidence = test_probabilities.max(axis=1)
    raw_test_pred = model.classes_[test_indices]
    test_pred = np.asarray([str(label) if score >= confidence_threshold else "unknown"
                            for label, score in zip(raw_test_pred, test_confidence)])
    test_known = test_pred != "unknown"
    report = {
        "scope": "command classification only; no speaker authentication or execution",
        "dataset": str(DATA_ROOT.relative_to(ROOT)),
        "conversion_manifest": "data/processed/custom_audio_v3/audio_conversion_integrity_manifest.csv",
        "usable_wav_counts": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "speaker_distribution": {"train": sorted(set(train_speakers)),
                                 "validation": sorted(set(val_speakers)),
                                 "test": sorted(set(test_speakers))},
        "speaker_counts": {split: {speaker: int(sum(s == speaker for s in speakers))
                                    for speaker in SPEAKERS}
                           for split, speakers in (("train", train_speakers),
                                                   ("validation", val_speakers),
                                                   ("test", test_speakers))},
        "class_distribution": {split: {command: int(sum(label == command for label in labels))
                                         for command in COMMANDS}
                               for split, labels in (("train", y_train), ("validation", y_val), ("test", y_test))},
        "selection": {name: {"validation_macro_f1": data["validation_macro_f1"],
                             "validation_accuracy": data["validation_accuracy"]}
                      for name, data in fitted.items()},
        "selected_model": selected,
        "unknown_rejection": {"confidence_threshold": confidence_threshold,
                              "selection": "minimum correct-class validation confidence; no unknown validation samples"},
        "test": {"accuracy": float(accuracy_score(y_test, test_pred)),
                 "macro_f1": float(f1_score(y_test, test_pred, labels=list(COMMANDS), average="macro", zero_division=0)),
                 "coverage": float(np.mean(test_known)),
                 "selective_accuracy": float(np.mean(test_pred[test_known] == y_test[test_known])) if np.any(test_known) else None,
                 "per_command": classification_report(y_test, test_pred, labels=list(COMMANDS),
                                                      output_dict=True, zero_division=0),
                 "per_speaker": {speaker: {"accuracy": float(accuracy_score(y_test[np.asarray(test_speakers) == speaker], test_pred[np.asarray(test_speakers) == speaker])),
                                             "samples": int(sum(s == speaker for s in test_speakers))}
                                 for speaker in SPEAKERS},
                 "confusion_matrix_labels": [*COMMANDS, "unknown"],
                 "confusion_matrix": confusion_matrix(y_test, test_pred, labels=[*COMMANDS, "unknown"]).tolist(),
                 "rows": [{"file": f, "speaker": s, "actual": str(a), "predicted": str(p), "confidence": float(c)}
                          for f, s, a, p, c in zip(test_files, test_speakers, y_test, test_pred, test_confidence)]},
        "limitation": "The command corpus has four training clips, two validation clips and two test clips per command (one clip per speaker in each split per command). The test set is small and the unknown confidence threshold has no unknown-command/noise examples.",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    joblib.dump({"model": model, "commands": list(COMMANDS), "feature_dimension": 30,
                 "sample_rate": TARGET_RATE, "selected_model": selected,
                 "confidence_threshold": confidence_threshold}, MODEL_DIR / "classifier.joblib")
    (RESULT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@lru_cache(maxsize=1)
def _load_classifier():
    return joblib.load(MODEL_DIR / "classifier.joblib")


def classify_custom_command(audio_bytes: bytes) -> dict:
    import io
    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError("Upload a valid WAV recording") from exc
    if data.size == 0:
        raise ValueError("Audio is empty")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size < sample_rate * 0.2:
        raise ValueError("Command recording is too short")
    if sample_rate != TARGET_RATE:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_RATE)
    mfcc = librosa.feature.mfcc(y=data, sr=TARGET_RATE, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(data)
    rms = librosa.feature.rms(y=data)
    features = []
    for row in mfcc:
        features.extend((float(row.mean()), float(row.std())))
    features.extend((float(zcr.mean()), float(zcr.std()), float(rms.mean()), float(rms.std())))
    bundle = _load_classifier()
    model = bundle["model"]
    vector = np.asarray(features, np.float32).reshape(1, -1)
    probabilities = model.predict_proba(vector)[0]
    index = int(np.argmax(probabilities))
    return {"command": str(model.classes_[index]), "score": float(probabilities[index]),
            "model": bundle["selected_model"], "scope": "classification only; not authenticated or authorized"}

