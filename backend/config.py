"""Paths and settings used by the backend, in one place."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root (override with BVS_PROJECT_ROOT, e.g. for tests).
ROOT = Path(os.environ.get("BVS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))

# The backend reuses the project's own preprocessing and feature code.
CODE_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/preprocessing", "src/features"):
    path = str(CODE_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

# Trained models and Session-1 enrollment features (git-ignored).
FACE_MODEL = ROOT / "saved_models" / "face" / "svm_face_security.joblib"
FACE_TRAIN = ROOT / "data" / "processed" / "splits" / "face_train.csv"
VOICE_MODEL = ROOT / "saved_models" / "voice" / "svm_voice_security.joblib"
VOICE_TRAIN = ROOT / "data" / "processed" / "splits" / "voice_train.csv"

# Validated hierarchical policy (joblib first, JSON record as fallback).
POLICY_FILE = ROOT / "saved_models" / "hierarchical" / "hierarchical_policy_audit.joblib"
POLICY_JSON = ROOT / "results" / "hierarchical" / "selected_thresholds_audit.json"

# People enrolled through the API (biometric data - git-ignored).
ENROLL_FILE = Path(os.environ.get("BVS_ENROLL_FILE", Path(__file__).resolve().parent / "enrolled" / "enrolled.json"))

FACE_ONLY_FAR = 0.01   # face-only operating point when no hierarchical policy is available
MIN_FRAMES = 8         # a single photo is augmented to this many frames
MAX_IMAGE_SIDE = 800   # large photos are shrunk first (the Haar detector is slow on big images)

# Pages allowed to call the API from the browser.
CORS_ORIGINS = ["http://127.0.0.1:5500", "http://localhost:5500"]
