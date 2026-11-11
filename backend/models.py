"""Loading the trained SVMs, the Session-1 enrollment references and the policy."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .policy import HierarchicalPolicy


class MissingFileError(RuntimeError):
    pass


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise MissingFileError(
            f"Missing {what}: {path}\n"
            "These files are git-ignored. Copy them from the machine that ran the "
            "pipeline, or re-run the pipeline (see backend/README.md)."
        )
    return path


@dataclass
class Verifier:
    """One trained SVM plus the enrollment reference of every speaker."""

    model: object
    columns: list[str]
    thresholds: dict[float, float]
    references: dict[str, np.ndarray]
    positive: int = field(init=False)

    def __post_init__(self):
        self.positive = list(self.model.classes_).index(1)

    @classmethod
    def load(cls, model_path: Path, train_path: Path, what: str) -> "Verifier":
        bundle = joblib.load(require(model_path, f"{what} model"))
        columns = list(bundle["feature_columns"])
        train = pd.read_csv(require(train_path, f"{what} enrollment features"))
        refs = {str(s): g[columns].to_numpy(np.float32).mean(0) for s, g in train.groupby("speaker")}
        thresholds = {float(k): float(v) for k, v in bundle.get("thresholds", {}).items()}
        return cls(bundle["model"], columns, thresholds, refs)

    def score(self, probe: np.ndarray, reference: np.ndarray) -> float:
        """Genuine probability for |probe - reference|, as in the training trials."""
        pair = np.abs(probe - reference).reshape(1, -1)
        return float(self.model.predict_proba(pair)[0, self.positive])


@dataclass
class Models:
    face: Verifier
    voice: Verifier | None
    policy: HierarchicalPolicy | None
    policy_source: str | None
    face_only_threshold: float

    @property
    def hierarchical(self) -> bool:
        return self.voice is not None and self.policy is not None

    @property
    def mode(self) -> str:
        return "hierarchical" if self.hierarchical else "face_only"


def load_policy() -> tuple[HierarchicalPolicy | None, str | None]:
    if config.POLICY_FILE.exists():
        return HierarchicalPolicy.from_mapping(joblib.load(config.POLICY_FILE)), config.POLICY_FILE.name
    if config.POLICY_JSON.exists():
        data = json.loads(config.POLICY_JSON.read_text(encoding="utf-8"))
        return HierarchicalPolicy.from_mapping(data), config.POLICY_JSON.name
    return None, None


def load_models() -> Models:
    face = Verifier.load(config.FACE_MODEL, config.FACE_TRAIN, "face")
    voice = None
    if config.VOICE_MODEL.exists() and config.VOICE_TRAIN.exists():
        voice = Verifier.load(config.VOICE_MODEL, config.VOICE_TRAIN, "voice")
    policy, source = load_policy()
    face_only = face.thresholds.get(config.FACE_ONLY_FAR) or min(face.thresholds.values())
    return Models(face, voice, policy, source, face_only)
