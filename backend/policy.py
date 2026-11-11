"""The decision rules, kept free of models and files so they are easy to test."""
from __future__ import annotations

from dataclasses import dataclass

FACE_ACCEPT = "face_accept"
FACE_REJECT = "face_reject"
VOICE_ACCEPT = "voice_accept"
VOICE_REJECT = "voice_reject"
VOICE_REQUIRED = "voice_required"


@dataclass(frozen=True)
class HierarchicalPolicy:
    """Validated Session-2 policy (src/verification/hierarchical_verification.py,
    apply_sequential_policy): face first, voice only when the face is unsure."""

    low_face: float
    high_face: float
    voice: float

    @classmethod
    def from_mapping(cls, m) -> "HierarchicalPolicy":
        return cls(float(m["low_face_threshold"]), float(m["high_face_threshold"]), float(m["voice_threshold"]))

    def face_stage(self, face_score: float) -> str | None:
        """FACE_ACCEPT / FACE_REJECT when the face is confident, None when voice must decide."""
        if face_score >= self.high_face:
            return FACE_ACCEPT
        if face_score <= self.low_face:
            return FACE_REJECT
        return None

    def voice_stage(self, voice_score: float) -> str:
        return VOICE_ACCEPT if voice_score >= self.voice else VOICE_REJECT

    def as_dict(self) -> dict:
        return {"low_face_threshold": self.low_face, "high_face_threshold": self.high_face, "voice_threshold": self.voice}


def face_only_stage(face_score: float, threshold: float) -> str:
    """Single-threshold decision used when no voice model is available."""
    return FACE_ACCEPT if face_score >= threshold else FACE_REJECT


def is_accepted(stage: str) -> bool:
    return stage in (FACE_ACCEPT, VOICE_ACCEPT)
