"""People enrolled through the API. Stored as JSON (git-ignored: it is biometric data)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def clean_name(name: str) -> str:
    return re.sub(r"[^\w\- ]", "", name).strip()[:40]


class EnrollmentStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.people: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.people))

    def __contains__(self, name: str) -> bool:
        return name in self.people

    def add(self, name: str, face_ref: np.ndarray, voice_ref: np.ndarray | None, samples: int) -> None:
        self.people[name] = {
            "face": face_ref.tolist(),
            "voice": voice_ref.tolist() if voice_ref is not None else None,
            "samples": samples,
        }
        self._save()

    def remove(self, name: str) -> bool:
        if self.people.pop(name, None) is None:
            return False
        self._save()
        return True

    def references(self, name: str) -> tuple[np.ndarray, np.ndarray | None]:
        p = self.people[name]
        voice = np.asarray(p["voice"], np.float32) if p.get("voice") is not None else None
        return np.asarray(p["face"], np.float32), voice

    def summary(self) -> list[dict]:
        return [{"id": k, "samples": v["samples"], "voice": v.get("voice") is not None}
                for k, v in sorted(self.people.items())]
