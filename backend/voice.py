"""Voice: audio -> 32 kHz mono WAV -> the project's voice features."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np

from . import config  # noqa: F401  (puts src/ on the import path - must come first)

# Project code (src/features).
from voice_features import EXPECTED_SAMPLE_RATE, extract_audio_features  # noqa: E402


class BadAudioError(ValueError):
    """Raised when a voice sample can't be used."""


def features_from_bytes(data: bytes, columns: list[str]) -> np.ndarray:
    """Read any WAV, resample to 32 kHz mono, and run
    src/features/voice_features.extract_audio_features on it unchanged."""
    import librosa
    import soundfile as sf

    try:
        signal, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise BadAudioError("Could not read the audio. Send a WAV file (the UI converts recordings to WAV).") from exc
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if len(signal) < sr * 0.5:
        raise BadAudioError("The voice sample is too short. Record at least 1-2 seconds.")
    if sr != EXPECTED_SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=EXPECTED_SAMPLE_RATE)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        sf.write(path, signal, EXPECTED_SAMPLE_RATE, subtype="PCM_16")
        feats, _, _ = extract_audio_features(path)
    finally:
        path.unlink(missing_ok=True)
    return np.asarray([feats[c] for c in columns], np.float32)
