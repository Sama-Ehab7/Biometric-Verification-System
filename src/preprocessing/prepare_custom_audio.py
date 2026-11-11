"""Validate and convert custom WAV/MP4 recordings into isolated 32 kHz mono WAVs.

On Windows, MP4/AAC decoding uses the built-in Media Foundation APIs; no package
installation or source-file mutation is performed. Outputs are versioned and
the script refuses to overwrite an existing derived-data directory.
"""
from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
import struct
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "raw" / "custom" / "voice"
OUTPUT_ROOT = ROOT / "data" / "processed" / "custom_audio_v3"
OUTPUT_RATE = 32_000
OUTPUT_CHANNELS = 1
VOICE_SPLITS = ("train", "validation", "test")
IDENTITIES = ("nour", "sama")
COMMANDS = ("open", "close", "start", "stop", "help")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guid(value: str):
    parsed = uuid.UUID(value)
    return _GUID(parsed.time_low, parsed.time_mid, parsed.time_hi_version,
                 (ctypes.c_ubyte * 8)(*parsed.bytes[8:]))


def _guid_text(value) -> str:
    tail = bytes(value.Data4).hex()
    return f"{value.Data1:08x}-{value.Data2:04x}-{value.Data3:04x}-{tail[:4]}-{tail[4:]}"


def _com_method(pointer, slot: int, result_type, argument_types):
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result_type, ctypes.c_void_p, *argument_types)(table[slot])


def _release(pointer) -> None:
    if pointer is not None and pointer.value:
        _com_method(pointer, 2, ctypes.c_ulong, [])(pointer)
        pointer.value = None


class MediaFoundationAACDecoder:
    """Minimal Media Foundation Source Reader wrapper for AAC audio-to-PCM."""

    FIRST_AUDIO_STREAM = 0xFFFFFFFD
    END_OF_STREAM = 0x00000002
    MF_VERSION = 0x00020070
    MF_MT_MAJOR_TYPE = "48eba18e-f8c9-4687-bf11-0a74c9f96a8f"
    MF_MT_SUBTYPE = "f7e34c9a-42e8-4714-b74b-cb29d72c35e5"
    MF_MT_AUDIO_NUM_CHANNELS = "37e48bf5-645e-4c5b-89de-ada9e29b696a"
    MF_MT_AUDIO_SAMPLES_PER_SECOND = "5faeeae7-0290-4c31-9e8a-c534f68d9dba"
    MF_MT_AUDIO_BITS_PER_SAMPLE = "f2deb57f-40fa-4764-aa33-ed4f2d1ff669"
    MF_MT_AUDIO_BLOCK_ALIGNMENT = "322de230-9eeb-43bd-ab7a-ff412251541d"
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND = "1aab75c8-cfef-451c-ab95-ac034b8e1731"
    MF_MEDIA_TYPE_AUDIO = "73647561-0000-0010-8000-00aa00389b71"
    MF_AUDIO_FORMAT_PCM = "00000001-0000-0010-8000-00aa00389b71"
    MF_AUDIO_FORMAT_AAC = "00001610-0000-0010-8000-00aa00389b71"

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Media Foundation MP4 conversion requires Windows")
        self.mfplat = ctypes.WinDLL("mfplat.dll")
        self.mfreadwrite = ctypes.WinDLL("mfreadwrite.dll")
        self.mfplat.MFStartup.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.mfplat.MFStartup.restype = ctypes.c_long
        self.mfplat.MFShutdown.restype = ctypes.c_long
        result = self.mfplat.MFStartup(self.MF_VERSION, 0)
        if result < 0:
            raise RuntimeError(f"MFStartup failed: 0x{result & 0xffffffff:08x}")

    @staticmethod
    def _check(result: int, stage: str) -> None:
        if result < 0:
            raise RuntimeError(f"{stage} failed: HRESULT 0x{result & 0xffffffff:08x}")

    @staticmethod
    def _attribute_u32(media_type, key: str) -> int:
        value = ctypes.c_uint32()
        fn = _com_method(media_type, 7, ctypes.c_long,
                         [ctypes.POINTER(type(_guid(key))), ctypes.POINTER(ctypes.c_uint32)])
        MediaFoundationAACDecoder._check(fn(media_type, ctypes.byref(_guid(key)), ctypes.byref(value)),
                                          f"GetUINT32({key})")
        return int(value.value)

    def decode(self, source: Path) -> tuple[np.ndarray, dict[str, Any]]:
        """Decode one AAC MP4 to normalized float32, mono at OUTPUT_RATE."""
        from ctypes import wintypes

        reader = ctypes.c_void_p()
        native_type = ctypes.c_void_p()
        output_type = ctypes.c_void_p()
        self.mfreadwrite.MFCreateSourceReaderFromURL.argtypes = [
            wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.mfreadwrite.MFCreateSourceReaderFromURL.restype = ctypes.c_long
        self._check(self.mfreadwrite.MFCreateSourceReaderFromURL(
            str(source.resolve()), None, ctypes.byref(reader)), "Open MP4 source")
        try:
            get_native = _com_method(reader, 5, ctypes.c_long,
                                     [wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)])
            self._check(get_native(reader, self.FIRST_AUDIO_STREAM, 0, ctypes.byref(native_type)),
                        "Find native audio stream")
            subtype = _GUID()
            get_guid = _com_method(native_type, 10, ctypes.c_long,
                                   [ctypes.POINTER(type(subtype)), ctypes.POINTER(type(subtype))])
            self._check(get_guid(native_type, ctypes.byref(_guid(self.MF_MT_SUBTYPE)),
                                 ctypes.byref(subtype)), "Read audio codec")
            native_codec = _guid_text(subtype)
            native_rate = self._attribute_u32(native_type, self.MF_MT_AUDIO_SAMPLES_PER_SECOND)
            native_channels = self._attribute_u32(native_type, self.MF_MT_AUDIO_NUM_CHANNELS)

            self.mfplat.MFCreateMediaType.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self.mfplat.MFCreateMediaType.restype = ctypes.c_long
            self._check(self.mfplat.MFCreateMediaType(ctypes.byref(output_type)), "Create PCM media type")
            set_guid = _com_method(output_type, 24, ctypes.c_long,
                                   [ctypes.POINTER(_GUID), ctypes.POINTER(_GUID)])
            set_u32 = _com_method(output_type, 21, ctypes.c_long,
                                  [ctypes.POINTER(_GUID), wintypes.UINT])
            self._check(set_guid(output_type, ctypes.byref(_guid(self.MF_MT_MAJOR_TYPE)),
                                 ctypes.byref(_guid(self.MF_MEDIA_TYPE_AUDIO))), "Set audio major type")
            self._check(set_guid(output_type, ctypes.byref(_guid(self.MF_MT_SUBTYPE)),
                                 ctypes.byref(_guid(self.MF_AUDIO_FORMAT_PCM))), "Set PCM subtype")
            for key, value in (
                (self.MF_MT_AUDIO_NUM_CHANNELS, OUTPUT_CHANNELS),
                (self.MF_MT_AUDIO_SAMPLES_PER_SECOND, OUTPUT_RATE),
                (self.MF_MT_AUDIO_BITS_PER_SAMPLE, 16),
                (self.MF_MT_AUDIO_BLOCK_ALIGNMENT, 2),
                (self.MF_MT_AUDIO_AVG_BYTES_PER_SECOND, OUTPUT_RATE * 2),
            ):
                self._check(set_u32(output_type, ctypes.byref(_guid(key)), value), f"Set PCM {key}")

            set_current = _com_method(reader, 7, ctypes.c_long,
                                      [wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p])
            self._check(set_current(reader, self.FIRST_AUDIO_STREAM, None, output_type),
                        "Negotiate PCM output")
            read_sample = _com_method(reader, 9, ctypes.c_long,
                [wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                 ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(ctypes.c_longlong),
                 ctypes.POINTER(ctypes.c_void_p)])
            pieces: list[bytes] = []
            for _ in range(100_000):
                actual = wintypes.DWORD()
                flags = wintypes.DWORD()
                timestamp = ctypes.c_longlong()
                sample = ctypes.c_void_p()
                self._check(read_sample(reader, self.FIRST_AUDIO_STREAM, 0,
                                        ctypes.byref(actual), ctypes.byref(flags),
                                        ctypes.byref(timestamp), ctypes.byref(sample)), "Read AAC sample")
                if sample.value:
                    try:
                        buffer = ctypes.c_void_p()
                        self._check(_com_method(sample, 41, ctypes.c_long,
                                                [ctypes.POINTER(ctypes.c_void_p)])(
                            sample, ctypes.byref(buffer)), "Get decoded PCM buffer")
                        try:
                            data_pointer = ctypes.POINTER(ctypes.c_ubyte)()
                            maximum = wintypes.DWORD()
                            current = wintypes.DWORD()
                            self._check(_com_method(buffer, 3, ctypes.c_long,
                                [ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                                 ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD)])(
                                    buffer, ctypes.byref(data_pointer), ctypes.byref(maximum),
                                    ctypes.byref(current)), "Lock decoded PCM")
                            try:
                                if current.value:
                                    pieces.append(ctypes.string_at(data_pointer, current.value))
                            finally:
                                _com_method(buffer, 4, ctypes.c_long, [])(buffer)
                        finally:
                            _release(buffer)
                    finally:
                        _release(sample)
                if flags.value & self.END_OF_STREAM:
                    break
            else:
                raise RuntimeError("MP4 decoder exceeded the sample limit")
            pcm = b"".join(pieces)
            if not pcm or len(pcm) % 2:
                raise ValueError("MP4 produced empty or malformed 16-bit PCM audio")
            signal = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            if not np.isfinite(signal).all() or not np.any(signal):
                raise ValueError("Decoded MP4 contains silent or non-finite audio")
            info = {"container": "ISO Base Media MP4", "codec_guid": native_codec,
                    "codec": "AAC" if native_codec == self.MF_AUDIO_FORMAT_AAC else native_codec,
                    "source_sample_rate": native_rate, "source_channels": native_channels,
                    "output_sample_rate": OUTPUT_RATE, "output_channels": OUTPUT_CHANNELS,
                    "output_subtype": "signed 16-bit PCM", "duration_seconds": len(signal) / OUTPUT_RATE}
            return signal, info
        finally:
            _release(output_type)
            _release(native_type)
            _release(reader)

    def close(self) -> None:
        self.mfplat.MFShutdown()


def _record_metadata(path: Path) -> dict[str, str]:
    relative = path.relative_to(SOURCE_ROOT)
    if relative.parts[0] == "commands":
        _, split, speaker, command, *_ = relative.parts
        task = "command"
        identity = speaker
        original_split = split
    else:
        split, speaker, *_ = relative.parts
        task = "voice"
        identity = speaker
        command = ""
        original_split = split
    return {"source_path": path.relative_to(ROOT).as_posix(), "task": task,
            "identity": identity, "command": command,
            "original_split": original_split, "extension": path.suffix.lower()}


def _read_wav(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    info = sf.info(str(path))
    if info.duration <= 0 or info.channels < 1 or info.samplerate <= 0:
        raise ValueError(f"Invalid WAV metadata: {path}")
    signal, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Unreadable or non-finite WAV: {path}")
    mono = signal.mean(axis=1)
    if rate != OUTPUT_RATE:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=OUTPUT_RATE)
    if len(mono) == 0 or not np.isfinite(mono).all() or not np.any(mono):
        raise ValueError(f"WAV contains no usable signal: {path}")
    return mono.astype(np.float32), {
        "container": "WAV", "codec": "PCM (decoded by libsndfile)",
        "source_sample_rate": int(rate), "source_channels": int(info.channels),
        "output_sample_rate": OUTPUT_RATE, "output_channels": OUTPUT_CHANNELS,
        "output_subtype": "signed 16-bit PCM", "duration_seconds": len(mono) / OUTPUT_RATE,
    }


def _voice_split_assignments(unique_voice_records: list[dict[str, Any]]) -> dict[str, str]:
    """Assign unique recordings, not duplicate file copies, to new stratified splits."""
    assignments: dict[str, str] = {}
    for identity, seed in (("nour", 17), ("sama", 29)):
        hashes = sorted(r["source_sha256"] for r in unique_voice_records if r["identity"] == identity)
        rng = np.random.default_rng(seed)
        ordered = [hashes[i] for i in rng.permutation(len(hashes))]
        train_count = int(round(len(ordered) * 0.6))
        test_count = max(1, int(round(len(ordered) * 0.2)))
        val_count = len(ordered) - train_count - test_count
        if min(train_count, val_count, test_count) < 1:
            raise ValueError(f"Not enough unique recordings for three-way split: {identity}={len(ordered)}")
        for h in ordered[:train_count]:
            assignments[h] = "train"
        for h in ordered[train_count:train_count + val_count]:
            assignments[h] = "validation"
        for h in ordered[train_count + val_count:]:
            assignments[h] = "test"
    return assignments


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty report: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_custom_audio() -> dict[str, Any]:
    """Decode all available clips, record integrity/splits, and write derived WAVs."""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite derived audio: {OUTPUT_ROOT}")
    sources = sorted(p for p in SOURCE_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".mp4"})
    if len(sources) != 60:
        raise ValueError(f"Expected to inspect all 60 custom voice/command files, found {len(sources)}")
    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in sources:
        hashes[sha256_file(path)].append(path)
    duplicates = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    hash_identities: dict[str, set[str]] = defaultdict(set)
    for path in sources:
        hash_identities[sha256_file(path)].add(_record_metadata(path)["identity"])
    conflicting_identity_groups = {h: ids for h, ids in hash_identities.items() if len(ids) > 1}
    if conflicting_identity_groups:
        raise ValueError(f"Byte-identical audio maps to multiple identities: {conflicting_identity_groups}")
    decoder = MediaFoundationAACDecoder()
    prepared: list[dict[str, Any]] = []
    failures = []
    try:
        for source in sources:
            source_hash = sha256_file(source)
            try:
                if source.suffix.lower() == ".mp4":
                    signal, info = decoder.decode(source)
                else:
                    signal, info = _read_wav(source)
                if len(signal) < int(0.2 * OUTPUT_RATE):
                    raise ValueError("Audio shorter than 0.2 seconds")
                relative = source.relative_to(SOURCE_ROOT)
                if relative.parts[0] == "commands":
                    rel_output = Path("commands") / Path(*relative.parts[1:]).with_suffix(".wav")
                else:
                    rel_output = Path("voice") / relative.with_suffix(".wav")
                dest = OUTPUT_ROOT / "converted" / rel_output
                metadata = _record_metadata(source)
                prepared.append({**metadata, **info, "source_sha256": source_hash,
                                 "relative_output": rel_output.as_posix(), "destination": dest,
                                 "signal": signal, "pcm_sha256": hashlib.sha256(
                                     np.asarray(signal, dtype="<f4").tobytes()).hexdigest()})
            except Exception as exc:
                failures.append({"source_path": source.relative_to(ROOT).as_posix(), "error": repr(exc)})
    finally:
        decoder.close()
    if failures:
        summary = "; ".join(f"{r['source_path']}: {r['error']}" for r in failures)
        raise RuntimeError(f"Audio preflight failed; no derived output was written: {summary}")
    if len(prepared) != len(sources):
        raise RuntimeError("Audio preflight did not produce exactly one output per source")
    destinations = [r["destination"] for r in prepared]
    if len(set(destinations)) != len(destinations):
        raise RuntimeError("Two source files map to the same derived output path")

    unique_voice_records = []
    for record in prepared:
        if record["task"] == "voice" and record["source_sha256"] not in {
            r["source_sha256"] for r in unique_voice_records
        }:
            unique_voice_records.append(record)
    split_assignments = _voice_split_assignments(unique_voice_records)
    canonical_source = {h: min(p.as_posix() for p in ps) for h, ps in hashes.items()}
    for record in prepared:
        if record["task"] == "voice":
            record["new_split"] = split_assignments[record["source_sha256"]]
            record["duplicate_group"] = record["source_sha256"] if len(hashes[record["source_sha256"]]) > 1 else ""
            record["use_as_split_sample"] = record["source_path"] == Path(
                canonical_source[record["source_sha256"]]).relative_to(ROOT).as_posix()
        else:
            record["new_split"] = record["original_split"]
            record["duplicate_group"] = record["source_sha256"] if len(hashes[record["source_sha256"]]) > 1 else ""
            record["use_as_split_sample"] = record["source_path"] == Path(
                canonical_source[record["source_sha256"]]).relative_to(ROOT).as_posix()

    # All source media have decoded and all paths have passed collision checks.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    for record in prepared:
        dest: Path = record["destination"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), record["signal"], OUTPUT_RATE, subtype="PCM_16")

    manifest_fields = ["source_path", "task", "identity", "command", "original_split", "new_split",
                       "extension", "container", "codec", "codec_guid", "source_sample_rate",
                       "source_channels", "output_sample_rate", "output_channels", "output_subtype",
                       "duration_seconds", "source_sha256", "pcm_sha256", "duplicate_group",
                       "use_as_split_sample", "relative_output"]
    manifest = [{k: r.get(k, "") for k in manifest_fields} for r in prepared]
    _write_csv(OUTPUT_ROOT / "audio_conversion_integrity_manifest.csv", manifest)
    voice_manifest = [{k: r.get(k, "") for k in manifest_fields if k not in {"task", "command"}}
                      for r in prepared if r["task"] == "voice"]
    _write_csv(OUTPUT_ROOT / "custom_voice_grouped_split_manifest.csv", voice_manifest)
    command_manifest = [{k: r.get(k, "") for k in manifest_fields if k not in {"task", "new_split", "duplicate_group"}}
                        for r in prepared if r["task"] == "command"]
    _write_csv(OUTPUT_ROOT / "custom_command_conversion_manifest.csv", command_manifest)
    duplicate_rows = []
    for h, paths in duplicates.items():
        identity_rows = [_record_metadata(p) for p in paths]
        for i, path_a in enumerate(paths):
            for path_b in paths[i + 1:]:
                a, b = _record_metadata(path_a), _record_metadata(path_b)
                duplicate_rows.append({"duplicate_group": h, "sha256": h,
                    "identity_a": a["identity"], "original_split_a": a["original_split"],
                    "original_path_a": a["source_path"], "identity_b": b["identity"],
                    "original_split_b": b["original_split"], "original_path_b": b["source_path"],
                    "same_identity": a["identity"] == b["identity"],
                    "new_split": split_assignments.get(h, a["original_split"])})
    if duplicate_rows:
        _write_csv(OUTPUT_ROOT / "exact_duplicate_relationships.csv", duplicate_rows)

    split_counts = defaultdict(lambda: defaultdict(int))
    for record in unique_voice_records:
        split_counts[record["identity"]][split_assignments[record["source_sha256"]]] += 1
    summary = {
        "source_root": str(SOURCE_ROOT.relative_to(ROOT)),
        "derived_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "source_file_count": len(sources), "successful_decodes": len(prepared),
        "mp4_count": sum(p.suffix.lower() == ".mp4" for p in sources),
        "wav_count": sum(p.suffix.lower() == ".wav" for p in sources),
        "mp4_codec_counts": dict(__import__("collections").Counter(
            r["codec"] for r in prepared if r["extension"] == ".mp4")),
        "source_native_formats": sorted({(r["source_sample_rate"], r["source_channels"], r["codec"])
                                          for r in prepared if r["extension"] == ".mp4"}),
        "output_format": {"sample_rate": OUTPUT_RATE, "channels": OUTPUT_CHANNELS,
                          "subtype": "PCM_16 WAV"},
        "exact_duplicate_groups": len(duplicates),
        "duplicate_pairs": sum(len(ps) * (len(ps) - 1) // 2 for ps in duplicates.values()),
        "leakage_free_unique_voice_split_counts": {k: dict(v) for k, v in split_counts.items()},
        "command_original_split_counts": {
            split: {command: sum(1 for r in prepared if r["task"] == "command"
                                 and r["original_split"] == split and r["command"] == command)
                    for command in COMMANDS}
            for split in VOICE_SPLITS},
        "conversion_method": "Windows Media Foundation Source Reader AAC decode; downmix/convert to 32 kHz mono signed PCM16 WAV. Existing WAVs were read with libsndfile, downmixed and resampled with librosa to the same format.",
        "original_files_modified": False,
    }
    (OUTPUT_ROOT / "audio_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare_custom_audio(), indent=2))

