"""Validate custom Face files and group perceptual duplicates without moving data."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/raw/custom/face"
OUTPUT_DIR = ROOT / "data/processed/custom_face_v1"
PHASH_MAX_DISTANCE = 4
IDENTITIES = ("nour", "sama")
SPLITS = ("train", "validation", "test")


def _phash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(reduced)[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_custom_face_manifest() -> dict:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT_DIR}")
    rows = []
    for split in SPLITS:
        for identity in IDENTITIES:
            folder = DATA_ROOT / split / identity
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Unreadable custom Face image: {path}")
                rows.append({"path": path.relative_to(ROOT).as_posix(), "identity": identity,
                             "original_split": split, "new_split": split,
                             "sha256": _sha(path), "phash64": f"{_phash(image):016x}",
                             "width": int(image.shape[1]), "height": int(image.shape[0])})
    if len(rows) != 40:
        raise ValueError(f"Expected 40 custom Face images; found {len(rows)}")

    exact_groups = defaultdict(list)
    for index, row in enumerate(rows):
        exact_groups[row["sha256"]].append(index)
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    distances = []
    for i, a in enumerate(rows):
        ha = int(a["phash64"], 16)
        for j in range(i + 1, len(rows)):
            b = rows[j]
            if a["identity"] != b["identity"]:
                continue
            distance = (ha ^ int(b["phash64"], 16)).bit_count()
            if distance <= PHASH_MAX_DISTANCE:
                union(i, j)
                distances.append({"path_a": a["path"], "path_b": b["path"],
                                  "identity": a["identity"], "phash_hamming": distance,
                                  "exact_duplicate": a["sha256"] == b["sha256"]})
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    for members in groups.values():
        if len(members) <= 1:
            continue
        member_splits = {rows[i]["original_split"] for i in members}
        # If a group touches test, keep it wholly in test. Otherwise merge
        # train/validation near copies into training so validation stays held out.
        assigned = "test" if "test" in member_splits else ("train" if "train" in member_splits else "validation")
        group_id = hashlib.sha256("\n".join(sorted(rows[i]["sha256"] for i in members)).encode()).hexdigest()[:12]
        for i in members:
            rows[i]["new_split"] = assigned
            rows[i]["near_duplicate_group"] = group_id
    for row in rows:
        row.setdefault("near_duplicate_group", "")

    counts = Counter((r["new_split"], r["identity"]) for r in rows)
    leak_groups = sum(len({rows[i]["original_split"] for i in members}) > 1
                      for members in groups.values() if len(members) > 1)
    summary = {
        "source_root": str(DATA_ROOT.relative_to(ROOT)), "image_count": len(rows),
        "identities": {identity: sum(r["identity"] == identity for r in rows) for identity in IDENTITIES},
        "original_split_counts": {split: sum(r["original_split"] == split for r in rows) for split in SPLITS},
        "new_split_counts": {split: {identity: counts[(split, identity)] for identity in IDENTITIES}
                             for split in SPLITS},
        "exact_duplicate_groups": sum(len(v) > 1 for v in exact_groups.values()),
        "perceptual_near_duplicate_pairs": len(distances),
        "perceptual_hash_method": "64-bit pHash via 32x32 grayscale DCT; group at Hamming distance <=4; visually inspect candidate list.",
        "groups_spanning_original_splits": leak_groups,
        "test_split_used_for_fit_or_threshold_selection": False,
        "caveat": "Near-duplicate detection is a screening heuristic; session metadata is unavailable.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    with (OUTPUT_DIR / "custom_face_split_manifest.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "custom_face_near_duplicate_pairs.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = ["path_a", "path_b", "identity", "phash_hamming", "exact_duplicate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(distances)
    (OUTPUT_DIR / "custom_face_integrity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare_custom_face_manifest(), indent=2))

