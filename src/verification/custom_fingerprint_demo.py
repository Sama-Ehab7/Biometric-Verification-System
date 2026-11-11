"""Custom Team Demo aliases for the standalone fingerprint verifier.

This adapter is intentionally separate from ``fingerprint_inference``. The
mapping CSV is the source of truth, and demo verification accepts only images
listed in the custom fingerprint test split.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .fingerprint_inference import TEMPLATE_ARTIFACT, verify_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT_ROOT = PROJECT_ROOT / "custom-data" / "fingerprint"
MAPPING_CSV = FINGERPRINT_ROOT / "metadata" / "fingerprint_mapping.csv"
TEST_ROOT = FINGERPRINT_ROOT / "test"

_REQUIRED_COLUMNS = {
    "custom_identity",
    "source_dataset",
    "source_database",
    "source_fingerprint_id",
    "copied_file_path",
    "assigned_role",
    "association_type",
}


def _mapping_data() -> tuple[dict[str, str], dict[str, dict[str, Path]]]:
    if not MAPPING_CSV.is_file():
        raise FileNotFoundError(f"Custom fingerprint mapping not found: {MAPPING_CSV}")

    aliases: dict[str, str] = {}
    reverse: dict[str, str] = {}
    test_images: dict[str, dict[str, Path]] = {}
    with MAPPING_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Fingerprint mapping is missing columns: {sorted(missing)}")

        for row in reader:
            alias = (row["custom_identity"] or "").strip().casefold()
            identity = (row["source_fingerprint_id"] or "").strip()
            if not alias or not identity:
                raise ValueError("Fingerprint mapping contains an empty alias or source ID")
            if row["association_type"] != "assumed_demo_mapping":
                raise ValueError(f"Unexpected association type for custom identity {alias!r}")
            if row["source_dataset"] != "FVC2002 Set B":
                raise ValueError(f"Unexpected source dataset for custom identity {alias!r}")

            previous = aliases.setdefault(alias, identity)
            if previous != identity:
                raise ValueError(f"Custom alias {alias!r} maps to multiple fingerprint IDs")
            previous_alias = reverse.setdefault(identity, alias)
            if previous_alias != alias:
                raise ValueError(f"Fingerprint ID {identity!r} maps to multiple custom aliases")

            if row["assigned_role"] == "test":
                copied = (PROJECT_ROOT / row["copied_file_path"]).resolve()
                expected_dir = (TEST_ROOT / alias).resolve()
                try:
                    copied.relative_to(expected_dir)
                except ValueError as exc:
                    raise ValueError(
                        f"Test mapping for {alias!r} points outside its custom test folder"
                    ) from exc
                filename = copied.name
                if filename in test_images.setdefault(alias, {}):
                    raise ValueError(f"Duplicate mapped test image for {alias!r}: {filename}")
                test_images[alias][filename] = copied

    if not aliases:
        raise ValueError("Fingerprint mapping contains no custom identities")

    try:
        with np.load(TEMPLATE_ARTIFACT, allow_pickle=False) as artifact:
            template_ids = set(artifact["identities"].astype(str).tolist())
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Unable to read fingerprint templates: {TEMPLATE_ARTIFACT}") from exc

    missing_ids = set(aliases.values()) - template_ids
    if missing_ids:
        raise ValueError(f"Mapped fingerprint IDs are absent from saved templates: {sorted(missing_ids)}")
    return aliases, test_images


def list_custom_fingerprint_aliases() -> list[str]:
    """Return aliases defined by the mapping CSV."""
    aliases, _ = _mapping_data()
    return sorted(aliases)


def resolve_custom_fingerprint_identity(custom_alias: str) -> str:
    """Resolve a selected custom-demo alias to its saved database ID."""
    alias = (custom_alias or "").strip().casefold()
    aliases, _ = _mapping_data()
    try:
        return aliases[alias]
    except KeyError as exc:
        raise ValueError(f"Unknown custom fingerprint alias: {custom_alias!r}") from exc


def list_custom_fingerprint_test_images() -> dict[str, list[str]]:
    """Return only test-split image names recorded in the mapping CSV."""
    aliases, test_images = _mapping_data()
    result: dict[str, list[str]] = {}
    for alias in sorted(aliases):
        images = test_images.get(alias, {})
        if not images:
            raise ValueError(f"No custom test images are mapped for {alias!r}")
        missing = [name for name, path in images.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Mapped test images are missing for {alias!r}: {missing}")
        result[alias] = sorted(images)
    return result


def resolve_custom_fingerprint_test_image(probe_alias: str, image_name: str) -> Path:
    """Resolve a mapped test image by its demo folder and basename."""
    alias = (probe_alias or "").strip().casefold()
    _, test_images = _mapping_data()
    if alias not in test_images:
        raise ValueError(f"Unknown custom fingerprint probe identity: {probe_alias!r}")
    if Path(image_name).name != image_name:
        raise ValueError("Fingerprint test image must be selected by file name")
    try:
        path = test_images[alias][image_name]
    except KeyError as exc:
        raise ValueError(f"Image is not mapped in the custom test split: {image_name!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Mapped custom test image not found: {path}")
    return path


def verify_custom_fingerprint(image_path: str | Path, custom_alias: str) -> dict:
    """Verify a mapped custom test image using its selected custom alias.

    This demo adapter leaves the original database-qualified inference API
    unchanged and returns both the custom alias and resolved source identity.
    """
    identity = resolve_custom_fingerprint_identity(custom_alias)
    path = Path(image_path).resolve()
    _, test_images = _mapping_data()
    allowed = {mapped.resolve() for rows in test_images.values() for mapped in rows.values()}
    if path not in allowed:
        raise ValueError("Custom demo verification accepts only mapped test-split images")
    result = verify_fingerprint(path, identity)
    return {"custom_identity": custom_alias.strip().casefold(), **result}


def verify_custom_fingerprint_test_image(
    claimed_alias: str, probe_alias: str, image_name: str
) -> dict:
    """Verify one test-split probe against the selected custom alias."""
    path = resolve_custom_fingerprint_test_image(probe_alias, image_name)
    return verify_custom_fingerprint(path, claimed_alias)


def verify_custom_fingerprint_upload(image_path: str | Path, custom_alias: str) -> dict:
    """Verify an uploaded probe using the saved SVM and CSV-backed alias.

    This custom-demo adapter does not train, calibrate, or alter the original
    database-qualified fingerprint inference function or its artifacts.
    """
    identity = resolve_custom_fingerprint_identity(custom_alias)
    result = verify_fingerprint(Path(image_path), identity)
    return {"custom_identity": custom_alias.strip().casefold(), **result}
