"""Focused smoke tests for the standalone Fingerprint inference module."""

import csv
import unittest
from pathlib import Path

import numpy as np

from src.verification.fingerprint_inference import (
    MODEL_PATH,
    TEMPLATE_ARTIFACT,
    create_enrollment_artifact,
    extract_inference_features,
    verify_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]


class FingerprintInferenceTests(unittest.TestCase):
    def test_model_template_artifact_and_decision(self):
        self.assertTrue(MODEL_PATH.is_file())
        self.assertTrue(TEMPLATE_ARTIFACT.is_file())
        result = verify_fingerprint(
            ROOT / "data/raw/fingerprint/DB1_B/101_7.tif",
            "DB1_B_101",
        )
        self.assertEqual(result["threshold"], 0.4540606024646377)
        self.assertIn(result["decision"], {"ACCEPT", "REJECT"})
        self.assertTrue(0.0 <= result["score"] <= 1.0)

    def test_raw_image_features_match_saved_training_extraction(self):
        image_path = ROOT / "data/raw/fingerprint/DB1_B/101_7.tif"
        extracted = extract_inference_features(image_path)
        feature_csv = ROOT / "data/processed/fingerprint/fingerprint_features.csv"
        with feature_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            row = next(
                record for record in csv.DictReader(handle)
                if record["identity"] == "DB1_B_101"
                and record["image_name"] == "101_7.png"
            )
        saved = np.asarray(
            [float(row[f"fingerprint_hog_{i:04d}"]) for i in range(8100)]
            + [float(row[f"fingerprint_lbp_{i:02d}"]) for i in range(10)]
        )
        self.assertEqual(extracted.shape, (8110,))
        np.testing.assert_allclose(extracted, saved, rtol=0, atol=1e-6)

    def test_invalid_path_and_unknown_claim_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            extract_inference_features(ROOT / "missing-fingerprint.tif")
        with self.assertRaisesRegex(ValueError, "Unable to read fingerprint image"):
            extract_inference_features(ROOT / "README.md")
        with self.assertRaisesRegex(ValueError, "Unknown enrolled claim"):
            verify_fingerprint(
                ROOT / "data/raw/fingerprint/DB1_B/101_7.tif",
                "NOT_A_REAL_ID",
            )

    def test_template_export_refuses_to_overwrite(self):
        with self.assertRaises(FileExistsError):
            create_enrollment_artifact()


if __name__ == "__main__":
    unittest.main()
