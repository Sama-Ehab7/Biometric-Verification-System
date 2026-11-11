"""Focused checks for the isolated custom Face models and inference API logic."""
import unittest
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.verification.custom_face import (
    MODEL_DIR, RESULT_DIR, extract_custom_face_features, identify_custom_face,
    verify_custom_face,
)

ROOT = Path(__file__).resolve().parents[1]


class CustomFaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_image = next((ROOT / "data/raw/custom/face/test/nour").glob("*"))
        cls.image_bytes = cls.test_image.read_bytes()

    def test_feature_width_and_image_validation(self):
        image = cv2.imread(str(self.test_image))
        self.assertEqual(extract_custom_face_features(image).shape, (3548,))
        with self.assertRaisesRegex(ValueError, "Unable to read"):
            verify_custom_face(b"not an image", "nour")

    def test_unknown_claim_and_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown custom Face identity"):
            verify_custom_face(self.image_bytes, "unknown")
        with self.assertRaisesRegex(ValueError, "Model must be"):
            verify_custom_face(self.image_bytes, "nour", "other")

    def test_svm_and_random_forest_use_saved_validation_thresholds(self):
        for name in ("svm", "random_forest"):
            with self.subTest(model=name):
                bundle = joblib.load(MODEL_DIR / f"{name}.joblib")
                result = verify_custom_face(self.image_bytes, "nour", name)
                self.assertEqual(result["threshold"], bundle["threshold"])
                self.assertEqual(result["model"], name)
                self.assertIn(result["decision"], {"ACCEPT", "REJECT"})
                self.assertEqual(result["accepted"], result["score"] >= result["threshold"])
                self.assertEqual(bundle["threshold_split"], "validation")

    def test_identification_returns_known_class_or_unknown(self):
        result = identify_custom_face(self.image_bytes, "svm")
        self.assertIn(result["identity"], {"nour", "sama", "unknown"})
        self.assertEqual(result["known"], result["identity"] != "unknown")

    def test_custom_outputs_are_separate_from_original_artifacts(self):
        self.assertTrue((MODEL_DIR / "svm.joblib").is_file())
        self.assertTrue((MODEL_DIR / "random_forest.joblib").is_file())
        self.assertTrue((RESULT_DIR / "evaluation.json").is_file())
        self.assertFalse((ROOT / "saved_models/face/custom_face.joblib").exists())
        self.assertFalse((ROOT / "results/face/custom_face_evaluation.csv").exists())


if __name__ == "__main__":
    unittest.main()
