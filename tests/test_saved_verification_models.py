"""Read-only smoke checks for saved Session 1 verification estimators."""

import csv
import unittest
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SavedVerificationModelTests(unittest.TestCase):
    def test_saved_security_models_accept_expected_pair_dimensions(self):
        cases = [
            ("face SVM", "saved_models/face/svm_face_security.joblib", "face", 3548),
            ("face RF", "saved_models/face/random_forest_face_security.joblib", "face", 3548),
            ("voice SVM", "saved_models/voice/svm_voice_security.joblib", "voice", 30),
            ("voice RF", "saved_models/voice/random_forest_voice_security.joblib", "voice", 30),
            ("fusion SVM", "saved_models/fusion/svm_early_fusion_security.joblib", "multimodal", 3578),
            ("fusion RF", "saved_models/fusion/random_forest_early_fusion_security.joblib", "multimodal", 3578),
        ]
        split_cache = {}
        for model_name, relative_model_path, prefix, expected_width in cases:
            with self.subTest(model=model_name):
                bundle = joblib.load(ROOT / relative_model_path)
                columns = bundle["feature_columns"]
                self.assertEqual(len(columns), expected_width)
                self.assertEqual(bundle["model"].n_features_in_, expected_width)

                if prefix not in split_cache:
                    train_path = ROOT / f"data/processed/splits/{prefix}_train.csv"
                    test_path = ROOT / f"data/processed/splits/{prefix}_test.csv"
                    with train_path.open("r", newline="", encoding="utf-8-sig") as handle:
                        train_rows = list(csv.DictReader(handle))
                    with test_path.open("r", newline="", encoding="utf-8-sig") as handle:
                        probe_row = next(csv.DictReader(handle))
                    split_cache[prefix] = train_rows, probe_row
                train_rows, probe_row = split_cache[prefix]

                speaker = probe_row["speaker"]
                enrollment = [r for r in train_rows if r["speaker"] == speaker]
                self.assertEqual(len(enrollment), 6)
                reference = np.mean(
                    [[float(row[c]) for c in columns] for row in enrollment], axis=0
                )
                probe = np.asarray([float(probe_row[c]) for c in columns])
                pair = np.abs(probe - reference).reshape(1, -1)
                score = float(bundle["model"].predict_proba(pair)[0, 1])
                self.assertTrue(np.isfinite(score))
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                self.assertIn(0.01, bundle["thresholds"])


if __name__ == "__main__":
    unittest.main()
