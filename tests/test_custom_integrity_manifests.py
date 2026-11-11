"""Read-only checks for the grouped custom-data manifests and command metrics."""
import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CustomIntegrityManifestTests(unittest.TestCase):
    def test_voice_exact_duplicate_groups_stay_in_one_split(self):
        path = ROOT / "data/processed/custom_audio_v3/custom_voice_grouped_split_manifest.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        groups = defaultdict(set)
        counts = defaultdict(lambda: defaultdict(int))
        for row in rows:
            if row["duplicate_group"]:
                groups[row["duplicate_group"]].add(row["new_split"])
            if row["use_as_split_sample"].lower() == "true":
                counts[row["new_split"]][row["identity"]] += 1
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(dict(counts["train"]), {"nour": 3, "sama": 6})
        self.assertEqual(dict(counts["validation"]), {"nour": 1, "sama": 2})
        self.assertEqual(dict(counts["test"]), {"nour": 1, "sama": 2})

    def test_face_near_duplicate_groups_do_not_cross_new_splits(self):
        path = ROOT / "data/processed/custom_face_v1/custom_face_split_manifest.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        groups = defaultdict(set)
        counts = defaultdict(lambda: defaultdict(int))
        for row in rows:
            if row["near_duplicate_group"]:
                groups[row["near_duplicate_group"]].add(row["new_split"])
            counts[row["new_split"]][row["identity"]] += 1
        self.assertEqual(len(rows), 40)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(dict(counts["train"]), {"nour": 12, "sama": 13})
        self.assertEqual(dict(counts["validation"]), {"nour": 4, "sama": 3})
        self.assertEqual(dict(counts["test"]), {"nour": 4, "sama": 4})

    def test_command_confusion_matrix_labels_match_saved_matrix(self):
        report = json.loads((ROOT / "results/custom/commands_v3/evaluation.json").read_text(encoding="utf-8"))
        labels = json.loads((ROOT / "results/custom/commands_v3/confusion_matrix_labels.json").read_text(encoding="utf-8"))
        matrix = report["test"]["confusion_matrix"]
        self.assertEqual(len(matrix), len(labels["axis_order"]))
        self.assertTrue(all(len(row) == len(labels["axis_order"]) for row in matrix))
        self.assertEqual(sum(map(sum, matrix)), 10)


if __name__ == "__main__":
    unittest.main()
