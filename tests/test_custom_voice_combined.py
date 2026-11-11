"""Inference/API tests for the isolated custom Voice and all-of-three demo."""
import io
import unittest
from pathlib import Path

import joblib
import soundfile as sf
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from backend.custom_voice_demo import identify_route, status as voice_status, verify_route
from backend.custom_combined_demo import (all_modalities_accept, status as combined_status,
                                          verify_route as combined_route)
from backend.custom_commands_demo import execute_route, status as command_status
from src.verification.custom_voice import (FEATURE_DIMENSION, MODEL_DIR, RESULT_DIR,
    extract_voice_features, verify_custom_voice)

ROOT = Path(__file__).resolve().parents[1]


def upload(path: Path, content_type: str):
    return UploadFile(filename=path.name, file=io.BytesIO(path.read_bytes()),
                      headers=Headers({"content-type": content_type}))


def manifest_test_voice():
    import csv
    manifest = ROOT / "data/processed/custom_audio_v3/custom_voice_grouped_split_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        row = next(r for r in csv.DictReader(stream)
                   if r["new_split"] == "test" and r["identity"] == "nour"
                   and r["use_as_split_sample"].lower() == "true")
    return ROOT / "data/processed/custom_audio_v3/converted" / row["relative_output"]


class CustomVoiceTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = manifest_test_voice()
        cls.audio = cls.probe.read_bytes()

    def test_feature_width_persisted_threshold_and_inference(self):
        self.assertEqual(extract_voice_features(self.probe).shape, (FEATURE_DIMENSION,))
        bundle = joblib.load(MODEL_DIR / "verifier.joblib")
        self.assertEqual(bundle["threshold_split"], "validation")
        result = verify_custom_voice(self.audio, "nour")
        self.assertEqual(result["threshold"], bundle["threshold"])
        self.assertEqual(result["decision"], "ACCEPT" if result["score"] >= result["threshold"] else "REJECT")
        self.assertTrue((MODEL_DIR / "identifier.joblib").is_file())
        self.assertTrue((RESULT_DIR / "evaluation.json").is_file())

    async def test_voice_routes_and_invalid_claim_or_wav(self):
        self.assertEqual(voice_status()["identities"], ["nour", "sama"])
        result = await verify_route("nour", upload(self.probe, "audio/wav"))
        self.assertIn(result["decision"], {"ACCEPT", "REJECT"})
        identified = await identify_route(upload(self.probe, "audio/wav"))
        self.assertIn(identified["identity"], {"nour", "sama", "unknown"})
        with self.assertRaises(HTTPException) as error:
            await verify_route("unknown", upload(self.probe, "audio/wav"))
        self.assertEqual(error.exception.status_code, 400)
        with self.assertRaises(HTTPException) as error:
            await verify_route("nour", upload(self.probe, "audio/mpeg"))
        self.assertEqual(error.exception.status_code, 415)

    async def test_combined_policy_reports_each_modality_and_fails_closed(self):
        face = ROOT / "data/raw/custom/face/test/nour/face_01.jpg"
        voice = upload(self.probe, "audio/wav")
        result = await combined_route("nour", "svm", upload(face, "image/jpeg"), voice,
                                      "nour", "101_8.tif")
        self.assertEqual(set(result["components"]), {"face", "voice", "fingerprint"})
        expected = all(part.get("accepted", part.get("decision") == "ACCEPT")
                       for part in result["components"].values())
        self.assertEqual(result["accepted"], expected)
        self.assertEqual(result["decision"], "ACCEPT" if expected else "REJECT")
        self.assertFalse(combined_status()["scientifically_validated_fusion"])

    def test_all_of_three_policy_rejects_missing_or_failed_modality(self):
        passing = {name: {"decision": "ACCEPT"} for name in ("face", "voice", "fingerprint")}
        self.assertTrue(all_modalities_accept(passing))
        for failed in passing:
            case = {name: dict(value) for name, value in passing.items()}
            case[failed] = {"decision": "REJECT"}
            with self.subTest(failed=failed):
                self.assertFalse(all_modalities_accept(case))
        self.assertFalse(all_modalities_accept({"face": {"decision": "ACCEPT"}}))

    async def test_command_execution_fails_closed_for_all_requesters(self):
        self.assertFalse(command_status()["execution_enabled"])
        with self.assertRaises(HTTPException) as error:
            await execute_route(upload(self.probe, "audio/wav"))
        self.assertEqual(error.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
