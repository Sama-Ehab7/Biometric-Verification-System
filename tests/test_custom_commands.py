"""Tests for the limited custom command classifier and fail-closed action route."""
import io
import unittest
from pathlib import Path

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from backend.custom_commands_demo import classify_route, execute_route, status
from src.verification.custom_commands import COMMANDS, classify_custom_command, extract_command_features

ROOT = Path(__file__).resolve().parents[1]


class CustomCommandTests(unittest.IsolatedAsyncioTestCase):
    def test_feature_dimension_and_saved_classifier(self):
        probe = ROOT / "data/raw/custom/voice/commands/test/nour/open/open_004.wav"
        self.assertEqual(extract_command_features(probe).shape, (30,))
        result = classify_custom_command(probe.read_bytes())
        self.assertIn(result["command"], COMMANDS)
        self.assertIn("classification only", result["scope"])

    async def test_route_validates_audio_and_protected_execution_fails_closed(self):
        probe = ROOT / "data/raw/custom/voice/commands/test/nour/open/open_004.wav"
        upload = UploadFile(filename=probe.name, file=io.BytesIO(probe.read_bytes()),
                            headers=Headers({"content-type": "audio/wav"}))
        result = await classify_route(upload)
        self.assertIn(result["command"], COMMANDS)
        self.assertFalse(status()["execution_enabled"])
        with self.assertRaises(HTTPException) as exc:
            await execute_route(upload)
        self.assertEqual(exc.exception.status_code, 503)

    def test_malformed_audio_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid WAV"):
            classify_custom_command(b"not audio")


if __name__ == "__main__":
    unittest.main()
