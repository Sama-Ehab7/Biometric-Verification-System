"""Tests for the CSV-backed custom fingerprint demo alias adapter.

Valid inference calls use only images in custom-data/fingerprint/test/.
"""

import unittest
import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile

from backend.custom_fingerprint_demo import (
    custom_fingerprint_identities,
    custom_fingerprint_test_images,
    verify_custom_fingerprint_route,
    verify_custom_fingerprint_upload_route,
)
from src.verification.custom_fingerprint_demo import (
    resolve_custom_fingerprint_identity,
    resolve_custom_fingerprint_test_image,
    verify_custom_fingerprint,
)
from src.verification.fingerprint_inference import verify_fingerprint


ROOT = Path(__file__).resolve().parents[1]


class CustomFingerprintDemoTests(unittest.TestCase):
    def test_aliases_resolve_from_mapping_csv(self):
        self.assertEqual(resolve_custom_fingerprint_identity("nour"), "DB1_B_101")
        self.assertEqual(resolve_custom_fingerprint_identity("sama"), "DB1_B_102")
        self.assertEqual(resolve_custom_fingerprint_identity(" Nour "), "DB1_B_101")

    def test_unknown_alias_rejected_safely(self):
        with self.assertRaisesRegex(ValueError, "Unknown custom fingerprint alias"):
            resolve_custom_fingerprint_identity("unknown")
        with self.assertRaises(HTTPException) as raised:
            verify_custom_fingerprint_route("unknown", "nour", "101_7.tif")
        self.assertEqual(raised.exception.status_code, 400)

    def test_api_lists_only_custom_test_images(self):
        self.assertEqual(custom_fingerprint_identities(), {"identities": ["nour", "sama"]})
        self.assertEqual(
            custom_fingerprint_test_images(),
            {"images": {"nour": ["101_7.tif", "101_8.tif"], "sama": ["102_7.tif", "102_8.tif"]}},
        )

    def test_alias_and_legacy_database_qualified_inference(self):
        cases = [
            ("nour", "DB1_B_101", "nour", "101_7.tif"),
            ("sama", "DB1_B_102", "sama", "102_7.tif"),
        ]
        for alias, database_id, probe_alias, image_name in cases:
            with self.subTest(alias=alias):
                image = resolve_custom_fingerprint_test_image(probe_alias, image_name)
                self.assertIn("custom-data/fingerprint/test", image.as_posix())

                aliased = verify_custom_fingerprint(image, alias)
                direct = verify_fingerprint(image, database_id)
                self.assertEqual(aliased["custom_identity"], alias)
                self.assertEqual(aliased["claimed_identity"], database_id)
                self.assertEqual(aliased["score"], direct["score"])
                self.assertEqual(aliased["threshold"], direct["threshold"])
                self.assertEqual(aliased["decision"], direct["decision"])
                self.assertIn(direct["decision"], {"ACCEPT", "REJECT"})

    def test_api_verifies_using_selected_alias(self):
        for alias, probe_alias, image_name, expected_id in [
            ("nour", "nour", "101_7.tif", "DB1_B_101"),
            ("sama", "sama", "102_7.tif", "DB1_B_102"),
        ]:
            with self.subTest(alias=alias):
                result = verify_custom_fingerprint_route(alias, probe_alias, image_name)
                self.assertEqual(result["custom_identity"], alias)
                self.assertEqual(result["claimed_identity"], expected_id)
                self.assertIn(result["decision"], {"ACCEPT", "REJECT"})

    def test_uploaded_fingerprint_uses_saved_svm_and_mapping(self):
        source = ROOT / "custom-data/fingerprint/test/nour/101_8.tif"
        with source.open("rb") as handle:
            image = UploadFile(filename="probe.tif", file=BytesIO(handle.read()))
        result = asyncio.run(verify_custom_fingerprint_upload_route("nour", image))
        direct = verify_fingerprint(source, "DB1_B_101")
        self.assertEqual(result["claimed_identity"], "DB1_B_101")
        self.assertEqual(result["score"], direct["score"])
        self.assertEqual(result["threshold"], direct["threshold"])
        self.assertEqual(result["decision"], direct["decision"])

    def test_uploaded_fingerprint_rejects_unknown_alias(self):
        source = ROOT / "custom-data/fingerprint/test/nour/101_8.tif"
        with source.open("rb") as handle:
            image = UploadFile(filename="probe.tif", file=BytesIO(handle.read()))
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(verify_custom_fingerprint_upload_route("unknown", image))
        self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
