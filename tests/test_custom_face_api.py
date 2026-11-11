"""Route-level checks without requiring an external HTTP test client."""
import io
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from backend.custom_face_demo import identify_route, status, verify_camera_route, verify_route

ROOT = Path(__file__).resolve().parents[1]


def upload(path: Path, content_type="image/jpeg"):
    return UploadFile(filename=path.name, file=io.BytesIO(path.read_bytes()),
                      headers=Headers({"content-type": content_type}))


class CustomFaceApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_and_verify_route(self):
        self.assertEqual(status()["identities"], ["nour", "sama"])
        result = await verify_route("nour", "svm", upload(ROOT / "data/raw/custom/face/test/nour/face_01.jpg"))
        self.assertIn(result["decision"], {"ACCEPT", "REJECT"})
        self.assertGreaterEqual(result["score"], 0)

    async def test_identify_route_and_invalid_inputs(self):
        probe = next((ROOT / "data/raw/custom/face/test/sama").iterdir())
        result = await identify_route("random_forest", upload(probe))
        self.assertIn(result["identity"], {"nour", "sama", "unknown"})
        with self.assertRaises(HTTPException) as exc:
            await verify_route("unknown", "svm", upload(ROOT / "data/raw/custom/face/test/nour/face_01.jpg"))
        self.assertEqual(exc.exception.status_code, 400)
        with self.assertRaises(HTTPException) as exc:
            await verify_route("nour", "svm", upload(ROOT / "data/raw/custom/face/test/nour/face_01.jpg", "text/plain"))
        self.assertEqual(exc.exception.status_code, 415)

    async def test_live_camera_guard_rejects_no_face(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((180, 180, 3), dtype=np.uint8))
        self.assertTrue(ok)
        image = UploadFile(filename="empty-scene.jpg", file=io.BytesIO(encoded.tobytes()),
                           headers=Headers({"content-type": "image/jpeg"}))
        with self.assertRaises(HTTPException) as exc:
            await verify_camera_route("nour", "svm", image)
        self.assertEqual(exc.exception.status_code, 422)
        self.assertIn("No face detected", exc.exception.detail)


if __name__ == "__main__":
    unittest.main()
