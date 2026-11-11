"""Unit tests for the backend's hierarchical decision rules (no models needed)."""

import unittest

from backend.policy import (
    FACE_ACCEPT,
    FACE_REJECT,
    VOICE_ACCEPT,
    VOICE_REJECT,
    HierarchicalPolicy,
    face_only_stage,
    is_accepted,
)

# The validated policy from results/hierarchical/selected_thresholds_audit.json
POLICY = HierarchicalPolicy(low_face=0.17164957054608299, high_face=0.290093, voice=0.20283866686207327)


class HierarchicalPolicyTests(unittest.TestCase):
    def test_confident_face_scores_decide_without_voice(self):
        self.assertEqual(POLICY.face_stage(0.90), FACE_ACCEPT)
        self.assertEqual(POLICY.face_stage(0.01), FACE_REJECT)

    def test_boundaries_match_apply_sequential_policy(self):
        # face >= high -> accept, face <= low -> reject (both inclusive)
        self.assertEqual(POLICY.face_stage(POLICY.high_face), FACE_ACCEPT)
        self.assertEqual(POLICY.face_stage(POLICY.low_face), FACE_REJECT)

    def test_uncertain_face_scores_go_to_voice(self):
        self.assertIsNone(POLICY.face_stage(0.20))
        self.assertEqual(POLICY.voice_stage(POLICY.voice), VOICE_ACCEPT)
        self.assertEqual(POLICY.voice_stage(0.10), VOICE_REJECT)

    def test_face_only_mode(self):
        self.assertEqual(face_only_stage(0.5, 0.29), FACE_ACCEPT)
        self.assertEqual(face_only_stage(0.1, 0.29), FACE_REJECT)

    def test_is_accepted(self):
        self.assertTrue(is_accepted(FACE_ACCEPT))
        self.assertTrue(is_accepted(VOICE_ACCEPT))
        self.assertFalse(is_accepted(FACE_REJECT))
        self.assertFalse(is_accepted(VOICE_REJECT))


if __name__ == "__main__":
    unittest.main()
