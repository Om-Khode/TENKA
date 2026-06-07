"""tests/test_reflection.py — Tests for assistant/reflection.py"""

import json
import unittest


class TestParseReflectionResponse(unittest.TestCase):
    """Verify _parse_reflection_response handles valid and invalid LLM output."""

    def _parse(self, response: str):
        from assistant.reflection import _parse_reflection_response
        return _parse_reflection_response(response)

    def test_valid_response_with_deltas_and_preferences(self):
        response = json.dumps({
            "deltas": {"trust": 0.03, "warmth": -0.02, "sass": 0.0},
            "reasoning": "User shared personal info",
            "preferences": [
                {"key": "music_app", "value": "spotify",
                 "category": "app_routing", "confidence": 0.6,
                 "evidence": "User said play on spotify 3 times"}
            ]
        })
        deltas, reasoning, prefs = self._parse(response)
        self.assertAlmostEqual(deltas["trust"], 0.03)
        self.assertAlmostEqual(deltas["warmth"], -0.02)
        self.assertNotIn("sass", deltas)  # zero deltas skipped
        self.assertIn("personal", reasoning)
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["key"], "music_app")

    def test_zero_deltas_all_skipped(self):
        response = json.dumps({
            "deltas": {"trust": 0.0, "warmth": 0.0, "sass": 0.0,
                       "openness": 0.0, "patience": 0.0, "playfulness": 0.0},
            "reasoning": "Routine interactions",
            "preferences": []
        })
        deltas, reasoning, prefs = self._parse(response)
        self.assertEqual(deltas, {})
        self.assertEqual(prefs, [])

    def test_deltas_clamped_to_005(self):
        response = json.dumps({
            "deltas": {"trust": 0.5, "warmth": -0.3},
            "reasoning": "Extreme values",
            "preferences": []
        })
        deltas, _, _ = self._parse(response)
        self.assertAlmostEqual(deltas["trust"], 0.05)
        self.assertAlmostEqual(deltas["warmth"], -0.05)

    def test_invalid_json_returns_empty(self):
        deltas, reasoning, prefs = self._parse("not json at all")
        self.assertEqual(deltas, {})
        self.assertEqual(reasoning, "")
        self.assertEqual(prefs, [])

    def test_markdown_fenced_json_parsed(self):
        inner = json.dumps({
            "deltas": {"trust": 0.01},
            "reasoning": "test",
            "preferences": []
        })
        response = f"```json\n{inner}\n```"
        deltas, _, _ = self._parse(response)
        self.assertAlmostEqual(deltas["trust"], 0.01)

    def test_invalid_trait_names_ignored(self):
        response = json.dumps({
            "deltas": {"trust": 0.01, "aggression": 0.05, "humor": -0.02},
            "reasoning": "test",
            "preferences": []
        })
        deltas, _, _ = self._parse(response)
        self.assertIn("trust", deltas)
        self.assertNotIn("aggression", deltas)
        self.assertNotIn("humor", deltas)

    def test_invalid_preference_category_ignored(self):
        response = json.dumps({
            "deltas": {},
            "reasoning": "test",
            "preferences": [
                {"key": "k", "value": "v", "category": "invalid_cat",
                 "confidence": 0.5, "evidence": "test"}
            ]
        })
        _, _, prefs = self._parse(response)
        self.assertEqual(len(prefs), 0)

    def test_preference_confidence_clamped(self):
        response = json.dumps({
            "deltas": {},
            "reasoning": "test",
            "preferences": [
                {"key": "k", "value": "v", "category": "app_routing",
                 "confidence": 5.0, "evidence": "test"}
            ]
        })
        _, _, prefs = self._parse(response)
        self.assertEqual(len(prefs), 1)
        self.assertAlmostEqual(prefs[0]["confidence"], 1.0)

    def test_preference_missing_required_fields_skipped(self):
        response = json.dumps({
            "deltas": {},
            "reasoning": "test",
            "preferences": [
                {"key": "", "value": "v", "category": "app_routing"},
                {"key": "k", "value": "", "category": "app_routing"},
                {"key": "k", "value": "v", "category": ""},
            ]
        })
        _, _, prefs = self._parse(response)
        self.assertEqual(len(prefs), 0)


class TestReflectionModuleStructure(unittest.TestCase):
    """Verify the reflection module has the expected public API."""

    def test_importable(self):
        import assistant.reflection as ref
        self.assertTrue(callable(ref.start))
        self.assertTrue(callable(ref.stop))

    def test_constants_present(self):
        from assistant.reflection import (
            REFLECTION_INTERVAL_CONVERSATIONS,
            REFLECTION_INTERVAL_HOURS,
        )
        self.assertEqual(REFLECTION_INTERVAL_CONVERSATIONS, 20)
        self.assertEqual(REFLECTION_INTERVAL_HOURS, 24)


if __name__ == "__main__":
    unittest.main()
