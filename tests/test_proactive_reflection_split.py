"""
test_proactive_reflection_split.py — Verify S13 Task 2 split of proactive.py → reflection.py

Tests:
  1. Module-level structure: reflection.py has the right public API / constants
  2. proactive.py no longer contains reflection code
  3. _parse_reflection_response handles valid / invalid / edge-case JSON
  4. call_async is used instead of asyncio.new_event_loop
  5. start_analyzer delegates to reflection.start; stop_analyzer delegates to reflection.stop
"""

import inspect
import json
import textwrap
import unittest


class TestModuleStructure(unittest.TestCase):
    """Verify the split produced correct module-level exports."""

    def test_reflection_has_public_api(self):
        """reflection.py exposes start() and stop()."""
        from assistant import reflection
        self.assertTrue(callable(reflection.start))
        self.assertTrue(callable(reflection.stop))

    def test_reflection_has_constants(self):
        """reflection.py has the reflection constants."""
        from assistant import reflection
        self.assertEqual(reflection.REFLECTION_INTERVAL_CONVERSATIONS, 20)
        self.assertEqual(reflection.REFLECTION_INTERVAL_HOURS, 24)
        self.assertEqual(reflection._REFLECTION_CHECK_SECONDS, 30 * 60)
        self.assertEqual(reflection._REFLECTION_MEMORY_LIMIT, 20)
        self.assertEqual(reflection._PREFERENCE_DECAY_INTERVAL_DAYS, 30)

    def test_reflection_has_private_functions(self):
        """reflection.py has all the private reflection functions."""
        from assistant import reflection
        private_names = [
            "_reflection_loop",
            "_maybe_run_reflection",
            "_run_reflection_cycle",
            "_process_discovered_preferences",
            "_maybe_run_preference_decay",
            "_gather_reflection_context",
            "_build_reflection_prompt",
            "_parse_reflection_response",
        ]
        for name in private_names:
            self.assertTrue(
                hasattr(reflection, name),
                f"reflection.py missing function: {name}",
            )

    def test_proactive_public_api_unchanged(self):
        """proactive.py still exports start_analyzer, stop_analyzer, get_queue."""
        from assistant import proactive
        self.assertTrue(callable(proactive.start_analyzer))
        self.assertTrue(callable(proactive.stop_analyzer))
        self.assertTrue(callable(proactive.get_queue))

    def test_proactive_no_reflection_code(self):
        """proactive.py no longer contains reflection functions or constants."""
        from assistant import proactive
        reflection_names = [
            "_start_reflection",
            "_stop_reflection",
            "_reflection_loop",
            "_maybe_run_reflection",
            "_run_reflection_cycle",
            "_process_discovered_preferences",
            "_maybe_run_preference_decay",
            "_gather_reflection_context",
            "_build_reflection_prompt",
            "_parse_reflection_response",
            "REFLECTION_INTERVAL_CONVERSATIONS",
            "REFLECTION_INTERVAL_HOURS",
            "_REFLECTION_CHECK_SECONDS",
            "_REFLECTION_MEMORY_LIMIT",
            "_PREFERENCE_DECAY_INTERVAL_DAYS",
            "_reflection_thread",
            "_reflection_stop_event",
        ]
        for name in reflection_names:
            self.assertFalse(
                hasattr(proactive, name),
                f"proactive.py still has reflection code: {name}",
            )

    def test_proactive_no_sqlite3_import(self):
        """proactive.py should not import sqlite3 (unused after reflection moved out)."""
        import assistant.proactive as mod
        source = inspect.getsource(mod)
        self.assertNotIn("import sqlite3", source)

    def test_proactive_no_asyncio_new_event_loop(self):
        """proactive.py should not use asyncio.new_event_loop()."""
        import assistant.proactive as mod
        source = inspect.getsource(mod)
        self.assertNotIn("asyncio.new_event_loop", source)
        self.assertNotIn("loop.run_until_complete", source)
        self.assertNotIn("loop.close()", source)

    def test_reflection_no_asyncio_new_event_loop(self):
        """reflection.py should not use asyncio.new_event_loop()."""
        import assistant.reflection as mod
        source = inspect.getsource(mod)
        self.assertNotIn("asyncio.new_event_loop", source)
        self.assertNotIn("loop.run_until_complete", source)
        self.assertNotIn("loop.close()", source)

    def test_proactive_uses_call_async(self):
        """proactive._analyze_with_llm uses call_async."""
        import assistant.proactive as mod
        source = inspect.getsource(mod._analyze_with_llm)
        self.assertIn("call_async", source)
        self.assertIn("from .core.asyncio_utils import call_async", source)

    def test_reflection_uses_call_async(self):
        """reflection._run_reflection_cycle uses call_async."""
        import assistant.reflection as mod
        source = inspect.getsource(mod._run_reflection_cycle)
        self.assertIn("call_async", source)
        self.assertIn("from .core.asyncio_utils import call_async", source)

    def test_reflection_logger_name(self):
        """reflection.py uses logger name 'proactive' for log continuity."""
        import assistant.reflection as mod
        self.assertEqual(mod.logger.name, "proactive")


class TestParseReflectionResponse(unittest.TestCase):
    """Verify _parse_reflection_response handles valid and invalid LLM output."""

    def _parse(self, response: str):
        from assistant.reflection import _parse_reflection_response
        return _parse_reflection_response(response)

    def test_valid_response(self):
        """Valid JSON with deltas + preferences parses correctly."""
        response = json.dumps({
            "deltas": {
                "trust": 0.02,
                "warmth": -0.01,
                "sass": 0.0,
                "openness": 0.0,
                "patience": 0.0,
                "playfulness": 0.03,
            },
            "reasoning": "User shared personal details, increasing trust.",
            "preferences": [
                {
                    "key": "music_app",
                    "value": "spotify",
                    "category": "app_routing",
                    "confidence": 0.5,
                    "evidence": "User said 'play on Spotify' three times.",
                }
            ],
        })
        deltas, reasoning, prefs = self._parse(response)
        self.assertAlmostEqual(deltas["trust"], 0.02)
        self.assertAlmostEqual(deltas["warmth"], -0.01)
        self.assertAlmostEqual(deltas["playfulness"], 0.03)
        # Zero deltas should be excluded
        self.assertNotIn("sass", deltas)
        self.assertNotIn("openness", deltas)
        self.assertNotIn("patience", deltas)
        self.assertIn("trust", reasoning.lower())
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["key"], "music_app")

    def test_empty_preferences(self):
        """Response with empty preferences array is fine."""
        response = json.dumps({
            "deltas": {"trust": 0.01},
            "reasoning": "Minor trust bump.",
            "preferences": [],
        })
        deltas, reasoning, prefs = self._parse(response)
        self.assertAlmostEqual(deltas["trust"], 0.01)
        self.assertEqual(prefs, [])

    def test_no_preferences_key(self):
        """Response without preferences key defaults to empty list."""
        response = json.dumps({
            "deltas": {"warmth": 0.02},
            "reasoning": "Warmth bump.",
        })
        deltas, reasoning, prefs = self._parse(response)
        self.assertAlmostEqual(deltas["warmth"], 0.02)
        self.assertEqual(prefs, [])

    def test_delta_clamping(self):
        """Deltas exceeding +-0.05 are clamped."""
        response = json.dumps({
            "deltas": {"trust": 0.9, "sass": -0.3},
            "reasoning": "Big changes.",
            "preferences": [],
        })
        deltas, _, _ = self._parse(response)
        self.assertAlmostEqual(deltas["trust"], 0.05)
        self.assertAlmostEqual(deltas["sass"], -0.05)

    def test_invalid_json(self):
        """Non-JSON response returns empty tuple."""
        deltas, reasoning, prefs = self._parse("not json at all")
        self.assertEqual(deltas, {})
        self.assertEqual(reasoning, "")
        self.assertEqual(prefs, [])

    def test_markdown_fenced_json(self):
        """JSON wrapped in markdown fences is handled."""
        inner = json.dumps({
            "deltas": {"trust": 0.01},
            "reasoning": "Fence test.",
            "preferences": [],
        })
        response = f"```json\n{inner}\n```"
        deltas, _, _ = self._parse(response)
        self.assertAlmostEqual(deltas["trust"], 0.01)

    def test_invalid_category_filtered(self):
        """Preferences with invalid categories are silently dropped."""
        response = json.dumps({
            "deltas": {},
            "reasoning": "No changes.",
            "preferences": [
                {
                    "key": "test",
                    "value": "val",
                    "category": "INVALID_CATEGORY",
                    "confidence": 0.5,
                    "evidence": "test",
                }
            ],
        })
        _, _, prefs = self._parse(response)
        self.assertEqual(prefs, [])

    def test_unknown_trait_ignored(self):
        """Traits not in the valid set are silently ignored."""
        response = json.dumps({
            "deltas": {"trust": 0.01, "aggression": 0.5},
            "reasoning": "Test.",
            "preferences": [],
        })
        deltas, _, _ = self._parse(response)
        self.assertIn("trust", deltas)
        self.assertNotIn("aggression", deltas)

    def test_confidence_clamping(self):
        """Preference confidence is clamped to [0.0, 1.0]."""
        response = json.dumps({
            "deltas": {},
            "reasoning": "No changes.",
            "preferences": [
                {
                    "key": "test",
                    "value": "val",
                    "category": "app_routing",
                    "confidence": 5.0,
                    "evidence": "test",
                }
            ],
        })
        _, _, prefs = self._parse(response)
        self.assertEqual(len(prefs), 1)
        self.assertAlmostEqual(prefs[0]["confidence"], 1.0)


class TestDelegation(unittest.TestCase):
    """Verify start_analyzer/stop_analyzer delegates to reflection.start/stop."""

    def test_start_analyzer_calls_reflection_start(self):
        """start_analyzer source contains reflection.start()."""
        import assistant.proactive as mod
        source = inspect.getsource(mod.start_analyzer)
        self.assertIn("reflection.start()", source)

    def test_stop_analyzer_calls_reflection_stop(self):
        """stop_analyzer source contains reflection.stop()."""
        import assistant.proactive as mod
        source = inspect.getsource(mod.stop_analyzer)
        self.assertIn("reflection.stop()", source)


class TestBuildReflectionPrompt(unittest.TestCase):
    """Verify _build_reflection_prompt produces valid prompt text."""

    def test_prompt_contains_traits_and_prefs(self):
        from assistant.reflection import _build_reflection_prompt
        traits_json = '{"trust": 0.5}'
        memory_summaries = "[2026-05-10 10:00] (small_talk) User: hi"
        preferences_json = '[{"key": "music_app", "value": "spotify"}]'
        prompt = _build_reflection_prompt(traits_json, memory_summaries, preferences_json)
        self.assertIn("trust", prompt)
        self.assertIn("music_app", prompt)
        self.assertIn("hi", prompt)
        self.assertIn("TRAIT MEANINGS", prompt)
        self.assertIn("Preference discovery rules", prompt)


if __name__ == "__main__":
    unittest.main()
