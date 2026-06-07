"""
test_wake_word_fallback.py — Regression test for the wake-word rename feedback-loop bug.

Scenario: user renames assistant via /set assistant_name Luna → restart.
WAKE_WORD_MODEL_PATH becomes models/luna.onnx (missing).
Old code silently fell back to hey_jarvis_v0.1 with the custom-tuned 0.02 threshold,
which false-fired on TTS audio → infinite "Yes!" loop.

New policy (tested here):
  - Custom model present  → load it
  - Custom missing + builtin explicitly set → opt-in fallback with loud warning
  - Custom missing + no builtin → return None (wake word disabled, PTT works)

Run: python test_wake_word_fallback.py
"""

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from assistant.io.audio import wake_word
from assistant import config


class _CaptureHandler(logging.Handler):
    """Collect log records for assertions."""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def messages_at(self, level):
        return [r.getMessage() for r in self.records if r.levelno == level]


class TestWakeModelFallback(unittest.TestCase):
    """_load_wake_model() policy under each config combination."""

    def setUp(self):
        # Snapshot config so we can restore
        self._saved = {
            "path": config.WAKE_WORD_MODEL_PATH,
            "builtin": config.WAKE_WORD_BUILTIN,
            "threshold": config.WAKE_WORD_THRESHOLD,
            "framework": config.WAKE_WORD_INFERENCE_FRAMEWORK,
        }

        # Attach capture handler to the wake_word logger
        self._handler = _CaptureHandler()
        self._handler.setLevel(logging.DEBUG)
        wake_word.logger.addHandler(self._handler)
        wake_word.logger.setLevel(logging.DEBUG)

        # Fake Model constructor — records its calls, returns a mock
        self.model_calls = []
        def _fake_model(**kwargs):
            self.model_calls.append(kwargs)
            m = MagicMock()
            m.models = {"fake": object()}
            return m
        self._FakeModel = _fake_model

    def tearDown(self):
        config.WAKE_WORD_MODEL_PATH = self._saved["path"]
        config.WAKE_WORD_BUILTIN = self._saved["builtin"]
        config.WAKE_WORD_THRESHOLD = self._saved["threshold"]
        config.WAKE_WORD_INFERENCE_FRAMEWORK = self._saved["framework"]
        wake_word.logger.removeHandler(self._handler)

    def _point_to_missing(self):
        tmp = Path(tempfile.mkdtemp()) / "nope_does_not_exist.onnx"
        config.WAKE_WORD_MODEL_PATH = tmp
        self.assertFalse(tmp.exists())
        return tmp

    def _point_to_existing(self):
        tmp = Path(tempfile.mkdtemp()) / "exists.onnx"
        tmp.write_bytes(b"not a real onnx, but the file exists")
        config.WAKE_WORD_MODEL_PATH = tmp
        return tmp

    def test_custom_model_present_loads_custom(self):
        path = self._point_to_existing()
        config.WAKE_WORD_BUILTIN = ""

        result = wake_word._load_wake_model(self._FakeModel)

        self.assertIsNotNone(result)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(self.model_calls[0]["wakeword_models"], [str(path)])

    def test_custom_missing_no_builtin_disables_wake(self):
        """The main bug: previously this path silently loaded hey_jarvis."""
        self._point_to_missing()
        config.WAKE_WORD_BUILTIN = ""

        result = wake_word._load_wake_model(self._FakeModel)

        self.assertIsNone(result, "Wake word must be disabled, not fall back silently")
        self.assertEqual(self.model_calls, [],
                         "Must not construct any Model when disabled")
        warnings = self._handler.messages_at(logging.WARNING)
        self.assertTrue(
            any("disabled" in w.lower() for w in warnings),
            f"Expected a 'disabled' warning, got: {warnings}",
        )
        self.assertTrue(
            any("push-to-talk" in w.lower() or "press 'v'" in w.lower()
                for w in warnings),
            "Warning should mention push-to-talk alternative",
        )

    def test_custom_missing_with_builtin_opts_in_with_warning(self):
        """Opt-in fallback: user explicitly sets WAKE_WORD_BUILTIN in .env."""
        self._point_to_missing()
        config.WAKE_WORD_BUILTIN = "hey_jarvis_v0.1"

        result = wake_word._load_wake_model(self._FakeModel)

        self.assertIsNotNone(result, "Opt-in fallback should load the built-in")
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(self.model_calls[0]["wakeword_models"],
                         ["hey_jarvis_v0.1"])

        warnings = self._handler.messages_at(logging.WARNING)
        self.assertTrue(
            any("threshold" in w.lower() for w in warnings),
            "Must warn about threshold mismatch when using built-in fallback",
        )
        self.assertTrue(
            any("hey_jarvis" in w.lower() for w in warnings),
            "Warning should name the built-in being used",
        )

    def test_builtin_default_is_empty(self):
        """Config default must be empty so builtin is opt-in, not silent."""
        # Reload config fresh
        import importlib
        importlib.reload(config)
        self.assertEqual(
            config.WAKE_WORD_BUILTIN, "",
            "WAKE_WORD_BUILTIN must default to empty so missing custom "
            "models disable wake instead of silently falling back.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
