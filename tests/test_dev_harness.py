"""
test_dev_harness.py — Validates the headless HTTP test harness.

Tests the mock functions, log capture, HTTP parsing, and response format
WITHOUT booting the full pipeline (no LLM keys or audio hardware needed).

Run: pytest tests/test_test_harness.py -v
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestMockSpeak(unittest.TestCase):
    """_mock_speak captures text and emotion correctly."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_captures_text_and_emotion(self):
        from assistant.dev_harness import _mock_speak, _captured
        _captured.clear()
        self._run(_mock_speak("Hello world", bridge=None, emotion="happy"))
        self.assertEqual(len(_captured), 1)
        self.assertEqual(_captured[0]["text"], "Hello world")
        self.assertEqual(_captured[0]["emotion"], "happy")

    def test_default_emotion_is_neutral(self):
        from assistant.dev_harness import _mock_speak, _captured
        _captured.clear()
        self._run(_mock_speak("Test"))
        self.assertEqual(_captured[0]["emotion"], "neutral")

    def test_returns_true(self):
        from assistant.test_harness import _mock_speak
        result = self._run(_mock_speak("anything"))
        self.assertTrue(result)

    def test_multiple_captures(self):
        from assistant.dev_harness import _mock_speak, _captured
        _captured.clear()
        self._run(_mock_speak("First"))
        self._run(_mock_speak("Second"))
        self.assertEqual(len(_captured), 2)
        self.assertEqual(_captured[0]["text"], "First")
        self.assertEqual(_captured[1]["text"], "Second")


class TestMockSpeakStreaming(unittest.TestCase):
    """_mock_speak_streaming consumes async generator and captures full text."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_consumes_stream(self):
        from assistant.dev_harness import _mock_speak_streaming, _captured

        async def token_gen():
            for tok in ["Hello ", "world ", "!"]:
                yield tok

        _captured.clear()
        success, text = self._run(_mock_speak_streaming(token_gen(), emotion="excited"))
        self.assertTrue(success)
        self.assertEqual(text, "Hello world !")
        self.assertEqual(_captured[0]["text"], "Hello world !")
        self.assertEqual(_captured[0]["emotion"], "excited")

    def test_empty_stream(self):
        from assistant.dev_harness import _mock_speak_streaming, _captured

        async def empty_gen():
            return
            yield  # noqa: unreachable — makes it an async generator

        _captured.clear()
        success, text = self._run(_mock_speak_streaming(empty_gen()))
        self.assertTrue(success)
        self.assertEqual(text, "")


class TestMockFinishTurn(unittest.TestCase):
    """_mock_finish_turn is a no-op that doesn't raise."""

    def test_noop(self):
        from assistant.dev_harness import _mock_finish_turn
        result = asyncio.get_event_loop().run_until_complete(
            _mock_finish_turn(bridge=None)
        )
        self.assertIsNone(result)


class TestLogCapture(unittest.TestCase):
    """_LogCapture handler collects INFO+ log records."""

    def test_captures_info(self):
        from assistant.dev_harness import _LogCapture
        handler = _LogCapture()
        test_logger = logging.getLogger("test.logcapture")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        try:
            test_logger.info("visible message")
            test_logger.debug("hidden message")
            self.assertEqual(len(handler.records), 1)
            self.assertIn("visible message", handler.records[0])
        finally:
            test_logger.removeHandler(handler)

    def test_format_includes_level(self):
        from assistant.dev_harness import _LogCapture
        handler = _LogCapture()
        test_logger = logging.getLogger("test.logcapture.fmt")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        try:
            test_logger.warning("uh oh")
            self.assertTrue(handler.records[0].startswith("WARNING"))
        finally:
            test_logger.removeHandler(handler)


class TestBuildHttpResponse(unittest.TestCase):
    """_build_http_response produces valid HTTP/1.1 response bytes."""

    def test_structure(self):
        from assistant.dev_harness import _build_http_response
        raw = _build_http_response("200 OK", {"key": "value"})
        header, body = raw.split(b"\r\n\r\n", 1)
        self.assertIn(b"HTTP/1.1 200 OK", header)
        self.assertIn(b"Content-Type: application/json", header)
        parsed = json.loads(body)
        self.assertEqual(parsed["key"], "value")

    def test_content_length_matches(self):
        from assistant.dev_harness import _build_http_response
        raw = _build_http_response("200 OK", {"unicode": "☃"})
        header, body = raw.split(b"\r\n\r\n", 1)
        for line in header.decode().split("\r\n"):
            if line.lower().startswith("content-length:"):
                declared = int(line.split(":")[1].strip())
                self.assertEqual(declared, len(body))
                break
        else:
            self.fail("Content-Length header missing")

    def test_unicode_body(self):
        from assistant.dev_harness import _build_http_response
        raw = _build_http_response("200 OK", {"text": "café"})
        _, body = raw.split(b"\r\n\r\n", 1)
        parsed = json.loads(body)
        self.assertEqual(parsed["text"], "café")


class TestIntentParsing(unittest.TestCase):
    """Intent is extracted from captured log lines."""

    def test_parses_intent_from_log_line(self):
        logs = [
            "INFO Transcription (Chat): \"what time is it\"",
            "INFO [REGEX] Matched → get_time",
            "INFO Intent: get_time",
            "INFO Response: \"It is 3:45 PM\"",
        ]
        intent = "unknown"
        for line in logs:
            if "Intent: " in line:
                intent = line.split("Intent: ", 1)[1].strip()
                break
        self.assertEqual(intent, "get_time")

    def test_defaults_to_unknown(self):
        logs = ["INFO some random log"]
        intent = "unknown"
        for line in logs:
            if "Intent: " in line:
                intent = line.split("Intent: ", 1)[1].strip()
                break
        self.assertEqual(intent, "unknown")


if __name__ == "__main__":
    unittest.main()
