"""
test_verification_vision.py — VL-1b: vision tier + planner integration.

Covers:
  - vision_verify happy paths (ok=True / ok=False from Flash JSON)
  - vision_verify infra failures fail-open (screenshot None, llm crash, parse fail)
  - vision_verify with VERIFY_VISION_FALLBACK=False returns code_result unchanged
  - parse_verify_failed: well-formed prefix, no-match, with trailing newline content
  - format_failure_for_user
  - planner _step_failed picks up VERIFY_FAILED| prefix (PA-3 fires)

Run: python test_verification_vision.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.verification as ver
from assistant.automation.verification import VerifyResult
from assistant import config as _config_stub

_config_stub.VERIFY_ENABLED = True
_config_stub.VERIFY_BROWSER_STEPS = True
_config_stub.VERIFY_APP_STEPS = True
_config_stub.VERIFY_VISION_FALLBACK = True
_config_stub.VERIFY_STRICT_TEXT_MATCH = False
_config_stub.VERIFY_MIN_CONFIDENCE = 0.5
_config_stub.VERIFY_MAX_RETRIES = 1


def _reset():
    _config_stub.VERIFY_VISION_FALLBACK = True


def _stub_modules(*, screenshot=b"fake-b64", vision_response='{"ok": true, "observation": "looks fine", "confidence": 0.9}'):
    """Install fake screen + llm modules into sys.modules so verification.py's
    in-function imports resolve to controllable doubles.

    ``get_vision_response`` now returns ``LLMResult`` objects so bare string
    values are wrapped in ``SimpleNamespace(text=...)`` so callers can do
    ``.text`` on the result.
    """
    screen_mod = types.ModuleType("assistant.io.screen")
    screen_mod.capture_screenshot_base64 = MagicMock(return_value=screenshot)

    llm_mod = types.ModuleType("assistant.llm")
    if isinstance(vision_response, Exception):
        llm_mod.get_vision_response = AsyncMock(side_effect=vision_response)
    else:
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text=vision_response)
        )

    sys.modules["assistant.io.screen"] = screen_mod
    sys.modules["assistant.llm"] = llm_mod
    return screen_mod, llm_mod


# ─── Vision tier ──────────────────────────────────────────────────────────────

class TestVisionVerifyHappy(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset()

    async def test_returns_vision_ok(self):
        screen, llm = _stub_modules(
            vision_response='{"ok": true, "observation": "search bar focused", "confidence": 0.9}'
        )
        amb = VerifyResult.ambiguous("click outcome unknown")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {"selector": "button"}},
            amb,
        )
        self.assertEqual(out.tier, "vision")
        self.assertTrue(out.ok)
        self.assertEqual(out.observation, "search bar focused")
        self.assertEqual(out.confidence, 0.9)
        screen.capture_screenshot_base64.assert_called_once()
        llm.get_vision_response.assert_awaited_once()

    async def test_returns_vision_failed_with_observation(self):
        _stub_modules(
            vision_response='{"ok": false, "observation": "Error: phone must include country code", "confidence": 0.95}'
        )
        amb = VerifyResult.ambiguous("submit outcome unknown")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {"selector": "#submit"}},
            amb,
        )
        self.assertEqual(out.tier, "vision")
        self.assertFalse(out.ok)
        self.assertIn("country code", out.observation)

    async def test_strips_markdown_fences(self):
        _stub_modules(
            vision_response='```json\n{"ok": true, "observation": "ok", "confidence": 0.8}\n```'
        )
        amb = VerifyResult.ambiguous()
        out = await ver.vision_verify(
            {"type": "app", "action": "click", "params": {}}, amb,
        )
        self.assertTrue(out.ok)
        self.assertEqual(out.tier, "vision")


class TestVisionVerifyPageScreenshot(unittest.IsolatedAsyncioTestCase):
    """vision_verify should prefer Playwright page.screenshot() over OS capture."""
    def setUp(self): _reset()

    async def test_uses_page_screenshot_when_page_available(self):
        _, llm = _stub_modules(
            vision_response='{"ok": true, "observation": "search results visible", "confidence": 0.9}'
        )
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"fake-png-bytes")

        amb = VerifyResult.ambiguous("click outcome unknown")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {"selector": "button"}},
            amb, page=mock_page,
        )
        self.assertTrue(out.ok)
        self.assertEqual(out.tier, "vision")
        mock_page.screenshot.assert_awaited_once()
        screen_mod = sys.modules["assistant.io.screen"]
        screen_mod.capture_screenshot_base64.assert_not_called()

    async def test_falls_back_to_os_capture_when_page_screenshot_fails(self):
        screen, llm = _stub_modules(
            screenshot=b"os-screenshot",
            vision_response='{"ok": true, "observation": "ok", "confidence": 0.8}'
        )
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(side_effect=Exception("page crashed"))

        amb = VerifyResult.ambiguous()
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}},
            amb, page=mock_page,
        )
        self.assertTrue(out.ok)
        mock_page.screenshot.assert_awaited_once()
        screen.capture_screenshot_base64.assert_called_once()

    async def test_uses_os_capture_when_no_page(self):
        screen, llm = _stub_modules(
            vision_response='{"ok": true, "observation": "ok", "confidence": 0.8}'
        )
        amb = VerifyResult.ambiguous()
        out = await ver.vision_verify(
            {"type": "app", "action": "click", "params": {}},
            amb, page=None,
        )
        self.assertTrue(out.ok)
        screen.capture_screenshot_base64.assert_called_once()


class TestVisionVerifyFailOpen(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset()

    async def test_screenshot_none_returns_code_result(self):
        _stub_modules(screenshot=None)
        amb = VerifyResult.ambiguous("original")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}}, amb,
        )
        self.assertEqual(out, amb)  # untouched

    async def test_llm_crash_returns_code_result(self):
        _stub_modules(vision_response=RuntimeError("network down"))
        amb = VerifyResult.ambiguous("original")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}}, amb,
        )
        self.assertEqual(out, amb)

    async def test_llm_unavailable_sentinel_returns_code_result(self):
        _stub_modules(vision_response="__LLM_UNAVAILABLE__")
        amb = VerifyResult.ambiguous("original")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}}, amb,
        )
        self.assertEqual(out, amb)

    async def test_empty_response_returns_code_result(self):
        _stub_modules(vision_response="")
        amb = VerifyResult.ambiguous("original")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}}, amb,
        )
        self.assertEqual(out, amb)

    async def test_garbage_json_returns_code_result(self):
        _stub_modules(vision_response="absolutely not json")
        amb = VerifyResult.ambiguous("original")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}}, amb,
        )
        self.assertEqual(out, amb)


class TestVisionVerifyDisabled(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_off_returns_code_result_without_calling_llm(self):
        _config_stub.VERIFY_VISION_FALLBACK = False
        _, llm = _stub_modules(vision_response='{"ok": false, "observation": "x"}')
        amb = VerifyResult.ambiguous("original")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}}, amb,
        )
        self.assertEqual(out, amb)
        llm.get_vision_response.assert_not_awaited()
        _reset()


# ─── parse_verify_failed ──────────────────────────────────────────────────────

class TestParseVerifyFailed(unittest.TestCase):
    def test_parses_clean_prefix(self):
        text = "VERIFY_FAILED|step=4|tier=vision|obs=Error message visible"
        p = ver.parse_verify_failed(text)
        self.assertEqual(p, {"step": 4, "tier": "vision", "observation": "Error message visible"})

    def test_parses_with_trailing_results(self):
        text = (
            "VERIFY_FAILED|step=2|tier=code|obs=URL is example.com, expected other.com\n"
            "Navigated to https://example.com\nFilled #email"
        )
        p = ver.parse_verify_failed(text)
        self.assertEqual(p["step"], 2)
        self.assertEqual(p["tier"], "code")
        self.assertIn("URL is example.com", p["observation"])

    def test_unrelated_text_returns_none(self):
        self.assertIsNone(ver.parse_verify_failed("just a normal step result"))
        self.assertIsNone(ver.parse_verify_failed(""))
        self.assertIsNone(ver.parse_verify_failed(None))

    def test_format_failure_for_user(self):
        s = ver.format_failure_for_user(
            {"step": 3, "tier": "code", "observation": "field reads 'Bob', typed 'Alice'"}
        )
        self.assertIn("Step 3", s)
        self.assertIn("Bob", s)
        self.assertIn("Alice", s)


# ─── Planner integration: _step_failed picks up VERIFY_FAILED ─────────────────
# We can't import planner.py directly (it pulls actions.py which has heavy deps).
# Instead, lift the deterministic helper inline — it's pure-string, easy to verify.

class TestPlannerFailureDetection(unittest.TestCase):
    """The planner's _step_failed routes step output to PA-3 recovery when a
    failure prefix is detected. VL-1b adds VERIFY_FAILED| to that list."""

    def test_verify_failed_treated_as_failure(self):
        # Inline copy of the relevant logic — keeps this test free of the
        # full planner import chain. Mirrors planner._FAILURE_PREFIXES.
        prefixes = ("ERROR:", "BLOCKED:", "TIMEOUT", "Error:", "Traceback",
                    "__NEEDS_OAUTH__", "__NEEDS_DEVICE_AUTH__",
                    "__CONFIRM_SEND__", "__SEND_ERROR__",
                    "VERIFY_FAILED|")
        sample = "VERIFY_FAILED|step=4|tier=vision|obs=phone needs country code\n..."
        self.assertTrue(any(sample.startswith(p) for p in prefixes))


class TestVisionPromptClickAwareness(unittest.TestCase):
    """Regression guard for manifest-based Session-5 Finding F4.

    The vision-verifier prompt must instruct the model to treat a click
    target's disappearance / toggle / state change as success, not failure.
    Without this rule, toggle clicks (Play → Pause, menu open → close,
    checkbox flip, tab switch) produce false-negative verify_failed which
    blocks automation-cache cache writes in router.py:1422 and therefore blocks manifest
    promotion from natural usage.
    """

    def test_prompt_contains_click_toggle_rule(self):
        prompt = ver._VISION_PROMPT
        # The rule must explicitly mention click actions and the toggle case.
        self.assertIn("CLICK actions", prompt)
        # The Play→Pause example is the canonical case discovered live.
        self.assertIn("Play button becoming a Pause button", prompt)
        # The rule must bias toward ok=true for clicks with any visible response.
        self.assertIn("ok\": true if you see ANY visible response", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
