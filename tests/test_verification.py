"""
test_verification.py — VL-1a: tiered step verification.

Covers:
  - VerifyResult constructors / tier semantics
  - Settings gate (verify_enabled, per-backend toggles, non-verifiable actions)
  - Text matching (loose default, strict mode, password heuristic)
  - URL normalization
  - Browser pre/post-check (Playwright mocked)
  - App pre/post-check (app_automation.get_text + pygetwindow mocked)
  - Dispatcher routing for unknown / read-only actions
  - Vision tier stub is a no-op pass-through (VL-1b will replace)

Run: python test_verification.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
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


def _reset_config():
    _config_stub.VERIFY_ENABLED = True
    _config_stub.VERIFY_BROWSER_STEPS = True
    _config_stub.VERIFY_APP_STEPS = True
    _config_stub.VERIFY_VISION_FALLBACK = True
    _config_stub.VERIFY_STRICT_TEXT_MATCH = False
    _config_stub.VERIFY_MIN_CONFIDENCE = 0.5


# ─── VerifyResult helpers ─────────────────────────────────────────────────────

class TestVerifyResult(unittest.TestCase):
    def test_ok_(self):
        r = VerifyResult.ok_()
        self.assertTrue(r.ok)
        self.assertEqual(r.tier, "code")
        self.assertEqual(r.confidence, 1.0)
        self.assertFalse(r.skipped)

    def test_fail(self):
        r = VerifyResult.fail("URL mismatch")
        self.assertFalse(r.ok)
        self.assertEqual(r.observation, "URL mismatch")
        self.assertEqual(r.tier, "code")

    def test_ambiguous_marks_for_escalation(self):
        r = VerifyResult.ambiguous("click outcome unknown")
        self.assertTrue(r.ok)  # ok=True so callers don't fail-stop
        self.assertEqual(r.tier, "ambiguous")
        self.assertEqual(r.confidence, 0.5)

    def test_skip(self):
        r = VerifyResult.skip("non-verifiable")
        self.assertTrue(r.skipped)
        self.assertTrue(r.ok)
        self.assertEqual(r.tier, "skipped")


# ─── Settings gate ────────────────────────────────────────────────────────────

class TestGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _reset_config()

    async def test_master_off_skips_everything(self):
        _config_stub.VERIFY_ENABLED = False
        r = await ver.post_verify({"type": "browser", "action": "navigate", "params": {"url": "x"}})
        self.assertTrue(r.skipped)

    async def test_browser_toggle(self):
        _config_stub.VERIFY_BROWSER_STEPS = False
        r = await ver.post_verify({"type": "browser", "action": "navigate", "params": {"url": "x"}})
        self.assertTrue(r.skipped)
        # App still runs
        r2 = await ver.post_verify({"type": "app", "action": "open", "params": {"name": "Notepad"}})
        self.assertFalse(r2.skipped or r2.ok and r2.tier == "skipped",
                         "app gate shouldn't skip when only browser is off")

    async def test_app_toggle(self):
        _config_stub.VERIFY_APP_STEPS = False
        r = await ver.post_verify({"type": "app", "action": "open", "params": {"name": "X"}})
        self.assertTrue(r.skipped)

    async def test_non_verifiable_actions_skip(self):
        for action in ("wait", "extract_text", "screenshot", "extract_selector"):
            r = await ver.post_verify({"type": "browser", "action": action, "params": {}})
            self.assertTrue(r.skipped, f"browser/{action} should skip")
        for action in ("wait", "get_text", "list"):
            r = await ver.post_verify({"type": "app", "action": action, "params": {}})
            self.assertTrue(r.skipped, f"app/{action} should skip")


# ─── Text matching ────────────────────────────────────────────────────────────

class TestTextMatch(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def test_loose_contains(self):
        self.assertTrue(ver._text_matches("john@x.com", "john@x.com"))
        self.assertTrue(ver._text_matches("john", "John Doe"))     # case-insensitive
        self.assertTrue(ver._text_matches("555", "555-1234"))      # phone autoformat
        self.assertTrue(ver._text_matches("john", "john@gmail.com"))  # autocomplete

    def test_loose_rejects_unrelated(self):
        self.assertFalse(ver._text_matches("john", "alice"))

    def test_strict_mode(self):
        _config_stub.VERIFY_STRICT_TEXT_MATCH = True
        self.assertTrue(ver._text_matches("john", "john"))
        self.assertFalse(ver._text_matches("john", "John"))
        self.assertFalse(ver._text_matches("john", "john@gmail.com"))

    def test_password_detection(self):
        self.assertTrue(ver._is_password_selector("input[type=password]"))
        self.assertTrue(ver._is_password_selector("name:Password"))
        self.assertTrue(ver._is_password_selector("#pwd"))
        self.assertFalse(ver._is_password_selector("input[type=text]"))
        self.assertFalse(ver._is_password_selector("name:Email"))


class TestUrlNormalize(unittest.TestCase):
    def test_strips_scheme_www_trailing_slash(self):
        self.assertEqual(ver._normalize_url("https://www.example.com/"), "example.com")
        self.assertEqual(ver._normalize_url("http://example.com"), "example.com")
        self.assertEqual(ver._normalize_url("example.com/path/"), "example.com/path")

    def test_empty_returns_empty(self):
        self.assertEqual(ver._normalize_url(""), "")
        self.assertEqual(ver._normalize_url(None), "")


# ─── Browser checkers (Playwright mocked) ─────────────────────────────────────

def _make_locator(*, visible=True, enabled=True, value=""):
    """Mock a Playwright Locator chain: page.locator(s).first.<methods>()"""
    first = MagicMock()
    first.is_visible = AsyncMock(return_value=visible)
    first.is_enabled = AsyncMock(return_value=enabled)
    first.is_hidden = AsyncMock(return_value=not visible)
    first.input_value = AsyncMock(return_value=value)
    first.bounding_box = AsyncMock(return_value={"x": 0, "y": 0, "width": 10, "height": 10})
    locator = MagicMock()
    locator.first = first
    # Newer pre/post fill paths log diagnostics: await loc.count() before the
    # visibility check, and await page.evaluate(...) on mismatch in post.
    locator.count = AsyncMock(return_value=1)
    return locator


def _make_page(url="https://example.com/", locators=None):
    page = MagicMock()
    page.url = url
    page.is_closed = MagicMock(return_value=False)
    page.evaluate = AsyncMock(return_value="complete")
    locators = locators or {}
    page.locator = MagicMock(side_effect=lambda sel: locators.get(sel, _make_locator()))
    return page


class TestBrowserPreCheck(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset_config()

    async def test_fill_pre_ok(self):
        page = _make_page(locators={"#email": _make_locator(visible=True, enabled=True)})
        r = await ver.pre_check(
            {"type": "browser", "action": "fill", "params": {"selector": "#email", "value": "x"}},
            page=page,
        )
        self.assertTrue(r.ok)
        self.assertEqual(r.tier, "pre_check")

    async def test_fill_pre_fails_when_invisible(self):
        page = _make_page(locators={"#email": _make_locator(visible=False)})
        r = await ver.pre_check(
            {"type": "browser", "action": "fill", "params": {"selector": "#email", "value": "x"}},
            page=page,
        )
        self.assertFalse(r.ok)
        self.assertIn("not visible", r.observation)

    async def test_fill_pre_fails_when_disabled(self):
        page = _make_page(locators={"#email": _make_locator(visible=True, enabled=False)})
        r = await ver.pre_check(
            {"type": "browser", "action": "fill", "params": {"selector": "#email", "value": "x"}},
            page=page,
        )
        self.assertFalse(r.ok)
        self.assertIn("disabled", r.observation)

    async def test_navigate_no_pre_check(self):
        r = await ver.pre_check(
            {"type": "browser", "action": "navigate", "params": {"url": "https://x"}},
            page=_make_page(),
        )
        self.assertTrue(r.skipped)


class TestBrowserPostVerify(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset_config()

    async def test_navigate_ok(self):
        page = _make_page(url="https://www.example.com/path")
        r = await ver.post_verify(
            {"type": "browser", "action": "navigate", "params": {"url": "example.com"}},
            page=page,
        )
        self.assertTrue(r.ok)

    async def test_navigate_url_mismatch(self):
        page = _make_page(url="https://other.com/")
        r = await ver.post_verify(
            {"type": "browser", "action": "navigate", "params": {"url": "https://example.com"}},
            page=page,
        )
        self.assertFalse(r.ok)
        self.assertIn("other.com", r.observation)

    async def test_navigate_same_host_redirect_is_ambiguous(self):
        """URL slug redirect within same host → ambiguous, not fail."""
        page = _make_page(url="https://in.bookmyshow.com/movies/berlin/spider-man-brand-new-day/ET00447840")
        r = await ver.post_verify(
            {"type": "browser", "action": "navigate",
             "params": {"url": "https://in.bookmyshow.com/movies/berlin/spiderman-brand-new-day/ET00447840"}},
            page=page,
        )
        self.assertTrue(r.ok)
        self.assertEqual(r.tier, "ambiguous")
        self.assertIn("same host", r.observation)

    async def test_navigate_different_host_still_fails(self):
        """Different host redirect → still a hard fail."""
        page = _make_page(url="https://help.example.com/article/123")
        r = await ver.post_verify(
            {"type": "browser", "action": "navigate",
             "params": {"url": "https://www.example.com/page"}},
            page=page,
        )
        self.assertFalse(r.ok)
        self.assertIn("help.example.com", r.observation)

    async def test_fill_ok_loose(self):
        page = _make_page(locators={"#e": _make_locator(value="john@gmail.com")})
        r = await ver.post_verify(
            {"type": "browser", "action": "fill", "params": {"selector": "#e", "value": "john"}},
            page=page,
        )
        self.assertTrue(r.ok, f"loose match should pass autocomplete; got {r.observation}")

    async def test_fill_fail_strict(self):
        _config_stub.VERIFY_STRICT_TEXT_MATCH = True
        page = _make_page(locators={"#e": _make_locator(value="john@gmail.com")})
        r = await ver.post_verify(
            {"type": "browser", "action": "fill", "params": {"selector": "#e", "value": "john"}},
            page=page,
        )
        self.assertFalse(r.ok)

    async def test_fill_password_non_empty_ok(self):
        page = _make_page(locators={"#pwd": _make_locator(value="********")})
        r = await ver.post_verify(
            {"type": "browser", "action": "fill", "params": {"selector": "#pwd", "value": "secret"}},
            page=page,
        )
        self.assertTrue(r.ok)

    async def test_fill_password_empty_fails(self):
        page = _make_page(locators={"#pwd": _make_locator(value="")})
        r = await ver.post_verify(
            {"type": "browser", "action": "fill", "params": {"selector": "#pwd", "value": "secret"}},
            page=page,
        )
        self.assertFalse(r.ok)

    async def test_click_is_ambiguous(self):
        r = await ver.post_verify(
            {"type": "browser", "action": "click", "params": {"selector": "button"}},
            page=_make_page(),
        )
        self.assertEqual(r.tier, "ambiguous")
        self.assertTrue(r.ok)  # ambiguous == pass at code tier; vision can override

    async def test_press_is_ambiguous(self):
        r = await ver.post_verify(
            {"type": "browser", "action": "press", "params": {"key": "Enter"}},
            page=_make_page(),
        )
        self.assertEqual(r.tier, "ambiguous")


# ─── App checkers (app_automation.get_text + pygetwindow mocked) ──────────────

class _FakeWindow:
    def __init__(self, title): self.title = title

class TestAppPostVerify(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset_config()

    async def test_open_window_appears(self):
        with patch("pygetwindow.getAllWindows", return_value=[_FakeWindow("Notepad - Untitled")]):
            r = await ver.post_verify({"type": "app", "action": "open", "params": {"name": "Notepad"}})
        self.assertTrue(r.ok)

    async def test_open_window_missing_fails(self):
        with patch("pygetwindow.getAllWindows", return_value=[_FakeWindow("Calculator")]):
            r = await ver.post_verify({"type": "app", "action": "open", "params": {"name": "Notepad"}})
        self.assertFalse(r.ok)
        self.assertIn("Notepad", r.observation)

    async def test_close_window_gone(self):
        with patch("pygetwindow.getAllWindows", return_value=[_FakeWindow("Calculator")]):
            r = await ver.post_verify({"type": "app", "action": "close", "params": {"name": "Notepad"}})
        self.assertTrue(r.ok)

    async def test_close_window_still_present_fails(self):
        with patch("pygetwindow.getAllWindows", return_value=[_FakeWindow("Notepad - Untitled")]):
            r = await ver.post_verify({"type": "app", "action": "close", "params": {"name": "Notepad"}})
        self.assertFalse(r.ok)

    async def test_focus_active_matches(self):
        with patch("pygetwindow.getActiveWindow", return_value=_FakeWindow("Notepad - Untitled")):
            r = await ver.post_verify({"type": "app", "action": "focus", "params": {"name": "Notepad"}})
        self.assertTrue(r.ok)

    async def test_focus_active_mismatches(self):
        with patch("pygetwindow.getActiveWindow", return_value=_FakeWindow("Calculator")):
            r = await ver.post_verify({"type": "app", "action": "focus", "params": {"name": "Notepad"}})
        self.assertFalse(r.ok)

    async def test_type_readback_ok(self):
        fake_aa = MagicMock()
        fake_aa.get_text = AsyncMock(return_value="hello world")
        with patch.object(ver, "_get_app_automation", return_value=fake_aa):
            r = await ver.post_verify({
                "type": "app", "action": "type",
                "params": {"text": "hello", "selector": "name:Edit", "window": "Notepad"},
            })
        self.assertTrue(r.ok)

    async def test_type_readback_mismatch(self):
        fake_aa = MagicMock()
        fake_aa.get_text = AsyncMock(return_value="goodbye")
        with patch.object(ver, "_get_app_automation", return_value=fake_aa):
            r = await ver.post_verify({
                "type": "app", "action": "type",
                "params": {"text": "hello", "selector": "name:Edit", "window": "Notepad"},
            })
        self.assertFalse(r.ok)
        self.assertIn("hello", r.observation)
        self.assertIn("goodbye", r.observation)

    async def test_type_at_focus_is_ambiguous(self):
        # No selector → can't read back deterministically → vision tier should answer
        r = await ver.post_verify({
            "type": "app", "action": "type",
            "params": {"text": "hello"},
        })
        self.assertEqual(r.tier, "ambiguous")

    async def test_type_password_is_ambiguous(self):
        r = await ver.post_verify({
            "type": "app", "action": "type",
            "params": {"text": "secret", "selector": "name:Password", "window": "Login"},
        })
        self.assertEqual(r.tier, "ambiguous")

    async def test_app_click_is_ambiguous(self):
        r = await ver.post_verify({
            "type": "app", "action": "click",
            "params": {"selector": "name:Submit", "window": "Form"},
        })
        self.assertEqual(r.tier, "ambiguous")


class TestAppPreCheck(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset_config()

    async def test_type_pre_focus_matches_window(self):
        # Focus is on Notepad and step targets Notepad → pre-check passes
        with patch("pygetwindow.getActiveWindow", return_value=_FakeWindow("Notepad - Untitled")):
            r = await ver.pre_check({
                "type": "app", "action": "type",
                "params": {"text": "x", "selector": "name:Edit", "window": "Notepad"},
            })
        self.assertTrue(r.ok)
        self.assertEqual(r.tier, "pre_check")

    async def test_type_pre_focus_drift_caught(self):
        # Step targets Notepad but VS Code is foreground → pre-check fails
        # (catches the bug class observed in the calculator click incident)
        with patch("pygetwindow.getActiveWindow", return_value=_FakeWindow("VS Code")):
            r = await ver.pre_check({
                "type": "app", "action": "type",
                "params": {"text": "x", "selector": "name:Edit", "window": "Notepad"},
            })
        self.assertFalse(r.ok)
        self.assertIn("focus drift", r.observation)

    async def test_type_pre_at_focus_ambiguous(self):
        r = await ver.pre_check({
            "type": "app", "action": "type",
            "params": {"text": "x"},  # no selector
        })
        self.assertEqual(r.tier, "ambiguous")

    async def test_type_pre_no_window_skips(self):
        # No window param and no active_window context → nothing cheap to check
        r = await ver.pre_check({
            "type": "app", "action": "type",
            "params": {"text": "x", "selector": "name:Edit"},
        })
        self.assertTrue(r.skipped)


# ─── Vision tier stub ─────────────────────────────────────────────────────────

class TestVisionStub(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset_config()

    async def test_vision_returns_result(self):
        # Stub screen/llm so no real API calls occur.
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value=None)
        sys.modules["assistant.io.screen"] = screen_mod

        amb = VerifyResult.ambiguous("click")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}},
            amb, page=_make_page(),
        )
        # screenshot=None → fail-open, returns code_result unchanged.
        self.assertEqual(out, amb)

    async def test_vision_fallback_off_returns_code_result(self):
        _config_stub.VERIFY_VISION_FALLBACK = False
        amb = VerifyResult.ambiguous("click")
        out = await ver.vision_verify(
            {"type": "browser", "action": "click", "params": {}},
            amb, page=_make_page(),
        )
        self.assertEqual(out, amb)


# ─── Dispatcher edge cases ────────────────────────────────────────────────────

class TestDispatcher(unittest.IsolatedAsyncioTestCase):
    def setUp(self): _reset_config()

    async def test_unknown_browser_action_ambiguous(self):
        r = await ver.post_verify({"type": "browser", "action": "frobnicate", "params": {}})
        self.assertEqual(r.tier, "ambiguous")

    async def test_unknown_app_action_ambiguous(self):
        r = await ver.post_verify({"type": "app", "action": "frobnicate", "params": {}})
        self.assertEqual(r.tier, "ambiguous")

    async def test_pre_check_unknown_action_skipped(self):
        # No pre-check defined → skipped (don't block execution)
        r = await ver.pre_check({"type": "browser", "action": "navigate", "params": {"url": "x"}})
        self.assertTrue(r.skipped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
