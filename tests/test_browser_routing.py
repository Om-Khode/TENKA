"""
test_browser_routing.py — Phase 1D: routing decision (`_choose_browser_mode`).

Pure decision-table tests. The function takes (goal, cdp_state) and returns
(mode, reason_meta). No I/O, no LLM, no side effects — straightforward
unit tests.

Covers all 7 priority branches of the decision tree:
  1. Master kill-switch (BROWSER_DOM_MODE_ENABLED=False) → bundled
  2. Canvas/WebGL keyword → vision (always, regardless of CDP)
  3. CDP unavailable → bundled
  4. User preference override → that mode
  5. Form-intent keyword → DOM
  6. Extraction-intent keyword → bundled
  7. Default with CDP up → DOM

Plus integration with detect_backend's `browser_content` branch.

Run: python test_browser_routing.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.router as da
import assistant.config as cfg


class _FakeCdpState:
    """Minimal stand-in for browser_cdp.CdpProbeResult."""
    def __init__(self, available: bool):
        self.available = available


# ─── _choose_browser_mode: each priority branch ──────────────────────────


class TestChooseBrowserMode(unittest.TestCase):
    def setUp(self):
        # Restore default flag at start of each test so test order doesn't matter
        cfg.BROWSER_DOM_MODE_ENABLED = True

    def tearDown(self):
        cfg.BROWSER_DOM_MODE_ENABLED = True

    # ── Priority 1: kill-switch ──
    def test_kill_switch_off_returns_bundled(self):
        cfg.BROWSER_DOM_MODE_ENABLED = False
        mode, meta = da._choose_browser_mode("fill the form", _FakeCdpState(True))
        self.assertEqual(mode, "playwright_bundled")
        self.assertEqual(meta["reason"], "dom_mode_flag_off")

    def test_kill_switch_off_canvas_app_still_bundled(self):
        # Kill-switch wins over canvas — both want non-DOM, but the user
        # explicitly disabled DOM so we honor that intent literally.
        cfg.BROWSER_DOM_MODE_ENABLED = False
        mode, _ = da._choose_browser_mode("draw in figma", _FakeCdpState(True))
        self.assertEqual(mode, "playwright_bundled")

    # ── Priority 2: canvas / WebGL keywords ──
    def test_canvas_figma_routes_to_vision(self):
        mode, meta = da._choose_browser_mode("draw a square in figma", _FakeCdpState(True))
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "canvas_intent")

    def test_canvas_miro_routes_to_vision(self):
        mode, _ = da._choose_browser_mode("add a sticky note on miro", _FakeCdpState(True))
        self.assertEqual(mode, "vision")

    def test_canvas_google_slides_routes_to_vision(self):
        mode, _ = da._choose_browser_mode(
            "edit the deck on google slides", _FakeCdpState(True),
        )
        self.assertEqual(mode, "vision")

    def test_canvas_overrides_form_intent(self):
        # "fill" is a form-intent keyword, but figma is canvas.
        # Canvas wins — DOM-mode would fail on canvas pages.
        mode, _ = da._choose_browser_mode(
            "fill in details on the figma board", _FakeCdpState(True),
        )
        self.assertEqual(mode, "vision")

    # ── Priority 3: CDP unavailable ──
    def test_cdp_unavailable_form_intent_bundled(self):
        mode, meta = da._choose_browser_mode("fill the form", _FakeCdpState(False))
        self.assertEqual(mode, "playwright_bundled")
        self.assertEqual(meta["reason"], "cdp_unavailable")

    def test_cdp_state_none_treated_as_unavailable(self):
        # When the cache hasn't probed yet, cdp_state may be None.
        # Treat as unavailable.
        mode, _ = da._choose_browser_mode("fill the form", None)
        self.assertEqual(mode, "playwright_bundled")

    def test_cdp_unavailable_canvas_still_vision(self):
        # Canvas check fires before CDP check — vision regardless of CDP.
        mode, meta = da._choose_browser_mode("draw in figma", _FakeCdpState(False))
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "canvas_intent")

    # ── Priority 4: user preference ──
    def test_user_preference_dom_wins(self):
        mode, meta = da._choose_browser_mode(
            "summarize this page", _FakeCdpState(True), user_preference="dom",
        )
        # Preference overrides extraction-intent's bundled default
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "user_preference")

    def test_user_preference_vision_wins(self):
        mode, meta = da._choose_browser_mode(
            "fill the form", _FakeCdpState(True), user_preference="vision",
        )
        # Preference overrides form-intent's DOM default
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "user_preference")

    def test_user_preference_invalid_value_ignored(self):
        # Unknown preference values fall through to heuristic
        mode, meta = da._choose_browser_mode(
            "fill the form", _FakeCdpState(True), user_preference="garbage",
        )
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "form_intent")

    def test_user_preference_does_not_override_canvas_or_kill_switch(self):
        # Canvas wins over preference — preference can't make us run DOM
        # mode against a Figma canvas (would fail anyway)
        mode, _ = da._choose_browser_mode(
            "draw in figma", _FakeCdpState(True), user_preference="dom",
        )
        self.assertEqual(mode, "vision")

    # ── Priority 5: form-intent keywords ──
    def test_form_intent_fill_routes_to_dom(self):
        mode, meta = da._choose_browser_mode("fill the form", _FakeCdpState(True))
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "form_intent")

    def test_form_intent_login_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("log in to truein", _FakeCdpState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_signup_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("sign up for a new account", _FakeCdpState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_signin_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("sign in with my credentials", _FakeCdpState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_book_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("book a demo", _FakeCdpState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_register_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("register for the event", _FakeCdpState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_complete_form_routes_to_dom(self):
        mode, _ = da._choose_browser_mode(
            "complete the demo form", _FakeCdpState(True),
        )
        self.assertEqual(mode, "dom")

    # ── Priority 6: extraction-intent keywords ──
    def test_extraction_summarize_routes_to_bundled(self):
        mode, meta = da._choose_browser_mode("summarize this page", _FakeCdpState(True))
        self.assertEqual(mode, "playwright_bundled")
        self.assertEqual(meta["reason"], "extraction_intent")

    def test_extraction_read_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode("read the article", _FakeCdpState(True))
        self.assertEqual(mode, "playwright_bundled")

    def test_extraction_what_does_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode(
            "what does this page say about pricing", _FakeCdpState(True),
        )
        self.assertEqual(mode, "playwright_bundled")

    def test_extraction_tell_me_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode(
            "tell me the headline of this article", _FakeCdpState(True),
        )
        self.assertEqual(mode, "playwright_bundled")

    # ── Priority 7: default ──
    def test_default_with_cdp_up_routes_to_dom(self):
        # Goal doesn't match any specific keyword — default to DOM
        mode, meta = da._choose_browser_mode(
            "do something on the page", _FakeCdpState(True),
        )
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "cdp_default")

    def test_default_no_cdp_routes_to_bundled(self):
        # Same goal but CDP down — bundled
        mode, _ = da._choose_browser_mode(
            "do something on the page", _FakeCdpState(False),
        )
        self.assertEqual(mode, "playwright_bundled")

    # ── Empty / edge ──
    def test_empty_goal_with_cdp_up_routes_to_dom(self):
        mode, meta = da._choose_browser_mode("", _FakeCdpState(True))
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "cdp_default")

    def test_empty_goal_cdp_down_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode("", _FakeCdpState(False))
        self.assertEqual(mode, "playwright_bundled")


# ─── _route_browser_content integration ─────────────────────────────────


class TestRouteBrowserContent(unittest.TestCase):
    """
    `_route_browser_content` bridges _choose_browser_mode's return into
    detect_backend's vocabulary. Tests focus on:
      - cdp_state read from browser_cdp module
      - user_preference read from preferences
      - meta tagged with running_window
      - error swallowing on import/lookup failures
    """

    def test_cdp_unavailable_returns_vision_not_playwright_bundled(self):
        # Phase 1E hotfix: in browser-content scenarios (user has their
        # own browser open at the page), "playwright_bundled" doesn't
        # make sense. _route_browser_content translates it to "vision"
        # and tags meta with translated_from for telemetry.
        with patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(False)), \
             patch("assistant.preferences.get_preference", return_value=None):
            mode, meta = da._route_browser_content(
                "fill form", "Firefox - Truein"
            )
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["app"], "Firefox - Truein")
        self.assertEqual(meta["reason"], "cdp_unavailable")
        self.assertEqual(meta["translated_from"], "playwright_bundled")

    def test_cdp_available_form_intent_returns_dom(self):
        with patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            mode, meta = da._route_browser_content(
                "fill the form", "Chrome",
            )
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["app"], "Chrome")

    def test_user_preference_propagated(self):
        # preferences.get_preference returns a dict like
        # {"key": ..., "value": ..., "confidence": ...}
        with patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference",
                   return_value={"key": "automation_browser_mode", "value": "vision"}):
            mode, meta = da._route_browser_content("fill the form", "Chrome")
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "user_preference")

    def test_browser_cdp_import_failure_falls_back_safely(self):
        # If browser_cdp can't import (unlikely but defensive), the bridge
        # should not raise — treat as cdp unavailable. After Phase 1E
        # hotfix, "playwright_bundled" gets translated to "vision" in the
        # browser-content scenario.
        with patch.dict("sys.modules", {"assistant.automation.browser.cdp": None}):
            mode, meta = da._route_browser_content("fill form", "Chrome")
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "cdp_unavailable")
        self.assertEqual(meta["translated_from"], "playwright_bundled")
        self.assertEqual(meta["app"], "Chrome")


# ─── detect_backend Phase 1D fallback: form-intent + browser open ────────


class TestDetectBackendFallback(unittest.TestCase):
    """
    The strict _BROWSER_INTENT_PATTERNS regex misses phrasings like
    "fill this form" because of its rigid (the\\s+)? clause. The Phase 1D
    fallback in detect_backend catches these by checking: any open
    window is a browser AND goal matches _FORM_INTENT_RE → delegate to
    _route_browser_content.

    Regression guard: this is exactly the bug that made "fill this form
    with testing values" fall through to vision-loop in the live test.
    """

    def test_fill_this_form_with_chrome_open_routes_to_browser_content(self):
        # Mock screen.get_open_windows to include a Chrome window
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["Truein - Google Chrome", "Notepad"]), \
             patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend("fill this form with testing values")
        # CDP up + form-intent → DOM-mode
        self.assertEqual(backend, "dom")
        self.assertEqual(meta["reason"], "form_intent")
        self.assertEqual(meta["app"], "Truein - Google Chrome")

    def test_fill_form_falls_to_vision_when_cdp_down(self):
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["Truein - Google Chrome"]), \
             patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(False)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend("fill this form with testing values")
        # CDP down → playwright_bundled → translated to vision
        self.assertEqual(backend, "vision")
        self.assertEqual(meta["reason"], "cdp_unavailable")

    def test_fallback_inactive_when_no_browser_window(self):
        # No browser open — fallback shouldn't fire
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["Notepad", "VS Code"]):
            backend, meta = da.detect_backend("fill this form with testing values")
        self.assertEqual(backend, "unknown")

    def test_open_chrome_and_fill_form_routes_to_browser_content(self):
        """Regression guard for the live failure: goal containing both
        'open chrome' (which used to disable browser-content routing via
        run_app_match) AND a form-intent verb should still route to
        browser-content because the form-fill is the actual task."""
        with patch("assistant.automation.router._detect_running_app",
                   return_value="Truein - Google Chrome"), \
             patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend(
                "open chrome and fill that form with testing values"
            )
        # form-intent overrides the run_app_match gating
        self.assertEqual(backend, "dom")
        self.assertEqual(meta["reason"], "form_intent")

    def test_open_chrome_alone_still_routes_to_native(self):
        """Counter-test: bare 'open chrome' (no form-fill verb) must still
        route native, otherwise we'd break the basic 'open browser' command."""
        with patch("assistant.automation.router._detect_running_app",
                   return_value="Google Chrome"):
            backend, meta = da.detect_backend("open chrome")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "running_app_detected")

    def test_fallback_inactive_for_non_form_goal(self):
        # Browser is open but goal isn't form-shape — fallback skipped
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["Truein - Google Chrome"]):
            backend, meta = da.detect_backend("what time is it")
        # Falls through to "unknown" (no other heuristic matches)
        self.assertEqual(backend, "unknown")

    def test_screen_import_failure_does_not_raise(self):
        # Defensive: if screen.get_open_windows raises, fallback silently
        # gives up and detect_backend returns "unknown".
        with patch("assistant.io.screen.get_open_windows",
                   side_effect=RuntimeError("no display")):
            backend, _ = da.detect_backend("fill this form")
        self.assertEqual(backend, "unknown")


# ─── can_handle: dom backend now accepted ────────────────────────────────


class TestCanHandleDom(unittest.IsolatedAsyncioTestCase):
    async def test_dom_backend_returns_handleable(self):
        with patch.object(da, "detect_backend",
                          return_value=("dom", {"reason": "form_intent"})):
            ok, backend = await da.can_handle("fill the form")
        self.assertTrue(ok)
        self.assertEqual(backend, "dom")

    async def test_dom_backend_falls_back_when_playwright_missing(self):
        # Defensive: deployment without Playwright installed
        with patch.object(da, "detect_backend",
                          return_value=("dom", {"reason": "form_intent"})), \
             patch("assistant.automation.browser.automation.PLAYWRIGHT_AVAILABLE", False):
            ok, backend = await da.can_handle("fill the form")
        self.assertFalse(ok)
        self.assertEqual(backend, "vision")


# ─── Bug A: app_context_pattern skipped when form-intent present ─────────


class TestAppContextPatternFormGuard(unittest.TestCase):
    """Bug A fix: 'Fill the subjects field with Maths' was routing to
    native with app='maths' because the app_context_pattern regex extracted
    the last word after 'with'. When a form-intent verb is present, the
    tail word is data (not an app) — skip app_context_pattern."""

    def test_fill_with_maths_not_treated_as_app(self):
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["demosite - Google Chrome"]), \
             patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend("Fill the subjects field with Maths")
        self.assertNotEqual(meta.get("reason"), "app_context_pattern")

    def test_play_music_on_spotify_still_routes_native(self):
        backend, meta = da.detect_backend("play music on spotify")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "app_context_pattern")
        self.assertEqual(meta["app"], "spotify")

    def test_fill_form_with_value_routes_dom(self):
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["App - Google Chrome"]), \
             patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend("fill this form with testing values")
        self.assertEqual(backend, "dom")
        self.assertEqual(meta["reason"], "form_intent")


# ─── Bug C: _FORM_INTENT_RE expanded keywords ───────────────────────────


class TestFormIntentExpandedKeywords(unittest.TestCase):
    """Bug C fix: 'Set State to NCR in this form' skipped DOM mode because
    'set' was missing from _FORM_INTENT_RE. Added set/choose/pick."""

    def test_set_matches_form_intent(self):
        self.assertIsNotNone(da._FORM_INTENT_RE.search("Set State to NCR in this form"))

    def test_choose_matches_form_intent(self):
        self.assertIsNotNone(da._FORM_INTENT_RE.search("choose the country in this form"))

    def test_pick_matches_form_intent(self):
        self.assertIsNotNone(da._FORM_INTENT_RE.search("pick a date"))

    def test_set_routes_to_dom_with_browser_open(self):
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["demosite - Google Chrome"]), \
             patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend("Set State to NCR in this form")
        self.assertEqual(backend, "dom")
        self.assertEqual(meta["reason"], "form_intent")


def _run(coro):
    """Run a coroutine in a new event loop and return its result.

    Uses `asyncio.new_event_loop()` directly — `get_event_loop()` is
    deprecated since 3.10 and raises RuntimeError when no loop is set
    in the current thread (which is the case under pytest's default
    test isolation).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── _execute_dom_task: form-intent routing to run_dom_form_fill ─────────


class TestDomFormFillRouting(unittest.TestCase):
    """Verify _execute_dom_task calls run_dom_form_fill for form-intent goals."""

    # _execute_dom_task imports browser_cdp / browser_dom_orchestrator
    # LOCALLY (lazy import at line ~1565 of router.py), so patches must
    # target the source modules, not `router.<name>` which never exists.
    @patch("assistant.automation.browser.dom_orchestrator.run_dom_form_fill", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.run_dom_task", new_callable=AsyncMock)
    @patch("assistant.automation.router._pick_active_page", new_callable=AsyncMock)
    @patch("assistant.automation.browser.cdp.get_or_attach_browser", new_callable=AsyncMock)
    def test_form_intent_uses_form_fill(self, mock_cdp, mock_pick, mock_old, mock_new):
        mock_cdp.return_value = MagicMock(kind="cdp", attachment=MagicMock())
        mock_pick.return_value = MagicMock()
        mock_new.return_value = MagicMock(
            success=True, final_summary="Form submitted.", reason="completed",
        )
        _run(da._execute_dom_task("Fill the registration form with test data"))
        mock_new.assert_called_once()
        mock_old.assert_not_called()

    @patch("assistant.automation.browser.dom_orchestrator.run_dom_form_fill", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.run_dom_task", new_callable=AsyncMock)
    @patch("assistant.automation.router._pick_active_page", new_callable=AsyncMock)
    @patch("assistant.automation.browser.cdp.get_or_attach_browser", new_callable=AsyncMock)
    def test_non_form_uses_old_loop(self, mock_cdp, mock_pick, mock_old, mock_new):
        mock_cdp.return_value = MagicMock(kind="cdp", attachment=MagicMock())
        mock_pick.return_value = MagicMock()
        mock_old.return_value = MagicMock(
            success=True, final_summary="Done.", reason="completed",
        )
        _run(da._execute_dom_task("Click the search button"))
        mock_old.assert_called_once()
        mock_new.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
