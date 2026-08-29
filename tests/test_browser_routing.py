"""
test_browser_routing.py — Phase 1D: routing decision (`_choose_browser_mode`).

Pure decision-table tests. The function takes (goal, driver_state) and returns
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


class _FakeDriverState:
    """Minimal stand-in for `extension_ws.DroverState`.

    The field is `connected`, not `available`: an extension either has a live
    socket or it has not, and there is no probe whose cached answer could be
    stale. Renamed with the mechanism rather than aliased, so a call site still
    reading `.available` fails loudly instead of quietly reading False and
    routing every browser task to the bundled browser.
    """

    def __init__(self, connected: bool):
        self.connected = connected


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
        mode, meta = da._choose_browser_mode("fill the form", _FakeDriverState(True))
        self.assertEqual(mode, "playwright_bundled")
        self.assertEqual(meta["reason"], "dom_mode_flag_off")

    def test_kill_switch_off_canvas_app_still_bundled(self):
        # Kill-switch wins over canvas — both want non-DOM, but the user
        # explicitly disabled DOM so we honor that intent literally.
        cfg.BROWSER_DOM_MODE_ENABLED = False
        mode, _ = da._choose_browser_mode("draw in figma", _FakeDriverState(True))
        self.assertEqual(mode, "playwright_bundled")

    # ── Priority 2: canvas / WebGL keywords ──
    def test_canvas_figma_routes_to_vision(self):
        mode, meta = da._choose_browser_mode("draw a square in figma", _FakeDriverState(True))
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "canvas_intent")

    def test_canvas_miro_routes_to_vision(self):
        mode, _ = da._choose_browser_mode("add a sticky note on miro", _FakeDriverState(True))
        self.assertEqual(mode, "vision")

    def test_canvas_google_slides_routes_to_vision(self):
        mode, _ = da._choose_browser_mode(
            "edit the deck on google slides", _FakeDriverState(True),
        )
        self.assertEqual(mode, "vision")

    def test_canvas_overrides_form_intent(self):
        # "fill" is a form-intent keyword, but figma is canvas.
        # Canvas wins — DOM-mode would fail on canvas pages.
        mode, _ = da._choose_browser_mode(
            "fill in details on the figma board", _FakeDriverState(True),
        )
        self.assertEqual(mode, "vision")

    # ── Priority 3: CDP unavailable ──
    def test_extension_unavailable_form_intent_bundled(self):
        mode, meta = da._choose_browser_mode("fill the form", _FakeDriverState(False))
        self.assertEqual(mode, "playwright_bundled")
        self.assertEqual(meta["reason"], "extension_unavailable")

    def test_a_none_driver_state_is_treated_as_unavailable(self):
        # Nothing has looked yet, so `driver_state` may be None. None refuses.
        # Treat as unavailable.
        mode, _ = da._choose_browser_mode("fill the form", None)
        self.assertEqual(mode, "playwright_bundled")

    def test_extension_unavailable_canvas_still_vision(self):
        # Canvas check fires before CDP check — vision regardless of CDP.
        mode, meta = da._choose_browser_mode("draw in figma", _FakeDriverState(False))
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "canvas_intent")

    # ── Priority 4: user preference ──
    def test_user_preference_dom_wins(self):
        mode, meta = da._choose_browser_mode(
            "summarize this page", _FakeDriverState(True), user_preference="dom",
        )
        # Preference overrides extraction-intent's bundled default
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "user_preference")

    def test_user_preference_vision_wins(self):
        mode, meta = da._choose_browser_mode(
            "fill the form", _FakeDriverState(True), user_preference="vision",
        )
        # Preference overrides form-intent's DOM default
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "user_preference")

    def test_user_preference_invalid_value_ignored(self):
        # Unknown preference values fall through to heuristic
        mode, meta = da._choose_browser_mode(
            "fill the form", _FakeDriverState(True), user_preference="garbage",
        )
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "form_intent")

    def test_user_preference_does_not_override_canvas_or_kill_switch(self):
        # Canvas wins over preference — preference can't make us run DOM
        # mode against a Figma canvas (would fail anyway)
        mode, _ = da._choose_browser_mode(
            "draw in figma", _FakeDriverState(True), user_preference="dom",
        )
        self.assertEqual(mode, "vision")

    # ── Priority 5: form-intent keywords ──
    def test_form_intent_fill_routes_to_dom(self):
        mode, meta = da._choose_browser_mode("fill the form", _FakeDriverState(True))
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "form_intent")

    def test_form_intent_login_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("log in to truein", _FakeDriverState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_signup_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("sign up for a new account", _FakeDriverState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_signin_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("sign in with my credentials", _FakeDriverState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_book_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("book a demo", _FakeDriverState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_register_routes_to_dom(self):
        mode, _ = da._choose_browser_mode("register for the event", _FakeDriverState(True))
        self.assertEqual(mode, "dom")

    def test_form_intent_complete_form_routes_to_dom(self):
        mode, _ = da._choose_browser_mode(
            "complete the demo form", _FakeDriverState(True),
        )
        self.assertEqual(mode, "dom")

    # ── Priority 6: extraction-intent keywords ──
    def test_extraction_summarize_routes_to_bundled(self):
        mode, meta = da._choose_browser_mode("summarize this page", _FakeDriverState(True))
        self.assertEqual(mode, "playwright_bundled")
        self.assertEqual(meta["reason"], "extraction_intent")

    def test_extraction_read_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode("read the article", _FakeDriverState(True))
        self.assertEqual(mode, "playwright_bundled")

    def test_extraction_what_does_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode(
            "what does this page say about pricing", _FakeDriverState(True),
        )
        self.assertEqual(mode, "playwright_bundled")

    def test_extraction_tell_me_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode(
            "tell me the headline of this article", _FakeDriverState(True),
        )
        self.assertEqual(mode, "playwright_bundled")

    # ── Priority 7: default ──
    def test_default_with_a_browser_connected_routes_to_dom(self):
        # Goal doesn't match any specific keyword — default to DOM
        mode, meta = da._choose_browser_mode(
            "do something on the page", _FakeDriverState(True),
        )
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "extension_default")

    def test_default_with_nothing_connected_routes_to_bundled(self):
        # Same goal but CDP down — bundled
        mode, _ = da._choose_browser_mode(
            "do something on the page", _FakeDriverState(False),
        )
        self.assertEqual(mode, "playwright_bundled")

    # ── Empty / edge ──
    def test_empty_goal_with_a_browser_connected_routes_to_dom(self):
        mode, meta = da._choose_browser_mode("", _FakeDriverState(True))
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["reason"], "extension_default")

    def test_empty_goal_with_nothing_connected_routes_to_bundled(self):
        mode, _ = da._choose_browser_mode("", _FakeDriverState(False))
        self.assertEqual(mode, "playwright_bundled")


# ─── _route_browser_content integration ─────────────────────────────────


class TestRouteBrowserContent(unittest.TestCase):
    """
    `_route_browser_content` bridges _choose_browser_mode's return into
    detect_backend's vocabulary. Tests focus on:
      - driver_state read from the extension module
      - user_preference read from preferences
      - meta tagged with running_window
      - error swallowing on import/lookup failures
    """

    def test_extension_unavailable_returns_vision_not_playwright_bundled(self):
        # Phase 1E hotfix: in browser-content scenarios (user has their
        # own browser open at the page), "playwright_bundled" doesn't
        # make sense. _route_browser_content translates it to "vision"
        # and tags meta with translated_from for telemetry.
        with patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(False)), \
             patch("assistant.preferences.get_preference", return_value=None):
            mode, meta = da._route_browser_content(
                "fill form", "Firefox - Truein"
            )
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["app"], "Firefox - Truein")
        self.assertEqual(meta["reason"], "extension_unavailable")
        self.assertEqual(meta["translated_from"], "playwright_bundled")

    def test_a_connected_extension_with_a_form_intent_returns_dom(self):
        with patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            mode, meta = da._route_browser_content(
                "fill the form", "Chrome",
            )
        self.assertEqual(mode, "dom")
        self.assertEqual(meta["app"], "Chrome")

    def test_user_preference_propagated(self):
        # `get_preference` does `SELECT *`, so a real row carries every
        # column including `source` -- which `set_preference` requires.
        # This fixture used to omit it, and the omission stopped being
        # harmless the moment the consumer began consulting provenance
        # (D2, §10.3): a row with no source is refused, correctly, and the
        # test then failed for a reason unrelated to what it checks.
        with patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
             patch("assistant.preferences.get_preference",
                   return_value={"key": "automation_browser_mode",
                                 "value": "vision", "source": "correction",
                                 "confidence": 0.85}):
            mode, meta = da._route_browser_content("fill the form", "Chrome")
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "user_preference")

    def test_a_guessed_preference_does_not_override_browser_routing(self):
        """The other half, and the reason the check exists.

        `_choose_browser_mode` treats `user_preference` as an override:
        set, it returns that mode and the heuristics never run. A nightly
        reflection cycle inventing this key at confidence 0.4 would
        silently take over browser routing, so the value has to be one a
        person actually stated.
        """
        with patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
             patch("assistant.preferences.get_preference",
                   return_value={"key": "automation_browser_mode",
                                 "value": "vision", "source": "reflection",
                                 "confidence": 0.95}):
            mode, meta = da._route_browser_content("fill the form", "Chrome")
        self.assertNotEqual(
            meta.get("reason"), "user_preference",
            "a model-written preference overrode browser routing")

    def test_a_preference_with_no_provenance_is_refused(self):
        """Fail closed. A row that cannot say where it came from is not a
        user statement, and this consumer accepts nothing less."""
        with patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
             patch("assistant.preferences.get_preference",
                   return_value={"key": "automation_browser_mode",
                                 "value": "vision"}):
            mode, meta = da._route_browser_content("fill the form", "Chrome")
        self.assertNotEqual(meta.get("reason"), "user_preference")

    def test_a_driver_import_failure_falls_back_safely(self):
        # If the extension module cannot import (unlikely, but this is the
        # bridge and it must not raise), treat it as no extension connected.
        # "playwright_bundled" is then translated to "vision" in the
        # browser-content scenario.
        with patch.dict("sys.modules", {"assistant.io.api.extension_ws": None}):
            mode, meta = da._route_browser_content("fill form", "Chrome")
        self.assertEqual(mode, "vision")
        self.assertEqual(meta["reason"], "extension_unavailable")
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
             patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend("fill this form with testing values")
        # CDP up + form-intent → DOM-mode
        self.assertEqual(backend, "dom")
        self.assertEqual(meta["reason"], "form_intent")
        self.assertEqual(meta["app"], "Truein - Google Chrome")

    def test_fill_form_falls_to_vision_when_cdp_down(self):
        with patch("assistant.io.screen.get_open_windows",
                   return_value=["Truein - Google Chrome"]), \
             patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(False)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend("fill this form with testing values")
        # CDP down → playwright_bundled → translated to vision
        self.assertEqual(backend, "vision")
        self.assertEqual(meta["reason"], "extension_unavailable")

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
             patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
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
             patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
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
             patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
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
             patch("assistant.io.api.extension_ws.drover_state_snapshot",
                   return_value=_FakeDriverState(True)), \
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
    """Verify `_execute_dom_task` calls `run_dom_form_fill` for form goals.

    `_execute_dom_task` imports the handle and orchestrator LOCALLY, so patches
    target the source modules rather than `router.<name>`, which never exists.

    The page-picking mock these tests used to carry is gone with the function it
    stood in for: the extension resolves every page verb to the active tab of
    the current window, so there is no list of contexts to sift.
    """

    @patch("assistant.automation.browser.dom_orchestrator.run_dom_form_fill", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.run_dom_task", new_callable=AsyncMock)
    @patch("assistant.automation.browser.handle.get_browser_handle", new_callable=AsyncMock)
    def test_form_intent_uses_form_fill(self, mock_handle, mock_old, mock_new):
        mock_handle.return_value = MagicMock(kind="drover", page=MagicMock())
        mock_new.return_value = MagicMock(
            success=True, final_summary="Form submitted.", reason="completed",
        )
        _run(da._execute_dom_task("Fill the registration form with test data"))
        mock_new.assert_called_once()
        mock_old.assert_not_called()

    @patch("assistant.automation.browser.dom_orchestrator.run_dom_form_fill", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.run_dom_task", new_callable=AsyncMock)
    @patch("assistant.automation.browser.handle.get_browser_handle", new_callable=AsyncMock)
    def test_non_form_uses_old_loop(self, mock_handle, mock_old, mock_new):
        mock_handle.return_value = MagicMock(kind="drover", page=MagicMock())
        mock_old.return_value = MagicMock(
            success=True, final_summary="Done.", reason="completed",
        )
        _run(da._execute_dom_task("Click the search button"))
        mock_old.assert_called_once()
        mock_new.assert_not_called()

    @patch("assistant.automation.browser.dom_orchestrator.run_dom_form_fill", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.run_dom_task", new_callable=AsyncMock)
    @patch("assistant.automation.browser.handle.get_browser_handle", new_callable=AsyncMock)
    def test_a_bundled_handle_falls_back_rather_than_driving_it(
        self, mock_handle, mock_old, mock_new
    ):
        """The router said "dom" and the extension went away in between.

        Driving the bundled browser here would run the task against a browser
        with none of the user's sessions -- on a goal the router chose precisely
        because her own browser was open at the page.
        """
        mock_handle.return_value = MagicMock(kind="bundled", page=MagicMock())
        result = _run(da._execute_dom_task("Fill the registration form"))
        self.assertEqual(result, "__FALLBACK__")
        mock_new.assert_not_called()
        mock_old.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
