"""
test_router_content_verb_routing.py — guards the three routing bugs that
fired during the README-demo live test on 2026-06-08:

  1. `_detect_running_app` matched the *site name* in a goal against the
     browser's window-title brand suffix (e.g. goal "...on youtube" matched
     the Firefox window already playing an Among Us video), so "play piano
     on youtube" tried to focus the stale tab instead of searching.

  2. The `launch_keyword` regex `(open|launch|start|run)\\s+(\\w+)` captured
     content verbs as app names — "open play cat videos…" got app="play".

  3. `app_context_pattern` and `_extract_target_app` treated the generic
     category "browser" (and other category nouns) as a specific app, so
     "X on browser" drove `_resolve_target_window("browser")` → Win-key
     search opened the local `browser-cache` Playwright folder in File
     Explorer.

The fix lives in `assistant/automation/router.py`:
  - `_GENERIC_CATEGORY_WORDS` / `_LAUNCH_VERB_STOPLIST` / `_BROWSER_ONLY_GOAL_RE`
  - `detect_backend` priority-3 split: browser-only-goal → focus,
    form-intent → `_route_browser_content`, content-goal → bundled-browser
  - `detect_backend` priority-4: launch_keyword rejects content verbs
  - `detect_backend` priority-5: "X on browser" → browser, other
    generic categories rejected
  - `_extract_target_app` rejects generic categories
  - `_execute_browser_task` strips generic-category suffix before LLM

Run: python -m pytest tests/test_router_content_verb_routing.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.router as da


class _FakeCdpState:
    def __init__(self, available: bool):
        self.available = available


# ─── Bug 1: site-name false-match against running browser ───────────────


class TestRunningBrowserContentRouting(unittest.TestCase):
    """When a browser is already running and the goal contains a site
    word (e.g. "youtube"), `_detect_running_app` matches the window's
    brand suffix. The fix: route to bundled-browser content, NOT native
    focus or vision-loop, so the user actually gets the requested content."""

    def test_play_x_on_youtube_with_firefox_open_routes_to_browser(self):
        # The bug case: Firefox is on a YouTube tab playing X, user says
        # "play piano on youtube". Previously routed to native focus
        # (stale tab) or vision-loop (no CDP).
        with patch("assistant.automation.router._detect_running_app",
                   return_value="Among Us | Paramount+ - YouTube — Mozilla Firefox"):
            backend, meta = da.detect_backend("play piano music on youtube")
        self.assertEqual(backend, "browser")
        self.assertEqual(meta["reason"], "browser_content_via_running_browser")
        self.assertIn("Firefox", meta["running"])

    def test_open_play_cat_videos_on_youtube_in_browser_routes_to_browser(self):
        # The exact log-reproduced phrase. Previously cycled through
        # native-focus → launch_keyword(play) → File-Explorer disaster.
        with patch("assistant.automation.router._detect_running_app",
                   return_value="something - YouTube — Mozilla Firefox"):
            backend, meta = da.detect_backend(
                "open play cat videos on youtube in browser"
            )
        self.assertEqual(backend, "browser")

    def test_open_chrome_alone_still_focuses_chrome(self):
        # Regression guard: the ONE case where browser-window detection
        # should still route to native focus — the user literally said
        # "open chrome" with nothing else.
        with patch("assistant.automation.router._detect_running_app",
                   return_value="Google Chrome"):
            backend, meta = da.detect_backend("open chrome")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "running_app_detected")

    def test_switch_to_firefox_focuses_firefox(self):
        with patch("assistant.automation.router._detect_running_app",
                   return_value="something - Mozilla Firefox"):
            backend, meta = da.detect_backend("switch to firefox")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "running_app_detected")

    def test_form_intent_with_chrome_open_still_uses_route_browser_content(self):
        # Existing carve-out preserved: "fill the form" with Chrome open
        # AND CDP up → DOM mode (not bundled-browser).
        with patch("assistant.automation.router._detect_running_app",
                   return_value="Truein - Google Chrome"), \
             patch("assistant.automation.browser.cdp.cdp_state_snapshot",
                   return_value=_FakeCdpState(True)), \
             patch("assistant.preferences.get_preference", return_value=None):
            backend, meta = da.detect_backend(
                "open chrome and fill the form with my email"
            )
        self.assertEqual(backend, "dom")
        self.assertEqual(meta["reason"], "form_intent")

    def test_non_browser_running_app_still_routes_native(self):
        # Notepad open → goal "type hello in notepad" → native, unchanged.
        with patch("assistant.automation.router._detect_running_app",
                   return_value="Untitled - Notepad"):
            backend, meta = da.detect_backend("type hello in notepad")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "running_app_detected")


# ─── Bug 2: content verbs captured as app names by launch_keyword ───────


class TestLaunchKeywordVerbStoplist(unittest.TestCase):
    """`(open|launch|start|run)\\s+(\\w+)` is too greedy. Content verbs
    like "play", "watch", "search" are never app names — fall through."""

    def test_open_play_does_not_become_app_named_play(self):
        # No running app, no "on X" suffix → priority 4 fires. The bug:
        # captured "play" as app_name. The fix: stoplist rejects, falls
        # through to "unknown".
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("open play cat videos")
        # Either fell to "unknown", or to a downstream priority — but
        # NEVER native with reason=launch_keyword + app=play.
        self.assertFalse(
            meta.get("reason") == "launch_keyword" and meta.get("app") == "play",
            f"content verb 'play' should not be captured as launch app; got {meta}",
        )

    def test_open_watch_movies_does_not_become_app_named_watch(self):
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("open watch movies")
        self.assertFalse(
            meta.get("reason") == "launch_keyword" and meta.get("app") == "watch"
        )

    def test_open_search_does_not_become_app_named_search(self):
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("open search the web")
        self.assertFalse(
            meta.get("reason") == "launch_keyword" and meta.get("app") == "search"
        )

    def test_open_chrome_still_captured_as_launch_keyword(self):
        # Negative: real app names still route via launch_keyword when
        # no running window detected.
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("open chrome")
        # Either launch_keyword(chrome) or browser_intent — both fine,
        # not the stoplist-filtered "unknown" case.
        self.assertNotEqual(meta.get("reason"), "no_match")

    def test_open_notepad_still_captured_as_launch_keyword(self):
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("open notepad")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "launch_keyword")
        self.assertEqual(meta["app"], "notepad")


# ─── Bug 3: generic category "browser" treated as app name ──────────────


class TestGenericCategoryRouting(unittest.TestCase):
    """`app_context_pattern` and `_extract_target_app` both used to take
    "browser" / "music" / etc. as specific app names. The fix routes
    "X on browser" to the browser backend (default browser via webbrowser
    module / Playwright bundled) and falls through for other categories."""

    def test_x_on_browser_routes_to_browser_category(self):
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("play cat videos on browser")
        self.assertEqual(backend, "browser")
        self.assertEqual(meta["reason"], "browser_category")

    def test_x_in_browser_routes_to_browser_category(self):
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("search piano tutorials in browser")
        self.assertEqual(backend, "browser")

    def test_x_on_music_does_not_route_to_native_music(self):
        # No app literally called "music" should be invoked. Fall through
        # so the LLM planner handles it.
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("play lofi on music")
        self.assertFalse(
            meta.get("reason") == "app_context_pattern" and meta.get("app") == "music",
            f"generic 'music' should not be a target app; got {meta}",
        )

    def test_x_on_spotify_still_routes_to_native(self):
        # Regression guard: real app names still work via app_context_pattern.
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("play piano on spotify")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "app_context_pattern")
        self.assertEqual(meta["app"], "spotify")

    def test_type_hello_in_notepad_unchanged(self):
        # Regression guard for `_extract_target_app` — Notepad still
        # resolves as a target app, not blocked by the category list.
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend("type hello world in notepad")
        self.assertEqual(backend, "native")
        self.assertEqual(meta["reason"], "app_context_pattern")
        self.assertEqual(meta["app"], "notepad")


# ─── _extract_target_app — unit tests for the category guard ────────────


class TestExtractTargetAppCategoryGuard(unittest.TestCase):
    def test_browser_rejected(self):
        target, stripped = da._extract_target_app("play cat videos on browser")
        self.assertIsNone(target)
        self.assertEqual(stripped, "play cat videos on browser")

    def test_music_rejected(self):
        target, stripped = da._extract_target_app("play lofi on music")
        self.assertIsNone(target)

    def test_video_rejected(self):
        target, stripped = da._extract_target_app("watch cat videos on video")
        self.assertIsNone(target)

    def test_player_rejected(self):
        target, stripped = da._extract_target_app("play song on player")
        self.assertIsNone(target)

    def test_notepad_passes_through(self):
        target, stripped = da._extract_target_app("type hello in notepad")
        self.assertEqual(target, "notepad")
        self.assertEqual(stripped, "type hello")

    def test_spotify_passes_through(self):
        target, stripped = da._extract_target_app("play lofi on spotify")
        self.assertEqual(target, "spotify")
        self.assertEqual(stripped, "play lofi")

    def test_no_suffix_unchanged(self):
        target, stripped = da._extract_target_app("hello world")
        self.assertIsNone(target)
        self.assertEqual(stripped, "hello world")


# ─── _strip_generic_category_suffix — unit tests ───────────────────────


class TestStripGenericCategorySuffix(unittest.TestCase):
    """The browser-task LLM planner doesn't need "on browser" suffixes —
    they carry no routing signal and can mislead it into navigating to
    `browser.com`. Strip them before handoff."""

    def test_strips_on_browser(self):
        self.assertEqual(
            da._strip_generic_category_suffix("play cat videos on youtube on browser"),
            "play cat videos on youtube",
        )

    def test_strips_in_browser(self):
        self.assertEqual(
            da._strip_generic_category_suffix("search piano in browser"),
            "search piano",
        )

    def test_strips_with_my_music(self):
        self.assertEqual(
            da._strip_generic_category_suffix("play this with my music"),
            "play this",
        )

    def test_does_not_strip_specific_brand(self):
        # "on youtube" stays — that's a specific routing signal.
        self.assertEqual(
            da._strip_generic_category_suffix("play piano on youtube"),
            "play piano on youtube",
        )

    def test_does_not_strip_specific_app(self):
        self.assertEqual(
            da._strip_generic_category_suffix("play piano on spotify"),
            "play piano on spotify",
        )

    def test_empty_string_no_op(self):
        self.assertEqual(da._strip_generic_category_suffix(""), "")

    def test_no_suffix_no_op(self):
        self.assertEqual(
            da._strip_generic_category_suffix("just play piano"),
            "just play piano",
        )

    def test_strips_videos_category(self):
        self.assertEqual(
            da._strip_generic_category_suffix("show me cats on videos"),
            "show me cats",
        )


# ─── End-to-end: the original log-reproduced phrases ────────────────────


class TestLogReproducedPhrases(unittest.TestCase):
    """Direct guards on the three log phrases that motivated this fix.
    The intent layer routes these to either `computer_task` or
    `code_executor`; whichever they reach, the router must NOT:
      - launch app named 'play'
      - focus a stale YouTube tab
      - open the File Explorer 'browser-cache' folder
    """

    def test_phrase_open_play_cat_videos_on_youtube_in_browser(self):
        # The first log phrase. With Firefox on YouTube already, route
        # to browser-content. With nothing running, fall through cleanly.
        with patch("assistant.automation.router._detect_running_app",
                   return_value="YouTube — Mozilla Firefox"):
            backend, meta = da.detect_backend(
                "open play cat videos on youtube in browser"
            )
        self.assertEqual(backend, "browser")
        # And nothing claims app='play':
        self.assertNotEqual(meta.get("app"), "play")

    def test_phrase_play_cat_videos_on_youtube_on_browser_no_browser_running(self):
        # The third log phrase, with no browser detected. Route via
        # browser_category — must NOT call _resolve_target_window('browser').
        with patch("assistant.automation.router._detect_running_app",
                   return_value=None):
            backend, meta = da.detect_backend(
                "play cat videos on youtube on browser"
            )
        self.assertEqual(backend, "browser")
        # Must not have parsed "browser" as a target_app
        target, _ = da._extract_target_app(
            "play cat videos on youtube on browser"
        )
        self.assertIsNone(target)


if __name__ == "__main__":
    unittest.main()
