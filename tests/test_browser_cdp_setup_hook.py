"""
test_browser_cdp_setup_hook.py — Phase 1F: pipeline-level browser-cdp-setup intercept.

Verifies the regex pre-router catches every phrasing that should run the
Chrome CDP setup script BEFORE intent classification — eliminating the
runaway-vision-loop hazard where a "set up chrome" goal misroutes to
computer_task and the vision loop types the literal goal string into
search bars indefinitely.

Coverage:
  - Setup phrasings (verb→noun, noun→verb, multi-word nouns)
  - Undo phrasings (must require unambiguous CDP-related noun)
  - Preview detection (preview / show me / what would / dry run)
  - Negative cases (must NOT match: "open chrome", "close chrome",
    "remove chrome" without setup-noun, generic computer tasks)
  - Intent + params shape (browser_cdp_setup with mode key)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from assistant import regex_router


class TestChromeSetupRoute(unittest.TestCase):
    """Setup phrasings should resolve to browser_cdp_setup with mode=setup."""

    def _assert_setup(self, text: str):
        r = regex_router.pre_route(text)
        self.assertIsNotNone(r, f"expected match for: {text!r}")
        self.assertEqual(r.intent, "browser_cdp_setup", f"wrong intent for: {text!r}")
        self.assertEqual(r.params.get("mode"), "setup", f"wrong mode for: {text!r}")

    def test_set_up_chrome(self):
        self._assert_setup("set up chrome")

    def test_set_up_chrome_for_yourself(self):
        self._assert_setup("set up chrome for yourself")

    def test_setup_chrome_no_space(self):
        self._assert_setup("setup chrome")

    def test_configure_chrome(self):
        self._assert_setup("configure chrome")

    def test_configure_chrome_cdp(self):
        self._assert_setup("configure chrome cdp")

    def test_enable_cdp(self):
        self._assert_setup("enable cdp")

    def test_enable_remote_debugging(self):
        self._assert_setup("enable remote debugging")

    def test_enable_remote_debugging_port(self):
        self._assert_setup("enable remote debugging port")

    def test_prepare_chrome(self):
        self._assert_setup("prepare chrome")

    def test_activate_cdp(self):
        self._assert_setup("activate cdp")

    # Reverse word order — noun first, verb after — must also match
    def test_browser_cdp_setup_noun_first(self):
        self._assert_setup("chrome setup")

    def test_do_browser_cdp_setup(self):
        self._assert_setup("do chrome setup")

    def test_chrome_cdp_setup(self):
        self._assert_setup("chrome cdp setup")

    def test_cdp_setup(self):
        self._assert_setup("cdp setup")

    # Filler words between verb and noun should be tolerated
    def test_set_up_my_chrome(self):
        self._assert_setup("set up my chrome")

    def test_configure_chrome_for_tenka(self):
        self._assert_setup("configure chrome for tenka")

    def test_case_insensitive(self):
        self._assert_setup("SET UP CHROME")
        self._assert_setup("Configure Chrome CDP")


class TestChromeSetupPreview(unittest.TestCase):
    """Preview markers should produce mode=preview, not mode=setup."""

    def _assert_preview(self, text: str):
        r = regex_router.pre_route(text)
        self.assertIsNotNone(r, f"expected match for: {text!r}")
        self.assertEqual(r.intent, "browser_cdp_setup")
        self.assertEqual(r.params.get("mode"), "preview", f"wrong mode for: {text!r}")

    def test_preview_browser_cdp_setup(self):
        self._assert_preview("preview chrome setup")

    def test_show_me_browser_cdp_setup(self):
        self._assert_preview("show me chrome setup")

    def test_what_would_browser_cdp_setup_do(self):
        self._assert_preview("what would chrome setup do")

    def test_dry_run_browser_cdp_setup(self):
        self._assert_preview("dry run chrome setup")

    def test_dry_dash_run(self):
        self._assert_preview("dry-run chrome setup")


class TestChromeSetupUndo(unittest.TestCase):
    """Undo phrasings should resolve to browser_cdp_setup with mode=undo."""

    def _assert_undo(self, text: str):
        r = regex_router.pre_route(text)
        self.assertIsNotNone(r, f"expected match for: {text!r}")
        self.assertEqual(r.intent, "browser_cdp_setup")
        self.assertEqual(r.params.get("mode"), "undo", f"wrong mode for: {text!r}")

    def test_undo_browser_cdp_setup(self):
        self._assert_undo("undo chrome setup")

    def test_reverse_browser_cdp_setup(self):
        self._assert_undo("reverse chrome setup")

    def test_unset_cdp(self):
        self._assert_undo("unset cdp")

    def test_revert_chrome_cdp(self):
        self._assert_undo("revert chrome cdp")

    def test_restore_browser_cdp_setup(self):
        self._assert_undo("restore chrome setup")

    def test_deactivate_cdp(self):
        self._assert_undo("deactivate cdp")

    def test_deactivate_remote_debugging(self):
        self._assert_undo("deactivate remote debugging")


class TestChromeSetupNegatives(unittest.TestCase):
    """Goals that must NOT route to browser_cdp_setup."""

    def _assert_not_browser_cdp_setup(self, text: str):
        r = regex_router.pre_route(text)
        if r is not None:
            self.assertNotEqual(
                r.intent, "browser_cdp_setup",
                f"unexpectedly matched browser_cdp_setup for: {text!r}",
            )

    def test_open_chrome(self):
        # Should route to open_browser or computer_task, never browser_cdp_setup
        self._assert_not_browser_cdp_setup("open chrome")

    def test_close_chrome(self):
        self._assert_not_browser_cdp_setup("close chrome")

    def test_remove_chrome_alone(self):
        # "remove chrome" without a setup/cdp/debug noun is ambiguous
        # (could mean uninstall the browser) — must not match undo
        self._assert_not_browser_cdp_setup("remove chrome")

    def test_disable_chrome_alone(self):
        # "disable" was removed from undo verbs because it's too generic
        self._assert_not_browser_cdp_setup("disable chrome")

    def test_what_time_is_it(self):
        self._assert_not_browser_cdp_setup("what time is it")

    def test_fill_this_form(self):
        # The exact goal that worked yesterday — must NOT misroute
        self._assert_not_browser_cdp_setup("fill this form with testing values")

    def test_open_settings(self):
        self._assert_not_browser_cdp_setup("open settings")

    def test_take_a_screenshot(self):
        self._assert_not_browser_cdp_setup("take a screenshot")

    def test_play_music(self):
        self._assert_not_browser_cdp_setup("play my liked songs")

    def test_chrome_only(self):
        # Bare "chrome" should not match — too ambiguous
        self._assert_not_browser_cdp_setup("chrome")


class TestChromeSetupPriority(unittest.TestCase):
    """The browser-cdp-setup check must run BEFORE other pre-routes that could
    accidentally match these phrases (e.g. _OPEN_APP_RE on "set up chrome")."""

    def test_setup_beats_open_app(self):
        # "set up chrome" starts with "set" — _OPEN_APP_RE would match
        # "start ..." patterns. Make sure browser_cdp_setup wins.
        r = regex_router.pre_route("set up chrome")
        self.assertEqual(r.intent, "browser_cdp_setup")

    def test_undo_with_setup_keyword_routes_to_undo(self):
        # "undo chrome setup" contains both undo-verb AND setup-verb.
        # Undo must win because it's checked first (and is the user's intent).
        r = regex_router.pre_route("undo chrome setup")
        self.assertEqual(r.intent, "browser_cdp_setup")
        self.assertEqual(r.params.get("mode"), "undo")


class TestIntentRegistration(unittest.TestCase):
    """The browser_cdp_setup intent must be wired through config + actions."""

    def test_intent_in_allowed_set(self):
        from assistant import config
        self.assertIn("browser_cdp_setup", config.ALLOWED_INTENTS)

    def test_intent_in_intents_list(self):
        from assistant import config
        self.assertIn("browser_cdp_setup", config.INTENTS)

    def test_handler_registered(self):
        from assistant import actions
        self.assertIn("browser_cdp_setup", actions._TOOLS)
        self.assertTrue(callable(actions._TOOLS["browser_cdp_setup"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
