"""
test_browser_extension_setup_hook.py — the pre-router catches setup phrasings.

The hazard this guards is specific and was expensive: a "set up the browser
extension" goal that misses the regex falls through to intent classification,
gets called a `computer_task`, and the vision loop types the literal goal string
into whatever search bar is on screen — repeatedly. So the phrasings are caught
before classification, and the negative cases matter as much as the positive
ones.

**The vocabulary changed with the mechanism.** The old pattern matched "chrome"
and "cdp", because the thing being set up was Chrome's remote-debugging port.
The extension drives whatever browser is already open, so a brand name in this
regex would be both wrong and a THE-rule violation. The tests that used to
assert "set up chrome" routes here now assert it does **not** — there is nothing
to set up in Chrome any more, and claiming that intent would send the user
somewhere that cannot help them.

Run: py -3.11 -m pytest tests/test_browser_extension_setup_hook.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.regex_router import pre_route  # noqa: E402

INTENT = "browser_extension_setup"


class _Base(unittest.TestCase):
    def _assert_mode(self, text: str, mode: str):
        result = pre_route(text)
        self.assertIsNotNone(result, f"{text!r} was not caught by the pre-router")
        self.assertEqual(result.intent, INTENT, f"{text!r} routed to {result.intent}")
        self.assertEqual(result.params.get("mode"), mode,
                         f"{text!r} got mode={result.params.get('mode')!r}")

    def _assert_setup(self, text: str):
        self._assert_mode(text, "setup")

    def _assert_undo(self, text: str):
        self._assert_mode(text, "undo")

    def _assert_preview(self, text: str):
        self._assert_mode(text, "preview")

    def _assert_not_this_intent(self, text: str):
        result = pre_route(text)
        if result is not None:
            self.assertNotEqual(
                result.intent, INTENT,
                f"{text!r} was claimed by {INTENT}, which cannot help with it",
            )


class TestSetupPhrasings(_Base):
    def test_verb_then_noun(self):
        for text in [
            "set up the browser extension",
            "setup the browser extension",
            "configure the browser extension",
            "enable the browser extension",
            "prepare the browser extension",
            "activate the browser extension",
            "connect the browser extension",
        ]:
            with self.subTest(text=text):
                self._assert_setup(text)

    def test_noun_then_verb(self):
        for text in [
            "browser extension setup",
            "browser extension, set it up",
            "extension setup please",
        ]:
            with self.subTest(text=text):
                self._assert_setup(text)

    def test_the_short_noun_alone(self):
        self._assert_setup("set up the extension")
        self._assert_setup("configure drover")

    def test_words_between_verb_and_noun(self):
        self._assert_setup("set up that browser extension for me")
        self._assert_setup("please configure the browser extension now")


class TestUndoPhrasings(_Base):
    def test_undo_verbs(self):
        for text in [
            "undo the browser extension",
            "reverse the browser extension setup",
            "revert the browser extension",
            "deactivate the browser extension",
            "disconnect the browser extension",
        ]:
            with self.subTest(text=text):
                self._assert_undo(text)

    def test_undo_wins_over_setup_when_both_verbs_appear(self):
        # Checked first, deliberately: "undo the extension setup" contains a
        # setup verb, and reading it as `setup` would re-mint a credential the
        # user just asked to remove.
        self._assert_undo("undo the browser extension setup")


class TestPreviewPhrasings(_Base):
    def test_preview_markers(self):
        for text in [
            "preview the browser extension setup",
            "show me what the browser extension setup would do",
            "dry run the browser extension setup",
        ]:
            with self.subTest(text=text):
                self._assert_preview(text)


class TestTheOldVocabularyNoLongerClaims(_Base):
    """Chrome and CDP phrasings must fall through now.

    There is no Chrome-specific setup left. A regex that still claimed these
    would answer a question about the browser with instructions for a mechanism
    that no longer exists — worse than not matching, because the user would
    follow them.
    """

    def test_chrome_phrasings_are_not_claimed(self):
        for text in [
            "set up chrome",
            "configure chrome",
            "enable cdp",
            "activate cdp",
            "set up the remote debugging port",
            "configure chrome cdp",
        ]:
            with self.subTest(text=text):
                self._assert_not_this_intent(text)


class TestNegativeCases(_Base):
    def test_plain_browser_actions_are_not_claimed(self):
        for text in [
            "open chrome",
            "close firefox",
            "switch to the browser",
            "open the browser and search for cats",
        ]:
            with self.subTest(text=text):
                self._assert_not_this_intent(text)

    def test_unrelated_goals_are_not_claimed(self):
        for text in [
            "set up my email",
            "configure the microphone",
            "what time is it",
            "enable dark mode",
        ]:
            with self.subTest(text=text):
                self._assert_not_this_intent(text)

    def test_a_bare_browser_name_does_not_trigger_undo(self):
        # The undo pattern requires an unambiguous noun. "remove chrome" is a
        # request about an application, not about this credential.
        self._assert_not_this_intent("remove chrome")
        self._assert_not_this_intent("uninstall firefox")


if __name__ == "__main__":
    unittest.main(verbosity=2)
