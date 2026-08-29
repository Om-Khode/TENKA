"""test_window_target_resolution.py — KI-39.

A goal that names a window must reach that window, or reach nothing. It must
never quietly become "whatever is focused right now".

**Observed live, 2026-08-29.** The goal was "open notepad and type hello
world". Step 2 failed post-verification because focus had moved, the planner
built a recovery step that named the target correctly --

    type "hello world" in the 'hunter2.txt - Notepad' window

-- and "hello world" was typed into a Visual Studio Code document, which the
planner then marked `recovered`.

The guard that should have stopped it already existed. `_execute_native_task`
refuses the active-window fallback when `_extract_target_app` reports an
explicit target, with a comment naming this exact hazard. **Quotes stopped it
being reached**, in two independent places:

    _extract_target_app   `(?:the\\s+)?(\\w+)\\s*$` matches neither a quoted
                          title nor the trailing noun in "in the X window",
                          so it reported no target at all
    _detect_running_app   `.strip(".,!?")` leaves the apostrophe attached, so
                          the candidate is `notepad'`, and
                          `"notepad'" in "notepad"` is False

And the phrasing both of them failed on is the phrasing the planner's own
recovery prompt *produces*. One component wrote a name its siblings could not
read; neither was wrong alone.

All-stub: string functions plus one stubbed window list. Nothing enumerates
windows and nothing reaches the desktop.

Run: py -3.11 -m pytest tests/test_window_target_resolution.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from assistant.automation import router


class TestQuotedTargetIsExtracted(unittest.TestCase):
    """`_extract_target_app` must see a target wherever a target was named."""

    # (goal, expected target)  -- the first two are the shapes the planner
    # emits, and the exact string from the live failure.
    NAMED = [
        ("type \"hello world\" in the 'hunter2.txt - Notepad' window",
         "hunter2.txt - notepad"),
        ("type hello world in 'hunter2.txt - Notepad'",
         "hunter2.txt - notepad"),
        ('type hello world in "Untitled - Notepad"', "untitled - notepad"),
        ("type hello world in notepad", "notepad"),
        ("type hello world in the notepad window", "notepad"),
        ("type hello world in the notepad app", "notepad"),
        ("multiply 3 and 4 on calculator", "calculator"),
        # `into` -- the planner wrote this four times in one session and the
        # first version of this fix did not accept it, so two of four runs
        # still fell through to the active-window fallback.
        ('type "hello world" into the Notepad window', "notepad"),
        ('type "hello world" into notepad', "notepad"),
        ("type hello world into 'hunter2.txt - Notepad'", "hunter2.txt - notepad"),
    ]

    def test_every_named_target_is_found(self):
        for goal, want in self.NAMED:
            with self.subTest(goal=goal):
                got, _ = router._extract_target_app(goal)
                self.assertEqual(
                    got, want,
                    f"no target extracted from {goal!r}, so the explicit-target "
                    f"guard in _execute_native_task is skipped and the step "
                    f"falls back to the foreground window",
                )

    def test_the_live_failure_case_specifically(self):
        # Named on its own so a regression reads as itself in the output.
        goal = "type \"hello world\" in the 'hunter2.txt - Notepad' window"
        target, stripped = router._extract_target_app(goal)
        self.assertEqual(target, "hunter2.txt - notepad")
        self.assertEqual(stripped, 'type "hello world"')


class TestNothingIsClaimedThatWasNotNamed(unittest.TestCase):
    """The control, and the half a quoting change most easily breaks.

    Widening a suffix pattern to accept quotes risks reading quoted *content*
    as a target. Each of these must still report no target.
    """

    NOT_NAMED = [
        # A quote at the end of the sentence, with "in" inside it.
        'search for "cats in hats"',
        'remind me to buy "milk in a carton"',
        # Generic nouns the stop-word list exists to reject.
        "type X in the box",
        "type X in the form",
        "put it in the field",
        # A category rather than an app -- this one opened a Playwright cache
        # folder in File Explorer once.
        "play music on browser",
        # No target clause at all.
        "type hello world",
        "open notepad",
        # A goal that names no window. The planner wrote this one too, and
        # "the current document" is not something this function can resolve --
        # `_execute_native_task` must treat the absence as a reason for care,
        # not as permission to use the foreground.
        'type "hello world" into the current document',
        # `to` is ordinary English and is deliberately NOT a target
        # preposition. Adding it resolved this to an app called "list", which
        # `_resolve_target_window` would Win-key search for.
        "add it to the list",
        "remind me to buy milk",
        "send this to john",
    ]

    def test_none_of_these_yields_a_target(self):
        for goal in self.NOT_NAMED:
            with self.subTest(goal=goal):
                got, stripped = router._extract_target_app(goal)
                self.assertIsNone(
                    got, f"{goal!r} produced target {got!r} -- a goal that "
                         f"names no window must not acquire one")
                self.assertEqual(stripped, goal, "the goal was altered anyway")

    def test_a_quoted_target_still_honours_the_length_floor(self):
        # Two characters or fewer is noise however it is punctuated.
        self.assertIsNone(router._extract_target_app("type X in 'ab'")[0])


class TestQuotedWindowNamesSurviveWordSplitting(unittest.TestCase):
    """`_detect_running_app`'s candidate words must not carry punctuation.

    Driven through the real function with `get_open_windows` stubbed on the
    module the function-local import reads, so nothing enumerates windows.
    """

    def test_the_live_failure_case_resolves_through_the_real_function(self):
        """Driven through `_detect_running_app`, not through the constant.

        The first draft asserted on `_WORD_PUNCT` directly and did **not** red
        when the call site was reverted to `.strip(".,!?")` -- it was testing
        the constant's contents rather than that anything uses them. Only the
        AST sweep caught that mutation, which is one guard too few for the
        behaviour that actually broke.

        `get_open_windows` is patched as an attribute on the real
        `assistant.io.screen` module, which is where `_detect_running_app`'s
        function-local `from ..io import screen` looks. No window enumeration
        happens.
        """
        import assistant.io.screen as screen_mod
        from unittest.mock import patch

        goal = "type \"hello world\" in the 'hunter2.txt - Notepad' window"
        windows = ["hunter2.txt - Notepad",
                   "Untitled-1 - TENKA - Visual Studio Code"]

        with patch.object(screen_mod, "get_open_windows", return_value=windows):
            got = router._detect_running_app(goal)

        self.assertEqual(
            got, "hunter2.txt - Notepad",
            "the quoted window name did not resolve, so _execute_native_task "
            "falls through to the active-window fallback and types into "
            "whatever is foreground",
        )

    def test_an_unnamed_window_still_resolves_to_nothing(self):
        """The control: the matcher must not start claiming windows."""
        import assistant.io.screen as screen_mod
        from unittest.mock import patch

        with patch.object(screen_mod, "get_open_windows",
                          return_value=["hunter2.txt - Notepad"]):
            self.assertIsNone(
                router._detect_running_app("what is the weather in pune"))

    def test_every_quote_style_is_stripped(self):
        for raw, want in [("notepad'", "notepad"), ('"notepad"', "notepad"),
                          ("'notepad'", "notepad"), ("(notepad)", "notepad"),
                          ("notepad;", "notepad"), ("[notepad]", "notepad"),
                          ("`notepad`", "notepad"), ("notepad.", "notepad")]:
            with self.subTest(raw=raw):
                self.assertEqual(raw.strip(router._WORD_PUNCT), want)

    def test_the_constant_is_used_everywhere_a_goal_is_split(self):
        """Anti-drift: four copies of a punctuation list is four chances for
        the next character to be added to three of them."""
        import ast

        src = (Path(__file__).parent.parent / "assistant" / "automation"
               / "router.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        # Any `.strip("...")` on a literal, other than the constant's own
        # definition, is a site that will drift.
        literal_strips = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) == "strip"
            and len(n.args) == 1
            and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
            and any(ch in n.args[0].value for ch in ".,!?")
        ]
        self.assertFalse(
            literal_strips,
            f"router.py strips a literal punctuation set at lines "
            f"{literal_strips} instead of using _WORD_PUNCT",
        )

        used = src.count("_WORD_PUNCT")
        self.assertGreaterEqual(
            used, 4, "walked nothing useful: _WORD_PUNCT appears "
                     f"{used} times, expected its definition plus the "
                     f"goal-splitting sites")


if __name__ == "__main__":
    unittest.main(verbosity=2)
