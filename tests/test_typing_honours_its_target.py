"""test_typing_honours_its_target.py — KI-39, second half.

`native.type_text` with no selector types via `pyautogui.write`, which goes
wherever the OS focus is. It accepted a `window` argument, logged it, and never
consulted it -- so a step reading

    type - {'text': 'hello world', 'window': '*hello world - Notepad'}

typed into whatever the operator had clicked since, and the turn reported
success naming the window it had not typed into.

**Observed live, 2026-08-29.** Fourteen seconds separated the `focus` step from
the keystrokes -- a Ctrl+N, a vision call and a one-second wait in between --
the operator clicked their editor during it, and "hello world" landed in a
source file. The reply said Notepad.

The guard already existed one tier up: `vision/agent.py:keyboard_type` takes
`expected_window` and returns `ABORTED_WRONG_FOCUS`. The deterministic tier had
the parameter and not the check, which put the safety property in the fallback
path and not in the one that runs first.

`pyautogui`, `psutil` and the screen module are all stubbed. Nothing here can
press a key: the tests assert on what `type_text` *returns*, and the ones that
let typing proceed assert the stub was called rather than that anything
happened on screen.

Run: py -3.11 -m pytest tests/test_typing_honours_its_target.py -q
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.native as native


def _run(coro):
    return asyncio.run(coro)


class _Fixture:
    """Stub the three things `type_text`'s untargeted path can reach.

    `pyautogui` goes in `sys.modules` -- `type_text` does a plain
    `import pyautogui`, which consults it first. The screen module is patched
    as an **attribute** on `assistant.io.screen`, because `_focus_matches`
    reads `from ..io import screen`, and that resolves the attribute on the
    package before `sys.modules` is considered. Patching the wrong one of
    those two is how a test in this project once drove the real desktop.
    """

    def __init__(self, active: str, *, exe: str = "code") -> None:
        self.active = active
        self.exe = exe
        self.typed: list[str] = []
        self.focused: list[str] = []
        self._patches: list = []

    def __enter__(self) -> "_Fixture":
        pag = types.ModuleType("pyautogui")
        pag.write = lambda t, interval=0: self.typed.append(t)
        pag.hotkey = lambda *a: self.typed.append("<paste>")
        pag.press = lambda k: self.typed.append(k)
        self.pyautogui = pag

        import assistant.io.screen as screen_mod

        self._patches = [
            patch.dict(sys.modules, {"pyautogui": pag, "psutil": self._psutil()}),
            patch.object(screen_mod, "get_active_window",
                         side_effect=lambda: self.active),
            # Force the branch under test rather than depending on which
            # automation backend happens to be installed on this machine.
            patch.object(native, "_backend", "terminator"),
            patch.object(native, "is_available", return_value=True),
            patch.object(native, "focus_window",
                         new=AsyncMock(side_effect=self._focus)),
        ]
        for p in self._patches:
            p.start()
        return self

    async def _focus(self, name: str) -> str:
        self.focused.append(name)
        return f"Focused window: {name}"

    def _psutil(self):
        mod = types.ModuleType("psutil")
        proc = MagicMock()
        proc.name.return_value = f"{self.exe}.exe"
        mod.Process = lambda pid: proc
        return mod

    def __exit__(self, *exc) -> None:
        for p in reversed(self._patches):
            p.stop()


class TestRefusesWhenFocusMoved(unittest.TestCase):
    def test_it_refuses_rather_than_typing_into_the_wrong_window(self):
        with _Fixture("Untitled-1 - Visual Studio Code") as f:
            out = _run(native.type_text("hello world",
                                        window="hunter2.txt - Notepad"))

        self.assertIn("ABORTED_WRONG_FOCUS", out)
        self.assertEqual(
            f.typed, [],
            "keystrokes were sent to the foreground window anyway")

    def test_the_refusal_names_both_windows(self):
        # The planner reads this string and the spoken summary is built from
        # it; "it didn't work" is not something either can act on.
        with _Fixture("Untitled-1 - Visual Studio Code"):
            out = _run(native.type_text("hi", window="hunter2.txt - Notepad"))
        self.assertIn("hunter2.txt - Notepad", out)
        self.assertIn("Visual Studio Code", out)

    def test_it_tries_one_refocus_before_giving_up(self):
        # A transient steal is the common case; refusing without trying would
        # trade one failure mode for another.
        with _Fixture("Untitled-1 - Visual Studio Code") as f:
            _run(native.type_text("hi", window="hunter2.txt - Notepad"))
        self.assertEqual(f.focused, ["hunter2.txt - Notepad"])

    def test_a_refocus_that_works_lets_the_text_through(self):
        f = _Fixture("Untitled-1 - Visual Studio Code")
        with f:
            async def _recover(name):
                f.focused.append(name)
                f.active = name          # the re-focus actually succeeds
                return "ok"

            with patch.object(native, "focus_window",
                              new=AsyncMock(side_effect=_recover)):
                out = _run(native.type_text("hello world",
                                            window="hunter2.txt - Notepad"))

        self.assertNotIn("ABORTED", out)
        self.assertEqual(f.typed, ["hello world"])


class TestTypesWhenTheTargetIsFocused(unittest.TestCase):
    """The control. A guard that refuses correctly while breaking the path it
    permits passes every red-green check there is."""

    def test_the_named_window_being_focused_types_normally(self):
        with _Fixture("hunter2.txt - Notepad") as f:
            out = _run(native.type_text("hello world",
                                        window="hunter2.txt - Notepad"))
        self.assertEqual(f.typed, ["hello world"])
        self.assertNotIn("ABORTED", out)
        self.assertEqual(f.focused, [], "an unnecessary re-focus was issued")

    def test_no_window_named_means_no_check(self):
        # Plenty of callers type into "whatever is focused" on purpose -- a
        # dialog, a search box, a field just clicked. The guard is about a
        # *claim* being wrong, not about typing without one.
        with _Fixture("Anything At All") as f:
            out = _run(native.type_text("hello world"))
        self.assertEqual(f.typed, ["hello world"])
        self.assertNotIn("ABORTED", out)

    def test_a_renamed_window_of_the_right_process_still_types(self):
        # Media players rename their window while playing ("Artist - Song").
        # Title matching alone would refuse the correct app, so the process
        # check exists -- the same reasoning the vision tier gives.
        with _Fixture("Some Artist - A Song", exe="spotify") as f:
            out = _run(native.type_text("hello", window="Spotify Premium"))
        self.assertEqual(f.typed, ["hello"])
        self.assertNotIn("ABORTED", out)


class TestUnreadableFocusFailsClosed(unittest.TestCase):
    def test_it_refuses_when_the_foreground_cannot_be_read(self):
        """Unknown is not a match. A keystroke is not the place to assume."""
        import assistant.io.screen as screen_mod

        with _Fixture("hunter2.txt - Notepad") as f:
            with patch.object(screen_mod, "get_active_window",
                              side_effect=RuntimeError("no desktop")):
                out = _run(native.type_text("hello",
                                            window="hunter2.txt - Notepad"))
        self.assertIn("ABORTED_WRONG_FOCUS", out)
        self.assertEqual(f.typed, [])


class TestTheVocabularyIsShared(unittest.TestCase):
    def test_the_refusal_uses_the_tag_the_rest_of_the_tree_knows(self):
        """`ABORTED_WRONG_FOCUS` already means this.

        `recovery.py` skips results containing it, the vision planner prompt
        tells the model to recover from it by adding a `focus_application`
        step, and `agent._action_failed` treats it as a non-success. Inventing
        a second spelling for the deterministic tier would leave all three
        blind to it.
        """
        root = Path(__file__).parent.parent / "assistant" / "automation"
        agent_src = (root / "vision" / "agent.py").read_text(encoding="utf-8")
        recovery_src = (root / "recovery.py").read_text(encoding="utf-8")
        self.assertIn("ABORTED_WRONG_FOCUS", agent_src)
        self.assertIn("aborted_wrong_focus", recovery_src.lower())

        with _Fixture("Untitled-1 - Visual Studio Code"):
            out = _run(native.type_text("hi", window="hunter2.txt - Notepad"))
        self.assertTrue(out.startswith("ABORTED_WRONG_FOCUS"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
