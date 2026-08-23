"""
test_planner_integration.py — planner-vision part 3: 3-pass pipeline + ESC race fix.

Covers the integrated _update_todos_after_batch behaviour:
  - Pass 1 confirms a deferred select-TODO via vision; rules+LLM still see
    the post-confirmation state.
  - Pass 2 rules deterministically mark a type-TODO; Pass 3 LLM is skipped
    when rules covered the whole batch.
  - Pass 3 LLM fires for kind="other" TODOs (e.g. "Submit form"); LLM marks
    pass through.
  - Pass 3 visible_ids guard: LLM cannot mark a select-TODO that's already
    pending_visual_confirm (would re-introduce hallucination class).
  - Kill-switch (config.DETERMINISTIC_MATCHING_ENABLED=False) reverts
    to PE-1 LLM-only path.
  - run_computer_task must NOT manage the session ESC/abort lifecycle (that
    moved to main.py), and delegates to the inner loop exactly once. Checked
    by reading the source -- see TestEscRaceFix for why running it was the
    problem rather than the test.

Run: py -3.11 -m pytest tests/test_planner_integration.py -q
"""

from __future__ import annotations

import ast
import pathlib
import textwrap
import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca
import assistant.config as cfg


def _run(coro):
    return asyncio.run(coro)


def _wrap_vision(v):
    """Wrap a bare string in SimpleNamespace(text=...) for LLMResult compat."""
    if isinstance(v, Exception):
        return v
    return SimpleNamespace(text=v)


_ORIGINAL_ATTRS: list = []
"""Package attributes `_install_screen_llm` overwrote, oldest first.

A stack rather than a dict: a test may install twice, and restoring in reverse
puts back what was there rather than what a later call happened to see.
"""


def _restore_stubs() -> None:
    """Undo every package-attribute overwrite, newest first.

    Called from `tearDown`. Leaving a fake `assistant.io.screen` bound on the
    package would hand every later test file in the same process a screen that
    returns `"fakeb64"` -- passing for the wrong reason rather than failing,
    which is the harder direction to notice.
    """
    while _ORIGINAL_ATTRS:
        obj, name, original = _ORIGINAL_ATTRS.pop()
        if original is None:
            if hasattr(obj, name):
                delattr(obj, name)
        else:
            setattr(obj, name, original)


def _install_screen_llm(*, screenshot="fakeb64",
                       vision_responses=None,
                       text_responses=None):
    """Install fake screen + llm modules for Pass 1 and Pass 3 stubbing.

    ``get_vision_response`` now returns ``LLMResult`` objects so each mock
    value is wrapped via ``_wrap_vision`` so callers can do ``.text`` on the
    result.
    """
    screen_mod = types.ModuleType("assistant.io.screen")
    screen_mod.capture_screenshot_base64 = MagicMock(return_value=screenshot)

    llm_mod = types.ModuleType("assistant.llm")
    if vision_responses is None:
        llm_mod.get_vision_response = AsyncMock(return_value=_wrap_vision("YES"))
    elif isinstance(vision_responses, list):
        llm_mod.get_vision_response = AsyncMock(
            side_effect=[_wrap_vision(v) for v in vision_responses]
        )
    else:
        llm_mod.get_vision_response = AsyncMock(return_value=_wrap_vision(vision_responses))

    if text_responses is None:
        text_mock = AsyncMock(return_value='{"completed":[],"new":[]}')
    elif isinstance(text_responses, list):
        text_mock = AsyncMock(side_effect=text_responses)
    else:
        text_mock = AsyncMock(return_value=text_responses)
    llm_mod.get_llm_response = text_mock

    # Both, and the second one is the one that actually works.
    #
    # `agent.py` reaches these as `from ...io import screen` and `from ... import
    # llm`, which read an ATTRIBUTE on the already-imported parent package.
    # Once `assistant.io.screen` is attribute-bound -- and something in the
    # import chain binds it -- `getattr` succeeds and returns the real module,
    # so the `sys.modules` entry below is never consulted. That makes the stub
    # order-dependent: it holds when this file runs alone and stops holding the
    # moment another file imports the real one first, at which point
    # `_confirm_pending_select_todos` takes an actual screenshot of whatever
    # the operator has on screen.
    #
    # The same trap has now cost this tree four separate incidents. The fix is
    # always the same: patch where the importing module will LOOK, not where
    # the name happens to live. `_restore_stubs` puts both back.
    import assistant.io as _io_pkg
    import assistant as _pkg

    # ── Import first, overwrite second. The order is load-bearing. ──
    #
    # `import assistant.llm.contracts` REBINDS `assistant.llm` on the parent
    # package as a side effect -- that is how submodule import works. Doing it
    # after `_pkg.llm = llm_mod` therefore silently reinstates the real
    # package, and pass 1's `from ... import llm` then reaches the real
    # `get_vision_response`: a live Gemini call, from a unit test, on the
    # operator's quota. Observed exactly that way ("Gemini vision error:
    # Incorrect padding" on the fake base64, then a Groq fallback attempt).
    #
    # So every real import this helper needs happens before any attribute is
    # overwritten.
    import assistant.llm.contracts as _contracts

    _ORIGINAL_ATTRS.append((_io_pkg, "screen", getattr(_io_pkg, "screen", None)))
    _ORIGINAL_ATTRS.append((_pkg, "llm", getattr(_pkg, "llm", None)))
    _ORIGINAL_ATTRS.append(
        (_contracts, "ask_for_synthesis", _contracts.ask_for_synthesis))
    _io_pkg.screen = screen_mod
    _pkg.llm = llm_mod

    sys.modules["assistant.io.screen"] = screen_mod

    # `assistant.llm` is deliberately NOT replaced in `sys.modules`, and pass 3
    # is the reason. `agent.py` reaches its text model as
    # `from ...llm.contracts import ask_for_synthesis`, which needs
    # `assistant.llm` to still be a real *package* -- a plain `ModuleType` has
    # no `__path__`, so the import died with
    # `ModuleNotFoundError: 'assistant.llm' is not a package` and took four of
    # this file's tests with it. They had been red on `main` since the call
    # moved off `llm.get_llm_response` onto the contracts wrapper, so the stub
    # was pinning an interface the code no longer had.
    #
    # Repairing the import alone would have been worse than the failure: with
    # `ask_for_synthesis` reachable and unstubbed, pass 3 would have made a
    # real Gemini call, on the operator's quota, in a unit test. So the
    # function is patched on the real contracts module -- the only thing pass 3
    # actually needs -- and the fake `llm` module above continues to serve
    # `from ... import llm` for the vision call in pass 1.
    # The same mock the old `get_llm_response` stub used, so `text_responses`
    # keeps meaning exactly what every caller in this file already expects --
    # and so a test can still assert the text model was or was not consulted.
    _contracts.ask_for_synthesis = text_mock
    llm_mod.ask_for_synthesis = text_mock
    return screen_mod, llm_mod


# ─── 3-Pass Pipeline ───────────────────────────────────────────────────────


class Test3PassPipeline(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()
        cfg.DETERMINISTIC_MATCHING_ENABLED = True

    def tearDown(self):
        ca._task_state.reset()
        cfg.DETERMINISTIC_MATCHING_ENABLED = True
        sys.modules.pop("assistant.io.screen", None)
        # `assistant.llm` is deliberately NOT popped. It is no longer faked in
        # `sys.modules` (see `_install_screen_llm`), so popping it here evicted
        # the REAL package -- and the next test in the class then failed with
        # `KeyError: 'assistant.llm'` reaching for it. `_restore_stubs` undoes
        # everything this file actually installed.
        _restore_stubs()

    def test_rule_T_only_skips_LLM(self):
        """Rule T marks a type-TODO; no LLM call needed."""
        _, llm_mod = _install_screen_llm()
        ca._task_state.set_initial_todos(["Type 'John' in First Name"])
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "John"},
        ]
        results = ["Clicked", 'Typed: "John"']
        marked, added = _run(ca._update_todos_after_batch(actions, results, "fill name"))
        self.assertEqual(marked, 1)
        self.assertEqual(added, 0)
        self.assertTrue(ca._task_state.todo_list[0]["done"])
        # LLM should NOT have been called — rules covered the batch.
        llm_mod.get_llm_response.assert_not_called()

    def test_rule_C_only_skips_LLM(self):
        _, llm_mod = _install_screen_llm()
        ca._task_state.set_initial_todos(["Click 'Submit' button"])
        actions = [{"type": "vision_guided_click", "text": "Submit"}]
        results = ["Clicked 'Submit'"]
        marked, _ = _run(ca._update_todos_after_batch(actions, results, "submit"))
        self.assertEqual(marked, 1)
        llm_mod.get_llm_response.assert_not_called()

    def test_rule_S_defers_LLM_skipped_for_addressed_select(self):
        """Rule S defers a select-TODO to pending; LLM cannot re-mark it."""
        _, llm_mod = _install_screen_llm(
            text_responses='{"completed":[1],"new":[]}'  # LLM tries to mark deferred
        )
        ca._task_state.set_initial_todos(["Select '1-50' from Staff Size dropdown"])
        actions = [{"type": "vision_guided_click", "text": "1-50"}]
        results = ["Clicked '1-50'"]
        marked, _ = _run(ca._update_todos_after_batch(actions, results, "select size"))
        # Select deferred — not done. LLM is NOT called because rules covered
        # the action AND the only TODO is now pending_visual_confirm (not in
        # visible/unresolved set).
        self.assertEqual(marked, 0)
        self.assertFalse(ca._task_state.todo_list[0]["done"])
        self.assertTrue(ca._task_state.todo_list[0]["pending_visual_confirm"])
        llm_mod.get_llm_response.assert_not_called()

    def test_pass1_visual_confirm_runs_when_pending(self):
        """Pass 1 visual-confirm marks a previously-deferred select-TODO."""
        _, llm_mod = _install_screen_llm(vision_responses="YES")
        ca._task_state.set_initial_todos(["Select 'IT' from Industry"])
        # Simulate prior batch that deferred this TODO.
        ca._task_state.todo_list[0]["pending_visual_confirm"] = True
        # Now this batch has no relevant action — but Pass 1 should still
        # run the visual-confirm and mark done if vision says YES.
        actions = [{"type": "screenshot_and_continue"}]
        results = ["SCREENSHOT_AND_CONTINUE"]
        marked, _ = _run(ca._update_todos_after_batch(actions, results, "wait"))
        self.assertEqual(marked, 1)
        self.assertTrue(ca._task_state.todo_list[0]["done"])
        # Vision LLM called for confirm; text LLM not needed (rules cover screenshot).
        llm_mod.get_vision_response.assert_called_once()

    def test_pass3_LLM_only_for_other_kind(self):
        """LLM only sees kind=other TODOs; type/click/select are rule-handled."""
        _, llm_mod = _install_screen_llm(
            text_responses='{"completed":[2],"new":[]}'  # mark the "Submit form" TODO
        )
        ca._task_state.set_initial_todos([
            "Type 'John' in First Name",  # id=1, kind=type
            "Submit form",                  # id=2, kind=other
        ])
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "John"},
            {"type": "vision_guided_click", "text": "Submit"},  # ambiguous — no Click TODO
        ]
        results = ["ok", 'Typed: "John"', "Clicked"]
        marked, _ = _run(ca._update_todos_after_batch(actions, results, "fill"))
        # Rule T marks #1; LLM marks #2.
        self.assertEqual(marked, 2)
        self.assertTrue(all(t["done"] for t in ca._task_state.todo_list))
        # LLM was invoked exactly once for the "other" TODO.
        llm_mod.get_llm_response.assert_called_once()
        # Verify the OPEN TODOS section narrowed scope to TODO #2 only.
        # (The action-context block legitimately mentions all actions including
        # the First Name click — that's there for LLM context, not for marking.)
        prompt_arg = llm_mod.get_llm_response.call_args[0][0]
        self.assertIn("Submit form", prompt_arg)
        self.assertIn("[2]", prompt_arg)
        # Extract the TODOS-only section and verify First Name's TODO isn't there.
        todos_section = prompt_arg.split("OPEN TODOS YOU MAY MARK:", 1)[1]
        todos_section = todos_section.split("\n\n", 1)[0]
        self.assertNotIn("[1]", todos_section)
        self.assertNotIn("First Name", todos_section)

    def test_pass3_guard_rejects_marks_outside_visible_set(self):
        """LLM hallucinating a TODO id outside the visible set must NOT mark it."""
        _, _ = _install_screen_llm(
            text_responses='{"completed":[1, 2],"new":[]}'  # tries to mark BOTH
        )
        ca._task_state.set_initial_todos([
            "Select '1-50' from Staff Size dropdown",  # id=1, kind=select
            "Submit form",                              # id=2, kind=other
        ])
        # Defer #1 via Rule S
        ca._task_state.todo_list[0]["pending_visual_confirm"] = True
        actions = [{"type": "screenshot_and_continue"}]
        results = ["SCREENSHOT_AND_CONTINUE"]
        # We need the LLM to fire (Pass 3) for the "Submit form" TODO. Rules
        # don't address screenshot_and_continue, so Pass 3 will run with
        # only id=2 visible.
        # But we need vision_responses=NO so the deferred #1 doesn't get
        # confirmed and stays out of unresolved.
        # Via the package attribute, which is where `agent.py` looks -- the
        # fake is bound there, not in `sys.modules`.
        import assistant as _pkg
        _pkg.llm.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text="NO"))
        marked, _ = _run(ca._update_todos_after_batch(actions, results, "x"))
        # LLM tried to mark BOTH 1 and 2. Guard should only allow 2 (since
        # 1 is pending_visual_confirm and not in visible set).
        self.assertEqual(marked, 1)
        self.assertFalse(ca._task_state.todo_list[0]["done"])  # protected
        self.assertTrue(ca._task_state.todo_list[1]["done"])

    def test_pass3_can_add_new_todos(self):
        """LLM-discovered new TODOs (cascading dropdowns) get appended."""
        _, _ = _install_screen_llm(
            text_responses='{"completed":[],"new":["Type \'NY\' in City"]}'
        )
        ca._task_state.set_initial_todos(["Submit form"])  # kind=other
        actions = [{"type": "vision_guided_click", "text": "State"}]
        results = ["Clicked 'State'"]
        marked, added = _run(ca._update_todos_after_batch(actions, results, "x"))
        self.assertEqual(marked, 0)
        self.assertEqual(added, 1)
        self.assertEqual(ca._task_state.todo_list[-1]["task"], "Type 'NY' in City")
        # The new TODO was classified by add_todo's call to _make_todo_dict.
        self.assertEqual(ca._task_state.todo_list[-1]["kind"], "type")


# ─── Kill switch ───────────────────────────────────────────────────────────


class TestKillSwitch(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()
        cfg.DETERMINISTIC_MATCHING_ENABLED = True
        sys.modules.pop("assistant.io.screen", None)
        # `assistant.llm` is deliberately NOT popped. It is no longer faked in
        # `sys.modules` (see `_install_screen_llm`), so popping it here evicted
        # the REAL package -- and the next test in the class then failed with
        # `KeyError: 'assistant.llm'` reaching for it. `_restore_stubs` undoes
        # everything this file actually installed.
        _restore_stubs()

    def test_disabled_falls_back_to_pe1_LLM_only(self):
        """When the kill-switch is False, Passes 1+2 are skipped — pure LLM."""
        cfg.DETERMINISTIC_MATCHING_ENABLED = False
        _, llm_mod = _install_screen_llm(
            text_responses='{"completed":[1],"new":[]}'
        )
        ca._task_state.set_initial_todos(["Type 'John' in First Name"])
        # Same action+anchor that would normally hit Rule T deterministically.
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "John"},
        ]
        results = ["Clicked", 'Typed: "John"']
        marked, _ = _run(ca._update_todos_after_batch(actions, results, "fill"))
        self.assertEqual(marked, 1)
        # Vision-LLM must NOT have been called (Pass 1 skipped).
        llm_mod.get_vision_response.assert_not_called()
        # Text-LLM must have been called (Pass 3 with all TODOs visible).
        llm_mod.get_llm_response.assert_called_once()


# ─── ESC race fix (audit #1/#6) ────────────────────────────────────────────


class TestEscRaceFix(unittest.TestCase):
    """`run_computer_task` must not touch the ESC monitor lifecycle.

    Asserted **by reading the source, not by running it**, and both halves of
    that sentence were learned the hard way.

    *What it used to assert.* The ordering
    `stop_esc_monitor -> reset_abort -> start_esc_monitor -> inner`. That
    ordering was deliberately removed: the monitor is a session-level singleton
    owned by `main.py`, and restarting it per task killed ESC abort for every
    later task in the session. So the test had been red on an arrangement
    nobody wants back.

    *Why it no longer executes anything.* It called the real
    `run_computer_task` with `patch.object(ca, "_run_computer_task_inner", ...)`
    -- but `ca` is `assistant.automation.vision`, the PACKAGE, while
    `run_computer_task` resolves that name as a module global inside
    `agent.py`. The patch bound an attribute nothing reads, so **the real
    vision agent loop ran**: 84 seconds, nineteen real screenshots, and a loop
    that drives `pyautogui`. It also minimized every terminal window on the
    machine first, via an unmocked `pygetwindow` in the entry point's own body.
    It did this while the operator was working.

    Two separate lessons, and the second is the general one:

    1. `patch.object` on a package is not `patch.object` on the module that
       defines the function. A name is resolved in the namespace where the
       *caller* was compiled.
    2. A test whose only claim is "this function does not call those three
       functions" does not need to call it. Executing a real entry point to
       observe an absence buys nothing and costs everything -- the thing being
       measured is a property of the code, so it is read off the code.

    Mutation: add a `start_esc_monitor()` call to `run_computer_task` and the
    first test reds.
    """

    def _entry_point_source(self) -> str:
        """`run_computer_task`'s body, and only its body.

        Sliced by AST rather than by line numbers so that editing anything
        above or below it cannot silently change what is being measured, and so
        that a rename fails loudly instead of matching nothing.
        """
        import assistant.automation.vision.agent as agent_mod
        src = pathlib.Path(agent_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                    and node.name == "run_computer_task"):
                body = ast.get_source_segment(src, node)
                self.assertTrue(body, "AST returned no source for the entry point")
                return body
        self.fail("run_computer_task no longer exists in agent.py -- if it was "
                  "renamed, move this check with it")

    def test_the_entry_point_does_not_touch_the_esc_monitor(self):
        """The monitor is a session singleton. A task that stops and restarts
        it kills ESC abort for every later task in that session, which is a
        silent failure: the first task aborts fine and nobody tries a second
        until it matters."""
        body = self._entry_point_source()
        tree = ast.parse(textwrap.dedent(body))
        called = {
            ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)
        }
        self.assertTrue(called, "walked nothing -- the body parsed to no calls")

        forbidden = {"stop_esc_monitor", "start_esc_monitor", "reset_abort"}
        self.assertEqual(
            called & forbidden, set(),
            f"the entry point manages session ESC/abort state: "
            f"{sorted(called & forbidden)}. That lifecycle belongs to main.py."
        )

    def test_the_entry_point_delegates_to_the_inner_loop_once(self):
        """The original double-reset check, kept as what it actually pins: one
        delegation, so no path runs the task twice."""
        body = self._entry_point_source()
        self.assertEqual(
            body.count("_run_computer_task_inner("), 1,
            "the entry point calls the inner loop other than exactly once"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
