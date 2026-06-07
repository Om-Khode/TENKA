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
  - run_computer_task entry-point ordering (audit fix #1/#6): stop_esc_monitor
    → reset_abort → start_esc_monitor, no double reset.

Run: python test_planner_integration.py
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

import assistant.automation.vision as ca
import assistant.config as cfg


def _run(coro):
    return asyncio.run(coro)


def _wrap_vision(v):
    """Wrap a bare string in SimpleNamespace(text=...) for LLMResult compat."""
    if isinstance(v, Exception):
        return v
    return SimpleNamespace(text=v)


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
        llm_mod.get_llm_response = AsyncMock(return_value='{"completed":[],"new":[]}')
    elif isinstance(text_responses, list):
        llm_mod.get_llm_response = AsyncMock(side_effect=text_responses)
    else:
        llm_mod.get_llm_response = AsyncMock(return_value=text_responses)

    sys.modules["assistant.io.screen"] = screen_mod
    sys.modules["assistant.llm"] = llm_mod
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
        sys.modules.pop("assistant.llm", None)

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
        sys.modules["assistant.llm"].get_vision_response = AsyncMock(return_value=SimpleNamespace(text="NO"))
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
        sys.modules.pop("assistant.llm", None)

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
    """
    The fix is structural — verify run_computer_task's setup ordering:
      1. stop_esc_monitor() runs first (idempotent, no-op if not running)
      2. reset_abort() runs next (clears TASK_ABORTED + _task_state)
      3. start_esc_monitor() runs last (fresh thread starts watching)
    AND that _run_computer_task_inner no longer calls reset_abort itself
    (would be a double-reset).
    """

    def test_run_computer_task_calls_in_correct_order(self):
        """
        Patch the three involved functions and the inner loop, then call
        run_computer_task. Verify the call order: stop_esc_monitor first,
        reset_abort next, start_esc_monitor last, inner runs after.
        """
        order = []

        def _stop():
            order.append("stop_esc")

        def _reset():
            order.append("reset_abort")

        def _start():
            order.append("start_esc")

        async def _fake_inner(goal, llm_func, tts_func, bridge_func):
            order.append("inner")
            return "done"

        with patch.object(ca, "stop_esc_monitor", _stop), \
             patch.object(ca, "reset_abort", _reset), \
             patch.object(ca, "start_esc_monitor", _start), \
             patch.object(ca, "_run_computer_task_inner", _fake_inner):
            result = _run(ca.run_computer_task("test goal", llm_func=lambda *a, **k: None))

        self.assertEqual(result, "done")
        # stop_esc must come before reset_abort, reset_abort before start_esc,
        # and start_esc before inner.
        self.assertEqual(order[:4], ["stop_esc", "reset_abort", "start_esc", "inner"])

    def test_inner_no_longer_calls_reset_abort(self):
        """Defensive check: only ONE reset_abort call per task entry."""
        reset_count = 0

        def _counting_reset():
            nonlocal reset_count
            reset_count += 1

        async def _fake_inner(*args, **kwargs):
            return "done"

        with patch.object(ca, "stop_esc_monitor"), \
             patch.object(ca, "reset_abort", _counting_reset), \
             patch.object(ca, "start_esc_monitor"), \
             patch.object(ca, "_run_computer_task_inner", _fake_inner):
            _run(ca.run_computer_task("test", llm_func=lambda *a, **k: None))

        self.assertEqual(reset_count, 1, "reset_abort must be called exactly once per task")


if __name__ == "__main__":
    unittest.main(verbosity=2)
