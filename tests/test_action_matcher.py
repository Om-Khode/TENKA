"""
test_action_matcher.py — planner-vision part 2: action matcher + visual confirm.

Covers:
  - _action_failed: failure prefixes, empty, normal text
  - _texts_overlap: substring either direction, fuzz threshold, empty
  - _is_yes_answer: YES, yes., Y, NO, garbage, mixed prefix
  - _find_focus_anchor_in_batch: click-target overlap, tab anchor,
    keyboard_hotkey with tab, no anchor before, broken chain by other action
  - _match_action_to_todo:
    * Rule T happy (type after click-focus, type after tab)
    * Rule T miss (no anchor, value mismatch, failed action)
    * Rule S happy defer (target matches value, target matches field)
    * Rule S double-defer protection (already pending — no re-set)
    * Rule S precedence over Rule C (same target, both kinds present)
    * Rule C happy (vision_guided_click, find_and_click_text, double-click)
    * Rule C miss (target mismatch, failed result)
  - _match_actions_to_todos: orchestrator returns set + count, mixed batch
  - _confirm_pending_select_todos:
    * No pending → no-op (no LLM calls)
    * YES on first call → mark done, strikes reset, returns (1, 0)
    * NO + strikes < 3 → increment, leave pending
    * NO × 3 + fallback YES → mark done, fallback_count++
    * NO × 3 + fallback NO → give up, clear pending, fallback_count++
    * Screenshot capture failure → no state change, returns (0, 0)
    * LLM unavailable → fail-open, no state change

Run: python test_action_matcher.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca


def _run(coro):
    return asyncio.run(coro)


def _wrap_vision(v):
    """Wrap a bare string in SimpleNamespace(text=...) for LLMResult compat."""
    if isinstance(v, Exception):
        return v
    return SimpleNamespace(text=v)


def _install_screen_llm(*, screenshot="fakeb64", vision_responses=None):
    """
    Install fake screen + llm modules. vision_responses can be:
      - a single string (returned every call, wrapped in SimpleNamespace)
      - a list (one wrapped response per call, in order)
      - an Exception (raised every call)
    ``get_vision_response`` now returns ``LLMResult`` objects so each value
    is wrapped via ``_wrap_vision`` so callers can do ``.text`` on the result.
    """
    screen_mod = types.ModuleType("assistant.io.screen")
    screen_mod.capture_screenshot_base64 = MagicMock(return_value=screenshot)

    llm_mod = types.ModuleType("assistant.llm")
    if isinstance(vision_responses, Exception):
        llm_mod.get_vision_response = AsyncMock(side_effect=vision_responses)
    elif isinstance(vision_responses, list):
        llm_mod.get_vision_response = AsyncMock(
            side_effect=[_wrap_vision(v) for v in vision_responses]
        )
    else:
        llm_mod.get_vision_response = AsyncMock(
            return_value=_wrap_vision(vision_responses or "YES")
        )

    sys.modules["assistant.io.screen"] = screen_mod
    sys.modules["assistant.llm"] = llm_mod
    return screen_mod, llm_mod


def _stub_type_todo(field: str, value: str) -> dict:
    """Add a classified type-TODO to _task_state and return its dict."""
    ca._task_state.add_todo(f"Type '{value}' in {field}")
    return ca._task_state.todo_list[-1]


def _stub_select_todo(field: str, value: str, *, pending=False, strikes=0) -> dict:
    ca._task_state.add_todo(f"Select '{value}' from {field} dropdown")
    t = ca._task_state.todo_list[-1]
    t["pending_visual_confirm"] = pending
    t["confirm_strikes"] = strikes
    return t


def _stub_click_todo(target: str) -> dict:
    ca._task_state.add_todo(f"Click '{target}' button")
    return ca._task_state.todo_list[-1]


# ─── _action_failed ────────────────────────────────────────────────────────


class TestActionFailed(unittest.TestCase):
    def test_failed_prefix(self):
        self.assertTrue(ca._action_failed("Failed: timeout"))

    def test_error_prefix(self):
        self.assertTrue(ca._action_failed("Error: bad coords"))

    def test_aborted_prefix(self):
        self.assertTrue(ca._action_failed("ABORTED: user pressed esc"))

    def test_aborted_wrong_focus_with_aborted(self):
        # Realistic message: "ACTION ABORTED_WRONG_FOCUS expected Notepad"
        self.assertTrue(ca._action_failed("ACTION ABORTED_WRONG_FOCUS expected Notepad"))

    def test_normal_success_message(self):
        self.assertFalse(ca._action_failed("Clicked 'Submit' at (100,200)"))

    def test_empty_string(self):
        self.assertTrue(ca._action_failed(""))

    def test_non_string(self):
        self.assertTrue(ca._action_failed(None))


# ─── _texts_overlap ────────────────────────────────────────────────────────


class TestTextsOverlap(unittest.TestCase):
    def test_substring_a_in_b(self):
        self.assertTrue(ca._texts_overlap("OK", "click OK button"))

    def test_substring_b_in_a(self):
        self.assertTrue(ca._texts_overlap("Schedule a Demo button", "Schedule a Demo"))

    def test_fuzz_above_threshold(self):
        # "Submit" vs "Sumbit" (typo)
        self.assertTrue(ca._texts_overlap("Submit", "Sumbit"))

    def test_no_overlap_below_threshold(self):
        self.assertFalse(ca._texts_overlap("Cancel", "Submit"))

    def test_empty(self):
        self.assertFalse(ca._texts_overlap("", "Submit"))
        self.assertFalse(ca._texts_overlap("Submit", ""))

    def test_canonicalization_strips_quotes(self):
        self.assertTrue(ca._texts_overlap("'Submit'", "submit"))


# ─── _is_yes_answer ────────────────────────────────────────────────────────


class TestIsYesAnswer(unittest.TestCase):
    def test_yes(self):
        self.assertTrue(ca._is_yes_answer("YES"))

    def test_yes_lowercase(self):
        self.assertTrue(ca._is_yes_answer("yes"))

    def test_yes_with_period(self):
        self.assertTrue(ca._is_yes_answer("Yes."))

    def test_yes_with_explanation(self):
        self.assertTrue(ca._is_yes_answer("YES — the value is visible"))

    def test_y_alone(self):
        self.assertTrue(ca._is_yes_answer("Y"))

    def test_no(self):
        self.assertFalse(ca._is_yes_answer("NO"))

    def test_no_with_explanation(self):
        self.assertFalse(ca._is_yes_answer("No, the field is empty"))

    def test_garbage(self):
        self.assertFalse(ca._is_yes_answer("maybe"))

    def test_empty(self):
        self.assertFalse(ca._is_yes_answer(""))

    def test_non_string(self):
        self.assertFalse(ca._is_yes_answer(None))

    def test_leading_punctuation_stripped(self):
        self.assertTrue(ca._is_yes_answer("**YES**"))
        self.assertFalse(ca._is_yes_answer("**NO**"))


# ─── _find_focus_anchor_in_batch ───────────────────────────────────────────


class TestFindFocusAnchor(unittest.TestCase):
    def test_click_overlap_anchor(self):
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "John"},
        ]
        anchor = ca._find_focus_anchor_in_batch(actions, 1, "First Name")
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["type"], "vision_guided_click")

    def test_tab_anchor(self):
        actions = [
            {"type": "keyboard_press", "key": "tab"},
            {"type": "keyboard_type", "text": "Doe"},
        ]
        anchor = ca._find_focus_anchor_in_batch(actions, 1, "Last Name")
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["key"], "tab")

    def test_keyboard_hotkey_with_tab(self):
        actions = [
            {"type": "keyboard_hotkey", "keys": ["ctrl", "tab"]},
            {"type": "keyboard_type", "text": "X"},
        ]
        self.assertIsNotNone(ca._find_focus_anchor_in_batch(actions, 1, "any"))

    def test_no_actions_before(self):
        actions = [{"type": "keyboard_type", "text": "X"}]
        self.assertIsNone(ca._find_focus_anchor_in_batch(actions, 0, "Field"))

    def test_click_on_unrelated_target_breaks_chain(self):
        actions = [
            {"type": "vision_guided_click", "text": "Random Button"},
            {"type": "keyboard_type", "text": "Doe"},
        ]
        self.assertIsNone(ca._find_focus_anchor_in_batch(actions, 1, "Last Name"))

    def test_non_tab_keyboard_press_breaks_chain(self):
        actions = [
            {"type": "keyboard_press", "key": "f5"},
            {"type": "keyboard_type", "text": "X"},
        ]
        self.assertIsNone(ca._find_focus_anchor_in_batch(actions, 1, "F"))

    def test_screenshot_and_continue_skipped(self):
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "screenshot_and_continue"},
            {"type": "keyboard_type", "text": "John"},
        ]
        anchor = ca._find_focus_anchor_in_batch(actions, 2, "First Name")
        self.assertIsNotNone(anchor)


# ─── _match_action_to_todo: Rule T ─────────────────────────────────────────


class TestMatchRuleType(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def test_happy_with_click_anchor(self):
        _stub_type_todo("First Name", "John")
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "John"},
        ]
        results = ["Clicked 'First Name' at (100,200)", 'Typed: "John"']
        tid, marked = ca._match_action_to_todo(actions[1], results[1], 1, actions)
        self.assertIsNotNone(tid)
        self.assertTrue(marked)
        self.assertTrue(ca._task_state.todo_list[0]["done"])

    def test_happy_with_tab_anchor(self):
        _stub_type_todo("Last Name", "Doe")
        actions = [
            {"type": "keyboard_press", "key": "tab"},
            {"type": "keyboard_type", "text": "Doe"},
        ]
        results = ["Pressed key: tab", 'Typed: "Doe"']
        tid, marked = ca._match_action_to_todo(actions[1], results[1], 1, actions)
        self.assertTrue(marked)

    def test_value_mismatch_no_match(self):
        _stub_type_todo("First Name", "John")
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "Jane"},  # wrong value
        ]
        results = ["Clicked", 'Typed: "Jane"']
        tid, marked = ca._match_action_to_todo(actions[1], results[1], 1, actions)
        self.assertIsNone(tid)
        self.assertFalse(ca._task_state.todo_list[0]["done"])

    def test_no_anchor_no_match(self):
        _stub_type_todo("First Name", "John")
        actions = [{"type": "keyboard_type", "text": "John"}]  # no preceding focus
        tid, marked = ca._match_action_to_todo(actions[0], 'Typed: "John"', 0, actions)
        self.assertIsNone(tid)
        self.assertFalse(ca._task_state.todo_list[0]["done"])

    def test_failed_action_no_match(self):
        _stub_type_todo("First Name", "John")
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "John"},
        ]
        tid, marked = ca._match_action_to_todo(actions[1], "Failed: keyboard error", 1, actions)
        self.assertIsNone(tid)


# ─── _match_action_to_todo: Rule S ─────────────────────────────────────────


class TestMatchRuleSelect(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def test_click_on_value_defers(self):
        _stub_select_todo("Staff Size", "1-50")
        actions = [{"type": "vision_guided_click", "text": "1-50"}]
        tid, marked = ca._match_action_to_todo(
            actions[0], "Clicked '1-50' at (100,200)", 0, actions
        )
        self.assertIsNotNone(tid)
        self.assertFalse(marked)
        self.assertTrue(ca._task_state.todo_list[0]["pending_visual_confirm"])
        self.assertFalse(ca._task_state.todo_list[0]["done"])

    def test_click_on_field_defers(self):
        _stub_select_todo("Industry", "IT")
        actions = [{"type": "vision_guided_click", "text": "Industry"}]
        tid, marked = ca._match_action_to_todo(
            actions[0], "Clicked 'Industry' at (100,200)", 0, actions
        )
        self.assertIsNotNone(tid)
        self.assertFalse(marked)
        self.assertTrue(ca._task_state.todo_list[0]["pending_visual_confirm"])

    def test_already_pending_not_re_deferred(self):
        # A select-TODO already pending from an earlier batch should NOT be
        # re-deferred (would reset its strikes counter erroneously).
        _stub_select_todo("Industry", "IT", pending=True, strikes=2)
        actions = [{"type": "vision_guided_click", "text": "Industry"}]
        tid, marked = ca._match_action_to_todo(actions[0], "Clicked", 0, actions)
        # Skipped — no re-defer; strikes preserved.
        self.assertIsNone(tid)
        self.assertEqual(ca._task_state.todo_list[0]["confirm_strikes"], 2)

    def test_select_takes_precedence_over_click_on_same_target(self):
        # Both a click-TODO ("Click 'IT'") and a select-TODO ("Select 'IT'")
        # could match a click on 'IT'. Rule S must run first.
        _stub_click_todo("IT")
        _stub_select_todo("Industry", "IT")
        actions = [{"type": "vision_guided_click", "text": "IT"}]
        tid, marked = ca._match_action_to_todo(actions[0], "Clicked 'IT'", 0, actions)
        # Rule S match → defer; click-TODO untouched.
        self.assertEqual(tid, ca._task_state.todo_list[1]["id"])
        self.assertFalse(marked)
        self.assertFalse(ca._task_state.todo_list[0]["done"])  # click-TODO untouched
        self.assertTrue(ca._task_state.todo_list[1]["pending_visual_confirm"])


# ─── _match_action_to_todo: Rule C ─────────────────────────────────────────


class TestMatchRuleClick(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def test_vision_guided_click_match(self):
        _stub_click_todo("Submit")
        actions = [{"type": "vision_guided_click", "text": "Submit"}]
        tid, marked = ca._match_action_to_todo(actions[0], "Clicked 'Submit'", 0, actions)
        self.assertTrue(marked)
        self.assertTrue(ca._task_state.todo_list[0]["done"])

    def test_find_and_click_text_match(self):
        _stub_click_todo("Save")
        actions = [{"type": "find_and_click_text", "text": "Save"}]
        tid, marked = ca._match_action_to_todo(actions[0], "Found 'Save'", 0, actions)
        self.assertTrue(marked)

    def test_double_click_match(self):
        _stub_click_todo("song.mp3")
        actions = [{"type": "find_and_double_click_text", "text": "song.mp3"}]
        tid, marked = ca._match_action_to_todo(actions[0], "Double-clicked", 0, actions)
        self.assertTrue(marked)

    def test_target_mismatch_no_match(self):
        _stub_click_todo("Submit")
        actions = [{"type": "vision_guided_click", "text": "Cancel"}]
        tid, marked = ca._match_action_to_todo(actions[0], "Clicked", 0, actions)
        self.assertIsNone(tid)
        self.assertFalse(ca._task_state.todo_list[0]["done"])

    def test_failed_click_no_match(self):
        _stub_click_todo("Submit")
        actions = [{"type": "vision_guided_click", "text": "Submit"}]
        tid, marked = ca._match_action_to_todo(actions[0], "Failed: not found", 0, actions)
        self.assertIsNone(tid)
        self.assertFalse(ca._task_state.todo_list[0]["done"])


# ─── _match_actions_to_todos: orchestrator ────────────────────────────────


class TestMatchActionsToTodos(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def test_mixed_batch(self):
        _stub_type_todo("First Name", "John")
        _stub_click_todo("Submit")
        actions = [
            {"type": "vision_guided_click", "text": "First Name"},
            {"type": "keyboard_type", "text": "John"},
            {"type": "vision_guided_click", "text": "Submit"},
        ]
        results = ["ok", 'Typed: "John"', "Clicked Submit"]
        addressed, marked = ca._match_actions_to_todos(actions, results)
        self.assertEqual(marked, 2)
        self.assertEqual(len(addressed), 2)
        self.assertTrue(all(t["done"] for t in ca._task_state.todo_list))

    def test_deferred_select_addressed_but_not_marked(self):
        _stub_select_todo("Industry", "IT")
        actions = [{"type": "vision_guided_click", "text": "IT"}]
        results = ["Clicked 'IT'"]
        addressed, marked = ca._match_actions_to_todos(actions, results)
        self.assertEqual(len(addressed), 1)
        self.assertEqual(marked, 0)


# ─── _confirm_pending_select_todos ────────────────────────────────────────


class TestConfirmPendingSelect(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()
        sys.modules.pop("assistant.io.screen", None)
        sys.modules.pop("assistant.llm", None)

    def test_no_pending_short_circuits(self):
        _, llm_mod = _install_screen_llm(vision_responses="YES")
        _stub_select_todo("Industry", "IT")  # not pending
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        self.assertEqual((confirmed, fb), (0, 0))
        llm_mod.get_vision_response.assert_not_called()

    def test_yes_marks_done(self):
        _install_screen_llm(vision_responses="YES")
        _stub_select_todo("Industry", "IT", pending=True, strikes=0)
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        self.assertEqual(confirmed, 1)
        self.assertEqual(fb, 0)
        t = ca._task_state.todo_list[0]
        self.assertTrue(t["done"])
        self.assertFalse(t["pending_visual_confirm"])
        self.assertEqual(t["confirm_strikes"], 0)

    def test_no_increments_strike(self):
        _install_screen_llm(vision_responses="NO")
        _stub_select_todo("Industry", "IT", pending=True, strikes=0)
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        self.assertEqual(confirmed, 0)
        self.assertEqual(fb, 0)
        t = ca._task_state.todo_list[0]
        self.assertFalse(t["done"])
        self.assertTrue(t["pending_visual_confirm"])
        self.assertEqual(t["confirm_strikes"], 1)

    def test_three_strikes_fallback_yes_marks_done(self):
        # Sequence: strike#1 NO, then fallback YES on the same call sequence.
        # _confirm_pending_select_todos makes ONE strict call per pending TODO,
        # then if strikes hit 3, fires ONE fallback call.
        _install_screen_llm(vision_responses=["NO", "YES"])
        _stub_select_todo("Industry", "IT", pending=True, strikes=2)
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        # Strict NO took strikes from 2 → 3, then fallback YES marked done.
        self.assertEqual(confirmed, 0)  # strict pass didn't confirm
        self.assertEqual(fb, 1)
        t = ca._task_state.todo_list[0]
        self.assertTrue(t["done"])
        self.assertFalse(t["pending_visual_confirm"])
        self.assertEqual(ca._task_state.confirm_fallback_count, 1)

    def test_three_strikes_fallback_no_marks_abandoned(self):
        # Fix A (2026-04-26): contract changed. Previous behaviour was
        # `done=False stays not-done — planner re-encounters` which caused
        # an infinite retry loop on Truein form. New behaviour: trust the
        # action signature, mark done=True, set confirm_abandoned=True for
        # honest disclosure in the final TTS reply.
        _install_screen_llm(vision_responses=["NO", "NO"])
        _stub_select_todo("Industry", "IT", pending=True, strikes=2)
        # Bump batch_idx so we can verify mark_todo_done stamped it.
        ca._task_state.batch_idx = 7
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        # Strict-NO didn't confirm; fallback fired and abandoned.
        self.assertEqual(confirmed, 0)
        self.assertEqual(fb, 1)
        t = ca._task_state.todo_list[0]
        self.assertTrue(t["done"], "Fix A: abandoned TODOs are marked done")
        self.assertTrue(t["confirm_abandoned"], "Fix A: confirm_abandoned flag set")
        self.assertFalse(t["pending_visual_confirm"], "pending cleared")
        self.assertEqual(t["confirm_strikes"], 0, "strikes reset on abandonment")
        self.assertEqual(t["batch_marked_done"], 7,
                         "batch_marked_done stamped via mark_todo_done")
        # Telemetry counters
        self.assertEqual(ca._task_state.confirm_fallback_count, 1)
        self.assertEqual(ca._task_state.confirm_abandoned_count, 1,
                         "Fix A: dedicated abandoned counter")

    def test_screenshot_failure_no_state_change(self):
        _install_screen_llm(screenshot=None, vision_responses="YES")
        _stub_select_todo("Industry", "IT", pending=True, strikes=1)
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        self.assertEqual((confirmed, fb), (0, 0))
        t = ca._task_state.todo_list[0]
        self.assertEqual(t["confirm_strikes"], 1)
        self.assertTrue(t["pending_visual_confirm"])

    def test_llm_unavailable_no_state_change(self):
        _install_screen_llm(vision_responses="__LLM_UNAVAILABLE__")
        _stub_select_todo("Industry", "IT", pending=True, strikes=1)
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        self.assertEqual((confirmed, fb), (0, 0))
        t = ca._task_state.todo_list[0]
        self.assertEqual(t["confirm_strikes"], 1)
        self.assertTrue(t["pending_visual_confirm"])

    def test_llm_crash_fail_open(self):
        _install_screen_llm(vision_responses=RuntimeError("boom"))
        _stub_select_todo("Industry", "IT", pending=True, strikes=1)
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        self.assertEqual((confirmed, fb), (0, 0))
        t = ca._task_state.todo_list[0]
        self.assertTrue(t["pending_visual_confirm"])

    def test_multiple_pending_one_screenshot(self):
        # Two pending TODOs, both get confirmed YES — only one screenshot taken.
        screen_mod, llm_mod = _install_screen_llm(vision_responses="YES")
        _stub_select_todo("Industry", "IT", pending=True, strikes=0)
        _stub_select_todo("Staff Size", "1-50", pending=True, strikes=0)
        confirmed, fb = _run(ca._confirm_pending_select_todos())
        self.assertEqual(confirmed, 2)
        # Screenshot is captured ONCE per pass (not once per TODO)
        self.assertEqual(screen_mod.capture_screenshot_base64.call_count, 1)
        # LLM called twice (once per TODO)
        self.assertEqual(llm_mod.get_vision_response.call_count, 2)


# ─── Visual-confirm prompt softening (2026-04-26) ──────────────────────────
#
# After the live test where Gemini Flash strict yes/no produced false-NOs
# on dropdown selections (chevron-styled value confused as "not visible"),
# the prompt was softened to explicitly accept dropdown-styled values inside
# the control's value area while still rejecting empty / placeholder /
# open-menu-list states. These tests verify the prompt body has the
# expected language so future edits don't silently re-tighten it.


# ─── Fix A — abandoned-confirm state (2026-04-26) ─────────────────────────
#
# After 3 strict NOs + permissive fallback NO, the TODO is marked done with
# confirm_abandoned=True (trusting the action signature over unreliable
# vision). Stops the infinite-retry loop observed when the planner kept
# re-attempting a deferred TODO that visual-confirm couldn't ratify.


class TestFixAAbandonedConfirm(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()
        sys.modules.pop("assistant.io.screen", None)
        sys.modules.pop("assistant.llm", None)

    async def test_abandoned_todo_not_redeferred_by_rule_s(self):
        # An abandoned TODO is done=True, so Rule S's `if todo["done"]`
        # filter must skip it on subsequent clicks (no re-defer would also
        # mean no recovery anyway, but the test guards against future
        # regressions where the filter is loosened).
        ca._task_state.set_initial_todos(["Select '1-50' from Staff Size dropdown"])
        t = ca._task_state.todo_list[0]
        t["done"] = True
        t["confirm_abandoned"] = True
        # Now run a click that WOULD have matched Rule S if the TODO were open.
        actions = [{"type": "vision_guided_click", "text": "1-50"}]
        results = ["Clicked '1-50' at (100,200)"]
        addressed, marked = ca._match_actions_to_todos(actions, results)
        # Rule S didn't fire (done filter). Rule C didn't either (kind=select,
        # not click). So no addressed_ids and no new marks.
        self.assertEqual(addressed, set())
        self.assertEqual(marked, 0)
        # Critical: the abandoned TODO state is sticky.
        self.assertTrue(t["done"])
        self.assertTrue(t["confirm_abandoned"])
        self.assertFalse(t["pending_visual_confirm"])

    async def test_all_todos_done_treats_abandoned_as_done(self):
        # all_todos_done() must return True even when some TODOs are
        # abandoned-done — otherwise the verifier sanity check at end-of-loop
        # never fires and the agent never finishes.
        ca._task_state.set_initial_todos([
            "Type 'John' in First Name",
            "Select '1-50' from Staff Size dropdown",
        ])
        ca._task_state.todo_list[0]["done"] = True  # normal-done
        ca._task_state.todo_list[1]["done"] = True
        ca._task_state.todo_list[1]["confirm_abandoned"] = True  # abandoned-done
        self.assertTrue(ca._task_state.all_todos_done(),
                        "Fix A: abandoned counts as done")

    async def test_progress_str_marks_abandoned_distinctly_for_humans(self):
        # The annotation is for debug.log readability — planner sees ✓
        # either way (the loop-prevention property of Fix A).
        ca._task_state.set_initial_todos([
            "Type 'John' in First Name",
            "Select '1-50' from Staff Size dropdown",
        ])
        ca._task_state.todo_list[0]["done"] = True  # normal
        ca._task_state.todo_list[1]["done"] = True
        ca._task_state.todo_list[1]["confirm_abandoned"] = True
        out = ca._task_state.todo_progress_str()
        self.assertIn("✓ Type 'John' in First Name", out)
        self.assertIn("(unconfirmed)", out)
        # Counts as done in the header
        self.assertIn("2 of 2 done", out)
        # The annotation only appears next to the abandoned line, not the normal
        normal_line = [ln for ln in out.split("\n") if "First Name" in ln][0]
        self.assertNotIn("unconfirmed", normal_line)

    async def test_strict_yes_does_not_set_abandoned(self):
        # Regression: the strict YES path must leave confirm_abandoned=False.
        _install_screen_llm(vision_responses="YES")
        _stub_select_todo("Industry", "IT", pending=True, strikes=0)
        confirmed, fb = await ca._confirm_pending_select_todos()
        self.assertEqual(confirmed, 1)
        t = ca._task_state.todo_list[0]
        self.assertTrue(t["done"])
        self.assertFalse(t["confirm_abandoned"],
                         "strict YES is genuine confirmation, not abandonment")
        self.assertEqual(ca._task_state.confirm_abandoned_count, 0)

    async def test_fallback_yes_does_not_set_abandoned(self):
        # Regression: fallback-YES (permissive vision saved us) is genuine.
        _install_screen_llm(vision_responses=["NO", "YES"])
        _stub_select_todo("Industry", "IT", pending=True, strikes=2)
        confirmed, fb = await ca._confirm_pending_select_todos()
        self.assertEqual(fb, 1)
        t = ca._task_state.todo_list[0]
        self.assertTrue(t["done"])
        self.assertFalse(t["confirm_abandoned"],
                         "fallback YES is permissive confirmation, not abandonment")
        # confirm_fallback_count goes up (we used the fallback) but
        # confirm_abandoned_count stays 0 (vision DID save us this time).
        self.assertEqual(ca._task_state.confirm_fallback_count, 1)
        self.assertEqual(ca._task_state.confirm_abandoned_count, 0)

    async def test_abandoned_field_summary_truncates_at_2(self):
        # TTS suffix builder caps the field list to keep replies ≤200 chars.
        ca._task_state.set_initial_todos([
            "Select 'A' from Field One dropdown",
            "Select 'B' from Field Two dropdown",
            "Select 'C' from Field Three dropdown",
            "Select 'D' from Field Four dropdown",
        ])
        for t in ca._task_state.todo_list:
            t["done"] = True
            t["confirm_abandoned"] = True
        summary = ca._task_state.abandoned_field_summary(max_fields=2)
        # First two field names + "and 2 more"
        self.assertIn("Field One", summary)
        self.assertIn("Field Two", summary)
        self.assertIn("and 2 more", summary)
        self.assertNotIn("Field Three", summary)

    async def test_abandoned_field_summary_empty_when_none(self):
        ca._task_state.set_initial_todos(["Type 'X' in Y"])
        ca._task_state.todo_list[0]["done"] = True  # done but NOT abandoned
        self.assertEqual(ca._task_state.abandoned_field_summary(), "")

    async def test_append_abandoned_suffix_appends_disclosure(self):
        ca._task_state.set_initial_todos(["Select '1-50' from Staff Size dropdown"])
        t = ca._task_state.todo_list[0]
        t["done"] = True
        t["confirm_abandoned"] = True
        out = ca._append_abandoned_suffix("Booked the demo.")
        self.assertEqual(out, "Booked the demo (couldn't visually confirm: Staff Size).")

    async def test_append_abandoned_suffix_noop_when_no_abandoned(self):
        ca._task_state.set_initial_todos(["Type 'John' in First Name"])
        ca._task_state.todo_list[0]["done"] = True
        out = ca._append_abandoned_suffix("All done.")
        self.assertEqual(out, "All done.")

    async def test_pipeline_abandoned_then_no_replan(self):
        # End-to-end: deferred TODO → 3 strikes → fallback NO → done +
        # abandoned. A subsequent batch with a click on the same field
        # MUST NOT re-defer (the done filter holds).
        _install_screen_llm(vision_responses=["NO", "NO", "NO", "NO"])
        _stub_select_todo("Industry", "IT", pending=True, strikes=2)
        ca._task_state.batch_idx = 5
        # First confirm pass: strict NO → strike 3 → fallback NO → abandoned
        await ca._confirm_pending_select_todos()
        t = ca._task_state.todo_list[0]
        self.assertTrue(t["done"])
        self.assertTrue(t["confirm_abandoned"])
        # Now a follow-up batch clicks the same target.
        actions = [{"type": "vision_guided_click", "text": "Industry"}]
        results = ["Clicked 'Industry' at (100,200)"]
        ca._task_state.batch_idx = 6
        addressed, marked = ca._match_actions_to_todos(actions, results)
        # Rule S filtered out by done=True. No re-defer.
        self.assertEqual(addressed, set())
        self.assertFalse(t["pending_visual_confirm"])  # still cleared
        self.assertTrue(t["confirm_abandoned"])  # sticky


class TestVisualConfirmPromptShape(unittest.TestCase):
    def test_template_accepts_dropdown_styling(self):
        # Softened prompt must explicitly mention dropdown-style rendering
        # so the LLM doesn't penalize the value for visual styling alone.
        prompt = ca._VISUAL_CONFIRM_PROMPT_TEMPLATE
        self.assertIn("dropdown", prompt.lower())
        self.assertIn("regardless of styling", prompt.lower())

    def test_template_still_rejects_placeholder(self):
        # Hard requirement: empty/placeholder states must still return NO.
        prompt = ca._VISUAL_CONFIRM_PROMPT_TEMPLATE
        self.assertIn("placeholder", prompt.lower())
        self.assertIn("empty", prompt.lower())

    def test_template_still_rejects_open_menu_list(self):
        # The original strict-version case: a value visible in the OPEN
        # dropdown menu list (highlighted but not committed) must NOT count
        # as selected. The softened version must preserve this guard.
        prompt = ca._VISUAL_CONFIRM_PROMPT_TEMPLATE
        self.assertIn("open dropdown menu", prompt.lower())

    def test_template_field_value_substitution(self):
        # Sanity: format() substitution still works on the new template.
        out = ca._VISUAL_CONFIRM_PROMPT_TEMPLATE.format(
            field="Staff Size", value="1-50"
        )
        self.assertIn("Staff Size", out)
        self.assertIn("1-50", out)
        # No KeyError or stray placeholders.
        self.assertNotIn("{field}", out)
        self.assertNotIn("{value}", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
