"""
test_browser_dom_executor.py — Phase 1C-b: DOM action executor.

Tests the executor against a stubbed Playwright Locator. Coverage:
  - Each action type's happy path
  - Read-back verification: mismatch detected for form_input/select
  - Lenient value-match (whitespace, case, contains-either-way)
  - Per-action exception → recorded + continue (no batch short-circuit)
  - Reperceive halts the batch and signals requires_reperceive
  - tree_dirty set by clicks/presses/selects, not by fills
  - wait_ms sleeps and always succeeds
  - Missing ref / unknown type → recorded as failure
  - Batch result aggregation (all_succeeded, succeeded/failed properties)

Run: python test_browser_dom_executor.py
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom_executor as bde


def _run(coro):
    return asyncio.run(coro)


# ─── Stub Locator ────────────────────────────────────────────────────────


class _StubLocator:
    """
    Minimal Playwright Locator stand-in. Records every call and lets the
    test control return values + exceptions.
    """
    def __init__(
        self,
        *,
        fill_raises=None,
        click_raises=None,
        select_label_raises=None,
        select_value_raises=None,
        press_raises=None,
        input_value_return="",
        input_value_raises=None,
        evaluate_return="",
        evaluate_raises=None,
    ):
        self.fill_raises = fill_raises
        self.click_raises = click_raises
        self.select_label_raises = select_label_raises
        self.select_value_raises = select_value_raises
        self.press_raises = press_raises
        self.input_value_return = input_value_return
        self.input_value_raises = input_value_raises
        self.evaluate_return = evaluate_return
        self.evaluate_raises = evaluate_raises

        self.fill_calls: list = []
        self.click_calls: list = []
        self.select_calls: list = []
        self.press_calls: list = []

    async def fill(self, value, timeout=None):
        self.fill_calls.append({"value": value, "timeout": timeout})
        if self.fill_raises is not None:
            raise self.fill_raises

    async def click(self, timeout=None):
        self.click_calls.append({"timeout": timeout})
        if self.click_raises is not None:
            raise self.click_raises

    async def select_option(self, label=None, value=None, timeout=None):
        self.select_calls.append({"label": label, "value": value, "timeout": timeout})
        if label is not None and self.select_label_raises is not None:
            raise self.select_label_raises
        if value is not None and self.select_value_raises is not None:
            raise self.select_value_raises

    async def press(self, key, timeout=None):
        self.press_calls.append({"key": key, "timeout": timeout})
        if self.press_raises is not None:
            raise self.press_raises

    async def input_value(self, timeout=None):
        if self.input_value_raises is not None:
            raise self.input_value_raises
        return self.input_value_return

    async def evaluate(self, js):
        if self.evaluate_raises is not None:
            raise self.evaluate_raises
        return self.evaluate_return


# ─── form_input ──────────────────────────────────────────────────────────


class TestFormInput(unittest.TestCase):
    def test_happy_path(self):
        loc = _StubLocator(input_value_return="John")
        action = {"type": "form_input", "ref": "r1", "value": "John"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.all_succeeded)
        self.assertEqual(loc.fill_calls[0]["value"], "John")
        self.assertEqual(result.results[0].observed_value, "John")

    def test_read_back_mismatch_fails(self):
        loc = _StubLocator(input_value_return="something else")
        action = {"type": "form_input", "ref": "r1", "value": "John"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertFalse(result.all_succeeded)
        self.assertIn("read-back mismatch", result.results[0].error)

    def test_fill_exception_records_failure(self):
        loc = _StubLocator(fill_raises=RuntimeError("element detached"))
        action = {"type": "form_input", "ref": "r1", "value": "John"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertFalse(result.all_succeeded)
        self.assertIn("fill failed", result.results[0].error)

    def test_read_back_exception_does_not_fail_action(self):
        # If fill() succeeds but input_value() raises, we don't fail —
        # recovery is the safety net.
        loc = _StubLocator(input_value_raises=RuntimeError("eval timeout"))
        action = {"type": "form_input", "ref": "r1", "value": "John"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.all_succeeded)
        self.assertIn("read-back failed", result.results[0].error)

    def test_form_input_does_not_set_tree_dirty(self):
        # Filling text doesn't mutate other DOM nodes (typically). The
        # orchestrator can keep its tree cache. Distinguishes form_input
        # from clicks/selects.
        loc = _StubLocator(input_value_return="x")
        action = {"type": "form_input", "ref": "r1", "value": "x"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertFalse(result.tree_dirty)

    def test_lenient_value_match(self):
        # Phone field auto-formats — observed differs from input but
        # contains the digits. Should pass.
        loc = _StubLocator(input_value_return="(123) 456-7890")
        action = {"type": "form_input", "ref": "r1", "value": "1234567890"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        # Lenient match handles substring either way after whitespace strip
        # — but "1234567890" is not a substring of "(123) 456-7890" because
        # of the punctuation. So this should fail strict-or-substring.
        # Confirms _values_match is permissive but not naive.
        self.assertFalse(result.all_succeeded)

    def test_whitespace_normalized_in_match(self):
        loc = _StubLocator(input_value_return="  John  ")
        action = {"type": "form_input", "ref": "r1", "value": "John"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.all_succeeded)

    def test_case_insensitive_match(self):
        loc = _StubLocator(input_value_return="JANE@example.com")
        action = {"type": "form_input", "ref": "r1", "value": "jane@example.com"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.all_succeeded)


# ─── click_ref ───────────────────────────────────────────────────────────


class TestClickRef(unittest.TestCase):
    def test_happy_path(self):
        loc = _StubLocator()
        action = {"type": "click_ref", "ref": "r1"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.all_succeeded)
        self.assertEqual(len(loc.click_calls), 1)

    def test_click_failure_recorded(self):
        loc = _StubLocator(click_raises=RuntimeError("not visible"))
        action = {"type": "click_ref", "ref": "r1"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertFalse(result.all_succeeded)
        self.assertIn("click failed", result.results[0].error)

    def test_click_sets_tree_dirty(self):
        loc = _StubLocator()
        action = {"type": "click_ref", "ref": "r1"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.tree_dirty)


# ─── select_option_ref ───────────────────────────────────────────────────


class TestSelectOption(unittest.TestCase):
    def test_happy_path_label_match(self):
        loc = _StubLocator(evaluate_return="Canada")
        action = {"type": "select_option_ref", "ref": "r1", "option": "Canada"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.all_succeeded)
        self.assertEqual(loc.select_calls[0]["label"], "Canada")
        self.assertEqual(result.results[0].observed_value, "Canada")

    def test_label_fails_falls_back_to_value(self):
        # First call (label) raises with "United Kingdom" — a label that
        # has no overlap with the eventual read-back. Value-fallback
        # succeeds, but the read-back returns a different country than
        # the planner asked for, so the action is recorded as failed.
        loc = _StubLocator(
            select_label_raises=RuntimeError("no option with that label"),
            evaluate_return="Mexico",
        )
        action = {"type": "select_option_ref", "ref": "r1", "option": "United Kingdom"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        # Label failed → value-fallback succeeded with no exception → but
        # read-back observed "Mexico" while expected was "United Kingdom"
        # (no substring overlap, no case match) → record as soft failure.
        self.assertFalse(result.all_succeeded)
        self.assertIn("read-back mismatch", result.results[0].error)
        # Both select attempts recorded.
        self.assertEqual(len(loc.select_calls), 2)

    def test_label_fails_value_succeeds_substring_passes(self):
        # Documents the lenient-match behaviour: when the planner passes
        # an option that's a substring of the actual selected text (e.g.
        # planner used a value attr "ca" that's inside "Canada"), we
        # accept it. This is intentional — the field contains a
        # superset of what the planner asked for.
        loc = _StubLocator(
            select_label_raises=RuntimeError("no option with that label"),
            evaluate_return="Canada",
        )
        action = {"type": "select_option_ref", "ref": "r1", "option": "ca"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        # Lenient match: "ca" is a substring of "canada" (case-folded) → pass.
        self.assertTrue(result.all_succeeded)

    def test_both_attempts_fail_records_both_errors(self):
        loc = _StubLocator(
            select_label_raises=RuntimeError("label miss"),
            select_value_raises=RuntimeError("value miss"),
        )
        action = {"type": "select_option_ref", "ref": "r1", "option": "X"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertFalse(result.all_succeeded)
        err = result.results[0].error
        self.assertIn("by-label", err)
        self.assertIn("by-value", err)

    def test_select_sets_tree_dirty(self):
        # Select changes can trigger onChange → new fields appear.
        loc = _StubLocator(evaluate_return="A")
        action = {"type": "select_option_ref", "ref": "r1", "option": "A"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.tree_dirty)

    def test_read_back_mismatch_fails(self):
        loc = _StubLocator(evaluate_return="Wrong")
        action = {"type": "select_option_ref", "ref": "r1", "option": "Canada"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertFalse(result.all_succeeded)
        self.assertIn("read-back mismatch", result.results[0].error)
        self.assertEqual(result.results[0].observed_value, "Wrong")


# ─── press_ref ───────────────────────────────────────────────────────────


class TestPressRef(unittest.TestCase):
    def test_happy_path(self):
        loc = _StubLocator()
        action = {"type": "press_ref", "ref": "r1", "key": "Enter"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.all_succeeded)
        self.assertEqual(loc.press_calls[0]["key"], "Enter")

    def test_press_sets_tree_dirty(self):
        # Enter on a form often submits → tree changes.
        loc = _StubLocator()
        action = {"type": "press_ref", "ref": "r1", "key": "Enter"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertTrue(result.tree_dirty)

    def test_press_failure_recorded(self):
        loc = _StubLocator(press_raises=RuntimeError("element gone"))
        action = {"type": "press_ref", "ref": "r1", "key": "Enter"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {"r1": loc}))
        self.assertFalse(result.all_succeeded)
        self.assertIn("press failed", result.results[0].error)


# ─── wait_ms ─────────────────────────────────────────────────────────────


class TestWaitMs(unittest.TestCase):
    def test_happy_path_no_sleep_actually_taken(self):
        # ms=0 should not actually sleep (faster tests)
        action = {"type": "wait_ms", "ms": 0}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {}))
        self.assertTrue(result.all_succeeded)

    def test_short_sleep_runs(self):
        action = {"type": "wait_ms", "ms": 10}  # 10ms — fast enough for tests
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {}))
        self.assertTrue(result.all_succeeded)

    def test_does_not_set_tree_dirty(self):
        action = {"type": "wait_ms", "ms": 0}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {}))
        self.assertFalse(result.tree_dirty)


# ─── reperceive ──────────────────────────────────────────────────────────


class TestReperceive(unittest.TestCase):
    def test_halts_batch_and_signals_reperceive(self):
        loc = _StubLocator()
        actions = [
            {"type": "click_ref", "ref": "r1"},
            {"type": "reperceive"},
            {"type": "click_ref", "ref": "r1"},  # should be skipped
        ]
        result = _run(bde.execute_dom_batch(MagicMock(), actions, {"r1": loc}))
        self.assertTrue(result.requires_reperceive)
        self.assertTrue(result.tree_dirty)
        # Only 2 results (click + reperceive); the post-reperceive click
        # was dropped.
        self.assertEqual(len(result.results), 2)
        self.assertEqual(len(loc.click_calls), 1)


# ─── Defensive paths ─────────────────────────────────────────────────────


class TestDefensive(unittest.TestCase):
    def test_missing_locator_recorded_not_raised(self):
        # Validator should have caught this, but if a stale ref slips through
        # (e.g. post-reperceive ref-map), we must record + continue.
        action = {"type": "form_input", "ref": "missing", "value": "x"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {}))
        self.assertFalse(result.all_succeeded)
        self.assertIn("locator not found", result.results[0].error)

    def test_unknown_type_recorded_not_raised(self):
        action = {"type": "totally_unknown"}
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {}))
        self.assertFalse(result.all_succeeded)
        self.assertIn("unknown action type", result.results[0].error)

    def test_batch_continues_after_failure(self):
        # First action fails, second action still runs.
        loc1 = _StubLocator(fill_raises=RuntimeError("boom"))
        loc2 = _StubLocator()
        actions = [
            {"type": "form_input", "ref": "r1", "value": "x"},
            {"type": "click_ref", "ref": "r2"},
        ]
        result = _run(bde.execute_dom_batch(
            MagicMock(), actions, {"r1": loc1, "r2": loc2},
        ))
        self.assertEqual(len(result.results), 2)
        self.assertFalse(result.results[0].succeeded)
        self.assertTrue(result.results[1].succeeded)
        self.assertEqual(len(loc2.click_calls), 1)
        # Mixed batch: some succeeded, some failed
        self.assertFalse(result.all_succeeded)
        self.assertEqual(len(result.succeeded), 1)
        self.assertEqual(len(result.failed), 1)

    def test_action_missing_type_recorded(self):
        action = {"ref": "r1"}  # no type
        result = _run(bde.execute_dom_batch(MagicMock(), [action], {}))
        self.assertFalse(result.all_succeeded)


# ─── _values_match ──────────────────────────────────────────────────────


class TestValuesMatch(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(bde._values_match("John", "John"))

    def test_case_insensitive(self):
        self.assertTrue(bde._values_match("john", "John"))

    def test_whitespace_normalized(self):
        self.assertTrue(bde._values_match("  John  ", "John"))
        self.assertTrue(bde._values_match("John\tDoe", "John Doe"))

    def test_substring_either_way(self):
        # "Canada" inside "Canada (CA)"
        self.assertTrue(bde._values_match("Canada (CA)", "Canada"))
        self.assertTrue(bde._values_match("ca", "Canada"))  # 'ca' in 'canada'

    def test_both_empty_match(self):
        self.assertTrue(bde._values_match("", ""))

    def test_one_empty_no_match(self):
        self.assertFalse(bde._values_match("John", ""))
        self.assertFalse(bde._values_match("", "John"))

    def test_non_string_no_match(self):
        self.assertFalse(bde._values_match(None, "x"))
        self.assertFalse(bde._values_match("x", None))


# ─── DomBatchResult properties ───────────────────────────────────────────


class TestBatchResultProperties(unittest.TestCase):
    def test_all_succeeded_empty_results_is_false(self):
        r = bde.DomBatchResult()
        self.assertFalse(r.all_succeeded)

    def test_all_succeeded_when_all_pass(self):
        r = bde.DomBatchResult(results=[
            bde.DomActionResult(action={}, succeeded=True),
            bde.DomActionResult(action={}, succeeded=True),
        ])
        self.assertTrue(r.all_succeeded)

    def test_succeeded_failed_partition(self):
        r = bde.DomBatchResult(results=[
            bde.DomActionResult(action={}, succeeded=True),
            bde.DomActionResult(action={}, succeeded=False),
            bde.DomActionResult(action={}, succeeded=True),
        ])
        self.assertEqual(len(r.succeeded), 2)
        self.assertEqual(len(r.failed), 1)
        self.assertFalse(r.all_succeeded)


# ─── Bug 7: fast-fail caps ───────────────────────────────────────────────


class TestFastFailConsecutiveFailures(unittest.TestCase):
    """The executor must not waste 60s on a stuck page. After N actions
    fail in a row, the rest of the batch is aborted with a tagged error."""

    def test_two_failures_in_a_row_aborts_remainder(self):
        # 5 actions, each with a locator whose .fill raises. With
        # max_consecutive_failures=2, only the first 2 should be dispatched
        # to the locator; the remaining 3 must be marked aborted.
        bad_locators = {
            f"r{i}": _StubLocator(fill_raises=Exception("element not found"))
            for i in range(5)
        }
        actions = [
            {"type": "form_input", "ref": f"r{i}", "value": "x"}
            for i in range(5)
        ]
        result = _run(bde.execute_dom_batch(
            page=MagicMock(), actions=actions, ref_to_locator=bad_locators,
            max_consecutive_failures=2,
        ))
        self.assertEqual(len(result.results), 5)
        # First 2 dispatched (.fill raised) → error contains "element not found"
        for r in result.results[:2]:
            self.assertFalse(r.succeeded)
            self.assertIn("element not found", r.error)
        # Last 3 aborted with the synthetic error
        for r in result.results[2:]:
            self.assertFalse(r.succeeded)
            self.assertIn("batch aborted", r.error)
            self.assertIn("consecutive", r.error)

    def test_single_failure_does_not_abort(self):
        # One transient failure should not abort the batch — only N-in-a-row.
        good = _StubLocator(input_value_return="x")
        bad = _StubLocator(fill_raises=Exception("flaky"))
        ref_map = {"good": good, "bad": bad}
        actions = [
            {"type": "form_input", "ref": "good", "value": "x"},
            {"type": "form_input", "ref": "bad", "value": "x"},
            {"type": "form_input", "ref": "good", "value": "x"},
            {"type": "form_input", "ref": "good", "value": "x"},
        ]
        result = _run(bde.execute_dom_batch(
            page=MagicMock(), actions=actions, ref_to_locator=ref_map,
            max_consecutive_failures=2,
        ))
        # All 4 attempted (no abort): 3 succeed, 1 fails (the bad one).
        self.assertEqual(len(result.results), 4)
        succ = sum(1 for r in result.results if r.succeeded)
        self.assertEqual(succ, 3)
        # No aborted-tag errors
        for r in result.results:
            if not r.succeeded:
                self.assertNotIn("batch aborted", r.error)

    def test_failure_then_success_resets_counter(self):
        # fail, success, fail, fail → no abort: streak goes 1, 0, 1, 2.
        # The check is `streak >= max` BEFORE the action's own dispatch,
        # so 2-in-a-row only aborts the *3rd* action onwards.
        good = _StubLocator(input_value_return="x")
        bad = _StubLocator(fill_raises=Exception("nope"))
        ref_map = {"good": good, "bad": bad}
        actions = [
            {"type": "form_input", "ref": "bad", "value": "x"},   # fail (streak=1)
            {"type": "form_input", "ref": "good", "value": "x"},  # ok (streak=0)
            {"type": "form_input", "ref": "bad", "value": "x"},   # fail (streak=1)
            {"type": "form_input", "ref": "bad", "value": "x"},   # fail (streak=2)
        ]
        result = _run(bde.execute_dom_batch(
            page=MagicMock(), actions=actions, ref_to_locator=ref_map,
            max_consecutive_failures=2,
        ))
        # All 4 attempted — the streak never reached 2 BEFORE an action's
        # check (only after the last one's dispatch).
        self.assertEqual(len(result.results), 4)
        for r in result.results:
            self.assertNotIn("batch aborted", r.error or "")

    def test_aborted_actions_have_clean_error_message(self):
        # The synthetic error must mention the cause clearly so the
        # orchestrator's AR-1c feedback is honest to the planner.
        bad = _StubLocator(fill_raises=Exception("X"))
        ref_map = {"r": bad}
        actions = [
            {"type": "form_input", "ref": "r", "value": "x"},
            {"type": "form_input", "ref": "r", "value": "x"},
            {"type": "form_input", "ref": "r", "value": "x"},
        ]
        result = _run(bde.execute_dom_batch(
            page=MagicMock(), actions=actions, ref_to_locator=ref_map,
            max_consecutive_failures=2,
        ))
        aborted_msg = result.results[2].error
        self.assertIn("batch aborted", aborted_msg)
        self.assertIn("page state likely wrong", aborted_msg)


class TestBatchBudgetCap(unittest.TestCase):
    """Wall-clock budget — even all-succeeding actions can't exceed N ms."""

    def test_budget_exceeded_aborts_remainder(self):
        # Use a real async fn that actually sleeps. AsyncMock with
        # side_effect=lambda returning asyncio.sleep doesn't propagate the
        # await, so the sleep was never honoured in the previous draft.
        async def slow_fill(*args, **kwargs):
            await asyncio.sleep(0.05)

        async def slow_input_value(*args, **kwargs):
            return "x"

        slow_loc = MagicMock()
        slow_loc.fill = slow_fill
        slow_loc.input_value = slow_input_value

        ref_map = {f"r{i}": slow_loc for i in range(6)}
        actions = [
            {"type": "form_input", "ref": f"r{i}", "value": "x"}
            for i in range(6)
        ]
        result = _run(bde.execute_dom_batch(
            page=MagicMock(), actions=actions, ref_to_locator=ref_map,
            batch_budget_ms=80,
            max_consecutive_failures=99,  # disable consecutive check
        ))
        # First 1-2 actions complete; remainder are aborted.
        succ = sum(1 for r in result.results if r.succeeded)
        aborted = sum(
            1 for r in result.results
            if not r.succeeded and "budget" in (r.error or "")
        )
        self.assertGreaterEqual(succ, 1)
        self.assertGreaterEqual(aborted, 1)
        self.assertEqual(len(result.results), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
