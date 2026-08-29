"""
test_browser_dom_orchestrator.py — Phase 1C-c: orchestrator + AR-1c +
multi-form disambiguation.

Stub-based unit tests. Real-Playwright integration of the full loop is
deferred to Phase 1F's test matrix (where it runs against the actual
Truein form). Here we verify:

  - _looks_like_submit token matching
  - _select_target_form: single form, modal preference, goal-vs-submit
    scoring, no-form fallback, deterministic tiebreak
  - _scope_tree_to_elements: ref_to_locator filtered to match
  - _format_failures_for_planner: builds AR-1c feedback string with cap
  - run_dom_task: every branch of the perceive→plan→execute loop
    * happy single-loop completion
    * tree_dirty triggers cache invalidation
    * failed batch → AR-1c feedback plumbed into next iteration
    * max_loops exhausted
    * empty plan + done → success
    * empty plan + not done → planner_failed
    * empty tree → empty_tree result
    * perceive raises → perceive_failed result

Run: python test_browser_dom_orchestrator.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom as bdom
import assistant.automation.browser.dom_planner as bdp
import assistant.automation.browser.dom_executor as bde
import assistant.automation.browser.dom_orchestrator as bdo
# P13. `handle` and `router` are stdlib+config at import time -- neither pulls
# Playwright or pyautogui, so importing them here cannot reach the desktop.
import assistant.automation.browser.handle as bhandle
import assistant.automation.router as router
from assistant.core.abort import UserAborted
from assistant.core.verdict import Outcome, speaks_as_done


def _run(coro):
    return asyncio.run(coro)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _elem(ref, **kwargs) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input", form_id="", in_dialog=False,
    )
    base.update(kwargs)
    return bdom.ElementInfo(ref=ref, **base)


def _make_tree(elements: list[bdom.ElementInfo]) -> bdom.PageDomTree:
    ref_map = {e.ref: MagicMock(name=f"loc-{e.ref}") for e in elements}
    return bdom.PageDomTree(
        elements=elements, ref_to_locator=ref_map, truncated=0,
        read_at=time.monotonic(), viewport=(1280, 800),
    )


def _batch_result(*, failed_actions=None, all_actions=None,
                  tree_dirty=False, reperceive=False) -> bde.DomBatchResult:
    """Quick batch-result builder for stubs."""
    if all_actions is None and failed_actions is None:
        all_actions = []
    if all_actions is None:
        # Treat failed_actions as the only actions
        results = [
            bde.DomActionResult(action=a, succeeded=False, error="stub fail")
            for a in (failed_actions or [])
        ]
    else:
        results = []
        failed_set = {id(a) for a in (failed_actions or [])}
        for a in all_actions:
            if id(a) in failed_set:
                results.append(bde.DomActionResult(action=a, succeeded=False, error="stub fail"))
            else:
                results.append(bde.DomActionResult(action=a, succeeded=True))
    return bde.DomBatchResult(
        results=results, requires_reperceive=reperceive, tree_dirty=tree_dirty,
    )


# ─── _looks_like_submit ──────────────────────────────────────────────────


class TestLooksLikeSubmit(unittest.TestCase):
    def test_common_submit_phrases(self):
        for name in ["Submit", "SUBMIT", "Send", "Schedule a Demo",
                     "Sign in", "Log in", "Subscribe", "Continue"]:
            self.assertTrue(bdo._looks_like_submit(name), f"missed {name!r}")

    def test_non_submit_buttons(self):
        for name in ["Play video", "Show details", "Edit", "Close",
                     "Cancel", "Filter results"]:
            self.assertFalse(bdo._looks_like_submit(name), f"false positive on {name!r}")

    def test_empty_or_none(self):
        self.assertFalse(bdo._looks_like_submit(""))
        self.assertFalse(bdo._looks_like_submit(None))


# ─── _select_target_form ─────────────────────────────────────────────────


class TestSelectTargetForm(unittest.TestCase):
    def test_no_forms_returns_none(self):
        elements = [_elem("r1"), _elem("r2")]  # no form_id
        self.assertIsNone(bdo._select_target_form(elements, "fill the form"))

    def test_single_form_used(self):
        elements = [_elem("r1", form_id="form-0"), _elem("r2", form_id="form-0")]
        result = bdo._select_target_form(elements, "fill")
        self.assertIsNotNone(result)
        fid, els = result
        self.assertEqual(fid, "form-0")
        self.assertEqual(len(els), 2)

    def test_modal_preference_wins(self):
        # Two forms; one inside a dialog. The dialog one wins regardless
        # of goal text.
        elements = [
            _elem("r1", form_id="form-0", role="button",
                  name="Submit", in_dialog=False),
            _elem("r2", form_id="form-1", role="button",
                  name="Submit", in_dialog=True),
        ]
        result = bdo._select_target_form(elements, "submit")
        fid, _ = result
        self.assertEqual(fid, "form-1", "dialog form should win")

    def test_goal_vs_submit_scoring(self):
        # Two forms with different submit names. Goal mentions "demo"
        # → form whose submit is "Schedule a Demo" wins.
        elements = [
            _elem("r1", form_id="form-0", role="button", name="Subscribe"),
            _elem("r2", form_id="form-1", role="button", name="Schedule a Demo"),
        ]
        result = bdo._select_target_form(
            elements, "fill the demo form with testing values"
        )
        fid, _ = result
        self.assertEqual(fid, "form-1")

    def test_no_obvious_match_falls_back_to_first(self):
        # Goal text doesn't match anything specific; deterministic tiebreak
        # by sorted form_id picks form-0.
        elements = [
            _elem("r1", form_id="form-0", role="button", name="Submit"),
            _elem("r2", form_id="form-1", role="button", name="Send"),
        ]
        result = bdo._select_target_form(elements, "fill this form")
        fid, _ = result
        self.assertEqual(fid, "form-0")

    def test_form_with_no_submit_button_still_eligible(self):
        # Forms might not have a recognized "submit" button (e.g. forms
        # using a custom div as the action element). Should still get
        # picked when it's the only candidate, or by tiebreak.
        elements = [_elem("r1", form_id="form-0", role="textbox", name="A")]
        result = bdo._select_target_form(elements, "x")
        self.assertIsNotNone(result)
        fid, _ = result
        self.assertEqual(fid, "form-0")

    def test_modal_with_two_dialogs_picks_first(self):
        # Edge case: two modal forms simultaneously visible. Token-overlap
        # scoring still applies, falling back to alphabetical tiebreak.
        elements = [
            _elem("r1", form_id="form-0", role="button",
                  name="Save", in_dialog=True),
            _elem("r2", form_id="form-1", role="button",
                  name="Save", in_dialog=True),
        ]
        result = bdo._select_target_form(elements, "save")
        fid, _ = result
        self.assertEqual(fid, "form-0")  # tiebreak

    def test_goal_demo_with_truein_two_form_setup(self):
        # Replicates Truein's actual structure: two identical forms
        # except one has submit "Schedule a Demo" (modal) and the other
        # has submit "Submit" (footer).
        elements = [
            _elem("r1", form_id="form-0", role="textbox", name="First name"),
            _elem("r2", form_id="form-0", role="button",
                  name="Schedule a Demo"),
            _elem("r3", form_id="form-1", role="textbox", name="First Name"),
            _elem("r4", form_id="form-1", role="button", name="Submit"),
        ]
        result = bdo._select_target_form(
            elements, "fill the demo form with testing values"
        )
        fid, els = result
        self.assertEqual(fid, "form-0")
        self.assertEqual(len(els), 2)


# ─── _scope_tree_to_elements ─────────────────────────────────────────────


class TestScopeTreeToElements(unittest.TestCase):
    def test_filters_ref_map(self):
        full_tree = _make_tree([_elem("r1"), _elem("r2"), _elem("r3")])
        target = [full_tree.elements[0], full_tree.elements[2]]
        scoped = bdo._scope_tree_to_elements(full_tree, target)
        self.assertEqual(len(scoped.elements), 2)
        self.assertEqual(set(scoped.ref_to_locator.keys()), {"r1", "r3"})
        self.assertEqual(scoped.viewport, full_tree.viewport)


# ─── _format_failures_for_planner ────────────────────────────────────────


class TestFormatFailuresForPlanner(unittest.TestCase):
    def test_no_failures_returns_empty(self):
        batch = bde.DomBatchResult(results=[
            bde.DomActionResult(action={}, succeeded=True),
        ])
        self.assertEqual(bdo._format_failures_for_planner(batch), "")

    def test_failures_formatted_with_action_context(self):
        batch = bde.DomBatchResult(results=[
            bde.DomActionResult(
                action={"type": "form_input", "ref": "r1"},
                succeeded=False, error="read-back mismatch",
                observed_value="",
            ),
            bde.DomActionResult(
                action={"type": "click_ref", "ref": "r2"},
                succeeded=False, error="click failed: TimeoutError",
            ),
        ])
        out = bdo._format_failures_for_planner(batch)
        self.assertIn("Previous batch had failures", out)
        self.assertIn("form_input ref=r1", out)
        self.assertIn("read-back mismatch", out)
        self.assertIn("click_ref ref=r2", out)

    def test_observed_value_included_when_present(self):
        batch = bde.DomBatchResult(results=[
            bde.DomActionResult(
                action={"type": "form_input", "ref": "r1"},
                succeeded=False, error="mismatch",
                observed_value="Wrong text",
            ),
        ])
        out = bdo._format_failures_for_planner(batch)
        self.assertIn("Wrong text", out)

    def test_truncates_at_max_lines(self):
        results = []
        for i in range(10):
            results.append(bde.DomActionResult(
                action={"type": "form_input", "ref": f"r{i}"},
                succeeded=False, error=f"err{i}",
            ))
        batch = bde.DomBatchResult(results=results)
        out = bdo._format_failures_for_planner(batch, max_lines=3)
        # 10 failures, max_lines=3 → first 3 emitted, "(7 more)" suffix
        self.assertIn("7 more", out)
        # First 3 referenced
        self.assertIn("r0", out)
        self.assertIn("r2", out)
        self.assertNotIn("r5", out)


# ─── run_dom_task: full loop ─────────────────────────────────────────────


class TestRunDomTask(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_happy_single_loop(self):
        # Perceive returns a tree; planner emits a batch; executor succeeds.
        # Phase 2E: a successful submit triggers a post-submit perceive pass
        # — when that pass finds no validation errors, success is returned
        # without a second plan call. So loops_used=2 (two perceives) but
        # only one planner call.
        tree = _make_tree([
            _elem("r1", form_id="form-0", role="textbox", name="First name"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        plan_actions = [
            {"type": "form_input", "ref": "r1", "value": "John"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="fill+submit", plan="ok",
            actions=plan_actions, done=True, needs_reperceive=False,
        )
        batch = _batch_result(all_actions=plan_actions)

        plan_mock = AsyncMock(return_value=plan)
        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=plan_mock), \
             patch.object(bde, "execute_dom_batch", new=AsyncMock(return_value=batch)):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.loops_used, 2)
        # Planner called exactly once — the post-submit pass short-circuits
        # before re-invoking the LLM.
        self.assertEqual(plan_mock.await_count, 1)

    async def test_failed_actions_feed_back_then_succeed(self):
        # Loop 1: planner emits 2 actions, one fails. Loop 2: planner
        # corrects, all succeed.
        tree = _make_tree([
            _elem("r1", form_id="form-0", role="textbox", name="A"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        actions_1 = [
            {"type": "form_input", "ref": "r1", "value": "X"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan_1 = bdp.DomPlan(
            thinking="initial", plan="x",
            actions=actions_1, done=True, needs_reperceive=False,
        )
        batch_1 = _batch_result(
            all_actions=actions_1, failed_actions=[actions_1[0]],
            tree_dirty=True,
        )

        actions_2 = [
            {"type": "form_input", "ref": "r1", "value": "Corrected"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan_2 = bdp.DomPlan(
            thinking="retry", plan="ok",
            actions=actions_2, done=True, needs_reperceive=False,
        )
        batch_2 = _batch_result(all_actions=actions_2)

        plan_calls: list = []
        async def _plan_side_effect(goal, scoped, *, feedback=""):
            plan_calls.append({"feedback": feedback})
            return plan_1 if len(plan_calls) == 1 else plan_2

        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(side_effect=_plan_side_effect)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=[batch_1, batch_2])):
            result = await bdo.run_dom_task("fill", MagicMock())

        self.assertTrue(result.success)
        # Phase 2E: 2 plan-and-execute loops + 1 post-submit verification loop = 3
        self.assertEqual(result.loops_used, 3)
        # Second plan call received feedback from the failure of loop 1.
        # The third loop is the post-submit short-circuit and does NOT call
        # the planner — so plan_calls stays at 2.
        self.assertEqual(len(plan_calls), 2)
        self.assertEqual(plan_calls[0]["feedback"], "")
        self.assertIn("Previous batch had failures", plan_calls[1]["feedback"])
        self.assertIn("form_input", plan_calls[1]["feedback"])

    async def test_max_loops_exhausted(self):
        tree = _make_tree([
            _elem("r1", form_id="form-0", role="button", name="Submit"),
        ])
        actions = [{"type": "click_ref", "ref": "r1"}]
        # Plan never says done; batch always fails.
        plan = bdp.DomPlan(
            thinking="x", plan="x", actions=actions, done=False,
            needs_reperceive=False,
        )
        batch = _batch_result(failed_actions=actions, all_actions=actions,
                              tree_dirty=True)

        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch", new=AsyncMock(return_value=batch)):
            result = await bdo.run_dom_task("fill", MagicMock(), max_loops=3)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "loop_failure_at_max")
        self.assertEqual(result.loops_used, 3)

    async def test_empty_plan_with_done_is_success(self):
        # Planner says nothing to do — accept as completion.
        tree = _make_tree([_elem("r1", form_id="form-0")])
        plan = bdp.DomPlan(
            thinking="nothing to do", plan="all good",
            actions=[], done=True, needs_reperceive=False,
        )

        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(return_value=plan)):
            result = await bdo.run_dom_task("g", MagicMock())

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed_no_actions")

    async def test_empty_plan_without_done_eventually_fails(self):
        tree = _make_tree([_elem("r1", form_id="form-0")])
        plan = bdp.DomPlan(
            thinking="x", plan="x", actions=[], done=False,
            needs_reperceive=False, rejection_notes=["llm_unavailable"],
        )

        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(return_value=plan)):
            result = await bdo.run_dom_task("g", MagicMock(), max_loops=2)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "planner_failed")

    async def test_empty_tree_eventually_fails(self):
        empty_tree = _make_tree([])

        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=empty_tree)), \
             patch.object(bdom, "invalidate_tree_cache", new=MagicMock()):
            result = await bdo.run_dom_task("g", MagicMock(), max_loops=2)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "empty_tree")

    async def test_perceive_raises_returns_failure(self):
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=RuntimeError("page closed"))):
            result = await bdo.run_dom_task("g", MagicMock())

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "perceive_failed")

    async def test_tree_dirty_invalidates_cache(self):
        tree = _make_tree([
            _elem("r1", form_id="form-0", role="button", name="Submit"),
        ])
        actions = [{"type": "click_ref", "ref": "r1"}]
        plan = bdp.DomPlan(
            thinking="x", plan="x", actions=actions, done=True,
            needs_reperceive=False,
        )
        batch = _batch_result(all_actions=actions, tree_dirty=True)

        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch", new=AsyncMock(return_value=batch)), \
             patch.object(bdom, "invalidate_tree_cache", new=MagicMock()) as inv:
            result = await bdo.run_dom_task("g", MagicMock())

        self.assertTrue(result.success)
        # invalidate_tree_cache called at least once due to tree_dirty=True
        self.assertGreaterEqual(inv.call_count, 1)

    async def test_no_form_falls_back_to_full_tree(self):
        # Page has no <form> ancestors (e.g. SPA with floating search).
        tree = _make_tree([
            _elem("r1", role="textbox", name="Search", form_id=""),
        ])
        actions = [{"type": "form_input", "ref": "r1", "value": "octopus"}]
        plan = bdp.DomPlan(
            thinking="search", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        batch = _batch_result(all_actions=actions)

        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(return_value=plan)) as p, \
             patch.object(bde, "execute_dom_batch", new=AsyncMock(return_value=batch)):
            result = await bdo.run_dom_task("search octopus", MagicMock())

        self.assertTrue(result.success)
        # Planner received the full tree (1 element, no scoping)
        called_tree = p.call_args[0][1]  # second positional arg
        self.assertEqual(len(called_tree.elements), 1)


# ─── P13: the state machine ──────────────────────────────────────────────
#
# TENKA-v2 §17.P13. This loop gave up its own status vocabulary for
# `core/verdict.py`'s. These tests pin the two properties the phase requires:
# identical observable behaviour (the 25 tests above, unchanged), and no new
# success path.


class TestReasonOutcomeTable(unittest.TestCase):
    """`speaks_as_done(outcome)` must agree with the old boolean, tag for tag.

    The migration's whole claim is that `success` carried no information the
    `reason` tag did not already carry. This is that claim as an assertion,
    enumerated rather than sampled: the three tags that were constructed with
    `success=True` in the pre-P13 tree, and the twelve that were constructed
    with `success=False`, taken from all 31 construction sites.

    If a future tag is added to the loop and mapped so that it speaks as done,
    this test is what says so.
    """

    # Exactly the tags that appeared as `success=True` before P13.
    SPOKE_AS_DONE = frozenset({
        "completed", "completed_no_actions", "completed_no_submit",
    })
    # Exactly the tags that appeared as `success=False` before P13.
    DID_NOT = frozenset({
        "max_loops", "max_attempts", "validation_unresolved",
        "validation_no_progress", "user_value_rejected", "planner_failed",
        "mapper_failed", "fills_failed", "submit_failed",
        "loop_failure_at_max", "perceive_failed", "empty_tree",
    })

    def test_table_covers_every_tag_and_nothing_else(self):
        # A tag in the loop with no row falls through to _UNMAPPED_OUTCOME,
        # and a row with no tag is dead weight that will drift. Both are
        # bugs, so the table and the census must match exactly.
        self.assertEqual(
            set(bdo._REASON_OUTCOME), self.SPOKE_AS_DONE | self.DID_NOT,
        )

    def test_success_matches_pre_p13_boolean_for_every_tag(self):
        for reason in sorted(self.SPOKE_AS_DONE):
            with self.subTest(reason=reason):
                self.assertTrue(bdo.DomTaskResult(reason=reason).success)
        for reason in sorted(self.DID_NOT):
            with self.subTest(reason=reason):
                self.assertFalse(bdo.DomTaskResult(reason=reason).success)

    def test_only_completed_is_evidence_of_success(self):
        # The three done-speaking tags are not equally strong, and flattening
        # them back into a boolean is what P13 removed. `completed_no_actions`
        # dispatched nothing; `completed_no_submit` never pressed submit.
        # Both may speak as done (V6) and neither is evidence (V4).
        self.assertIs(
            bdo.DomTaskResult(reason="completed").outcome, Outcome.SUCCEEDED,
        )
        for reason in ("completed_no_actions", "completed_no_submit"):
            with self.subTest(reason=reason):
                r = bdo.DomTaskResult(reason=reason)
                self.assertIs(r.outcome, Outcome.UNVERIFIED)
                self.assertFalse(r.outcome.is_evidence_of_success)
                self.assertTrue(r.success)

    def test_budget_exhaustion_is_uncertain_not_failed(self):
        # Nobody looked and nobody knows. Reporting FAILED here would be the
        # inverse of KI-31: positive evidence claimed where there is none.
        for reason in ("max_loops", "max_attempts", "validation_unresolved"):
            with self.subTest(reason=reason):
                self.assertIs(
                    bdo.DomTaskResult(reason=reason).outcome,
                    Outcome.UNCERTAIN,
                )

    def test_no_foundation_is_unsupported(self):
        # The two tags router.py already routes to the vision tier.
        for reason in ("perceive_failed", "empty_tree"):
            with self.subTest(reason=reason):
                self.assertIs(
                    bdo.DomTaskResult(reason=reason).outcome,
                    Outcome.UNSUPPORTED,
                )

    def test_unmapped_reason_fails_closed(self):
        # A tag added to the loop and forgotten here must not read as done.
        # Same shape as `brain/task.may_transition` answering False for an
        # unknown state: the omission goes nowhere good, loudly.
        r = bdo.DomTaskResult(reason="some_tag_nobody_mapped")
        with self.assertLogs("browser_dom_orchestrator", level="WARNING") as cm:
            self.assertIs(r.outcome, Outcome.FAILED)
        self.assertIn("no row in _REASON_OUTCOME", "\n".join(cm.output))
        self.assertFalse(r.success)

    def test_nothing_outside_the_completed_tags_speaks_as_done(self):
        # The invariant, stated over the table itself rather than over the
        # census: nothing that is not a `completed_*` tag may speak as done.
        # This is what catches a *new* tag mapped to SUCCEEDED or UNVERIFIED.
        for reason, outcome in bdo._REASON_OUTCOME.items():
            with self.subTest(reason=reason):
                self.assertEqual(
                    speaks_as_done(outcome), reason in self.SPOKE_AS_DONE,
                )


# ─── P13: abort is not a failure mode ────────────────────────────────────


class TestAbortIsNotAFallback(unittest.IsolatedAsyncioTestCase):
    """ESC held mid-batch must reach the caller as `UserAborted`.

    `dom_executor.execute_dom_batch` raises it at every action boundary and
    `run_dom_task` deliberately does not catch it. The hole was one level up:
    `router._execute_dom_task` caught it with a bare `except Exception` and
    returned `"__FALLBACK__"`, which is not an error string but an instruction
    to escalate to the vision tier -- so an abort re-triggered TTS, minimized
    the user's terminals and spent a vision call before anything stopped.

    `.claude/rules/automation.md`: "Never swallow `UserAborted` into a string
    error."
    """

    async def test_orchestrator_lets_user_aborted_through(self):
        tree = _make_tree([
            _elem("r1", form_id="form-0", role="textbox", name="A"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        actions = [{"type": "form_input", "ref": "r1", "value": "X"}]
        plan = bdp.DomPlan(
            thinking="fill", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=UserAborted("esc_hold"))):
            with self.assertRaises(UserAborted):
                await bdo.run_dom_task("fill the form", MagicMock())

    async def _run_router_with_abort(self):
        """Drive `router._execute_dom_task` to the point of abort.

        The driver is stubbed on the module the deferred
        `from .browser import handle` binds, which is where the lookup happens
        at call time. There is no page-selection stub any more: the extension
        resolves every verb to the active tab itself.
        """
        page = MagicMock()
        page.url = "https://example.invalid/form"
        handle = MagicMock()
        handle.kind = "latch"
        handle.page = page

        with patch.object(bhandle, "get_browser_handle",
                          new=AsyncMock(return_value=handle)),              patch.object(bdo, "run_dom_task",
                          new=AsyncMock(side_effect=UserAborted("esc_hold"))):
            return await router._execute_dom_task("do the thing")

    async def test_router_reraises_instead_of_falling_back(self):
        with self.assertRaises(UserAborted):
            await self._run_router_with_abort()

    async def test_router_does_not_log_abort_as_a_crash(self):
        # The log line is the user-visible symptom in the transcript, and it
        # is what made this look like an orchestrator bug rather than an
        # abort. Asserted separately from the raise because a guard placed
        # after the `logger.error` would pass the test above and still lie.
        with self.assertLogs(router.logger, level="INFO") as cm:
            with self.assertRaises(UserAborted):
                await self._run_router_with_abort()
        self.assertNotIn("crashed", "\n".join(cm.output))


class TestTheRaiseHasSomewhereToLand(unittest.TestCase):
    """The receiver of the re-raise, pinned structurally.

    CLAUDE.md rule 10: a boundary is only as good as the enumeration of paths
    around it. Making `_execute_dom_task` raise is worth nothing if its caller
    stops re-raising, and that caller is code this commit did not touch — so
    nothing else would notice. `da_handlers.handle_computer_task` wraps the
    `execute_automation` call in `except UserAborted: raise` and answers it
    with "Stopped." at the function level.

    Structural rather than executed, deliberately: driving `handle_computer_task`
    for real means importing and stubbing `automation.vision`, and a stub that
    misses (`stubs must patch where the import looks`) hands the mouse and
    keyboard to the vision loop. The end-to-end proof is the live test in the
    commit message; this is the regression guard that costs nothing.
    """

    def test_computer_task_reraises_abort_around_execute_automation(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).parent.parent / "assistant" / "actions"
               / "da_handlers.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "handle_computer_task")

        # The property is about *handler order*, not about a re-raise existing
        # somewhere in the function. The first draft of this test asserted the
        # latter and passed with the guard deleted: the function-level `try`
        # also wraps the call and also re-raises, so `any(...)` found it and
        # said nothing. Python picks the first matching handler, so what
        # matters is that no broad `except Exception` around this call can be
        # reached by a `UserAborted` first.
        def _is_broad(h: ast.ExceptHandler) -> bool:
            if h.type is None:
                return True
            names = ([e for e in h.type.elts] if isinstance(h.type, ast.Tuple)
                     else [h.type])
            return any(getattr(n, "id", None) in ("Exception", "BaseException")
                       for n in names)

        def _reraises_abort(h: ast.ExceptHandler) -> bool:
            return (getattr(h.type, "id", None) == "UserAborted"
                    and any(isinstance(s, ast.Raise) and s.exc is None
                            for s in ast.walk(h)))

        guarding = [
            t for t in ast.walk(fn)
            if isinstance(t, ast.Try)
            and any(isinstance(c, ast.Call)
                    and getattr(c.func, "attr", None) == "execute_automation"
                    for c in ast.walk(t))
        ]
        self.assertTrue(
            guarding,
            "handle_computer_task no longer calls execute_automation inside a "
            "try -- re-check where a raised UserAborted now lands",
        )

        unshielded = []
        for t in guarding:
            shielded = False
            for h in t.handlers:
                if _reraises_abort(h):
                    shielded = True
                elif _is_broad(h) and not shielded:
                    unshielded.append((h.lineno, ast.unparse(h.type)
                                       if h.type else "bare except"))
        self.assertFalse(
            unshielded,
            f"a broad handler around execute_automation is reachable by "
            f"UserAborted at {unshielded} -- an abort raised by "
            f"router._execute_dom_task decays into the vision-loop fallback "
            f"again, which is the defect P13 removed",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
