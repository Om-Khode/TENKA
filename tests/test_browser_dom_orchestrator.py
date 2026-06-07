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


if __name__ == "__main__":
    unittest.main(verbosity=2)
