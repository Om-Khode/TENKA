"""
test_post_submit_navigation.py — Bug 8: post-submit navigation success detection.

Stub-based unit tests for the orchestrator's nav-success classification.
Three signals matter, each tested end-to-end via run_dom_task plus a
focused helper-level test:

  1. evaluate_failed (Playwright "Execution context destroyed")
  2. URL changed since the pre-submit baseline
  3. All baseline form refs absent-or-invisible after submit (Webflow soft
     transition)

Plus regression guards:
  - validation errors present → soft-transition signal MUST NOT fire
  - no prior submit → nav check is silent (orchestrator doesn't false-success
    on a page that arrives in transitional state)
  - URL change with same baseline-empty → no false success

Run: python test_post_submit_navigation.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom as bdom
import assistant.automation.browser.dom_planner as bdp
import assistant.automation.browser.dom_executor as bde
import assistant.automation.browser.dom_orchestrator as bdo


# ─── Helpers ─────────────────────────────────────────────────────────────


def _elem(ref, **kwargs) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input", form_id="", in_dialog=False,
        aria_invalid=False,
    )
    base.update(kwargs)
    return bdom.ElementInfo(ref=ref, **base)


def _make_tree(
    elements: list[bdom.ElementInfo],
    *,
    validation_errors: tuple[bdom.ValidationError, ...] = (),
    evaluate_failed: bool = False,
    url: str = "",
) -> bdom.PageDomTree:
    ref_map = {e.ref: MagicMock(name=f"loc-{e.ref}") for e in elements}
    return bdom.PageDomTree(
        elements=elements, ref_to_locator=ref_map, truncated=0,
        read_at=time.monotonic(), viewport=(1280, 800),
        validation_errors=validation_errors,
        evaluate_failed=evaluate_failed,
        url=url,
    )


def _batch_ok(actions: list[dict], *, tree_dirty: bool = True) -> bde.DomBatchResult:
    return bde.DomBatchResult(
        results=[bde.DomActionResult(action=a, succeeded=True) for a in actions],
        tree_dirty=tree_dirty,
    )


# ─── _post_submit_navigated helper ───────────────────────────────────────


class TestPostSubmitNavigated(unittest.TestCase):
    """Direct tests of the detection helper, isolated from the loop."""

    def test_evaluate_failed_fires_signal(self):
        tree = _make_tree([], evaluate_failed=True, url="https://x/")
        nav, reason = bdo._post_submit_navigated(
            tree, baseline_url="https://x/", baseline_refs={"r1"},
        )
        self.assertTrue(nav)
        self.assertIn("evaluate context", reason)

    def test_url_change_fires_signal(self):
        tree = _make_tree(
            [_elem("r1", visible=True)],
            url="https://x/thank-you",
        )
        nav, reason = bdo._post_submit_navigated(
            tree, baseline_url="https://x/form", baseline_refs={"r1"},
        )
        self.assertTrue(nav)
        self.assertIn("navigated", reason)
        self.assertIn("thank-you", reason)

    def test_url_unchanged_no_signal(self):
        tree = _make_tree(
            [_elem("r1", visible=True)],
            url="https://x/",
        )
        nav, _ = bdo._post_submit_navigated(
            tree, baseline_url="https://x/", baseline_refs={"r1"},
        )
        self.assertFalse(nav)

    def test_baseline_refs_all_invisible_fires_signal(self):
        # Webflow-style soft transition: form node still in DOM but visible=False.
        tree = _make_tree(
            [_elem("r1", visible=False), _elem("r2", visible=False)],
            url="https://x/",
        )
        nav, reason = bdo._post_submit_navigated(
            tree, baseline_url="https://x/", baseline_refs={"r1", "r2"},
        )
        self.assertTrue(nav)
        self.assertIn("transitioned away", reason)

    def test_baseline_refs_all_absent_fires_signal(self):
        # Hard removal: form node replaced with success message DOM.
        tree = _make_tree(
            [_elem("rNEW", visible=True, name="Thank you!")],
            url="https://x/",
        )
        nav, reason = bdo._post_submit_navigated(
            tree, baseline_url="https://x/", baseline_refs={"r1", "r2"},
        )
        self.assertTrue(nav)
        self.assertIn("transitioned away", reason)

    def test_baseline_partial_visible_no_signal(self):
        # If even ONE baseline ref is still visible, the form is still up.
        tree = _make_tree(
            [_elem("r1", visible=True), _elem("r2", visible=False)],
            url="https://x/",
        )
        nav, _ = bdo._post_submit_navigated(
            tree, baseline_url="https://x/", baseline_refs={"r1", "r2"},
        )
        self.assertFalse(nav)

    def test_validation_errors_block_soft_transition_signal(self):
        # All baseline refs invisible BUT validation errors present →
        # form is rebuilt-with-errors, not transitioned. Don't claim success.
        tree = _make_tree(
            [_elem("r1", visible=False)],
            url="https://x/",
            validation_errors=(
                bdom.ValidationError("r1", "Required field", "describedby"),
            ),
        )
        nav, _ = bdo._post_submit_navigated(
            tree, baseline_url="https://x/", baseline_refs={"r1"},
        )
        self.assertFalse(nav)

    def test_no_baseline_no_signal(self):
        # Without baseline state, only the strong evaluate_failed/url-change
        # signals can fire. An invisible-form perception alone cannot.
        tree = _make_tree(
            [_elem("r1", visible=False)],
            url="https://x/",
        )
        nav, _ = bdo._post_submit_navigated(
            tree, baseline_url="", baseline_refs=set(),
        )
        self.assertFalse(nav)

    def test_evaluate_failed_dominates_no_baseline(self):
        # Even without baseline, evaluate_failed alone fires (it's an
        # unambiguous mid-navigation signal). Note: orchestrator gates the
        # whole nav-check on baseline being set, so this only matters
        # if the helper is reused elsewhere — but the helper itself is honest.
        tree = _make_tree([], evaluate_failed=True, url="")
        nav, _ = bdo._post_submit_navigated(
            tree, baseline_url="", baseline_refs=set(),
        )
        self.assertTrue(nav)


# ─── End-to-end: run_dom_task with nav scenarios ─────────────────────────


class TestNavSuccessIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_evaluate_failed_after_submit_returns_success(self):
        # Loop 1: fill + submit. Loop 2: page is mid-navigation, perception
        # raises "Execution context was destroyed" → evaluate_failed=True.
        # Orchestrator must classify as success without burning a planner call.
        tree_before = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email"),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            url="https://site/form",
        )
        tree_navigating = _make_tree([], evaluate_failed=True, url="https://site/form")
        actions = [
            {"type": "form_input", "ref": "r1", "value": "x@y.z"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="fill+submit", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        plan_mock = AsyncMock(return_value=plan)
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[tree_before, tree_navigating])), \
             patch.object(bdp, "plan_dom_actions", new=plan_mock), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertTrue(result.success, msg=f"got {result!r}")
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.final_summary, "Form submitted.")
        self.assertEqual(plan_mock.await_count, 1, "no second planner call")

    async def test_url_change_after_submit_returns_success(self):
        # Loop 1: fill + submit on form URL. Loop 2: page navigated to
        # /thanks → URL different from baseline. Returns success.
        tree_before = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email"),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            url="https://site/form",
        )
        # Post-nav: a thank-you page might have unrelated elements OR be
        # mostly empty. Either way URL change alone is the signal.
        tree_thanks = _make_tree(
            [_elem("rNEW", role="link", name="Back to home")],
            url="https://site/thank-you",
        )
        actions = [
            {"type": "form_input", "ref": "r1", "value": "x@y.z"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="fill+submit", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        plan_mock = AsyncMock(return_value=plan)
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[tree_before, tree_thanks])), \
             patch.object(bdp, "plan_dom_actions", new=plan_mock), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(plan_mock.await_count, 1)

    async def test_webflow_soft_transition_returns_success(self):
        # Webflow: form stays in DOM, all inputs go visible=False, no errors.
        tree_before = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email", visible=True),
                _elem("r2", form_id="form-0", role="button", name="Submit", visible=True),
            ],
            url="https://webflow.io/",
        )
        # Same URL, same refs, but everything invisible after submit.
        tree_after = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email", visible=False),
                _elem("r2", form_id="form-0", role="button", name="Submit", visible=False),
            ],
            url="https://webflow.io/",
        )
        actions = [
            {"type": "form_input", "ref": "r1", "value": "x@y.z"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="fill+submit", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        plan_mock = AsyncMock(return_value=plan)
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[tree_before, tree_after])), \
             patch.object(bdp, "plan_dom_actions", new=plan_mock), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")

    async def test_validation_errors_with_invisible_refs_does_not_succeed(self):
        # Regression guard: form rebuilds with errors, refs go invisible,
        # but validation_errors is non-empty → MUST NOT report success.
        tree_before = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email", visible=True),
                _elem("r2", form_id="form-0", role="button", name="Submit", visible=True),
            ],
            url="https://x/",
        )
        tree_with_errors_invisible = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email", visible=False),
                _elem("r2", form_id="form-0", role="button", name="Submit", visible=False),
            ],
            url="https://x/",
            validation_errors=(
                bdom.ValidationError("r1", "Invalid email", "describedby"),
            ),
        )
        actions = [
            {"type": "form_input", "ref": "r1", "value": "bad"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="fill+submit", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        # Loop 2 will fall through to corrective fill. Loop 3 (also errors)
        # eventually hits no-progress bail or replan exhaustion. Either
        # way, the nav-success path MUST NOT fire.
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[
                              tree_before,
                              tree_with_errors_invisible,
                              tree_with_errors_invisible,
                              tree_with_errors_invisible,
                              tree_with_errors_invisible,
                          ])), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertFalse(result.success,
                         msg=f"nav-success false-fired with errors present: {result!r}")

    async def test_first_perception_evaluate_failed_no_prior_submit_does_not_succeed(self):
        # Test 2 scenario: TENKA arrives at a page that's mid-navigation,
        # NO prior submit happened in this run. The orchestrator must NOT
        # claim success — there's no submit to verify against. Falls
        # through to the existing empty_tree retry logic.
        empty_navigating = _make_tree([], evaluate_failed=True, url="https://x/")
        # Make subsequent perceptions return a stable invisible-form state
        # so we exit cleanly via planner_failed (not nav-success).
        invisible_tree = _make_tree(
            [_elem("r1", visible=False)],
            url="https://x/",
        )
        empty_plan = bdp.DomPlan(
            thinking="all invisible", plan="",
            actions=[], done=False, needs_reperceive=False,
            rejection_notes=["all refs invisible"],
        )
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[
                              empty_navigating,
                              invisible_tree, invisible_tree,
                              invisible_tree, invisible_tree,
                          ])), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(return_value=empty_plan)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok([]))):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        # Exact reason isn't load-bearing for this guard — the only
        # invariant is "nav-success did NOT fire". Any non-success outcome
        # satisfies it.
        self.assertFalse(result.success,
                         msg=f"nav-success false-fired w/o prior submit: {result!r}")

    async def test_nav_detected_on_loop_3_after_corrective_batch(self):
        # Test 3 scenario in compressed form: loop 1 submit succeeds,
        # loop 2 sees errors and runs a corrective fill + resubmit,
        # loop 3 perception finds the form transitioned away. The baseline
        # must persist across the error-replan path so loop 3 catches it.
        tree_before = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email", visible=True),
                _elem("r2", form_id="form-0", role="button", name="Submit", visible=True),
            ],
            url="https://x/",
        )
        tree_after_bad = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email",
                      visible=True, aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit", visible=True),
            ],
            url="https://x/",
            validation_errors=(
                bdom.ValidationError("r1", "Invalid email", "describedby"),
            ),
        )
        # Loop 3 perception: form transitioned away (Webflow soft).
        tree_transitioned = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox", name="Email", visible=False),
                _elem("r2", form_id="form-0", role="button", name="Submit", visible=False),
            ],
            url="https://x/",
        )
        actions_1 = [
            {"type": "form_input", "ref": "r1", "value": "bad"},
            {"type": "click_ref", "ref": "r2"},
        ]
        actions_2 = [
            {"type": "form_input", "ref": "r1", "value": "good@x.com"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan_1 = bdp.DomPlan(
            thinking="fill+submit", plan="initial",
            actions=actions_1, done=True, needs_reperceive=False,
        )
        plan_2 = bdp.DomPlan(
            thinking="corrective", plan="fix",
            actions=actions_2, done=True, needs_reperceive=False,
        )

        plan_calls = []

        async def _plan_se(goal, scoped, *, feedback=""):
            plan_calls.append(feedback)
            return plan_1 if len(plan_calls) == 1 else plan_2

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[
                              tree_before, tree_after_bad, tree_transitioned,
                          ])), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(side_effect=_plan_se)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=[
                              _batch_ok(actions_1), _batch_ok(actions_2),
                          ])):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertTrue(result.success, msg=f"got {result!r}")
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.final_summary, "Form submitted.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
