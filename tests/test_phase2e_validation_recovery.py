"""
test_phase2e_validation_recovery.py — Phase 2E: post-submit validation recovery.

Stub-based unit tests for the orchestrator's submit-detection and
validation-error feedback path, plus the perception-side hydration of
ValidationError records from the JS pass output.

Coverage:
  - _batch_contained_submit: submit-shape clicks, type="submit" inputs,
    non-submit clicks (Cancel, opening dropdown), wrong-shape actions.
  - _format_validation_for_planner: per-field anchored, page-level,
    truncation cap.
  - _scope_tree_to_elements: filters validation_errors to the scoped form
    (drops out-of-scope refs, keeps page-level).
  - Hydration: read_page_dom(stub) builds ValidationError records from a
    raw JS payload, mapping field_idx → ref via the captured elements.
  - Hydration: errors anchored to a pruned/unknown idx are dropped (so
    the planner never sees a feedback line referencing an invisible field).
  - Orchestrator integration:
      * happy submit + clean post-submit perceive → success on loop 2
        without a second plan call
      * submit + post-submit errors → planner gets validation feedback
        and emits a corrective fill batch
      * post-submit override is suppressed at the last loop (no budget
        for verification — accept what we have)

Run: python test_phase2e_validation_recovery.py
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


# ─── Helpers (mirrored from test_browser_dom_orchestrator.py) ────────────


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
) -> bdom.PageDomTree:
    ref_map = {e.ref: MagicMock(name=f"loc-{e.ref}") for e in elements}
    return bdom.PageDomTree(
        elements=elements, ref_to_locator=ref_map, truncated=0,
        read_at=time.monotonic(), viewport=(1280, 800),
        validation_errors=validation_errors,
    )


def _batch_ok(actions: list[dict], *, tree_dirty: bool = True) -> bde.DomBatchResult:
    return bde.DomBatchResult(
        results=[bde.DomActionResult(action=a, succeeded=True) for a in actions],
        tree_dirty=tree_dirty,
    )


# ─── _batch_contained_submit ─────────────────────────────────────────────


class TestBatchContainedSubmit(unittest.TestCase):
    def test_submit_button_click_detected(self):
        tree = _make_tree([
            _elem("r1", role="textbox", name="Email"),
            _elem("r2", role="button", name="Submit"),
        ])
        actions = [
            {"type": "form_input", "ref": "r1", "value": "x@y.z"},
            {"type": "click_ref", "ref": "r2"},
        ]
        self.assertTrue(bdo._batch_contained_submit(actions, tree))

    def test_input_type_submit_detected(self):
        # <input type="submit"> is a button without role=button per impliedRole
        # — we treat it as submit via the type field.
        tree = _make_tree([
            _elem("r1", role="button", name="Go", type="submit"),
        ])
        actions = [{"type": "click_ref", "ref": "r1"}]
        self.assertTrue(bdo._batch_contained_submit(actions, tree))

    def test_non_submit_click_not_detected(self):
        tree = _make_tree([
            _elem("r1", role="button", name="Cancel"),
            _elem("r2", role="combobox", name="State"),
        ])
        actions = [
            {"type": "click_ref", "ref": "r2"},  # opening a dropdown
            {"type": "click_ref", "ref": "r1"},  # Cancel — not a submit token
        ]
        self.assertFalse(bdo._batch_contained_submit(actions, tree))

    def test_form_input_only_not_detected(self):
        tree = _make_tree([_elem("r1", role="textbox", name="Email")])
        actions = [{"type": "form_input", "ref": "r1", "value": "x"}]
        self.assertFalse(bdo._batch_contained_submit(actions, tree))

    def test_unknown_ref_skipped(self):
        # Defensive: action refs a button not in the tree (post-validation
        # drift edge case). Helper must not crash.
        tree = _make_tree([_elem("r1", role="textbox")])
        actions = [{"type": "click_ref", "ref": "ghost"}]
        self.assertFalse(bdo._batch_contained_submit(actions, tree))

    def test_empty_actions(self):
        tree = _make_tree([_elem("r1", role="button", name="Submit")])
        self.assertFalse(bdo._batch_contained_submit([], tree))


# ─── _format_validation_for_planner ──────────────────────────────────────


class TestFormatValidationForPlanner(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(
            bdo._format_validation_for_planner((), [_elem("r1")]),
            "",
        )

    def test_field_anchored_and_page_level(self):
        elements = [
            _elem("r1", role="textbox", name="Email"),
            _elem("r2", role="textbox", name="Phone"),
        ]
        errs = (
            bdom.ValidationError("r1", "Please enter a valid email", "describedby"),
            bdom.ValidationError("", "Some fields are missing", "alert"),
        )
        out = bdo._format_validation_for_planner(errs, elements)
        self.assertIn("VALIDATION ERRORS FROM PRIOR SUBMIT", out)
        self.assertIn("'Email'", out)
        self.assertIn("ref=r1", out)
        self.assertIn("Please enter a valid email", out)
        self.assertIn("page-level", out)

    def test_unknown_ref_falls_back(self):
        # field_ref not in elements → "<unnamed field>" placeholder.
        out = bdo._format_validation_for_planner(
            (bdom.ValidationError("r9", "Bad", "alert"),), [_elem("r1")],
        )
        self.assertIn("ref=r9", out)
        self.assertIn("<unnamed field>", out)

    def test_truncates_at_max_lines(self):
        elements = [_elem(f"r{i}", name=f"F{i}") for i in range(10)]
        errs = tuple(
            bdom.ValidationError(f"r{i}", f"err{i}", "alert") for i in range(10)
        )
        out = bdo._format_validation_for_planner(errs, elements, max_lines=3)
        self.assertIn("err0", out)
        self.assertIn("err2", out)
        self.assertNotIn("err5", out)
        self.assertIn("7 more", out)


# ─── _scope_tree_to_elements forwarding of validation errors ─────────────


class TestScopeTreeForwardsValidation(unittest.TestCase):
    def test_drops_out_of_scope_keeps_page_level(self):
        e_in = _elem("r1", form_id="form-0")
        e_out = _elem("r2", form_id="form-1")
        errs = (
            bdom.ValidationError("r1", "in scope error", "alert"),
            bdom.ValidationError("r2", "out of scope error", "alert"),
            bdom.ValidationError("", "page-level", "alert"),
        )
        full = _make_tree([e_in, e_out], validation_errors=errs)
        scoped = bdo._scope_tree_to_elements(full, [e_in])
        msgs = [ve.message for ve in scoped.validation_errors]
        self.assertIn("in scope error", msgs)
        self.assertIn("page-level", msgs)
        self.assertNotIn("out of scope error", msgs)


# ─── read_page_dom hydration of validation_errors ────────────────────────


class TestAriaInvalidJsContract(unittest.TestCase):
    """
    Bug 10 (2026-04-28): the JS perception pass must NOT match
    `el.matches(':invalid')` when computing aria_invalid. The HTML5
    `:invalid` pseudo-class fires on any unmet `required`/`pattern`/
    `minlength`/`type=email` constraint — even for fields the form's
    visible UI accepts. On Webflow forms with strict patterns this
    generated 6+ spurious synthetic errors per perception, drowning
    the one real rejection in noise.

    Live evidence (2026-04-28 20:54): Truein form perception found
    7 validation errors (1 real + 6 synthetic) when ONLY the
    Contact Number was actually rejected per the form's UI.

    Regression guard: assert `:invalid` no longer appears in the JS
    block that computes `ariaInvalid`. Acceptable elsewhere (e.g. in
    a class-name selector matching `[class*="invalid"]` for error-
    container detection — that's unrelated and intentional).
    """

    def test_js_does_not_call_matches_invalid_for_aria_invalid(self):
        # The aria-invalid computation block must not call
        # `el.matches(':invalid')` — that's the noisy HTML5 pseudo-class
        # that fired on every unmet pattern/minlength/required constraint
        # and drowned real rejections in synthetic noise (Bug 10).
        # The whole JS source is checked: the call signature itself is the
        # canonical anti-pattern (any place re-introduces it would re-break
        # the contract).
        js = bdom._DOM_QUERY_JS
        # Defensive: allow the string `:invalid` in comments and in the
        # unrelated [class*="invalid"] error-container CSS selector — only
        # the .matches(':invalid') call is forbidden.
        self.assertNotIn(
            "matches(':invalid')", js,
            msg=(
                "aria-invalid computation must not call "
                "el.matches(':invalid') — pseudo-class fires on any "
                "unmet HTML5 constraint, generating spurious synthetic "
                "errors (Bug 10)."
            ),
        )
        self.assertNotIn(
            'matches(":invalid")', js,
            msg="(double-quoted variant of the same forbidden call)",
        )

    def test_synthetic_error_loop_gates_on_aria_invalid_attr(self):
        # The synthetic-error generation loop must iterate over fields
        # whose aria_invalid is True (now meaning explicit attr only).
        # Sanity guard that no PR accidentally re-broadens the gate.
        js = bdom._DOM_QUERY_JS
        # Find the synthetic loop preamble.
        marker = "Synthetic entries for aria-invalid fields"
        idx = js.find(marker)
        self.assertGreater(idx, -1, "synthetic-error loop preamble missing")
        loop_block = js[idx:idx + 800]
        self.assertIn("if (!cap.aria_invalid) continue", loop_block)


class TestReadPageDomHydratesValidation(unittest.IsolatedAsyncioTestCase):
    """
    Stub a Playwright `Page` whose `evaluate()` returns a fabricated payload
    matching the shape the JS function emits. The Python-side hydration
    should map field_idx → ref correctly and drop errors anchored to refs
    pruned by the token budget or never captured.
    """

    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    def _stub_page(self, payload: dict):
        page = MagicMock(name="page")
        page.evaluate = AsyncMock(return_value=payload)
        page.locator = MagicMock(side_effect=lambda sel: MagicMock(name=f"loc({sel})"))
        return page

    async def test_hydrates_field_anchored_error(self):
        payload = {
            "elements": [
                {
                    "idx": 0, "tag": "input", "role": "textbox",
                    "name": "Email", "placeholder": "", "value": "",
                    "options": [], "bounds": [10, 10, 100, 30],
                    "visible": True, "enabled": True, "type": "email",
                    "form_id": "form-0", "in_dialog": False,
                    "aria_invalid": True,
                },
            ],
            "viewport": [1280, 800],
            "validation_errors": [
                {
                    "field_idx": 0,
                    "message": "Please enter a valid email address",
                    "source": "describedby",
                },
            ],
        }
        page = self._stub_page(payload)
        tree = await bdom.read_page_dom(page, use_cache=False)

        self.assertEqual(len(tree.elements), 1)
        self.assertTrue(tree.elements[0].aria_invalid)
        self.assertEqual(len(tree.validation_errors), 1)
        ve = tree.validation_errors[0]
        self.assertEqual(ve.field_ref, tree.elements[0].ref)
        self.assertEqual(ve.message, "Please enter a valid email address")
        self.assertEqual(ve.source, "describedby")

    async def test_hydrates_page_level_error(self):
        payload = {
            "elements": [],
            "viewport": [1280, 800],
            "validation_errors": [
                {"field_idx": -1, "message": "Form rejected", "source": "alert"},
            ],
        }
        tree = await bdom.read_page_dom(self._stub_page(payload), use_cache=False)
        self.assertEqual(len(tree.validation_errors), 1)
        self.assertEqual(tree.validation_errors[0].field_ref, "")
        self.assertEqual(tree.validation_errors[0].source, "alert")

    async def test_drops_error_for_unknown_field_idx(self):
        # field_idx points at an element not in the captured list — drop
        # rather than letting the planner see a dangling ref.
        payload = {
            "elements": [
                {
                    "idx": 0, "tag": "input", "role": "textbox",
                    "name": "Email", "placeholder": "", "value": "",
                    "options": [], "bounds": [0, 0, 100, 30],
                    "visible": True, "enabled": True, "type": "email",
                    "form_id": "", "in_dialog": False, "aria_invalid": False,
                },
            ],
            "viewport": [1280, 800],
            "validation_errors": [
                {"field_idx": 999, "message": "Orphan", "source": "alert"},
            ],
        }
        tree = await bdom.read_page_dom(self._stub_page(payload), use_cache=False)
        self.assertEqual(len(tree.validation_errors), 0)

    async def test_skips_empty_messages(self):
        payload = {
            "elements": [],
            "viewport": [1280, 800],
            "validation_errors": [
                {"field_idx": -1, "message": "", "source": "alert"},
                {"field_idx": -1, "message": "   ", "source": "alert"},
            ],
        }
        tree = await bdom.read_page_dom(self._stub_page(payload), use_cache=False)
        self.assertEqual(len(tree.validation_errors), 0)


# ─── Orchestrator integration: post-submit override end-to-end ──────────


class TestPostSubmitOverride(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_clean_post_submit_short_circuits_to_success(self):
        # Loop 1: fill + submit; planner says done. Submit click detected.
        # Loop 2: post-submit perceive returns no validation errors →
        # success with one planner call.
        tree = _make_tree([
            _elem("r1", form_id="form-0", role="textbox", name="Email"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        actions = [
            {"type": "form_input", "ref": "r1", "value": "test@example.com"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="fill+submit", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        plan_mock = AsyncMock(return_value=plan)
        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=plan_mock), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.loops_used, 2)
        # Planner called once — post-submit pass short-circuits without
        # invoking the LLM when no errors are present.
        self.assertEqual(plan_mock.await_count, 1)

    async def test_post_submit_errors_trigger_corrective_fill(self):
        # Loop 1: fill + submit; submit detected. Loop 2: perceive returns
        # a validation error; planner receives the error feedback and
        # emits a corrective fill + resubmit. Loop 3: post-submit clean.
        tree_before = _make_tree([
            _elem("r1", form_id="form-0", role="textbox",
                  name="Email", value=""),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        # After first submit: same fields, but Email has aria_invalid=True
        # and the tree carries a ValidationError pointing at r1.
        tree_after_bad = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox",
                      name="Email", value="bad", aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError(
                    "r1", "Please enter a valid email address", "describedby",
                ),
            ),
        )
        # After corrective fill + second submit: clean tree, no errors.
        tree_after_good = _make_tree([
            _elem("r1", form_id="form-0", role="textbox",
                  name="Email", value="test@example.com"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])

        actions_1 = [
            {"type": "form_input", "ref": "r1", "value": "bad"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan_1 = bdp.DomPlan(
            thinking="fill+submit", plan="initial",
            actions=actions_1, done=True, needs_reperceive=False,
        )
        actions_2 = [
            {"type": "form_input", "ref": "r1", "value": "test@example.com"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan_2 = bdp.DomPlan(
            thinking="fix email", plan="corrective",
            actions=actions_2, done=True, needs_reperceive=False,
        )

        # Perception sequence: initial → after-bad → after-good
        perceive_seq = [tree_before, tree_after_bad, tree_after_good]

        plan_calls: list = []

        async def _plan_side_effect(goal, scoped, *, feedback=""):
            plan_calls.append({"feedback": feedback,
                               "n_errors": len(scoped.validation_errors)})
            return plan_1 if len(plan_calls) == 1 else plan_2

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=perceive_seq)), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(side_effect=_plan_side_effect)), \
             patch.object(bde, "execute_dom_batch", new=AsyncMock(side_effect=[
                 _batch_ok(actions_1), _batch_ok(actions_2),
             ])):
            result = await bdo.run_dom_task("fill the form", MagicMock())

        self.assertTrue(result.success, msg=f"got {result!r}")
        self.assertEqual(result.reason, "completed")
        # 1 plan+exec, 1 post-submit-with-errors plan+exec, 1 final clean check
        self.assertEqual(result.loops_used, 3)
        # Planner called twice (NOT three times — the final post-submit
        # short-circuit doesn't invoke the LLM when the tree is clean).
        self.assertEqual(len(plan_calls), 2)
        # The corrective plan call must have received the validation feedback.
        self.assertEqual(plan_calls[0]["feedback"], "")
        self.assertIn("VALIDATION ERRORS", plan_calls[1]["feedback"])
        self.assertIn("Email", plan_calls[1]["feedback"])
        self.assertIn("valid email", plan_calls[1]["feedback"])

    async def test_last_loop_inline_verify_clean(self):
        # Bug 4 fix: when the last loop ends in submit, the orchestrator
        # MUST inline-verify the page before claiming success. With no
        # validation errors after the submit, success is reported.
        tree = _make_tree([
            _elem("r1", form_id="form-0", role="button", name="Submit"),
        ])
        actions = [{"type": "click_ref", "ref": "r1"}]
        plan = bdp.DomPlan(
            thinking="just submit", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )
        with patch.object(bdom, "read_page_dom", new=AsyncMock(return_value=tree)), \
             patch.object(bdp, "plan_dom_actions", new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("g", MagicMock(), max_loops=1)

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.loops_used, 1)
        self.assertEqual(result.final_summary, "Form submitted.")


# ─── Bug 4: last-loop inline verification with errors ────────────────────


class TestLastLoopInlineVerify(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_last_loop_with_lingering_errors_reports_failure(self):
        # Bug 4: max_loops=1, planner says done=True after submit, but the
        # inline verify perceive finds validation errors. Old behaviour:
        # report SUCCESS (false positive). New behaviour: report
        # loop_failure_at_max with a TTS-friendly summary naming the field.
        tree_before = _make_tree([
            _elem("r1", form_id="form-0", role="textbox", name="Phone"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        tree_after_bad = _make_tree(
            [
                _elem("r1", form_id="form-0", role="textbox",
                      name="Contact Number", value="9999999",
                      aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError(
                    "r1", "Please enter a valid phone number.", "describedby",
                ),
            ),
        )
        actions = [
            {"type": "form_input", "ref": "r1", "value": "9999999"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="fill+submit", plan="ok",
            actions=actions, done=True, needs_reperceive=False,
        )

        # Perception sequence: initial perceive (clean) → inline verify (errors)
        perceive_seq = [tree_before, tree_after_bad]

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=perceive_seq)), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("fill the form", MagicMock(),
                                            max_loops=1)

        # The critical assertion — was SUCCESS, must now be FAILURE.
        self.assertFalse(result.success, msg=f"got {result!r}")
        self.assertEqual(result.reason, "loop_failure_at_max")
        self.assertEqual(result.loops_used, 1)
        # Final summary must mention the offending field by NAME.
        self.assertIn("Contact Number", result.final_summary)
        # Must be TTS-friendly: short, no refs/codes/paths.
        self.assertLess(len(result.final_summary), 200)
        self.assertNotIn("ref", result.final_summary)
        self.assertNotIn("aria_invalid", result.final_summary)

    async def test_last_loop_inline_verify_failure_returns_short_summary(self):
        # Multiple flagged fields → summary lists them, capped at 3.
        before = _make_tree([
            _elem("r1", form_id="form-0", name="Email"),
            _elem("r2", form_id="form-0", name="Phone"),
            _elem("r3", form_id="form-0", name="ZIP"),
            _elem("r4", form_id="form-0", role="button", name="Submit"),
        ])
        after = _make_tree(
            [
                _elem("r1", form_id="form-0", name="Email", aria_invalid=True),
                _elem("r2", form_id="form-0", name="Phone", aria_invalid=True),
                _elem("r3", form_id="form-0", name="ZIP", aria_invalid=True),
                _elem("r4", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError("r1", "bad email", "describedby"),
                bdom.ValidationError("r2", "bad phone", "describedby"),
                bdom.ValidationError("r3", "bad zip", "describedby"),
            ),
        )
        actions = [{"type": "click_ref", "ref": "r4"}]
        plan = bdp.DomPlan(
            thinking="x", plan="x", actions=actions,
            done=True, needs_reperceive=False,
        )
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[before, after])), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(return_value=_batch_ok(actions))):
            result = await bdo.run_dom_task("g", MagicMock(), max_loops=1)
        self.assertFalse(result.success)
        self.assertIn("Email", result.final_summary)
        self.assertIn("Phone", result.final_summary)
        self.assertIn("ZIP", result.final_summary)


# ─── Bug 5: no-progress detection ────────────────────────────────────────


class TestValidationNoProgress(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_repeated_identical_errors_exits_early(self):
        # Bug 5: after a corrective pass, the same validation errors come
        # back unchanged → orchestrator must give up rather than burn the
        # remaining loop budget on identical re-fills.
        before = _make_tree([
            _elem("r1", form_id="form-0", name="Phone"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        bad_errs = (
            bdom.ValidationError(
                "r1", "Please enter a valid phone number.", "describedby",
            ),
        )
        after_bad_1 = _make_tree(
            [
                _elem("r1", form_id="form-0", name="Phone",
                      value="9999999", aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=bad_errs,
        )
        # Same errors after the corrective pass — no progress.
        after_bad_2 = _make_tree(
            [
                _elem("r1", form_id="form-0", name="Phone",
                      value="9999999", aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=bad_errs,
        )

        actions = [
            {"type": "form_input", "ref": "r1", "value": "9999999"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="x", plan="x", actions=actions,
            done=True, needs_reperceive=False,
        )

        # Perception sequence:
        #  loop 1: before
        #  loop 2: after_bad_1 (post-submit, errors detected)
        #  loop 3: after_bad_2 (post-submit, SAME errors → exit)
        perceive_seq = [before, after_bad_1, after_bad_2]

        plan_calls: list = []

        async def _plan_side_effect(goal, scoped, *, feedback=""):
            plan_calls.append(feedback)
            return plan

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=perceive_seq)), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(side_effect=_plan_side_effect)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=[
                              _batch_ok(actions), _batch_ok(actions),
                          ])):
            result = await bdo.run_dom_task(
                "fill the form", MagicMock(), max_loops=5,
            )

        # Must exit at loop 3 with no_progress, NOT burn loops 4 and 5.
        self.assertFalse(result.success, msg=f"got {result!r}")
        self.assertEqual(result.reason, "validation_no_progress")
        self.assertEqual(result.loops_used, 3)
        # Only 2 plan calls — the third is the no-progress exit before any
        # planning happens.
        self.assertEqual(len(plan_calls), 2)
        # Summary names the stuck field.
        self.assertIn("Phone", result.final_summary)

    async def test_progress_does_not_trigger_early_exit(self):
        # Negative case: errors change between passes → no early exit.
        # Loop 2 sees error A, loop 3 sees error B (different message),
        # loop 4 clean. Should reach success normally.
        before = _make_tree([
            _elem("r1", form_id="form-0", name="Phone"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        after_v1 = _make_tree(
            [
                _elem("r1", form_id="form-0", name="Phone",
                      value="9999999", aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError("r1", "Too short", "describedby"),
            ),
        )
        after_v2 = _make_tree(
            [
                _elem("r1", form_id="form-0", name="Phone",
                      value="abc", aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                # Different message — different signature.
                bdom.ValidationError("r1", "Must be digits", "describedby"),
            ),
        )
        clean = _make_tree([
            _elem("r1", form_id="form-0", name="Phone", value="1234567890"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        actions = [
            {"type": "form_input", "ref": "r1", "value": "x"},
            {"type": "click_ref", "ref": "r2"},
        ]
        plan = bdp.DomPlan(
            thinking="x", plan="x", actions=actions,
            done=True, needs_reperceive=False,
        )
        perceive_seq = [before, after_v1, after_v2, clean]

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=perceive_seq)), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(return_value=plan)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=[
                              _batch_ok(actions),
                              _batch_ok(actions),
                              _batch_ok(actions),
                          ])):
            result = await bdo.run_dom_task(
                "fill the form", MagicMock(), max_loops=5,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")


# ─── Bug 6: wasteful refill filter ───────────────────────────────────────


class TestFilterWastefulRefills(unittest.TestCase):
    def test_drops_fill_on_unflagged_field(self):
        actions = [
            {"type": "form_input", "ref": "r1", "value": "Alex"},     # flagged
            {"type": "form_input", "ref": "r2", "value": "Smith"},  # un-flagged
            {"type": "click_ref", "ref": "r9"},                     # always kept
        ]
        flagged = {"r1"}
        kept, dropped = bdo._filter_wasteful_refills(actions, flagged)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["ref"], "r1")
        self.assertEqual(kept[1]["ref"], "r9")

    def test_no_flagged_refs_returns_input_untouched(self):
        # Page-level errors only → no field-level signal → don't filter.
        actions = [
            {"type": "form_input", "ref": "r1", "value": "x"},
            {"type": "form_input", "ref": "r2", "value": "y"},
        ]
        kept, dropped = bdo._filter_wasteful_refills(actions, set())
        self.assertEqual(dropped, 0)
        self.assertEqual(kept, actions)

    def test_clicks_always_kept(self):
        # Clicks (re-submit) and selects on un-flagged fields must survive.
        actions = [
            {"type": "click_ref", "ref": "r5"},
            {"type": "select_option", "ref": "r6", "value": "X"},
        ]
        flagged = {"r1"}  # neither r5 nor r6 in flagged
        kept, dropped = bdo._filter_wasteful_refills(actions, flagged)
        self.assertEqual(dropped, 0)
        self.assertEqual(kept, actions)

    def test_non_dict_action_passes_through(self):
        actions = [
            "garbage",
            {"type": "form_input", "ref": "r1", "value": "x"},
        ]
        kept, dropped = bdo._filter_wasteful_refills(actions, {"r1"})
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)

    def test_empty_actions(self):
        kept, dropped = bdo._filter_wasteful_refills([], {"r1"})
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 0)


class TestFilterWastefulRefillsIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_corrective_pass_drops_unflagged_refills(self):
        # Loop 1: planner emits 2 fills + submit. Flagged set is empty
        # (initial pass) → no filter. After submit, only r2 has errors.
        # Loop 2: planner emits 2 fills + submit (refilling r1 too).
        # Filter must drop r1's fill, keeping only r2's fill + submit.
        before = _make_tree([
            _elem("r1", form_id="form-0", name="First Name"),
            _elem("r2", form_id="form-0", name="Phone"),
            _elem("r3", form_id="form-0", role="button", name="Submit"),
        ])
        after_bad = _make_tree(
            [
                _elem("r1", form_id="form-0", name="First Name", value="Alex"),
                _elem("r2", form_id="form-0", name="Phone",
                      value="9999999", aria_invalid=True),
                _elem("r3", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError("r2", "bad phone", "describedby"),
            ),
        )
        clean = _make_tree([
            _elem("r1", form_id="form-0", name="First Name", value="Alex"),
            _elem("r2", form_id="form-0", name="Phone", value="1234567890"),
            _elem("r3", form_id="form-0", role="button", name="Submit"),
        ])

        actions_1 = [
            {"type": "form_input", "ref": "r1", "value": "Alex"},
            {"type": "form_input", "ref": "r2", "value": "9999999"},
            {"type": "click_ref", "ref": "r3"},
        ]
        # Planner re-fills BOTH fields on the corrective pass — wasteful.
        actions_2 = [
            {"type": "form_input", "ref": "r1", "value": "Alex"},
            {"type": "form_input", "ref": "r2", "value": "1234567890"},
            {"type": "click_ref", "ref": "r3"},
        ]
        plan_1 = bdp.DomPlan(
            thinking="initial", plan="initial",
            actions=actions_1, done=True, needs_reperceive=False,
        )
        plan_2 = bdp.DomPlan(
            thinking="corrective", plan="corrective",
            actions=actions_2, done=True, needs_reperceive=False,
        )

        executor_calls: list = []

        async def _exec_side_effect(page, actions, ref_map):
            executor_calls.append(list(actions))
            return _batch_ok(actions)

        plan_n = [0]
        async def _plan_side_effect(goal, scoped, *, feedback=""):
            plan_n[0] += 1
            return plan_1 if plan_n[0] == 1 else plan_2

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[before, after_bad, clean])), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(side_effect=_plan_side_effect)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=_exec_side_effect)):
            result = await bdo.run_dom_task("fill", MagicMock(), max_loops=5)

        self.assertTrue(result.success)
        # Loop 1 executor saw all 3 actions (no flagged refs yet).
        self.assertEqual(len(executor_calls[0]), 3)
        # Loop 2 executor saw the FILTERED batch — r1's refill dropped.
        loop2_refs = [a.get("ref") for a in executor_calls[1]]
        self.assertNotIn("r1", loop2_refs, msg=f"got {executor_calls[1]!r}")
        self.assertIn("r2", loop2_refs)
        self.assertIn("r3", loop2_refs)


# ─── Helper: signature stability ─────────────────────────────────────────


class TestValidationSignature(unittest.TestCase):
    def test_signature_stable_across_order(self):
        a = (
            bdom.ValidationError("r1", "Too short", "describedby"),
            bdom.ValidationError("r2", "Required", "alert"),
        )
        b = (
            bdom.ValidationError("r2", "Required", "alert"),
            bdom.ValidationError("r1", "Too short", "describedby"),
        )
        self.assertEqual(bdo._validation_signature(a),
                         bdo._validation_signature(b))

    def test_signature_differs_when_message_changes(self):
        a = (bdom.ValidationError("r1", "Too short", "describedby"),)
        b = (bdom.ValidationError("r1", "Required", "describedby"),)
        self.assertNotEqual(bdo._validation_signature(a),
                            bdo._validation_signature(b))

    def test_signature_message_case_insensitive(self):
        # We lowercase the message before signing to absorb cosmetic
        # capitalisation differences from re-renders.
        a = (bdom.ValidationError("r1", "Too short", "describedby"),)
        b = (bdom.ValidationError("r1", "TOO SHORT", "describedby"),)
        self.assertEqual(bdo._validation_signature(a),
                         bdo._validation_signature(b))


# ─── Helper: TTS-friendly summary ────────────────────────────────────────


class TestFormatUnresolvedForSummary(unittest.TestCase):
    def test_single_field(self):
        elements = [_elem("r1", name="Phone")]
        errs = (bdom.ValidationError("r1", "bad", "x"),)
        s = bdo._format_unresolved_for_summary(errs, elements)
        self.assertIn("Phone", s)
        self.assertIn("valid value", s)
        self.assertLess(len(s), 200)

    def test_multiple_fields_listed(self):
        elements = [
            _elem("r1", name="Email"), _elem("r2", name="Phone"),
        ]
        errs = (
            bdom.ValidationError("r1", "x", "x"),
            bdom.ValidationError("r2", "x", "x"),
        )
        s = bdo._format_unresolved_for_summary(errs, elements)
        self.assertIn("Email", s)
        self.assertIn("Phone", s)

    def test_capped_at_five_when_errors_le_five(self):
        # Bug 9: adaptive cap — with 4 errors (≤ 5), all four are listed.
        elements = [
            _elem("r1", name="A"), _elem("r2", name="B"),
            _elem("r3", name="C"), _elem("r4", name="D"),
        ]
        errs = tuple(bdom.ValidationError(f"r{i}", "x", "x") for i in (1, 2, 3, 4))
        s = bdo._format_unresolved_for_summary(errs, elements)
        self.assertIn("A", s)
        self.assertIn("B", s)
        self.assertIn("C", s)
        self.assertIn("D", s)

    def test_capped_at_three_when_errors_gt_five(self):
        # Bug 9: with > 5 errors, cap stays at 3 to keep TTS short.
        elements = [
            _elem(f"r{i}", name=chr(ord("A") + i - 1))
            for i in range(1, 8)  # A..G
        ]
        errs = tuple(
            bdom.ValidationError(f"r{i}", "x", "x") for i in range(1, 8)
        )
        s = bdo._format_unresolved_for_summary(errs, elements)
        # First 3 named fields appear; 4th onward dropped.
        self.assertIn("A", s)
        self.assertIn("B", s)
        self.assertIn("C", s)
        self.assertNotIn("D", s)
        self.assertNotIn("G", s)

    def test_explicit_max_fields_override(self):
        # Caller can still pin the cap (backwards-compat / specialized callers).
        elements = [_elem(f"r{i}", name=chr(ord("A") + i - 1))
                    for i in range(1, 6)]
        errs = tuple(bdom.ValidationError(f"r{i}", "x", "x") for i in range(1, 6))
        s = bdo._format_unresolved_for_summary(errs, elements, max_fields=2)
        self.assertIn("A", s)
        self.assertIn("B", s)
        self.assertNotIn("C", s)

    def test_page_level_errors_only_falls_back_to_count(self):
        # No field-anchored errors → fallback summary.
        errs = (bdom.ValidationError("", "general", "alert"),)
        s = bdo._format_unresolved_for_summary(errs, [])
        self.assertIn("1 errors", s)
        self.assertLess(len(s), 200)

    def test_empty_errors_returns_done(self):
        s = bdo._format_unresolved_for_summary((), [])
        self.assertEqual(s, "Done.")

    def test_dedupe_case_insensitive(self):
        # Real-world: a form may anchor two errors at the same field with
        # different label casing ("First name" vs "First Name"). Should
        # appear ONCE in the summary, not twice.
        elements = [
            _elem("r1", name="First name"),
            _elem("r2", name="First Name"),
        ]
        errs = (
            bdom.ValidationError("r1", "required", "x"),
            bdom.ValidationError("r2", "required", "x"),
        )
        s = bdo._format_unresolved_for_summary(errs, elements)
        # The phrase "first name" appears exactly once (case-insensitive).
        self.assertEqual(s.lower().count("first name"), 1)


# ─── Bug 6 side-effect: empty-batch false positive ───────────────────────


class TestFilterEmptyBatchGuard(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_filter_does_not_empty_batch(self):
        # Bug 6 side-effect: when the filter would drop ALL planner actions
        # (refs disagree between planner and form anchors), we MUST keep
        # the original batch — letting the orchestrator return
        # completed_no_actions on an empty plan is a false-positive success.
        before = _make_tree([
            _elem("r1", form_id="form-0", name="Phone"),
            _elem("r2", form_id="form-0", role="button", name="Submit"),
        ])
        # Post-submit: form anchors error to r99 (a phantom ref the planner
        # doesn't know about — simulates re-render mismatch).
        after_bad = _make_tree(
            [
                _elem("r1", form_id="form-0", name="Phone",
                      value="9999999"),
                _elem("r99", form_id="form-0", name="Phone-rerendered",
                      aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError("r99", "bad phone", "describedby"),
            ),
        )
        # Loop 2's planner emits a fill on r1 (its known ref) — every action
        # would be dropped by the filter (r1 not in flagged={r99}).
        actions_1 = [
            {"type": "form_input", "ref": "r1", "value": "9999999"},
            {"type": "click_ref", "ref": "r2"},
        ]
        actions_2 = [
            {"type": "form_input", "ref": "r1", "value": "1234567890"},
        ]
        plan_1 = bdp.DomPlan(
            thinking="x", plan="x",
            actions=actions_1, done=True, needs_reperceive=False,
        )
        plan_2 = bdp.DomPlan(
            thinking="x", plan="x",
            actions=actions_2, done=True, needs_reperceive=False,
        )

        executor_calls: list = []

        async def _exec_side_effect(page, actions, ref_map):
            executor_calls.append(list(actions))
            return _batch_ok(actions)

        plan_n = [0]
        async def _plan_side_effect(goal, scoped, *, feedback=""):
            plan_n[0] += 1
            return plan_1 if plan_n[0] == 1 else plan_2

        # After loop 2's batch, post-submit shows still-bad — but the path
        # we care about is whether loop 2 SHORT-CIRCUITED to false success.
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[before, after_bad, after_bad])), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(side_effect=_plan_side_effect)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=_exec_side_effect)):
            result = await bdo.run_dom_task(
                "fill the form", MagicMock(), max_loops=3,
            )

        # The pre-fix bug: orchestrator returned success=True with reason
        # 'completed_no_actions' here. Post-fix: filter is bypassed when it
        # would empty the batch, so loop 2's batch executes (1 action), and
        # the run continues to its real conclusion (success or no-progress).
        # We assert the orchestrator did NOT short-circuit to a fake success.
        self.assertNotEqual(
            result.reason, "completed_no_actions",
            msg=f"filter empty-batch guard failed: {result!r}",
        )
        # Loop 2 must have executed something (proves filter did not empty).
        self.assertEqual(len(executor_calls), 2,
                         msg=f"got {len(executor_calls)} batches: {executor_calls!r}")
        self.assertEqual(len(executor_calls[1]), 1)


# ─── completed_no_actions guard: errors present ──────────────────────────


class TestCompletedNoActionsGuard(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_empty_plan_with_errors_reports_unresolved(self):
        # Defence in depth: even if the planner LEGITIMATELY emits
        # actions=[] done=True (which can happen when the planner mis-reads
        # the corrective state), the orchestrator must NOT declare success
        # while the scoped tree still carries validation_errors.
        bad_tree = _make_tree(
            [
                _elem("r1", form_id="form-0", name="Phone",
                      value="9999999", aria_invalid=True),
                _elem("r2", form_id="form-0", role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError("r1", "bad phone", "describedby"),
            ),
        )
        # Planner says "nothing to do", done=True — in production this
        # would have been the corrupt success path.
        plan = bdp.DomPlan(
            thinking="all good", plan="nothing left",
            actions=[], done=True, needs_reperceive=False,
        )

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(return_value=bad_tree)), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(return_value=plan)):
            result = await bdo.run_dom_task("g", MagicMock(), max_loops=2)

        self.assertFalse(result.success, msg=f"got {result!r}")
        self.assertEqual(result.reason, "validation_unresolved")
        # Summary names the offending field.
        self.assertIn("Phone", result.final_summary)

    async def test_empty_plan_with_no_errors_still_succeeds(self):
        # Negative: empty plan + clean tree → completed_no_actions remains
        # the correct success path. We didn't break the legitimate case.
        clean_tree = _make_tree([_elem("r1", form_id="form-0", name="X")])
        plan = bdp.DomPlan(
            thinking="x", plan="x",
            actions=[], done=True, needs_reperceive=False,
        )
        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(return_value=clean_tree)), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(return_value=plan)):
            result = await bdo.run_dom_task("g", MagicMock(), max_loops=2)

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed_no_actions")


if __name__ == "__main__":
    unittest.main(verbosity=2)
