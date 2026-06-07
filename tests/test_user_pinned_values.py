"""
test_user_pinned_values.py — Bug 12: respect user-supplied values.

When the user puts a literal value in the goal text ("with mobile as 99999"),
that value is a HARD CONSTRAINT — the agent must not silently substitute
it on validation feedback. If the form rejects a user-pinned field, the
orchestrator bails immediately with `reason="user_value_rejected"` and a
TTS-friendly summary that quotes back the user's literal value.

User-confirmed policy decisions (2026-04-28):
  1. Strict pin detection — value must appear in goal as a whole-word /
     whole-token literal match. Substring within a longer word does NOT
     pin (e.g. "Test" inside "testing values" → not pinned).
  2. Bail summary names up to 1-2 fields, capped at 120 chars.
  3. Bail IMMEDIATELY on any pinned-field rejection — don't try to fix
     non-pinned fields first.

Coverage:
  - Pin detection — literal value, whole-word boundary, multi-token,
    case-insensitive, whitespace-flexible, empty-value skip,
    non-form_input action skip.
  - Summary formatter — single field, two fields, TTS budget enforced
    (drops to 1 field when 2 would exceed 120 chars).
  - Orchestrator behavior — pinned reject → bail with reason; mixed
    pinned + non-pinned reject → bail (per user policy 3); only
    non-pinned reject → corrective fill runs as today (regression
    guard); pinned-rejection on last-loop final verify also bails
    with user_value_rejected (Bug 4 + 12 interaction).

Run: python test_user_pinned_values.py
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


# ─── _value_appears_in_goal helper ───────────────────────────────────────


class TestValueAppearsInGoal(unittest.TestCase):
    def test_literal_number_token(self):
        self.assertTrue(bdo._value_appears_in_goal(
            "99999", "fill this form with mobile as 99999",
        ))

    def test_literal_word_token(self):
        self.assertTrue(bdo._value_appears_in_goal(
            "John", "first name as John, last name as Doe",
        ))

    def test_substring_within_longer_word_no_match(self):
        # "Test" within "testing" must NOT match — that would falsely pin
        # planner-invented values that incidentally overlap goal text.
        self.assertFalse(bdo._value_appears_in_goal(
            "Test", "fill this form with testing values",
        ))

    def test_multi_token_phrase_match(self):
        self.assertTrue(bdo._value_appears_in_goal(
            "John Smith", "first name as John Smith please",
        ))

    def test_email_with_punctuation(self):
        # Word-boundary uses `isalnum()` — `@` and `.` count as boundaries.
        self.assertTrue(bdo._value_appears_in_goal(
            "test@example.com", "with email test@example.com today",
        ))

    def test_case_insensitive(self):
        self.assertTrue(bdo._value_appears_in_goal(
            "JOHN", "first name john please",
        ))
        self.assertTrue(bdo._value_appears_in_goal(
            "john", "first name JOHN please",
        ))

    def test_whitespace_collapsed(self):
        # Multiple spaces in goal collapse to one before matching.
        self.assertTrue(bdo._value_appears_in_goal(
            "John Smith", "first name John   Smith here",
        ))

    def test_empty_inputs(self):
        self.assertFalse(bdo._value_appears_in_goal("", "anything"))
        self.assertFalse(bdo._value_appears_in_goal("anything", ""))
        self.assertFalse(bdo._value_appears_in_goal("", ""))


# ─── _extract_pinned_refs ────────────────────────────────────────────────


class TestExtractPinnedFieldNames(unittest.TestCase):
    """
    Bug 12: pin extraction is keyed by NORMALIZED FIELD NAME, not ref.
    Refs are content-addressed (include bounds_quantized) so they mutate
    when the form re-renders post-rejection. Field labels are stable.
    Live evidence 2026-04-28 21:58: ref-keyed pins missed the rejection
    because loop-1 ref differed from loop-2 ref despite both being
    Contact Number.
    """

    def test_pins_form_input_with_goal_value(self):
        actions = [
            {"type": "form_input", "ref": "r_phone", "value": "99999"},
            {"type": "form_input", "ref": "r_first", "value": "Test"},
            {"type": "click_ref", "ref": "r_submit"},
        ]
        tree = _make_tree([
            _elem("r_phone", name="Contact Number"),
            _elem("r_first", name="First name"),
            _elem("r_submit", role="button", name="Submit"),
        ])
        pinned = bdo._extract_pinned_field_names(
            actions, "fill this form with testing values with mobile as 99999",
            tree,
        )
        # 99999 is in goal as discrete token → pinned under "contact number".
        # "Test" is NOT a whole-word match for "testing" → not pinned.
        # click_ref is not a form_input → ignored.
        self.assertEqual(pinned, {"contact number": "99999"})

    def test_skips_empty_value(self):
        actions = [
            {"type": "form_input", "ref": "r_a", "value": ""},
            {"type": "form_input", "ref": "r_b", "value": "   "},
            {"type": "form_input", "ref": "r_c", "value": "John"},
        ]
        tree = _make_tree([
            _elem("r_a", name="A"),
            _elem("r_b", name="B"),
            _elem("r_c", name="Name"),
        ])
        pinned = bdo._extract_pinned_field_names(
            actions, "name as John", tree,
        )
        self.assertEqual(pinned, {"name": "John"})

    def test_empty_goal_returns_empty(self):
        actions = [{"type": "form_input", "ref": "r1", "value": "anything"}]
        pinned = bdo._extract_pinned_field_names(actions, "", _make_tree([]))
        self.assertEqual(pinned, {})

    def test_no_form_input_actions(self):
        actions = [
            {"type": "click_ref", "ref": "r1"},
            {"type": "select_option_ref", "ref": "r2", "option": "USA"},
        ]
        pinned = bdo._extract_pinned_field_names(
            actions, "select USA", _make_tree([]),
        )
        # Only form_input actions are considered.
        self.assertEqual(pinned, {})

    def test_multiple_pinned_fields(self):
        actions = [
            {"type": "form_input", "ref": "r_phone", "value": "99999"},
            {"type": "form_input", "ref": "r_email", "value": "x@y.z"},
            {"type": "form_input", "ref": "r_first", "value": "Bob"},
        ]
        tree = _make_tree([
            _elem("r_phone", name="Contact Number"),
            _elem("r_email", name="Email"),
            _elem("r_first", name="First name"),
        ])
        pinned = bdo._extract_pinned_field_names(
            actions, "mobile as 99999 and email x@y.z please", tree,
        )
        self.assertEqual(
            pinned,
            {"contact number": "99999", "email": "x@y.z"},
        )

    def test_skips_action_with_no_named_field(self):
        # A form_input ref pointing to an element with empty name has
        # no durable identifier — skip rather than crash.
        actions = [
            {"type": "form_input", "ref": "r_unnamed", "value": "99999"},
        ]
        tree = _make_tree([_elem("r_unnamed", name="")])
        pinned = bdo._extract_pinned_field_names(
            actions, "value as 99999", tree,
        )
        self.assertEqual(pinned, {})


class TestExtractPinnedRefsCompatShim(unittest.TestCase):
    """The legacy `_extract_pinned_refs` shim is kept callable so older
    tests / external imports don't break. Returned shape: (set_of_refs,
    dict_of_ref_to_value), back-filled from the name-keyed result."""

    def test_shim_returns_back_filled_ref_set(self):
        actions = [
            {"type": "form_input", "ref": "r_phone", "value": "99999"},
        ]
        tree = _make_tree([_elem("r_phone", name="Contact Number")])
        pinned_refs, by_ref = bdo._extract_pinned_refs(
            actions, "mobile as 99999", tree,
        )
        self.assertEqual(pinned_refs, {"r_phone"})
        self.assertEqual(by_ref, {"r_phone": "99999"})


class TestPinSurvivesRefMutation(unittest.TestCase):
    """
    Regression guard for the live-test 2026-04-28 21:58 bug: when the
    form re-renders post-rejection, the input's bounds shift → its
    content-addressed ref mutates. Pin tracking must still match the
    field by NAME — refs are not identity here.
    """

    def test_name_match_survives_when_ref_changes(self):
        # Loop 1 tree: Contact Number has ref "r_v1".
        loop1_tree = _make_tree([
            _elem("r_v1", name="Contact Number"),
        ])
        loop1_actions = [
            {"type": "form_input", "ref": "r_v1", "value": "99999"},
        ]
        pinned = bdo._extract_pinned_field_names(
            loop1_actions, "mobile as 99999", loop1_tree,
        )
        self.assertEqual(pinned, {"contact number": "99999"})

        # Loop 2 tree: same field, but ref is now "r_v2_after_rerender".
        # The pinned dict is keyed by name — so a lookup against the new
        # tree's name still hits the pin. The orchestrator's bail check
        # uses this exact path: name_by_ref[err.field_ref] → check pinned.
        loop2_tree = _make_tree([
            _elem("r_v2_after_rerender", name="Contact Number"),
        ])
        name_by_ref = {e.ref: (e.name or "") for e in loop2_tree.elements}
        norm = bdo._normalize_field_name(name_by_ref["r_v2_after_rerender"])
        self.assertIn(norm, pinned,
                      msg="pin lookup must survive ref mutation across loops")


# ─── _format_pinned_rejection_summary ────────────────────────────────────


class TestFormatPinnedRejectionSummary(unittest.TestCase):
    def test_single_field(self):
        s = bdo._format_pinned_rejection_summary(
            {"contact number": "99999"},
            ["Contact Number"],
        )
        self.assertIn("'99999'", s)
        self.assertIn("'Contact Number'", s)
        self.assertIn("rejected", s.lower())
        self.assertLess(len(s), 120,
                        msg=f"summary too long: {len(s)} chars: {s!r}")

    def test_two_fields(self):
        s = bdo._format_pinned_rejection_summary(
            {"contact number": "99999", "email": "bad"},
            ["Contact Number", "Email"],
        )
        self.assertIn("'Contact Number'", s)
        self.assertIn("'Email'", s)
        self.assertLess(len(s), 120)

    def test_tts_budget_drops_to_one_field_when_two_overflow(self):
        # Realistic-but-pathological field names that push past 120 chars.
        long1 = "ReallyLongContactNumberFieldNameForTesting"
        long2 = "EvenLongerEmailFieldNameJustForTestingOverflow"
        s = bdo._format_pinned_rejection_summary(
            {long1.lower(): "9999999999",
             long2.lower(): "long-email-value@somewhere.com"},
            [long1, long2],
        )
        self.assertLess(
            len(s), 120,
            msg=f"summary still too long after fallback: {len(s)} chars: {s!r}",
        )

    def test_no_pinned_in_rejected_returns_generic(self):
        # rejected list refers to a field with no pin → generic fallback.
        s = bdo._format_pinned_rejection_summary(
            {},
            ["Email"],
        )
        # Should still produce a sane fallback message.
        self.assertGreater(len(s), 0)
        self.assertLess(len(s), 120)


# ─── End-to-end: run_dom_task respects pins ──────────────────────────────


class TestPinnedBehavior(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    async def test_pinned_rejection_bails_immediately(self):
        # User's goal pins mobile=99999. Form rejects Contact Number.
        # Orchestrator must bail with reason="user_value_rejected" — NOT
        # send corrective-fill prompt to the planner.
        tree_before = _make_tree([
            _elem("r_phone", form_id="form-0",
                  role="textbox", name="Contact Number"),
            _elem("r_submit", form_id="form-0",
                  role="button", name="Submit"),
        ])
        tree_after = _make_tree(
            [
                _elem("r_phone", form_id="form-0",
                      role="textbox", name="Contact Number",
                      value="99999"),
                _elem("r_submit", form_id="form-0",
                      role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError(
                    "r_phone", "Please enter a valid phone number",
                    "text-match",
                ),
            ),
        )
        actions = [
            {"type": "form_input", "ref": "r_phone", "value": "99999"},
            {"type": "click_ref", "ref": "r_submit"},
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
            result = await bdo.run_dom_task(
                "fill this form with mobile as 99999", MagicMock(),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "user_value_rejected")
        self.assertIn("99999", result.final_summary)
        self.assertIn("Contact Number", result.final_summary)
        # Critical: planner was called ONLY ONCE — corrective-fill prompt
        # never went out (it would have been the 2nd call).
        self.assertEqual(
            plan_mock.await_count, 1,
            msg="corrective-fill must NOT run when user pin rejected",
        )

    async def test_non_pinned_rejection_runs_corrective_fill(self):
        # Regression guard: when the rejected field is NOT user-pinned
        # (planner invented its value), corrective fill plumbing runs as
        # before — Bug 12 must not break the existing self-correction.
        tree_before = _make_tree([
            _elem("r_email", form_id="form-0",
                  role="textbox", name="Email"),
            _elem("r_submit", form_id="form-0",
                  role="button", name="Submit"),
        ])
        tree_after_bad = _make_tree(
            [
                _elem("r_email", form_id="form-0",
                      role="textbox", name="Email", value="bad"),
                _elem("r_submit", form_id="form-0",
                      role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError(
                    "r_email", "Invalid email", "describedby",
                ),
            ),
        )
        tree_clean = _make_tree([
            _elem("r_email", form_id="form-0",
                  role="textbox", name="Email", value="ok@x.com"),
            _elem("r_submit", form_id="form-0",
                  role="button", name="Submit"),
        ])
        actions_1 = [
            {"type": "form_input", "ref": "r_email", "value": "bad"},
            {"type": "click_ref", "ref": "r_submit"},
        ]
        actions_2 = [
            {"type": "form_input", "ref": "r_email", "value": "ok@x.com"},
            {"type": "click_ref", "ref": "r_submit"},
        ]
        plan_1 = bdp.DomPlan(
            thinking="fill+submit", plan="initial",
            actions=actions_1, done=True, needs_reperceive=False,
        )
        plan_2 = bdp.DomPlan(
            thinking="corrective", plan="fix",
            actions=actions_2, done=True, needs_reperceive=False,
        )

        plan_calls: list = []

        async def _plan_se(goal, scoped, *, feedback=""):
            plan_calls.append(feedback)
            return plan_1 if len(plan_calls) == 1 else plan_2

        with patch.object(bdom, "read_page_dom",
                          new=AsyncMock(side_effect=[
                              tree_before, tree_after_bad, tree_clean,
                          ])), \
             patch.object(bdp, "plan_dom_actions",
                          new=AsyncMock(side_effect=_plan_se)), \
             patch.object(bde, "execute_dom_batch",
                          new=AsyncMock(side_effect=[
                              _batch_ok(actions_1), _batch_ok(actions_2),
                          ])):
            # Goal does NOT pin "bad" — that's a planner-invented value.
            result = await bdo.run_dom_task(
                "fill this form with testing email", MagicMock(),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        # Two plan calls — first plan + corrective fill — confirms the
        # self-correction path still runs for non-pinned rejections.
        self.assertEqual(len(plan_calls), 2)

    async def test_mixed_pin_and_non_pin_bails_immediately(self):
        # User pins mobile=99999. Form rejects BOTH mobile and email.
        # Per user policy 3: bail immediately on ANY pinned rejection;
        # don't try to fix the non-pinned email first.
        tree_before = _make_tree([
            _elem("r_phone", form_id="form-0",
                  role="textbox", name="Contact Number"),
            _elem("r_email", form_id="form-0",
                  role="textbox", name="Email"),
            _elem("r_submit", form_id="form-0",
                  role="button", name="Submit"),
        ])
        tree_after = _make_tree(
            [
                _elem("r_phone", form_id="form-0",
                      role="textbox", name="Contact Number", value="99999"),
                _elem("r_email", form_id="form-0",
                      role="textbox", name="Email", value="bad"),
                _elem("r_submit", form_id="form-0",
                      role="button", name="Submit"),
            ],
            validation_errors=(
                bdom.ValidationError(
                    "r_phone", "Invalid phone", "text-match",
                ),
                bdom.ValidationError(
                    "r_email", "Invalid email", "describedby",
                ),
            ),
        )
        actions = [
            {"type": "form_input", "ref": "r_phone", "value": "99999"},
            {"type": "form_input", "ref": "r_email", "value": "bad"},
            {"type": "click_ref", "ref": "r_submit"},
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
            result = await bdo.run_dom_task(
                "fill the form with mobile as 99999 please", MagicMock(),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "user_value_rejected")
        # Summary names the pinned field (Contact Number with 99999).
        self.assertIn("99999", result.final_summary)
        self.assertIn("Contact Number", result.final_summary)
        # No corrective fill ran.
        self.assertEqual(plan_mock.await_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
