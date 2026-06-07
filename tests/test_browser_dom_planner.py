"""
test_browser_dom_planner.py — Phase 1C-a: DOM-aware planner.

Stub-based unit tests for the planner module. The LLM is mocked; we
verify the module's:
  - JSON parsing tolerance (bare, fenced, prose-wrapped, truncated)
  - Per-action validation rules
  - Ref-existence enforcement against the tree
  - Visible/enabled gating
  - Type-specific schema (form_input, click_ref, select_option_ref,
    press_ref, wait_ms, reperceive)
  - LLM unavailable / crash → empty plan with rejection notes
  - Feedback string plumbs into the user prompt

Run: python test_browser_dom_planner.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom as bdom
import assistant.automation.browser.dom_planner as bdp


def _run(coro):
    return asyncio.run(coro)


def _install_llm_stub(text_response=None):
    """Patch ask_for_plan in llm.contracts to return a controlled response.

    Returns a mock whose .call_count / .call_args can be inspected.
    """
    if isinstance(text_response, Exception):
        mock = AsyncMock(side_effect=text_response)
    elif isinstance(text_response, list):
        mock = AsyncMock(side_effect=text_response)
    else:
        mock = AsyncMock(return_value=text_response or "")
    patcher = unittest.mock.patch(
        "assistant.llm.contracts.ask_for_plan", mock,
    )
    patcher.start()
    # Return a namespace with get_llm_response pointing to the mock
    # so existing tests that inspect call_args still work.
    ns = types.SimpleNamespace(get_llm_response=mock, _patcher=patcher)
    return ns


def _make_tree(elements: list[bdom.ElementInfo], ref_map=None) -> bdom.PageDomTree:
    if ref_map is None:
        ref_map = {e.ref: MagicMock(name=f"locator-{e.ref}") for e in elements}
    return bdom.PageDomTree(
        elements=elements, ref_to_locator=ref_map, truncated=0,
        read_at=time.monotonic(), viewport=(1280, 800),
    )


def _elem(ref, **kwargs) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input",
    )
    base.update(kwargs)
    return bdom.ElementInfo(ref=ref, **base)


# ─── _parse_planner_response ─────────────────────────────────────────────


class TestParsePlannerResponse(unittest.TestCase):
    def test_bare_json(self):
        raw = '{"thinking":"x","plan":"p","actions":[],"done":false}'
        out = bdp._parse_planner_response(raw)
        self.assertIsNotNone(out)
        self.assertEqual(out["thinking"], "x")

    def test_closed_code_fence(self):
        raw = '```json\n{"thinking":"x","actions":[]}\n```'
        out = bdp._parse_planner_response(raw)
        self.assertIsNotNone(out)

    def test_unclosed_fence_with_truncation(self):
        # The 2026-04-26 vision-planner failure shape applied to DOM mode.
        raw = (
            '```json\n'
            '{"thinking":"fill 3 fields then submit",'
            '"plan":"fill","actions":[{"type":"form_input","ref":"abc",'
            '"value":"Te'
        )
        out = bdp._parse_planner_response(raw)
        # Recovery may close the string + array + object; structure is
        # what matters.
        self.assertIsNotNone(out, "must recover a parseable structure")
        self.assertIn("actions", out)

    def test_gibberish_returns_none(self):
        self.assertIsNone(bdp._parse_planner_response("not json at all"))

    def test_empty_returns_none(self):
        self.assertIsNone(bdp._parse_planner_response(""))
        self.assertIsNone(bdp._parse_planner_response("   "))

    def test_non_string_returns_none(self):
        self.assertIsNone(bdp._parse_planner_response(None))
        self.assertIsNone(bdp._parse_planner_response(42))


# ─── _validate_action: per-type schema ───────────────────────────────────


class TestValidateAction(unittest.TestCase):
    def setUp(self):
        self.refs = {"good", "select_with_opts", "invisible", "disabled"}
        self.lookup = {
            "good": {"ref": "good", "role": "textbox", "name": "X"},
            "select_with_opts": {
                "ref": "select_with_opts", "role": "combobox", "name": "Y",
                "options": ["A", "B", "C"],
            },
            "invisible": {"ref": "invisible", "role": "textbox", "name": "I", "visible": False},
            "disabled": {"ref": "disabled", "role": "textbox", "name": "D", "enabled": False},
        }

    def test_form_input_valid(self):
        ok, _ = bdp._validate_action(
            {"type": "form_input", "ref": "good", "value": "John"},
            self.refs, self.lookup,
        )
        self.assertTrue(ok)

    def test_form_input_unknown_ref(self):
        ok, reason = bdp._validate_action(
            {"type": "form_input", "ref": "nonexistent", "value": "x"},
            self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("not in tree", reason)

    def test_form_input_invisible_ref_rejected(self):
        ok, reason = bdp._validate_action(
            {"type": "form_input", "ref": "invisible", "value": "x"},
            self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("not visible", reason)

    def test_form_input_disabled_ref_rejected(self):
        ok, reason = bdp._validate_action(
            {"type": "form_input", "ref": "disabled", "value": "x"},
            self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("not enabled", reason)

    def test_form_input_missing_value_rejected(self):
        ok, reason = bdp._validate_action(
            {"type": "form_input", "ref": "good"}, self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("missing keys", reason)

    def test_select_option_valid_when_in_options(self):
        ok, _ = bdp._validate_action(
            {"type": "select_option_ref", "ref": "select_with_opts", "option": "B"},
            self.refs, self.lookup,
        )
        self.assertTrue(ok)

    def test_select_option_rejects_unknown_option(self):
        ok, reason = bdp._validate_action(
            {"type": "select_option_ref", "ref": "select_with_opts", "option": "Z"},
            self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("not in element's options", reason)

    def test_select_option_no_options_array_allows_any(self):
        # Custom comboboxes have empty options at perception time; the
        # planner shouldn't be blocked from select_option_ref against them
        # if they DO ship with options later. (Native selects without
        # options is a corner case that should have been rejected upstream.)
        refs = {"loose"}
        lookup = {"loose": {"ref": "loose", "role": "combobox"}}  # no options
        ok, _ = bdp._validate_action(
            {"type": "select_option_ref", "ref": "loose", "option": "Anything"},
            refs, lookup,
        )
        self.assertTrue(ok)

    def test_click_ref_valid(self):
        ok, _ = bdp._validate_action(
            {"type": "click_ref", "ref": "good"}, self.refs, self.lookup,
        )
        self.assertTrue(ok)

    def test_press_ref_requires_key(self):
        ok, reason = bdp._validate_action(
            {"type": "press_ref", "ref": "good"}, self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("missing keys", reason)

    def test_wait_ms_valid(self):
        ok, _ = bdp._validate_action(
            {"type": "wait_ms", "ms": 500}, self.refs, self.lookup,
        )
        self.assertTrue(ok)

    def test_wait_ms_negative_rejected(self):
        ok, reason = bdp._validate_action(
            {"type": "wait_ms", "ms": -1}, self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("0..30000", reason)

    def test_wait_ms_too_long_rejected(self):
        ok, reason = bdp._validate_action(
            {"type": "wait_ms", "ms": 60000}, self.refs, self.lookup,
        )
        self.assertFalse(ok)

    def test_reperceive_no_ref_required(self):
        ok, _ = bdp._validate_action(
            {"type": "reperceive"}, self.refs, self.lookup,
        )
        self.assertTrue(ok)

    def test_unknown_type_rejected(self):
        ok, reason = bdp._validate_action(
            {"type": "screenshot"}, self.refs, self.lookup,
        )
        self.assertFalse(ok)
        self.assertIn("unknown type", reason)

    def test_non_dict_action_rejected(self):
        ok, reason = bdp._validate_action(
            "not a dict", self.refs, self.lookup,
        )
        self.assertFalse(ok)


# ─── plan_dom_actions: full pipeline ─────────────────────────────────────


class TestPlanDomActions(unittest.TestCase):
    _llm_ns = None

    def tearDown(self):
        if self._llm_ns and hasattr(self._llm_ns, "_patcher"):
            self._llm_ns._patcher.stop()
        self._llm_ns = None

    def test_happy_path_form_fill(self):
        tree = _make_tree([
            _elem("ref0001", name="First name"),
            _elem("ref0002", name="Last name"),
            _elem("ref0003", role="combobox", name="Country",
                  options=("USA", "Canada", "UK"), tag="select"),
            _elem("ref0004", role="button", name="Submit", tag="button"),
        ])
        self._llm_ns = _install_llm_stub(text_response=json.dumps({
            "thinking": "fill name+country, submit",
            "plan": "fill all",
            "actions": [
                {"type": "form_input", "ref": "ref0001", "value": "Jane"},
                {"type": "form_input", "ref": "ref0002", "value": "Doe"},
                {"type": "select_option_ref", "ref": "ref0003", "option": "Canada"},
                {"type": "click_ref", "ref": "ref0004"},
            ],
            "done": True,
            "needs_reperceive": False,
        }))
        plan = _run(bdp.plan_dom_actions("fill the signup form", tree))
        self.assertEqual(len(plan.actions), 4)
        self.assertTrue(plan.done)
        self.assertFalse(plan.needs_reperceive)
        self.assertEqual(plan.rejection_notes, [])

    def test_invalid_ref_filtered_out(self):
        tree = _make_tree([_elem("ref0001", name="A")])
        self._llm_ns = _install_llm_stub(text_response=json.dumps({
            "thinking": "x", "plan": "p",
            "actions": [
                {"type": "form_input", "ref": "ref0001", "value": "John"},
                {"type": "click_ref", "ref": "fabricated"},  # not in tree
            ],
            "done": True, "needs_reperceive": False,
        }))
        plan = _run(bdp.plan_dom_actions("g", tree))
        self.assertEqual(len(plan.actions), 1)  # only the valid one kept
        self.assertEqual(plan.actions[0]["ref"], "ref0001")
        self.assertEqual(len(plan.rejection_notes), 1)
        self.assertIn("fabricated", plan.rejection_notes[0])

    def test_invisible_action_filtered(self):
        tree = _make_tree([
            _elem("ref0001", name="A"),
            _elem("ref0002", name="B", visible=False),
        ])
        self._llm_ns = _install_llm_stub(text_response=json.dumps({
            "thinking": "x", "plan": "p",
            "actions": [
                {"type": "form_input", "ref": "ref0001", "value": "ok"},
                {"type": "form_input", "ref": "ref0002", "value": "no"},
            ],
            "done": False, "needs_reperceive": False,
        }))
        plan = _run(bdp.plan_dom_actions("g", tree))
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0]["ref"], "ref0001")
        self.assertTrue(any("not visible" in n for n in plan.rejection_notes))

    def test_select_unknown_option_filtered(self):
        tree = _make_tree([
            _elem("ref0001", role="combobox", name="X",
                  options=("A", "B"), tag="select"),
        ])
        self._llm_ns = _install_llm_stub(text_response=json.dumps({
            "thinking": "x", "plan": "p",
            "actions": [
                {"type": "select_option_ref", "ref": "ref0001", "option": "C"},
            ],
            "done": False, "needs_reperceive": False,
        }))
        plan = _run(bdp.plan_dom_actions("g", tree))
        self.assertEqual(len(plan.actions), 0)
        self.assertEqual(len(plan.rejection_notes), 1)

    def test_llm_unavailable_returns_empty_plan(self):
        tree = _make_tree([_elem("ref0001", name="A")])
        self._llm_ns = _install_llm_stub(text_response="__LLM_UNAVAILABLE__")
        plan = _run(bdp.plan_dom_actions("g", tree))
        self.assertEqual(plan.actions, [])
        self.assertFalse(plan.done)
        self.assertIn("llm_unavailable", plan.rejection_notes)

    def test_llm_crash_returns_empty_plan(self):
        tree = _make_tree([_elem("ref0001", name="A")])
        self._llm_ns = _install_llm_stub(text_response=RuntimeError("network down"))
        plan = _run(bdp.plan_dom_actions("g", tree))
        self.assertEqual(plan.actions, [])
        self.assertTrue(any("llm_crash" in n for n in plan.rejection_notes))

    def test_garbage_response_returns_empty_plan(self):
        tree = _make_tree([_elem("ref0001", name="A")])
        self._llm_ns = _install_llm_stub(text_response="this is not json")
        plan = _run(bdp.plan_dom_actions("g", tree))
        self.assertEqual(plan.actions, [])
        self.assertIn("parse_failed", plan.rejection_notes)

    def test_feedback_passed_to_llm(self):
        tree = _make_tree([_elem("ref0001", name="A")])
        self._llm_ns = _install_llm_stub(text_response=json.dumps({
            "thinking": "x", "plan": "p", "actions": [],
            "done": False, "needs_reperceive": False,
        }))
        _run(bdp.plan_dom_actions("g", tree, feedback="email empty after submit"))
        mock = self._llm_ns.get_llm_response
        self.assertEqual(mock.call_count, 1)
        prompt_arg = mock.call_args[0][0]
        self.assertIn("FEEDBACK FROM PREVIOUS ITERATION", prompt_arg)
        self.assertIn("email empty after submit", prompt_arg)

    def test_needs_reperceive_propagated(self):
        tree = _make_tree([
            _elem("ref0001", role="combobox", name="Industry"),  # no options
        ])
        self._llm_ns = _install_llm_stub(text_response=json.dumps({
            "thinking": "open combobox first",
            "plan": "open then re-read",
            "actions": [
                {"type": "click_ref", "ref": "ref0001"},
                {"type": "reperceive"},
            ],
            "done": False, "needs_reperceive": True,
        }))
        plan = _run(bdp.plan_dom_actions("pick IT", tree))
        self.assertTrue(plan.needs_reperceive)
        self.assertEqual(len(plan.actions), 2)

    def test_empty_actions_array_valid(self):
        # Planner can return an empty action list when nothing to do.
        tree = _make_tree([_elem("ref0001", name="A")])
        self._llm_ns = _install_llm_stub(text_response=json.dumps({
            "thinking": "nothing to do",
            "plan": "noop",
            "actions": [],
            "done": True,
            "needs_reperceive": False,
        }))
        plan = _run(bdp.plan_dom_actions("g", tree))
        self.assertEqual(plan.actions, [])
        self.assertTrue(plan.done)
        self.assertEqual(plan.rejection_notes, [])


# ─── System prompt invariants ────────────────────────────────────────────


class TestSystemPromptInvariants(unittest.TestCase):
    def test_prompt_documents_all_action_types(self):
        for atype in bdp._VALID_ACTION_TYPES:
            self.assertIn(atype, bdp.DOM_PLANNER_SYSTEM_PROMPT,
                          f"action type {atype!r} missing from system prompt")

    def test_prompt_warns_against_fabricated_refs(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn("never fabricate", prompt)

    def test_prompt_warns_against_invisible_action(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn("visible", prompt)
        self.assertIn("enabled", prompt)


# ─── Combobox prompt rules (Phase 2A) ────────────────────────────────────


class TestComboboxPromptRules(unittest.TestCase):
    """Phase 2A: planner prompt must teach the autocomplete combobox flow."""

    def test_rule_3a_autocomplete_combobox_uses_form_input(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT
        self.assertIn("autocomplete", prompt.lower())
        self.assertIn("form_input", prompt)

    def test_rule_3b_non_autocomplete_uses_click(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT
        self.assertIn("click_ref", prompt)

    def test_rule_3c_option_selection_uses_click_not_select(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn('role="option"', prompt)

    def test_rule_11_multi_select_one_value_per_loop(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn("multi", prompt)
        self.assertIn("one value", prompt)

    def test_native_select_rule_unchanged(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT
        self.assertIn("select_option_ref", prompt)
        self.assertIn("EXACT", prompt)

    def test_elements_section_documents_autocomplete_field(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT
        elements_section = prompt.split("ELEMENTS YOU RECEIVE")[1].split("OUTPUT")[0]
        self.assertIn("autocomplete", elements_section.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
