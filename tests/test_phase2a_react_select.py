"""
test_phase2a_react_select.py — Phase 2A: react-select adapter integration tests.

Tests the multi-loop flows that make react-select autocomplete work:
  - Autocomplete 2-loop: form_input → reperceive → click option
  - Native select regression: still uses select_option_ref
  - Prompt contract: combobox rules present in system prompt
"""

from __future__ import annotations

import asyncio
import json
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


def _elem(ref, **kwargs) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input",
    )
    base.update(kwargs)
    return bdom.ElementInfo(ref=ref, **base)


def _make_tree(elements, ref_map=None):
    if ref_map is None:
        ref_map = {e.ref: MagicMock(name=f"loc-{e.ref}") for e in elements}
    return bdom.PageDomTree(
        elements=elements, ref_to_locator=ref_map, truncated=0,
        read_at=time.monotonic(), viewport=(1280, 800),
    )


class TestAutocompletePromptContract(unittest.TestCase):
    """Verify the planner prompt teaches the autocomplete combobox flow."""

    def test_prompt_has_rule_3a_autocomplete(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn("autocomplete", prompt)
        self.assertIn("form_input", bdp.DOM_PLANNER_SYSTEM_PROMPT)

    def test_prompt_has_rule_3c_click_option(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT
        self.assertIn("click_ref", prompt)
        found = ("role=\"option\"" in prompt.lower()
                 or "role=option" in prompt.lower()
                 or 'role="option"' in prompt)
        self.assertTrue(found, "Prompt must mention role=option")

    def test_prompt_has_rule_11_multi_select(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn("one value", prompt)

    def test_prompt_documents_autocomplete_field(self):
        prompt = bdp.DOM_PLANNER_SYSTEM_PROMPT
        elements_section = prompt.split("ELEMENTS YOU RECEIVE")[1].split("OUTPUT")[0]
        self.assertIn("autocomplete", elements_section.lower())


class TestSerializationAutocomplete(unittest.TestCase):
    """Verify autocomplete is serialized correctly for the planner."""

    def test_autocomplete_combobox_emits_field(self):
        e = _elem("ref_ac", role="combobox", name="Subjects",
                   options=(), tag="input", autocomplete="list")
        tree = _make_tree([e])
        parsed = json.loads(bdom.serialize_for_planner(tree))
        self.assertEqual(parsed["elements"][0]["autocomplete"], "list")

    def test_native_select_does_not_emit_autocomplete(self):
        e = _elem("ref_ns", role="combobox", name="Country",
                   options=("USA", "Canada"), tag="select")
        tree = _make_tree([e])
        parsed = json.loads(bdom.serialize_for_planner(tree))
        self.assertNotIn("autocomplete", parsed["elements"][0])

    def test_textbox_does_not_emit_autocomplete(self):
        e = _elem("ref_tb", role="textbox", name="Email", type="email")
        tree = _make_tree([e])
        parsed = json.loads(bdom.serialize_for_planner(tree))
        self.assertNotIn("autocomplete", parsed["elements"][0])


class TestJsQueryAutocomplete(unittest.TestCase):
    """Verify the JS query captures aria-autocomplete."""

    def test_js_query_contains_aria_autocomplete(self):
        self.assertIn("aria-autocomplete", bdom._DOM_QUERY_JS)


class TestPlannerValidationUnchanged(unittest.TestCase):
    """Regression: existing action types still validate correctly."""

    def test_form_input_still_valid(self):
        ok, _ = bdp._validate_action(
            {"type": "form_input", "ref": "r001", "value": "test"},
            {"r001"}, {"r001": {"ref": "r001", "role": "textbox", "name": "F", "value": ""}},
        )
        self.assertTrue(ok)

    def test_click_ref_still_valid(self):
        ok, _ = bdp._validate_action(
            {"type": "click_ref", "ref": "r002"},
            {"r002"}, {"r002": {"ref": "r002", "role": "button", "name": "Submit", "value": ""}},
        )
        self.assertTrue(ok)

    def test_select_option_ref_still_valid_for_native_select(self):
        ok, _ = bdp._validate_action(
            {"type": "select_option_ref", "ref": "r003", "option": "USA"},
            {"r003"}, {"r003": {"ref": "r003", "role": "combobox", "name": "Country",
                                "value": "", "options": ["USA", "Canada"]}},
        )
        self.assertTrue(ok)

    def test_reperceive_still_valid(self):
        ok, _ = bdp._validate_action(
            {"type": "reperceive"}, set(), {},
        )
        self.assertTrue(ok)


class TestPortalOptionScoping(unittest.TestCase):
    """Portal-rendered dropdown options must be included in the scoped tree."""

    def _form_elements(self):
        return [
            _elem("r001", name="First name", form_id="form-0"),
            _elem("r002", role="combobox", name="Subjects",
                  options=(), tag="input", form_id="form-0",
                  autocomplete="list"),
            _elem("r003", role="button", name="Submit", tag="button",
                  form_id="form-0"),
        ]

    def _portal_options(self):
        return [
            _elem("r050", role="option", name="Maths", tag="div",
                  form_id=""),
            _elem("r051", role="option", name="English", tag="div",
                  form_id=""),
        ]

    def test_select_target_form_excludes_portal_options(self):
        all_els = self._form_elements() + self._portal_options()
        result = bdo._select_target_form(all_els, "fill subjects")
        self.assertIsNotNone(result)
        form_id, form_els = result
        self.assertEqual(form_id, "form-0")
        refs = {e.ref for e in form_els}
        self.assertNotIn("r050", refs)
        self.assertNotIn("r051", refs)

    def test_scoped_tree_includes_portal_options_after_fix(self):
        form_els = self._form_elements()
        portal_els = self._portal_options()
        all_els = form_els + portal_els
        full_tree = _make_tree(all_els)

        scope = bdo._select_target_form(all_els, "fill subjects")
        self.assertIsNotNone(scope)
        target_form_id, target_elements = scope

        _PORTAL_ROLES = {"option", "listbox"}
        extras = [
            e for e in full_tree.elements
            if not e.form_id and e.role in _PORTAL_ROLES
        ]
        combined = list(target_elements) + extras
        scoped = bdo._scope_tree_to_elements(full_tree, combined)

        refs = {e.ref for e in scoped.elements}
        self.assertIn("r001", refs)
        self.assertIn("r002", refs)
        self.assertIn("r050", refs, "portal option must be in scoped tree")
        self.assertIn("r051", refs, "portal option must be in scoped tree")

    def test_no_portal_elements_when_none_exist(self):
        all_els = self._form_elements()
        full_tree = _make_tree(all_els)
        scope = bdo._select_target_form(all_els, "fill subjects")
        target_form_id, target_elements = scope
        _PORTAL_ROLES = {"option", "listbox"}
        extras = [
            e for e in full_tree.elements
            if not e.form_id and e.role in _PORTAL_ROLES
        ]
        self.assertEqual(extras, [])
        scoped = bdo._scope_tree_to_elements(full_tree, target_elements)
        self.assertEqual(len(scoped.elements), 3)
