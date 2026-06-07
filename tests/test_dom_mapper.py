"""
test_dom_mapper.py — Option B: LLM goal-to-field mapper.

Tests:
  - FormMapping / FillInstruction dataclass construction
  - Prompt construction from ElementInfo list + goal
  - Response parsing (happy path, multi-value, missing fields)
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom_mapper as mapper


class TestDataclasses(unittest.TestCase):
    def test_fill_instruction_basic(self):
        fi = mapper.FillInstruction(ref="abc123", field_name="Email", value="test@example.com")
        self.assertEqual(fi.ref, "abc123")
        self.assertEqual(fi.field_name, "Email")
        self.assertEqual(fi.value, "test@example.com")

    def test_fill_instruction_multi_value(self):
        fi = mapper.FillInstruction(ref="abc123", field_name="Subjects", value=["Maths", "English"])
        self.assertIsInstance(fi.value, list)
        self.assertEqual(fi.value, ["Maths", "English"])

    def test_form_mapping_basic(self):
        fills = [mapper.FillInstruction(ref="r1", field_name="Name", value="Test")]
        fm = mapper.FormMapping(fills=fills, submit_ref="s1", thinking="fill the name")
        self.assertEqual(len(fm.fills), 1)
        self.assertEqual(fm.submit_ref, "s1")
        self.assertFalse(fm.skip_submit)

    def test_form_mapping_skip_submit(self):
        fm = mapper.FormMapping(fills=[], submit_ref="", thinking="no submit", skip_submit=True)
        self.assertTrue(fm.skip_submit)


import assistant.automation.browser.dom as bdom


def _elem(ref, **kw) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input", form_id="form-0", in_dialog=False,
    )
    base.update(kw)
    return bdom.ElementInfo(ref=ref, **base)


class TestPromptConstruction(unittest.TestCase):
    def test_prompt_contains_goal(self):
        elements = [_elem("r1", name="First Name")]
        prompt = mapper.build_mapper_prompt("Fill the form with name John", elements)
        self.assertIn("Fill the form with name John", prompt)

    def test_prompt_contains_field_names(self):
        elements = [
            _elem("r1", name="First Name"),
            _elem("r2", name="Email", type="email"),
        ]
        prompt = mapper.build_mapper_prompt("Fill the form", elements)
        self.assertIn("First Name", prompt)
        self.assertIn("Email", prompt)
        self.assertIn("r1", prompt)
        self.assertIn("r2", prompt)

    def test_prompt_marks_combobox(self):
        elements = [
            _elem("r1", role="combobox", name="Subjects", autocomplete="list", options=()),
        ]
        prompt = mapper.build_mapper_prompt("Fill subjects with Maths", elements)
        self.assertIn("combobox", prompt.lower())

    def test_prompt_marks_radio(self):
        elements = [
            _elem("r1", role="radio", name="Male", tag="input", type="radio"),
            _elem("r2", role="radio", name="Female", tag="input", type="radio"),
        ]
        prompt = mapper.build_mapper_prompt("Select male gender", elements)
        self.assertIn("radio", prompt.lower())

    def test_prompt_marks_submit_button(self):
        elements = [
            _elem("r1", name="Name"),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        prompt = mapper.build_mapper_prompt("Fill the form", elements)
        self.assertIn("Submit", prompt)
        self.assertIn("button", prompt.lower())

    def test_prompt_includes_current_values(self):
        elements = [_elem("r1", name="Name", value="Already Filled")]
        prompt = mapper.build_mapper_prompt("Fill the form", elements)
        self.assertIn("Already Filled", prompt)

    def test_prompt_includes_select_options(self):
        elements = [
            _elem("r1", role="combobox", name="State", tag="select",
                  options=("-- Select --", "NCR", "Uttar Pradesh")),
        ]
        prompt = mapper.build_mapper_prompt("Select NCR as state", elements)
        self.assertIn("NCR", prompt)
        self.assertIn("Uttar Pradesh", prompt)


class TestParseResponse(unittest.TestCase):
    def test_happy_path(self):
        raw = json.dumps({
            "thinking": "fill name and email",
            "fills": [
                {"ref": "r1", "value": "John"},
                {"ref": "r2", "value": "john@example.com"},
            ],
            "submit_ref": "s1",
            "skip_submit": False,
        })
        valid_refs = {"r1", "r2", "s1"}
        result = mapper.parse_mapper_response(raw, valid_refs)
        self.assertEqual(len(result.fills), 2)
        self.assertEqual(result.fills[0].value, "John")
        self.assertEqual(result.submit_ref, "s1")
        self.assertFalse(result.skip_submit)

    def test_multi_value(self):
        raw = json.dumps({
            "thinking": "fill subjects",
            "fills": [{"ref": "r1", "value": ["Maths", "English"]}],
            "submit_ref": "",
        })
        result = mapper.parse_mapper_response(raw, {"r1"})
        self.assertEqual(result.fills[0].value, ["Maths", "English"])

    def test_unknown_ref_dropped(self):
        raw = json.dumps({
            "thinking": "fill",
            "fills": [
                {"ref": "r1", "value": "A"},
                {"ref": "BOGUS", "value": "B"},
            ],
            "submit_ref": "",
        })
        result = mapper.parse_mapper_response(raw, {"r1"})
        self.assertEqual(len(result.fills), 1)
        self.assertEqual(result.fills[0].ref, "r1")

    def test_invalid_json_returns_empty(self):
        result = mapper.parse_mapper_response("not json", {"r1"})
        self.assertEqual(len(result.fills), 0)
        self.assertEqual(result.submit_ref, "")

    def test_submit_ref_validated(self):
        raw = json.dumps({
            "thinking": "fill",
            "fills": [{"ref": "r1", "value": "A"}],
            "submit_ref": "BOGUS",
        })
        result = mapper.parse_mapper_response(raw, {"r1"})
        self.assertEqual(result.submit_ref, "")

    def test_field_name_populated_from_ref_map(self):
        raw = json.dumps({
            "thinking": "fill",
            "fills": [{"ref": "r1", "value": "Test"}],
            "submit_ref": "",
        })
        ref_to_name = {"r1": "First Name"}
        result = mapper.parse_mapper_response(raw, {"r1"}, ref_to_name=ref_to_name)
        self.assertEqual(result.fills[0].field_name, "First Name")


import asyncio
from unittest.mock import AsyncMock, patch


def _run(coro):
    return asyncio.run(coro)


class TestMapGoalToFields(unittest.TestCase):
    @patch("assistant.automation.browser.dom_mapper.ask_for_plan", new_callable=AsyncMock)
    def test_happy_path(self, mock_ask):
        mock_ask.return_value = json.dumps({
            "thinking": "fill name and email, submit",
            "fills": [
                {"ref": "r1", "value": "Test"},
                {"ref": "r2", "value": "test@example.com"},
            ],
            "submit_ref": "s1",
        })
        elements = [
            _elem("r1", name="First Name"),
            _elem("r2", name="Email", type="email"),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        tree = bdom.PageDomTree(
            elements=elements,
            ref_to_locator={"r1": None, "r2": None, "s1": None},
        )
        result = _run(mapper.map_goal_to_fields("Fill the form with test data", tree))
        self.assertEqual(len(result.fills), 2)
        self.assertEqual(result.submit_ref, "s1")
        mock_ask.assert_called_once()
        call_kwargs = mock_ask.call_args
        self.assertTrue(call_kwargs.kwargs.get("json_mode"))

    @patch("assistant.automation.browser.dom_mapper.ask_for_plan", new_callable=AsyncMock)
    def test_llm_returns_garbage(self, mock_ask):
        mock_ask.return_value = "I don't understand"
        elements = [_elem("r1", name="Name")]
        tree = bdom.PageDomTree(
            elements=elements,
            ref_to_locator={"r1": None},
        )
        result = _run(mapper.map_goal_to_fields("Fill name", tree))
        self.assertEqual(len(result.fills), 0)

    @patch("assistant.automation.browser.dom_mapper.ask_for_plan", new_callable=AsyncMock)
    def test_validation_feedback_included(self, mock_ask):
        mock_ask.return_value = json.dumps({
            "thinking": "fix email",
            "fills": [{"ref": "r1", "value": "real@email.com"}],
            "submit_ref": "s1",
        })
        elements = [
            _elem("r1", name="Email", type="email", value="bad"),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        tree = bdom.PageDomTree(
            elements=elements,
            ref_to_locator={"r1": None, "s1": None},
        )
        feedback = 'VALIDATION ERRORS:\n- "Email" (ref=r1): Please enter a valid email'
        result = _run(mapper.map_goal_to_fields("Fill email", tree, feedback=feedback))
        self.assertEqual(len(result.fills), 1)
        # Verify feedback was included in the prompt
        prompt_arg = mock_ask.call_args.args[0]
        self.assertIn("VALIDATION ERRORS", prompt_arg)
