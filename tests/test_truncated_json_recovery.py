"""
test_truncated_json_recovery.py — robust mid-stream JSON recovery for the
vision planner LLM (2026-04-26).

Bug context: vision planner LLM hit max_output_tokens mid-string, leaving
a JSON like:

    ```json
    {
      "thinking": "I have already clicked... I will

The prior `_recover_truncated_json` only handled brace-count mismatches —
appending `]}` blindly. With an unclosed string AND an unclosed code fence,
that produced unparseable output and the agent aborted with
"Sorry, I couldn't understand the action plan from the LLM."

Fix: walk char-by-char tracking string-quote state + bracket stack, then
close everything in the right order. _parse_plan is also extended to strip
leading-only (unclosed) code fences.

Covers:
  - _recover_truncated_json:
    * valid JSON unchanged (idempotent)
    * unclosed string at end → close string
    * unclosed array → close array
    * unclosed nested object inside array → close in correct order
    * trailing comma + truncation → strip comma
    * dangling backslash before truncation → drop and close
    * empty/None inputs → unchanged
  - _parse_plan:
    * pure JSON object
    * closed code fence
    * unclosed code fence with mid-string truncation (the live-test case)
    * unclosed code fence with mid-array truncation
    * recovered plan with no `actions` key still parses (caller handles)
    * gibberish returns None
    * non-string input returns None

Run: python test_truncated_json_recovery.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca


# ─── _recover_truncated_json ──────────────────────────────────────────────


class TestRecoverTruncatedJson(unittest.TestCase):
    def test_valid_json_unchanged(self):
        valid = '{"a": 1, "b": [1, 2, 3]}'
        self.assertEqual(ca._recover_truncated_json(valid), valid)

    def test_unclosed_string_at_end(self):
        truncated = '{"thinking": "I will'
        out = ca._recover_truncated_json(truncated)
        self.assertEqual(out, '{"thinking": "I will"}')
        # And it parses cleanly.
        self.assertEqual(json.loads(out), {"thinking": "I will"})

    def test_unclosed_array(self):
        truncated = '{"actions": [{"type": "click"}'
        out = ca._recover_truncated_json(truncated)
        parsed = json.loads(out)
        self.assertEqual(parsed["actions"], [{"type": "click"}])

    def test_truncation_inside_nested_object_in_array(self):
        truncated = '{"actions": [{"type": "click", "text": "OK"}, {"type": "type", "text": "Hello'
        out = ca._recover_truncated_json(truncated)
        parsed = json.loads(out)
        self.assertEqual(len(parsed["actions"]), 2)
        self.assertEqual(parsed["actions"][1]["text"], "Hello")

    def test_trailing_comma_then_truncation(self):
        truncated = '{"actions": [{"type": "click"},'
        out = ca._recover_truncated_json(truncated)
        parsed = json.loads(out)
        self.assertEqual(parsed["actions"], [{"type": "click"}])

    def test_dangling_backslash_dropped(self):
        # Truncated mid-escape sequence: `... "thinking": "He said \`
        truncated = '{"thinking": "He said \\'
        out = ca._recover_truncated_json(truncated)
        parsed = json.loads(out)
        # The dangling backslash is dropped, then string closed
        self.assertEqual(parsed["thinking"], "He said ")

    def test_empty_string(self):
        self.assertEqual(ca._recover_truncated_json(""), "")

    def test_none_input(self):
        self.assertIsNone(ca._recover_truncated_json(None))

    def test_string_containing_braces_not_double_closed(self):
        # Braces inside string values must NOT bump the open-stack counter.
        valid = '{"description": "function foo() { return 1; }"}'
        self.assertEqual(ca._recover_truncated_json(valid), valid)
        # Should parse to the original.
        self.assertEqual(
            json.loads(ca._recover_truncated_json(valid))["description"],
            "function foo() { return 1; }",
        )

    def test_escaped_quote_inside_string_doesnt_close(self):
        # A `\"` inside a string must not be treated as a closing quote.
        truncated = '{"a": "he said \\"hi\\" then'
        out = ca._recover_truncated_json(truncated)
        parsed = json.loads(out)
        self.assertEqual(parsed["a"], 'he said "hi" then')

    def test_deep_nesting_closes_in_correct_order(self):
        truncated = '{"a": {"b": {"c": ["one", "two'
        out = ca._recover_truncated_json(truncated)
        parsed = json.loads(out)
        self.assertEqual(parsed, {"a": {"b": {"c": ["one", "two"]}}})


# ─── _parse_plan with code-fence handling ─────────────────────────────────


class TestParsePlanRobustness(unittest.TestCase):
    def test_pure_json(self):
        raw = '{"thinking": "x", "actions": []}'
        self.assertEqual(ca._parse_plan(raw),
                         {"thinking": "x", "actions": []})

    def test_closed_code_fence(self):
        raw = '```json\n{"thinking": "x", "actions": []}\n```'
        self.assertEqual(ca._parse_plan(raw),
                         {"thinking": "x", "actions": []})

    def test_unclosed_code_fence_with_truncation(self):
        # Exact shape from the 2026-04-26 live-test failure.
        raw = (
            '```json\n'
            '{\n'
            '  "thinking": "I have already clicked on the \'Industry\' '
            'dropdown. Now I need to select \'IT\' from the list of '
            'industries. I can see \'Industry\' highlighted, and '
            '\'Construction\' below it. I will'
        )
        result = ca._parse_plan(raw)
        # The recovered plan should at least parse and have a thinking field.
        self.assertIsNotNone(result, "parser must recover from truncated fence")
        self.assertIn("thinking", result)
        # No actions key in this truncation point — caller's loop just
        # asks for a fresh plan next iteration. That's the right behaviour.

    def test_unclosed_fence_with_truncated_array(self):
        raw = (
            '```json\n'
            '{"thinking": "ok", "actions": [{"type": "vision_guided_click", '
            '"text": "Subm'
        )
        result = ca._parse_plan(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["actions"], [
            {"type": "vision_guided_click", "text": "Subm"}
        ])

    def test_gibberish_returns_none(self):
        self.assertIsNone(ca._parse_plan("this is not json at all"))

    def test_non_string_returns_none(self):
        self.assertIsNone(ca._parse_plan(None))
        self.assertIsNone(ca._parse_plan(42))

    def test_empty_returns_none(self):
        self.assertIsNone(ca._parse_plan(""))
        self.assertIsNone(ca._parse_plan("   "))

    def test_existing_pe1_recovery_path_still_works(self):
        # Regression guard for the original recovery case (unclosed brace,
        # no fence, no string truncation).
        raw = '{"thinking": "x", "actions": [{"type": "click"}'
        result = ca._parse_plan(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["actions"], [{"type": "click"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
