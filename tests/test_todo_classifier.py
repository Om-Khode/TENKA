"""
test_todo_classifier.py — planner-vision part 1: TODO classifier + state plumbing.

Covers:
  - _strip_matched_quotes: straight, smart, mixed, internal preserved, empty
  - _classify_todo: type/select/click/other branches with edges
    * type: quoted/unquoted value, in/into connector, trailing period
    * select: dropdown/menu/list/combobox suffixes, leading "the"
    * click: bare/quoted/role-suffixed, leading "the"
    * other: unrecognized verb, malformed, empty/None input
  - _make_todo_dict: schema completeness, classification populated
  - _TaskState integration: set_initial_todos calls classifier, add_todo too
  - State defaults: zero_progress_streak/loop_budget/confirm_fallback_count
  - reset() clears new fields back to defaults

Run: python test_todo_classifier.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca


# ─── _strip_matched_quotes ─────────────────────────────────────────────────


class TestStripMatchedQuotes(unittest.TestCase):
    def test_straight_single(self):
        self.assertEqual(ca._strip_matched_quotes("'hello'"), "hello")

    def test_straight_double(self):
        self.assertEqual(ca._strip_matched_quotes('"hello"'), "hello")

    def test_backtick(self):
        self.assertEqual(ca._strip_matched_quotes("`hello`"), "hello")

    def test_smart_single(self):
        self.assertEqual(ca._strip_matched_quotes("‘hello’"), "hello")

    def test_smart_double(self):
        self.assertEqual(ca._strip_matched_quotes("“hello”"), "hello")

    def test_mismatched_pair_preserved(self):
        # Single quote at start, double at end — not a recognized pair.
        self.assertEqual(ca._strip_matched_quotes("'hello\""), "'hello\"")

    def test_internal_quote_preserved(self):
        self.assertEqual(ca._strip_matched_quotes("'O'Brien'"), "O'Brien")

    def test_no_quotes_unchanged(self):
        self.assertEqual(ca._strip_matched_quotes("hello"), "hello")

    def test_empty(self):
        self.assertEqual(ca._strip_matched_quotes(""), "")

    def test_non_string(self):
        self.assertEqual(ca._strip_matched_quotes(None), "")
        self.assertEqual(ca._strip_matched_quotes(42), "")

    def test_only_one_char(self):
        self.assertEqual(ca._strip_matched_quotes("'"), "'")


# ─── _classify_todo: kind=type ────────────────────────────────────────────


class TestClassifyType(unittest.TestCase):
    def test_quoted_value_in_field(self):
        out = ca._classify_todo("Type 'John' in First Name")
        self.assertEqual(out, {"kind": "type", "target": "", "field": "First Name", "value": "John"})

    def test_double_quoted_value(self):
        out = ca._classify_todo('Type "John" in First Name')
        self.assertEqual(out["value"], "John")
        self.assertEqual(out["field"], "First Name")

    def test_unquoted_value_email(self):
        out = ca._classify_todo("Type test@example.com in Work Email")
        self.assertEqual(out["kind"], "type")
        self.assertEqual(out["value"], "test@example.com")
        self.assertEqual(out["field"], "Work Email")

    def test_into_connector(self):
        out = ca._classify_todo("Type 'hello' into Notepad")
        self.assertEqual(out["kind"], "type")
        self.assertEqual(out["value"], "hello")
        self.assertEqual(out["field"], "Notepad")

    def test_enter_verb(self):
        out = ca._classify_todo("Enter '1234' in Contact Number")
        self.assertEqual(out["kind"], "type")
        self.assertEqual(out["value"], "1234")

    def test_fill_verb(self):
        out = ca._classify_todo("Fill 'Jane' in Last Name")
        self.assertEqual(out["kind"], "type")

    def test_trailing_period(self):
        out = ca._classify_todo("Type 'John' in First Name.")
        self.assertEqual(out["field"], "First Name")

    def test_verb_only_no_connector_falls_to_other(self):
        # "Type the document" has no "in"/"into" — degrade to other.
        out = ca._classify_todo("Type the document")
        self.assertEqual(out["kind"], "other")

    def test_case_insensitive_verb(self):
        out = ca._classify_todo("TYPE 'X' IN Y")
        self.assertEqual(out["kind"], "type")


# ─── _classify_todo: kind=select ──────────────────────────────────────────


class TestClassifySelect(unittest.TestCase):
    def test_dropdown_suffix_stripped(self):
        out = ca._classify_todo("Select '1-50' from Staff Size dropdown")
        self.assertEqual(out, {"kind": "select", "target": "", "field": "Staff Size", "value": "1-50"})

    def test_no_suffix(self):
        out = ca._classify_todo("Select 'IT' from Industry")
        self.assertEqual(out["field"], "Industry")
        self.assertEqual(out["value"], "IT")

    def test_choose_verb(self):
        out = ca._classify_todo("Choose Construction from Industry dropdown")
        self.assertEqual(out["kind"], "select")
        self.assertEqual(out["value"], "Construction")
        self.assertEqual(out["field"], "Industry")

    def test_pick_verb(self):
        out = ca._classify_todo("Pick Yes from Confirm")
        self.assertEqual(out["kind"], "select")
        self.assertEqual(out["value"], "Yes")

    def test_menu_suffix_stripped(self):
        out = ca._classify_todo("Select 'Foo' from Bar menu")
        self.assertEqual(out["field"], "Bar")

    def test_list_suffix_stripped(self):
        out = ca._classify_todo("Select 'Foo' from Bar list")
        self.assertEqual(out["field"], "Bar")

    def test_combobox_suffix_stripped(self):
        out = ca._classify_todo("Select 'Foo' from Bar combobox")
        self.assertEqual(out["field"], "Bar")

    def test_leading_the_stripped_from_field(self):
        out = ca._classify_todo("Select 'IT' from the Industry dropdown")
        self.assertEqual(out["field"], "Industry")

    def test_in_connector_treated_as_select(self):
        out = ca._classify_todo("Select 'IT' in Industry")
        self.assertEqual(out["kind"], "select")
        self.assertEqual(out["field"], "Industry")


# ─── _classify_todo: kind=click ───────────────────────────────────────────


class TestClassifyClick(unittest.TestCase):
    def test_quoted_target_with_button(self):
        out = ca._classify_todo("Click 'Schedule a Demo' button")
        self.assertEqual(out, {"kind": "click", "target": "Schedule a Demo", "field": "", "value": ""})

    def test_bare_target(self):
        out = ca._classify_todo("Click Submit")
        self.assertEqual(out["target"], "Submit")

    def test_press_verb(self):
        out = ca._classify_todo("Press Save")
        self.assertEqual(out["kind"], "click")
        self.assertEqual(out["target"], "Save")

    def test_leading_the_stripped(self):
        out = ca._classify_todo("Press the Save button")
        self.assertEqual(out["target"], "Save")

    def test_link_suffix_stripped(self):
        out = ca._classify_todo("Click 'Sign in' link")
        self.assertEqual(out["target"], "Sign in")

    def test_tap_verb(self):
        out = ca._classify_todo("Tap OK")
        self.assertEqual(out["kind"], "click")
        self.assertEqual(out["target"], "OK")

    def test_quoted_no_suffix(self):
        out = ca._classify_todo("Click 'Cancel'")
        self.assertEqual(out["target"], "Cancel")


# ─── _classify_todo: kind=other / edge cases ──────────────────────────────


class TestClassifyOther(unittest.TestCase):
    def test_unrecognized_verb(self):
        self.assertEqual(ca._classify_todo("Submit form")["kind"], "other")

    def test_navigation_verb(self):
        self.assertEqual(ca._classify_todo("Open the new tab")["kind"], "other")

    def test_empty_string(self):
        self.assertEqual(ca._classify_todo("")["kind"], "other")

    def test_whitespace_only(self):
        self.assertEqual(ca._classify_todo("   ")["kind"], "other")

    def test_none_input(self):
        out = ca._classify_todo(None)
        self.assertEqual(out["kind"], "other")
        self.assertEqual(out["target"], "")
        self.assertEqual(out["field"], "")
        self.assertEqual(out["value"], "")

    def test_non_string_input(self):
        self.assertEqual(ca._classify_todo(42)["kind"], "other")

    def test_returns_full_dict_always(self):
        # Defensive: every return value must have all four keys.
        for case in ["", "Type 'X' in Y", "Click X", "Select 'A' from B", "junk"]:
            out = ca._classify_todo(case)
            self.assertEqual(set(out.keys()), {"kind", "target", "field", "value"})


# ─── _make_todo_dict ──────────────────────────────────────────────────────


class TestMakeTodoDict(unittest.TestCase):
    def test_schema_complete(self):
        d = ca._make_todo_dict(7, "Type 'X' in Y")
        expected_keys = {"id", "task", "done", "kind", "target", "field", "value",
                         "pending_visual_confirm", "confirm_strikes",
                         # Fix A — abandoned-confirm flag (default False)
                         "confirm_abandoned",
                         # recovery engagement timestamps (default -1)
                         "batch_marked_done", "batch_deferred"}
        self.assertEqual(set(d.keys()), expected_keys)

    def test_engagement_timestamps_default_to_negative_one(self):
        d = ca._make_todo_dict(1, "Click Submit")
        self.assertEqual(d["batch_marked_done"], -1)
        self.assertEqual(d["batch_deferred"], -1)

    def test_confirm_abandoned_defaults_to_false(self):
        # Fix A: every freshly-created TODO starts with confirm_abandoned=False.
        # The flag flips True only on the 3-strike fallback NO path.
        d = ca._make_todo_dict(1, "Select '1-50' from Staff Size dropdown")
        self.assertFalse(d["confirm_abandoned"])

    def test_id_and_task_set(self):
        d = ca._make_todo_dict(3, "Click Submit")
        self.assertEqual(d["id"], 3)
        self.assertEqual(d["task"], "Click Submit")
        self.assertFalse(d["done"])

    def test_classification_populated(self):
        d = ca._make_todo_dict(1, "Select '1-50' from Staff Size dropdown")
        self.assertEqual(d["kind"], "select")
        self.assertEqual(d["field"], "Staff Size")
        self.assertEqual(d["value"], "1-50")

    def test_runtime_fields_default(self):
        d = ca._make_todo_dict(1, "Type 'X' in Y")
        self.assertFalse(d["pending_visual_confirm"])
        self.assertEqual(d["confirm_strikes"], 0)


# ─── _TaskState integration ───────────────────────────────────────────────


class TestTaskStatePE1gPlumbing(unittest.TestCase):
    def setUp(self):
        self.s = ca._TaskState()

    def test_set_initial_todos_classifies_each(self):
        self.s.set_initial_todos([
            "Type 'John' in First Name",
            "Select '1-50' from Staff Size dropdown",
            "Click 'Submit' button",
            "Open the page",  # other
        ])
        kinds = [t["kind"] for t in self.s.todo_list]
        self.assertEqual(kinds, ["type", "select", "click", "other"])

    def test_add_todo_classifies(self):
        self.s.add_todo("Type 'extra' in Bonus Field")
        last = self.s.todo_list[-1]
        self.assertEqual(last["kind"], "type")
        self.assertEqual(last["value"], "extra")
        self.assertEqual(last["field"], "Bonus Field")

    def test_state_defaults(self):
        self.assertEqual(self.s.zero_progress_streak, 0)
        self.assertEqual(self.s.loop_budget, ca.MAX_LOOPS)
        self.assertEqual(self.s.confirm_fallback_count, 0)

    def test_reset_clears_state(self):
        self.s.zero_progress_streak = 5
        self.s.loop_budget = 13
        self.s.confirm_fallback_count = 2
        self.s.set_initial_todos(["Type 'X' in Y"])
        self.s.reset()
        self.assertEqual(self.s.zero_progress_streak, 0)
        self.assertEqual(self.s.loop_budget, ca.MAX_LOOPS)
        self.assertEqual(self.s.confirm_fallback_count, 0)
        self.assertEqual(self.s.todo_list, [])

    def test_pe1_existing_fields_still_present(self):
        """Regression: the PE-1 contract (id/task/done) survives planner-vision."""
        self.s.set_initial_todos(["Type 'X' in Y"])
        t = self.s.todo_list[0]
        self.assertIn("id", t)
        self.assertIn("task", t)
        self.assertIn("done", t)
        self.assertEqual(t["id"], 1)
        self.assertFalse(t["done"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
