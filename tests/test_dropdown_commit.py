"""
test_dropdown_commit.py — planner-vision dropdown-commit: dropdown commit safety guard.

Covers:
  - _canonical_token normalization (lowercase, quote-strip, whitespace collapse)
  - _action_target_text extraction priority
  - _batch_has_recent_dropdown_click: returns False when no select-TODOs,
    False when no overlap, True on field-text overlap, True on value-text
    overlap, case-insensitive
  - _inject_dropdown_commit_if_needed:
    * empty actions → no-op
    * non-keyboard trailing action → no-op
    * keyboard_press(enter) trailing → no-op
    * keyboard_press(down) trailing + dropdown context → injects Enter
    * keyboard_press(down) trailing + NO context → no-op (chat-box safety)
    * keyboard_press(down) trailing + pending_visual_confirm select-TODO → injects
    * arrow + screenshot_and_continue at tail → Enter injected BEFORE screenshot
    * config flag DROPDOWN_COMMIT_GUARD_ENABLED=False → no-op even with context

Run: python test_dropdown_commit.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca
import assistant.config as cfg


# ─── _canonical_token + _action_target_text ────────────────────────────────


class TestCanonicalToken(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(ca._canonical_token("Staff Size"), "staff size")

    def test_quote_strip_double(self):
        self.assertEqual(ca._canonical_token('"1-50"'), "1-50")

    def test_quote_strip_single(self):
        self.assertEqual(ca._canonical_token("'IT'"), "it")

    def test_smart_quotes(self):
        self.assertEqual(ca._canonical_token("“Work Email”"), "work email")

    def test_whitespace_collapse(self):
        self.assertEqual(ca._canonical_token("  Schedule   a  Demo  "), "schedule a demo")

    def test_non_string_returns_empty(self):
        self.assertEqual(ca._canonical_token(None), "")
        self.assertEqual(ca._canonical_token(42), "")


class TestActionTargetText(unittest.TestCase):
    def test_text_field_preferred(self):
        self.assertEqual(
            ca._action_target_text({"type": "vision_guided_click", "text": "OK", "name": "ignored"}),
            "OK",
        )

    def test_target_description_fallback(self):
        self.assertEqual(
            ca._action_target_text({"type": "mouse_click", "target_description": "the icon"}),
            "the icon",
        )

    def test_name_fallback(self):
        self.assertEqual(ca._action_target_text({"type": "open_application", "name": "Notepad"}), "Notepad")

    def test_empty_when_none_present(self):
        self.assertEqual(ca._action_target_text({"type": "wait", "seconds": 1.0}), "")


# ─── _batch_has_recent_dropdown_click ──────────────────────────────────────


class TestBatchHasRecentDropdownClick(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def _stub_select_todo(self, field, value):
        """Synthesize the post-planner-vision TODO state directly."""
        ca._task_state.set_initial_todos([f"Select '{value}' from {field} dropdown"])
        ca._task_state.todo_list[0]["kind"] = "select"
        ca._task_state.todo_list[0]["field"] = field
        ca._task_state.todo_list[0]["value"] = value
        ca._task_state.todo_list[0]["pending_visual_confirm"] = False
        ca._task_state.todo_list[0]["confirm_strikes"] = 0

    def test_no_select_todos_returns_false(self):
        actions = [{"type": "vision_guided_click", "text": "Staff Size"}]
        self.assertFalse(ca._batch_has_recent_dropdown_click(actions))

    def test_no_click_actions_returns_false(self):
        self._stub_select_todo("Staff Size", "1-50")
        actions = [{"type": "keyboard_type", "text": "x"}]
        self.assertFalse(ca._batch_has_recent_dropdown_click(actions))

    def test_field_text_overlap_returns_true(self):
        self._stub_select_todo("Staff Size", "1-50")
        actions = [{"type": "vision_guided_click", "text": "Staff Size"}]
        self.assertTrue(ca._batch_has_recent_dropdown_click(actions))

    def test_value_text_overlap_returns_true(self):
        self._stub_select_todo("Industry", "IT")
        actions = [{"type": "find_and_click_text", "text": "IT"}]
        self.assertTrue(ca._batch_has_recent_dropdown_click(actions))

    def test_case_insensitive(self):
        self._stub_select_todo("Industry", "Information Technology")
        actions = [{"type": "vision_guided_click", "text": "INDUSTRY"}]
        self.assertTrue(ca._batch_has_recent_dropdown_click(actions))

    def test_substring_either_direction(self):
        # Action target is a superset of TODO field
        self._stub_select_todo("Industry", "IT")
        actions = [{"type": "vision_guided_click", "text": "Industry dropdown"}]
        self.assertTrue(ca._batch_has_recent_dropdown_click(actions))

    def test_no_overlap_returns_false(self):
        self._stub_select_todo("Industry", "IT")
        actions = [{"type": "vision_guided_click", "text": "First Name"}]
        self.assertFalse(ca._batch_has_recent_dropdown_click(actions))


# ─── _inject_dropdown_commit_if_needed ─────────────────────────────────────


class TestInjectDropdownCommit(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()
        # Default: flag enabled — tests for the disabled path will override.
        cfg.DROPDOWN_COMMIT_GUARD_ENABLED = True

    def tearDown(self):
        ca._task_state.reset()
        cfg.DROPDOWN_COMMIT_GUARD_ENABLED = True

    def _stub_select_todo(self, field, value, pending=False):
        ca._task_state.set_initial_todos([f"Select '{value}' from {field} dropdown"])
        ca._task_state.todo_list[0]["kind"] = "select"
        ca._task_state.todo_list[0]["field"] = field
        ca._task_state.todo_list[0]["value"] = value
        ca._task_state.todo_list[0]["pending_visual_confirm"] = pending
        ca._task_state.todo_list[0]["confirm_strikes"] = 0

    def test_empty_actions_noop(self):
        self.assertEqual(ca._inject_dropdown_commit_if_needed([]), [])

    def test_only_screenshot_and_wait_noop(self):
        actions = [{"type": "wait", "seconds": 1.0}, {"type": "screenshot_and_continue"}]
        self.assertEqual(ca._inject_dropdown_commit_if_needed(actions), actions)

    def test_non_keyboard_trailing_noop(self):
        self._stub_select_todo("Industry", "IT")
        actions = [{"type": "vision_guided_click", "text": "Industry"}]
        self.assertEqual(ca._inject_dropdown_commit_if_needed(actions), actions)

    def test_keyboard_press_enter_already_present_noop(self):
        self._stub_select_todo("Industry", "IT")
        actions = [
            {"type": "vision_guided_click", "text": "Industry"},
            {"type": "keyboard_press", "key": "down"},
            {"type": "keyboard_press", "key": "enter"},
        ]
        self.assertEqual(ca._inject_dropdown_commit_if_needed(actions), actions)

    def test_arrow_down_with_dropdown_context_injects_enter(self):
        self._stub_select_todo("Industry", "IT")
        actions = [
            {"type": "vision_guided_click", "text": "Industry"},
            {"type": "keyboard_press", "key": "down"},
            {"type": "keyboard_press", "key": "down"},
        ]
        out = ca._inject_dropdown_commit_if_needed(actions)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[-1], {"type": "keyboard_press", "key": "enter"})

    def test_arrow_down_no_context_noop(self):
        # No select-TODOs at all — chat-box / search-list scenario, must not inject.
        actions = [
            {"type": "vision_guided_click", "text": "Message"},
            {"type": "keyboard_press", "key": "down"},
        ]
        self.assertEqual(ca._inject_dropdown_commit_if_needed(actions), actions)

    def test_pending_visual_confirm_triggers_inject(self):
        # Earlier batch already clicked the dropdown; this batch only has
        # the navigation. No click in THIS batch overlaps the field, but
        # pending_visual_confirm is set — guard should still fire.
        self._stub_select_todo("Industry", "IT", pending=True)
        actions = [{"type": "keyboard_press", "key": "down"}]
        out = ca._inject_dropdown_commit_if_needed(actions)
        self.assertEqual(out, [
            {"type": "keyboard_press", "key": "down"},
            {"type": "keyboard_press", "key": "enter"},
        ])

    def test_inject_before_trailing_screenshot(self):
        # Enter must land BEFORE screenshot_and_continue, not after it.
        self._stub_select_todo("Industry", "IT")
        actions = [
            {"type": "vision_guided_click", "text": "Industry"},
            {"type": "keyboard_press", "key": "down"},
            {"type": "screenshot_and_continue"},
        ]
        out = ca._inject_dropdown_commit_if_needed(actions)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[1], {"type": "keyboard_press", "key": "down"})
        self.assertEqual(out[2], {"type": "keyboard_press", "key": "enter"})
        self.assertEqual(out[3], {"type": "screenshot_and_continue"})

    def test_pageup_pagedown_arrow_keys_also_trigger(self):
        self._stub_select_todo("Industry", "IT")
        for key in ("up", "pagedown", "pageup"):
            actions = [
                {"type": "vision_guided_click", "text": "Industry"},
                {"type": "keyboard_press", "key": key},
            ]
            out = ca._inject_dropdown_commit_if_needed(actions)
            self.assertEqual(out[-1], {"type": "keyboard_press", "key": "enter"},
                             f"Expected Enter injection for trailing key={key}")

    def test_non_arrow_key_noop(self):
        # A non-arrow keyboard_press should NOT trigger the guard even with context.
        self._stub_select_todo("Industry", "IT")
        actions = [
            {"type": "vision_guided_click", "text": "Industry"},
            {"type": "keyboard_press", "key": "tab"},
        ]
        self.assertEqual(ca._inject_dropdown_commit_if_needed(actions), actions)

    def test_config_flag_off_disables_guard(self):
        cfg.DROPDOWN_COMMIT_GUARD_ENABLED = False
        self._stub_select_todo("Industry", "IT")
        actions = [
            {"type": "vision_guided_click", "text": "Industry"},
            {"type": "keyboard_press", "key": "down"},
        ]
        # With the flag off the guard MUST NOT modify the batch even when
        # all conditions are otherwise met.
        self.assertEqual(ca._inject_dropdown_commit_if_needed(actions), actions)


# ─── Audit fixes #2 + #9 in _parse_todo_list ───────────────────────────────


class TestParseTodoListAuditFixes(unittest.TestCase):
    def test_completed_key_no_longer_returns_updater_output(self):
        """Audit #2: 'completed' was wrongly listed as a generator-output key."""
        # If the generator ever returned this updater-shaped object, the old
        # code returned the empty completed list and disabled PE-1 silently.
        # Now we should fall through and find nothing usable → return [].
        out = ca._parse_todo_list('{"completed":[],"new":["A","B"]}')
        # 'new' is also no longer in the key list — total miss → empty list.
        self.assertEqual(out, [])

    def test_empty_list_at_top_level_does_not_short_circuit(self):
        """Audit #9: empty list at top-level used to return [] prematurely.

        With multi-candidate parsing, the `[]` then `{...}` candidates are
        tried in order. An empty top-level list now falls through so a later
        object-wrapper candidate can succeed.
        """
        # The whole text is an empty list — no other candidates can fire.
        # Verify it returns [] (nothing to extract) rather than crashing.
        self.assertEqual(ca._parse_todo_list("[]"), [])

    def test_empty_first_dict_key_falls_through(self):
        """Audit #9: when 'list' is empty but 'tasks' is populated, return tasks."""
        out = ca._parse_todo_list('{"list":[],"tasks":["A","B"]}')
        self.assertEqual(out, ["A", "B"])

    def test_pe1_basic_list_still_parses(self):
        """Regression guard: existing PE-1 happy paths must still work."""
        self.assertEqual(ca._parse_todo_list('["x","y"]'), ["x", "y"])

    def test_pe1_object_wrapper_todos_still_parses(self):
        """Regression guard: 'todos' key remains the canonical wrapper."""
        self.assertEqual(ca._parse_todo_list('{"todos":["A","B"]}'), ["A", "B"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
