"""
test_fix_b_pending_progress.py — Fix B (2026-04-26):
Render `pending_visual_confirm=True` TODOs distinctly in todo_progress_str()
so the planner skips them during the 1-3 strike confirm window.

Why: without Fix B, a Rule-S-deferred TODO renders identically to an
unattempted TODO (`✗`). The planner reads `✗` as "do this next" and
re-attempts the same select, forming an infinite retry loop until MAX_LOOPS.
Observed on Truein form 2026-04-26 third live test.

Fix B's contract:
  ✓ done           — completed (regular)
  ✓ done (unconfirmed) — abandoned (Fix A); rendered with annotation
                          for debug.log readers, planner just sees ✓
  · pending confirm — Rule S deferred; "(awaiting confirm — do not retry)"
                      annotation; "← NEXT" marker NEVER lands here
  ✗ open            — not yet attempted; "← NEXT" on the first one

Header gains a "(N awaiting confirm)" clause when any pending exists.

Planner system prompt has a new rule explaining the three symbols so the
LLM interprets `·` as "do not retry" rather than guessing.

Run: python test_fix_b_pending_progress.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.vision as ca


class TestPendingConfirmRendering(unittest.TestCase):
    def setUp(self):
        ca._task_state.reset()

    def tearDown(self):
        ca._task_state.reset()

    def test_pending_renders_with_dot_symbol(self):
        ca._task_state.set_initial_todos([
            "Type 'John' in First Name",
            "Select '1-50' from Staff Size dropdown",
        ])
        ca._task_state.todo_list[1]["pending_visual_confirm"] = True
        out = ca._task_state.todo_progress_str()
        # The pending line uses the · symbol, NOT ✗
        pending_line = [ln for ln in out.split("\n") if "Staff Size" in ln][0]
        self.assertIn("·", pending_line)
        self.assertNotIn("✗", pending_line)

    def test_pending_carries_do_not_retry_annotation(self):
        ca._task_state.set_initial_todos(["Select '1-50' from Staff Size dropdown"])
        ca._task_state.todo_list[0]["pending_visual_confirm"] = True
        out = ca._task_state.todo_progress_str()
        self.assertIn("awaiting confirm", out.lower())
        self.assertIn("do not retry", out.lower())

    def test_next_marker_skips_pending(self):
        # Critical: "← NEXT" must land on the first OPEN (✗) TODO, never on
        # a pending one. Otherwise the planner reads `· ... ← NEXT` and
        # retries, defeating Fix B's purpose.
        ca._task_state.set_initial_todos([
            "Type 'John' in First Name",
            "Select '1-50' from Staff Size dropdown",
            "Select 'IT' from Industry dropdown",
            "Type 'test@example.com' in Work Email",
        ])
        ca._task_state.todo_list[0]["done"] = True   # ✓
        ca._task_state.todo_list[1]["pending_visual_confirm"] = True  # ·
        # TODOs 2 and 3 stay open (✗)
        out = ca._task_state.todo_progress_str()
        # NEXT marker should land on TODO #2 (the FIRST ✗), not on #1 (·)
        next_line = [ln for ln in out.split("\n") if "← NEXT" in ln][0]
        self.assertIn("Industry", next_line, "← NEXT must land on first ✗, not ·")
        self.assertNotIn("Staff Size", next_line)

    def test_pending_does_not_count_as_done_in_header(self):
        # Header reports "X of N done". Pending TODOs are not done.
        ca._task_state.set_initial_todos([
            "Type 'A' in F1", "Type 'B' in F2", "Select 'C' from F3 dropdown",
        ])
        ca._task_state.todo_list[0]["done"] = True
        ca._task_state.todo_list[2]["pending_visual_confirm"] = True
        out = ca._task_state.todo_progress_str()
        # 1 done out of 3, 1 awaiting confirm
        self.assertIn("1 of 3 done", out)

    def test_header_includes_awaiting_confirm_count(self):
        ca._task_state.set_initial_todos([
            "Select 'A' from F1 dropdown",
            "Select 'B' from F2 dropdown",
            "Type 'C' in F3",
        ])
        ca._task_state.todo_list[0]["pending_visual_confirm"] = True
        ca._task_state.todo_list[1]["pending_visual_confirm"] = True
        out = ca._task_state.todo_progress_str()
        self.assertIn("2 awaiting confirm", out)

    def test_header_omits_awaiting_confirm_when_none(self):
        # When zero pending, header stays clean — no spurious annotation.
        ca._task_state.set_initial_todos(["Type 'A' in F1"])
        out = ca._task_state.todo_progress_str()
        self.assertNotIn("awaiting confirm", out)

    def test_three_states_render_simultaneously(self):
        # Mixed state: done, abandoned-done, pending, open. All render
        # distinctly; planner can disambiguate every line.
        ca._task_state.set_initial_todos([
            "Type 'John' in First Name",                # done
            "Select '1-50' from Staff Size dropdown",   # abandoned
            "Select 'IT' from Industry dropdown",       # pending
            "Type 'x@y.com' in Email",                  # open
        ])
        ca._task_state.todo_list[0]["done"] = True
        ca._task_state.todo_list[1]["done"] = True
        ca._task_state.todo_list[1]["confirm_abandoned"] = True
        ca._task_state.todo_list[2]["pending_visual_confirm"] = True
        out = ca._task_state.todo_progress_str()
        lines = out.split("\n")
        # Header: 2 of 4 done, 1 awaiting confirm
        self.assertIn("2 of 4 done", lines[0])
        self.assertIn("1 awaiting confirm", lines[0])
        # Per-line shape
        first_name_line = [ln for ln in lines if "First Name" in ln][0]
        staff_line = [ln for ln in lines if "Staff Size" in ln][0]
        industry_line = [ln for ln in lines if "Industry" in ln][0]
        email_line = [ln for ln in lines if "Email" in ln][0]
        self.assertIn("✓", first_name_line)
        self.assertNotIn("(unconfirmed)", first_name_line)
        self.assertIn("✓", staff_line)
        self.assertIn("(unconfirmed)", staff_line)
        self.assertIn("·", industry_line)
        self.assertIn("awaiting confirm", industry_line)
        self.assertIn("✗", email_line)
        self.assertIn("← NEXT", email_line)

    def test_all_pending_no_next_marker(self):
        # If ONLY pending and done exist (no open), no "← NEXT" should
        # appear (don't fake-mark a pending as next).
        ca._task_state.set_initial_todos([
            "Type 'A' in F1",
            "Select 'B' from F2 dropdown",
        ])
        ca._task_state.todo_list[0]["done"] = True
        ca._task_state.todo_list[1]["pending_visual_confirm"] = True
        out = ca._task_state.todo_progress_str()
        self.assertNotIn("← NEXT", out)

    def test_pending_with_done_flag_renders_as_done(self):
        # Defensive: if both done=True AND pending_visual_confirm=True somehow
        # exist (shouldn't, but Fix A's mark_todo_done doesn't clear pending
        # via the abandoned path — wait, it does. But future code paths
        # might.) The done branch must win since done is the terminal state.
        ca._task_state.set_initial_todos(["Select 'A' from F1 dropdown"])
        ca._task_state.todo_list[0]["done"] = True
        ca._task_state.todo_list[0]["pending_visual_confirm"] = True
        out = ca._task_state.todo_progress_str()
        self.assertIn("✓", out)
        self.assertNotIn("·", out)


class TestPlannerPromptSymbolGuidance(unittest.TestCase):
    """
    The planner system prompt must explain the · symbol so the LLM treats
    it as "do not retry" rather than a guess. Guard the prompt body so
    future edits don't silently break the contract.
    """

    def test_prompt_documents_three_symbols(self):
        prompt = ca.VISION_PLANNER_SYSTEM_PROMPT
        # All three symbols mentioned
        self.assertIn("✓", prompt)
        self.assertIn("·", prompt)
        self.assertIn("✗", prompt)

    def test_prompt_says_never_re_attempt_pending(self):
        prompt = ca.VISION_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn("never re-attempt", prompt)
        # And the canonical phrase the planner should pattern-match on
        self.assertIn("awaiting confirm", prompt)

    def test_prompt_warns_against_retry_loops(self):
        # The cost case ("burns the loop budget") is the actual failure
        # planner-vision is fixing — the prompt should mention it so the
        # model understands the "why" and doesn't second-guess.
        prompt = ca.VISION_PLANNER_SYSTEM_PROMPT.lower()
        self.assertIn("retry loop", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
