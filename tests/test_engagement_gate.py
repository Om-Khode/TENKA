"""
test_engagement_gate.py — recovery dialog-engagement gate (2026-04-26 fix).

Bug context: the recovery layer's _diagnose mis-classified the Truein "Book A Free Demo"
form-modal the agent was actively filling as an unwanted overlay, then
clicked the close X to "recover" — dismissing the form mid-task.

Fix: a code-level veto AFTER _diagnose. When recent agent actions show
successful engagement with the visible modal surface (TODO marked
done or deferred within a 2-batch window, OR — for non-TODO tasks — a
non-failed action in the recent batch), the overlay_appeared dispatch is
suppressed. Other recovery classes (error_shown, no_change, unknown) are
NOT gated — they are non-destructive.

Covers:
  - _is_dialog_engagement_active: TODO marked done within window, deferred
    within window, stale (outside window), no TODOs + recent success, no
    TODOs + all failures, empty inputs
  - checkpoint integration:
    * First-batch cookie banner (no engagement) → dismiss fires
    * Engaged modal (TODO done last batch) → gate suppresses dismiss
    * Engaged via pending_visual_confirm → gate suppresses
    * Stale engagement outside window → gate opens
    * No-TODO task with recent success → gate fires (signal B)
    * Kill switch disables gate
    * error_shown / no_change / unknown classes NOT gated
  - Truien scenario regression: 3 prior batches each marked a Type TODO done,
    then recovery diagnoses overlay → gate suppresses, no _recover_overlay call

Run: python test_engagement_gate.py
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.recovery as rec
from assistant import config as _config_stub


# ─── _is_dialog_engagement_active (pure function) ──────────────────────────


class TestEngagementSignal(unittest.TestCase):
    def _todo(self, **kw):
        """Build a minimal TODO dict for engagement testing."""
        base = {
            "id": 1, "task": "x", "done": False, "kind": "type",
            "target": "", "field": "", "value": "",
            "pending_visual_confirm": False, "confirm_strikes": 0,
            "batch_marked_done": -1, "batch_deferred": -1,
        }
        base.update(kw)
        return base

    def test_no_inputs_not_engaged(self):
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=None, recent_action_results=None, current_batch_idx=5,
        )
        self.assertFalse(engaged)

    def test_todo_marked_done_in_window(self):
        # Current batch=5, window=2 → threshold=3. Stamp at batch 4 → engaged.
        todos = [self._todo(done=True, batch_marked_done=4)]
        engaged, reason = rec._is_dialog_engagement_active(
            todo_snapshot=todos, recent_action_results=None, current_batch_idx=5,
        )
        self.assertTrue(engaged)
        self.assertIn("marked done in batch 4", reason)

    def test_todo_marked_done_at_threshold(self):
        # Current=5, window=2 → threshold=3. Stamp at exactly batch 3 → engaged.
        todos = [self._todo(done=True, batch_marked_done=3)]
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=todos, recent_action_results=None, current_batch_idx=5,
        )
        self.assertTrue(engaged)

    def test_todo_marked_done_outside_window(self):
        # Current=5, window=2 → threshold=3. Stamp at batch 2 → stale → not engaged.
        todos = [self._todo(done=True, batch_marked_done=2)]
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=todos, recent_action_results=None, current_batch_idx=5,
        )
        self.assertFalse(engaged)

    def test_todo_deferred_in_window(self):
        todos = [self._todo(kind="select", pending_visual_confirm=True, batch_deferred=4)]
        engaged, reason = rec._is_dialog_engagement_active(
            todo_snapshot=todos, recent_action_results=None, current_batch_idx=5,
        )
        self.assertTrue(engaged)
        self.assertIn("deferred in batch 4", reason)

    def test_todos_present_but_none_recent(self):
        # TODOs exist but none progressed recently — gate is open.
        todos = [self._todo(done=False, batch_marked_done=-1)]
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=todos, recent_action_results=None, current_batch_idx=5,
        )
        self.assertFalse(engaged)

    def test_default_stamp_negative_one_does_not_count(self):
        # Default -1 must never trigger engagement, even with current_batch_idx=0.
        todos = [self._todo(done=True, batch_marked_done=-1)]
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=todos, recent_action_results=None, current_batch_idx=0,
        )
        self.assertFalse(engaged)

    def test_no_todos_recent_success_falls_back(self):
        # No TODO tracking — fallback to action-result heuristic.
        engaged, reason = rec._is_dialog_engagement_active(
            todo_snapshot=None,
            recent_action_results=["Clicked 'Submit' at (100,200)"],
            current_batch_idx=3,
        )
        self.assertTrue(engaged)
        self.assertIn("non-failed action", reason)

    def test_no_todos_all_failures_not_engaged(self):
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=None,
            recent_action_results=["Failed: timeout", "ABORTED_WRONG_FOCUS"],
            current_batch_idx=3,
        )
        self.assertFalse(engaged)

    def test_no_todos_empty_results_not_engaged(self):
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=None, recent_action_results=[], current_batch_idx=3,
        )
        self.assertFalse(engaged)

    def test_empty_todo_list_falls_back_to_action_results(self):
        # Empty list (not None) = no TODO tracking active for this task.
        engaged, _ = rec._is_dialog_engagement_active(
            todo_snapshot=[],
            recent_action_results=["Clicked 'OK'"],
            current_batch_idx=3,
        )
        self.assertTrue(engaged)


# ─── checkpoint() integration with the gate ────────────────────────────────


def _stub_screen_llm(*, screenshot="fake-b64", vision_response="{}"):
    """Install fake screen + llm for checkpoint diagnose stage.

    ``get_vision_response`` now returns ``LLMResult`` objects so bare string
    values are wrapped in ``SimpleNamespace(text=...)`` so callers can do
    ``.text`` on the result.
    """
    screen_mod = types.ModuleType("assistant.io.screen")
    screen_mod.capture_screenshot_base64 = MagicMock(return_value=screenshot)
    llm_mod = types.ModuleType("assistant.llm")
    if isinstance(vision_response, Exception):
        llm_mod.get_vision_response = AsyncMock(side_effect=vision_response)
    else:
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text=vision_response)
        )
    sys.modules["assistant.io.screen"] = screen_mod
    sys.modules["assistant.llm"] = llm_mod
    return screen_mod, llm_mod


class TestCheckpointGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _config_stub.DIALOG_ENGAGEMENT_GATE_ENABLED = True

    def _todo_done(self, batch_idx):
        return {
            "id": 1, "task": "Type 'X' in Y", "done": True, "kind": "type",
            "target": "", "field": "Y", "value": "X",
            "pending_visual_confirm": False, "confirm_strikes": 0,
            "batch_marked_done": batch_idx, "batch_deferred": -1,
        }

    async def test_first_batch_cookie_banner_dismiss_fires(self):
        """No engagement signal → recovery dismisses (cookie-banner case preserved)."""
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared","detail":"cookie banner",'
                            '"recovery_target":"Accept All button"}'
        )
        # Patch _recover_overlay so we can verify it WAS called.
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            co = await rec.checkpoint(
                goal="read article",
                last_action={"type": "computer_agent", "action": "click", "params": {}, "goal": ""},
                todo_snapshot=None,            # no TODO tracking
                recent_action_results=None,     # no recent actions
                current_batch_idx=1,
            )
        mock_overlay.assert_awaited_once()
        self.assertEqual(co.diagnosed_class, "overlay_appeared")
        self.assertEqual(co.action_taken, "bbox_click")

    async def test_engaged_modal_suppresses_dismiss(self):
        """Engagement hot (TODO done last batch) → no _recover_overlay call."""
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared","detail":"book a demo modal",'
                            '"recovery_target":"close X button"}'
        )
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            co = await rec.checkpoint(
                goal="fill the form",
                last_action={"type": "computer_agent", "action": "type", "params": {}, "goal": ""},
                todo_snapshot=[self._todo_done(batch_idx=4)],
                recent_action_results=['Typed: "Test"'],
                current_batch_idx=5,
            )
        mock_overlay.assert_not_awaited()
        self.assertEqual(co.diagnosed_class, "overlay_appeared")
        self.assertEqual(co.action_taken, "none")
        self.assertFalse(co.recovered)
        self.assertIn("[gated:", co.detail)
        self.assertIn("TODO #1", co.detail)

    async def test_pending_select_confirm_triggers_gate(self):
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared","detail":"modal","recovery_target":"X"}'
        )
        todo = self._todo_done(batch_idx=-1)
        todo["done"] = False
        todo["batch_marked_done"] = -1
        todo["pending_visual_confirm"] = True
        todo["batch_deferred"] = 4
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            co = await rec.checkpoint(
                goal="x", last_action={"type": "computer_agent", "action": "click",
                                         "params": {}, "goal": ""},
                todo_snapshot=[todo],
                recent_action_results=["Clicked 'Industry'"],
                current_batch_idx=5,
            )
        mock_overlay.assert_not_awaited()
        self.assertEqual(co.action_taken, "none")

    async def test_stale_engagement_outside_window_does_not_gate(self):
        """Engagement >2 batches ago → gate is OPEN, dismiss proceeds."""
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared","detail":"fresh popup","recovery_target":"X"}'
        )
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            await rec.checkpoint(
                goal="x", last_action={"type": "computer_agent", "action": "click",
                                         "params": {}, "goal": ""},
                todo_snapshot=[self._todo_done(batch_idx=2)],   # batch 2
                recent_action_results=None,
                current_batch_idx=5,                              # current 5, threshold 3
            )
        mock_overlay.assert_awaited_once()

    async def test_no_todo_with_recent_success_gates(self):
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared","detail":"modal","recovery_target":"X"}'
        )
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            co = await rec.checkpoint(
                goal="x", last_action={"type": "computer_agent", "action": "click",
                                         "params": {}, "goal": ""},
                todo_snapshot=None,
                recent_action_results=["Clicked 'Some Field' at (100,200)"],
                current_batch_idx=3,
            )
        mock_overlay.assert_not_awaited()
        self.assertEqual(co.action_taken, "none")

    async def test_no_todo_all_failures_does_not_gate(self):
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared","detail":"banner","recovery_target":"X"}'
        )
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            await rec.checkpoint(
                goal="x", last_action={"type": "computer_agent", "action": "click",
                                         "params": {}, "goal": ""},
                todo_snapshot=None,
                recent_action_results=["Failed: timeout", "ABORTED_WRONG_FOCUS expected X"],
                current_batch_idx=3,
            )
        mock_overlay.assert_awaited_once()

    async def test_kill_switch_disables_gate(self):
        """When config flag is False, gate is bypassed even with engagement."""
        _config_stub.DIALOG_ENGAGEMENT_GATE_ENABLED = False
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared","detail":"modal","recovery_target":"X"}'
        )
        try:
            with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
                await rec.checkpoint(
                    goal="x", last_action={"type": "computer_agent", "action": "click",
                                             "params": {}, "goal": ""},
                    todo_snapshot=[self._todo_done(batch_idx=4)],
                    recent_action_results=None,
                    current_batch_idx=5,
                )
            mock_overlay.assert_awaited_once()
        finally:
            _config_stub.DIALOG_ENGAGEMENT_GATE_ENABLED = True

    async def test_error_shown_class_not_gated(self):
        """Only overlay_appeared is destructive — error_shown still dispatches."""
        _stub_screen_llm(
            vision_response='{"class":"error_shown","detail":"email invalid"}'
        )
        with patch.object(rec, "_recover_error", new=AsyncMock(return_value=(True, 1))) as mock_err:
            await rec.checkpoint(
                goal="x", last_action={"type": "computer_agent", "action": "type",
                                         "params": {}, "goal": ""},
                todo_snapshot=[self._todo_done(batch_idx=4)],   # engagement hot
                recent_action_results=None,
                current_batch_idx=5,
            )
        # Despite engagement, error_shown still fires its strategy.
        mock_err.assert_awaited_once()

    async def test_no_change_class_not_gated(self):
        _stub_screen_llm(
            vision_response='{"class":"no_change","detail":"page still loading"}'
        )
        with patch.object(rec, "_recover_no_change", new=AsyncMock(return_value=(True, 1))) as mock_nc:
            await rec.checkpoint(
                goal="x", last_action={"type": "computer_agent", "action": "click",
                                         "params": {}, "goal": ""},
                todo_snapshot=[self._todo_done(batch_idx=4)],
                recent_action_results=None,
                current_batch_idx=5,
            )
        mock_nc.assert_awaited_once()

    async def test_unknown_class_returns_early_unchanged(self):
        """Unknown short-circuits before the gate even runs — guard against
        the gate accidentally affecting the unknown path."""
        _stub_screen_llm(
            vision_response='{"class":"unknown","detail":"captcha appeared"}'
        )
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            co = await rec.checkpoint(
                goal="x", last_action={"type": "computer_agent", "action": "click",
                                         "params": {}, "goal": ""},
                todo_snapshot=[self._todo_done(batch_idx=4)],
                recent_action_results=None,
                current_batch_idx=5,
            )
        mock_overlay.assert_not_awaited()
        self.assertEqual(co.diagnosed_class, "unknown")
        self.assertEqual(co.action_taken, "none")
        self.assertNotIn("[gated:", co.detail)


# ─── End-to-end Truien scenario regression ─────────────────────────────────


class TestTrueinRegressionScenario(unittest.IsolatedAsyncioTestCase):
    """
    Replays the 2026-04-26 live-test failure shape:
      - Three prior batches each marked a Type TODO done (Rule T anchored)
      - Current batch: recovery diagnoses the form-modal as overlay_appeared
        with recovery_target='close button' (the bug target)
      - Expected: gate fires, _recover_overlay NOT called, the form is NOT
        dismissed, outcome is recorded with action_taken='none' and
        '[gated:' in detail.
    """

    async def test_truein_form_modal_not_dismissed(self):
        _stub_screen_llm(
            vision_response='{"class":"overlay_appeared",'
                            '"detail":"Book A Free Demo modal blocking the page",'
                            '"recovery_target":"close button on the dialog"}'
        )
        prior_todos = [
            {"id": 1, "task": "Type 'Test' in First Name", "done": True,
             "kind": "type", "target": "", "field": "First Name", "value": "Test",
             "pending_visual_confirm": False, "confirm_strikes": 0,
             "batch_marked_done": 1, "batch_deferred": -1},
            {"id": 2, "task": "Type 'User' in Last Name", "done": True,
             "kind": "type", "target": "", "field": "Last Name", "value": "User",
             "pending_visual_confirm": False, "confirm_strikes": 0,
             "batch_marked_done": 2, "batch_deferred": -1},
            {"id": 3, "task": "Type 'Test Co' in Company", "done": True,
             "kind": "type", "target": "", "field": "Company", "value": "Test Co",
             "pending_visual_confirm": False, "confirm_strikes": 0,
             "batch_marked_done": 3, "batch_deferred": -1},
        ]
        with patch.object(rec, "_recover_overlay", new=AsyncMock(return_value=(True, 1))) as mock_overlay:
            co = await rec.checkpoint(
                goal="fill the form",
                last_action={"type": "computer_agent", "action": "type",
                              "params": {"text": "Test Co"}, "goal": ""},
                todo_snapshot=prior_todos,
                recent_action_results=['Pressed key: tab', 'Typed: "Test Co"',
                                        'SCREENSHOT_AND_CONTINUE'],
                current_batch_idx=4,
            )
        # The bug: this would call _recover_overlay, click the close X,
        # and dismiss the form. The fix: gate fires, no dispatch.
        mock_overlay.assert_not_awaited()
        self.assertEqual(co.action_taken, "none")
        self.assertFalse(co.recovered)
        self.assertIn("[gated:", co.detail)
        self.assertIn("Book A Free Demo", co.detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
