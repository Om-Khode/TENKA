"""
test_recovery.py — AR-1a: recovery skeleton + diagnose + orchestrator loop.

Covers:
  - _diagnose JSON parsing (each of 5 classes, including "success")
  - _diagnose markdown-fence stripping
  - _diagnose fail-open paths (screenshot None, llm crash, llm unavailable
    sentinel, garbage JSON, invalid class)
  - attempt_recovery: unknown class → escalate immediately
  - attempt_recovery: success class → false alarm, return succeeded
  - attempt_recovery: same diagnose detail twice → escalate (loop guard)
  - attempt_recovery: max_attempts exhausted → escalate with last observation
  - attempt_recovery: success path (mocked strategy True + post_verify ok)
  - attempt_recovery: dispatch matrix — each class routes to its 1:1 strategy
  - attempt_recovery: post_verify crash bubbles up as escalation, not silent ok
  - checkpoint: success class → recovered=True, no action dispatched

Run: python test_recovery.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.recovery as rec
from assistant.automation.recovery import RecoveryAttempt, RecoveryOutcome
from assistant.automation.verification import VerifyResult
from assistant import config as _config_stub

_config_stub.VERIFY_ENABLED = True
_config_stub.VERIFY_BROWSER_STEPS = True
_config_stub.VERIFY_APP_STEPS = True
_config_stub.VERIFY_VISION_FALLBACK = True
_config_stub.VERIFY_STRICT_TEXT_MATCH = False
_config_stub.VERIFY_MIN_CONFIDENCE = 0.5
_config_stub.VERIFY_MAX_RETRIES = 1


# ─── Stub helpers ─────────────────────────────────────────────────────────────

def _stub_screen_llm(*, screenshot="fake-b64", vision_response='{"class":"overlay_appeared","detail":"calendar visible"}'):
    """Install fake screen + llm modules so recovery.py's in-function imports
    resolve to controllable doubles.

    ``get_vision_response`` now returns ``LLMResult`` objects so bare string
    values are wrapped in ``SimpleNamespace(text=...)`` so callers can do
    ``.text`` on the result.  Callables are wrapped so each invocation returns
    a ``SimpleNamespace`` around the callable's bare-string return value.
    Exceptions are passed through as-is (raised by the mock).
    """
    screen_mod = types.ModuleType("assistant.io.screen")
    screen_mod.capture_screenshot_base64 = MagicMock(return_value=screenshot)

    llm_mod = types.ModuleType("assistant.llm")
    if isinstance(vision_response, Exception):
        llm_mod.get_vision_response = AsyncMock(side_effect=vision_response)
    elif callable(vision_response):
        # Allow callable returning different responses on each call (for
        # multi-attempt loop tests).  Wrap each bare-string return value.
        async def _wrapped_callable(*args, **kwargs):
            raw = vision_response(*args, **kwargs)
            return SimpleNamespace(text=raw)
        llm_mod.get_vision_response = _wrapped_callable
    else:
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text=vision_response)
        )

    sys.modules["assistant.io.screen"] = screen_mod
    sys.modules["assistant.llm"] = llm_mod
    return screen_mod, llm_mod


def _stub_verification(verify_results):
    """Patch assistant.verification.post_verify with a sequence of return values
    (or a single value). Returns the patcher so callers can stop()."""
    import assistant.automation.verification as ver
    if isinstance(verify_results, list):
        mock = MagicMock(side_effect=verify_results)
    elif isinstance(verify_results, Exception):
        mock = MagicMock(side_effect=verify_results)
    else:
        mock = MagicMock(return_value=verify_results)
    return patch.object(ver, "post_verify", mock)


# ─── Diagnose: JSON parsing ───────────────────────────────────────────────────

class TestDiagnoseHappy(unittest.IsolatedAsyncioTestCase):
    async def test_overlay_appeared(self):
        _stub_screen_llm(vision_response='{"class":"overlay_appeared","detail":"calendar opened"}')
        out = await rec._diagnose("set DOB to 25 Apr 2026", "focus changed", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "overlay_appeared")
        self.assertEqual(out["detail"], "calendar opened")

    async def test_error_shown(self):
        _stub_screen_llm(vision_response='{"class":"error_shown","detail":"email format invalid"}')
        out = await rec._diagnose("submit form", "still on form", {"type": "browser", "action": "click"})
        self.assertEqual(out["class"], "error_shown")
        self.assertEqual(out["detail"], "email format invalid")

    async def test_no_change(self):
        _stub_screen_llm(vision_response='{"class":"no_change","detail":"page still loading"}')
        out = await rec._diagnose("click submit", "no visible change", {"type": "browser", "action": "click"})
        self.assertEqual(out["class"], "no_change")

    async def test_unknown(self):
        _stub_screen_llm(vision_response='{"class":"unknown","detail":"captcha appeared"}')
        out = await rec._diagnose("login", "halted", {"type": "browser", "action": "click"})
        self.assertEqual(out["class"], "unknown")
        self.assertIn("captcha", out["detail"])

    async def test_markdown_fence_stripped(self):
        _stub_screen_llm(vision_response='```json\n{"class":"error_shown","detail":"required"}\n```')
        out = await rec._diagnose("submit", "form rejected", {"type": "browser", "action": "click"})
        self.assertEqual(out["class"], "error_shown")
        self.assertEqual(out["detail"], "required")


# ─── Diagnose: fail-open paths ────────────────────────────────────────────────

class TestDiagnoseFailOpen(unittest.IsolatedAsyncioTestCase):
    async def test_screenshot_none(self):
        _stub_screen_llm(screenshot=None)
        out = await rec._diagnose("goal", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "unknown")
        self.assertIn("screenshot", out["detail"])

    async def test_llm_crash(self):
        _stub_screen_llm(vision_response=RuntimeError("network exploded"))
        out = await rec._diagnose("goal", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "unknown")
        self.assertIn("crashed", out["detail"])

    async def test_llm_unavailable_sentinel(self):
        _stub_screen_llm(vision_response="__LLM_UNAVAILABLE__")
        out = await rec._diagnose("goal", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "unknown")
        self.assertIn("no vision response", out["detail"])

    async def test_empty_response(self):
        _stub_screen_llm(vision_response="")
        out = await rec._diagnose("goal", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "unknown")

    async def test_garbage_json(self):
        _stub_screen_llm(vision_response="this is not json at all {")
        out = await rec._diagnose("goal", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "unknown")
        self.assertIn("parse", out["detail"])

    async def test_invalid_class_coerced_to_unknown(self):
        _stub_screen_llm(vision_response='{"class":"datepicker_present","detail":"calendar"}')
        out = await rec._diagnose("goal", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "unknown")
        # detail preserved so caller can report what the model actually saw
        self.assertEqual(out["detail"], "calendar")

    async def test_missing_class_field(self):
        _stub_screen_llm(vision_response='{"detail":"calendar"}')
        out = await rec._diagnose("goal", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "unknown")


# ─── attempt_recovery: escalation paths ───────────────────────────────────────

class TestRecoveryEscalation(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_class_escalates_immediately(self):
        _stub_screen_llm(vision_response='{"class":"unknown","detail":"captcha"}')
        vr_in = VerifyResult.fail("vision said no")
        outcome = await rec.attempt_recovery(
            step={"type": "browser", "action": "click", "params": {}},
            goal="login",
            verify_result=vr_in,
        )
        self.assertFalse(outcome.succeeded)
        self.assertTrue(outcome.escalated)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual(outcome.attempts[0].action_taken, "escalated")
        self.assertIn("captcha", outcome.final_observation)

    async def test_same_detail_twice_escalates(self):
        # Two diagnoses with identical detail → loop guard fires on attempt 2.
        # First diagnose: overlay_appeared (stub strategy returns False, so
        # post_verify will fail). Second diagnose: same detail → escalate.
        responses = iter([
            '{"class":"overlay_appeared","detail":"calendar grid visible"}',
            '{"class":"overlay_appeared","detail":"calendar grid visible"}',
        ])
        _stub_screen_llm(vision_response=lambda *a, **kw: next(responses))

        with _stub_verification(VerifyResult.fail("still on form")):
            outcome = await rec.attempt_recovery(
                step={"type": "browser", "action": "fill", "params": {"text": "2026-04-25"}},
                goal="set DOB to 25 Apr 2026",
                verify_result=VerifyResult.fail("focus changed"),
                max_attempts=3,
            )

        self.assertFalse(outcome.succeeded)
        self.assertTrue(outcome.escalated)
        # 2 attempts: first was bbox_click (failed), second was loop-guard escalate
        self.assertEqual(len(outcome.attempts), 2)
        self.assertEqual(outcome.attempts[0].action_taken, "bbox_click")
        self.assertEqual(outcome.attempts[1].action_taken, "escalated")

    async def test_max_attempts_exhausted_escalates(self):
        # Three diagnoses with DIFFERENT details (avoids loop guard) all
        # routed to a stub strategy → max_attempts hit, escalate.
        responses = iter([
            '{"class":"overlay_appeared","detail":"first overlay"}',
            '{"class":"overlay_appeared","detail":"second overlay"}',
            '{"class":"overlay_appeared","detail":"third overlay"}',
        ])
        _stub_screen_llm(vision_response=lambda *a, **kw: next(responses))

        with _stub_verification(VerifyResult.fail("nope")):
            outcome = await rec.attempt_recovery(
                step={"type": "browser", "action": "click", "params": {}},
                goal="click submit",
                verify_result=VerifyResult.fail("initial"),
                max_attempts=3,
            )

        self.assertFalse(outcome.succeeded)
        self.assertTrue(outcome.escalated)
        self.assertEqual(len(outcome.attempts), 3)
        for a in outcome.attempts:
            self.assertEqual(a.action_taken, "bbox_click")
            self.assertFalse(a.succeeded)
        self.assertEqual(outcome.final_observation, "nope")


# ─── attempt_recovery: success path ───────────────────────────────────────────

class TestRecoverySuccess(unittest.IsolatedAsyncioTestCase):
    async def test_overlay_recovery_success(self):
        _stub_screen_llm(vision_response='{"class":"overlay_appeared","detail":"calendar"}')

        # Patch the overlay strategy to return success, and verify ok.
        with patch.object(rec, "_recover_overlay", AsyncMock(return_value=(True, 1))):
            with _stub_verification(VerifyResult.ok_(observation="DOB filled")):
                outcome = await rec.attempt_recovery(
                    step={"type": "browser", "action": "fill", "params": {}},
                    goal="set DOB",
                    verify_result=VerifyResult.fail("focus drift"),
                )

        self.assertTrue(outcome.succeeded)
        self.assertFalse(outcome.escalated)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual(outcome.attempts[0].action_taken, "bbox_click")
        self.assertTrue(outcome.attempts[0].succeeded)
        self.assertEqual(outcome.final_observation, "DOB filled")

    async def test_recovery_succeeds_on_second_attempt(self):
        # First attempt fails (different detail), second succeeds.
        responses = iter([
            '{"class":"overlay_appeared","detail":"first state"}',
            '{"class":"error_shown","detail":"format invalid"}',
        ])
        _stub_screen_llm(vision_response=lambda *a, **kw: next(responses))

        with patch.object(rec, "_recover_overlay", AsyncMock(return_value=(False, 1))), \
             patch.object(rec, "_recover_error", AsyncMock(return_value=(True, 0))):
            verify_results = [
                VerifyResult.fail("still bad"),       # after attempt 1
                VerifyResult.ok_(observation="done"), # after attempt 2
            ]
            with _stub_verification(verify_results):
                outcome = await rec.attempt_recovery(
                    step={"type": "browser", "action": "fill", "params": {}},
                    goal="submit",
                    verify_result=VerifyResult.fail("initial"),
                    max_attempts=3,
                )

        self.assertTrue(outcome.succeeded)
        self.assertFalse(outcome.escalated)
        self.assertEqual(len(outcome.attempts), 2)
        self.assertEqual(outcome.attempts[0].action_taken, "bbox_click")
        self.assertFalse(outcome.attempts[0].succeeded)
        self.assertEqual(outcome.attempts[1].action_taken, "replan_input")
        self.assertTrue(outcome.attempts[1].succeeded)


# ─── attempt_recovery: dispatch matrix (1:1 strategy↔class) ───────────────────

class TestRecoveryDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_overlay_routes_to_recover_overlay(self):
        _stub_screen_llm(vision_response='{"class":"overlay_appeared","detail":"x"}')
        with patch.object(rec, "_recover_overlay", AsyncMock(return_value=(False, 0))) as mo, \
             patch.object(rec, "_recover_error", AsyncMock(return_value=(False, 0))) as me, \
             patch.object(rec, "_recover_no_change", AsyncMock(return_value=(False, 0))) as mn:
            with _stub_verification(VerifyResult.fail("no")):
                await rec.attempt_recovery(
                    step={"type": "browser", "action": "click", "params": {}},
                    goal="g", verify_result=VerifyResult.fail("i"), max_attempts=1,
                )
        mo.assert_awaited_once()
        me.assert_not_awaited()
        mn.assert_not_awaited()

    async def test_error_routes_to_recover_error(self):
        _stub_screen_llm(vision_response='{"class":"error_shown","detail":"x"}')
        with patch.object(rec, "_recover_overlay", AsyncMock(return_value=(False, 0))) as mo, \
             patch.object(rec, "_recover_error", AsyncMock(return_value=(False, 0))) as me, \
             patch.object(rec, "_recover_no_change", AsyncMock(return_value=(False, 0))) as mn:
            with _stub_verification(VerifyResult.fail("no")):
                await rec.attempt_recovery(
                    step={"type": "browser", "action": "fill", "params": {}},
                    goal="g", verify_result=VerifyResult.fail("i"), max_attempts=1,
                )
        me.assert_awaited_once()
        mo.assert_not_awaited()
        mn.assert_not_awaited()

    async def test_no_change_routes_to_recover_no_change(self):
        _stub_screen_llm(vision_response='{"class":"no_change","detail":"x"}')
        with patch.object(rec, "_recover_overlay", AsyncMock(return_value=(False, 0))) as mo, \
             patch.object(rec, "_recover_error", AsyncMock(return_value=(False, 0))) as me, \
             patch.object(rec, "_recover_no_change", AsyncMock(return_value=(False, 0))) as mn:
            with _stub_verification(VerifyResult.fail("no")):
                await rec.attempt_recovery(
                    step={"type": "browser", "action": "click", "params": {}},
                    goal="g", verify_result=VerifyResult.fail("i"), max_attempts=1,
                )
        mn.assert_awaited_once()
        mo.assert_not_awaited()
        me.assert_not_awaited()


# ─── attempt_recovery: infra failure during re-verify ─────────────────────────

class TestRecoveryReVerifyCrash(unittest.IsolatedAsyncioTestCase):
    async def test_post_verify_crash_escalates(self):
        _stub_screen_llm(vision_response='{"class":"overlay_appeared","detail":"x"}')
        with patch.object(rec, "_recover_overlay", AsyncMock(return_value=(True, 1))), \
             _stub_verification(RuntimeError("verifier blew up")):
            outcome = await rec.attempt_recovery(
                step={"type": "browser", "action": "click", "params": {}},
                goal="g",
                verify_result=VerifyResult.fail("i"),
            )
        self.assertFalse(outcome.succeeded)
        self.assertTrue(outcome.escalated)
        self.assertIn("crashed", outcome.final_observation)


# ─── Result types ─────────────────────────────────────────────────────────────

class TestResultTypes(unittest.TestCase):
    def test_recovery_attempt_dataclass(self):
        a = RecoveryAttempt(
            diagnose_class="overlay_appeared",
            detail="d",
            action_taken="bbox_click",
            succeeded=True,
            cost_calls=2,
        )
        self.assertEqual(a.diagnose_class, "overlay_appeared")
        self.assertTrue(a.succeeded)

    def test_recovery_outcome_defaults(self):
        o = RecoveryOutcome(succeeded=False)
        self.assertEqual(o.attempts, [])
        self.assertEqual(o.final_observation, "")
        self.assertFalse(o.escalated)


# ─── _overlay_goal_text helper (AR-1b) ────────────────────────────────────────

class TestOverlayGoalText(unittest.TestCase):
    def test_prefers_explicit_goal(self):
        out = rec._overlay_goal_text(
            "set DOB to 25 Apr 2026",
            {"action": "fill", "params": {"value": "ignored"}},
        )
        self.assertEqual(out, "set DOB to 25 Apr 2026")

    def test_strips_whitespace(self):
        out = rec._overlay_goal_text("   submit form  ", {})
        self.assertEqual(out, "submit form")

    def test_synthesizes_from_value(self):
        out = rec._overlay_goal_text("", {"action": "fill", "params": {"value": "2026-04-25"}})
        self.assertEqual(out, "fill: 2026-04-25")

    def test_synthesizes_from_selector_when_no_value(self):
        out = rec._overlay_goal_text("", {"action": "click", "params": {"selector": "Submit"}})
        self.assertEqual(out, "click: Submit")

    def test_value_preferred_over_selector(self):
        out = rec._overlay_goal_text(
            "",
            {"action": "fill", "params": {"selector": "#dob", "value": "2026-04-25"}},
        )
        self.assertEqual(out, "fill: 2026-04-25")

    def test_falls_back_to_action_when_params_empty(self):
        out = rec._overlay_goal_text("", {"action": "click", "params": {}})
        self.assertEqual(out, "click")

    def test_blank_goal_falls_through_to_synthesis(self):
        out = rec._overlay_goal_text("   ", {"action": "fill", "params": {"value": "x"}})
        self.assertEqual(out, "fill: x")


# ─── _recover_overlay (AR-1b) ─────────────────────────────────────────────────

class TestRecoverOverlay(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Stub pyautogui so click() doesn't move the real mouse during tests.
        self._pyauto = types.ModuleType("pyautogui")
        self._pyauto.click = MagicMock()
        sys.modules["pyautogui"] = self._pyauto

    async def test_happy_path_clicks_screen_coords(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(512, 384))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        ok, calls = await rec._recover_overlay(
            "set DOB to 25 Apr 2026", page=None, active_window=None,
            step={"action": "fill", "params": {"value": "2026-04-25"}},
        )
        self.assertTrue(ok)
        self.assertEqual(calls, 1)
        llm_mod.locate_element_bbox.assert_called_once()
        # Goal text: explicit goal preferred, not synthesized
        args, _ = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "set DOB to 25 Apr 2026")
        self._pyauto.click.assert_called_once_with(512, 384)

    async def test_synthesizes_goal_when_explicit_empty(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(100, 200))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        ok, calls = await rec._recover_overlay(
            "", page=None, active_window=None,
            step={"action": "fill", "params": {"value": "test@example.com"}},
        )
        self.assertTrue(ok)
        args, _ = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "fill: test@example.com")

    async def test_bbox_returns_none(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=None)
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        ok, calls = await rec._recover_overlay(
            "find non-existent thing", page=None, active_window=None,
            step={"action": "click", "params": {}},
        )
        self.assertFalse(ok)
        self.assertEqual(calls, 1)
        self._pyauto.click.assert_not_called()

    async def test_screenshot_none(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value=None)
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock()
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        ok, calls = await rec._recover_overlay(
            "g", page=None, active_window=None, step={"action": "click", "params": {}},
        )
        self.assertFalse(ok)
        self.assertEqual(calls, 0)
        llm_mod.locate_element_bbox.assert_not_called()

    async def test_bbox_locate_crash_fails_open(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(side_effect=RuntimeError("api down"))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        ok, calls = await rec._recover_overlay(
            "g", page=None, active_window=None, step={"action": "click", "params": {}},
        )
        self.assertFalse(ok)
        self.assertEqual(calls, 1)
        self._pyauto.click.assert_not_called()

    async def test_pyautogui_click_crash_fails_open(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(50, 50))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod
        self._pyauto.click = MagicMock(side_effect=OSError("display locked"))

        ok, calls = await rec._recover_overlay(
            "g", page=None, active_window=None, step={"action": "click", "params": {}},
        )
        self.assertFalse(ok)
        self.assertEqual(calls, 1)

    async def test_browser_path_calls_bring_to_front(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(10, 20))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        page = MagicMock()
        page.bring_to_front = AsyncMock()

        ok, _ = await rec._recover_overlay(
            "g", page=page, active_window=None, step={"action": "click", "params": {}},
        )
        self.assertTrue(ok)
        page.bring_to_front.assert_awaited_once()
        self._pyauto.click.assert_called_once_with(10, 20)

    async def test_bring_to_front_failure_does_not_block_click(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(10, 20))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        page = MagicMock()
        page.bring_to_front = AsyncMock(side_effect=RuntimeError("page closed"))

        ok, _ = await rec._recover_overlay(
            "g", page=page, active_window=None, step={"action": "click", "params": {}},
        )
        # bring_to_front failure is best-effort; click still fires.
        self.assertTrue(ok)
        self._pyauto.click.assert_called_once()


# ─── End-to-end via attempt_recovery: real overlay strategy + diagnose ────────

class TestAttemptRecoveryWithRealOverlay(unittest.IsolatedAsyncioTestCase):
    """Full loop using the real (non-stubbed) _recover_overlay strategy."""

    def setUp(self):
        self._pyauto = types.ModuleType("pyautogui")
        self._pyauto.click = MagicMock()
        sys.modules["pyautogui"] = self._pyauto

    async def test_overlay_recovers_datepicker_scenario(self):
        # Diagnose says overlay_appeared, bbox finds the calendar cell,
        # post-verify confirms the date is now filled.
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text='{"class":"overlay_appeared","detail":"calendar grid is open"}')
        )
        llm_mod.locate_element_bbox = MagicMock(return_value=(640, 400))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        with _stub_verification(VerifyResult.ok_(observation="DOB now reads 2026-04-25")):
            outcome = await rec.attempt_recovery(
                step={"type": "browser", "action": "fill",
                      "params": {"selector": "#dob", "value": "2026-04-25"},
                      "goal": "set DOB to 25 Apr 2026"},
                goal="set DOB to 25 Apr 2026",
                verify_result=VerifyResult.fail("focus drifted to calendar"),
            )

        self.assertTrue(outcome.succeeded)
        self.assertFalse(outcome.escalated)
        self.assertEqual(outcome.attempts[-1].diagnose_class, "overlay_appeared")
        self.assertEqual(outcome.attempts[-1].action_taken, "bbox_click")
        self._pyauto.click.assert_called_once_with(640, 400)
        # Goal threaded through: bbox locator received the planner's goal text
        args, _ = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "set DOB to 25 Apr 2026")


# ─── _synthesize_step_from_ca_action ───────────────────────────────────

class TestSynthesizeStepFromCAAction(unittest.TestCase):
    def test_keyboard_type(self):
        ca = {"action": "keyboard_type", "params": {"text": "Reading"}}
        out = rec._synthesize_step_from_ca_action(ca, "fill the form with testing values")
        self.assertEqual(out["type"], "computer_agent")
        self.assertEqual(out["action"], "type")
        self.assertEqual(out["params"]["text"], "Reading")
        self.assertEqual(out["goal"], "fill the form with testing values")

    def test_keyboard_press_with_top_level_key(self):
        # computer_agent puts 'key' at top level for keyboard_press
        ca = {"action": "keyboard_press", "key": "tab"}
        out = rec._synthesize_step_from_ca_action(ca, "g")
        self.assertEqual(out["action"], "press")
        self.assertEqual(out["params"]["key"], "tab")

    def test_vision_guided_click_normalizes(self):
        ca = {"action": "vision_guided_click", "params": {"x": 100, "y": 200, "text": "Submit"}}
        out = rec._synthesize_step_from_ca_action(ca, "g")
        self.assertEqual(out["action"], "click")
        self.assertEqual(out["params"]["x"], 100)

    def test_unknown_action_passes_through(self):
        ca = {"action": "scroll", "params": {"clicks": 3}}
        out = rec._synthesize_step_from_ca_action(ca, "g")
        self.assertEqual(out["action"], "scroll")
        self.assertEqual(out["params"]["clicks"], 3)

    def test_type_field_used_when_action_missing(self):
        # Some emitters use 'type' instead of 'action'
        ca = {"type": "mouse_click", "params": {"x": 10, "y": 20}}
        out = rec._synthesize_step_from_ca_action(ca, "g")
        self.assertEqual(out["action"], "click")

    def test_empty_action_defaults(self):
        out = rec._synthesize_step_from_ca_action({}, "")
        self.assertEqual(out["action"], "?")
        self.assertEqual(out["goal"], "")


# ─── checkpoint() ──────────────────────────────────────────────────────

class TestCheckpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._pyauto = types.ModuleType("pyautogui")
        self._pyauto.click = MagicMock()
        sys.modules["pyautogui"] = self._pyauto

    async def test_unknown_returns_no_recovery(self):
        _stub_screen_llm(vision_response='{"class":"unknown","detail":"all clear"}')
        out = await rec.checkpoint(
            goal="fill form",
            last_action={"type": "computer_agent", "action": "type", "params": {"text": "John"}, "goal": "fill form"},
        )
        self.assertEqual(out.diagnosed_class, "unknown")
        self.assertFalse(out.recovered)
        self.assertEqual(out.action_taken, "none")
        self.assertEqual(out.cost_calls, 1)

    async def test_overlay_recovers(self):
        # diagnose says overlay, bbox locates, click fires.
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text='{"class":"overlay_appeared","detail":"autocomplete dropdown over Hobbies"}')
        )
        llm_mod.locate_element_bbox = MagicMock(return_value=(700, 628))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        out = await rec.checkpoint(
            goal="check the Reading checkbox under Hobbies",
            last_action={
                "type": "computer_agent", "action": "type",
                "params": {"text": "Reading"},
                "goal": "check the Reading checkbox under Hobbies",
            },
        )
        self.assertEqual(out.diagnosed_class, "overlay_appeared")
        self.assertEqual(out.action_taken, "bbox_click")
        self.assertTrue(out.recovered)
        # 1 diagnose + 1 bbox = 2 calls
        self.assertEqual(out.cost_calls, 2)
        self._pyauto.click.assert_called_once_with(700, 628)
        # The bbox locator was given the GOAL, not the action's typed text
        args, _ = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "check the Reading checkbox under Hobbies")

    async def test_overlay_diagnosed_but_strategy_fails(self):
        # diagnose ok, bbox returns None → recovered=False but class still reported
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img-b64")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text='{"class":"overlay_appeared","detail":"some overlay"}')
        )
        llm_mod.locate_element_bbox = MagicMock(return_value=None)
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        out = await rec.checkpoint(
            goal="g", last_action={"type": "computer_agent", "action": "type", "params": {}},
        )
        self.assertEqual(out.diagnosed_class, "overlay_appeared")
        self.assertEqual(out.action_taken, "bbox_click")
        self.assertFalse(out.recovered)
        self._pyauto.click.assert_not_called()

    async def test_error_shown_dispatches_to_recover_error(self):
        _stub_screen_llm(vision_response='{"class":"error_shown","detail":"email format invalid"}')
        out = await rec.checkpoint(
            goal="submit", last_action={"type": "computer_agent", "action": "type", "params": {"text": "bad"}},
        )
        # AR-1c stub returns (False, 0) — recovered=False but class+action reported
        self.assertEqual(out.diagnosed_class, "error_shown")
        self.assertEqual(out.action_taken, "replan_input")
        self.assertFalse(out.recovered)

    async def test_no_change_dispatches_to_recover_no_change(self):
        _stub_screen_llm(vision_response='{"class":"no_change","detail":"page still loading"}')
        out = await rec.checkpoint(
            goal="click submit", last_action={"type": "computer_agent", "action": "click", "params": {}},
        )
        self.assertEqual(out.diagnosed_class, "no_change")
        self.assertEqual(out.action_taken, "retry")
        self.assertFalse(out.recovered)

    async def test_diagnose_screenshot_none_returns_unknown(self):
        _stub_screen_llm(screenshot=None)
        out = await rec.checkpoint(
            goal="g", last_action={"type": "computer_agent", "action": "type", "params": {}},
        )
        self.assertEqual(out.diagnosed_class, "unknown")
        self.assertFalse(out.recovered)
        self.assertEqual(out.action_taken, "none")

    async def test_diagnose_llm_crash_returns_unknown(self):
        _stub_screen_llm(vision_response=RuntimeError("api down"))
        out = await rec.checkpoint(
            goal="g", last_action={"type": "computer_agent", "action": "type", "params": {}},
        )
        self.assertEqual(out.diagnosed_class, "unknown")
        self.assertFalse(out.recovered)


# ─── recovery_target threading ─────────────────────────────

class TestRecoveryTargetParsing(unittest.IsolatedAsyncioTestCase):
    """diagnose now returns recovery_target — verify it's parsed and defaulted."""

    async def test_diagnose_parses_recovery_target(self):
        _stub_screen_llm(vision_response='{"class":"overlay_appeared","detail":"calendar open","recovery_target":"day 25 in April calendar"}')
        out = await rec._diagnose("set DOB", "focus drift", {"type": "browser", "action": "fill"})
        self.assertEqual(out["class"], "overlay_appeared")
        self.assertEqual(out["recovery_target"], "day 25 in April calendar")

    async def test_diagnose_recovery_target_defaults_empty(self):
        # Older diagnose responses without recovery_target still parse fine.
        _stub_screen_llm(vision_response='{"class":"overlay_appeared","detail":"x"}')
        out = await rec._diagnose("g", "obs", {"type": "browser", "action": "fill"})
        self.assertEqual(out["recovery_target"], "")

    async def test_diagnose_recovery_target_in_unknown_branch(self):
        # Even when class is unknown, recovery_target slot exists.
        _stub_screen_llm(vision_response='{"class":"unknown","detail":"all clear"}')
        out = await rec._diagnose("g", "(none)", {"type": "browser", "action": "fill"})
        self.assertEqual(out["recovery_target"], "")

    async def test_diagnose_failopen_paths_include_recovery_target(self):
        _stub_screen_llm(screenshot=None)
        out = await rec._diagnose("g", "obs", {"type": "browser", "action": "fill"})
        self.assertIn("recovery_target", out)
        self.assertEqual(out["recovery_target"], "")


class TestOverlayGoalTextPriority(unittest.TestCase):
    """target_hint is the new highest-priority source for bbox text."""

    def test_target_hint_beats_goal(self):
        out = rec._overlay_goal_text(
            "fill this form with testing values",
            {"action": "fill", "params": {"value": "x"}},
            target_hint="Maths option in suggestion list",
        )
        self.assertEqual(out, "Maths option in suggestion list")

    def test_falls_back_to_goal_when_hint_empty(self):
        out = rec._overlay_goal_text(
            "click Submit", {"action": "click", "params": {}}, target_hint="",
        )
        self.assertEqual(out, "click Submit")

    def test_falls_back_to_goal_when_hint_whitespace(self):
        out = rec._overlay_goal_text(
            "click Submit", {"action": "click", "params": {}}, target_hint="   ",
        )
        self.assertEqual(out, "click Submit")

    def test_clips_long_paragraph_goal_to_max_chars(self):
        # The exact bug from 2026-04-26 01:12:52 — multi-sentence planner thinking
        # passed as goal would otherwise reach the bbox locator verbatim.
        long = (
            "The form is partially filled. I need to fill in the 'Mobile Number', "
            "'Subjects', 'Hobbies', 'Picture', and 'Current Address' fields. Then "
            "I need to submit the form."
        )
        out = rec._overlay_goal_text(long, {"action": "click", "params": {}}, target_hint="")
        self.assertLessEqual(len(out), rec._MAX_BBOX_TARGET_CHARS)
        self.assertTrue(long.startswith(out))

    def test_clips_long_target_hint_to_max_chars(self):
        long = "x" * 500
        out = rec._overlay_goal_text("g", {"action": "click"}, target_hint=long)
        self.assertEqual(len(out), rec._MAX_BBOX_TARGET_CHARS)


class TestRecoverOverlayUsesTargetHint(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._pyauto = types.ModuleType("pyautogui")
        self._pyauto.click = MagicMock()
        sys.modules["pyautogui"] = self._pyauto

    async def test_target_hint_passed_to_bbox(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(100, 200))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        ok, _ = await rec._recover_overlay(
            goal="fill this form with testing values",  # would have been bad input
            page=None, active_window=None,
            step={"action": "type", "params": {"text": "Maths"}},
            target_hint="Maths in suggestion list",
        )
        self.assertTrue(ok)
        args, _kw = llm_mod.locate_element_bbox.call_args
        # target_hint wins over goal — this is the architectural fix.
        self.assertEqual(args[0], "Maths in suggestion list")

    async def test_falls_back_to_goal_when_no_hint(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(50, 50))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        ok, _ = await rec._recover_overlay(
            goal="set DOB to 25 Apr 2026", page=None, active_window=None,
            step={"action": "fill", "params": {"value": "2026-04-25"}},
            target_hint="",
        )
        self.assertTrue(ok)
        args, _kw = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "set DOB to 25 Apr 2026")

    async def test_long_paragraph_goal_does_not_reach_bbox(self):
        # Direct regression test for the 2026-04-26 01:12:52 bug.
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.locate_element_bbox = MagicMock(return_value=(0, 0))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        long_goal = (
            "The form is partially filled. I need to fill in the 'Mobile Number', "
            "'Subjects', 'Hobbies', 'Picture', and 'Current Address' fields. Then "
            "I need to submit the form. I will start by tabbing to the Mobile "
            "Number field and typing a value."
        )
        await rec._recover_overlay(
            goal=long_goal, page=None, active_window=None,
            step={"action": "type", "params": {"text": "Maths"}},
            target_hint="",
        )
        args, _kw = llm_mod.locate_element_bbox.call_args
        # Whatever reaches bbox must be clipped to the safe ceiling.
        self.assertLessEqual(len(args[0]), rec._MAX_BBOX_TARGET_CHARS)


class TestCheckpointThreadsRecoveryTarget(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._pyauto = types.ModuleType("pyautogui")
        self._pyauto.click = MagicMock()
        sys.modules["pyautogui"] = self._pyauto

    async def test_checkpoint_passes_recovery_target_to_overlay(self):
        # Full chain: diagnose returns recovery_target, checkpoint dispatches,
        # _recover_overlay receives it via target_hint, bbox locator gets it.
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text='{"class":"overlay_appeared","detail":"autocomplete dropdown for Maths","recovery_target":"Maths in autocomplete list"}')
        )
        llm_mod.locate_element_bbox = MagicMock(return_value=(700, 600))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        # The goal here is a long paragraph (the actual bug condition).
        long_goal = (
            "The form is partially filled. I need to fill in 'Mobile Number', "
            "'Subjects', 'Hobbies', 'Picture', and 'Current Address'. I will "
            "start by typing in Subjects."
        )

        out = await rec.checkpoint(
            goal=long_goal,
            last_action={"type": "computer_agent", "action": "type",
                         "params": {"text": "Maths"}, "goal": long_goal},
        )
        self.assertEqual(out.diagnosed_class, "overlay_appeared")
        self.assertTrue(out.recovered)
        # Critical: the bbox locator received the SHORT recovery_target,
        # NOT the long goal paragraph.
        args, _kw = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "Maths in autocomplete list")
        self._pyauto.click.assert_called_once_with(700, 600)

    async def test_checkpoint_falls_back_when_diagnose_omits_target(self):
        # Older / less-compliant diagnose responses without recovery_target
        # still recover via goal fallback (clipped to safe length).
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text='{"class":"overlay_appeared","detail":"calendar"}')  # no recovery_target
        )
        llm_mod.locate_element_bbox = MagicMock(return_value=(100, 200))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        out = await rec.checkpoint(
            goal="click April",  # short, safe to use as fallback
            last_action={"type": "computer_agent", "action": "click", "params": {}},
        )
        self.assertTrue(out.recovered)
        args, _kw = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "click April")


class TestAttemptRecoveryThreadsRecoveryTarget(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._pyauto = types.ModuleType("pyautogui")
        self._pyauto.click = MagicMock()
        sys.modules["pyautogui"] = self._pyauto

    async def test_attempt_recovery_uses_recovery_target_for_bbox(self):
        screen_mod = types.ModuleType("assistant.io.screen")
        screen_mod.capture_screenshot_base64 = MagicMock(return_value="img")
        llm_mod = types.ModuleType("assistant.llm")
        llm_mod.get_vision_response = AsyncMock(
            return_value=SimpleNamespace(text='{"class":"overlay_appeared","detail":"x","recovery_target":"phone field with country code"}')
        )
        llm_mod.locate_element_bbox = MagicMock(return_value=(40, 50))
        sys.modules["assistant.io.screen"] = screen_mod
        sys.modules["assistant.llm"] = llm_mod

        with _stub_verification(VerifyResult.ok_(observation="done")):
            out = await rec.attempt_recovery(
                step={"type": "browser", "action": "fill",
                      "params": {"selector": "#phone"}, "goal": "fill phone"},
                goal="fill phone",
                verify_result=VerifyResult.fail("phone needs country code"),
            )
        self.assertTrue(out.succeeded)
        args, _kw = llm_mod.locate_element_bbox.call_args
        self.assertEqual(args[0], "phone field with country code")


# ─── "success" class: false alarm recovery ──────────────────────────────────

class TestDiagnoseSuccessClass(unittest.IsolatedAsyncioTestCase):
    async def test_diagnose_parses_success(self):
        _stub_screen_llm(vision_response='{"class":"success","detail":"page loaded correctly"}')
        out = await rec._diagnose("navigate to site", "URL mismatch", {"type": "browser", "action": "navigate"})
        self.assertEqual(out["class"], "success")
        self.assertEqual(out["detail"], "page loaded correctly")


class TestRecoverySuccessClass(unittest.IsolatedAsyncioTestCase):
    async def test_success_class_returns_succeeded_immediately(self):
        _stub_screen_llm(vision_response='{"class":"success","detail":"page shows movie details"}')
        vr_in = VerifyResult.fail("URL slug mismatch")
        outcome = await rec.attempt_recovery(
            step={"type": "browser", "action": "navigate", "params": {}},
            goal="open movie page",
            verify_result=vr_in,
        )
        self.assertTrue(outcome.succeeded)
        self.assertFalse(outcome.escalated)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual(outcome.attempts[0].diagnose_class, "success")
        self.assertEqual(outcome.attempts[0].action_taken, "false_alarm")
        self.assertTrue(outcome.attempts[0].succeeded)

    async def test_success_class_no_strategy_dispatched(self):
        _stub_screen_llm(vision_response='{"class":"success","detail":"navigated ok"}')
        with patch.object(rec, "_recover_overlay", AsyncMock()) as mo, \
             patch.object(rec, "_recover_error", AsyncMock()) as me, \
             patch.object(rec, "_recover_no_change", AsyncMock()) as mn:
            outcome = await rec.attempt_recovery(
                step={"type": "browser", "action": "navigate", "params": {}},
                goal="go to site",
                verify_result=VerifyResult.fail("url redirect"),
            )
            mo.assert_not_awaited()
            me.assert_not_awaited()
            mn.assert_not_awaited()
        self.assertTrue(outcome.succeeded)


class TestCheckpointSuccessClass(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_success_returns_recovered_true(self):
        _stub_screen_llm(vision_response='{"class":"success","detail":"field filled correctly"}')
        out = await rec.checkpoint(
            goal="fill email",
            last_action={"type": "computer_agent", "action": "type",
                         "params": {"text": "test@example.com"}, "goal": "fill email"},
        )
        self.assertEqual(out.diagnosed_class, "success")
        self.assertTrue(out.recovered)
        self.assertEqual(out.action_taken, "none")
        self.assertEqual(out.cost_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
