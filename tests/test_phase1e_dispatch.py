"""
test_phase1e_dispatch.py — Phase 1E: DOM-mode dispatch wiring.

Tests the integration of detect_backend → execute_automation when the
backend is "dom":
  - _execute_dom_task happy path (attach → page → run_dom_task → success)
  - Failure modes that should fall back to vision-loop:
    * CDP attach returns kind="bundled" (race: probe stale)
    * No usable page found
    * perceive_failed / empty_tree from orchestrator
  - Failure modes that should NOT fall back (return summary):
    * max_loops, planner_failed, loop_failure_at_max
  - execute_automation routes "dom" backend correctly
  - Unrelated backends still work (regression guard)

Run: python test_phase1e_dispatch.py
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.router as da
import assistant.automation.browser.dom_orchestrator as bdo
import assistant.automation.browser.handle as bhandle


def _run(coro):
    return asyncio.run(coro)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_handle(kind: str = "latch"):
    """A driver handle, as `get_browser_handle` returns one.

    The context/page fakes that stood here are gone with the tab picker they
    fed. The extension resolves every page verb to the active tab of the
    current window, so there is no list of contexts for the router to sift and
    nothing for a fake to stand in for.
    """
    return bhandle.BrowserHandle(kind=kind, page=MagicMock(), connection=MagicMock())


# ─── _execute_dom_task ───────────────────────────────────────────────────


class TestExecuteDomTask(unittest.IsolatedAsyncioTestCase):
    async def test_a_bundled_handle_falls_back(self):
        # The router chose "dom" because the user's own browser was open at the
        # page. Driving the bundled one instead would run the task against a
        # browser with none of her sessions.
        with patch.object(bhandle, "get_browser_handle",
                          new=AsyncMock(return_value=_make_handle(kind="bundled"))):
            result = await da._execute_dom_task("fill the form")
        self.assertEqual(result, "__FALLBACK__")

    async def test_a_resolve_failure_falls_back(self):
        with patch.object(bhandle, "get_browser_handle",
                          new=AsyncMock(side_effect=RuntimeError("network"))):
            result = await da._execute_dom_task("fill the form")
        self.assertEqual(result, "__FALLBACK__")

    async def test_happy_path_returns_summary(self):
        # `success` is derived from `reason`, not stored -- a result whose tag
        # and verdict disagree cannot be constructed.
        success = bdo.DomTaskResult(
            reason="completed", loops_used=1,
            final_summary="Filled all 7 fields and submitted.",
            history=[],
        )
        # Both orchestrator entry points are patched. A form-shaped goal is
        # routed to `run_dom_form_fill`, not `run_dom_task`, so patching only
        # the latter let the real one run against a MagicMock page -- which
        # failed with an empty tree and returned the sentinel, in a test named
        # "happy path".
        with patch.object(bhandle, "get_browser_handle",
                          new=AsyncMock(return_value=_make_handle())),              patch.object(bdo, "run_dom_form_fill", new=AsyncMock(return_value=success)),              patch.object(bdo, "run_dom_task", new=AsyncMock(return_value=success)):
            result = await da._execute_dom_task("fill the demo form")
        self.assertEqual(result, "Filled all 7 fields and submitted.")

    async def test_orchestrator_crash_falls_back(self):
        with patch.object(bhandle, "get_browser_handle",
                          new=AsyncMock(return_value=_make_handle())),              patch.object(bdo, "run_dom_form_fill",
                          new=AsyncMock(side_effect=RuntimeError("crash"))),              patch.object(bdo, "run_dom_task",
                          new=AsyncMock(side_effect=RuntimeError("crash"))):
            result = await da._execute_dom_task("fill")
        self.assertEqual(result, "__FALLBACK__")

    async def test_a_failed_task_reports_its_own_summary_not_a_sentinel(self):
        """A task that ran and failed is not a reason to escalate a tier.

        `.claude/rules/automation.md` records what a sentinel cost here once:
        "__FALLBACK__" is not a failure report, it is an instruction, and
        returning it for a real failure spends a vision call on a task that
        already knows why it did not work.
        """
        failed = bdo.DomTaskResult(
            reason="max_loops", loops_used=6,
            final_summary="Could not find the submit button.", history=[],
        )
        with patch.object(bhandle, "get_browser_handle",
                          new=AsyncMock(return_value=_make_handle())),              patch.object(bdo, "run_dom_form_fill", new=AsyncMock(return_value=failed)),              patch.object(bdo, "run_dom_task", new=AsyncMock(return_value=failed)):
            result = await da._execute_dom_task("fill the form")
        self.assertNotEqual(result, "__FALLBACK__")
        self.assertIn("submit button", result)


# ─── execute_automation routing ──────────────────────────────────────────


class TestExecuteAutomationRouting(unittest.IsolatedAsyncioTestCase):
    async def test_dom_backend_routes_to_dom_handler(self):
        # Mock detect_backend to force the dom branch
        with patch.object(da, "detect_backend",
                          return_value=("dom", {"reason": "form_intent", "app": "Chrome"})), \
             patch.object(da, "_execute_dom_task",
                          new=AsyncMock(return_value="DOM-mode reply")) as h:
            result = await da.execute_automation("fill the form", llm_func=None)
        self.assertEqual(result, "DOM-mode reply")
        # The router also passes the foreground window title; asserted on the
        # call happening rather than on an exact signature, which is the
        # caller's business and not this test's subject.
        h.assert_awaited_once()

    async def test_dom_fallback_propagates(self):
        # When _execute_dom_task returns __FALLBACK__, execute_automation
        # propagates so the caller routes to vision-loop.
        with patch.object(da, "detect_backend",
                          return_value=("dom", {"reason": "form_intent"})), \
             patch.object(da, "_execute_dom_task",
                          new=AsyncMock(return_value="__FALLBACK__")):
            result = await da.execute_automation("fill the form", llm_func=None)
        self.assertEqual(result, "__FALLBACK__")

    async def test_browser_backend_unaffected(self):
        # Regression: existing "browser" backend still routes to its handler
        with patch.object(da, "detect_backend",
                          return_value=("browser", {"reason": "browser_intent"})), \
             patch.object(da, "_execute_browser_task",
                          new=AsyncMock(return_value="browser reply")) as h:
            result = await da.execute_automation("visit example.com", llm_func=None)
        self.assertEqual(result, "browser reply")
        h.assert_awaited_once()

    async def test_native_backend_unaffected(self):
        with patch.object(da, "detect_backend",
                          return_value=("native", {"reason": "running_app_detected"})), \
             patch.object(da, "_execute_native_task",
                          new=AsyncMock(return_value="native reply")) as h:
            result = await da.execute_automation("open notepad", llm_func=None)
        self.assertEqual(result, "native reply")
        h.assert_awaited_once()

    async def test_unknown_backend_falls_back(self):
        with patch.object(da, "detect_backend",
                          return_value=("unknown", {"reason": "no_match"})):
            result = await da.execute_automation("xyz", llm_func=None)
        self.assertEqual(result, "__FALLBACK__")


if __name__ == "__main__":
    unittest.main(verbosity=2)
