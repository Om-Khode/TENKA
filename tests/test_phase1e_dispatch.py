"""
test_phase1e_dispatch.py — Phase 1E: DOM-mode dispatch wiring.

Tests the integration of detect_backend → execute_automation when the
backend is "dom":
  - _pick_active_page heuristics (skip chrome://, pick first user page)
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
import assistant.automation.browser.cdp as bcdp


def _run(coro):
    return asyncio.run(coro)


# ─── Helpers ─────────────────────────────────────────────────────────────


class _FakePage:
    def __init__(self, url: str = "https://example.com/"):
        self.url = url


class _FakeContext:
    def __init__(self, pages):
        self.pages = pages


class _FakeAttachment:
    def __init__(self, contexts):
        self.contexts = contexts


def _make_handle(kind: str = "cdp", contexts=None):
    return bcdp.BrowserHandle(
        kind=kind,
        browser=MagicMock(name="browser"),
        attachment=_FakeAttachment(contexts or []) if kind == "cdp" else None,
    )


# ─── _pick_active_page ───────────────────────────────────────────────────


class TestPickActivePage(unittest.TestCase):
    def test_returns_none_when_no_attachment(self):
        self.assertIsNone(da._pick_active_page(None))

    def test_returns_none_when_no_contexts(self):
        att = _FakeAttachment(contexts=[])
        self.assertIsNone(da._pick_active_page(att))

    def test_returns_none_when_contexts_have_no_pages(self):
        att = _FakeAttachment(contexts=[_FakeContext([]), _FakeContext([])])
        self.assertIsNone(da._pick_active_page(att))

    def test_picks_first_user_page(self):
        truein = _FakePage("https://truein.com/demo")
        att = _FakeAttachment(contexts=[_FakeContext([truein])])
        self.assertIs(da._pick_active_page(att), truein)

    def test_skips_chrome_internal_pages(self):
        new_tab = _FakePage("chrome://newtab/")
        truein = _FakePage("https://truein.com/")
        att = _FakeAttachment(contexts=[_FakeContext([new_tab, truein])])
        self.assertIs(da._pick_active_page(att), truein)

    def test_skips_extension_popups(self):
        ext = _FakePage("chrome-extension://abc123/popup.html")
        site = _FakePage("https://example.com/")
        att = _FakeAttachment(contexts=[_FakeContext([ext, site])])
        self.assertIs(da._pick_active_page(att), site)

    def test_skips_devtools(self):
        dt = _FakePage("devtools://devtools/inspector.html")
        site = _FakePage("https://example.com/")
        att = _FakeAttachment(contexts=[_FakeContext([dt, site])])
        self.assertIs(da._pick_active_page(att), site)

    def test_falls_back_to_chrome_internal_when_no_user_pages(self):
        # All pages are chrome:// — return one anyway. Better than None
        # because the user might be planning to navigate via the agent.
        new_tab = _FakePage("chrome://newtab/")
        att = _FakeAttachment(contexts=[_FakeContext([new_tab])])
        self.assertIs(da._pick_active_page(att), new_tab)

    def test_url_read_exception_skips_page(self):
        bad = MagicMock()
        type(bad).url = property(lambda _: (_ for _ in ()).throw(RuntimeError("dead")))
        good = _FakePage("https://example.com/")
        att = _FakeAttachment(contexts=[_FakeContext([bad, good])])
        # The bad page raises on url read; we skip it and pick the good one.
        self.assertIs(da._pick_active_page(att), good)

    def test_walks_multiple_contexts(self):
        # Different incognito-style contexts with separate pages
        ctx1 = _FakeContext([_FakePage("chrome://newtab/")])
        ctx2 = _FakeContext([_FakePage("https://example.com/")])
        att = _FakeAttachment(contexts=[ctx1, ctx2])
        page = da._pick_active_page(att)
        self.assertEqual(page.url, "https://example.com/")


# ─── _execute_dom_task ───────────────────────────────────────────────────


class TestExecuteDomTask(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset CDP state between tests so cached attachments don't leak
        bcdp.reset_state_for_test()

    def tearDown(self):
        bcdp.reset_state_for_test()

    async def test_attach_returns_bundled_falls_back(self):
        handle = _make_handle(kind="bundled")
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)):
            result = await da._execute_dom_task("fill the form")
        self.assertEqual(result, "__FALLBACK__")

    async def test_attach_raises_falls_back(self):
        with patch.object(bcdp, "get_or_attach_browser",
                          new=AsyncMock(side_effect=RuntimeError("network"))):
            result = await da._execute_dom_task("fill the form")
        self.assertEqual(result, "__FALLBACK__")

    async def test_no_pages_falls_back(self):
        handle = _make_handle(contexts=[_FakeContext([])])
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)):
            result = await da._execute_dom_task("fill the form")
        self.assertEqual(result, "__FALLBACK__")

    async def test_happy_path_returns_summary(self):
        page = _FakePage("https://truein.com/demo")
        handle = _make_handle(contexts=[_FakeContext([page])])
        success = bdo.DomTaskResult(
            success=True, reason="completed", loops_used=1,
            final_summary="Filled all 7 fields and submitted.",
            history=[],
        )
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)), \
             patch.object(bdo, "run_dom_task", new=AsyncMock(return_value=success)):
            result = await da._execute_dom_task("fill the demo form")
        self.assertEqual(result, "Filled all 7 fields and submitted.")

    async def test_orchestrator_crash_falls_back(self):
        page = _FakePage("https://example.com/")
        handle = _make_handle(contexts=[_FakeContext([page])])
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)), \
             patch.object(bdo, "run_dom_task",
                          new=AsyncMock(side_effect=RuntimeError("crash"))):
            result = await da._execute_dom_task("fill")
        self.assertEqual(result, "__FALLBACK__")

    async def test_perceive_failed_falls_back_to_vision(self):
        page = _FakePage("https://example.com/")
        handle = _make_handle(contexts=[_FakeContext([page])])
        # perceive_failed = vision should retry — fall back
        failed = bdo.DomTaskResult(
            success=False, reason="perceive_failed", loops_used=0,
            final_summary="Could not read the page.",
            history=[],
        )
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)), \
             patch.object(bdo, "run_dom_task", new=AsyncMock(return_value=failed)):
            result = await da._execute_dom_task("fill")
        self.assertEqual(result, "__FALLBACK__")

    async def test_empty_tree_falls_back_to_vision(self):
        page = _FakePage("https://example.com/")
        handle = _make_handle(contexts=[_FakeContext([page])])
        empty = bdo.DomTaskResult(
            success=False, reason="empty_tree", loops_used=2,
            final_summary="No interactive elements found on page.",
            history=[],
        )
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)), \
             patch.object(bdo, "run_dom_task", new=AsyncMock(return_value=empty)):
            result = await da._execute_dom_task("fill")
        self.assertEqual(result, "__FALLBACK__")

    async def test_max_loops_returns_summary_no_fallback(self):
        # max_loops means DOM-mode tried but couldn't finish — vision-loop
        # would just burn the same budget over again. Honest summary
        # to the user instead.
        page = _FakePage("https://example.com/")
        handle = _make_handle(contexts=[_FakeContext([page])])
        maxed = bdo.DomTaskResult(
            success=False, reason="max_loops", loops_used=5,
            final_summary="Could not complete within 5 steps.",
            history=[],
        )
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)), \
             patch.object(bdo, "run_dom_task", new=AsyncMock(return_value=maxed)):
            result = await da._execute_dom_task("fill")
        self.assertEqual(result, "Could not complete within 5 steps.")

    async def test_planner_failed_returns_summary_no_fallback(self):
        page = _FakePage("https://example.com/")
        handle = _make_handle(contexts=[_FakeContext([page])])
        bad_plan = bdo.DomTaskResult(
            success=False, reason="planner_failed", loops_used=2,
            final_summary="Planner produced no usable actions.",
            history=[],
        )
        with patch.object(bcdp, "get_or_attach_browser", new=AsyncMock(return_value=handle)), \
             patch.object(bdo, "run_dom_task", new=AsyncMock(return_value=bad_plan)):
            result = await da._execute_dom_task("fill")
        self.assertEqual(result, "Planner produced no usable actions.")


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
        h.assert_awaited_once_with("fill the form")

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
