"""
test_da_tab_selection.py — Phase 2E follow-up: CDP tab selection bug fix.

Covers desktop_automation._pick_active_page() and _strip_browser_window_suffix():

Bug context:
  When the user has multiple Chrome tabs open (e.g. Truein + leftover demoqa),
  the OS foreground window's active tab can disagree with Playwright's CDP
  page-list MRU ordering. The old picker walked the list and returned the
  first non-internal page, often picking the wrong tab.

  Fix: when the foreground window title is supplied, strip the browser-name
  suffix and match the remainder against each candidate page's <title>.

Run: python test_da_tab_selection.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.router as da


def _run(coro):
    return asyncio.run(coro)


def _make_page(url: str, title: str = "") -> MagicMock:
    """Stub a Playwright Page with .url (sync property) and .title() (async)."""
    p = MagicMock()
    p.url = url
    p.title = AsyncMock(return_value=title)
    return p


def _make_attachment(*pages_per_context) -> MagicMock:
    """Stub a CDP attachment with one or more contexts, each holding pages."""
    contexts = []
    for pages in pages_per_context:
        ctx = MagicMock()
        ctx.pages = list(pages)
        contexts.append(ctx)
    att = MagicMock()
    att.contexts = contexts
    return att


# ─── _strip_browser_window_suffix ────────────────────────────────────────


class TestStripBrowserWindowSuffix(unittest.TestCase):
    def test_chrome_suffix_stripped(self):
        title = "Truein: AI Based Time Tracking - Google Chrome"
        self.assertEqual(
            da._strip_browser_window_suffix(title),
            "Truein: AI Based Time Tracking",
        )

    def test_brave_suffix_stripped(self):
        self.assertEqual(
            da._strip_browser_window_suffix("DemoQA - Brave"),
            "DemoQA",
        )

    def test_brave_browser_suffix_stripped(self):
        self.assertEqual(
            da._strip_browser_window_suffix("DemoQA - Brave Browser"),
            "DemoQA",
        )

    def test_edge_suffix_stripped(self):
        self.assertEqual(
            da._strip_browser_window_suffix("Page - Microsoft Edge"),
            "Page",
        )

    def test_em_dash_separator(self):
        self.assertEqual(
            da._strip_browser_window_suffix("Stripe Checkout — Google Chrome"),
            "Stripe Checkout",
        )

    def test_en_dash_separator(self):
        self.assertEqual(
            da._strip_browser_window_suffix("Stripe – Google Chrome"),
            "Stripe",
        )

    def test_chromium_suffix(self):
        self.assertEqual(
            da._strip_browser_window_suffix("My Page - Chromium"),
            "My Page",
        )

    def test_empty_input(self):
        self.assertEqual(da._strip_browser_window_suffix(""), "")
        self.assertEqual(da._strip_browser_window_suffix(None), "")

    def test_unknown_browser_suffix_returns_as_is(self):
        # Unknown browser? Don't strip — caller's substring match still works.
        self.assertEqual(
            da._strip_browser_window_suffix("Page - SuperBrowser"),
            "Page - SuperBrowser",
        )


# ─── _pick_active_page ───────────────────────────────────────────────────


class TestPickActivePage(unittest.TestCase):
    def test_none_attachment(self):
        self.assertIsNone(_run(da._pick_active_page(None)))

    def test_attachment_with_no_contexts(self):
        att = MagicMock()
        att.contexts = []
        self.assertIsNone(_run(da._pick_active_page(att)))

    def test_no_hint_picks_first_non_internal(self):
        # No hint → MRU walk: first non-internal page wins (backward compat).
        chrome_internal = _make_page("chrome://newtab/", "New Tab")
        truein = _make_page("https://truein.com/", "Truein")
        demoqa = _make_page("https://demoqa.com/", "DemoQA")
        att = _make_attachment([chrome_internal, demoqa, truein])
        page = _run(da._pick_active_page(att))
        self.assertIs(page, demoqa)  # first non-internal in list order

    def test_chrome_internals_skipped(self):
        # chrome://, chrome-extension://, devtools://, edge://, brave://, about:
        bad_pages = [
            _make_page("chrome://settings/", "Settings"),
            _make_page("chrome-extension://abc/popup.html", "Ext"),
            _make_page("devtools://devtools/inspector.html", "DevTools"),
            _make_page("edge://newtab/", "Edge New"),
            _make_page("brave://welcome/", "Brave"),
            _make_page("about:blank", ""),
        ]
        good = _make_page("https://truein.com/", "Truein")
        att = _make_attachment(bad_pages + [good])
        page = _run(da._pick_active_page(att))
        self.assertIs(page, good)

    def test_only_internal_pages_falls_through_to_first(self):
        # Goal "navigate to X" wants any page back. Fallback returns first.
        newtab = _make_page("chrome://newtab/", "New Tab")
        att = _make_attachment([newtab])
        page = _run(da._pick_active_page(att))
        self.assertIs(page, newtab)

    def test_title_match_picks_correct_tab(self):
        # The bug case: foreground=Truein, MRU order says demoqa first.
        truein = _make_page(
            "https://truein.com/",
            "Truein: AI Based Time Tracking Software for Multi-Site Workforce",
        )
        demoqa = _make_page(
            "https://demoqa.com/automation-practice-form",
            "ToolsQA",
        )
        att = _make_attachment([demoqa, truein])  # demoqa first in MRU
        window_title = (
            "Truein: AI Based Time Tracking Software for Multi-Site Workforce "
            "- Google Chrome"
        )
        page = _run(da._pick_active_page(att, prefer_window_title=window_title))
        self.assertIs(page, truein)

    def test_title_match_substring_in_window(self):
        # Chrome window title may contain the page title as a prefix even
        # when page <title> is shorter (rare but possible).
        truein = _make_page("https://truein.com/", "Truein")
        demoqa = _make_page("https://demoqa.com/", "DemoQA")
        att = _make_attachment([demoqa, truein])
        window_title = "Truein: AI Based Time Tracking - Google Chrome"
        page = _run(da._pick_active_page(att, prefer_window_title=window_title))
        self.assertIs(page, truein)

    def test_title_match_window_substring_of_page_title(self):
        # The reverse: Chrome truncates long page titles in its window
        # chrome. Window says "Truein: AI Based..." and page <title> is the
        # full string.
        full_title = "Truein: AI Based Time Tracking Software for Multi-Site Workforce"
        truein = _make_page("https://truein.com/", full_title)
        demoqa = _make_page("https://demoqa.com/", "DemoQA")
        att = _make_attachment([demoqa, truein])
        window_title = "Truein: AI Based Time Tracking - Google Chrome"  # truncated
        page = _run(da._pick_active_page(att, prefer_window_title=window_title))
        self.assertIs(page, truein)

    def test_no_title_match_falls_through_to_mru(self):
        # Hint provided but no candidate matches → fall through to first.
        truein = _make_page("https://truein.com/", "Truein")
        demoqa = _make_page("https://demoqa.com/", "DemoQA")
        att = _make_attachment([demoqa, truein])
        window_title = "Some Other Site - Google Chrome"
        page = _run(da._pick_active_page(att, prefer_window_title=window_title))
        self.assertIs(page, demoqa)  # MRU fallback

    def test_single_candidate_skips_title_check(self):
        # Optimization: with a single candidate, we don't await page.title()
        # at all. Verify by making title() raise — the picker should still
        # return the page.
        truein = _make_page("https://truein.com/", "Truein")
        truein.title = AsyncMock(side_effect=RuntimeError("don't call me"))
        att = _make_attachment([truein])
        window_title = "Anything - Google Chrome"
        page = _run(da._pick_active_page(att, prefer_window_title=window_title))
        self.assertIs(page, truein)
        truein.title.assert_not_awaited()

    def test_title_call_raising_is_skipped(self):
        # If page.title() raises (page closing mid-pick), skip it and try
        # the next candidate.
        broken = _make_page("https://broken.com/", "")
        broken.title = AsyncMock(side_effect=RuntimeError("page closed"))
        truein = _make_page("https://truein.com/", "Truein")
        att = _make_attachment([broken, truein])
        page = _run(da._pick_active_page(
            att, prefer_window_title="Truein - Google Chrome"
        ))
        self.assertIs(page, truein)

    def test_empty_window_title_skips_match(self):
        # Empty / None hint → no title check, MRU fallback.
        a = _make_page("https://a.com/", "A")
        b = _make_page("https://b.com/", "B")
        att = _make_attachment([a, b])
        page = _run(da._pick_active_page(att, prefer_window_title=""))
        self.assertIs(page, a)
        page = _run(da._pick_active_page(att, prefer_window_title=None))
        self.assertIs(page, a)

    def test_generic_chrome_window_title_skips_match(self):
        # Window title is just "Google Chrome" (no page) → strip leaves
        # nothing → no hint match attempted.
        a = _make_page("https://a.com/", "A")
        b = _make_page("https://b.com/", "B")
        att = _make_attachment([a, b])
        page = _run(da._pick_active_page(att, prefer_window_title="Google Chrome"))
        self.assertIs(page, a)  # MRU

    def test_match_across_multiple_contexts(self):
        # Title-match must look across all contexts, not just the first.
        ctx1_page = _make_page("https://demoqa.com/", "DemoQA")
        ctx2_page = _make_page("https://truein.com/", "Truein")
        att = _make_attachment([ctx1_page], [ctx2_page])
        window_title = "Truein - Google Chrome"
        page = _run(da._pick_active_page(att, prefer_window_title=window_title))
        self.assertIs(page, ctx2_page)

    def test_case_insensitive_match(self):
        truein = _make_page("https://truein.com/", "TRUEIN")
        demoqa = _make_page("https://demoqa.com/", "DemoQA")
        att = _make_attachment([demoqa, truein])
        window_title = "truein - Google Chrome"
        page = _run(da._pick_active_page(att, prefer_window_title=window_title))
        self.assertIs(page, truein)


# ─── _execute_dom_task wiring ────────────────────────────────────────────


class TestExecuteDomTaskPassesWindowTitle(unittest.TestCase):
    """Verify the foreground_window_title parameter actually flows into
    _pick_active_page when _execute_dom_task is invoked."""

    def test_window_title_forwarded_to_picker(self):
        import assistant.automation.router as da_mod

        # Stub browser_cdp.get_or_attach_browser → returns a CDP handle
        # with a stub attachment carrying two pages.
        truein = _make_page("https://truein.com/", "Truein")
        demoqa = _make_page("https://demoqa.com/", "DemoQA")
        attachment = _make_attachment([demoqa, truein])

        handle = MagicMock()
        handle.kind = "cdp"
        handle.attachment = attachment

        cdp_stub = types.SimpleNamespace(
            get_or_attach_browser=AsyncMock(return_value=handle),
        )

        # Stub orchestrator.run_dom_task → returns success result with the
        # picked page recorded so we can assert which one was chosen.
        chosen_page_box = {}

        async def fake_run_dom_task(goal, page):
            chosen_page_box["page"] = page
            res = MagicMock()
            res.success = True
            res.final_summary = "Done."
            res.reason = "ok"
            res.loops_used = 1
            return res

        orch_stub = types.SimpleNamespace(run_dom_task=fake_run_dom_task)

        # Patch the lazy imports inside _execute_dom_task.
        sys.modules["assistant.automation.browser.cdp"] = cdp_stub
        sys.modules["assistant.automation.browser.dom_orchestrator"] = orch_stub
        try:
            result = _run(da_mod._execute_dom_task(
                "fill the truein form",
                foreground_window_title="Truein - Google Chrome",
            ))
        finally:
            del sys.modules["assistant.automation.browser.cdp"]
            del sys.modules["assistant.automation.browser.dom_orchestrator"]

        self.assertEqual(result, "Done.")
        self.assertIs(chosen_page_box["page"], truein)


if __name__ == "__main__":
    unittest.main(verbosity=2)
