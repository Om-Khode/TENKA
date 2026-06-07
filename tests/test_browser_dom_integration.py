"""
test_browser_dom_integration.py — Phase 1B real-Playwright integration tests.

These tests drive an actual headless Chromium against the static fixture
HTML pages in tests/fixtures/dom/. They validate the JS perception query
against a real DOM (the unit tests in test_browser_dom.py only exercise
the Python-side processing of canned JS returns).

GATED by env var DOM_REAL_BROWSER=1. When unset, the entire module is
skipped — CI without Chromium installed sees a clean SKIP, no failures.

Run locally:
  set DOM_REAL_BROWSER=1
  python test_browser_dom_integration.py

Or one-shot (Windows cmd):
  cmd /c "set DOM_REAL_BROWSER=1 && python test_browser_dom_integration.py"
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Gate: env var DOM_REAL_BROWSER=1 enables the real-Chromium tests.
# Without it, the test classes are defined but every test is skipped via
# the @unittest.skipUnless decorator below — both `python test_X.py` and
# `python -m unittest` show clean SKIPPED output, no tracebacks.
_GATE = os.environ.get("DOM_REAL_BROWSER", "").strip()
_REAL_BROWSER_ENABLED = _GATE in ("1", "true", "yes", "on")
_SKIP_REASON = (
    "DOM_REAL_BROWSER not set. Set =1 to run (requires Chromium via "
    "`playwright install chromium`)."
)


# Always import the modules under test — `read_page_dom` needs no Chromium.
import assistant.automation.browser.dom as bdom
import assistant.config as cfg

# Defer Playwright import to test setUp so module load doesn't fail when
# Playwright isn't installed.
async_playwright = None
if _REAL_BROWSER_ENABLED:
    try:
        from playwright.async_api import async_playwright as _ap
        async_playwright = _ap
    except ImportError:
        _REAL_BROWSER_ENABLED = False
        _SKIP_REASON = "Playwright not installed (pip install playwright)."


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dom"


def _fixture_uri(name: str) -> str:
    """Convert tests/fixtures/dom/<name> to a file:// URI Playwright can load."""
    p = (_FIXTURE_DIR / name).resolve()
    if not p.exists():
        raise FileNotFoundError(f"fixture missing: {p}")
    return p.as_uri()


# ─── Base class with per-test browser setup ──────────────────────────────


# Decorator applied per-subclass below — Python's skipUnless doesn't
# inherit, so the base class would otherwise hide the skip from concrete
# test classes.
class _RealBrowserTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Per-test Chromium launch. Slower than session-scoped, but
    IsolatedAsyncioTestCase makes a fresh event loop per test, which
    conflicts with sharing a long-lived Playwright instance across the
    suite. ~2s per test in headless mode — acceptable for a manual
    smoke run.
    """

    async def asyncSetUp(self):
        bdom.reset_state_for_test()
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)
        self._page = await self._browser.new_page()

    async def asyncTearDown(self):
        try:
            await self._browser.close()
        finally:
            await self._pw.stop()
        bdom.reset_state_for_test()

    async def _load(self, fixture: str):
        """Navigate to a fixture and wait for it to load."""
        await self._page.goto(_fixture_uri(fixture), wait_until="domcontentloaded")


# ─── form_basic — canonical interactive form ─────────────────────────────


@unittest.skipUnless(_REAL_BROWSER_ENABLED, _SKIP_REASON)
class TestFormBasic(_RealBrowserTestCase):
    async def test_captures_8_inputs_and_select_and_button(self):
        await self._load("form_basic.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        # 8 inputs + 1 select + 1 submit button = 10
        self.assertEqual(len(tree.elements), 10,
                         f"expected 10, got {[e.name for e in tree.elements]}")
        # Roles distributed correctly
        roles = [e.role for e in tree.elements]
        self.assertEqual(roles.count("textbox"), 8)  # text/email/tel/password/date/url all → textbox
        self.assertEqual(roles.count("combobox"), 1)  # native <select>
        self.assertEqual(roles.count("button"), 1)
        # Names captured via labels
        names = {e.name for e in tree.elements}
        self.assertIn("First name", names)
        self.assertIn("Last name", names)
        self.assertIn("Email", names)
        self.assertIn("Country", names)

    async def test_native_select_options_enumerated(self):
        await self._load("form_basic.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        country = next(e for e in tree.elements if e.name == "Country")
        self.assertEqual(country.role, "combobox")
        # First option is the placeholder, then the real four
        self.assertIn("United States", country.options)
        self.assertIn("Canada", country.options)
        self.assertIn("United Kingdom", country.options)
        self.assertIn("India", country.options)

    async def test_input_type_attribute_captured(self):
        await self._load("form_basic.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        types = {e.name: e.type for e in tree.elements if e.role == "textbox"}
        self.assertEqual(types["Email"], "email")
        self.assertEqual(types["Phone"], "tel")
        self.assertEqual(types["Password"], "password")
        self.assertEqual(types["Birthday"], "date")
        self.assertEqual(types["Website"], "url")
        self.assertEqual(types["First name"], "text")

    async def test_filter_form_returns_only_form_descendants(self):
        await self._load("form_basic.html")
        tree = await bdom.read_page_dom(self._page, "form")
        # All 10 form descendants — the page has exactly one form
        self.assertEqual(len(tree.elements), 10)

    async def test_refs_stable_across_two_reads(self):
        await self._load("form_basic.html")
        t1 = await bdom.read_page_dom(self._page, "interactive")
        bdom.invalidate_tree_cache(self._page)
        t2 = await bdom.read_page_dom(self._page, "interactive")
        # Same elements, same refs (content is identical)
        refs1 = sorted(e.ref for e in t1.elements)
        refs2 = sorted(e.ref for e in t2.elements)
        self.assertEqual(refs1, refs2)


# ─── form_combobox — native select + custom ARIA combobox ───────────────


@unittest.skipUnless(_REAL_BROWSER_ENABLED, _SKIP_REASON)
class TestFormCombobox(_RealBrowserTestCase):
    async def test_native_select_enumerates_options(self):
        await self._load("form_combobox.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        size = next(e for e in tree.elements if e.name == "Staff size (native <select>)")
        self.assertEqual(size.role, "combobox")
        self.assertEqual(set(size.options),
                         {"1-50", "51-100", "101-200", "201-500", "501+"})

    async def test_custom_combobox_returns_empty_options(self):
        # Per design: perceiver does NOT auto-open custom comboboxes —
        # that side-effect belongs in the orchestrator's open-then-reperceive.
        await self._load("form_combobox.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        industry = next(
            (e for e in tree.elements if e.role == "combobox" and "Industry" in e.name),
            None,
        )
        self.assertIsNotNone(industry, "custom combobox not captured")
        self.assertEqual(industry.options, ())  # empty until orchestrator opens it

    async def test_combobox_has_correct_role(self):
        await self._load("form_combobox.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        # Both should report role="combobox"
        comboboxes = [e for e in tree.elements if e.role == "combobox"]
        self.assertEqual(len(comboboxes), 2)


# ─── form_hidden — hidden / disabled / visually-hidden edge cases ───────


@unittest.skipUnless(_REAL_BROWSER_ENABLED, _SKIP_REASON)
class TestFormHidden(_RealBrowserTestCase):
    async def test_hidden_input_excluded(self):
        await self._load("form_hidden.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        names = [e.name for e in tree.elements]
        # input[type=hidden] has no role — excluded entirely
        for n in names:
            self.assertNotIn("csrf", n.lower())

    async def test_disabled_input_included_with_enabled_false(self):
        await self._load("form_hidden.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        disabled = next(
            (e for e in tree.elements if "Disabled" in e.name),
            None,
        )
        self.assertIsNotNone(disabled)
        self.assertFalse(disabled.enabled)

    async def test_aria_disabled_input_included_with_enabled_false(self):
        await self._load("form_hidden.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        aria_dis = next(
            (e for e in tree.elements if "aria-disabled" in e.name),
            None,
        )
        self.assertIsNotNone(aria_dis)
        self.assertFalse(aria_dis.enabled)

    async def test_display_none_included_visible_false(self):
        await self._load("form_hidden.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        none = next(
            (e for e in tree.elements if "display-none" in e.name),
            None,
        )
        self.assertIsNotNone(none)
        self.assertFalse(none.visible)

    async def test_visibility_hidden_included_visible_false(self):
        await self._load("form_hidden.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        vh = next(
            (e for e in tree.elements if "visibility-hidden" in e.name),
            None,
        )
        self.assertIsNotNone(vh)
        self.assertFalse(vh.visible)

    async def test_opacity_zero_included_visible_false(self):
        await self._load("form_hidden.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        op = next(
            (e for e in tree.elements if "opacity-zero" in e.name),
            None,
        )
        self.assertIsNotNone(op)
        self.assertFalse(op.visible)


# ─── form_collision — duplicate-content elements ────────────────────────


@unittest.skipUnless(_REAL_BROWSER_ENABLED, _SKIP_REASON)
class TestFormCollision(_RealBrowserTestCase):
    async def test_duplicate_content_gets_disambiguated_refs(self):
        await self._load("form_collision.html")
        tree = await bdom.read_page_dom(self._page, "interactive")
        comments = [e for e in tree.elements if e.name == "Comment"]
        self.assertEqual(len(comments), 2, "expected 2 'Comment' inputs")
        # The two refs differ — the second appends `:2`
        ref1, ref2 = comments[0].ref, comments[1].ref
        self.assertNotEqual(ref1, ref2)
        self.assertTrue(ref2.endswith(":2"), f"expected `:2` suffix, got {ref2!r}")
        # Both refs must be in ref_to_locator (executor safety)
        self.assertIn(ref1, tree.ref_to_locator)
        self.assertIn(ref2, tree.ref_to_locator)


# ─── form_oversized — token-budget truncation pipeline ──────────────────


@unittest.skipUnless(_REAL_BROWSER_ENABLED, _SKIP_REASON)
class TestFormOversized(_RealBrowserTestCase):
    async def test_truncation_fires_at_default_budget(self):
        await self._load("form_oversized.html")
        # Use the actual default budget (4000 tokens).
        tree = await bdom.read_page_dom(self._page, "interactive")
        self.assertGreater(tree.truncated, 0,
                           "expected token-budget truncation to fire on 250-field form")
        # Pruned refs MUST NOT remain in ref_to_locator (executor safety
        # invariant — never let the executor act on a ref the planner
        # never saw).
        self.assertEqual(
            set(e.ref for e in tree.elements),
            set(tree.ref_to_locator.keys()),
            "pruned refs leaked into ref_to_locator",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
