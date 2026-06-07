"""
test_browser_dom.py — Phase 1B: Accessibility-tree perception.

Strategy: stub-based tests against a fake Playwright Page. The JS query
itself can't be unit-tested without a real browser; that's covered by
the integration test gate (env var `DOM_REAL_BROWSER=1`, manual). Here
we test:

  - _build_ref content-addressing (same content → same ref; bounds
    quantization tolerates 1-px reflow)
  - _disambiguate_ref collision handling
  - _apply_token_budget truncation strategy (drop bounds → drop
    placeholders → prune invisible → tail-prune)
  - read_page_dom on canned JS-return JSON: empty page, normal form,
    malformed element, missing fields, oversized tree, page.evaluate
    crash, malformed top-level return
  - cache TTL behavior (within TTL returns cached; manual invalidation
    forces re-read)
  - ref_to_locator built via page.locator() with the data-tenka-idx
    selector
  - Pruned elements DROP from ref_to_locator (executor can't try to act
    on a ref the planner never saw)
  - serialize_for_planner shape

Run: python test_browser_dom.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom as bdom
import assistant.config as cfg


def _run(coro):
    return asyncio.run(coro)


# ─── Fake Playwright Page ────────────────────────────────────────────────


class _FakeLocator:
    """Lightweight Locator stand-in. Only needs to be hashable + comparable
    by reference for the tests."""
    def __init__(self, selector):
        self.selector = selector

    def __repr__(self):
        return f"FakeLocator({self.selector!r})"


class _FakePage:
    """Minimal Page stub: .evaluate() returns canned data, .locator()
    returns a FakeLocator carrying the selector string. Captures the JS
    + arg passed to evaluate so tests can assert on them."""
    def __init__(self, evaluate_return=None, evaluate_raises=None):
        self._return = evaluate_return
        self._raises = evaluate_raises
        self.evaluate_calls: list[tuple[str, dict]] = []
        self.locator_calls: list[str] = []

    async def evaluate(self, js, arg=None):
        self.evaluate_calls.append((js, arg))
        if self._raises is not None:
            raise self._raises
        return self._return

    def locator(self, selector):
        self.locator_calls.append(selector)
        return _FakeLocator(selector)


def _row(idx: int, **overrides) -> dict:
    """Default-ish element row matching what the JS would return."""
    base = {
        "idx": idx,
        "tag": "input",
        "role": "textbox",
        "name": f"Field {idx}",
        "placeholder": "",
        "value": "",
        "options": [],
        "bounds": [10, 100 + idx * 50, 200, 30],
        "visible": True,
        "enabled": True,
        "type": "text",
    }
    base.update(overrides)
    return base


def _evaluate_return(rows: list[dict], viewport=(1280, 800)) -> dict:
    return {"elements": rows, "viewport": list(viewport)}


# ─── _build_ref / _disambiguate_ref ──────────────────────────────────────


class TestRefScheme(unittest.TestCase):
    def test_same_content_same_ref(self):
        a = bdom._build_ref("textbox", "First name", "", (100, 200, 300, 30))
        b = bdom._build_ref("textbox", "First name", "", (100, 200, 300, 30))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)

    def test_bounds_quantization_tolerates_small_reflow(self):
        # 1px shift falls within the same 8-px bucket, so ref unchanged.
        a = bdom._build_ref("textbox", "F", "", (100, 200, 300, 30))
        b = bdom._build_ref("textbox", "F", "", (101, 200, 300, 30))
        self.assertEqual(a, b)

    def test_bounds_change_beyond_bucket_changes_ref(self):
        a = bdom._build_ref("textbox", "F", "", (100, 200, 300, 30))
        b = bdom._build_ref("textbox", "F", "", (120, 200, 300, 30))  # 20px later
        self.assertNotEqual(a, b)

    def test_different_role_changes_ref(self):
        a = bdom._build_ref("textbox", "Submit", "", (100, 200, 300, 30))
        b = bdom._build_ref("button", "Submit", "", (100, 200, 300, 30))
        self.assertNotEqual(a, b)

    def test_disambiguate_first_returns_base(self):
        used: dict[str, int] = {}
        ref = bdom._disambiguate_ref("abc1234567", used)
        self.assertEqual(ref, "abc1234567")
        self.assertEqual(used["abc1234567"], 1)

    def test_disambiguate_collision_appends_counter(self):
        used: dict[str, int] = {"abc1234567": 1}
        r2 = bdom._disambiguate_ref("abc1234567", used)
        r3 = bdom._disambiguate_ref("abc1234567", used)
        self.assertEqual(r2, "abc1234567:2")
        self.assertEqual(r3, "abc1234567:3")


# ─── Token-budget enforcement ────────────────────────────────────────────


def _make_element(idx: int, *, visible: bool = True, placeholder: str = "") -> bdom.ElementInfo:
    return bdom.ElementInfo(
        ref=f"r{idx:04d}aaaa",
        role="textbox",
        name=f"Field {idx}",
        placeholder=placeholder,
        value="",
        options=(),
        bounds=(10, 100 + idx * 30, 200, 30),
        visible=visible,
        enabled=True,
        type="text",
        tag="input",
    )


class TestTokenBudget(unittest.TestCase):
    def test_under_budget_no_change(self):
        elements = [_make_element(i) for i in range(5)]
        kept, truncated, flags = bdom._apply_token_budget(elements, budget=4000)
        self.assertEqual(len(kept), 5)
        self.assertEqual(truncated, 0)
        self.assertFalse(flags["drop_bounds"])

    def test_drop_bounds_first(self):
        # 200 elements × 40 tokens = 8000 — exceeds 4000. Dropping bounds
        # → 200 × 25 = 5000, still over → also drops placeholder → 200 × 22 = 4400.
        # Still over → prunes. We just verify the flag flipped.
        elements = [_make_element(i) for i in range(200)]
        kept, truncated, flags = bdom._apply_token_budget(elements, budget=4000)
        self.assertTrue(flags["drop_bounds"])

    def test_drop_placeholder_when_bounds_alone_insufficient(self):
        # 150 × 25 = 3750 (fits with bounds dropped)
        # 200 × 25 = 5000 (doesn't fit, need placeholder dropped → 4400, still over)
        elements = [_make_element(i, placeholder="hint") for i in range(170)]
        kept, truncated, flags = bdom._apply_token_budget(elements, budget=4000)
        self.assertTrue(flags["drop_bounds"])
        self.assertTrue(flags["drop_placeholder"])

    def test_invisible_pruned_before_visible(self):
        # Make a tree where dropping flags isn't enough; invisible elements
        # should be removed first.
        elements = (
            [_make_element(i, visible=True) for i in range(150)]
            + [_make_element(i, visible=False) for i in range(150, 200)]
        )
        kept, truncated, flags = bdom._apply_token_budget(elements, budget=4000)
        # All kept should be visible (invisible pruned in step 4)
        self.assertTrue(all(e.visible for e in kept))
        self.assertEqual(truncated + len(kept), 200)

    def test_tail_prune_when_invisible_pass_insufficient(self):
        # All visible, 200 of them, all with placeholder — token budget
        # forces tail-pruning even after dropping bounds + placeholder.
        elements = [_make_element(i, placeholder="hint") for i in range(200)]
        kept, truncated, flags = bdom._apply_token_budget(elements, budget=2000)
        self.assertGreater(truncated, 0)
        self.assertLessEqual(len(kept) * 22, 2000 + 22)

    def test_estimate_tokens_increases_with_options(self):
        # Element with options should cost more than one without.
        e_no_opts = _make_element(0)
        e_with_opts = bdom.ElementInfo(
            ref=e_no_opts.ref, role=e_no_opts.role, name=e_no_opts.name,
            placeholder=e_no_opts.placeholder, value=e_no_opts.value,
            options=tuple(f"opt{i}" for i in range(20)),
            bounds=e_no_opts.bounds, visible=True, enabled=True,
            type=e_no_opts.type, tag=e_no_opts.tag,
        )
        n0 = bdom._estimate_tokens([e_no_opts], drop_bounds=False, drop_placeholder=False)
        n1 = bdom._estimate_tokens([e_with_opts], drop_bounds=False, drop_placeholder=False)
        self.assertGreater(n1, n0)


# ─── read_page_dom — happy path ──────────────────────────────────────────


class TestReadPageDomHappy(unittest.TestCase):
    def setUp(self):
        bdom.reset_state_for_test()

    def tearDown(self):
        bdom.reset_state_for_test()

    def test_empty_page(self):
        page = _FakePage(evaluate_return=_evaluate_return([]))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(tree.elements, [])
        self.assertEqual(tree.ref_to_locator, {})
        self.assertEqual(tree.truncated, 0)
        self.assertEqual(tree.viewport, (1280, 800))

    def test_normal_form(self):
        rows = [
            _row(0, name="First name"),
            _row(1, name="Last name"),
            _row(2, name="Email", type="email"),
            _row(3, tag="select", role="combobox", name="Country",
                 options=["USA", "Canada", "UK"]),
            _row(4, tag="button", role="button", name="Submit"),
        ]
        page = _FakePage(evaluate_return=_evaluate_return(rows))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(len(tree.elements), 5)
        self.assertEqual(tree.elements[0].name, "First name")
        self.assertEqual(tree.elements[3].role, "combobox")
        self.assertEqual(tree.elements[3].options, ("USA", "Canada", "UK"))
        # All refs unique
        self.assertEqual(len(set(e.ref for e in tree.elements)), 5)
        # ref_to_locator built for every element
        self.assertEqual(len(tree.ref_to_locator), 5)
        # Locators built via data-tenka-idx selectors
        for idx, sel in enumerate(page.locator_calls):
            self.assertEqual(sel, f"[data-tenka-idx='{idx}']")

    def test_evaluate_called_with_filter_and_options_args(self):
        page = _FakePage(evaluate_return=_evaluate_return([]))
        _run(bdom.read_page_dom(page, filter="form", open_comboboxes=True))
        self.assertEqual(len(page.evaluate_calls), 1)
        _, arg = page.evaluate_calls[0]
        self.assertEqual(arg, {"filter": "form", "openComboboxes": True})

    def test_placeholder_used_as_name_when_label_missing(self):
        # Move 1: Webflow/React forms lack <label> association. The
        # placeholder is the visible identifier — fall back to it.
        rows = [
            _row(0, name="", placeholder="First name"),
            _row(1, name="", placeholder="Work Email", type="email"),
            _row(2, name="Has Real Label", placeholder="example@x.com"),  # name wins
        ]
        page = _FakePage(evaluate_return=_evaluate_return(rows))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(tree.elements[0].name, "First name")
        self.assertEqual(tree.elements[1].name, "Work Email")
        # Placeholder still preserved on the ElementInfo for any downstream
        # consumer that wants to know the original source
        self.assertEqual(tree.elements[0].placeholder, "First name")
        # When real label exists, name wins; placeholder is separate
        self.assertEqual(tree.elements[2].name, "Has Real Label")
        self.assertEqual(tree.elements[2].placeholder, "example@x.com")

    def test_placeholder_fallback_handles_empty_both(self):
        # Both name and placeholder empty — name stays empty (no fabrication)
        rows = [_row(0, name="", placeholder="")]
        page = _FakePage(evaluate_return=_evaluate_return(rows))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(tree.elements[0].name, "")

    def test_collision_disambiguated(self):
        # Two identical rows → same base_ref → second gets `:2` suffix.
        rows = [
            _row(0, name="Name", bounds=[10, 100, 200, 30]),
            _row(1, name="Name", bounds=[10, 100, 200, 30]),  # same content
        ]
        page = _FakePage(evaluate_return=_evaluate_return(rows))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(len(tree.elements), 2)
        ref0, ref1 = tree.elements[0].ref, tree.elements[1].ref
        self.assertNotEqual(ref0, ref1)
        self.assertTrue(ref1.endswith(":2"))


# ─── read_page_dom — failure / edge cases ────────────────────────────────


class TestReadPageDomFailures(unittest.TestCase):
    def setUp(self):
        bdom.reset_state_for_test()

    def tearDown(self):
        bdom.reset_state_for_test()

    def test_evaluate_raises_returns_empty_tree(self):
        page = _FakePage(evaluate_raises=RuntimeError("page closed"))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(tree.elements, [])
        self.assertEqual(tree.ref_to_locator, {})

    def test_malformed_top_level_return(self):
        page = _FakePage(evaluate_return="garbage string")
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(tree.elements, [])

    def test_malformed_element_row_skipped(self):
        rows = [
            _row(0, name="Good"),
            "this is not a dict",   # garbage row — should be skipped
            _row(1, name="Also good"),
        ]
        page = _FakePage(evaluate_return=_evaluate_return(rows))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(len(tree.elements), 2)

    def test_missing_idx_skipped(self):
        rows = [
            _row(0, name="Good"),
            {**_row(1, name="No idx"), "idx": -1},  # idx<0 → skip
        ]
        page = _FakePage(evaluate_return=_evaluate_return(rows))
        tree = _run(bdom.read_page_dom(page))
        self.assertEqual(len(tree.elements), 1)

    def test_truncated_elements_dropped_from_ref_map(self):
        # Force token budget to prune. Pruned refs MUST NOT remain in the
        # locator map — the executor would otherwise act on refs the
        # planner never saw.
        rows = [_row(i, name=f"F{i}") for i in range(200)]
        page = _FakePage(evaluate_return=_evaluate_return(rows))
        # Tighten budget to force truncation
        original_budget = cfg.BROWSER_DOM_TREE_TOKEN_BUDGET
        cfg.BROWSER_DOM_TREE_TOKEN_BUDGET = 1500
        try:
            tree = _run(bdom.read_page_dom(page))
            self.assertGreater(tree.truncated, 0)
            self.assertEqual(set(e.ref for e in tree.elements), set(tree.ref_to_locator.keys()))
        finally:
            cfg.BROWSER_DOM_TREE_TOKEN_BUDGET = original_budget


# ─── Cache behavior ─────────────────────────────────────────────────────


class TestCache(unittest.TestCase):
    def setUp(self):
        bdom.reset_state_for_test()

    def tearDown(self):
        bdom.reset_state_for_test()

    def test_cache_within_ttl_skips_evaluate(self):
        page = _FakePage(evaluate_return=_evaluate_return([_row(0)]))
        _run(bdom.read_page_dom(page))
        self.assertEqual(len(page.evaluate_calls), 1)
        # Second call within TTL must NOT re-evaluate.
        tree2 = _run(bdom.read_page_dom(page))
        self.assertEqual(len(page.evaluate_calls), 1)
        self.assertEqual(len(tree2.elements), 1)

    def test_use_cache_false_forces_reread(self):
        page = _FakePage(evaluate_return=_evaluate_return([_row(0)]))
        _run(bdom.read_page_dom(page))
        _run(bdom.read_page_dom(page, use_cache=False))
        self.assertEqual(len(page.evaluate_calls), 2)

    def test_invalidate_tree_cache_forces_reread(self):
        page = _FakePage(evaluate_return=_evaluate_return([_row(0)]))
        _run(bdom.read_page_dom(page))
        bdom.invalidate_tree_cache(page)
        _run(bdom.read_page_dom(page))
        self.assertEqual(len(page.evaluate_calls), 2)

    def test_invalidate_unknown_page_safe(self):
        # No-op when no cache entry for the page
        page = _FakePage(evaluate_return=_evaluate_return([]))
        bdom.invalidate_tree_cache(page)
        # Just verify no exception

    def test_cache_separates_pages(self):
        page_a = _FakePage(evaluate_return=_evaluate_return([_row(0, name="A")]))
        page_b = _FakePage(evaluate_return=_evaluate_return([_row(0, name="B")]))
        tree_a = _run(bdom.read_page_dom(page_a))
        tree_b = _run(bdom.read_page_dom(page_b))
        self.assertEqual(tree_a.elements[0].name, "A")
        self.assertEqual(tree_b.elements[0].name, "B")
        # Both pages get separate cache entries
        self.assertIn(id(page_a), bdom._tree_cache)
        self.assertIn(id(page_b), bdom._tree_cache)


# ─── serialize_for_planner ──────────────────────────────────────────────


class TestSerializeForPlanner(unittest.TestCase):
    def test_basic_serialization_shape(self):
        e = _make_element(0)
        tree = bdom.PageDomTree(
            elements=[e], ref_to_locator={e.ref: _FakeLocator("[data-tenka-idx='0']")},
            truncated=0, read_at=time.monotonic(), viewport=(800, 600),
        )
        s = bdom.serialize_for_planner(tree)
        parsed = json.loads(s)
        self.assertIn("elements", parsed)
        self.assertEqual(len(parsed["elements"]), 1)
        self.assertEqual(parsed["elements"][0]["ref"], e.ref)
        self.assertEqual(parsed["elements"][0]["role"], "textbox")
        self.assertEqual(parsed["elements"][0]["name"], "Field 0")
        self.assertNotIn("_truncated", parsed)

    def test_truncation_marker_present_when_pruned(self):
        e = _make_element(0)
        tree = bdom.PageDomTree(
            elements=[e], ref_to_locator={},
            truncated=42, read_at=time.monotonic(), viewport=(800, 600),
        )
        s = bdom.serialize_for_planner(tree)
        parsed = json.loads(s)
        self.assertEqual(parsed["_truncated"], 42)

    def test_options_included_when_present(self):
        e = bdom.ElementInfo(
            ref="abc1234567", role="combobox", name="Country", placeholder="",
            value="USA", options=("USA", "Canada", "UK"), bounds=(0, 0, 100, 30),
            visible=True, enabled=True, type="", tag="select",
        )
        tree = bdom.PageDomTree(
            elements=[e], ref_to_locator={},
            truncated=0, read_at=time.monotonic(), viewport=(800, 600),
        )
        s = bdom.serialize_for_planner(tree)
        parsed = json.loads(s)
        self.assertEqual(parsed["elements"][0]["options"], ["USA", "Canada", "UK"])

    def test_placeholder_suppressed_when_equal_to_name(self):
        # Move 1: when placeholder was used as the name fallback, emitting
        # both would waste tokens. Suppress the redundant placeholder.
        e_redundant = bdom.ElementInfo(
            ref="r0000aaaa", role="textbox", name="First name",
            placeholder="First name",  # equal to name (placeholder fallback)
            value="", options=(), bounds=(0, 0, 100, 30),
            visible=True, enabled=True, type="text", tag="input",
        )
        e_distinct = bdom.ElementInfo(
            ref="r0001aaaa", role="textbox", name="First name",
            placeholder="Jane",  # distinct — keep
            value="", options=(), bounds=(0, 0, 100, 30),
            visible=True, enabled=True, type="text", tag="input",
        )
        tree = bdom.PageDomTree(
            elements=[e_redundant, e_distinct], ref_to_locator={},
            truncated=0, read_at=time.monotonic(), viewport=(800, 600),
        )
        parsed = json.loads(bdom.serialize_for_planner(tree))
        rows = parsed["elements"]
        self.assertNotIn("placeholder", rows[0],
                         "redundant placeholder must be suppressed")
        self.assertEqual(rows[1]["placeholder"], "Jane",
                         "distinct placeholder must be kept")

    def test_invisible_and_disabled_flags_emitted(self):
        # When visible=False or enabled=False, the planner needs to know.
        # When True (default), they should NOT appear (token-saving).
        e_visible_enabled = _make_element(0, visible=True)
        e_invisible = bdom.ElementInfo(
            ref="ref0000001", role="textbox", name="A", placeholder="",
            value="", options=(), bounds=(0, 0, 100, 30),
            visible=False, enabled=True, type="text", tag="input",
        )
        e_disabled = bdom.ElementInfo(
            ref="ref0000002", role="textbox", name="B", placeholder="",
            value="", options=(), bounds=(0, 0, 100, 30),
            visible=True, enabled=False, type="text", tag="input",
        )
        tree = bdom.PageDomTree(
            elements=[e_visible_enabled, e_invisible, e_disabled],
            ref_to_locator={}, truncated=0, read_at=time.monotonic(),
            viewport=(800, 600),
        )
        parsed = json.loads(bdom.serialize_for_planner(tree))
        rows = parsed["elements"]
        self.assertNotIn("visible", rows[0])
        self.assertNotIn("enabled", rows[0])
        self.assertEqual(rows[1]["visible"], False)
        self.assertEqual(rows[2]["enabled"], False)

    def test_autocomplete_emitted_for_combobox_with_attribute(self):
        e = bdom.ElementInfo(
            ref="ref_ac_ser", role="combobox", name="Subjects",
            placeholder="", value="", options=(),
            bounds=(0, 0, 200, 30), visible=True, enabled=True,
            type="", tag="input", autocomplete="list",
        )
        tree = bdom.PageDomTree(
            elements=[e], ref_to_locator={},
            truncated=0, read_at=time.monotonic(), viewport=(800, 600),
        )
        parsed = json.loads(bdom.serialize_for_planner(tree))
        self.assertEqual(parsed["elements"][0]["autocomplete"], "list")

    def test_autocomplete_omitted_for_non_combobox(self):
        e = bdom.ElementInfo(
            ref="ref_ac_none", role="textbox", name="Email",
            placeholder="", value="", options=(),
            bounds=(0, 0, 200, 30), visible=True, enabled=True,
            type="email", tag="input",
        )
        tree = bdom.PageDomTree(
            elements=[e], ref_to_locator={},
            truncated=0, read_at=time.monotonic(), viewport=(800, 600),
        )
        parsed = json.loads(bdom.serialize_for_planner(tree))
        self.assertNotIn("autocomplete", parsed["elements"][0])

    def test_autocomplete_omitted_for_combobox_without_attribute(self):
        e = bdom.ElementInfo(
            ref="ref_ac_empty", role="combobox", name="State",
            placeholder="", value="", options=(),
            bounds=(0, 0, 200, 30), visible=True, enabled=True,
            type="", tag="div",
        )
        tree = bdom.PageDomTree(
            elements=[e], ref_to_locator={},
            truncated=0, read_at=time.monotonic(), viewport=(800, 600),
        )
        parsed = json.loads(bdom.serialize_for_planner(tree))
        self.assertNotIn("autocomplete", parsed["elements"][0])

    def test_autocomplete_omitted_for_native_select_with_options(self):
        e = bdom.ElementInfo(
            ref="ref_ac_nat", role="combobox", name="Country",
            placeholder="", value="USA", options=("USA", "Canada", "UK"),
            bounds=(0, 0, 200, 30), visible=True, enabled=True,
            type="", tag="select",
        )
        tree = bdom.PageDomTree(
            elements=[e], ref_to_locator={},
            truncated=0, read_at=time.monotonic(), viewport=(800, 600),
        )
        parsed = json.loads(bdom.serialize_for_planner(tree))
        self.assertNotIn("autocomplete", parsed["elements"][0])


# ─── ElementInfo dataclass guarantees ────────────────────────────────────


class TestElementInfoFrozen(unittest.TestCase):
    def test_frozen_immutable(self):
        e = _make_element(0)
        with self.assertRaises(Exception):
            e.ref = "mutated"  # frozen dataclass — must raise


# ─── autocomplete field ───────────────────────────────────────────────────


class TestAutocompleteField(unittest.TestCase):
    def test_element_info_has_autocomplete_field(self):
        e = bdom.ElementInfo(
            ref="ref_ac_test", role="combobox", name="Subjects",
            placeholder="", value="", options=(),
            bounds=(0, 0, 200, 30), visible=True, enabled=True,
            type="", tag="input",
        )
        self.assertEqual(e.autocomplete, "")

    def test_element_info_autocomplete_set(self):
        e = bdom.ElementInfo(
            ref="ref_ac_test", role="combobox", name="Subjects",
            placeholder="", value="", options=(),
            bounds=(0, 0, 200, 30), visible=True, enabled=True,
            type="", tag="input", autocomplete="list",
        )
        self.assertEqual(e.autocomplete, "list")

    def test_js_query_captures_aria_autocomplete(self):
        js = bdom._DOM_QUERY_JS
        self.assertIn("aria-autocomplete", js)

    def test_hydration_reads_autocomplete_from_raw(self):
        raw_return = {
            "elements": [{
                "idx": 0, "tag": "input", "role": "combobox",
                "name": "Subjects", "placeholder": "",
                "value": "", "options": [],
                "bounds": [10, 100, 200, 30],
                "visible": True, "enabled": True,
                "type": "", "form_id": "", "in_dialog": False,
                "aria_invalid": False, "describedby": [], "el_id": "",
                "autocomplete": "list",
            }],
            "viewport": [1280, 800],
            "validation_errors": [],
        }
        page = _FakePage(evaluate_return=raw_return)
        tree = _run(bdom.read_page_dom(page, use_cache=False))
        self.assertEqual(len(tree.elements), 1)
        self.assertEqual(tree.elements[0].autocomplete, "list")

    def test_hydration_defaults_autocomplete_when_missing(self):
        raw_return = {
            "elements": [{
                "idx": 0, "tag": "input", "role": "textbox",
                "name": "Email", "placeholder": "",
                "value": "", "options": [],
                "bounds": [10, 100, 200, 30],
                "visible": True, "enabled": True,
                "type": "email", "form_id": "", "in_dialog": False,
                "aria_invalid": False, "describedby": [], "el_id": "",
            }],
            "viewport": [1280, 800],
            "validation_errors": [],
        }
        page = _FakePage(evaluate_return=raw_return)
        tree = _run(bdom.read_page_dom(page, use_cache=False))
        self.assertEqual(len(tree.elements), 1)
        self.assertEqual(tree.elements[0].autocomplete, "")


# ─── accessibleName textContent fallback for option/treeitem ─────────────


class TestAccessibleNameOptionFallback(unittest.TestCase):
    """Bug B fix: react-select option elements have no aria-label or
    <label>, so accessibleName must fall back to textContent for
    role='option' and role='treeitem'."""

    def test_js_accessible_name_includes_option_role(self):
        js = bdom._DOM_QUERY_JS
        self.assertIn("explicit === 'option'", js)

    def test_js_accessible_name_includes_treeitem_role(self):
        js = bdom._DOM_QUERY_JS
        self.assertIn("explicit === 'treeitem'", js)


class TestComboboxFallbackName(unittest.TestCase):
    """Issue F: react-select comboboxes have name='' because they lack
    aria-label / <label for>. The JS comboboxFallbackName function walks
    up the DOM to find placeholder text or ancestor IDs."""

    def test_js_contains_combobox_fallback_function(self):
        js = bdom._DOM_QUERY_JS
        self.assertIn("comboboxFallbackName", js)

    def test_fallback_called_only_for_combobox(self):
        js = bdom._DOM_QUERY_JS
        self.assertIn("role === 'combobox'", js)
        self.assertIn("comboboxFallbackName(el)", js)

    def test_fallback_skips_svg_and_buttons(self):
        js = bdom._DOM_QUERY_JS
        self.assertIn("child.tagName === 'svg'", js)
        self.assertIn('[role="button"]', js)

    def test_fallback_ancestor_id_strips_noise(self):
        js = bdom._DOM_QUERY_JS
        for word in ("container", "wrapper", "field", "group"):
            self.assertIn(word, js)


if __name__ == "__main__":
    unittest.main(verbosity=2)
