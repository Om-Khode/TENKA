"""
test_dom_filler.py — Option B: deterministic per-widget form filler.

Tests:
  - FillResult / FormFillResult dataclass construction
  - Widget type classification from ElementInfo
  - Per-widget fill sequences against stub Playwright locators
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom as bdom
import assistant.automation.browser.dom_filler as filler
from assistant.automation.browser.dom_mapper import FillInstruction, FormMapping


def _elem(ref, **kw) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input", form_id="form-0", in_dialog=False,
    )
    base.update(kw)
    return bdom.ElementInfo(ref=ref, **base)


def _run(coro):
    return asyncio.run(coro)


class TestDataclasses(unittest.TestCase):
    def test_fill_result(self):
        r = filler.FillResult(
            ref="r1", field_name="Email",
            intended_value="a@b.com", observed_value="a@b.com",
            succeeded=True,
        )
        self.assertTrue(r.succeeded)
        self.assertEqual(r.error, "")

    def test_form_fill_result_all_succeeded(self):
        fills = [
            filler.FillResult(ref="r1", field_name="A", intended_value="x",
                              observed_value="x", succeeded=True),
            filler.FillResult(ref="r2", field_name="B", intended_value="y",
                              observed_value="y", succeeded=True),
        ]
        r = filler.FormFillResult(fills=fills, submit_clicked=True)
        self.assertTrue(r.all_succeeded)

    def test_form_fill_result_partial_failure(self):
        fills = [
            filler.FillResult(ref="r1", field_name="A", intended_value="x",
                              observed_value="x", succeeded=True),
            filler.FillResult(ref="r2", field_name="B", intended_value="y",
                              observed_value="", succeeded=False, error="fill failed"),
        ]
        r = filler.FormFillResult(fills=fills, submit_clicked=False)
        self.assertFalse(r.all_succeeded)


class TestClassifyWidget(unittest.TestCase):
    def test_textbox(self):
        e = _elem("r1", role="textbox")
        self.assertEqual(filler.classify_widget(e), "textbox")

    def test_native_select(self):
        e = _elem("r1", role="combobox", tag="select",
                  options=("A", "B", "C"))
        self.assertEqual(filler.classify_widget(e), "native_select")

    def test_autocomplete_combobox(self):
        e = _elem("r1", role="combobox", autocomplete="list", options=())
        self.assertEqual(filler.classify_widget(e), "autocomplete_combobox")

    def test_click_combobox(self):
        e = _elem("r1", role="combobox", autocomplete="", options=())
        self.assertEqual(filler.classify_widget(e), "click_combobox")

    def test_radio(self):
        e = _elem("r1", role="radio", type="radio")
        self.assertEqual(filler.classify_widget(e), "radio")

    def test_checkbox(self):
        e = _elem("r1", role="checkbox", type="checkbox")
        self.assertEqual(filler.classify_widget(e), "checkbox")

    def test_button(self):
        e = _elem("r1", role="button", tag="button")
        self.assertEqual(filler.classify_widget(e), "button")

    def test_textarea(self):
        e = _elem("r1", role="textbox", tag="textarea")
        self.assertEqual(filler.classify_widget(e), "textbox")

    def test_input_email(self):
        e = _elem("r1", role="textbox", type="email")
        self.assertEqual(filler.classify_widget(e), "textbox")


class _StubLocator:
    """Minimal Playwright Locator stub."""
    def __init__(self, *, input_value_return="", check_state=False):
        self._input_value = input_value_return
        self._checked = check_state
        self.fill = AsyncMock()
        self.click = AsyncMock()
        self.select_option = AsyncMock()
        self.check = AsyncMock()
        self.uncheck = AsyncMock()
        self.is_checked = AsyncMock(return_value=self._checked)
        self.input_value = AsyncMock(return_value=self._input_value)
        self.evaluate = AsyncMock(return_value=self._input_value)


class TestFillTextbox(unittest.TestCase):
    def test_happy_path(self):
        loc = _StubLocator(input_value_return="John")
        elem = _elem("r1", name="First Name")
        result = _run(filler.fill_textbox(loc, elem, "John"))
        self.assertTrue(result.succeeded)
        self.assertEqual(result.observed_value, "John")
        loc.fill.assert_called_once()

    def test_readback_mismatch(self):
        loc = _StubLocator(input_value_return="wrong")
        elem = _elem("r1", name="First Name")
        result = _run(filler.fill_textbox(loc, elem, "John"))
        self.assertFalse(result.succeeded)
        self.assertIn("mismatch", result.error)

    def test_fill_raises(self):
        loc = _StubLocator()
        loc.fill = AsyncMock(side_effect=Exception("timeout"))
        elem = _elem("r1", name="First Name")
        result = _run(filler.fill_textbox(loc, elem, "John"))
        self.assertFalse(result.succeeded)


class TestFillRadio(unittest.TestCase):
    def test_click_matching_radio(self):
        loc = _StubLocator()
        elem = _elem("r1", role="radio", name="Male", type="radio")
        result = _run(filler.fill_radio(loc, elem, "Male"))
        self.assertTrue(result.succeeded)
        loc.click.assert_called_once()

    def test_click_raises(self):
        loc = _StubLocator()
        loc.click = AsyncMock(side_effect=Exception("timeout"))
        elem = _elem("r1", role="radio", name="Male", type="radio")
        result = _run(filler.fill_radio(loc, elem, "Male"))
        self.assertFalse(result.succeeded)


class TestFillCheckbox(unittest.TestCase):
    def test_check_unchecked(self):
        loc = _StubLocator(check_state=False)
        elem = _elem("r1", role="checkbox", name="Terms")
        result = _run(filler.fill_checkbox(loc, elem, "check"))
        self.assertTrue(result.succeeded)
        loc.check.assert_called_once()

    def test_uncheck_checked(self):
        loc = _StubLocator(check_state=True)
        elem = _elem("r1", role="checkbox", name="Terms")
        result = _run(filler.fill_checkbox(loc, elem, "uncheck"))
        self.assertTrue(result.succeeded)
        loc.uncheck.assert_called_once()

    def test_already_correct_state(self):
        loc = _StubLocator(check_state=True)
        elem = _elem("r1", role="checkbox", name="Terms")
        result = _run(filler.fill_checkbox(loc, elem, "check"))
        self.assertTrue(result.succeeded)
        loc.check.assert_not_called()
        loc.uncheck.assert_not_called()


class TestFillNativeSelect(unittest.TestCase):
    def test_select_by_label(self):
        loc = _StubLocator(input_value_return="NCR")
        elem = _elem("r1", role="combobox", tag="select",
                      name="State", options=("-- Select --", "NCR", "UP"))
        result = _run(filler.fill_native_select(loc, elem, "NCR"))
        self.assertTrue(result.succeeded)
        loc.select_option.assert_called_once()

    def test_select_raises(self):
        loc = _StubLocator()
        loc.select_option = AsyncMock(side_effect=Exception("not found"))
        elem = _elem("r1", role="combobox", tag="select",
                      name="State", options=("A", "B"))
        result = _run(filler.fill_native_select(loc, elem, "C"))
        self.assertFalse(result.succeeded)


class _StubPage:
    """Minimal Playwright Page stub that returns a controllable DOM tree."""
    def __init__(self, portal_options=None):
        self._portal_options = portal_options or []

    async def evaluate(self, *args, **kwargs):
        return {}


def _make_tree_with_options(option_names, combobox_ref="cb1"):
    """Build a PageDomTree that includes portal-rendered option elements."""
    elems = []
    locs = {}
    for i, name in enumerate(option_names):
        ref = f"opt_{i}"
        elems.append(_elem(ref, role="option", name=name, form_id="", tag="div"))
        locs[ref] = _StubLocator()
    elems.append(_elem(combobox_ref, role="combobox", name="Subjects",
                        autocomplete="list", form_id="form-0"))
    locs[combobox_ref] = _StubLocator(input_value_return="Math")
    return bdom.PageDomTree(elements=elems, ref_to_locator=locs)


class TestFillCombobox(unittest.TestCase):
    @patch("assistant.automation.browser.dom_filler.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_filler.browser_dom.invalidate_tree_cache")
    def test_single_value_happy_path(self, mock_invalidate, mock_read):
        tree_with_opts = _make_tree_with_options(["Maths", "Physics", "Chemistry"])
        mock_read.return_value = tree_with_opts

        cb_loc = _StubLocator(input_value_return="Math")
        elem = _elem("cb1", role="combobox", name="Subjects", autocomplete="list")
        page = _StubPage()

        result = _run(filler.fill_combobox(cb_loc, elem, "Maths", page=page))
        self.assertTrue(result.succeeded)
        cb_loc.fill.assert_called_once()  # typed prefix
        # Option click happened on the matching option locator
        opt_loc = tree_with_opts.ref_to_locator["opt_0"]
        opt_loc.click.assert_called_once()

    @patch("assistant.automation.browser.dom_filler.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_filler.browser_dom.invalidate_tree_cache")
    def test_no_matching_option(self, mock_invalidate, mock_read):
        tree_with_opts = _make_tree_with_options(["Physics", "Chemistry"])
        mock_read.return_value = tree_with_opts

        cb_loc = _StubLocator(input_value_return="Xyz")
        elem = _elem("cb1", role="combobox", name="Subjects", autocomplete="list")
        page = _StubPage()

        result = _run(filler.fill_combobox(cb_loc, elem, "Xyzzy", page=page))
        self.assertFalse(result.succeeded)
        self.assertIn("no matching option", result.error.lower())

    @patch("assistant.automation.browser.dom_filler.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_filler.browser_dom.invalidate_tree_cache")
    def test_multi_value(self, mock_invalidate, mock_read):
        # First call: options for "Maths" prefix
        tree1 = _make_tree_with_options(["Maths", "Physics"])
        # Second call: options for "English" prefix
        tree2 = _make_tree_with_options(["English", "Hindi"])
        mock_read.side_effect = [tree1, tree2]

        cb_loc = _StubLocator(input_value_return="")
        elem = _elem("cb1", role="combobox", name="Subjects", autocomplete="list")
        page = _StubPage()

        results = _run(filler.fill_combobox_multi(
            cb_loc, elem, ["Maths", "English"], page=page,
        ))
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].succeeded)
        self.assertTrue(results[1].succeeded)

    @patch("assistant.automation.browser.dom_filler.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_filler.browser_dom.invalidate_tree_cache")
    def test_click_combobox_opens_then_selects(self, mock_invalidate, mock_read):
        tree_with_opts = _make_tree_with_options(["Maths", "Physics"])
        mock_read.return_value = tree_with_opts

        cb_loc = _StubLocator()
        elem = _elem("cb1", role="combobox", name="Category", autocomplete="")
        page = _StubPage()

        result = _run(filler.fill_combobox(cb_loc, elem, "Maths", page=page))
        self.assertTrue(result.succeeded)
        cb_loc.click.assert_called_once()  # opened dropdown


class TestFillForm(unittest.TestCase):
    @patch("assistant.automation.browser.dom_filler.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_filler.browser_dom.invalidate_tree_cache")
    def test_mixed_widget_form(self, mock_invalidate, mock_read):
        """Fill a form with textbox + radio + native select."""
        text_loc = _StubLocator(input_value_return="John")
        radio_loc = _StubLocator()
        select_loc = _StubLocator(input_value_return="NCR")
        submit_loc = _StubLocator()

        elements = [
            _elem("r1", name="First Name"),
            _elem("r2", role="radio", name="Male", type="radio"),
            _elem("r3", role="combobox", tag="select", name="State",
                  options=("-- Select --", "NCR", "UP")),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        ref_to_locator = {
            "r1": text_loc, "r2": radio_loc,
            "r3": select_loc, "s1": submit_loc,
        }
        tree = bdom.PageDomTree(elements=elements, ref_to_locator=ref_to_locator)

        mapping = FormMapping(
            fills=[
                FillInstruction(ref="r1", field_name="First Name", value="John"),
                FillInstruction(ref="r2", field_name="Male", value="Male"),
                FillInstruction(ref="r3", field_name="State", value="NCR"),
            ],
            submit_ref="s1",
            thinking="fill name, gender, state, submit",
        )

        page = _StubPage()
        result = _run(filler.fill_form(mapping, tree, page))
        self.assertTrue(result.all_succeeded)
        self.assertTrue(result.submit_clicked)
        text_loc.fill.assert_called_once()
        radio_loc.click.assert_called_once()
        select_loc.select_option.assert_called_once()
        submit_loc.click.assert_called_once()

    def test_skip_submit(self):
        text_loc = _StubLocator(input_value_return="X")
        elements = [_elem("r1", name="Field")]
        ref_to_locator = {"r1": text_loc}
        tree = bdom.PageDomTree(elements=elements, ref_to_locator=ref_to_locator)

        mapping = FormMapping(
            fills=[FillInstruction(ref="r1", field_name="Field", value="X")],
            submit_ref="",
            thinking="just fill",
            skip_submit=True,
        )
        page = _StubPage()
        result = _run(filler.fill_form(mapping, tree, page))
        self.assertTrue(result.all_succeeded)
        self.assertFalse(result.submit_clicked)

    def test_missing_locator_skipped(self):
        elements = [_elem("r1", name="Field")]
        ref_to_locator = {}  # no locator for r1
        tree = bdom.PageDomTree(elements=elements, ref_to_locator=ref_to_locator)

        mapping = FormMapping(
            fills=[FillInstruction(ref="r1", field_name="Field", value="X")],
            submit_ref="",
            thinking="fill",
        )
        page = _StubPage()
        result = _run(filler.fill_form(mapping, tree, page))
        self.assertFalse(result.all_succeeded)
        self.assertIn("no locator", result.fills[0].error.lower())


if __name__ == "__main__":
    unittest.main()
