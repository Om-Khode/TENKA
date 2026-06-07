"""
test_dom_form_fill_integration.py — Option B: mapper→filler→verify integration.

Tests:
  - Happy-path single-loop completion (1 LLM call)
  - Validation error → corrective map → re-fill (2 LLM calls)
  - Empty mapping → failure
  - Perceive failure → graceful exit
  - Fill failures reported as failure, not false success (Bug 2)
  - Cascading/dependent fields: second pass for newly-enabled fields (Bug 1)
  - No false dependent-field detection for combobox option elements
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom as bdom
import assistant.automation.browser.dom_mapper as mapper
import assistant.automation.browser.dom_filler as filler_mod
import assistant.automation.browser.dom_orchestrator as bdo


def _run(coro):
    return asyncio.run(coro)


def _elem(ref, **kw) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input", form_id="form-0", in_dialog=False,
    )
    base.update(kw)
    return bdom.ElementInfo(ref=ref, **base)


def _make_page(url="https://example.com/form"):
    page = MagicMock()
    page.url = url
    return page


def _make_tree(elements, url="https://example.com/form"):
    locs = {e.ref: AsyncMock() for e in elements}
    return bdom.PageDomTree(
        elements=elements,
        ref_to_locator=locs,
        url=url,
        read_at=time.monotonic(),
    )


class TestHappyPath(unittest.TestCase):
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_filler.fill_form", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_mapper.map_goal_to_fields", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.invalidate_tree_cache")
    def test_single_loop_success(self, mock_invalidate, mock_read, mock_map, mock_fill):
        elements = [
            _elem("r1", name="Name"),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        tree = _make_tree(elements)
        # No disabled elements → no dependent field re-perceive needed.
        # Post-submit: URL changed → navigation success.
        tree_clean = _make_tree(elements, url="https://example.com/thank-you")
        mock_read.side_effect = [tree, tree_clean]

        mock_map.return_value = mapper.FormMapping(
            fills=[mapper.FillInstruction(ref="r1", field_name="Name", value="Test")],
            submit_ref="s1",
            thinking="fill name and submit",
        )
        mock_fill.return_value = filler_mod.FormFillResult(
            fills=[filler_mod.FillResult(
                ref="r1", field_name="Name",
                intended_value="Test", observed_value="Test",
                succeeded=True,
            )],
            submit_clicked=False,
        )

        page = _make_page()
        result = _run(bdo.run_dom_form_fill("Fill name with Test", page))
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        mock_map.assert_called_once()


class TestValidationRecovery(unittest.TestCase):
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_filler.fill_form", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_mapper.map_goal_to_fields", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.invalidate_tree_cache")
    def test_validation_error_triggers_remap(self, mock_invalidate, mock_read, mock_map, mock_fill):
        elements = [
            _elem("r1", name="Email", type="email"),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        tree1 = _make_tree(elements)
        # After first submit: validation error
        tree_err = _make_tree(elements)
        tree_err.validation_errors = (
            bdom.ValidationError(field_ref="r1", message="Invalid email", source="aria-invalid"),
        )
        # After corrective submit: URL changed (navigation success)
        tree_clean = _make_tree(elements, url="https://example.com/thanks")
        # Initial perceive, post-submit verify (err),
        # corrective re-perceive, post-submit verify (clean)
        mock_read.side_effect = [tree1, tree_err, tree_err, tree_clean]

        mock_map.side_effect = [
            mapper.FormMapping(
                fills=[mapper.FillInstruction(ref="r1", field_name="Email", value="bad")],
                submit_ref="s1", thinking="fill email",
            ),
            mapper.FormMapping(
                fills=[mapper.FillInstruction(ref="r1", field_name="Email", value="real@email.com")],
                submit_ref="s1", thinking="fix email",
            ),
        ]
        mock_fill.side_effect = [
            filler_mod.FormFillResult(
                fills=[filler_mod.FillResult(
                    ref="r1", field_name="Email",
                    intended_value="bad", observed_value="bad", succeeded=True,
                )],
                submit_clicked=False,
            ),
            filler_mod.FormFillResult(
                fills=[filler_mod.FillResult(
                    ref="r1", field_name="Email",
                    intended_value="real@email.com", observed_value="real@email.com",
                    succeeded=True,
                )],
                submit_clicked=False,
            ),
        ]

        page = _make_page()
        result = _run(bdo.run_dom_form_fill("Fill email", page))
        self.assertTrue(result.success)
        self.assertEqual(mock_map.call_count, 2)


class TestEmptyMapping(unittest.TestCase):
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_mapper.map_goal_to_fields", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.invalidate_tree_cache")
    def test_empty_mapping_fails(self, mock_invalidate, mock_read, mock_map):
        elements = [_elem("r1", name="Name")]
        tree = _make_tree(elements)
        mock_read.return_value = tree
        mock_map.return_value = mapper.FormMapping(
            fills=[], submit_ref="", thinking="parse_error",
        )
        page = _make_page()
        result = _run(bdo.run_dom_form_fill("Fill name", page))
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "mapper_failed")


class TestPerceiveFailure(unittest.TestCase):
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.read_page_dom", new_callable=AsyncMock)
    def test_perceive_raises(self, mock_read):
        mock_read.side_effect = Exception("page crashed")
        page = _make_page()
        result = _run(bdo.run_dom_form_fill("Fill form", page))
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "perceive_failed")


class TestFillFailureReported(unittest.TestCase):
    """Bug 2: fill failures must NOT be reported as success."""

    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_filler.fill_form", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_mapper.map_goal_to_fields", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.invalidate_tree_cache")
    def test_fill_failure_not_false_success(self, mock_invalidate, mock_read, mock_map, mock_fill):
        elements = [
            _elem("r1", name="State", role="combobox", autocomplete="list"),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        tree = _make_tree(elements)
        mock_read.return_value = tree

        mock_map.return_value = mapper.FormMapping(
            fills=[
                mapper.FillInstruction(ref="r1", field_name="State", value="NCR"),
                mapper.FillInstruction(ref="s1", field_name="Submit", value="Delhi"),
            ],
            submit_ref="s1",
            thinking="fill state and city",
        )
        mock_fill.return_value = filler_mod.FormFillResult(
            fills=[
                filler_mod.FillResult(
                    ref="r1", field_name="State",
                    intended_value="NCR", observed_value="NCR",
                    succeeded=True,
                ),
                filler_mod.FillResult(
                    ref="s1", field_name="Submit",
                    intended_value="Delhi", observed_value="",
                    succeeded=False, error="unknown widget type: button",
                ),
            ],
            submit_clicked=False,
        )

        page = _make_page()
        result = _run(bdo.run_dom_form_fill("Set state to NCR and city to Delhi", page))
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "fills_failed")
        self.assertIn("Submit", result.final_summary)


class TestDependentFields(unittest.TestCase):
    """Bug 1: cascading fields — City enabled after State is filled."""

    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_filler.fill_form", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_mapper.map_goal_to_fields", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.invalidate_tree_cache")
    def test_newly_enabled_field_gets_second_pass(self, mock_invalidate, mock_read, mock_map, mock_fill):
        state_elem = _elem("r1", name="Select State", role="combobox", autocomplete="list")
        city_disabled = _elem("r2", name="Select City", role="combobox", autocomplete="list", enabled=False)
        city_enabled = _elem("r2b", name="Select City", role="combobox", autocomplete="list", enabled=True)
        submit = _elem("s1", role="button", name="Submit", tag="button")

        tree_initial = _make_tree([state_elem, city_disabled, submit])
        tree_after_state = _make_tree([state_elem, city_enabled, submit])
        tree_clean = _make_tree(
            [state_elem, city_enabled, submit],
            url="https://example.com/thanks",
        )
        mock_read.side_effect = [tree_initial, tree_after_state, tree_clean]

        mock_map.side_effect = [
            mapper.FormMapping(
                fills=[mapper.FillInstruction(ref="r1", field_name="Select State", value="NCR")],
                submit_ref="s1",
                thinking="fill state",
            ),
            mapper.FormMapping(
                fills=[
                    mapper.FillInstruction(ref="r1", field_name="Select State", value="NCR"),
                    mapper.FillInstruction(ref="r2b", field_name="Select City", value="Delhi"),
                ],
                submit_ref="s1",
                thinking="fill state and city",
            ),
        ]
        mock_fill.side_effect = [
            filler_mod.FormFillResult(
                fills=[filler_mod.FillResult(
                    ref="r1", field_name="Select State",
                    intended_value="NCR", observed_value="NCR",
                    succeeded=True,
                )],
                submit_clicked=False,
            ),
            filler_mod.FormFillResult(
                fills=[filler_mod.FillResult(
                    ref="r2b", field_name="Select City",
                    intended_value="Delhi", observed_value="Delhi",
                    succeeded=True,
                )],
                submit_clicked=False,
            ),
        ]

        page = _make_page()
        result = _run(bdo.run_dom_form_fill(
            "Set state to NCR and city to Delhi in this form", page,
        ))
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(mock_map.call_count, 2)
        self.assertEqual(mock_fill.call_count, 2)

    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_filler.fill_form", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom_mapper.map_goal_to_fields", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.read_page_dom", new_callable=AsyncMock)
    @patch("assistant.automation.browser.dom_orchestrator.browser_dom.invalidate_tree_cache")
    def test_no_disabled_elements_skips_reperceive(self, mock_invalidate, mock_read, mock_map, mock_fill):
        """When no elements are disabled, skip the dependent field check entirely."""
        elements = [
            _elem("r1", name="subjects", role="combobox", autocomplete="list"),
            _elem("s1", role="button", name="Submit", tag="button"),
        ]
        tree = _make_tree(elements)
        tree_clean = _make_tree(elements, url="https://example.com/thanks")
        # Only 2 reads: initial perceive + post-submit verify.
        # No re-perceive for dependent fields (none are disabled).
        mock_read.side_effect = [tree, tree_clean]

        mock_map.return_value = mapper.FormMapping(
            fills=[mapper.FillInstruction(ref="r1", field_name="subjects", value="Maths")],
            submit_ref="s1",
            thinking="fill subjects",
        )
        mock_fill.return_value = filler_mod.FormFillResult(
            fills=[filler_mod.FillResult(
                ref="r1", field_name="subjects",
                intended_value="Maths", observed_value="Maths",
                succeeded=True,
            )],
            submit_clicked=False,
        )

        page = _make_page()
        result = _run(bdo.run_dom_form_fill("Fill subjects with Maths", page))
        self.assertTrue(result.success)
        # Only 1 mapper call — no second pass triggered
        mock_map.assert_called_once()
        # Only 2 read_page_dom calls (initial + post-submit verify)
        self.assertEqual(mock_read.call_count, 2)


if __name__ == "__main__":
    unittest.main()
