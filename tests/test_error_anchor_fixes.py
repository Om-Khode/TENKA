"""
test_error_anchor_fixes.py — Bug 11 (2026-04-28): error-anchor accuracy.

Live-test 2026-04-28 21:28 surfaced three sub-bugs in the validation-
error anchoring layer of `browser_dom.py`:

  A. Wrong anchor: the "Please enter a valid phone number." error anchored
     to "First name" instead of "Contact Number". The DOM proximity walker
     used `cur.querySelector('[data-tenka-idx]')` which returns the FIRST
     descendant in DOM order — so when the error sits at form-level (below
     the submit button, no nearby input sibling), the walker arbitrarily
     picked the topmost form field.

  B. Duplicate detection: the same error message + same anchor appeared
     TWICE in `validation_errors`. The `[class*="error"]` selector matched
     both an outer wrapper (e.g. `class="error-msg-wrapper"`) AND a nested
     inner element (`class="error-text"`); both contained the same text.
     `seenErrEls` only deduped by DOM identity, not by (anchor, message).

  C. No fallback: when DOM proximity hits an ambiguous ancestor (multiple
     captures = form-level error), there was no semantic backup.

Coverage in this file (all string-level / hydration-level — JS runtime
tests live in test_browser_dom_integration.py with a real Chromium):
  - JS source asserts: ambiguous-ancestor branch, single-capture branch,
    text-match alias map present, dedup loop present, dedupedErrors
    returned (not raw validationErrors).
  - Python hydration: 'text-match' source tag round-trips through
    read_page_dom and shows up on the ValidationError.
  - Python hydration: deduplicated errors arrive de-duped (the JS dedup
    is upstream — but if a buggy JS pass returned dupes, the hydration
    layer should still surface them; this is a contract guard).

Run: python test_error_anchor_fixes.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.browser_dom as bdom


# ─── JS source-level contracts ───────────────────────────────────────────


class TestJsAnchorContracts(unittest.TestCase):
    """
    String-level guards that lock in the anchor-resolution structure in
    `_DOM_QUERY_JS`. If a future PR rewrites the walker without preserving
    these properties, these assertions catch it.
    """

    def setUp(self):
        self.js = bdom._DOM_QUERY_JS

    def test_proximity_walker_handles_single_capture_branch(self):
        # Walker must distinguish single-capture (trustworthy: error is
        # right next to its input) from multi-capture (ambiguous: form-
        # level error). Lookup pattern proves the branch exists.
        self.assertIn(
            "captures.length === 1", self.js,
            msg="single-capture trustworthy branch missing from walker",
        )

    def test_proximity_walker_breaks_on_ambiguous_ancestor(self):
        # Multi-capture branch must `break` out of the walk — otherwise
        # the walker would keep going and eventually grab some other
        # arbitrary input. Bug 11.A regression guard.
        self.assertIn(
            "captures.length > 1", self.js,
            msg="multi-capture (ambiguous) branch missing from walker",
        )
        # Confirm the branch breaks the walk-up loop.
        idx = self.js.find("captures.length > 1")
        self.assertGreater(idx, -1)
        next_50 = self.js[idx:idx + 200]
        self.assertIn(
            "break", next_50,
            msg="ambiguous-ancestor branch must `break` out of walker",
        )

    def test_text_match_fallback_present(self):
        # Text-match section must exist. Anchor by either the section
        # header comment or the source tag string — both are stable.
        self.assertIn(
            "'text-match'", self.js,
            msg="text-match source tag missing — fallback removed?",
        )

    def test_text_match_alias_map_covers_phone_and_email(self):
        # The alias map mirrors Python's _FIELD_ALIASES. Spot-check the
        # two most common groups (phone family, email family) so a
        # PR can't silently drop them.
        self.assertIn("'mobile'", self.js)
        self.assertIn("'phone'", self.js)
        self.assertIn("'contact'", self.js)
        self.assertIn("'tel'", self.js)
        self.assertIn("'email'", self.js)
        self.assertIn("'mail'", self.js)

    def test_text_match_includes_number_alias_for_phone_family(self):
        # The actual bug: "Please enter a valid phone number." → "Contact
        # Number" requires `number` to alias-equate with the phone family
        # (since the field name is "Contact Number" but the error said
        # "phone number"). Lock this in.
        self.assertIn("'number'", self.js)
        # And it must be in a group that contains 'phone' (so the alias
        # expansion ties them together). Check by carving the group line.
        # Find the line containing 'mobile' — it should also contain
        # 'phone' and 'number' since Bug 11 puts them all in one group.
        for line in self.js.split("\n"):
            if "'mobile'" in line and "'phone'" in line:
                self.assertIn(
                    "'number'", line,
                    msg=(
                        "phone-family alias group must include 'number' "
                        "so 'Please enter a valid phone number.' aliases "
                        "to a 'Contact Number' field via the 'number' "
                        "token (Bug 11.B)"
                    ),
                )
                break
        else:
            self.fail("phone-family alias group not found")

    def test_dedupe_loop_present(self):
        # Bug 11.C: dedupe by (field_idx, message). The dedup loop must
        # exist after the main error collection.
        self.assertIn("dedupedErrors", self.js)
        self.assertIn("seenSigs", self.js)
        # And the return statement must use dedupedErrors, NOT raw.
        ret_idx = self.js.rfind("validation_errors:")
        self.assertGreater(ret_idx, -1)
        ret_block = self.js[ret_idx:ret_idx + 80]
        self.assertIn(
            "dedupedErrors", ret_block,
            msg="JS return must surface dedupedErrors (not raw)",
        )


# ─── Python hydration: text-match source tag ─────────────────────────────


class TestTextMatchHydration(unittest.IsolatedAsyncioTestCase):
    """
    The Python hydration layer must round-trip the new 'text-match'
    source tag without dropping or relabeling it. Stub a fabricated JS
    payload (mimicking what the patched JS would emit for the Truein
    case) and verify the ValidationError exposes the tag.
    """

    async def asyncSetUp(self):
        bdom.reset_state_for_test()

    async def asyncTearDown(self):
        bdom.reset_state_for_test()

    def _stub_page(self, payload: dict):
        page = MagicMock(name="page")
        page.evaluate = AsyncMock(return_value=payload)
        page.locator = MagicMock(side_effect=lambda sel: MagicMock(name=f"loc({sel})"))
        return page

    async def test_text_match_source_round_trips(self):
        # Mimic what the patched JS emits for a phone-shaped error
        # anchored by text-match (not DOM proximity).
        payload = {
            "elements": [
                {
                    "idx": 0, "tag": "input", "role": "textbox",
                    "name": "First name", "placeholder": "", "value": "Test",
                    "options": [], "bounds": [10, 10, 100, 30],
                    "visible": True, "enabled": True, "type": "text",
                    "form_id": "form-0", "in_dialog": False,
                    "aria_invalid": False,
                },
                {
                    "idx": 1, "tag": "input", "role": "textbox",
                    "name": "Contact Number", "placeholder": "", "value": "99999",
                    "options": [], "bounds": [10, 50, 100, 30],
                    "visible": True, "enabled": True, "type": "tel",
                    "form_id": "form-0", "in_dialog": False,
                    "aria_invalid": False,
                },
            ],
            "viewport": [1280, 800],
            "validation_errors": [
                {
                    "field_idx": 1,  # text-match anchored to Contact Number
                    "message": "Please enter a valid phone number.",
                    "source": "text-match",
                },
            ],
        }
        tree = await bdom.read_page_dom(self._stub_page(payload), use_cache=False)

        self.assertEqual(len(tree.validation_errors), 1)
        ve = tree.validation_errors[0]
        self.assertEqual(ve.source, "text-match")
        self.assertEqual(ve.field_ref, tree.elements[1].ref,
                         msg="error must anchor to Contact Number, not First name")
        self.assertIn("phone number", ve.message.lower())

    async def test_no_duplicate_errors_passthrough(self):
        # If a buggy JS pass were to leak duplicate (field_idx, message)
        # entries past its dedup, the Python hydration layer still
        # passes them through (dedup is upstream). This test documents
        # that contract — it's a regression guard, not a behavior assert.
        payload = {
            "elements": [
                {
                    "idx": 0, "tag": "input", "role": "textbox",
                    "name": "First name", "placeholder": "", "value": "",
                    "options": [], "bounds": [0, 0, 100, 30],
                    "visible": True, "enabled": True, "type": "text",
                    "form_id": "", "in_dialog": False,
                    "aria_invalid": False,
                },
            ],
            "viewport": [1280, 800],
            "validation_errors": [
                {"field_idx": 0, "message": "Bad", "source": "error-class"},
                {"field_idx": 0, "message": "Bad", "source": "error-class"},
            ],
        }
        tree = await bdom.read_page_dom(self._stub_page(payload), use_cache=False)
        # Hydration passes them through — JS-side dedup is the source of
        # truth. Both arrive.
        self.assertEqual(len(tree.validation_errors), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
