"""
test_summary_goal_priority.py — Bug 9: goal-aware error prioritization.

When the user parameterizes a specific field in the goal text
("fill the form with mobile as 99999"), the form's rejection summary
should mention THAT field first — even if it's not the first field in
DOM order. The personality LLM downstream uses the summary to craft a
TTS reply; surfacing the user-supplied bad-value field gives it the
right anchor.

Coverage:
  - Direct token match: goal "email" + field "Work Email" → email first.
  - Synonym alias: goal "mobile" + field "Contact Number" → contact first.
  - Multiple aliases: goal "phone" + field "Contact Number" → match.
  - No goal token in any field name → fall back to DOM order (no regression).
  - Stable order within match groups (preserves DOM order for tiebreak).
  - Empty goal → no prioritization (no regression).
  - 120-char TTS budget preserved with 5-field summary.
  - Helper-level alias expansion test.

Run: python test_summary_goal_priority.py
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

import assistant.automation.browser.dom as bdom
import assistant.automation.browser.dom_orchestrator as bdo


# ─── Helpers ─────────────────────────────────────────────────────────────


def _elem(ref, **kwargs) -> bdom.ElementInfo:
    base = dict(
        role="textbox", name="Field", placeholder="", value="",
        options=(), bounds=(0, 0, 200, 30), visible=True, enabled=True,
        type="text", tag="input", form_id="", in_dialog=False,
        aria_invalid=False,
    )
    base.update(kwargs)
    return bdom.ElementInfo(ref=ref, **base)


# ─── Alias expansion (helper-level) ──────────────────────────────────────


class TestExpandGoalTokens(unittest.TestCase):
    def test_mobile_pulls_in_phone_contact_tel(self):
        toks = bdo._expand_goal_tokens("fill the form with mobile as 99999")
        self.assertIn("mobile", toks)
        self.assertIn("phone", toks)
        self.assertIn("contact", toks)
        self.assertIn("tel", toks)

    def test_email_pulls_in_mail(self):
        toks = bdo._expand_goal_tokens("set email to test@x.com")
        self.assertIn("email", toks)
        self.assertIn("mail", toks)

    def test_no_alias_keyword_no_expansion(self):
        toks = bdo._expand_goal_tokens("fill the booking widget")
        # "fill" / "the" / "booking" / "widget" don't intersect any alias
        # set, so the expansion equals the raw token set.
        self.assertIn("fill", toks)
        self.assertIn("booking", toks)
        self.assertNotIn("mobile", toks)

    def test_short_tokens_dropped(self):
        toks = bdo._expand_goal_tokens("a b cd email")
        self.assertNotIn("a", toks)
        self.assertNotIn("cd", toks)
        self.assertIn("email", toks)

    def test_punctuation_stripped(self):
        toks = bdo._expand_goal_tokens("set the email, please.")
        self.assertIn("email", toks)
        # No "email," with the comma attached.
        self.assertNotIn("email,", toks)

    def test_empty_goal(self):
        self.assertEqual(bdo._expand_goal_tokens(""), frozenset())
        self.assertEqual(bdo._expand_goal_tokens("   "), frozenset())


class TestFieldMatchesGoal(unittest.TestCase):
    def test_direct_match(self):
        toks = frozenset({"mobile", "phone", "contact"})
        self.assertTrue(bdo._field_matches_goal("Contact Number", toks))

    def test_no_match(self):
        toks = frozenset({"mobile", "phone"})
        self.assertFalse(bdo._field_matches_goal("First Name", toks))

    def test_short_field_token_ignored(self):
        # "Last Name" — "last" is 4 chars, OK. "Name" is 4 chars, OK.
        toks = frozenset({"name", "last"})
        self.assertTrue(bdo._field_matches_goal("Last Name", toks))

    def test_empty_inputs(self):
        self.assertFalse(bdo._field_matches_goal("", frozenset({"x"})))
        self.assertFalse(bdo._field_matches_goal("Field", frozenset()))


# ─── Goal-aware prioritization in summary ────────────────────────────────


class TestSummaryGoalPriority(unittest.TestCase):
    def test_direct_token_match_promoted(self):
        # Goal mentions "email"; field "Work Email" should appear first
        # even though "First name" is in DOM order earlier.
        elements = [
            _elem("r1", name="First name"),
            _elem("r2", name="Last name"),
            _elem("r3", name="Work Email"),
        ]
        errs = (
            bdom.ValidationError("r1", "Required", "describedby"),
            bdom.ValidationError("r2", "Required", "describedby"),
            bdom.ValidationError("r3", "Bad email", "describedby"),
        )
        s = bdo._format_unresolved_for_summary(
            errs, elements, goal="set email to test@x.com",
        )
        # 'Work Email' should appear before 'First name' in the summary.
        self.assertLess(s.index("Work Email"), s.index("First name"))

    def test_synonym_match_promoted(self):
        # Goal says "mobile"; field is "Contact Number". Without alias map,
        # there'd be no overlap. Bug 9: expand "mobile" → also matches
        # "contact" as a token — so "Contact Number" gets promoted.
        elements = [
            _elem("r1", name="First name"),
            _elem("r2", name="Last name"),
            _elem("r3", name="Company Name"),
            _elem("r4", name="Work Email"),
            _elem("r5", name="Contact Number"),
        ]
        errs = tuple(
            bdom.ValidationError(f"r{i}", "Bad", "describedby")
            for i in (1, 2, 3, 4, 5)
        )
        s = bdo._format_unresolved_for_summary(
            errs, elements,
            goal="fill this form with mobile number as 99999",
        )
        # Contact Number must appear FIRST in the summary.
        self.assertIn("Contact Number", s)
        self.assertLess(s.index("Contact Number"), s.index("First name"))

    def test_phone_alias_matches_contact(self):
        elements = [
            _elem("r1", name="First name"),
            _elem("r2", name="Contact Number"),
        ]
        errs = (
            bdom.ValidationError("r1", "Bad", "describedby"),
            bdom.ValidationError("r2", "Bad", "describedby"),
        )
        s = bdo._format_unresolved_for_summary(
            errs, elements, goal="enter phone as 1234",
        )
        self.assertLess(s.index("Contact Number"), s.index("First name"))

    def test_no_match_preserves_dom_order(self):
        # Goal has no field-related tokens; summary keeps DOM order.
        elements = [
            _elem("r1", name="Alpha"),
            _elem("r2", name="Beta"),
            _elem("r3", name="Gamma"),
        ]
        errs = (
            bdom.ValidationError("r1", "x", "x"),
            bdom.ValidationError("r2", "x", "x"),
            bdom.ValidationError("r3", "x", "x"),
        )
        s = bdo._format_unresolved_for_summary(
            errs, elements, goal="fill the form please",
        )
        self.assertLess(s.index("Alpha"), s.index("Beta"))
        self.assertLess(s.index("Beta"), s.index("Gamma"))

    def test_empty_goal_preserves_dom_order(self):
        elements = [
            _elem("r1", name="Email"),
            _elem("r2", name="First"),
        ]
        errs = (
            bdom.ValidationError("r1", "x", "x"),
            bdom.ValidationError("r2", "x", "x"),
        )
        s_no_goal = bdo._format_unresolved_for_summary(errs, elements)
        # Without a goal, no prioritization — DOM order preserved.
        self.assertLess(s_no_goal.index("Email"), s_no_goal.index("First"))

    def test_stable_within_match_group(self):
        # Two fields both match the goal → relative DOM order kept.
        elements = [
            _elem("r1", name="First name"),
            _elem("r2", name="Last name"),
        ]
        errs = (
            bdom.ValidationError("r1", "x", "x"),
            bdom.ValidationError("r2", "x", "x"),
        )
        s = bdo._format_unresolved_for_summary(
            errs, elements, goal="set the name fields",
        )
        # Both match "name"; first → second preserved.
        self.assertLess(s.index("First name"), s.index("Last name"))


# ─── TTS hygiene budget ─────────────────────────────────────────────────


class TestTtsHygiene(unittest.TestCase):
    def test_five_field_summary_under_120_chars(self):
        # Bumping the cap to 5 must still respect feedback_tts_hygiene
        # (<120 chars). Use realistic Truein-shaped names.
        elements = [
            _elem("r1", name="First name"),
            _elem("r2", name="Last Name"),
            _elem("r3", name="Company Name"),
            _elem("r4", name="Work Email"),
            _elem("r5", name="Contact Number"),
        ]
        errs = tuple(
            bdom.ValidationError(f"r{i}", "x", "x")
            for i in range(1, 6)
        )
        s = bdo._format_unresolved_for_summary(errs, elements)
        self.assertLess(
            len(s), 120,
            msg=f"summary exceeds TTS budget ({len(s)} chars): {s!r}",
        )

    def test_realistic_truein_summary_with_goal(self):
        # End-to-end shape: goal mentions mobile, 5 errors, summary should
        # lead with Contact Number AND fit budget.
        elements = [
            _elem("r1", name="First name"),
            _elem("r2", name="Last Name"),
            _elem("r3", name="Company Name"),
            _elem("r4", name="Work Email"),
            _elem("r5", name="Contact Number"),
        ]
        errs = tuple(
            bdom.ValidationError(f"r{i}", "x", "x")
            for i in range(1, 6)
        )
        s = bdo._format_unresolved_for_summary(
            errs, elements,
            goal="fill this form with testing values with mobile as 99999",
        )
        self.assertLess(len(s), 120)
        # Must start with Contact Number.
        self.assertIn("'Contact Number'", s)
        first_field_idx = s.index("'Contact Number'")
        for other in ("'First name'", "'Last Name'", "'Company Name'", "'Work Email'"):
            if other in s:
                self.assertLess(first_field_idx, s.index(other))


if __name__ == "__main__":
    unittest.main(verbosity=2)
