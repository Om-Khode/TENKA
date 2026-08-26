"""`_extract_target_app` finds the app in an `in/on/with/using <app>` suffix
and strips it from the goal.

Regression test for the `type X in notepad -> typed into the IDE` bug: the
target was not extracted, so the text went to whatever had focus.

**Was a manual script.** It sat in `tests/` with a `def run()` and an
`if __name__ == "__main__"` block, so pytest imported it, collected nothing,
and reported EMPTY -- a file that looks like coverage and asserts nothing. The
2026-08-25 baseline found six of these. The cases below are the originals,
unchanged; only the harness is new.

It could not even be run by hand from the repo root: no `sys.path` setup, so
`from assistant...` raised ModuleNotFoundError. It had not run anywhere, in any
form, for a while.

Run with:  py -3.11 -m pytest tests/test_da_target_app.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CASES = [
    # (goal, expected_target, expected_stripped)
    # ── Real targets that should be detected and stripped ──
    ("type café résumé 5 in notepad", "notepad", "type café résumé 5"),
    ("Type 'café résumé 5' in notepad", "notepad", "Type 'café résumé 5'"),
    ("write a poem in word", "word", "write a poem"),
    ("multiply 3 and 4 on calculator", "calculator", "multiply 3 and 4"),
    ("play music with spotify", "spotify", "play music"),
    ("search using chrome", "chrome", "search"),

    # ── Stop-words at end should NOT be treated as apps ──
    # (prevents noun-collisions like "in the form / in the field / in the input")
    ("type my email in the form", None, "type my email in the form"),
    ("click in the email field", None, "click in the email field"),
    ("paste it into the textbox", None, "paste it into the textbox"),
    ("search in the box", None, "search in the box"),

    # ── 2-letter or stop-word noun should not match ──
    ("type X in it", None, "type X in it"),
    ("write that in mode", None, "write that in mode"),

    # ── Goals without 'in/on/with/using' suffix at end ──
    ("type café résumé 5", None, "type café résumé 5"),
    ("open notepad", None, "open notepad"),

    # ── 'in' at end of a quoted phrase should not strip ──
    # The regex requires alpha word + END, so 'marketing"' has trailing quote.
    ("type 'I work in marketing'", None, "type 'I work in marketing'"),
]




@pytest.mark.parametrize("goal,expected_target,expected_stripped", CASES)
def test_target_app_is_extracted_and_stripped(goal, expected_target,
                                              expected_stripped):
    from assistant.automation.router import _extract_target_app

    target, stripped = _extract_target_app(goal)
    assert target == expected_target, f"target for {goal!r}"
    assert stripped == expected_stripped, f"stripped goal for {goal!r}"


def test_the_case_table_covers_both_directions():
    """Both halves matter: an extractor that found nothing would pass every
    `None` case, and one that matched everything would pass every named one."""
    assert len(CASES) >= 15
    assert any(t for _, t, _ in CASES), "no case expects an app to be found"
    assert any(t is None for _, t, _ in CASES), "no case expects no app"
