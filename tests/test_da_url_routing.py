"""URL detection must not mis-classify an email address as a URL.

The regression that produced it: `type test.dev20154@gmail.com in the
email field` matched the URL pattern, so a goal about typing into a form
was routed as a navigation.

**Was a manual script.** It sat in `tests/` with a `def run()` and an
`if __name__ == "__main__"` block, so pytest imported it, collected nothing,
and reported EMPTY -- a file that looks like coverage and asserts nothing. The
2026-08-25 baseline found six of these. The cases below are the originals,
unchanged; only the harness is new.

It could not even be run by hand from the repo root: no `sys.path` setup, so
`from assistant...` raised ModuleNotFoundError. It had not run anywhere, in any
form, for a while.

Run with:  py -3.11 -m pytest tests/test_da_url_routing.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CASES = [
    # --- Emails must NOT match ---
    ("type test.dev20154@gmail.com in the email field", False, "regression: email with .dev local + gmail.com"),
    ("type alice@example.io in the email field", False, "email with .io domain"),
    ("send to john.doe@example.com please", False, "email with .com"),
    ("Type 'alice@example.one' in the email field", False, ".one not in TLD list anyway"),
    ("fill the form with user@company.org", False, "email with .org"),

    # --- Real URLs must still match ---
    ("open chrome and youtube.com", True, "bare TLD URL"),
    ("go to https://example.com/path", True, "https URL"),
    ("visit www.bbc.com/news", True, "www. URL"),
    ("open google.com", True, "bare .com"),
    ("check test.dev tomorrow", True, "bare .dev domain (no @)"),

    # --- Goals with neither ---
    ("type 'café résumé 5'", False, "quoted literal text"),
    ("type my email in the form", False, "instruction without URL"),
    ("multiply 3 and 4 on calculator", False, "no URL"),
]



@pytest.mark.parametrize("goal,expected,note", CASES)
def test_url_pattern_matches_only_real_urls(goal, expected, note):
    from assistant.automation.router import _URL_PATTERN

    match = _URL_PATTERN.search(goal)
    assert bool(match) == expected, (
        f"{note}: expected match={expected}, got {match.group(0)!r}"
        if match else f"{note}: expected match={expected}, got no match")


def test_the_case_table_is_not_empty():
    """A parametrize over an empty list is a file that passes by not running."""
    assert len(CASES) >= 13
    assert any(e for _, e, _ in CASES), "no positive case -- only absences"
    assert any(not e for _, e, _ in CASES), "no negative case -- only presences"
