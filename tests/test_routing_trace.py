"""`detect_backend` routes browser-content goals to vision, launches to native.

**Was a manual script, and it was hiding a real defect.** It sat in `tests/`
with a `def _main()` and an `if __name__ == "__main__"` block, so pytest
imported it, collected nothing, and reported EMPTY. Run by hand it printed
`1 FAILED` — and nobody saw it, because nothing ran it. The 2026-08-25 baseline
found six files in this shape; this is the one that was covering for something.

**The fixture was also doing unintended work.** Its window list was
`"DummyForms - TENKA Testing — Mozilla Firefox"`, and `_detect_running_app`
matches the goal's words against open window titles — so
`fill out the form ...` "found" Firefox by way of the substring **Form**s in
the title. Change the title to anything that does not echo the goal and the
case routes somewhere else entirely. One of the three passing cases was passing
on a coincidence.

With an honest title, two of the three route to `browser` rather than `vision`.
See `_BROWSER_CONTENT_XFAIL` below for what that means and why it is marked
rather than silently re-baselined.

Run with:  py -3.11 -m pytest tests/test_routing_trace.py -v
"""
import pathlib
import sys
from unittest.mock import patch

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.automation import router as da  # noqa: E402

# A window title that does NOT contain any word from the goals below. The
# original said "DummyForms", which `_detect_running_app` matched against the
# word "form" in the goal -- so the fixture answered the question the test was
# asking.
FIREFOX_OPEN = ["Test Page - Mozilla Firefox", "Antigravity", "cmd"]
CHROME_OPEN = ["YouTube - Google Chrome", "cmd"]
NO_BROWSER = ["Antigravity", "Notepad", "cmd"]

# Goals that route to `browser` today and, per this file's original intent,
# should route to `vision`.
#
# `_BROWSER_INTENT_PATTERNS` claims `fill (out) (the) form` and `submit form`,
# and the branch that answers "browser" consults `_detect_running_app(goal)` --
# which reads the *goal text* for an app name and never looks at what is
# actually open. So a form action typed while Firefox sits in front of you
# routes to `browser`, meaning Playwright opens its own window rather than
# acting on the page you are looking at.
#
# Marked `strict` deliberately: when the routing is fixed these XPASS, and a
# strict xfail turns an unexpected pass into a failure. That is the point --
# nobody has to remember to come back and delete this list.
_BROWSER_CONTENT_XFAIL = {
    "fill out the form with random test data",
    "submit form",
}

FIREFOX_CASES = [
    ("fill out the form with random test data", "vision", "form-fill in running Firefox"),
    ("fill this form for testing", "vision", "form-fill alt phrasing"),
    ("submit form", "vision", "submit form intent"),
]

LAUNCH_CASES_CHROME = [
    ("open chrome", "native", "regression: open running Chrome -> native focus"),
    ("launch chrome", "native", "regression: launch verb -> native"),
    ("open chrome and youtube.com", "browser", "URL pattern still wins -> browser/Playwright"),
]

GENERIC_CASES = [
    ("type 'café résumé 5' in notepad", "native", "in-app context pattern"),
    ("multiply 3 and 4 on calculator", "native", "in-app context pattern"),
    ("search for cats on google", "browser", "search intent"),
    ("click the back button", "unknown", "no candidate match anywhere"),
]


def _backend(goal, windows):
    with patch("assistant.io.screen.get_open_windows", return_value=windows):
        backend, meta = da.detect_backend(goal)
    return backend, meta


@pytest.mark.parametrize("goal,expected,note", FIREFOX_CASES,
                         ids=[c[0] for c in FIREFOX_CASES])
def test_browser_content_goes_to_vision(goal, expected, note, request):
    if goal in _BROWSER_CONTENT_XFAIL:
        request.node.add_marker(pytest.mark.xfail(
            strict=True,
            reason="routes to `browser` and opens a second window instead of "
                   "acting on the page in front of the user; "
                   "`_detect_running_app` reads the goal, never the open "
                   "windows",
        ))
    backend, meta = _backend(goal, FIREFOX_OPEN)
    assert backend == expected, f"{note}: reason={meta.get('reason')}"


@pytest.mark.parametrize("goal,expected,note", LAUNCH_CASES_CHROME,
                         ids=[c[0] for c in LAUNCH_CASES_CHROME])
def test_a_launch_goal_does_not_go_to_vision(goal, expected, note):
    """The control for the block above. Routing everything to vision would
    satisfy every case there and make the cheap backends dead code."""
    backend, meta = _backend(goal, CHROME_OPEN)
    assert backend == expected, f"{note}: reason={meta.get('reason')}"


@pytest.mark.parametrize("goal,expected,note", GENERIC_CASES,
                         ids=[c[0] for c in GENERIC_CASES])
def test_ordinary_goals_are_unchanged(goal, expected, note):
    backend, meta = _backend(goal, NO_BROWSER)
    assert backend == expected, f"{note}: reason={meta.get('reason')}"


def test_the_fixture_title_does_not_echo_any_goal():
    """The accident this file was built on, pinned so it cannot come back.

    `_detect_running_app` matches goal words against open window titles. A
    fixture whose title contains a word from the goal answers the question the
    test is asking, and the test then passes whatever the router does.
    """
    # Only the browser-content cases, and only against their own fixture.
    # `type 'café résumé 5' in notepad` legitimately shares a word with a
    # `Notepad` window -- the goal names the app, the router is meant to match
    # it, and that case expects `native` for exactly that reason. The accident
    # is a title echoing a goal that names *no* app, which is what the three
    # FIREFOX_CASES are.
    titles = " ".join(FIREFOX_OPEN).lower()
    _NOISE = {"with", "this", "that", "test", "data", "random", "testing",
              "form", "out", "the", "for"}

    for goal, _, _ in FIREFOX_CASES:
        for word in goal.lower().split():
            if len(word) < 4 or word in _NOISE:
                continue
            assert word not in titles, (
                f"the fixture window titles contain {word!r} from the goal "
                f"{goal!r} -- `_detect_running_app` will match it and the "
                f"test will pass on the fixture rather than on the routing")

    # And the specific one that was wrong: the original title was
    # "DummyForms - TENKA Testing — Mozilla Firefox", and "form" is the noun
    # every one of these goals is about.
    assert "form" not in titles, (
        "the Firefox fixture title contains 'form', which is the word every "
        "browser-content goal here shares -- this is the original accident")
