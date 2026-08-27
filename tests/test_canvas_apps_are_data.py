"""A regex may not name a brand. THE rule, applied to canvas routing.

`automation/router.py` decided that a goal needs the vision tier by matching
nine product names inside a regular expression:

    _CANVAS_APP_RE = re.compile(
        r'\\b(figma|miro|whiteboard|canvas|excalidraw|flutter|tldraw|'
        r'google\\s+slides|google\\s+docs|google\\s+drawings|sketch|'
        r'draw|paint)\\b', re.IGNORECASE)

`CLAUDE.md`'s second bullet under THE rule is "❌ A regex that mentions a brand
name", and the prescribed fix is the one taken here: lift the specific
behaviour into data. The *behaviour* was right -- an app whose document is a
canvas exposes no accessible tree worth reading, so a step aimed at one has to
go to vision or it will confidently click on nothing. Only the location was
wrong. That is a fact about an app, and facts about apps live in
`core/known_apps.py`.

The generic words stayed in an expression, deliberately: "whiteboard",
"canvas", "draw" and "paint" mean the same thing whoever makes the app, and
turning them into rows would be data-modelling the English language.

What this buys: the tenth canvas app is a row, not an edit to a routing
expression -- and an app taught at runtime is found the same way a built-in
one is.

Run with:  py -3.11 -m pytest tests/test_canvas_apps_are_data.py -v
"""
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.known_apps import KNOWN_APPS, get_category  # noqa: E402

_ROUTER_PY = _ROOT / "assistant" / "automation" / "router.py"

# Brand names that were in the regex. Not an exhaustive list of canvas apps --
# the point is that these particular strings must no longer appear in one.
_MOVED = ("figma", "miro", "excalidraw", "tldraw", "sketch",
          "google slides", "google drawings", "flutter")


# ─── the rule ────────────────────────────────────────────────────────────────

def _compiled_patterns(path):
    """Every string literal handed to `re.compile` in `path`.

    By AST, and that matters. The first version of this check scanned the raw
    source with a regex of its own and reported two false positives: it saw
    "sketch" inside the generic word "sketching", and "google slides" inside
    the *docstring* explaining why the brands had been moved out. A sweep that
    reads its own prose is the trap this project keeps hitting.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "compile"):
            continue
        for arg in ast.walk(node):
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.append(arg.value)
    return found


@pytest.mark.parametrize("brand", _MOVED)
def test_no_brand_name_survives_in_the_routing_expression(brand):
    """THE rule, checked against the source rather than trusted."""
    patterns = _compiled_patterns(_ROUTER_PY)
    assert patterns, "walked nothing -- router.py compiles no expressions"

    word = re.compile(r"\b" + re.escape(brand) + r"\b", re.IGNORECASE)
    offenders = [pat for pat in patterns if word.search(pat)]
    assert not offenders, (
        f"{brand!r} is back inside a compiled expression: {offenders}")


def test_the_brand_detector_can_see_a_brand():
    """Positive control for the test above, which asserts an absence. A broken
    walk finds nothing either, and there is deliberately no brand left in
    router.py to prove it against."""
    import ast
    import tempfile

    src = "import re" + chr(10) + "X = re.compile(r'figma|whiteboard')" + chr(10)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        tmp = pathlib.Path(fh.name)
    try:
        patterns = _compiled_patterns(tmp)
    finally:
        tmp.unlink()

    assert any("figma" in pat for pat in patterns), (
        "the extractor cannot see a brand that is right there")


@pytest.mark.parametrize("brand", _MOVED)
def test_each_brand_is_a_row_instead(brand):
    """Moved, not deleted. A test that only checked the regex would pass just
    as well if the behaviour had been dropped on the floor."""
    assert get_category(brand) == "canvas_app", (
        f"{brand!r} left the regex without arriving in the app table")


def test_the_generic_words_stayed_generic():
    """The other direction. "whiteboard" and "draw" are not brands, and making
    rows of them would be data-modelling English."""
    from assistant.automation.router import _CANVAS_INTENT_RE

    for word in ("whiteboard", "canvas", "draw", "paint"):
        assert _CANVAS_INTENT_RE.search(f"open the {word}"), word
        assert get_category(word) is None, (
            f"{word!r} became an app row; it is a description, not a product")


# ─── the behaviour it replaced ───────────────────────────────────────────────

@pytest.mark.parametrize("goal", [
    "open figma and move the rectangle",
    "design something in miro",
    "open google slides and add a title",
    "sketch a logo",
    "use excalidraw",
])
def test_a_canvas_app_still_routes_to_vision(goal):
    from assistant.automation.router import _names_a_canvas_app
    from assistant.automation.router import _CANVAS_INTENT_RE

    assert _names_a_canvas_app(goal) or _CANVAS_INTENT_RE.search(goal), goal


@pytest.mark.parametrize("goal", [
    "open notepad and type hello",
    "play something on spotify",
    "open chrome",
    "send a message on whatsapp",
])
def test_an_ordinary_app_does_not(goal):
    """**The direction that makes this useless if broken.** Routing everything
    to vision would satisfy every test above and make the cheap tiers dead
    code."""
    from assistant.automation.router import _names_a_canvas_app
    from assistant.automation.router import _CANVAS_INTENT_RE

    assert not _names_a_canvas_app(goal), goal
    assert not _CANVAS_INTENT_RE.search(goal), goal


def test_a_longer_app_name_wins_over_a_shorter_one():
    """"google slides" must not lose to some shorter entry that happens to be a
    prefix or a substring of it. Longest-first is the ordering, and without it
    the answer depends on dict insertion order."""
    from assistant.automation.router import _names_a_canvas_app

    assert _names_a_canvas_app("open google slides")


def test_an_alias_resolves_the_same_way():
    """Aliases are how the table absorbs the way people actually speak, and a
    lookup that only matched canonical names would send "gslides" to a tier
    that cannot see it."""
    from assistant.automation.router import _names_a_canvas_app

    assert _names_a_canvas_app("open gslides")


def test_the_lookup_walks_a_non_empty_table():
    """Anti-vacuity: `_names_a_canvas_app` returning False for everything would
    pass every negative test above."""
    canvas = [n for n, e in KNOWN_APPS.items() if e.category == "canvas_app"]
    assert len(canvas) >= 8, (
        f"only {len(canvas)} canvas rows -- the move lost some")


def test_a_new_canvas_app_needs_no_code_change():
    """The claim THE rule makes about its own fix, exercised. If adding the
    tenth app still required touching a regex, the change would have been
    cosmetic."""
    from assistant.core import known_apps
    from assistant.automation import router

    before = router._names_a_canvas_app("open plotwright")
    known_apps.KNOWN_APPS["plotwright"] = known_apps.AppEntry("canvas_app", [])
    try:
        after = router._names_a_canvas_app("open plotwright")
    finally:
        known_apps.KNOWN_APPS.pop("plotwright", None)

    assert not before and after, (
        "a new canvas app was not picked up from the table alone")


# ─── the routing decision itself, not just its two predicates ────────────────
#
# **From a green mutant.** Deleting `or _names_a_canvas_app(goal)` from the
# routing call site changed no test result: everything above exercises the two
# predicates separately and nothing exercised the decision they feed. A move
# that is correct in both halves and unwired is still a regression.

class _FakeCdpState:
    """CDP up, so vision has to win on the canvas rule rather than by default.

    Shape copied from `tests/test_browser_routing.py`, whose canvas cases all
    say "draw in figma" -- the generic word carries every one of them, which is
    exactly why deleting the table lookup from the routing call site changed no
    result there.
    """

    def __init__(self, available=True):
        self.available = available


def _mode(goal, monkeypatch, cdp_up=True):
    from assistant.automation import router
    from assistant import config

    monkeypatch.setattr(config, "BROWSER_DOM_MODE_ENABLED", True,
                        raising=False)
    return router._choose_browser_mode(goal, _FakeCdpState(cdp_up))[0]


@pytest.mark.parametrize("goal", [
    "open figma and move the rectangle",
    "design something in miro",
    "open gslides and add a title",
    "open figma in chrome",
])
def test_a_canvas_goal_routes_to_vision(goal, monkeypatch):
    """The whole decision, with CDP up -- vision must win anyway, because a
    canvas element is one opaque node whatever the DOM tier can reach."""
    assert _mode(goal, monkeypatch) == "vision", goal


@pytest.mark.parametrize("goal", [
    "search for flights on chrome",
    "fill in the login form",
    "open notepad and type hello",
])
def test_an_ordinary_goal_does_not_route_to_vision(goal, monkeypatch):
    """The control. Routing everything to vision satisfies the test above and
    makes the cheap tiers dead code."""
    assert _mode(goal, monkeypatch) != "vision", goal


def test_a_second_app_in_the_sentence_does_not_hide_the_canvas(monkeypatch):
    """**A real bug this found.** The first implementation stopped at the
    longest app name it recognised and answered about that one, so "open figma
    in chrome" resolved on `chrome` -- longer than `figma` -- and routed away
    from the only tier that can see a canvas. A goal naming two apps needs the
    tier that handles the harder of them."""
    from assistant.automation.router import _names_a_canvas_app

    assert _names_a_canvas_app("open figma in chrome")
    assert _names_a_canvas_app("open chrome and then figma")
    assert not _names_a_canvas_app("open chrome and then notepad")


# ─── the derived lookup follows the table ────────────────────────────────────

def test_get_category_sees_an_app_added_at_runtime():
    """`_APP_LOOKUP` was built once at import, which contradicted the promise at
    the top of `known_apps.py`: *adding a new app requires one row in
    KNOWN_APPS, no code changes elsewhere*. True at edit time, false at
    runtime -- and THE rule says apps are "discovered, learned, or taught at
    runtime", which a lookup frozen at import cannot support.

    Tested directly rather than through the canvas router, which reads
    `KNOWN_APPS` itself and would pass either way -- a mutation restoring the
    freeze went green until this existed.
    """
    from assistant.core import known_apps

    assert known_apps.get_category("plotwright") is None
    known_apps.KNOWN_APPS["plotwright"] = known_apps.AppEntry(
        "canvas_app", ["plotw"])
    try:
        assert known_apps.get_category("plotwright") == "canvas_app"
        assert known_apps.get_category("plotw") == "canvas_app", (
            "the rebuilt lookup dropped the aliases")
        assert known_apps.resolve_app("plotw") == ("plotwright", "canvas_app")
    finally:
        known_apps.KNOWN_APPS.pop("plotwright", None)

    assert known_apps.get_category("plotwright") is None, (
        "the lookup kept a row that is no longer in the table")


def test_the_lookup_still_answers_for_the_built_in_rows():
    """The control. A lookup that rebuilt into an empty dict would satisfy the
    removal half of the test above."""
    from assistant.core.known_apps import get_category

    assert get_category("spotify") == "music_app"
    assert get_category("chrome") == "browser"
    assert get_category("vscode") == "text_editor"
