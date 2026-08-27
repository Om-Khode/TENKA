"""Two operator decisions, and one plan entry that turned out to be wrong.

**She/her, stated plainly.** She answered "what is your gender?" with "I don't
have a gender", while this project refers to her as *she* throughout, ships an
explicitly female personality, and has an identity block telling her she has
preferences and answers from them. Those disagreed, and the disagreement was
settled by the operator rather than by me.

**Canonical preference keys.** The reflection prompt lets the model invent keys
and `get_preference` matches exactly, so `music_app` / `Music App` / `music-app`
are three rows at `CONFIDENCE_FIRST_OBSERVATION`, none ever promoted --
promotion happens by finding the *same* key again and bumping it. The D3 ladder
built in P9 could never accumulate.

That is the salvageable half of decision D10. The other half -- pointing
reflection at the router's `automation_{word}` keys -- rested on a wrong premise
of mine and would have broken routing; the correction is written up in TENKA-v2
section 21 and one test below pins the fact that makes it wrong, so nobody
"fixes" it back.

Run with:  py -3.11 -m pytest tests/test_identity_gender_and_key_canon.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.llm.prompts import _build_identity_block  # noqa: E402
from assistant.reflection import _canonical_preference_key  # noqa: E402


# ─── she/her ─────────────────────────────────────────────────────────────────

def test_the_identity_block_states_her_pronouns():
    text = _build_identity_block().lower()
    assert "she/her" in text, "the identity block does not state her pronouns"
    assert "a woman" in text


def test_she_is_told_not_to_announce_it():
    """Answering when asked is identity; volunteering it is the disclaimer
    habit this block spent three commits removing. Nobody introduces themselves
    by their pronouns."""
    text = _build_identity_block().lower()
    assert "never announce it unprompted" in text


def test_the_gender_answer_is_not_left_to_a_personality():
    """The same rule as everything else in this block. `warm_honest` asserting
    "You are software" is what happens when identity is a personality's to
    write, and a personality inventing a different answer here would be the
    same failure with a different subject."""
    for folder in sorted((_ROOT / "assistant" / "personalities").iterdir()):
        prompt = folder / "prompt.txt"
        if not prompt.exists():
            continue
        text = prompt.read_text(encoding="utf-8").lower()
        assert "gender" not in text, (
            f"{folder.name}/prompt.txt has an opinion about her gender; that "
            f"belongs to the identity block"
        )


def test_the_no_denial_rule_still_covers_gender():
    """"I don't have a gender" was a denial, and the block already forbids
    denying what she does have. Both halves must be present or the new line is
    a fact she can still decline to state."""
    text = _build_identity_block().lower()
    assert "gender" in text
    assert "i don't have" in text, (
        "the block no longer forbids the bare denial that produced this"
    )


# ─── canonical preference keys ───────────────────────────────────────────────

@pytest.mark.parametrize("written,canonical", [
    ("music_app", "music_app"),
    ("Music App", "music_app"),
    ("music-app", "music_app"),
    ("  MUSIC   APP  ", "music_app"),
    ("Messaging Default", "messaging_default"),
    ("verbosity", "verbosity"),
])
def test_one_spelling_per_preference(written, canonical):
    """Case, spacing and punctuation fold together, so a re-observation lands on
    the row it is meant to promote."""
    assert _canonical_preference_key(written) == canonical


def test_semantically_different_keys_stay_different():
    """**The direction over-normalising breaks.** `music_app` and
    `music_player` might well mean the same thing, and deciding that is a
    judgement -- collapsing them here would silently merge two preferences and
    hand the survivor the other's confidence."""
    assert (_canonical_preference_key("music_app")
            != _canonical_preference_key("music_player"))
    assert (_canonical_preference_key("verbosity")
            != _canonical_preference_key("verbose"))


@pytest.mark.parametrize("odd", ["", "   ", "__", "!!!"])
def test_a_key_that_normalises_to_nothing_is_kept_as_written(odd):
    """Not a validation step. Losing a preference to a naming rule is worse
    than storing an oddly-named one -- the confidence ladder decides whether it
    is ever acted on."""
    assert _canonical_preference_key(odd) == odd


def test_reflection_canonicalises_before_it_looks_up():
    """Where it has to happen. Canonicalising only on write would still miss the
    existing row on read, so every night would create a new one and nothing
    would ever be bumped."""
    import ast
    import inspect

    from assistant import reflection

    # By AST, not by substring. The substring version matched the *comment*
    # above the assignment, which names the function -- so removing the actual
    # call was a GREEN mutation. Comments explaining a mechanism are not the
    # mechanism, and this file has now hit that twice.
    src = inspect.getsource(reflection._process_discovered_preferences)
    tree = ast.parse(src)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_canonical_preference_key" in calls, (
        "the key is used raw, so `get_preference` misses the row it should "
        "bump and the ladder never accumulates"
    )


# ─── the D10 premise that was wrong, pinned ──────────────────────────────────

def test_the_router_key_holds_a_backend_not_an_app_name():
    """Why "reconcile the namespaces" was the wrong instruction, kept as a test
    so it is not re-adopted.

    `detect_backend` returns `pref["value"]` straight out of
    `automation_{word}`, and its callers treat that as a backend. Reflection's
    `music_app` holds an application name. Same-shaped key, different value
    domain -- pointing one at the other puts "spotify" where "browser" belongs.
    """
    import inspect

    from assistant.automation import router
    src = inspect.getsource(router._check_routing_preference)
    assert 'get_preference(f"automation_{word}")' in src
    assert 'return pref["value"]' in src, (
        "the router no longer returns the preference value directly; re-check "
        "whether the two key spaces are still incompatible"
    )


def test_reflection_preferences_are_read_by_category_not_key():
    """The other false claim: that reflection's proposals were unreadable. They
    are read by category, and always were -- which is why P9 had to gate that
    path to user-stated provenance rather than leaving it open."""
    src = (_ROOT / "assistant" / "actions" / "__init__.py").read_text(
        encoding="utf-8")
    block = src[src.index("def _build_goal_hints"):]
    block = block[:block.index("def _extract_contact_from_goal")]
    assert 'p["category"] in' in block, (
        "_build_goal_hints no longer selects by category, so the D10 correction "
        "needs re-checking"
    )
    assert "app_routing" in block
