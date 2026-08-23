"""What TENKA is belongs to TENKA, not to whichever personality is loaded.

The live defect. Asked for her favourite colour under `warm_honest`, she said:

    "Honestly? I don't have one. I'm software, so colors don't really register
     for me in that way."

Thirty lines earlier in the same file she is "a desktop companion" who "lives in
the user's computer", and in the same session she answered "who are you" with
"I'm Tenka. I live in your computer and help you with whatever you need". One
personality, two identities, and the wrong one won the question that mattered.

**Why it won.** `_build_personality_rules` has said *"Don't say 'as an AI' or
'I'm just a program'"* all along. But `warm_honest/prompt.txt` said "You are
software" as a statement of fact, and a personality's own text and the shared
invariants are concatenated with no precedence -- so a personality can
contradict any invariant by asserting the opposite. Personality governs
**delivery**. What she is is not a personality's to write. Exactly the principle
that keeps personality from changing a verdict.

**Both directions, because a hard "never mention it" would be worse.** Refusing
to answer, or claiming to be a person, when someone sincerely asks is deception
-- and that is worse than the disclaimer this file exists to remove. So the rule
is *never volunteer, never deny*. "I live in this computer" is true, which is
why it answers nearly every real case without either failure.

Run with:  py -3.11 -m pytest tests/test_identity_is_not_a_personality.py -v
"""
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PERSONALITIES = _ROOT / "assistant" / "personalities"

# Claims about *substrate*, which is what a personality may not assert. Phrased
# as second-person statements because that is how a prompt file addresses her --
# "You are software", not "I am software".
#
# Deliberately not a general ban on the words "AI" or "software": the shared
# block below contains "not claim to be one", and `_build_personality_rules`
# names the phrases it forbids. A sweep that matched any mention would flag the
# rules that exist to prevent the thing.
_SUBSTRATE_CLAIMS = re.compile(
    r"you\s+are\s+(?:just\s+)?(?:an?\s+)?"
    r"(?:software|program|programme|ai|a\.i\.|bot|chatbot|language\s+model|"
    r"machine|algorithm|code|model)\b",
    re.IGNORECASE,
)


def _personality_dirs():
    return [d for d in sorted(_PERSONALITIES.iterdir()) if (d / "prompt.txt").exists()]


# ─── no personality asserts what she is made of ──────────────────────────────

def test_every_personality_prompt_is_walked():
    """A sweep over an empty set passes forever."""
    dirs = _personality_dirs()
    assert len(dirs) >= 3, f"only found {[d.name for d in dirs]}"


@pytest.mark.parametrize("folder", _personality_dirs(), ids=lambda d: d.name)
def test_no_personality_prompt_claims_to_be_software(folder):
    """The sweep that stops a fourth personality reintroducing this.

    `warm_honest` shipped "You are software." and nothing objected -- no test,
    no structure, and a shared rule saying the opposite that simply lost. The
    file is data, so the guard has to be a sweep over the data.
    """
    text = (folder / "prompt.txt").read_text(encoding="utf-8")
    hits = _SUBSTRATE_CLAIMS.findall(text)
    assert not hits, (
        f"{folder.name}/prompt.txt asserts what she is made of: {hits}. "
        f"Identity is composed once in `llm/prompts.py:_IDENTITY_BLOCK` and "
        f"inherited; a personality describes how she talks."
    )


@pytest.mark.parametrize("folder", _personality_dirs(), ids=lambda d: d.name)
def test_no_personality_prompt_denies_having_preferences(folder):
    """The second half of the same defect, and the one that produced the
    observed reply. "I don't have a favourite colour" is a refusal dressed as
    honesty: she has moods and opinions everywhere else in these files, and a
    personality that denies them for one question is inconsistent with itself,
    not careful."""
    text = (folder / "prompt.txt").read_text(encoding="utf-8")
    denials = re.findall(
        r"you\s+(?:do\s*n[o']t|don't|cannot|can't)\s+have\s+"
        r"(?:any\s+)?(?:real\s+)?(?:preferences|opinions|feelings|favourites|favorites)",
        text, re.IGNORECASE,
    )
    assert not denials, (
        f"{folder.name}/prompt.txt tells her she has no inner life: {denials}"
    )


# ─── the identity is actually composed in ────────────────────────────────────

def test_the_identity_block_reaches_every_personality(monkeypatch):
    """Composed, not merely defined. A block nothing concatenates is the most
    expensive kind of dead text: it reads as a fix."""
    from assistant.llm import prompts as p
    import assistant.personalities as personalities_mod
    from assistant.personalities import PersonalityLoader

    for folder in _personality_dirs():
        loader = PersonalityLoader(folder.name)
        monkeypatch.setattr(personalities_mod, "get_active_loader",
                            lambda ldr=loader: ldr)
        base = p._get_personality_base()
        assert "lives in this computer" in base, (
            f"{folder.name} does not inherit the identity block"
        )


def test_the_identity_block_comes_after_the_personality_text(monkeypatch):
    """Order is load-bearing. A personality that still describes itself early
    on must not get the last word on what she is."""
    from assistant.llm import prompts as p
    import assistant.personalities as personalities_mod
    from assistant.personalities import PersonalityLoader

    loader = PersonalityLoader("warm_honest")
    monkeypatch.setattr(personalities_mod, "get_active_loader", lambda: loader)
    base = p._get_personality_base()

    own_text = loader.get_prompt_base()
    assert base.index(own_text[:40]) < base.index("lives in this computer")


def test_the_identity_block_carries_the_configured_name():
    """Never a hardcoded "TENKA". The display name is configuration, and a
    prompt that disagrees with it hands her two names."""
    from assistant import config
    from assistant.llm.prompts import _build_identity_block

    assert config.ASSISTANT_NAME_DISPLAY in _build_identity_block()
    assert "{name}" not in _build_identity_block(), "the template was not rendered"


# ─── both directions of the rule ─────────────────────────────────────────────

def test_the_block_forbids_volunteering_what_she_is_made_of():
    from assistant.llm.prompts import _build_identity_block
    text = _build_identity_block().lower()
    assert "never volunteer" in text, (
        "nothing stops the unprompted disclaimer this whole file is about"
    )


def test_the_block_forbids_claiming_to_be_a_person():
    """**The carve-out, and it is not optional.** A rule that only said "never
    mention it" would be satisfied by a TENKA who insists she is human when
    someone sincerely asks -- worse than the disclaimer it replaced, and a
    thing no personality should be able to produce."""
    from assistant.llm.prompts import _build_identity_block
    text = _build_identity_block().lower()
    assert "not claim to be one" in text, (
        "the block permits denying what she is when asked directly"
    )


def test_an_ordinary_character_question_is_answered_not_explained():
    """The instruction that addresses the observed reply directly, rather than
    relying on the general ban to cover it."""
    from assistant.llm.prompts import _build_identity_block
    text = _build_identity_block().lower()
    assert "favourite colour" in text or "favorite color" in text, (
        "the block does not name the ordinary-character-question case, which is "
        "the one that actually went wrong"
    )


def test_the_older_rule_against_disclaimers_is_still_there():
    """It was right all along and it is kept. What changed is that it no longer
    stands alone against a personality asserting the opposite as fact."""
    from assistant.llm.prompts import _build_personality_rules
    rules = _build_personality_rules("neutral").lower()
    assert "as an ai" in rules and "just a program" in rules
