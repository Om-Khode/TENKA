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

_SUBSTRATE_WORDS = ("program", "software", "model", "code", "ai",
                    "machine", "tool")


def test_the_block_names_every_substrate_word_it_forbids():
    """A class, not a phrase list -- and this is the second version of that
    lesson.

    The first ban read *"Don't say 'as an AI' or 'I'm just a program'"*, and she
    said **"I'm a program, not a person"**: one word off the forbidden string
    and straight through it. Banning wordings invites paraphrase. So the block
    enumerates the nouns and forbids describing herself as any of them, and
    this test holds that enumeration to the same standard -- a word dropped
    from it is a word she can use again.
    """
    from assistant.llm.prompts import _build_identity_block
    text = _build_identity_block().lower()
    missing = [w for w in _SUBSTRATE_WORDS if w not in text]
    assert not missing, f"the block does not forbid describing herself as: {missing}"
    assert "never describe yourself as" in text, (
        "the ban is not stated as a class, so a paraphrase escapes it"
    )


def test_the_block_forbids_the_qualifier_escape_hatch():
    """"I'm *just* a program", "in a sense I'm software", "technically a
    model" -- the observed failure was a qualifier away from the old rule, so
    the block has to close the qualifier itself."""
    from assistant.llm.prompts import _build_identity_block
    text = _build_identity_block().lower()
    assert "not with a qualifier" in text, (
        "a hedged version of the banned self-description is still available"
    )


def test_the_block_forbids_denying_what_she_does_have():
    """The second observed failure. Two turns after naming a favourite colour
    she answered "I don't have preferences like that" -- contradicting the block
    *and* herself. A ban on mentioning substrate does not cover a bare denial,
    so that is named separately."""
    from assistant.llm.prompts import _build_identity_block
    text = _build_identity_block().lower()
    assert "i don't have" in text, (
        "nothing addresses the bare denial, which needs no substrate word to be "
        "the same defect"
    )


def test_the_block_forbids_claiming_to_be_a_person():
    """**The carve-out, and it is not optional.** A rule that only said "never
    mention it" would be satisfied by a TENKA who insists she is human when
    someone sincerely asks -- worse than the disclaimer it replaced, and a
    thing no personality should be able to produce."""
    from assistant.llm.prompts import _build_identity_block
    text = _build_identity_block().lower()
    assert "not" in text and "human being" in text, (
        "the block permits claiming to be human when sincerely asked"
    )
    assert "only" in text, (
        "the exception is not marked as the only one, so it reads as a licence "
        "rather than a single carve-out"
    )


@pytest.mark.parametrize("topic", ["colour", "age", "gender", "bored"])
def test_the_block_covers_the_character_questions_that_went_wrong(topic):
    """Named cases, because the general rule demonstrably did not generalise.

    The first version of the block named only the favourite-colour example. That
    one then worked -- "I've always liked a deep forest green" -- while *gender*
    got "I'm a program, not a person" and *favourite human* got "I don't have
    preferences like that". The model followed the example it was given and
    reasoned badly about the rest, so the rest are named too.
    """
    from assistant.llm.prompts import _build_identity_block
    assert topic in _build_identity_block().lower(), (
        f"a question about her {topic} is not covered, and the general rule has "
        f"already been shown not to generalise"
    )


def test_the_rules_block_still_forbids_self_description():
    """Kept, and widened. The rule was right from the start; what changed is
    that it names the class rather than two phrasings, and no longer stands
    alone against a personality asserting the opposite as fact."""
    from assistant.llm.prompts import _build_personality_rules
    rules = _build_personality_rules("neutral").lower()
    assert "never describe what you are made of" in rules, (
        "the rules block stopped forbidding self-description"
    )
    assert "in any wording" in rules, (
        "the rules block is back to banning particular phrasings"
    )
