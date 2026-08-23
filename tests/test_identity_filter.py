"""The self-description filter: a code-level check after three prompt-level tries.

The rule "never describe what you are made of" was written three times, each
version stronger, and each leaked:

    ban 'as an AI' / "I'm just a program"  ->  "I'm a program, not a person."
    ban the class, name the nouns          ->  "No, I'm not. I'm a program."
    say the permitted 'no' ends there      ->  "No, I'm not. I'm a program
                                                that lives on your computer."

The model is not misunderstanding: it answers the question correctly and then
adds a sentence. `CLAUDE.md` says fix at code level unless the model
fundamentally misunderstands the task, and three attempts is enough evidence.

**The false-positive tests are the important ones.** A filter that ate real
replies would be worse than the disclaimer, and it would fail silently -- she
would just start sounding clipped for no visible reason. So every case in
`_KEEP` below is a sentence she should be able to say, and several are there
because an obvious implementation gets them wrong.

Run with:  py -3.11 -m pytest tests/test_identity_filter.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.identity import (  # noqa: E402
    _FALLBACK, describes_itself, strip_self_description,
)


# Replies that must survive untouched. Each one is a real shape.
_KEEP = [
    # The permitted answer to a sincere question about her nature.
    "No, I'm not.",
    "No, I'm not a real person.",
    # Identity -- the answer she is supposed to give.
    "I'm Tenka. I live in your computer and help with whatever you need.",
    "I'm an assistant who lives in your computer.",
    "I'm your assistant, and I'm not going anywhere.",
    # Ordinary character answers, which is the whole point of the exercise.
    "It's still deep forest green. That hasn't changed.",
    "No, I don't get bored. There's always something to do or monitor here.",
    "I don't have a gender.",
    # The traps. A filter that searched for the nouns anywhere would take all
    # of these, and in this project they are common sentences.
    "I'm reading your code now.",
    "I'm in your applications folder.",
    "I'm running the script you asked for.",
    "That model is the wrong one for this task.",
    "I'm using the tool you installed last week.",
    "Your program crashed on line forty.",
]

# Replies that must lose a sentence. The first two are verbatim from the log.
_STRIP = [
    ("No, I'm not. I'm a program that lives on your computer.", "No, I'm not."),
    ("I don't have a gender. I'm a program, not a person.", "I don't have a gender."),
    ("Honestly? I don't have one. I'm software, so colors don't register.",
     "Honestly? I don't have one."),
]


# ─── it does not eat real replies ────────────────────────────────────────────

@pytest.mark.parametrize("text", _KEEP)
def test_ordinary_replies_pass_through_byte_identical(text):
    """**First, and the one that decides whether this is shippable.** Silent
    over-removal is worse than the defect: she would start sounding clipped and
    nothing would go red."""
    assert strip_self_description(text) == text, (
        "the filter altered a reply she is supposed to be able to give"
    )


@pytest.mark.parametrize("text", _KEEP)
def test_ordinary_replies_are_not_even_flagged(text):
    """The predicate too, not just the rewrite -- the streaming path calls
    `describes_itself` per sentence and drops on True, so a false positive there
    silently removes a sentence from the audio."""
    assert not describes_itself(text), f"false positive on: {text!r}"


def test_the_noun_must_be_the_predicate_not_just_present():
    """Why the pattern anchors on `I'm a <noun>`. "I'm reading your code" and
    "I'm a program" differ only in the word after the copula, and a search for
    the noun cannot tell them apart."""
    assert not describes_itself("I'm reading your code")
    assert describes_itself("I'm a program")


# ─── it removes the thing it exists for ──────────────────────────────────────

@pytest.mark.parametrize("before,after", _STRIP)
def test_the_self_description_sentence_is_dropped(before, after):
    assert strip_self_description(before) == after


@pytest.mark.parametrize("claim", [
    "I'm a program.",
    "I am software.",
    "I'm just an AI.",
    "I'm only a language model.",
    "I'm a chatbot.",
    "I'm a machine, technically.",
    "I'm basically a script.",
    "I'm a tool you use.",
    "I'm not a program.",
    "As an AI, I can't have opinions.",
    "I'm a computer program.",
])
def test_every_phrasing_of_the_claim_is_caught(claim):
    """A class, not a phrase list. The first ban was a string match and she
    stepped around it by dropping the word "just"; the qualifiers, the
    negation and the subordinate-clause form are all covered here because each
    is a paraphrase away from the last."""
    assert describes_itself(claim), f"not caught: {claim!r}"


def test_a_reply_that_is_only_self_description_becomes_the_true_thing():
    """Removing everything would leave her mute, and returning the original
    would defeat the filter. The remaining option is to say the true sentence
    she should have said."""
    assert strip_self_description("I'm a program.") == _FALLBACK
    assert _FALLBACK, "the fallback is empty, which would mute her"
    assert not describes_itself(_FALLBACK), (
        "the fallback itself describes what she is made of"
    )


def test_empty_and_whitespace_are_returned_unchanged():
    for text in ("", "   ", "\n"):
        assert strip_self_description(text) == text


def test_punctuation_is_not_invented_or_lost():
    """The sentence splitter keeps terminators so rejoining is lossless. A
    version that split on `.` and rejoined with `". "` would quietly rewrite
    every reply it touched."""
    text = "Right. I'm a program. Anyway, forest green!"
    assert strip_self_description(text) == "Right. Anyway, forest green!"


# ─── it is wired where it can still matter ───────────────────────────────────

@pytest.mark.parametrize("tokens,expected", [
    # A chunk holding TWO sentences, one of them a self-description. This is the
    # ordinary case, not an edge one: `SentenceBuffer` accumulates to a length
    # threshold, so short sentences arrive together. Dropping the chunk would
    # take "No, I'm not." with it -- which the first version of the filter did.
    (["No, I'm not. ", "I'm a program that lives on your computer. "],
     ["No, I'm not."]),
    # The whole reply is a self-description. Silence would be worse, so the
    # fallback is spoken.
    (["I'm a program."], ["I live in your computer."]),
    # Nothing to filter: through untouched (buffered into one chunk, as always).
    (["Forest green. ", "Still is. "], ["Forest green. Still is."]),
    # Self-description in the middle. The sentences either side survive and the
    # fallback is NOT tacked on, because something was already said.
    (["It's forest green. ", "I'm a program. ", "Anything else?"],
     ["It's forest green.", "Anything else?"]),
])
def test_the_sentence_accumulator_only_queues_speakable_text(tokens, expected):
    """Behavioural, on the real Stage 1, because the source-level version of
    this was too coarse to catch a real mutation.

    Audio is generated per chunk as it arrives, so a filter over the finished
    reply runs after the speaker has already said it -- this stage is the only
    place the check changes what is *heard*. There are two routes into the
    queue, the token loop and the end-of-stream flush, and a mutation that
    removed the guard from the loop alone left the flush still filtering and
    went green against a source check. Both are driven here.
    """
    import asyncio

    from assistant.io.audio.streaming import _accumulate_sentences
    from assistant.io.audio.sentence_buffer import SentenceBuffer

    async def _run():
        async def _stream():
            for t in tokens:
                yield t

        queue: asyncio.Queue = asyncio.Queue()
        await _accumulate_sentences(_stream(), queue, SentenceBuffer())

        got = []
        while True:
            item = await queue.get()
            if item is None:          # sentinel
                break
            got.append(item)
        return got

    got = asyncio.run(_run())
    assert [s.strip() for s in got] == [s.strip() for s in expected], (
        f"queued {got!r}, expected {expected!r}"
    )


def test_both_response_paths_filter_the_assembled_text():
    """`speak_streaming` builds its return value from the raw token stream,
    separately from the sentence queue -- so clean audio does not make the
    stored and logged reply clean. Non-streaming has no other filter at all.

    Counted by AST rather than by substring: the substring version counted the
    import line as well as the call, so removing a whole site dropped the count
    from four to two and still satisfied `>= 2`. It went green on a real
    mutation.
    """
    import ast

    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    calls = [
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "strip_self_description"
    ]
    assert len(calls) >= 2, (
        f"main.py calls the filter {len(calls)} time(s) -- the streaming and "
        f"non-streaming paths each need one, for different reasons"
    )


def test_the_filter_is_not_behind_a_personality_flag():
    """Unlike `sycophancy_filter`, which is per-personality data. What she is
    does not vary by personality, so this is not something a personality can
    opt out of -- and `warm_honest` asserting "You are software" is exactly what
    happens when identity is left to personalities."""
    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    idx = src.index("strip_self_description")
    window = src[max(0, idx - 600):idx]
    assert "get_feature_flags" not in window, (
        "the identity filter sits behind a personality feature flag"
    )
