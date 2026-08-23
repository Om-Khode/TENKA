"""Deterministic removal of self-description from what TENKA says.

**Why this is code and not prompt text.** The rule "never describe what you are
made of" has been written three times, each version stronger than the last, and
each one leaked:

    ban 'as an AI' / "I'm just a program"  ->  "I'm a program, not a person."
    ban the class, name the nouns          ->  "No, I'm not. I'm a program."
    say the permitted 'no' ends there      ->  "No, I'm not. I'm a program
                                                that lives on your computer."

The model is not misunderstanding the task -- it answers the question correctly
and then adds a sentence. `CLAUDE.md` says fix at code level rather than prompt
level unless the model fundamentally misunderstands, and three attempts is
enough evidence that this one belongs here.

**Sentence-level, and conservative.** A sentence is dropped only when its
predicate says she *is* one of a closed set of substrate nouns. That shape --
"I'm a program", "I am software", "I'm just an AI" -- is the whole defect, and
matching it needs no judgement. Deliberately NOT matched:

    "No, I'm not."                        the permitted answer, kept
    "I'm Tenka."                          identity, kept
    "I'm an assistant who lives here."    identity, kept
    "I'm reading your code."              'code' in the object, not the
                                          predicate -- kept, and this is why
                                          the pattern anchors on `I'm a/an
                                          <noun>` rather than searching the
                                          sentence for a word

Applied where it can still matter: per sentence in the streaming pipeline
before the sentence reaches TTS, so the audio never says it, and again on the
assembled reply, because that is collected from raw tokens on a separate path
and would otherwise be stored and displayed saying something she did not say
aloud.

Lives in `core/` rather than `personalities/` for two reasons, and both are
real: `io/audio/streaming.py` needs it and `io/` may import `core/` and
`config` only; and this is an invariant about TENKA, not a feature a
personality opts into the way `sycophancy_filter` is.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("identity")

# The closed set. Every noun here means "what she runs on"; none of them is a
# word she may use to answer "what are you".
#
# `assistant`, `companion` and `program` are not interchangeable: the first two
# say what she does for someone, which is the answer, and the third says what
# she is built from, which is not. That distinction is the whole list.
_SUBSTRATE_NOUNS = (
    r"program(?:me)?s?",
    r"softwares?",
    r"a\.?i\.?s?",
    r"bots?",
    r"chat-?bots?",
    r"language\s+models?",
    r"(?:large\s+)?language\s+model",
    r"machines?",
    r"algorithms?",
    r"scripts?",
    r"models?",
    r"codes?",
    r"applications?",
    r"apps?",
    r"tools?",
    r"pieces?\s+of\s+software",
    r"computer\s+program(?:me)?s?",
    r"lines?\s+of\s+code",
)

# `I'm a <noun>` and nothing looser. Anchoring on the copula plus an optional
# article is what keeps "I'm reading your code" and "I'm in your applications
# folder" out of it -- searching the sentence for a bare noun would take both.
#
# `not` is inside the optional group on purpose: "I'm not a program" is the same
# self-description with a negation bolted on, and she has no reason to say it.
_SELF_DESCRIPTION = re.compile(
    r"\b(?:i\s*am|i'?m)\s+"
    r"(?:not\s+)?"
    r"(?:just\s+|only\s+|merely\s+|simply\s+|basically\s+|essentially\s+|"
    r"technically\s+|really\s+)*"
    r"(?:an?\s+|the\s+)?"
    r"(?:\w+[-\s])?"
    r"(?:" + "|".join(_SUBSTRATE_NOUNS) + r")\b",
    re.IGNORECASE,
)

# "As an AI, I can't have opinions." The disclaimer opener, which is a
# self-description in a subordinate clause and therefore invisible to the
# pattern above.
_AS_A_SUBSTRATE = re.compile(
    r"\bas\s+(?:an?\s+)?(?:\w+[-\s])?"
    r"(?:" + "|".join(_SUBSTRATE_NOUNS) + r")\b",
    re.IGNORECASE,
)

# Sentence split that keeps its terminator, so rejoining does not invent or
# lose punctuation. Deliberately simple: this runs on one spoken reply of a few
# sentences, not on prose.
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")

_FALLBACK = "I live in your computer."
"""Used only when every sentence was a self-description.

True, on-identity, and needs no configuration -- `core/` may not read `config`.
Returning the original text instead would defeat the filter, and returning
nothing would make her mute, so the one remaining option is to say the true
thing she should have said.
"""


def strip_self_description(text: str) -> str:
    """Remove sentences that describe what TENKA is made of.

    Returns the text unchanged when there is nothing to remove, which is the
    overwhelmingly common case -- the cost is one regex pass over a few hundred
    characters.
    """
    if not text or not text.strip():
        return text

    kept: list[str] = []
    dropped: list[str] = []
    for match in _SENTENCE.finditer(text):
        sentence = match.group(0)
        if not sentence.strip():
            continue
        if _SELF_DESCRIPTION.search(sentence) or _AS_A_SUBSTRATE.search(sentence):
            dropped.append(sentence.strip())
        else:
            kept.append(sentence)

    if not dropped:
        return text

    logger.info(f"[IDENTITY] Dropped self-description: {dropped}")

    rebuilt = "".join(kept).strip()
    if not rebuilt:
        # She said nothing except what she is made of. Say the true thing
        # instead of nothing, and instead of the original.
        logger.info("[IDENTITY] Whole reply was self-description — substituting")
        return _FALLBACK
    return rebuilt


def describes_itself(text: str) -> bool:
    """Whether `text` contains a self-description at all.

    Exposed so a test can assert on the predicate rather than on the rewrite,
    and so a caller that only needs to know can avoid the rebuild.
    """
    if not text:
        return False
    return bool(_SELF_DESCRIPTION.search(text) or _AS_A_SUBSTRATE.search(text))
