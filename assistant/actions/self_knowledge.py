"""self_knowledge.py — the handler for questions about TENKA herself.

TENKA-v2 §13. The facts live in `brain/selfknowledge.py`; this is the door
`actions.execute()` opens onto them.

**The handler cannot reach the Brain**, and that is the fourth time this plan
has met the same wall: `actions/` sits below `brain/`, there is no
`actions -> brain` import in the tree, and adding one would be a layer
inversion. So the Brain injects its reader here, the way `main.py` injects the
event-bus dispatcher and the `ChatDispatch` protocol. With nothing injected the
handler answers `UNAVAILABLE` -- which is also the honest answer, since without
the registry there are no facts to report.

**K1 in one line: this handler does not compose prose.** It selects a fact and
returns what the fact read. The model's job is to explain the value it is
handed on the way out, never to supply it.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .registry import tool_registry

logger = logging.getLogger("actions")

# K2's sentence, duplicated as a literal rather than imported from `brain/`,
# because importing it would be the inversion this file exists to avoid. Pinned
# equal to the Brain's by `test_self_knowledge.py` -- two copies of a sentence
# is acceptable; two copies that can drift apart silently is not.
UNAVAILABLE = (
    "I don't have reliable information about that part of my current "
    "implementation."
)

# Set by `main.py` at startup. `Optional` and checked, never assumed: a handler
# that raises because nobody wired it is a worse answer than one that says it
# does not know.
_reader: Optional[Callable[..., object]] = None


def set_reader(reader: Callable[..., object]) -> None:
    """Injected by `main.py`, which may import both packages."""
    global _reader
    _reader = reader
    logger.info("[SELF] self-knowledge reader wired")


# Which fact answers which shape of question. Word overlap against the fact's
# own vocabulary -- no model call, and no table of phrasings to maintain.
#
# Deliberately small and deliberately not clever: a question this does not
# recognise gets `UNAVAILABLE`, which is correct. Guessing at the closest fact
# would be the same failure as a resolver returning a wrong affordance, except
# the answer is about herself and therefore harder for the asker to check.
_VOCABULARY: "dict[str, tuple[str, ...]]" = {
    "model_chain": ("model", "models", "llm", "provider", "gemini", "chain"),
    "personality": ("personality", "persona", "character", "mood"),
    # No "do". It appears in "what do you do", "which model do you use" and
    # "what commands do you know" alike, so it claimed every question and the
    # first key to score 1 won -- which was this one. Same over-claim shape as
    # `regex_router.pre_route`: a pattern that matches all of ordinary English
    # steals the request from whatever would have answered it correctly.
    "affordances": ("can", "capable", "capabilities", "able", "affordance",
                    "abilities", "skills", "tasks"),
    "intents": ("intent", "intents", "commands", "vocabulary"),
    # "How do you do X". Checked before the others in `_select` because these
    # words are about *mechanism*, and a question naming both a mechanism word
    # and an intent name ("what do you use for file_task") is asking how, not
    # which. Without this the only intent-shaped answer was the whole list,
    # which is what live testing produced.
    "mechanism": ("how", "mechanism", "implemented", "implementation",
                  "handler", "works", "uses", "use"),
}


def _names_an_intent(query: str) -> bool:
    """Does the question name one of TENKA's intents?

    The signal that separates "how do you do X" from every other question:
    a mechanism question is *about a specific capability*, and the only way to
    be about one is to name it.
    """
    from ..config import INTENTS

    lowered = (query or "").lower()
    words = {w.strip(".,!?;:'\"") for w in lowered.split()}
    return any(i in words or i.replace("_", " ") in lowered for i in INTENTS)


def _select(query: str) -> Optional[str]:
    """Which fact answers this question, or None.

    **Mechanism first, but only when the question names an intent.** Word
    overlap alone cannot separate these two:

        "what do you use for file_task"     -> mechanism
        "which model do you use"            -> model_chain

    Both contain "use". Ordering the dict does not help -- whichever comes
    first wins both. What actually distinguishes them is that the first names
    a capability and the second does not, so that is the test. Live testing
    produced exactly this confusion: asked how she searches files, she recited
    every intent she has, because "intent" scored for `intents` and won the
    tie.

    "can" is in the affordance vocabulary and "do" is not, which looks
    arbitrary until you count: "do" appears in "what do you do", "which model
    do you use" and "what commands do you know" alike, so it matched every
    question and the first key to score won. "can" carries capability in a way
    "do" does not -- and the risk is smaller here than in `pre_route` anyway,
    because this runs only after the classifier has decided the question is
    about TENKA herself.

    Ties otherwise go to the earlier key, which is why `model_chain` is first:
    "which model can you use" scores one for each, and the model is the answer.
    """
    words = {w.strip(".,!?;:'\"").lower() for w in (query or "").split()}
    if words & set(_VOCABULARY["mechanism"]) and _names_an_intent(query):
        return "mechanism"
    best, score = None, 0
    for key, vocabulary in _VOCABULARY.items():
        if key == "mechanism":
            # Handled above, and deliberately not eligible here: its words
            # ("how", "use", "works") are common enough to steal questions
            # they do not answer.
            continue
        hits = len(words & set(vocabulary))
        if hits > score:
            best, score = key, hits
    return best


@tool_registry.decorator("self_knowledge")
def handle_self_knowledge(params: dict, llm_response: str) -> str:
    """Answer a question about TENKA's own implementation, or say she cannot.

    The grant set is read from the turn's contextvar rather than passed in:
    every other capability decision on this path reads the same one, and a
    second channel for "what may this caller see" is a second thing to keep in
    sync.
    """
    query = (params.get("query") or llm_response or "").strip()
    key = _select(query)
    if key is None or _reader is None:
        return UNAVAILABLE

    from . import current_grants
    granted = current_grants.get()

    try:
        # The query goes through so a mechanism question can name what it is
        # about. Facts that do not want it ignore it -- `Fact.takes_query`
        # decides, not the caller.
        value = _reader(key, granted, query)
    except Exception as e:
        logger.debug(f"[SELF] read failed: {e}")
        return UNAVAILABLE

    if value == UNAVAILABLE or value is None:
        return UNAVAILABLE

    if isinstance(value, (tuple, list)):
        value = ", ".join(str(v) for v in value)
    return f"{key}: {value}"
