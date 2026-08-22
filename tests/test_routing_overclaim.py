"""`regex_router.pre_route` must not claim requests it cannot answer.

`pre_route` is a cost optimisation: it answers ~40-50% of daily commands with
zero LLM calls. The risk it carries is not being wrong about what it claims --
it is claiming too much. A pattern that matches more than its intent covers
silently steals the request from the classifier that would have got it right,
and the misroute is invisible because the fast path never asks a second opinion.

Measured on 2026-08-22: `_FORGET_MEMORY_RE` claimed **every** utterance starting
`forget|delete|remove|erase`, so `delete "C:/…/notes.txt"` routed to
`forget_memory` and answered "I don't have anything about that." The same
session shows the classifier getting the identical path right as `file_task`
one turn later -- the regex was the only thing that was wrong.

Two invariants, both zero-cost:

  1. **Declining is a behaviour worth testing.** Every entry below with
     `None` must fall through to the LLM. A pattern that starts claiming
     these has widened, and widening is the failure mode.
  2. **Claims are exact.** Every entry with an intent must produce that
     intent, so tightening a pattern cannot quietly stop it answering the
     thing it exists for.

The corpus is hand-written on purpose. The first version of this file was
generated from real interaction history, which pulled three live OAuth
credentials out of the database and was one `git add` from a public repo (see
KI-29). Anything here is either invented or a shape reduced to inert content.

Run with:  py -3.11 -m pytest tests/test_routing_overclaim.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# (utterance, expected intent or None) -- None means "must fall through to the
# classifier". Grouped by the pattern each group is aimed at, so a failure
# names the pattern that widened.
CASES = [
    # ── _FORGET_MEMORY_RE: memory phrasing it SHOULD claim ────────────────
    ("forget that I like coffee", "forget_memory"),
    ("forget about my old job", "forget_memory"),
    ("delete the fact that I live in Pune", "forget_memory"),
    ("remove the memory that I hate mornings", "forget_memory"),
    ("erase that I said yes", "forget_memory"),
    ("forget my old phone number", "forget_memory"),

    # ── _FORGET_MEMORY_RE: file work it must NOT claim ────────────────────
    # The reported bug and its neighbourhood. Every one of these answered
    # "I don't have anything about that" before the guard existed.
    ('delete "C:/Users/user/Desktop/Temp/notes.txt"', None),
    ("delete C:/Users/user/Desktop/notes.txt", None),
    ("delete the file notes.txt", None),
    ("delete notes.txt", None),
    ("remove report.pdf", None),
    ("delete my downloads folder", None),
    ("remove that screenshot from the desktop", None),
    ("erase the temp folder", None),
    ("delete everything in documents", None),

    # ── _FORGET_MEMORY_RE: app / window work it must NOT claim ────────────
    ("close chrome and delete the tab", None),
    ("remove the last row from the spreadsheet", None),

    # ── Deletion targets that are neither a file nor a memory ─────────────
    # These are what the mandatory-qualifier half of the pattern protects,
    # and the table missed them at first: none is file-shaped, so the
    # file-shape guard alone lets every one of them through. Found by a
    # mutation that should have gone red and did not.
    ("delete the last email", None),
    ("delete my playlist", None),
    ("remove the sticky note", None),
    ("delete the second item", None),
    ("remove the alarm", None),
    ("delete the draft", None),

    # ── _SEARCH_RE already declines file-shaped content — keep it that way ─
    ("search for my tax return pdf", None),
    ("find notes.txt", None),
    ("search for the best coffee in town", "web_search"),
    ("google the weather", "web_search"),

    # ── _REMEMBER_FACT_RE ─────────────────────────────────────────────────
    ("remember that I prefer dark mode", "store_memory"),
    ("remember my birthday is in March", "store_memory"),

    # ── monitors / schedules must keep winning over the memory pattern ────
    ("delete all monitors", "manage_monitor"),
    ("remove the song monitor", "manage_monitor"),
]


def _route(text):
    from assistant.regex_router import pre_route
    r = pre_route(text)
    return r.intent if r is not None else None


@pytest.mark.parametrize("text,expected", CASES,
                         ids=[c[0][:44] for c in CASES])
def test_pre_route_claims_exactly_what_it_can_answer(text, expected):
    got = _route(text)
    if expected is None:
        assert got is None, (
            f"pre_route claimed {got!r} for {text!r}.\n"
            "This request needs the classifier. A regex that claims it answers "
            "from the wrong handler and never asks for a second opinion."
        )
    else:
        assert got == expected, (
            f"pre_route returned {got!r} for {text!r}, expected {expected!r}.\n"
            "A pattern was tightened past the thing it exists to answer."
        )


def test_the_case_table_is_not_empty():
    """Anti-vacuity. A parametrised test over an empty list passes forever, and
    an import error that silently emptied CASES would look like a clean run."""
    assert len(CASES) >= 20, f"only {len(CASES)} cases -- the table shrank"
    assert any(e is None for _, e in CASES), "no decline cases left"
    assert any(e is not None for _, e in CASES), "no claim cases left"


def test_no_utterance_is_claimed_by_two_different_intents():
    """`pre_route` is an ordered chain, so order is the only disambiguation
    between patterns that overlap. This does not test the patterns in
    isolation -- it asserts the chain is deterministic and that no case in the
    table is answered differently on a second call, which would mean routing
    depends on state rather than text."""
    from assistant.regex_router import pre_route

    for text, _ in CASES:
        first = pre_route(text)
        second = pre_route(text)
        a = first.intent if first else None
        b = second.intent if second else None
        assert a == b, (
            f"pre_route is not deterministic for {text!r}: {a!r} then {b!r}. "
            "Routing depends on something other than the utterance."
        )


def test_a_declined_utterance_returns_none_not_unknown():
    """Declining means returning `None` so the caller falls through. Returning
    an `unknown` IntentResult would look like a decline in a log and dispatch
    to a handler in practice."""
    from assistant.regex_router import pre_route

    declined = [t for t, e in CASES if e is None]
    assert declined, "no decline cases -- this test would pass vacuously"
    for text in declined:
        r = pre_route(text)
        assert r is None, (
            f"{text!r} produced {r.intent!r} instead of None. A decline must be "
            "an absence of a result, not a result meaning 'nothing'."
        )
