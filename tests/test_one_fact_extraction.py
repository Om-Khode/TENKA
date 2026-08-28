"""One fact-extraction site, and the flag both response paths share.

TENKA-v2 §17.P9. `main.py` extracted facts twice — once on each response path —
and the copies had drifted in a way neither one shows on its own:

    non-streaming   extracts, sets `_facts_extracted`, then tells personality
    streaming       tells personality first, extracts a hundred lines later

So every streaming turn reported `facts_extracted=False` to
`personality.process_turn`. The streaming path is the conversational one — the
path where facts are actually learned — so the trait bump for learning
something about the user could only fire on the path that mostly does not.

The provenance changed too, and downward. Both sites wrote
`source="conversation"`, which reads like "the user said it". The user said
*something*; a model decided which part was a fact, what to call the key, and
what the value was. That is an inference over the user's words, so it is
`single_inference` — D3's rule that an unverifiable model claim stays a single
inference whatever it is labelled.

Run with:  py -3.11 -m pytest tests/test_one_fact_extraction.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MAIN = _ROOT / "assistant" / "main.py"


def _tree():
    return ast.parse(_MAIN.read_text(encoding="utf-8"))


def _calls(node, name):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) == name
                 or getattr(n.func, "id", None) == name)]


def _func(name):
    for n in ast.walk(_tree()):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n.name == name:
            return n
    return None


# ─── one site ────────────────────────────────────────────────────────────────

def test_facts_are_extracted_from_exactly_one_place():
    """The property. Two copies is how the flag diverged, and neither copy
    looked wrong on its own."""
    sites = _calls(_tree(), "extract_facts")
    assert sites, "walked nothing -- extract_facts is gone from main.py"
    assert len(sites) == 1, (
        f"facts are extracted from {len(sites)} places "
        f"(lines {[c.lineno for c in sites]}); there must be one")


def test_the_one_site_is_the_shared_helper():
    """Not merely one call -- one call *in the helper*. A single call left on
    the non-streaming path would satisfy the count and leave streaming turns
    storing nothing at all."""
    helper = _func("_extract_and_store_facts")
    assert helper is not None, "_extract_and_store_facts is gone"
    assert _calls(helper, "extract_facts"), (
        "the helper does not extract anything")


def test_both_paths_use_the_helper():
    pipeline = _func("_turn_pipeline")
    assert pipeline is not None
    uses = _calls(pipeline, "_extract_and_store_facts")
    assert len(uses) == 2, (
        f"the turn calls the extraction helper {len(uses)} times; both the "
        f"streaming and non-streaming paths need it")


# ─── personality hears the truth, once, on either path ───────────────────────

def test_personality_is_told_exactly_once_per_turn():
    """Two calls would double-bump every other trait this function moves."""
    calls = _calls(_tree(), "process_turn")
    assert calls, "walked nothing -- personality.process_turn is gone"
    assert len(calls) == 1, (
        f"personality.process_turn is called {len(calls)} times; the guard in "
        f"_note_personality exists so it is called once")


def test_the_personality_call_lives_behind_the_once_guard():
    noter = _func("_note_personality")
    assert noter is not None, "_note_personality is gone"
    assert _calls(noter, "process_turn"), (
        "the once-guard no longer wraps the personality call")


def test_both_paths_notify_personality():
    """The bug, structurally. The streaming path must reach it *after* its
    deferred extraction, or it reports False forever."""
    pipeline = _func("_turn_pipeline")
    uses = _calls(pipeline, "_note_personality")
    assert len(uses) == 2, (
        f"_note_personality is called from {len(uses)} places; the streaming "
        f"and non-streaming paths each need one")


def test_the_streaming_notification_comes_after_its_extraction():
    """Ordering is the whole defect. Notifying before extracting is exactly
    what the old code did, and it type-checks perfectly."""
    pipeline = _func("_turn_pipeline")
    extractions = sorted(c.lineno for c in
                         _calls(pipeline, "_extract_and_store_facts"))
    notifications = sorted(c.lineno for c in
                           _calls(pipeline, "_note_personality"))
    assert len(extractions) == 2 and len(notifications) == 2

    # The later notification (streaming) must follow the later extraction.
    assert notifications[-1] > extractions[-1], (
        f"personality is notified at line {notifications[-1]} but the "
        f"deferred extraction runs at {extractions[-1]} -- it would report "
        f"'no facts learned' on every streaming turn")


# ─── what gets written ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extracted_facts_are_stored_as_a_single_inference(monkeypatch):
    """Through the real helper, not by asserting on a constant.

    `source="conversation"` implied the user stated it. A model chose the key
    and the value; that is an inference over what they said.
    """
    from assistant import main as main_mod
    from assistant.core.provenance import Provenance, at_least

    written = []

    async def _facts(_text):
        return [{"key": "user_pet", "value": "a cat"}]

    async def _memtype(_k, _v):
        return "fact"

    monkeypatch.setattr(main_mod.llm, "extract_facts", _facts)
    monkeypatch.setattr(main_mod.llm, "ask_for_memory_type", _memtype)
    monkeypatch.setattr(main_mod.memory, "save_typed_fact",
                        lambda **kw: written.append(kw))

    count = await main_mod._extract_and_store_facts("I have a cat")

    assert count == 1
    assert written, "nothing was stored"
    source = written[0]["source"]
    assert source == Provenance.SINGLE_INFERENCE.value, (
        f"stored as {source!r}")
    assert not at_least(source, Provenance.EXPLICIT_USER_STATEMENT), (
        "an extracted fact outranks something the user actually stated")


@pytest.mark.asyncio
async def test_the_helper_reports_how_many_it_found(monkeypatch):
    """The return value is what both paths use to tell personality whether the
    turn taught anything. A helper that always returned 0 would restore the
    original bug while passing every structural test above."""
    from assistant import main as main_mod

    async def _none(_text):
        return []

    async def _memtype(_k, _v):
        return "fact"

    monkeypatch.setattr(main_mod.llm, "extract_facts", _none)
    monkeypatch.setattr(main_mod.llm, "ask_for_memory_type", _memtype)
    monkeypatch.setattr(main_mod.memory, "save_typed_fact",
                        lambda **kw: None)
    assert await main_mod._extract_and_store_facts("hello") == 0

    async def _three(_text):
        return [{"key": f"k{i}", "value": str(i)} for i in range(3)]

    monkeypatch.setattr(main_mod.llm, "extract_facts", _three)
    assert await main_mod._extract_and_store_facts("lots") == 3
