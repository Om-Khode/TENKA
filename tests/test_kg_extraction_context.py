"""Tests for knowledge-graph Session 2 Issue 1 — thread topic-tracker hint into the
entity-extraction prompt so the LLM can resolve pronouns.

Background: knowledge-graph Session 1 livetest produced no (aanya, works_at, Razorpay)
fact from the turn 'she works at Razorpay as a backend engineer'. The
extractor saw the turn in isolation and could not resolve 'she'. Fix:
pull the active topic from topic_tracker and inject it into the prompt
as a CONVERSATION CONTEXT section.

Pure unit + mocked extraction. Safe in isolation.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import _reset_for_testing, init_db


# ─── Prompt content — context_hint plumbing ───────────────────────────────


def _builder():
    from assistant.llm.contracts import _build_kg_extraction_prompt
    return _build_kg_extraction_prompt


def test_prompt_includes_context_section_when_hint_provided():
    """When context_hint is non-empty, the prompt must surface it in a
    section the LLM can read. Anchor: literal 'CONVERSATION CONTEXT' header."""
    prompt = _builder()("user_msg", context_hint="Active topic: Aanya")
    assert "CONVERSATION CONTEXT" in prompt
    assert "Aanya" in prompt


def test_prompt_omits_context_section_when_hint_is_none():
    """Default behavior unchanged when no hint is passed."""
    prompt = _builder()("user_msg", context_hint=None)
    assert "CONVERSATION CONTEXT" not in prompt


def test_prompt_omits_context_section_when_hint_is_empty_string():
    """Empty / whitespace hint must not produce a dangling header."""
    prompt = _builder()("user_msg", context_hint="")
    assert "CONVERSATION CONTEXT" not in prompt
    prompt = _builder()("user_msg", context_hint="   ")
    assert "CONVERSATION CONTEXT" not in prompt


# ─── topic_tracker module-level singleton ──────────────────────────────────


def test_topic_tracker_module_has_active_accessors():
    """topic_tracker exposes set_active / get_active so consumers outside
    main.py can share the same singleton instance."""
    import importlib
    import assistant.topic_tracker as tt
    importlib.reload(tt)  # clear any prior set_active state
    assert tt.get_active() is None
    tracker = tt.TopicTracker()
    tt.set_active(tracker)
    assert tt.get_active() is tracker
    tt.set_active(None)
    assert tt.get_active() is None


# ─── Integration: ingest_turn pulls hint from active tracker ──────────────


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def db(tmp_path):
    _reset_for_testing()
    db_obj = init_db(tmp_path / "test.db")
    yield db_obj
    db_obj.close()
    _reset_for_testing()


def test_ingest_turn_passes_active_tracker_hint_to_extraction(db):
    """ingest_turn must fetch the active tracker's topic hint and forward
    it to ask_for_entity_extraction as context_hint."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    kg_mod._repo = None
    kg_mod.init_kg()

    # Set up an active tracker with a known topic on the stack.
    tracker = tt.TopicTracker()
    tracker.push_turn("Tell me about Aanya", turn_number=1)
    tt.set_active(tracker)
    try:
        captured = {}

        async def fake_extract(text, source, context_hint=None):
            captured["text"] = text
            captured["source"] = source
            captured["context_hint"] = context_hint
            return {"entities": [], "facts": [], "relationships": []}

        with patch(
            "assistant.knowledge_graph.ask_for_entity_extraction",
            fake_extract, create=True,
        ):
            _run(kg_mod.ingest_turn("she works at Razorpay", "user_msg"))

        assert captured.get("context_hint") is not None
        assert "Aanya" in captured["context_hint"]
    finally:
        tt.set_active(None)


def test_ingest_turn_passes_none_hint_when_no_active_tracker(db):
    """When no tracker is active, ingest_turn must still work; context_hint
    is None and extraction proceeds normally."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    tt.set_active(None)
    kg_mod._repo = None
    kg_mod.init_kg()

    captured = {}

    async def fake_extract(text, source, context_hint=None):
        captured["context_hint"] = context_hint
        return {"entities": [], "facts": [], "relationships": []}

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        _run(kg_mod.ingest_turn("Aanya is in Bangalore", "user_msg"))

    # Either None or absent — both acceptable
    assert captured.get("context_hint") in (None, "")
