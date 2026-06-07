"""Tests for knowledge-graph Session 3 Path A — deterministic pre-resolution.

Before calling ask_for_entity_extraction, ingest_turn now substitutes
pronouns against the active topic_tracker via resolve_query(). This is a
zero-LLM, deterministic fix for the Session 2 finding that Flash-Lite
didn't reliably honour the context_hint string. Path A and the hint
both fire (belt-and-braces): pre-resolution gives the LLM a clean
concrete-name input; the hint remains as a fallback for cases
substitution can't reach.

Pure unit + mocked extraction. Safe in isolation.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import _reset_for_testing, init_db


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


def test_pre_resolved_text_reaches_extraction(db):
    """Pronoun in the user turn is substituted with the active topic before
    the extractor sees it. The LLM no longer has to guess what 'she' means."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    kg_mod._repo = None
    kg_mod.init_kg()

    tracker = tt.TopicTracker()
    tracker.push_turn("Tell me about Aanya", turn_number=1)
    tt.set_active(tracker)
    try:
        captured = {}

        async def fake_extract(text, source, context_hint=None):
            captured["text"] = text
            captured["context_hint"] = context_hint
            return {"entities": [], "facts": [], "relationships": []}

        with patch(
            "assistant.knowledge_graph.ask_for_entity_extraction",
            fake_extract, create=True,
        ):
            _run(kg_mod.ingest_turn("she works at Razorpay", "user_msg"))

        # The LLM now sees the resolved form: 'Aanya works at Razorpay'.
        assert "she" not in captured["text"].lower().split()
        assert "Aanya" in captured["text"]
        # Belt-and-braces: the context_hint is also still populated.
        assert captured["context_hint"] is not None
        assert "Aanya" in captured["context_hint"]
    finally:
        tt.set_active(None)


def test_no_pronoun_text_passes_through_unchanged(db):
    """When the user turn has no pronouns, pre-resolution is a no-op and the
    extractor receives the exact original text."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    kg_mod._repo = None
    kg_mod.init_kg()

    tracker = tt.TopicTracker()
    tracker.push_turn("Tell me about Aanya", turn_number=1)
    tt.set_active(tracker)
    try:
        captured = {}

        async def fake_extract(text, source, context_hint=None):
            captured["text"] = text
            return {"entities": [], "facts": [], "relationships": []}

        with patch(
            "assistant.knowledge_graph.ask_for_entity_extraction",
            fake_extract, create=True,
        ):
            _run(kg_mod.ingest_turn("Razorpay raised a Series E round", "user_msg"))

        assert captured["text"] == "Razorpay raised a Series E round"
    finally:
        tt.set_active(None)


def test_self_referent_fact_dropped(db):
    """Defensive filter (Session 4 livetest): facts where
    canonicalized subject == object are dropped at persist time.
    Flash-Lite occasionally mis-labels the object as the subject — the
    signal is corrupt either way, better to drop than store."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    tt.set_active(None)
    kg_mod._repo = None
    kg_mod.init_kg()

    async def fake_extract(text, source, context_hint=None):
        return {
            "entities": [
                {"type": "person", "name": "Razorpay", "confidence": 1.0},
            ],
            "facts": [
                {"subject": "Razorpay", "predicate": "works_at",
                 "object": "Razorpay", "confidence": 1.0},
                {"subject": "Razorpay", "predicate": "has_role",
                 "object": "backend engineer", "confidence": 1.0},
            ],
            "relationships": [],
            "commitments": [],
        }

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        _run(kg_mod.ingest_turn("Razorpay works at Razorpay", "user_msg"))

    rows = db.fetchall("SELECT subject_id, predicate, object FROM kg_facts")
    preds = [r["predicate"] for r in rows]
    # The (Razorpay, works_at, Razorpay) self-referent fact must be dropped.
    assert "works_at" not in preds
    # The other fact (different object) survives.
    assert "has_role" in preds


def test_no_active_tracker_falls_back_to_raw_text(db):
    """When no tracker is registered (boot ordering, isolated tests), ingest
    still works and the raw text reaches the extractor unchanged."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    tt.set_active(None)
    kg_mod._repo = None
    kg_mod.init_kg()

    captured = {}

    async def fake_extract(text, source, context_hint=None):
        captured["text"] = text
        captured["context_hint"] = context_hint
        return {"entities": [], "facts": [], "relationships": []}

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        _run(kg_mod.ingest_turn("she works at Razorpay", "user_msg"))

    # No tracker → no resolution. The LLM gets the raw user text and the
    # context_hint is None. This is the documented degraded mode.
    assert captured["text"] == "she works at Razorpay"
    assert captured["context_hint"] in (None, "")
