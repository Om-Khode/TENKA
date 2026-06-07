"""Tests for knowledge-graph Session 2 Issue 2 — strip unfounded event_at.

Background: Flash-Lite occasionally violates the prompt rule and emits
event_at for bare relative phrases ("last month" → "2024-05"), guessing
the year. This defensive filter at the persist layer strips event_at when
the turn text shows only relative phrasing without an explicit calendar
anchor (4-digit year or absolute date).

Pure unit + SQLite. Safe in isolation.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import _reset_for_testing, init_db


# ─── Pure helper ──────────────────────────────────────────────────────────


def _strip():
    from assistant.knowledge_graph import _strip_unfounded_event_at
    return _strip_unfounded_event_at


def test_strip_when_text_uses_bare_last_month():
    """'last month' with no explicit year — LLM likely guessed the year."""
    assert _strip()("she moved to Bangalore last month", "2024-05") is None


def test_strip_when_text_uses_yesterday():
    assert _strip()("I joined yesterday", "2026-06-02") is None


def test_strip_when_text_uses_next_friday():
    assert _strip()("the kickoff is next Friday afternoon", "2026-06-06") is None


def test_keep_when_text_has_explicit_4digit_year():
    """If the user said '2024' explicitly, event_at=2024-something is grounded."""
    assert _strip()("she moved to Bangalore in March 2024", "2024-03") == "2024-03"


def test_keep_when_text_has_explicit_year_only():
    assert _strip()("she joined in 2024", "2024") == "2024"


def test_strip_when_no_calendar_anchor_in_text():
    """Conservative defense (strengthened mid-Session-2 after livetest): if
    text has no explicit year, strip event_at regardless of other phrasing.
    Flash-Lite hallucinates years aggressively from month names alone."""
    # "in December" — no year, even though "December" is a calendar word
    assert _strip()("we are planning a Goa trip in December", "2023-12") is None
    # No temporal phrasing at all — strip (LLM made it up)
    assert _strip()("she works at Razorpay", "2024-08") is None
    # "joined Voyager" — no date cue, strip
    assert _strip()("I joined a new project called Voyager", "2024-05-16") is None


def test_keep_when_event_at_already_none():
    assert _strip()("anything goes here", None) is None
    assert _strip()("anything", "") is None


def test_strip_when_event_at_is_unparseable_placeholder():
    """Session 2 livetest: Flash-Lite sometimes echoes the schema
    placeholder ('YYYY-MM-DD or YYYY-MM or YYYY or omit') or invents
    placeholders ('XXXX-12'). Anything that isn't a parseable ISO date
    must be dropped before reaching the DB."""
    # Literal prompt-template leak
    assert _strip()("any text", "YYYY-MM-DD") is None
    assert _strip()("any text", "YYYY-MM-DD or YYYY-MM or YYYY or omit") is None
    # Invented placeholder with X / Y / Z stand-ins
    assert _strip()("plans for trip in December", "XXXX-12") is None
    # Garbage strings
    assert _strip()("Aanya moved last month", "soon") is None
    assert _strip()("text", "not-a-date") is None


# ─── Integration: _persist_extraction applies the filter ──────────────────


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


def test_persist_strips_unfounded_event_at_on_relative_phrase(db):
    """Mocked LLM payload mimics the bug from knowledge-graph Session 1 livetest:
    text contains only 'last month', LLM still emits event_at=2024-05.
    Persist must drop event_at before writing to the DB."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "person", "name": "Aanya", "confidence": 1.0},
        ],
        "facts": [
            {
                "subject": "Aanya",
                "predicate": "moved_to",
                "object": "Bangalore",
                "confidence": 1.0,
                "event_at": "2024-05",
            },
        ],
        "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        _run(kg_mod.ingest_turn(
            "Aanya moved to Bangalore last month", "user_msg",
        ))

    row = db.fetchone(
        "SELECT f.event_at, f.object FROM kg_facts f "
        "JOIN kg_entities e ON e.id = f.subject_id "
        "WHERE e.canonical_name = 'aanya' AND f.predicate = 'moved_to'"
    )
    assert row is not None
    assert row["object"] == "Bangalore"
    assert row["event_at"] is None, (
        f"event_at should have been stripped; got {row['event_at']!r}"
    )


def test_persist_keeps_event_at_when_text_has_explicit_year(db):
    """Same shape, but text contains '2024' — event_at survives."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()

    payload = {
        "entities": [
            {"type": "person", "name": "Mira", "confidence": 1.0},
        ],
        "facts": [
            {
                "subject": "Mira",
                "predicate": "moved_to",
                "object": "Hyderabad",
                "confidence": 1.0,
                "event_at": "2024-11",
            },
        ],
        "relationships": [],
    }

    async def fake_extract(text, source, context_hint=None):
        return payload

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        _run(kg_mod.ingest_turn(
            "Mira moved to Hyderabad in November 2024", "user_msg",
        ))

    row = db.fetchone(
        "SELECT event_at FROM kg_facts WHERE predicate = 'moved_to'"
    )
    assert row["event_at"] == "2024-11"
