"""Tests for knowledge-graph items A (event_at field) + K (subject-fact lifting).

A: extraction prompt must include event_at in the fact schema; _persist_extraction
   must pass event_at through to repo.add_fact.

K: extraction prompt must guide person-employer + family relations into
   subject-facts on the person, not concept-to-concept relationships.

Prompt-content tests assert anchor strings. They are intentionally
conservative — they verify intent without locking in exact wording. Per
CLAUDE.md gotcha, prompt examples must NOT match these test cases, so the
test placeholders ("Alpha Co", "uncle Damien") are deliberately distinct
from the prompt's own examples.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import _reset_for_testing, init_db


# ─── A part 2 + K: prompt content ──────────────────────────────────────────


def test_prompt_has_event_at_field_in_fact_schema():
    """A: fact schema in the extraction prompt must mention event_at."""
    from assistant.llm.contracts import _build_kg_extraction_prompt
    prompt = _build_kg_extraction_prompt("user_msg")
    assert "event_at" in prompt, (
        "extraction prompt must mention event_at so the LLM emits it on facts"
    )


def test_prompt_has_person_employer_guidance():
    """K: prompt must instruct LLM to emit person-employer subject-facts
    (works_at / has_role), not concept-to-concept relationships."""
    from assistant.llm.contracts import _build_kg_extraction_prompt
    prompt = _build_kg_extraction_prompt("user_msg")
    # Either explicit predicate name OR the guidance phrasing must appear.
    has_predicate_hint = "works_at" in prompt
    has_role_hint = "has_role" in prompt or "role" in prompt.lower()
    assert has_predicate_hint, (
        "expected 'works_at' guidance for person-employer facts"
    )
    assert has_role_hint, (
        "expected role-related guidance for person-job facts"
    )


def test_prompt_has_family_relation_guidance():
    """K: prompt must instruct LLM to emit family-relation subject-facts
    so 'my brother X' produces a fact about the relationship, not just an
    isolated entity."""
    from assistant.llm.contracts import _build_kg_extraction_prompt
    prompt = _build_kg_extraction_prompt("user_msg")
    lower = prompt.lower()
    family_terms = ("brother", "sister", "family", "sibling", "parent",
                    "mother", "father", "relative")
    matched = [t for t in family_terms if t in lower]
    assert matched, (
        f"expected at least one family-relation term in prompt; none of "
        f"{family_terms} appeared"
    )


# ─── A part 2: _persist_extraction passes event_at to add_fact ─────────────


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


def test_persist_extraction_passes_event_at_through_to_repo(db):
    """When a fact in the extraction payload carries event_at, it must
    land in the DB column. Closes the wire from contracts → persist → repo."""
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
                "event_at": "2025-11-20",
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
        # Text must contain an explicit 4-digit year so the strengthened
        # event_at filter (Session 2) preserves the value.
        _run(kg_mod.ingest_turn("Mira moved to Hyderabad in November 2025.", "user_msg"))

    rows = db.fetchall(
        "SELECT f.event_at, f.predicate, f.object "
        "FROM kg_facts f JOIN kg_entities e ON e.id = f.subject_id "
        "WHERE e.canonical_name = 'mira'"
    )
    assert len(rows) == 1
    assert rows[0]["predicate"] == "moved_to"
    assert rows[0]["object"] == "Hyderabad"
    assert rows[0]["event_at"] == "2025-11-20"
