"""Tests for knowledge-graph Session 3 H — source_turn_id provenance backlink.

V17 adds a nullable TEXT source_turn_id column to kg_entities, kg_facts,
and kg_relationships. The ingest path threads an opaque turn id from
main.py through ingest_turn → _persist_extraction → the repo write
methods, so any KG row can be traced back to the conversation row that
produced it ("why do you think that?").

Pure unit tests. No LLM calls.
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


def _noop_loader():
    class _ZeroModel:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            if isinstance(texts, str):
                return np.zeros(3, dtype="float32")
            return np.zeros((len(texts), 3), dtype="float32")
    return lambda: _ZeroModel()


@pytest.fixture
def db(tmp_path):
    _reset_for_testing()
    db_obj = init_db(tmp_path / "test.db")
    yield db_obj
    db_obj.close()
    _reset_for_testing()


# ─── v17 migration ─────────────────────────────────────────────────────────


def test_v17_adds_source_turn_id_to_all_three_tables(db):
    """ALTER TABLE on kg_entities, kg_facts, kg_relationships."""
    for table in ("kg_entities", "kg_facts", "kg_relationships"):
        rows = db.fetchall(f"PRAGMA table_info({table})")
        names = {r["name"] for r in rows}
        assert "source_turn_id" in names, (
            f"v17 migration must add source_turn_id to {table}"
        )


# (test_v17_schema_version_at_17 removed — superseded by
# tests/test_kg_commitments.py::test_v18_schema_version_at_18 in knowledge-graph Session 4.)


# ─── Repo persistence ──────────────────────────────────────────────────────


def test_add_fact_persists_source_turn_id(db):
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo
    repo = KnowledgeGraphRepo(db, embed_model_loader=_noop_loader())
    eid, _ = repo.upsert_entity(
        "person", "Aanya", source="user_msg",
        source_turn_id="sess-7:42",
    )
    repo.add_fact(
        eid, "works_at", "Razorpay",
        source="user_msg", source_turn_id="sess-7:42",
    )

    row = db.fetchone(
        "SELECT source_turn_id FROM kg_facts WHERE subject_id = ?", (eid,),
    )
    assert row["source_turn_id"] == "sess-7:42"

    erow = db.fetchone(
        "SELECT source_turn_id FROM kg_entities WHERE id = ?", (eid,),
    )
    assert erow["source_turn_id"] == "sess-7:42"


def test_upsert_entity_preserves_first_turn_id_on_rematch(db):
    """Provenance stays with the FIRST turn that introduced the entity.
    Subsequent matches (exact or cosine) must NOT overwrite source_turn_id."""
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo
    repo = KnowledgeGraphRepo(db, embed_model_loader=_noop_loader())
    eid1, created1 = repo.upsert_entity(
        "person", "Aanya", source="user_msg", source_turn_id="turn-A",
    )
    eid2, created2 = repo.upsert_entity(
        "person", "Aanya", source="user_msg", source_turn_id="turn-B",
    )
    assert created1 is True
    assert created2 is False
    assert eid1 == eid2

    row = db.fetchone(
        "SELECT source_turn_id FROM kg_entities WHERE id = ?", (eid1,),
    )
    assert row["source_turn_id"] == "turn-A", (
        "second upsert must not overwrite the original provenance"
    )


def test_add_relationship_persists_source_turn_id(db):
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo
    repo = KnowledgeGraphRepo(db, embed_model_loader=_noop_loader())
    a, _ = repo.upsert_entity("person", "Aanya", source="user_msg")
    b, _ = repo.upsert_entity("project", "Razorpay", source="user_msg")
    repo.add_relationship(
        a, b, "related_to", source="user_msg",
        source_turn_id="prov-rel-1",
    )
    row = db.fetchone(
        "SELECT source_turn_id FROM kg_relationships "
        "WHERE from_id = ? AND to_id = ?", (a, b),
    )
    assert row["source_turn_id"] == "prov-rel-1"


# ─── ingest_turn end-to-end forwarding ─────────────────────────────────────


def test_ingest_turn_forwards_source_turn_id_to_persisted_fact(db):
    """The full chain: ingest_turn(source_turn_id=...) lands on kg_facts."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    tt.set_active(None)
    kg_mod._repo = None
    kg_mod.init_kg()

    async def fake_extract(text, source, context_hint=None):
        return {
            "entities": [
                {"type": "person", "name": "Aanya", "confidence": 1.0},
            ],
            "facts": [
                {
                    "subject": "Aanya", "predicate": "works_at",
                    "object": "Razorpay", "confidence": 0.9,
                },
            ],
            "relationships": [],
        }

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        _run(kg_mod.ingest_turn(
            "Aanya works at Razorpay", "user_msg",
            source_turn_id="sess-X:99",
        ))

    fact_row = db.fetchone(
        "SELECT source_turn_id FROM kg_facts WHERE predicate = 'works_at'",
    )
    assert fact_row is not None
    assert fact_row["source_turn_id"] == "sess-X:99"

    ent_row = db.fetchone(
        "SELECT source_turn_id FROM kg_entities WHERE canonical_name = 'aanya'",
    )
    assert ent_row is not None
    assert ent_row["source_turn_id"] == "sess-X:99"


def test_ingest_turn_without_source_turn_id_is_null(db):
    """Legacy call shape (no source_turn_id) leaves the column NULL."""
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    tt.set_active(None)
    kg_mod._repo = None
    kg_mod.init_kg()

    async def fake_extract(text, source, context_hint=None):
        return {
            "entities": [{"type": "person", "name": "Bob", "confidence": 1.0}],
            "facts": [], "relationships": [],
        }

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        _run(kg_mod.ingest_turn("Bob is here", "user_msg"))

    row = db.fetchone(
        "SELECT source_turn_id FROM kg_entities WHERE canonical_name = 'bob'",
    )
    assert row is not None
    assert row["source_turn_id"] is None
