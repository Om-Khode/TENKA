"""Tests for knowledge-graph Session 4 H — why_do_you_think_that read-path helper.

Given a KG row (fact / entity / commitment), the helper resolves
source_turn_id back to the originating conversations row, closing the
Zep-pattern episodic backlink the schema-v17 column opened.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import _reset_for_testing, init_db


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


def _seed_conv_row(db_obj, session_id: str, user_input: str) -> int:
    """Insert a conversations row directly and return its id, matching the
    MemoryRepo.save_turn schema."""
    cur = db_obj.execute(
        "INSERT INTO conversations (timestamp, user_input, intent, response, "
        "session_id) VALUES (?, ?, ?, ?, ?)",
        ("2026-06-03T12:00:00", user_input, "small_talk", "ok", session_id),
    )
    db_obj.commit()
    return cur.lastrowid


def test_why_resolves_fact_to_conversation(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()

    conv_id = _seed_conv_row(db, "sess-A", "she works at Razorpay")
    turn_id = f"sess-A:{conv_id}"

    eid, _ = repo.upsert_entity(
        "person", "Aanya", source="user_msg", source_turn_id=turn_id,
    )
    repo.add_fact(
        eid, "works_at", "Razorpay",
        source="user_msg", source_turn_id=turn_id,
    )
    fact_row = db.fetchone("SELECT id FROM kg_facts WHERE subject_id = ?", (eid,))
    fact_id = fact_row["id"]

    out = kg_mod.why_do_you_think_that(fact_id=fact_id)
    assert out is not None
    assert out["source_turn_id"] == turn_id
    assert out["conversation"] is not None
    assert out["conversation"]["user_input"] == "she works at Razorpay"


def test_why_resolves_entity_to_conversation(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    conv_id = _seed_conv_row(db, "sess-B", "Aanya joined Razorpay")
    eid, _ = repo.upsert_entity(
        "person", "Aanya", source="user_msg",
        source_turn_id=f"sess-B:{conv_id}",
    )
    out = kg_mod.why_do_you_think_that(entity_id=eid)
    assert out["conversation"]["user_input"] == "Aanya joined Razorpay"


def test_why_resolves_commitment_to_conversation(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    uid, _ = repo.upsert_entity("person", "user", source="user_msg")
    conv_id = _seed_conv_row(db, "sess-C", "I'll send the deck tomorrow")
    cid = repo.add_commitment(
        uid, "send the deck", source="user_msg",
        source_turn_id=f"sess-C:{conv_id}",
    )
    out = kg_mod.why_do_you_think_that(commitment_id=cid)
    assert out["conversation"]["user_input"] == "I'll send the deck tomorrow"


def test_why_returns_none_when_no_provenance(db):
    """A KG row inserted with source_turn_id=None should resolve to None
    so the caller can fall back gracefully."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    eid, _ = repo.upsert_entity("person", "Aanya", source="user_msg")
    assert kg_mod.why_do_you_think_that(entity_id=eid) is None


def test_why_returns_none_for_unknown_id(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    assert kg_mod.why_do_you_think_that(fact_id=99999) is None


def test_why_tolerates_freeform_turn_id(db):
    """source_turn_id is opaque TEXT — callers may write any shape. The
    helper returns the id verbatim and conversation=None when it can't
    parse a conv_row_id back out of it."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    eid, _ = repo.upsert_entity(
        "person", "Aanya", source="user_msg",
        source_turn_id="freeform-no-colon",
    )
    out = kg_mod.why_do_you_think_that(entity_id=eid)
    assert out is not None
    assert out["source_turn_id"] == "freeform-no-colon"
    assert out["conversation"] is None


def test_why_rejects_no_arguments(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    assert kg_mod.why_do_you_think_that() is None
