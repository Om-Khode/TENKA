"""Tests for knowledge-graph Session 4 E — commitments table + extractor + intent.

Covers:
- schema v18 (kg_commitments + indexes)
- KnowledgeGraphRepo.add_commitment / list_open_commitments /
  mark_commitment_fulfilled / find_commitments_by_text
- expand_entity_context now returns commitments
- ingest_turn extracts and persists commitments (mocked LLM)
- assistant.knowledge_graph facade helpers (list_open_commitments_*)
- memory_search._is_commitment_query
- handle_memory_query surfaces commitments when the query mentions
  promises (mocked hybrid-retrieval returns nothing so the path is forced)
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


# ─── Schema v18 ────────────────────────────────────────────────────────────


def test_v18_creates_kg_commitments_table(db):
    rows = db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_commitments'"
    )
    assert len(rows) == 1


def test_v18_kg_commitments_columns_present(db):
    rows = db.fetchall("PRAGMA table_info(kg_commitments)")
    names = {r["name"] for r in rows}
    expected = {
        "id", "owner_id", "promise_text", "when_due", "created_at",
        "fulfilled_at", "source", "source_turn_id", "reminder_id",
    }
    assert expected <= names


def test_v18_schema_version_at_18(db):
    row = db.fetchone("SELECT version FROM _schema_version WHERE id = 1")
    assert row["version"] == 18


def test_v18_indexes_present(db):
    rows = db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND name LIKE 'idx_kg_commitments_%'"
    )
    names = {r["name"] for r in rows}
    expected = {
        "idx_kg_commitments_owner",
        "idx_kg_commitments_open",
        "idx_kg_commitments_due",
    }
    assert expected <= names


# ─── Repo ops ──────────────────────────────────────────────────────────────


def _seed_user(db_obj):
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo
    repo = KnowledgeGraphRepo(db_obj, embed_model_loader=_noop_loader())
    uid, _ = repo.upsert_entity("person", "user", source="user_msg")
    return repo, uid


def test_add_commitment_persists_fields(db):
    repo, uid = _seed_user(db)
    cid = repo.add_commitment(
        uid, "send the file by Friday",
        source="user_msg", when_due="2026-06-05",
        source_turn_id="t-1", reminder_id=42,
    )
    row = db.fetchone("SELECT * FROM kg_commitments WHERE id = ?", (cid,))
    assert row["owner_id"] == uid
    assert row["promise_text"] == "send the file by Friday"
    assert row["when_due"] == "2026-06-05"
    assert row["source_turn_id"] == "t-1"
    assert row["reminder_id"] == 42
    assert row["fulfilled_at"] is None


def test_list_open_commitments_excludes_fulfilled(db):
    repo, uid = _seed_user(db)
    open_id = repo.add_commitment(uid, "call mom", source="user_msg")
    closed_id = repo.add_commitment(uid, "buy milk", source="user_msg")
    repo.mark_commitment_fulfilled(closed_id)
    rows = repo.list_open_commitments(owner_id=uid)
    ids = {r["id"] for r in rows}
    assert open_id in ids
    assert closed_id not in ids


def test_mark_commitment_fulfilled_returns_false_for_unknown(db):
    repo, _ = _seed_user(db)
    assert repo.mark_commitment_fulfilled(99999) is False


def test_mark_commitment_fulfilled_idempotent(db):
    repo, uid = _seed_user(db)
    cid = repo.add_commitment(uid, "call mom", source="user_msg")
    assert repo.mark_commitment_fulfilled(cid) is True
    assert repo.mark_commitment_fulfilled(cid) is False  # already done


def test_find_commitments_by_text_substring(db):
    repo, uid = _seed_user(db)
    a = repo.add_commitment(uid, "send the report to Priya", source="user_msg")
    b = repo.add_commitment(uid, "buy milk", source="user_msg")
    rows = repo.find_commitments_by_text("report")
    ids = {r["id"] for r in rows}
    assert a in ids
    assert b not in ids


def test_expand_entity_context_includes_commitments(db):
    repo, uid = _seed_user(db)
    repo.add_commitment(uid, "review PR-42", source="user_msg")
    ctx = repo.expand_entity_context(uid)
    promises = [c["promise_text"] for c in ctx["commitments"]]
    assert "review PR-42" in promises


# ─── Extractor prompt + persistence ────────────────────────────────────────


def test_extraction_prompt_describes_commitments():
    from assistant.llm.contracts import _build_kg_extraction_prompt
    prompt = _build_kg_extraction_prompt("user_msg")
    assert "commitments" in prompt
    assert "promise" in prompt.lower()


def test_ingest_turn_persists_commitment_for_user(db):
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    tt.set_active(None)
    kg_mod._repo = None
    kg_mod.init_kg()

    async def fake_extract(text, source, context_hint=None):
        return {
            "entities": [],
            "facts": [],
            "relationships": [],
            "commitments": [
                {"owner": "user", "promise": "send the deck", "when_due": None},
            ],
        }

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        # Capitalized noun ("Priya") makes the text clear the
        # _has_entity_signal pre-filter so the extractor actually runs.
        _run(kg_mod.ingest_turn(
            "I promised Priya I would send the deck tomorrow", "user_msg",
            source_turn_id="t-9",
        ))

    rows = db.fetchall(
        "SELECT promise_text, source_turn_id FROM kg_commitments"
    )
    assert any(r["promise_text"] == "send the deck" for r in rows)
    assert any(r["source_turn_id"] == "t-9" for r in rows)


def test_ingest_turn_skips_commitment_without_promise_text(db):
    import assistant.knowledge_graph as kg_mod
    import assistant.topic_tracker as tt
    tt.set_active(None)
    kg_mod._repo = None
    kg_mod.init_kg()

    async def fake_extract(text, source, context_hint=None):
        return {
            "entities": [], "facts": [], "relationships": [],
            "commitments": [{"owner": "user", "promise": "  "}],
        }

    with patch(
        "assistant.knowledge_graph.ask_for_entity_extraction",
        fake_extract, create=True,
    ):
        # Must also pass the pre-filter so we genuinely exercise the
        # "promise text empty → skip" code path inside _persist_extraction.
        _run(kg_mod.ingest_turn("Aanya mentioned the project today", "user_msg"))

    rows = db.fetchall("SELECT COUNT(*) AS n FROM kg_commitments")
    assert rows[0]["n"] == 0


# ─── Facade helpers ────────────────────────────────────────────────────────


def test_list_open_commitments_for_user_no_user_entity(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    assert kg_mod.list_open_commitments_for_user() == []


def test_list_open_commitments_for_user_resolves_canonical(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    uid, _ = repo.upsert_entity("person", "user", source="user_msg")
    repo.add_commitment(uid, "draft the spec", source="user_msg")
    out = kg_mod.list_open_commitments_for_user()
    assert any(c["promise_text"] == "draft the spec" for c in out)


def test_list_open_commitments_for_entity(db):
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    aanya_id, _ = repo.upsert_entity("person", "Aanya", source="user_msg")
    repo.add_commitment(aanya_id, "review the design", source="user_msg")
    out = kg_mod.list_open_commitments_for_entity(aanya_id)
    assert len(out) == 1


# ─── _is_commitment_query + memory_query integration ──────────────────────


def test_is_commitment_query_positive_cases():
    from assistant.actions.memory_search import _is_commitment_query
    for q in [
        "what did I promise Aanya",
        "what have I committed to",
        "did I owe Priya something",
        "what am I supposed to do this week",
    ]:
        assert _is_commitment_query(q), q


def test_is_commitment_query_negative_cases():
    from assistant.actions.memory_search import _is_commitment_query
    for q in [
        "what's the weather like",
        "where does Aanya work",
        "play some music",
    ]:
        assert not _is_commitment_query(q), q


def test_handle_memory_query_surfaces_commitments(db, monkeypatch):
    """When the query mentions promises and the user has open commitments,
    they appear in the synthesis prompt."""
    import assistant.knowledge_graph as kg_mod
    kg_mod._repo = None
    kg_mod.init_kg()
    repo = kg_mod._get_repo()
    uid, _ = repo.upsert_entity("person", "user", source="user_msg")
    repo.add_commitment(uid, "send the report to Priya", source="user_msg")

    from assistant.actions import memory_search

    monkeypatch.setattr(
        "assistant.memory.hybrid_search_facts", lambda q, limit=10: []
    )
    monkeypatch.setattr(
        "assistant.memory.hybrid_search_conversations", lambda q, limit=5: []
    )
    monkeypatch.setattr(
        "assistant.memory.search_recording_sessions", lambda q, limit=3: []
    )
    monkeypatch.setattr(kg_mod, "search_entities", lambda q: [])

    captured = {}

    async def fake_synth(prompt, **kw):
        captured["prompt"] = prompt
        return "synth-out"

    monkeypatch.setattr("assistant.llm.contracts.ask_for_synthesis", fake_synth)
    _run(memory_search.handle_memory_query(
        {"query": "what did I promise this week"}, "",
    ))
    assert "OPEN PROMISES" in captured["prompt"]
    assert "send the report to Priya" in captured["prompt"]
