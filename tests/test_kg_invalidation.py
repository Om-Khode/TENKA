"""Tests for knowledge-graph item B — fact invalidation on object-change.

Semantics:
  * Same (subject, predicate, object) repeated → UPSERT, max(confidence).
    The existing row remains the only row for that triple.
  * Same (subject, predicate) with a DIFFERENT object, while any
    currently-valid row exists for that (subject, predicate): mark all
    currently-valid rows invalid_at=now, then INSERT a new row.
  * Re-adding the originally-invalidated (subject, predicate, object) while
    a newer object is currently valid: restore the old row
    (invalid_at = NULL, raise confidence) AND invalidate the currently-valid
    row. UNIQUE(subject_id, predicate, object) makes this the only sensible
    handling — we cannot INSERT a second row with the same triple.
  * Retrieval (get_facts_for_entity) must filter out invalid_at IS NOT NULL.

Pure SQLite — no audio, no browser. Safe in isolation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import Database, _reset_for_testing, init_db
from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo


def _noop_loader():
    class _M:
        def encode(self, *_a, **_kw):
            import numpy as np
            return np.zeros(3)
    return lambda: _M()


@pytest.fixture
def kg_repo(tmp_path):
    _reset_for_testing()
    db = init_db(tmp_path / "test.db")
    repo = KnowledgeGraphRepo(db, embed_model_loader=_noop_loader())
    yield repo
    db.close()
    _reset_for_testing()


def _seed_entity(repo: KnowledgeGraphRepo, name: str = "subj") -> int:
    eid, _ = repo.upsert_entity(entity_type="person", name=name, source="user_msg")
    return eid


def _all_facts(db: Database, subject_id: int) -> list[dict]:
    """Includes invalidated rows — for assertion only."""
    rows = db.fetchall(
        "SELECT id, predicate, object, confidence, invalid_at "
        "FROM kg_facts WHERE subject_id = ? ORDER BY id",
        (subject_id,),
    )
    return [dict(r) for r in rows]


def test_same_triple_upserts_max_confidence(kg_repo):
    """Adding the same triple twice keeps one row and raises confidence."""
    sid = _seed_entity(kg_repo)
    id1 = kg_repo.add_fact(sid, "is_a", "engineer", source="user_msg", confidence=0.6)
    id2 = kg_repo.add_fact(sid, "is_a", "engineer", source="user_msg", confidence=0.9)

    assert id1 == id2
    rows = _all_facts(kg_repo._db, sid)
    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.9
    assert rows[0]["invalid_at"] is None


def test_object_change_invalidates_old_and_inserts_new(kg_repo):
    """Same (subject, predicate) with a new object: old goes invalid, new is valid."""
    sid = _seed_entity(kg_repo)
    old_id = kg_repo.add_fact(sid, "works_at", "OldCo", source="user_msg")
    new_id = kg_repo.add_fact(sid, "works_at", "NewCo", source="user_msg")

    assert old_id != new_id
    rows = {r["id"]: r for r in _all_facts(kg_repo._db, sid)}
    assert len(rows) == 2
    assert rows[old_id]["object"] == "OldCo"
    assert rows[old_id]["invalid_at"] is not None
    # Verify ISO 8601-ish timestamp shape
    assert "T" in rows[old_id]["invalid_at"] or "-" in rows[old_id]["invalid_at"]
    assert rows[new_id]["object"] == "NewCo"
    assert rows[new_id]["invalid_at"] is None


def test_chained_object_changes_invalidate_each_previous(kg_repo):
    """intern → engineer → senior engineer: only the latest is currently valid."""
    sid = _seed_entity(kg_repo)
    kg_repo.add_fact(sid, "role", "intern", source="user_msg")
    kg_repo.add_fact(sid, "role", "engineer", source="user_msg")
    kg_repo.add_fact(sid, "role", "senior engineer", source="user_msg")

    rows = _all_facts(kg_repo._db, sid)
    assert len(rows) == 3
    valid = [r for r in rows if r["invalid_at"] is None]
    invalid = [r for r in rows if r["invalid_at"] is not None]
    assert len(valid) == 1
    assert valid[0]["object"] == "senior engineer"
    invalid_objs = sorted(r["object"] for r in invalid)
    assert invalid_objs == ["engineer", "intern"]


def test_get_facts_for_entity_excludes_invalidated(kg_repo):
    """The sole repo read-path must filter invalid_at IS NOT NULL."""
    sid = _seed_entity(kg_repo)
    kg_repo.add_fact(sid, "lives_in", "Berlin", source="user_msg")
    kg_repo.add_fact(sid, "lives_in", "Bangalore", source="user_msg")

    facts = kg_repo.get_facts_for_entity(sid)
    assert len(facts) == 1
    assert facts[0]["object"] == "Bangalore"
    assert facts[0]["invalid_at"] is None


def test_expand_entity_context_excludes_invalidated(kg_repo):
    """Integration: query-time context block must not see stale facts."""
    sid = _seed_entity(kg_repo)
    kg_repo.add_fact(sid, "works_at", "OldCo", source="user_msg")
    kg_repo.add_fact(sid, "works_at", "NewCo", source="user_msg")

    ctx = kg_repo.expand_entity_context(sid)
    objs = [f["object"] for f in ctx["facts"]]
    assert "OldCo" not in objs
    assert "NewCo" in objs


def test_restoration_revalidates_old_and_invalidates_current(kg_repo):
    """User says X is engineer, then manager, then engineer again: UNIQUE
    forces restoration of the original row (clear invalid_at), and the
    intermediate 'manager' row becomes invalid."""
    sid = _seed_entity(kg_repo)
    eng_id = kg_repo.add_fact(sid, "role", "engineer", source="user_msg")
    mgr_id = kg_repo.add_fact(sid, "role", "manager", source="user_msg")
    eng_id_again = kg_repo.add_fact(sid, "role", "engineer", source="user_msg", confidence=0.5)

    # Restoration UPDATES the original row, doesn't insert a new one
    assert eng_id_again == eng_id

    rows = {r["id"]: r for r in _all_facts(kg_repo._db, sid)}
    assert len(rows) == 2  # no third row inserted
    assert rows[eng_id]["invalid_at"] is None  # restored
    # Confidence: was 1.0 (default first call); 0.5 max is still 1.0
    assert rows[eng_id]["confidence"] == 1.0
    assert rows[mgr_id]["invalid_at"] is not None  # newly invalidated
