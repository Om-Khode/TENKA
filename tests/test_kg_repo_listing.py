"""Tests for KnowledgeGraphRepo.list_entities / list_facts / list_relationships
/ delete_entity — the Studio "browse and forget" surface. Real SQLite in a
tmp dir, not mocks: mocked DBs have masked migration failures here before.
"""
from pathlib import Path

import pytest

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo


@pytest.fixture
def repo(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield KnowledgeGraphRepo(db, embed_model_loader=lambda: None)
    db.close()
    _reset_for_testing()


def _seed(repo: KnowledgeGraphRepo) -> dict:
    """Two entities, three facts (one superseded), three relationships: one
    with the seeded entity as from_id, one as to_id (so delete_entity's
    "both directions" claim has something to prove), and one dangling --
    pointing at an entity id that was never created."""
    sister_id, _ = repo.upsert_entity("person", "sister", source="conversation")
    event_id, _ = repo.upsert_entity("event", "thesis defence", source="conversation")

    # lives_in Pune, then Bengaluru -- add_fact's own invalidation semantics
    # supersede the first row rather than deleting it.
    repo.add_fact(sister_id, "lives_in", "Pune", source="conversation")
    repo.add_fact(sister_id, "lives_in", "Bengaluru", source="conversation")
    repo.add_fact(event_id, "happens_on", "2026-09-04", source="conversation")

    repo.add_relationship(sister_id, event_id, "attending", source="conversation")
    repo.add_relationship(event_id, sister_id, "reminds", source="conversation")

    # A dangling edge: to_id 99999 was never inserted. FK enforcement (on in
    # this schema) refuses that INSERT outright, so this simulates the
    # legacy-row case the same way the Studio fake fixture does -- real
    # graphs do end up with edges whose far entity is gone. Foreign keys are
    # toggled off only for this one raw insert.
    repo._db.execute("PRAGMA foreign_keys = OFF")
    repo._db.execute(
        "INSERT INTO kg_relationships "
        "(from_id, to_id, type, properties_json, confidence, source, created_at) "
        "VALUES (?, ?, ?, '{}', 1.0, ?, ?)",
        (sister_id, 99_999, "mentions", "conversation", "2026-01-01T00:00:00"),
    )
    repo._db.commit()
    repo._db.execute("PRAGMA foreign_keys = ON")

    return {"sister_id": sister_id, "event_id": event_id}


# ─── list_entities ──────────────────────────────────────────────────────────
def test_list_entities_returns_every_entity(repo):
    _seed(repo)
    rows = repo.list_entities()
    assert {r["canonical_name"] for r in rows} == {"sister", "thesis defence"}


def test_list_entities_respects_limit(repo):
    _seed(repo)
    rows = repo.list_entities(limit=1)
    assert len(rows) == 1


# ─── list_facts ─────────────────────────────────────────────────────────────
def test_list_facts_includes_superseded_rows(repo):
    ids = _seed(repo)
    rows = repo.list_facts()
    assert len(rows) == 3
    pune = [r for r in rows if r["object"] == "Pune"][0]
    bengaluru = [r for r in rows if r["object"] == "Bengaluru"][0]
    assert pune["invalid_at"] is not None, "the superseded fact must still be listed"
    assert bengaluru["invalid_at"] is None
    assert pune["subject_id"] == ids["sister_id"]


# ─── list_relationships ─────────────────────────────────────────────────────
def test_list_relationships_includes_dangling_edges(repo):
    _seed(repo)
    rows = repo.list_relationships()
    assert len(rows) == 3
    dangling = [r for r in rows if r["to_id"] == 99_999]
    assert len(dangling) == 1
    assert dangling[0]["type"] == "mentions"


# ─── delete_entity ───────────────────────────────────────────────────────────
def test_delete_entity_removes_entity_facts_and_both_edge_directions(repo):
    ids = _seed(repo)

    assert repo.delete_entity(ids["sister_id"]) is True

    assert repo.get_entity(ids["sister_id"]) is None
    remaining_facts = repo.list_facts()
    assert all(f["subject_id"] != ids["sister_id"] for f in remaining_facts)
    remaining_rels = repo.list_relationships()
    assert all(
        r["from_id"] != ids["sister_id"] and r["to_id"] != ids["sister_id"]
        for r in remaining_rels
    )


def test_delete_entity_leaves_the_other_entity_untouched(repo):
    ids = _seed(repo)
    repo.delete_entity(ids["sister_id"])
    assert repo.get_entity(ids["event_id"]) is not None
    remaining_facts = repo.list_facts()
    assert any(f["subject_id"] == ids["event_id"] for f in remaining_facts)


def test_delete_entity_returns_false_for_an_unknown_id(repo):
    assert repo.delete_entity(123_456) is False
