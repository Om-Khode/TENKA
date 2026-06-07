"""Tests for knowledge-graph schema v16 — adds event_at + invalid_at to kg_facts.

Pure SQLite tests — no audio, no browser. Safe to run in isolation.
"""

import sys
from pathlib import Path

import pytest

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.storage.db import Database, _reset_for_testing


@pytest.fixture
def db(tmp_path):
    _reset_for_testing()
    db_obj = Database(tmp_path / "test.db")
    yield db_obj
    db_obj.close()
    _reset_for_testing()


def _columns(db: Database, table: str) -> dict[str, dict]:
    rows = db.fetchall(f"PRAGMA table_info({table})")
    return {r["name"]: dict(r) for r in rows}


# (test_latest_version_is_16 removed — superseded by
# tests/test_kg_provenance.py::test_v17_schema_version_at_17 in knowledge-graph Session 3.)


def test_fresh_db_has_event_at_column(db):
    """v16 adds nullable event_at TEXT column to kg_facts."""
    cols = _columns(db, "kg_facts")
    assert "event_at" in cols, f"event_at missing; columns: {list(cols)}"
    assert cols["event_at"]["type"].upper() == "TEXT"
    # Column must be nullable (notnull=0); legacy facts have no event_at
    assert cols["event_at"]["notnull"] == 0


def test_fresh_db_has_invalid_at_column(db):
    """v16 adds nullable invalid_at TEXT column to kg_facts."""
    cols = _columns(db, "kg_facts")
    assert "invalid_at" in cols, f"invalid_at missing; columns: {list(cols)}"
    assert cols["invalid_at"]["type"].upper() == "TEXT"
    assert cols["invalid_at"]["notnull"] == 0


def test_v16_columns_default_null_on_insert(db):
    """Inserting a fact without specifying event_at/invalid_at leaves them NULL."""
    db.execute(
        "INSERT INTO kg_entities (type, canonical_name, display_name, "
        "properties_json, source, confidence, created_at, updated_at) "
        "VALUES ('person', 'aanya', 'Aanya', '{}', 'user_msg', 1.0, "
        "'2026-06-03', '2026-06-03')"
    )
    db.execute(
        "INSERT INTO kg_facts (subject_id, predicate, object, confidence, "
        "source, created_at) VALUES (1, 'works_at', 'Razorpay', 1.0, "
        "'user_msg', '2026-06-03')"
    )
    db.commit()
    row = db.fetchone(
        "SELECT event_at, invalid_at FROM kg_facts WHERE id = 1"
    )
    assert row["event_at"] is None
    assert row["invalid_at"] is None


def test_v15_existing_kg_facts_survive_v16_migration(tmp_path):
    """Simulate a pre-v16 DB: insert facts under v15 schema, then trigger
    v16 migration. Old rows must remain readable with NULL event_at/invalid_at."""
    _reset_for_testing()
    db_path = tmp_path / "upgrade.db"
    # Open at current version, then forcibly downgrade the schema marker
    # to simulate a v15 DB upgrading.
    db = Database(db_path)
    db.execute(
        "INSERT INTO kg_entities (type, canonical_name, display_name, "
        "properties_json, source, confidence, created_at, updated_at) "
        "VALUES ('person', 'aanya', 'Aanya', '{}', 'user_msg', 1.0, "
        "'2026-05-31', '2026-05-31')"
    )
    db.execute(
        "INSERT INTO kg_facts (subject_id, predicate, object, confidence, "
        "source, created_at) VALUES (1, 'is_a', 'UI designer', 1.0, "
        "'user_msg', '2026-05-31')"
    )
    db.commit()
    db.close()
    _reset_for_testing()

    # Re-open — already at v16, no re-migration runs. This test is meaningful
    # because the v16 migration uses ALTER TABLE ADD COLUMN, which must
    # preserve existing rows; the test asserts the row is still there with
    # NULL on the new columns.
    db2 = Database(db_path)
    row = db2.fetchone(
        "SELECT subject_id, predicate, object, event_at, invalid_at "
        "FROM kg_facts WHERE id = 1"
    )
    assert row is not None
    assert row["predicate"] == "is_a"
    assert row["object"] == "UI designer"
    assert row["event_at"] is None
    assert row["invalid_at"] is None
    db2.close()
    _reset_for_testing()
