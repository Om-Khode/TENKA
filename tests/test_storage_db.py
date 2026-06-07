"""Tests for storage/db.py — Database class + migrations."""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.storage.db import Database, init_db, get_db, _reset_for_testing


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Reset the module singleton before/after each test."""
    _reset_for_testing()
    yield
    _reset_for_testing()


# --- Database init ---

def test_creates_db_file(tmp_path):
    db_path = tmp_path / "memory" / "personality.db"
    db = Database(db_path)
    assert db_path.exists()
    db.close()


def test_wal_mode(tmp_path):
    db = Database(tmp_path / "test.db")
    row = db.fetchone("PRAGMA journal_mode")
    assert row[0] == "wal"
    db.close()


def test_row_factory(tmp_path):
    db = Database(tmp_path / "test.db")
    db.execute("CREATE TABLE t (name TEXT)")
    db.execute("INSERT INTO t VALUES (?)", ("hello",))
    db.commit()
    row = db.fetchone("SELECT name FROM t")
    assert row["name"] == "hello"
    db.close()


# --- Schema versioning ---

def test_fresh_db_at_latest_version(tmp_path):
    db = Database(tmp_path / "test.db")
    assert db._get_version() == Database._LATEST_VERSION
    db.close()


def test_creates_all_tables(tmp_path):
    db = Database(tmp_path / "test.db")
    tables = {
        r["name"]
        for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    expected = {
        "_schema_version",
        "personality_state", "personality_log", "metadata",
        "user_preferences", "preference_log",
        "user_procedures",
        "runtime_settings",
        "user_shortcuts",
    }
    assert expected <= tables
    db.close()


def test_existing_db_no_data_loss(tmp_path):
    """Opening an existing DB with data preserves rows."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.execute(
        "INSERT INTO runtime_settings (key, value, updated_at, updated_source) "
        "VALUES (?, ?, ?, ?)",
        ("theme", '"dark"', "2026-01-01T00:00:00", "user"),
    )
    db.commit()
    db.close()

    db2 = Database(db_path)
    row = db2.fetchone("SELECT value FROM runtime_settings WHERE key = ?", ("theme",))
    assert row["value"] == '"dark"'
    db2.close()


def test_migration_idempotent(tmp_path):
    """Running migrations on an already-current DB is a no-op."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    v1 = db._get_version()
    db.close()

    db2 = Database(db_path)
    assert db2._get_version() == v1
    db2.close()


# --- Singleton ---

def test_init_db_returns_instance(tmp_path):
    db = init_db(tmp_path / "test.db")
    assert db is not None
    assert get_db() is db


def test_init_db_idempotent(tmp_path):
    db1 = init_db(tmp_path / "test.db")
    db2 = init_db(tmp_path / "test.db")
    assert db1 is db2


def test_get_db_before_init():
    assert get_db() is None


# --- Helpers ---

def test_fetchall(tmp_path):
    db = Database(tmp_path / "test.db")
    db.execute(
        "INSERT INTO runtime_settings (key, value, updated_at) VALUES (?, ?, ?)",
        ("a", '"1"', "2026-01-01"),
    )
    db.execute(
        "INSERT INTO runtime_settings (key, value, updated_at) VALUES (?, ?, ?)",
        ("b", '"2"', "2026-01-01"),
    )
    db.commit()
    rows = db.fetchall("SELECT key FROM runtime_settings ORDER BY key")
    assert [r["key"] for r in rows] == ["a", "b"]
    db.close()


def test_executemany(tmp_path):
    db = Database(tmp_path / "test.db")
    db.executemany(
        "INSERT INTO runtime_settings (key, value, updated_at) VALUES (?, ?, ?)",
        [("x", '"1"', "2026-01-01"), ("y", '"2"', "2026-01-01")],
    )
    db.commit()
    assert len(db.fetchall("SELECT * FROM runtime_settings")) == 2
    db.close()


def test_procedures_index_exists(tmp_path):
    db = Database(tmp_path / "test.db")
    idx = db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_procedures_trigger'"
    )
    assert idx is not None
    db.close()
