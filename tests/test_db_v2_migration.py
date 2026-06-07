"""Tests for db.py v2 migration — memory tables + legacy data migration."""

import sqlite3
from pathlib import Path

import pytest

from assistant.storage.db import Database, _reset_for_testing


@pytest.fixture(autouse=True)
def reset_db():
    yield
    _reset_for_testing()


class TestV2MigrationCreatesMemoryTables:
    def test_conversations_table_exists(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        )
        assert row is not None

    def test_facts_table_exists(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
        )
        assert row is not None

    def test_recording_sessions_table_exists(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recording_sessions'"
        )
        assert row is not None

    def test_schema_version_is_latest(self, tmp_path):
        # Fresh DBs land at the current _LATEST_VERSION (was 2 when this test
        # was written; the migration sequence has advanced since then).
        db = Database(tmp_path / "test.db")
        row = db.fetchone("SELECT version FROM _schema_version")
        assert row["version"] == Database._LATEST_VERSION


class TestV2LegacyDataMigration:
    def _seed_legacy_db(self, legacy_path: Path) -> None:
        conn = sqlite3.connect(str(legacy_path))
        conn.executescript("""
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, user_input TEXT, intent TEXT,
                response TEXT, session_id TEXT
            );
            CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, key TEXT, value TEXT, source TEXT
            );
            CREATE TABLE recording_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL, transcript TEXT NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) "
            "VALUES ('2026-01-01', 'hello', 'small_talk', 'hi there', 'sess1')"
        )
        conn.execute(
            "INSERT INTO facts (timestamp, key, value, source) "
            "VALUES ('2026-01-01', 'name', 'Alex', 'user')"
        )
        conn.execute(
            "INSERT INTO recording_sessions (session_id, chunk_index, timestamp, transcript) "
            "VALUES ('rec1', 0, '2026-01-01', 'test transcript')"
        )
        conn.commit()
        conn.close()

    def test_migrates_conversations_from_legacy(self, tmp_path):
        self._seed_legacy_db(tmp_path / "assistant_memory.db")
        db = Database(tmp_path / "personality.db")
        row = db.fetchone("SELECT user_input FROM conversations WHERE id = 1")
        assert row["user_input"] == "hello"

    def test_migrates_facts_from_legacy(self, tmp_path):
        self._seed_legacy_db(tmp_path / "assistant_memory.db")
        db = Database(tmp_path / "personality.db")
        row = db.fetchone("SELECT value FROM facts WHERE key = 'name'")
        assert row["value"] == "Alex"

    def test_migrates_recordings_from_legacy(self, tmp_path):
        self._seed_legacy_db(tmp_path / "assistant_memory.db")
        db = Database(tmp_path / "personality.db")
        row = db.fetchone("SELECT transcript FROM recording_sessions WHERE session_id = 'rec1'")
        assert row["transcript"] == "test transcript"

    def test_skips_migration_when_no_legacy_db(self, tmp_path):
        db = Database(tmp_path / "personality.db")
        rows = db.fetchall("SELECT * FROM conversations")
        assert rows == []

    def test_existing_db_stays_at_latest(self, tmp_path):
        # Re-opening an already-migrated DB must not re-migrate or downgrade.
        latest = Database._LATEST_VERSION
        db1 = Database(tmp_path / "test.db")
        assert db1.fetchone("SELECT version FROM _schema_version")["version"] == latest
        db1.close()
        _reset_for_testing()
        db2 = Database(tmp_path / "test.db")
        assert db2.fetchone("SELECT version FROM _schema_version")["version"] == latest
