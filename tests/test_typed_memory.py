"""Tests for Phase I1: Typed Memory with Governance."""

import asyncio
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Stub heavy modules before importing assistant packages
for mod_name in [
    "faster_whisper", "pyaudio", "sounddevice",
    "sentence_transformers", "faiss",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from pathlib import Path
from datetime import datetime, timedelta

import pytest

from assistant.storage.db import Database, _reset_for_testing


@pytest.fixture(autouse=True)
def reset_db():
    yield
    _reset_for_testing()


class TestConfigConstants:
    def test_expiry_days_fact(self):
        from assistant import config
        assert config.MEMORY_EXPIRY_DAYS_FACT == 30

    def test_expiry_days_how_to(self):
        from assistant import config
        assert config.MEMORY_EXPIRY_DAYS_HOW_TO == 14

    def test_expiry_days_blocker(self):
        from assistant import config
        assert config.MEMORY_EXPIRY_DAYS_BLOCKER == 14

    def test_valid_memory_types(self):
        from assistant import config
        assert config.VALID_MEMORY_TYPES == frozenset(
            {"preference", "identity", "fact", "how_to", "blocker"}
        )


class TestSchemaV3Migration:
    def test_fresh_db_has_memory_type_column(self, tmp_path):
        db = Database(tmp_path / "test.db")
        col_names = [r["name"] for r in db.fetchall("PRAGMA table_info(facts)")]
        assert "memory_type" in col_names
        assert "expires_at" in col_names

    def test_fresh_db_at_latest_version(self, tmp_path):
        # Fresh DBs land at _LATEST_VERSION (was 3 when this test was written).
        db = Database(tmp_path / "test.db")
        row = db.fetchone("SELECT version FROM _schema_version")
        assert row["version"] == Database._LATEST_VERSION

    def test_v2_to_v3_migration_backfills_existing_facts(self, tmp_path):
        """Simulate a v2 database with existing facts, then migrate."""
        db_path = tmp_path / "test.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE _schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL)")
        conn.execute("INSERT INTO _schema_version (id, version) VALUES (1, 2)")
        conn.execute("""
            CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, key TEXT, value TEXT, source TEXT
            )
        """)
        conn.execute(
            "INSERT INTO facts (timestamp, key, value, source) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), "user_name", "Alex", "user"),
        )
        conn.execute(
            "INSERT INTO facts (timestamp, key, value, source) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), "fav_food", "biryani", "conversation"),
        )
        # Create other required v1/v2 tables so Database() doesn't fail
        conn.execute("CREATE TABLE IF NOT EXISTS personality_state (trait TEXT PRIMARY KEY, value REAL NOT NULL, floor_val REAL NOT NULL, ceiling_val REAL NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS personality_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, trait TEXT, old_value REAL, new_value REAL, delta REAL, reason TEXT, trigger TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS user_preferences (key TEXT PRIMARY KEY, value TEXT, category TEXT, confidence REAL DEFAULT 0.5, source TEXT, times_used INTEGER DEFAULT 0, times_overridden INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS preference_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, key TEXT, old_value TEXT, new_value TEXT, old_confidence REAL, new_confidence REAL, source TEXT, reason TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS user_procedures (id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT, name TEXT, description TEXT DEFAULT '', steps TEXT, backend TEXT DEFAULT 'auto', created_at TEXT, updated_at TEXT, use_count INTEGER DEFAULT 0, last_used TEXT, enabled INTEGER DEFAULT 1)")
        conn.execute("CREATE TABLE IF NOT EXISTS runtime_settings (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT, updated_source TEXT DEFAULT 'user')")
        conn.execute("CREATE TABLE IF NOT EXISTS user_shortcuts (trigger TEXT PRIMARY KEY, intent TEXT, params_json TEXT DEFAULT '{}', description TEXT DEFAULT '', times_used INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, user_input TEXT, intent TEXT, response TEXT, session_id TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS recording_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, chunk_index INTEGER, timestamp TEXT, transcript TEXT)")
        conn.commit()
        conn.close()

        db = Database(db_path)
        rows = db.fetchall("SELECT memory_type, expires_at FROM facts ORDER BY id")
        assert len(rows) == 2
        assert rows[0]["memory_type"] == "fact"
        assert rows[1]["memory_type"] == "fact"
        for row in rows:
            assert row["expires_at"] is not None
            exp = datetime.fromisoformat(row["expires_at"])
            assert exp > datetime.now() + timedelta(days=29)
            assert exp < datetime.now() + timedelta(days=31)

    def test_default_for_new_row_is_fact(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.execute(
            "INSERT INTO facts (timestamp, key, value, source) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), "test_key", "test_val", "user"),
        )
        db.commit()
        row = db.fetchone("SELECT memory_type FROM facts WHERE key = 'test_key'")
        assert row["memory_type"] == "fact"


from assistant.storage.repos.memory import MemoryRepo


@pytest.fixture
def repo(tmp_path) -> MemoryRepo:
    db = Database(tmp_path / "test.db")
    return MemoryRepo(db, data_dir=tmp_path)


class TestSaveTypedFact:
    def test_save_with_explicit_type_and_expiry(self, repo):
        repo.save_typed_fact("name", "Alex", "user", "identity")
        rows = repo._db.fetchall("SELECT * FROM facts WHERE key = 'name'")
        assert len(rows) == 1
        assert rows[0]["memory_type"] == "identity"
        assert rows[0]["expires_at"] is None  # identity never expires

    def test_save_preference_no_expiry(self, repo):
        repo.save_typed_fact("browser", "Firefox", "user", "preference")
        row = repo._db.fetchone("SELECT * FROM facts WHERE key = 'browser'")
        assert row["memory_type"] == "preference"
        assert row["expires_at"] is None

    def test_save_fact_default_expiry_30_days(self, repo):
        repo.save_typed_fact("meeting", "Thursday", "conversation", "fact")
        row = repo._db.fetchone("SELECT * FROM facts WHERE key = 'meeting'")
        assert row["memory_type"] == "fact"
        exp = datetime.fromisoformat(row["expires_at"])
        assert exp > datetime.now() + timedelta(days=29)
        assert exp < datetime.now() + timedelta(days=31)

    def test_save_how_to_default_expiry_14_days(self, repo):
        repo.save_typed_fact("fix_printer", "restart spooler", "conversation", "how_to")
        row = repo._db.fetchone("SELECT * FROM facts WHERE key = 'fix_printer'")
        assert row["memory_type"] == "how_to"
        exp = datetime.fromisoformat(row["expires_at"])
        assert exp > datetime.now() + timedelta(days=13)
        assert exp < datetime.now() + timedelta(days=15)

    def test_save_blocker_default_expiry_14_days(self, repo):
        repo.save_typed_fact("vpn_issue", "drops at 3pm", "user", "blocker")
        row = repo._db.fetchone("SELECT * FROM facts WHERE key = 'vpn_issue'")
        assert row["memory_type"] == "blocker"
        exp = datetime.fromisoformat(row["expires_at"])
        assert exp > datetime.now() + timedelta(days=13)
        assert exp < datetime.now() + timedelta(days=15)

    def test_save_explicit_expires_at_overrides_default(self, repo):
        custom_exp = (datetime.now() + timedelta(days=7)).isoformat()
        repo.save_typed_fact("temp", "val", "user", "fact", expires_at=custom_exp)
        row = repo._db.fetchone("SELECT * FROM facts WHERE key = 'temp'")
        exp = datetime.fromisoformat(row["expires_at"])
        assert exp > datetime.now() + timedelta(days=6)
        assert exp < datetime.now() + timedelta(days=8)

    def test_invalid_type_coerces_to_fact(self, repo):
        repo.save_typed_fact("k", "v", "user", "nonsense_type")
        row = repo._db.fetchone("SELECT * FROM facts WHERE key = 'k'")
        assert row["memory_type"] == "fact"
        assert row["expires_at"] is not None


class TestSaveFactBackwardCompat:
    def test_old_save_fact_still_works(self, repo):
        repo.save_fact("color", "blue", "user")
        row = repo._db.fetchone("SELECT * FROM facts WHERE key = 'color'")
        assert row["memory_type"] == "fact"
        assert row["expires_at"] is not None


class TestGetActiveFacts:
    def test_returns_non_expired_facts(self, repo):
        repo.save_typed_fact("pref", "Firefox", "user", "preference")
        repo.save_typed_fact("temp", "meeting", "user", "fact")
        results = repo.get_active_facts()
        keys = [r["key"] for r in results]
        assert "pref" in keys
        assert "temp" in keys

    def test_excludes_expired_facts(self, repo):
        past = (datetime.now() - timedelta(days=1)).isoformat()
        repo.save_typed_fact("old", "stale", "user", "fact", expires_at=past)
        repo.save_typed_fact("fresh", "new", "user", "preference")
        results = repo.get_active_facts()
        keys = [r["key"] for r in results]
        assert "fresh" in keys
        assert "old" not in keys

    def test_query_filters_by_key(self, repo):
        repo.save_typed_fact("user_name", "Alex", "user", "identity")
        repo.save_typed_fact("fav_food", "biryani", "user", "preference")
        results = repo.get_active_facts(query="name")
        assert len(results) == 1
        assert results[0]["key"] == "user_name"

    def test_query_excludes_expired_even_if_key_matches(self, repo):
        past = (datetime.now() - timedelta(days=1)).isoformat()
        repo.save_typed_fact("user_name", "old_name", "user", "fact", expires_at=past)
        repo.save_typed_fact("user_email", "om@test.com", "user", "identity")
        results = repo.get_active_facts(query="user")
        keys = [r["key"] for r in results]
        assert "user_email" in keys
        assert "user_name" not in keys

    def test_no_query_returns_all_active(self, repo):
        repo.save_typed_fact("a", "1", "user", "preference")
        repo.save_typed_fact("b", "2", "user", "identity")
        results = repo.get_active_facts()
        assert len(results) == 2


class TestCleanupExpired:
    def test_deletes_expired_facts(self, repo):
        past = (datetime.now() - timedelta(days=1)).isoformat()
        repo.save_typed_fact("old", "stale", "user", "fact", expires_at=past)
        repo.save_typed_fact("fresh", "new", "user", "preference")
        count = repo.cleanup_expired()
        assert count == 1
        all_rows = repo._db.fetchall("SELECT * FROM facts")
        assert len(all_rows) == 1
        assert all_rows[0]["key"] == "fresh"

    def test_does_not_delete_null_expires(self, repo):
        repo.save_typed_fact("pref", "Firefox", "user", "preference")
        count = repo.cleanup_expired()
        assert count == 0
        all_rows = repo._db.fetchall("SELECT * FROM facts")
        assert len(all_rows) == 1

    def test_returns_zero_when_nothing_expired(self, repo):
        repo.save_typed_fact("fresh", "val", "user", "fact")
        count = repo.cleanup_expired()
        assert count == 0

    def test_deletes_multiple_expired(self, repo):
        past = (datetime.now() - timedelta(days=1)).isoformat()
        for i in range(5):
            repo.save_typed_fact(f"old_{i}", f"v{i}", "user", "fact", expires_at=past)
        repo.save_typed_fact("keep", "val", "user", "identity")
        count = repo.cleanup_expired()
        assert count == 5
        all_rows = repo._db.fetchall("SELECT * FROM facts")
        assert len(all_rows) == 1


class TestMemoryFacade:
    def test_facade_save_typed_fact(self, tmp_path):
        from assistant import memory as mem_mod
        db = Database(tmp_path / "test.db")
        from assistant.storage.repos.memory import MemoryRepo
        mem_mod._repo = MemoryRepo(db, data_dir=tmp_path)
        try:
            mem_mod.save_typed_fact("k", "v", "user", "identity")
            results = mem_mod.get_active_facts(query="k")
            assert len(results) == 1
            assert results[0]["memory_type"] == "identity"
        finally:
            mem_mod._repo = None

    def test_facade_cleanup_expired(self, tmp_path):
        from assistant import memory as mem_mod
        db = Database(tmp_path / "test.db")
        from assistant.storage.repos.memory import MemoryRepo
        mem_mod._repo = MemoryRepo(db, data_dir=tmp_path)
        try:
            past = (datetime.now() - timedelta(days=1)).isoformat()
            mem_mod.save_typed_fact("old", "val", "user", "fact", expires_at=past)
            count = mem_mod.cleanup_expired()
            assert count == 1
        finally:
            mem_mod._repo = None


class TestAskForMemoryType(unittest.IsolatedAsyncioTestCase):
    async def test_returns_valid_type(self):
        with patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = SimpleNamespace(text="preference")
            from assistant.llm.contracts import ask_for_memory_type
            result = await ask_for_memory_type("browser", "Firefox")
            assert result == "preference"

    async def test_strips_whitespace(self):
        with patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = SimpleNamespace(text="  identity\n")
            from assistant.llm.contracts import ask_for_memory_type
            result = await ask_for_memory_type("name", "Alex")
            assert result == "identity"

    async def test_invalid_type_coerces_to_fact(self):
        with patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = SimpleNamespace(text="some garbage response")
            from assistant.llm.contracts import ask_for_memory_type
            result = await ask_for_memory_type("key", "val")
            assert result == "fact"

    async def test_llm_unavailable_returns_fact(self):
        with patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = SimpleNamespace(text="__LLM_UNAVAILABLE__")
            from assistant.llm.contracts import ask_for_memory_type
            result = await ask_for_memory_type("key", "val")
            assert result == "fact"

    async def test_llm_crash_returns_fact(self):
        with patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM crashed")
            from assistant.llm.contracts import ask_for_memory_type
            result = await ask_for_memory_type("key", "val")
            assert result == "fact"

    async def test_uses_synthesis_task_type(self):
        with patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = SimpleNamespace(text="fact")
            from assistant.llm.contracts import ask_for_memory_type
            await ask_for_memory_type("key", "val")
            call_kwargs = mock_llm.call_args
            assert call_kwargs.kwargs.get("task_type") == "synthesis" or call_kwargs[1].get("task_type") == "synthesis"


class TestHandlerIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._tmp_path = Path(self._tmp)

    async def test_store_memory_calls_typed_save(self):
        """store_memory handler should classify and save typed facts."""
        from assistant import memory as mem_mod
        db = Database(self._tmp_path / "store.db")
        from assistant.storage.repos.memory import MemoryRepo
        mem_mod._repo = MemoryRepo(db, data_dir=self._tmp_path)

        try:
            with patch("assistant.llm.contracts.ask_for_memory_type", new_callable=AsyncMock) as mock_classify:
                mock_classify.return_value = "preference"
                from assistant.actions.memory_search import handle_store_memory
                result = await handle_store_memory(
                    {"content": "my browser is Firefox"}, "", None
                )
                assert "Firefox" in result
                rows = db.fetchall("SELECT memory_type FROM facts")
                assert len(rows) == 1
                assert rows[0]["memory_type"] == "preference"
        finally:
            mem_mod._repo = None
            _reset_for_testing()

    async def test_memory_query_uses_active_facts(self):
        """memory_query handler should use get_active_facts, not search_facts."""
        from assistant import memory as mem_mod
        db = Database(self._tmp_path / "query.db")
        from assistant.storage.repos.memory import MemoryRepo
        mem_mod._repo = MemoryRepo(db, data_dir=self._tmp_path)

        try:
            past = (datetime.now() - timedelta(days=1)).isoformat()
            mem_mod.save_typed_fact("user_name", "Alex", "user", "identity")
            mem_mod.save_typed_fact("old_name", "old", "user", "fact", expires_at=past)

            with patch("assistant.llm.contracts.ask_for_synthesis", new_callable=AsyncMock) as mock_synth:
                mock_synth.return_value = "Your name is Alex."
                from assistant.actions.memory_search import handle_memory_query
                result = await handle_memory_query({"query": "name"}, "", None)
                call_args = mock_synth.call_args[0][0]
                assert "Alex" in call_args
                assert "old" not in call_args
        finally:
            mem_mod._repo = None
            _reset_for_testing()


class TestAutoExtractionClassification(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._tmp_path = Path(self._tmp)

    async def test_extract_facts_saves_with_type(self):
        """Verify the extraction pattern: classify then save_typed_fact."""
        from assistant import memory as mem_mod
        db = Database(self._tmp_path / "test.db")
        from assistant.storage.repos.memory import MemoryRepo
        mem_mod._repo = MemoryRepo(db, data_dir=self._tmp_path)

        try:
            with patch("assistant.llm.contracts.ask_for_memory_type", new_callable=AsyncMock) as mock_classify:
                mock_classify.return_value = "identity"
                facts = [{"key": "user_name", "value": "Alex"}]
                for fact in facts:
                    memory_type = await mock_classify(fact["key"], fact["value"])
                    mem_mod.save_typed_fact(
                        key=fact["key"], value=fact["value"],
                        source="conversation", memory_type=memory_type,
                    )
                rows = db.fetchall("SELECT memory_type FROM facts")
                assert len(rows) == 1
                assert rows[0]["memory_type"] == "identity"
        finally:
            mem_mod._repo = None
            _reset_for_testing()


class TestPeriodicCleanup(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._tmp_path = Path(self._tmp)

    async def test_cleanup_loop_function(self):
        """Test the cleanup coroutine runs and cleans expired facts."""
        from assistant import memory as mem_mod
        db = Database(self._tmp_path / "test.db")
        from assistant.storage.repos.memory import MemoryRepo
        mem_mod._repo = MemoryRepo(db, data_dir=self._tmp_path)

        try:
            past = (datetime.now() - timedelta(days=1)).isoformat()
            mem_mod.save_typed_fact("old", "val", "user", "fact", expires_at=past)
            mem_mod.save_typed_fact("keep", "val", "user", "identity")

            count = mem_mod.cleanup_expired()
            assert count == 1
            remaining = db.fetchall("SELECT * FROM facts")
            assert len(remaining) == 1
            assert remaining[0]["key"] == "keep"
        finally:
            mem_mod._repo = None
            _reset_for_testing()
