"""Tests for Hybrid Retrieval with RRF."""

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

for mod_name in [
    "faster_whisper", "pyaudio", "sounddevice",
    "sentence_transformers", "faiss",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

import pytest

from assistant.storage.db import Database, _reset_for_testing


@pytest.fixture(autouse=True)
def reset_db():
    yield
    _reset_for_testing()


# ─── Schema & Trigger Tests ───────────────────────────────────────────


class TestSchemaV10Migration:
    def test_fts5_tables_exist(self, tmp_path):
        db = Database(tmp_path / "test.db")
        tables = [
            r["name"]
            for r in db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        assert "facts_fts" in tables
        assert "conversations_fts" in tables

    def test_schema_version_is_10(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone("SELECT version FROM _schema_version")
        assert row["version"] == 10

    def test_fact_insert_triggers_fts(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "john_email", "john@example.com", "user", "fact"),
        )
        db.commit()
        row = db.fetchone(
            "SELECT * FROM facts_fts WHERE facts_fts MATCH ?", ("john_email",)
        )
        assert row is not None

    def test_fact_delete_removes_fts(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "temp_key", "temp_val", "user", "fact"),
        )
        db.commit()
        db.execute("DELETE FROM facts WHERE key = 'temp_key'")
        db.commit()
        row = db.fetchone(
            "SELECT * FROM facts_fts WHERE facts_fts MATCH ?", ("temp_key",)
        )
        assert row is None

    def test_fact_update_updates_fts(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "color", "blue", "user", "preference"),
        )
        db.commit()
        db.execute("UPDATE facts SET value = 'red' WHERE key = 'color'")
        db.commit()
        row = db.fetchone(
            "SELECT * FROM facts_fts WHERE facts_fts MATCH ?", ("red",)
        )
        assert row is not None
        old = db.fetchone(
            "SELECT * FROM facts_fts WHERE facts_fts MATCH ?", ("blue",)
        )
        assert old is None

    def test_conversation_insert_triggers_fts(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.execute(
            "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "hello world", "small_talk", "hi there", "s1"),
        )
        db.commit()
        row = db.fetchone(
            "SELECT * FROM conversations_fts WHERE conversations_fts MATCH ?",
            ("hello",),
        )
        assert row is not None

    def test_backfill_existing_facts(self, tmp_path):
        """Pre-v10 facts appear in FTS after migration."""
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE _schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO _schema_version (id, version) VALUES (1, 9)")
        conn.executescript("""
            CREATE TABLE personality_state (
                personality_id TEXT NOT NULL, trait TEXT NOT NULL,
                value REAL NOT NULL, floor_val REAL NOT NULL,
                ceiling_val REAL NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (personality_id, trait)
            );
            CREATE TABLE personality_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                trait TEXT NOT NULL, old_value REAL NOT NULL, new_value REAL NOT NULL,
                delta REAL NOT NULL, reason TEXT NOT NULL, trigger TEXT NOT NULL,
                personality_id TEXT NOT NULL DEFAULT 'tsundere'
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE user_preferences (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, category TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5, source TEXT NOT NULL,
                times_used INTEGER DEFAULT 0, times_overridden INTEGER DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE preference_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                key TEXT NOT NULL, old_value TEXT, new_value TEXT NOT NULL,
                old_confidence REAL, new_confidence REAL NOT NULL,
                source TEXT NOT NULL, reason TEXT NOT NULL
            );
            CREATE TABLE user_procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT NOT NULL,
                name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                steps TEXT NOT NULL, backend TEXT NOT NULL DEFAULT 'auto',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 0, last_used TEXT DEFAULT NULL,
                enabled INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE runtime_settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL,
                updated_at TEXT NOT NULL, updated_source TEXT NOT NULL DEFAULT 'user'
            );
            CREATE TABLE user_shortcuts (
                trigger TEXT PRIMARY KEY, intent TEXT NOT NULL,
                params_json TEXT NOT NULL DEFAULT '{}', description TEXT NOT NULL DEFAULT '',
                times_used INTEGER DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
                user_input TEXT, intent TEXT, response TEXT, session_id TEXT
            );
            CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
                key TEXT, value TEXT, source TEXT,
                memory_type TEXT NOT NULL DEFAULT 'fact', expires_at TEXT
            );
            CREATE TABLE recording_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, timestamp TEXT NOT NULL, transcript TEXT NOT NULL
            );
            CREATE TABLE session_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL UNIQUE,
                started_at TEXT NOT NULL, ended_at TEXT, turn_count INTEGER DEFAULT 0,
                last_intent TEXT, task_summary TEXT, blocker TEXT, summarized INTEGER DEFAULT 0
            );
            CREATE TABLE schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                cron_expr TEXT NOT NULL, task_type TEXT NOT NULL, task_goal TEXT NOT NULL,
                notify_mode TEXT NOT NULL DEFAULT 'on_match_only', condition_text TEXT,
                last_result_hash TEXT, last_fired_at TEXT, next_fire_at TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
            );
            CREATE TABLE event_monitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                event_type TEXT NOT NULL, source_filter TEXT,
                condition_mode TEXT NOT NULL DEFAULT 'code', condition_expr TEXT,
                condition_prompt TEXT, action_type TEXT NOT NULL, action_payload TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1, cooldown_secs INTEGER NOT NULL DEFAULT 5,
                last_fired_at TEXT, fire_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, user_goal TEXT NOT NULL
            );
            CREATE TABLE interaction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL, input_modality TEXT NOT NULL, transcript TEXT,
                intent_detected TEXT, intent_source TEXT, action_dispatched TEXT,
                action_outcome TEXT, error_class TEXT, latency_total_ms INTEGER,
                latency_stt_ms INTEGER, latency_intent_ms INTEGER,
                latency_action_ms INTEGER, latency_tts_ms INTEGER,
                llm_calls_count INTEGER DEFAULT 0, llm_tokens_in INTEGER DEFAULT 0,
                llm_tokens_out INTEGER DEFAULT 0, fallback_chain_depth INTEGER DEFAULT 0,
                vision_calls_count INTEGER DEFAULT 0,
                user_corrected_within_30s INTEGER DEFAULT 0,
                same_intent_repeated INTEGER DEFAULT 0
            );
            CREATE TABLE automation_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT, backend TEXT NOT NULL,
                app_name TEXT NOT NULL, goal_slug TEXT NOT NULL, goal_text TEXT NOT NULL,
                steps_json TEXT NOT NULL, hit_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, last_hit_at TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1
            );
        """)
        conn.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "pet_name", "Bruno", "user", "preference"),
        )
        conn.commit()
        conn.close()

        db = Database(db_path)
        row = db.fetchone(
            "SELECT * FROM facts_fts WHERE facts_fts MATCH ?", ("Bruno",)
        )
        assert row is not None


from assistant.storage.repos.memory import MemoryRepo


# ─── RRF Fusion Tests ─────────────────────────────────────────────────


class TestRRFFusion:
    def test_single_source_preserves_order(self):
        ranked = [(10, 0.9), (20, 0.7), (30, 0.5)]
        result = MemoryRepo._rrf_fuse(ranked, limit=3)
        ids = [r[0] for r in result]
        assert ids == [10, 20, 30]

    def test_two_sources_no_overlap(self):
        list_a = [(1, 0.9), (2, 0.7)]
        list_b = [(3, 0.8), (4, 0.6)]
        result = MemoryRepo._rrf_fuse(list_a, list_b, limit=4)
        ids = [r[0] for r in result]
        assert set(ids) == {1, 2, 3, 4}

    def test_two_sources_overlap_boosts(self):
        list_a = [(1, 0.9), (2, 0.7)]
        list_b = [(2, 0.8), (3, 0.6)]
        result = MemoryRepo._rrf_fuse(list_a, list_b, limit=3)
        ids = [r[0] for r in result]
        assert ids[0] == 2

    def test_limit_respected(self):
        ranked = [(i, 1.0 - i * 0.1) for i in range(10)]
        result = MemoryRepo._rrf_fuse(ranked, limit=3)
        assert len(result) == 3

    def test_empty_inputs(self):
        result = MemoryRepo._rrf_fuse([], [], limit=5)
        assert result == []

    def test_one_empty_one_populated(self):
        ranked = [(1, 0.9), (2, 0.7)]
        result = MemoryRepo._rrf_fuse([], ranked, limit=5)
        ids = [r[0] for r in result]
        assert ids == [1, 2]


from pathlib import Path


# ─── FTS5 Search Tests ────────────────────────────────────────────────


class TestFTSSearch:
    @pytest.fixture()
    def repo(self, tmp_path):
        db = Database(tmp_path / "test.db")
        repo = MemoryRepo(db, tmp_path / "memory")
        db.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "john_email", "john@example.com", "user", "identity"),
        )
        db.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "favorite_food", "biryani", "user", "preference"),
        )
        db.commit()
        db.execute(
            "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "what is python", "small_talk",
             "Python is a programming language", "s1"),
        )
        db.commit()
        return repo

    def test_exact_keyword_match(self, repo):
        results = repo._search_facts_fts("john_email", limit=5)
        assert len(results) >= 1
        ids = [r[0] for r in results]
        fact = repo._db.fetchone("SELECT id FROM facts WHERE key = 'john_email'")
        assert fact["id"] in ids

    def test_value_keyword_match(self, repo):
        results = repo._search_facts_fts("biryani", limit=5)
        assert len(results) >= 1

    def test_multi_word_query(self, repo):
        results = repo._search_facts_fts("john example", limit=5)
        assert len(results) >= 1

    def test_special_characters_no_crash(self, repo):
        results = repo._search_facts_fts('test"with(special)*chars', limit=5)
        assert isinstance(results, list)

    def test_empty_query_returns_empty(self, repo):
        results = repo._search_facts_fts("", limit=5)
        assert results == []

    def test_conversation_fts_search(self, repo):
        results = repo._search_conversations_fts("python", limit=5)
        assert len(results) >= 1

    def test_sanitize_strips_quotes(self, repo):
        sanitized = repo._sanitize_fts_query('say "hello" world')
        assert sanitized == '"say" "hello" "world"'


import numpy as np


# ─── FAISS Facts Index Tests ──────────────────────────────────────────


class TestFAISSFactsIndex:
    @pytest.fixture()
    def repo_with_faiss(self, tmp_path):
        db = Database(tmp_path / "test.db")
        data_dir = tmp_path / "memory"
        data_dir.mkdir()
        repo = MemoryRepo(db, data_dir)
        repo.init_vector_store()
        return repo

    def test_new_fact_gets_indexed(self, repo_with_faiss):
        repo = repo_with_faiss
        if repo._facts_faiss_index is None:
            pytest.skip("FAISS not available")
        repo.save_typed_fact("dog_name", "Bruno", "user", "preference")
        assert len(repo._facts_id_map) >= 1
        fact = repo._db.fetchone("SELECT id FROM facts WHERE key = 'dog_name'")
        assert fact["id"] in repo._facts_id_map

    def test_semantic_search_returns_relevant_fact(self, repo_with_faiss):
        repo = repo_with_faiss
        if repo._facts_faiss_index is None:
            pytest.skip("FAISS not available")
        repo.save_typed_fact("john_email", "john@example.com", "user", "identity")
        results = repo._search_facts_semantic("contact information for john", limit=5)
        assert len(results) >= 1
        ids = [r[0] for r in results]
        fact = repo._db.fetchone("SELECT id FROM facts WHERE key = 'john_email'")
        assert fact["id"] in ids

    def test_unrelated_query_filtered_by_threshold(self, repo_with_faiss):
        repo = repo_with_faiss
        if repo._facts_faiss_index is None:
            pytest.skip("FAISS not available")
        repo.save_typed_fact("favorite_color", "blue", "user", "preference")
        results = repo._search_facts_semantic("quantum physics equations", limit=5)
        assert isinstance(results, list)
        for _id, score in results:
            assert score >= 0.25, f"Low-similarity result not filtered: {score}"


# ─── Hybrid Search Integration Tests ──────────────────────────────────


class TestHybridSearchFacts:
    @pytest.fixture()
    def repo(self, tmp_path):
        db = Database(tmp_path / "test.db")
        data_dir = tmp_path / "memory"
        data_dir.mkdir()
        repo = MemoryRepo(db, data_dir)
        repo.init_vector_store()
        repo.save_typed_fact("john_email", "john@example.com", "user", "identity")
        repo.save_typed_fact("favorite_food", "biryani", "user", "preference")
        repo.save_typed_fact("work_project", "TENKA assistant", "user", "fact")
        return repo

    def test_keyword_query_returns_results(self, repo):
        results = repo.hybrid_search_facts("john_email", limit=5)
        assert len(results) >= 1
        keys = [r["key"] for r in results]
        assert "john_email" in keys

    def test_results_have_rrf_score(self, repo):
        results = repo.hybrid_search_facts("john", limit=5)
        if results:
            assert "rrf_score" in results[0]
            assert isinstance(results[0]["rrf_score"], float)

    def test_expired_facts_filtered(self, repo):
        from datetime import datetime, timedelta
        expired = (datetime.now() - timedelta(days=1)).isoformat()
        repo._db.execute(
            "INSERT INTO facts (timestamp, key, value, source, memory_type, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), "old_fact", "expired_value", "user", "fact", expired),
        )
        repo._db.commit()
        results = repo.hybrid_search_facts("old_fact", limit=10)
        ids_returned = [r["id"] for r in results]
        expired_fact = repo._db.fetchone("SELECT id FROM facts WHERE key = 'old_fact'")
        assert expired_fact["id"] not in ids_returned

    def test_graceful_degradation_fts_only(self, repo):
        repo._facts_faiss_index = None
        repo._facts_id_map = []
        results = repo.hybrid_search_facts("biryani", limit=5)
        assert len(results) >= 1

    def test_empty_query_returns_empty(self, repo):
        results = repo.hybrid_search_facts("", limit=5)
        assert results == []


class TestHybridSearchConversations:
    @pytest.fixture()
    def repo(self, tmp_path):
        db = Database(tmp_path / "test.db")
        data_dir = tmp_path / "memory"
        data_dir.mkdir()
        repo = MemoryRepo(db, data_dir)
        repo.init_vector_store()
        repo.save_turn("what is python", "small_talk", "Python is a programming language", "s1")
        repo.save_turn("tell me about rust", "small_talk", "Rust is a systems language", "s1")
        return repo

    def test_keyword_query_returns_results(self, repo):
        results = repo.hybrid_search_conversations("python", limit=5)
        assert len(results) >= 1

    def test_results_have_rrf_score(self, repo):
        results = repo.hybrid_search_conversations("python", limit=5)
        if results:
            assert "rrf_score" in results[0]

    def test_graceful_degradation_fts_only(self, repo):
        repo._faiss_index = None
        repo._id_map = []
        results = repo.hybrid_search_conversations("rust", limit=5)
        assert len(results) >= 1

    def test_empty_query_returns_empty(self, repo):
        results = repo.hybrid_search_conversations("", limit=5)
        assert results == []



# ─── Handler Integration Tests ────────────────────────────────────────


class TestHandlerIntegration:
    def test_facade_hybrid_search_facts_exists(self):
        from assistant import memory as mem_facade
        assert hasattr(mem_facade, "hybrid_search_facts")
        assert callable(mem_facade.hybrid_search_facts)

    def test_facade_hybrid_search_conversations_exists(self):
        from assistant import memory as mem_facade
        assert hasattr(mem_facade, "hybrid_search_conversations")
        assert callable(mem_facade.hybrid_search_conversations)

    def test_handler_calls_hybrid_methods(self):
        """Verify the handler source code references hybrid_search, not old methods."""
        import inspect
        from assistant.actions.memory_search import handle_memory_query

        source = inspect.getsource(handle_memory_query)
        assert "hybrid_search_conversations" in source
        assert "hybrid_search_facts" in source
