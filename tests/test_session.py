"""Tests for Phase I2: Session Continuity Snapshots."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from datetime import datetime

# Stub heavy modules before importing assistant packages
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


class TestSchemaV4Migration:
    def test_fresh_db_has_session_snapshots_table(self, tmp_path):
        db = Database(tmp_path / "test.db")
        tables = [r["name"] for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        assert "session_snapshots" in tables

    def test_fresh_db_at_version_4(self, tmp_path):
        db = Database(tmp_path / "test.db")
        row = db.fetchone("SELECT version FROM _schema_version")
        assert row["version"] == 4

    def test_session_snapshots_columns(self, tmp_path):
        db = Database(tmp_path / "test.db")
        cols = [r["name"] for r in db.fetchall("PRAGMA table_info(session_snapshots)")]
        expected = ["id", "session_id", "started_at", "ended_at", "turn_count",
                    "last_intent", "task_summary", "blocker", "summarized"]
        for col in expected:
            assert col in cols

    def test_v3_to_v4_migration(self, tmp_path):
        """Simulate a v3 database, then verify v4 migration adds the table."""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE _schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL)")
        conn.execute("INSERT INTO _schema_version (id, version) VALUES (1, 3)")
        # Create minimal v3 tables so Database.__init__ doesn't fail
        conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, timestamp TEXT, key TEXT, value TEXT, source TEXT, memory_type TEXT NOT NULL DEFAULT 'fact', expires_at TEXT)")
        conn.execute("CREATE TABLE conversations (id INTEGER PRIMARY KEY, timestamp TEXT, user_input TEXT, intent TEXT, response TEXT, session_id TEXT)")
        conn.execute("CREATE TABLE personality_state (trait TEXT PRIMARY KEY, value REAL NOT NULL, floor_val REAL NOT NULL, ceiling_val REAL NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE personality_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, trait TEXT, old_value REAL, new_value REAL, delta REAL, reason TEXT, trigger TEXT)")
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE user_preferences (key TEXT PRIMARY KEY, value TEXT, category TEXT, confidence REAL, source TEXT, times_used INTEGER, times_overridden INTEGER, created_at TEXT, updated_at TEXT)")
        conn.execute("CREATE TABLE preference_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, key TEXT, old_value TEXT, new_value TEXT, old_confidence REAL, new_confidence REAL, source TEXT, reason TEXT)")
        conn.execute("CREATE TABLE user_procedures (id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT, name TEXT, description TEXT DEFAULT '', steps TEXT, backend TEXT DEFAULT 'auto', created_at TEXT, updated_at TEXT, use_count INTEGER DEFAULT 0, last_used TEXT, enabled INTEGER DEFAULT 1)")
        conn.execute("CREATE TABLE runtime_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL, updated_source TEXT NOT NULL DEFAULT 'user')")
        conn.execute("CREATE TABLE shortcuts (id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT NOT NULL, intent TEXT NOT NULL, params TEXT, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE recording_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, timestamp TEXT NOT NULL, transcript TEXT NOT NULL)")
        conn.commit()
        conn.close()

        # Now open with Database — should auto-migrate to v4
        db = Database(db_path)
        tables = [r["name"] for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        assert "session_snapshots" in tables
        row = db.fetchone("SELECT version FROM _schema_version")
        assert row["version"] == 4


from assistant.storage.repos.session import SessionRepo


class TestSessionRepo:
    def _make_repo(self, tmp_path):
        db = Database(tmp_path / "test.db")
        return SessionRepo(db), db

    def test_start_session_creates_row(self, tmp_path):
        repo, db = self._make_repo(tmp_path)
        repo.start_session("abc-123")
        row = db.fetchone("SELECT * FROM session_snapshots WHERE session_id = ?", ("abc-123",))
        assert row is not None
        assert row["session_id"] == "abc-123"
        assert row["started_at"] is not None
        assert row["turn_count"] == 0
        assert row["summarized"] == 0

    def test_increment_turn_count(self, tmp_path):
        repo, db = self._make_repo(tmp_path)
        repo.start_session("abc-123")
        repo.increment_turn_count("abc-123")
        repo.increment_turn_count("abc-123")
        row = db.fetchone("SELECT turn_count FROM session_snapshots WHERE session_id = ?", ("abc-123",))
        assert row["turn_count"] == 2

    def test_update_last_intent(self, tmp_path):
        repo, db = self._make_repo(tmp_path)
        repo.start_session("abc-123")
        repo.update_last_intent("abc-123", "web_search")
        row = db.fetchone("SELECT last_intent FROM session_snapshots WHERE session_id = ?", ("abc-123",))
        assert row["last_intent"] == "web_search"

    def test_end_session_sets_ended_at(self, tmp_path):
        repo, db = self._make_repo(tmp_path)
        repo.start_session("abc-123")
        repo.end_session("abc-123")
        row = db.fetchone("SELECT ended_at FROM session_snapshots WHERE session_id = ?", ("abc-123",))
        assert row["ended_at"] is not None

    def test_save_summary(self, tmp_path):
        repo, db = self._make_repo(tmp_path)
        repo.start_session("abc-123")
        repo.save_summary("abc-123", "web_search", "User searched for GPU prices", None)
        row = db.fetchone("SELECT * FROM session_snapshots WHERE session_id = ?", ("abc-123",))
        assert row["summarized"] == 1
        assert row["last_intent"] == "web_search"
        assert row["task_summary"] == "User searched for GPU prices"
        assert row["blocker"] is None

    def test_save_summary_with_blocker(self, tmp_path):
        repo, db = self._make_repo(tmp_path)
        repo.start_session("abc-123")
        repo.save_summary("abc-123", "computer_task", "Setting up Chrome", "Chrome crashed on launch")
        row = db.fetchone("SELECT * FROM session_snapshots WHERE session_id = ?", ("abc-123",))
        assert row["blocker"] == "Chrome crashed on launch"

    def test_get_last_snapshot_returns_most_recent_summarized(self, tmp_path):
        repo, _ = self._make_repo(tmp_path)
        repo.start_session("sess-1")
        repo.save_summary("sess-1", "web_search", "Searched for GPUs", None)
        repo.start_session("sess-2")
        repo.save_summary("sess-2", "code_executor", "Ran Python script", None)
        repo.start_session("sess-3")
        repo.increment_turn_count("sess-3")

        result = repo.get_last_snapshot()
        assert result is not None
        assert result["session_id"] == "sess-2"
        assert result["task_summary"] == "Ran Python script"

    def test_get_last_snapshot_returns_none_when_empty(self, tmp_path):
        repo, _ = self._make_repo(tmp_path)
        assert repo.get_last_snapshot() is None

    def test_get_last_interaction_time(self, tmp_path):
        repo, _ = self._make_repo(tmp_path)
        repo.start_session("sess-1")
        repo.start_session("sess-2")
        result = repo.get_last_interaction_time()
        assert result is not None

    def test_get_last_interaction_time_returns_none_when_empty(self, tmp_path):
        repo, _ = self._make_repo(tmp_path)
        assert repo.get_last_interaction_time() is None

    def test_get_unsummarized_session_finds_crash_candidate(self, tmp_path):
        repo, _ = self._make_repo(tmp_path)
        repo.start_session("sess-1")
        repo.increment_turn_count("sess-1")
        repo.increment_turn_count("sess-1")
        repo.increment_turn_count("sess-1")

        result = repo.get_unsummarized_session()
        assert result is not None
        assert result["session_id"] == "sess-1"
        assert result["turn_count"] == 3

    def test_get_unsummarized_session_skips_low_turn_count(self, tmp_path):
        repo, _ = self._make_repo(tmp_path)
        repo.start_session("sess-1")
        repo.increment_turn_count("sess-1")

        assert repo.get_unsummarized_session() is None

    def test_get_unsummarized_session_skips_summarized(self, tmp_path):
        repo, _ = self._make_repo(tmp_path)
        repo.start_session("sess-1")
        repo.increment_turn_count("sess-1")
        repo.increment_turn_count("sess-1")
        repo.save_summary("sess-1", "small_talk", "Chatted", None)

        assert repo.get_unsummarized_session() is None


from unittest.mock import patch, AsyncMock
from assistant.storage.db import init_db


class TestSessionFacade:
    def _init_facade(self, tmp_path):
        """Initialize DB + facade for testing."""
        import assistant.session as session_mod
        session_mod._repo = None
        session_mod._current_session_id = None
        init_db(tmp_path / "test.db")
        session_mod.init_session_db()
        return session_mod

    def test_start_session_returns_uuid(self, tmp_path):
        session = self._init_facade(tmp_path)
        sid = session.start_session()
        assert sid is not None
        assert len(sid) == 36  # UUID4 format: 8-4-4-4-12
        assert "-" in sid

    def test_get_current_session_id(self, tmp_path):
        session = self._init_facade(tmp_path)
        sid = session.start_session()
        assert session.get_current_session_id() == sid

    def test_record_turn_increments_and_updates_intent(self, tmp_path):
        session = self._init_facade(tmp_path)
        session.start_session()
        session.record_turn("web_search")
        session.record_turn("small_talk")
        # Verify via repo
        snapshot = session._repo.get_unsummarized_session()
        assert snapshot["turn_count"] == 2
        assert snapshot["last_intent"] == "small_talk"

    def test_end_session_sets_ended_at(self, tmp_path):
        session = self._init_facade(tmp_path)
        sid = session.start_session()
        session.end_session()
        row = session._repo._db.fetchone(
            "SELECT ended_at FROM session_snapshots WHERE session_id = ?", (sid,)
        )
        assert row["ended_at"] is not None

    def test_get_resume_context_empty_when_no_history(self, tmp_path):
        session = self._init_facade(tmp_path)
        assert session.get_resume_context() == ""

    def test_get_resume_context_formats_correctly(self, tmp_path):
        session = self._init_facade(tmp_path)
        # Create a past summarized session
        session._repo.start_session("old-session")
        session._repo.save_summary("old-session", "web_search", "Searched for GPU prices", None)
        # Start current session
        session.start_session()

        ctx = session.get_resume_context()
        assert "SESSION CONTEXT:" in ctx
        assert "Searched for GPU prices" in ctx
        assert "Nothing was left unfinished" in ctx

    def test_get_resume_context_includes_blocker(self, tmp_path):
        session = self._init_facade(tmp_path)
        session._repo.start_session("old-session")
        session._repo.save_summary("old-session", "computer_task", "Setting up Chrome", "Chrome crashed")
        session.start_session()

        ctx = session.get_resume_context()
        assert "Chrome crashed" in ctx

    def test_format_gap_just_now(self, tmp_path):
        from assistant.session import _format_gap
        from datetime import timedelta
        assert _format_gap(timedelta(seconds=30)) == "just now"
        assert _format_gap(timedelta(minutes=3)) == "just now"

    def test_format_gap_minutes(self, tmp_path):
        from assistant.session import _format_gap
        from datetime import timedelta
        assert _format_gap(timedelta(minutes=15)) == "15 minutes ago"
        assert _format_gap(timedelta(minutes=45)) == "45 minutes ago"

    def test_format_gap_hours(self, tmp_path):
        from assistant.session import _format_gap
        from datetime import timedelta
        assert _format_gap(timedelta(hours=2)) == "2 hours ago"
        assert _format_gap(timedelta(hours=1, minutes=30)) == "1 hours ago"

    def test_format_gap_days(self, tmp_path):
        from assistant.session import _format_gap
        from datetime import timedelta
        assert _format_gap(timedelta(days=3)) == "3 days ago"


class TestSessionSummaryContract(unittest.IsolatedAsyncioTestCase):
    @patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock)
    async def test_returns_parsed_json(self, mock_llm):
        mock_llm.return_value = SimpleNamespace(text='{"task_summary": "Searched for GPU prices", "blocker": null}')
        from assistant.llm.contracts import ask_for_session_summary
        turns = [
            {"user_input": "find me cheap GPUs", "response": "Here are some options..."},
            {"user_input": "which one is best?", "response": "The RTX 4060 is best value."},
        ]
        result = await ask_for_session_summary(turns)
        assert result["task_summary"] == "Searched for GPU prices"
        assert result["blocker"] is None

    @patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock)
    async def test_returns_blocker_when_present(self, mock_llm):
        mock_llm.return_value = SimpleNamespace(text='{"task_summary": "Setting up Chrome", "blocker": "Chrome crashed"}')
        from assistant.llm.contracts import ask_for_session_summary
        turns = [
            {"user_input": "set up chrome for truein", "response": "Starting..."},
            {"user_input": "it crashed", "response": "Let me try again."},
        ]
        result = await ask_for_session_summary(turns)
        assert result["task_summary"] == "Setting up Chrome"
        assert result["blocker"] == "Chrome crashed"

    @patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock)
    async def test_returns_fallback_on_invalid_json(self, mock_llm):
        mock_llm.return_value = SimpleNamespace(text="not valid json at all")
        from assistant.llm.contracts import ask_for_session_summary
        turns = [{"user_input": "hello", "response": "hi"}]
        result = await ask_for_session_summary(turns)
        assert result["task_summary"] == "General conversation"
        assert result["blocker"] is None

    @patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock)
    async def test_strips_markdown_code_fences(self, mock_llm):
        mock_llm.return_value = SimpleNamespace(text='```json\n{"task_summary": "Searched for GPU prices in INR", "blocker": null}\n```')
        from assistant.llm.contracts import ask_for_session_summary
        turns = [{"user_input": "convert to INR", "response": "Here are prices"}]
        result = await ask_for_session_summary(turns)
        assert result["task_summary"] == "Searched for GPU prices in INR"
        assert result["blocker"] is None

    @patch("assistant.llm.contracts.get_llm_response", new_callable=AsyncMock)
    async def test_returns_fallback_on_exception(self, mock_llm):
        mock_llm.side_effect = Exception("API down")
        from assistant.llm.contracts import ask_for_session_summary
        turns = [{"user_input": "hello", "response": "hi"}]
        result = await ask_for_session_summary(turns)
        assert result["task_summary"] == "General conversation"
        assert result["blocker"] is None


class TestSnapshotAndRecovery(unittest.IsolatedAsyncioTestCase):
    def _init_facade(self, tmp_path):
        import assistant.session as session_mod
        session_mod._repo = None
        session_mod._current_session_id = None
        _reset_for_testing()
        init_db(tmp_path / "test.db")
        session_mod.init_session_db()
        return session_mod

    @patch("assistant.llm.contracts.ask_for_session_summary", new_callable=AsyncMock)
    async def test_save_snapshot_calls_llm_and_stores(self, mock_summary):
        import tempfile, pathlib
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        session = self._init_facade(tmp_path)
        sid = session.start_session()
        session.record_turn("web_search")
        session.record_turn("small_talk")

        mock_summary.return_value = {"task_summary": "Searched for GPUs", "blocker": None}
        turns = [
            {"user_input": "find GPUs", "response": "Here are options"},
            {"user_input": "thanks", "response": "No problem!"},
        ]
        await session.save_snapshot(turns)

        snapshot = session._repo.get_last_snapshot()
        assert snapshot is not None
        assert snapshot["task_summary"] == "Searched for GPUs"
        assert snapshot["summarized"] == 1

    @patch("assistant.llm.contracts.ask_for_session_summary", new_callable=AsyncMock)
    async def test_save_snapshot_skips_if_low_turn_count(self, mock_summary):
        import tempfile, pathlib
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        session = self._init_facade(tmp_path)
        session.start_session()
        session.record_turn("small_talk")  # only 1 turn

        await session.save_snapshot([{"user_input": "hi", "response": "hey"}])
        mock_summary.assert_not_called()

    @patch("assistant.llm.contracts.ask_for_session_summary", new_callable=AsyncMock)
    async def test_recover_crashed_session(self, mock_summary):
        import tempfile, pathlib
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        session = self._init_facade(tmp_path)

        # Simulate crashed session: has turns in conversations table but no summary
        session._repo.start_session("crashed-sess")
        session._repo.increment_turn_count("crashed-sess")
        session._repo.increment_turn_count("crashed-sess")
        session._repo.increment_turn_count("crashed-sess")
        # Insert fake conversation turns
        db = session._repo._db
        db.execute(
            "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) VALUES (?, ?, ?, ?, ?)",
            ("2026-05-18T10:00:00", "find GPUs", "web_search", "Here are options", "crashed-sess"),
        )
        db.execute(
            "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) VALUES (?, ?, ?, ?, ?)",
            ("2026-05-18T10:01:00", "which is best?", "web_search", "RTX 4060", "crashed-sess"),
        )
        db.commit()

        mock_summary.return_value = {"task_summary": "Was comparing GPUs", "blocker": "Session crashed"}

        # Start new session and recover
        session.start_session()
        await session.recover_crashed_session()

        # Verify the crashed session is now summarized
        row = db.fetchone(
            "SELECT * FROM session_snapshots WHERE session_id = ?", ("crashed-sess",)
        )
        assert row["summarized"] == 1
        assert row["task_summary"] == "Was comparing GPUs"
        assert row["ended_at"] is not None

    @patch("assistant.llm.contracts.ask_for_session_summary", new_callable=AsyncMock)
    async def test_recover_does_nothing_when_no_crash(self, mock_summary):
        import tempfile, pathlib
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        session = self._init_facade(tmp_path)
        session.start_session()
        await session.recover_crashed_session()
        mock_summary.assert_not_called()

    @patch("assistant.llm.contracts.ask_for_session_summary", new_callable=AsyncMock)
    async def test_recover_handles_llm_failure_gracefully(self, mock_summary):
        import tempfile, pathlib
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        session = self._init_facade(tmp_path)

        # Simulate crashed session
        session._repo.start_session("crashed-sess")
        session._repo.increment_turn_count("crashed-sess")
        session._repo.increment_turn_count("crashed-sess")
        db = session._repo._db
        db.execute(
            "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) VALUES (?, ?, ?, ?, ?)",
            ("2026-05-18T10:00:00", "hello", "small_talk", "hi", "crashed-sess"),
        )
        db.commit()

        mock_summary.side_effect = Exception("API down")

        session.start_session()
        await session.recover_crashed_session()  # Should not raise

        # Still unsummarized — will be retried next startup
        row = db.fetchone(
            "SELECT summarized FROM session_snapshots WHERE session_id = ?", ("crashed-sess",)
        )
        assert row["summarized"] == 0


class TestFullLifecycle(unittest.IsolatedAsyncioTestCase):
    """End-to-end test: start → turns → shutdown → resume context available."""

    @patch("assistant.llm.contracts.ask_for_session_summary", new_callable=AsyncMock)
    async def test_full_session_lifecycle(self, mock_summary):
        import tempfile, pathlib
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        _reset_for_testing()

        import assistant.session as session_mod
        session_mod._repo = None
        session_mod._current_session_id = None
        db = init_db(tmp_path / "test.db")
        session_mod.init_session_db()

        # --- Session 1: normal usage ---
        sid1 = session_mod.start_session()
        assert session_mod.get_current_session_id() == sid1

        # Simulate 3 turns
        session_mod.record_turn("web_search")
        session_mod.record_turn("web_search")
        session_mod.record_turn("small_talk")

        # Insert fake turns into conversations table
        for i, (ui, resp) in enumerate([
            ("find cheap GPUs", "Here are some options"),
            ("compare prices", "RTX 4060 is cheapest"),
            ("thanks", "You're welcome!"),
        ]):
            db.execute(
                "INSERT INTO conversations (timestamp, user_input, intent, response, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"2026-05-18T10:0{i}:00", ui, "web_search", resp, sid1),
            )
        db.commit()

        # Shutdown
        mock_summary.return_value = {"task_summary": "Compared GPU prices", "blocker": None}
        turns = [{"user_input": "find cheap GPUs", "response": "Here are some options"},
                 {"user_input": "compare prices", "response": "RTX 4060 is cheapest"},
                 {"user_input": "thanks", "response": "You're welcome!"}]
        await session_mod.save_snapshot(turns)
        session_mod.end_session()

        # --- Session 2: verify resume context ---
        session_mod._current_session_id = None
        sid2 = session_mod.start_session()
        assert sid2 != sid1

        ctx = session_mod.get_resume_context()
        assert "Compared GPU prices" in ctx
        assert "Nothing was left unfinished" in ctx
        assert "SESSION CONTEXT:" in ctx
