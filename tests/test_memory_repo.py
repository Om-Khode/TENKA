"""Tests for MemoryRepo — SQLite path (no FAISS required)."""

from pathlib import Path

import pytest

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.memory import MemoryRepo


@pytest.fixture(autouse=True)
def reset_db():
    yield
    _reset_for_testing()


@pytest.fixture
def repo(tmp_path) -> MemoryRepo:
    db = Database(tmp_path / "test.db")
    return MemoryRepo(db, data_dir=tmp_path)


class TestSaveTurn:
    def test_save_and_retrieve(self, repo):
        repo.save_turn("hello", "small_talk", "hi there", "sess1")
        recent = repo.get_recent(1)
        assert len(recent) == 1
        assert recent[0]["user_input"] == "hello"
        assert recent[0]["intent"] == "small_talk"
        assert recent[0]["response"] == "hi there"
        assert recent[0]["session_id"] == "sess1"

    def test_get_recent_returns_chronological(self, repo):
        repo.save_turn("first", "a", "r1", "s")
        repo.save_turn("second", "b", "r2", "s")
        repo.save_turn("third", "c", "r3", "s")
        recent = repo.get_recent(3)
        assert [r["user_input"] for r in recent] == ["first", "second", "third"]

    def test_get_recent_limit(self, repo):
        for i in range(5):
            repo.save_turn(f"msg{i}", "x", f"resp{i}", "s")
        assert len(repo.get_recent(2)) == 2


class TestBuildRecentContext:
    def test_empty_returns_empty(self, repo):
        assert repo.build_recent_context() == ""

    def test_formats_with_header(self, repo):
        repo.save_turn("hi", "small_talk", "hello", "s")
        ctx = repo.build_recent_context(limit=1, header="HISTORY:")
        assert ctx.startswith("HISTORY:")
        assert "User: hi" in ctx
        assert "Assistant: hello" in ctx

    def test_no_header(self, repo):
        repo.save_turn("hi", "small_talk", "hello", "s")
        ctx = repo.build_recent_context(limit=1, header="")
        assert not ctx.startswith("HISTORY")
        assert "User: hi" in ctx


class TestSessionScopedContext:
    def test_get_recent_filters_by_session(self, repo):
        repo.save_turn("old fail", "planner", "error on old session", "sess_old")
        repo.save_turn("hello", "small_talk", "hi", "sess_new")
        repo.save_turn("do thing", "planner", "done", "sess_new")
        recent = repo.get_recent(10, session_id="sess_new")
        assert len(recent) == 2
        assert all(r["session_id"] == "sess_new" for r in recent)

    def test_get_recent_no_session_returns_all(self, repo):
        repo.save_turn("msg1", "a", "r1", "sess1")
        repo.save_turn("msg2", "b", "r2", "sess2")
        recent = repo.get_recent(10)
        assert len(recent) == 2

    def test_build_recent_context_session_excludes_other_sessions(self, repo):
        repo.save_turn("old fail", "planner", "couldn't do it", "sess_old")
        repo.save_turn("new task", "planner", "done", "sess_new")
        ctx = repo.build_recent_context(limit=10, session_id="sess_new")
        assert "new task" in ctx
        assert "old fail" not in ctx

    def test_build_recent_context_empty_session_returns_all(self, repo):
        repo.save_turn("msg1", "a", "r1", "sess1")
        repo.save_turn("msg2", "b", "r2", "sess2")
        ctx = repo.build_recent_context(limit=10)
        assert "msg1" in ctx
        assert "msg2" in ctx


class TestSearchConversationsSqlFallback:
    def test_finds_matching_input(self, repo):
        repo.save_turn("tell me about python", "small_talk", "python is great", "s")
        repo.save_turn("hello", "small_talk", "hi", "s")
        results = repo.search_conversations("python")
        assert len(results) >= 1
        assert any("python" in r["user_input"] for r in results)

    def test_returns_empty_on_no_match(self, repo):
        repo.save_turn("hello", "small_talk", "hi", "s")
        results = repo.search_conversations("zzzznotfound")
        assert results == []

    def test_similarity_score_is_zero_for_sql(self, repo):
        repo.save_turn("test", "x", "resp", "s")
        results = repo.search_conversations("test")
        assert all(r["similarity_score"] == 0.0 for r in results)


class TestSummarizeSession:
    def test_summarize_existing_session(self, repo):
        repo.save_turn("q1", "intent1", "a1", "session_abc")
        repo.save_turn("q2", "intent2", "a2", "session_abc")
        summary = repo.summarize_session("session_abc")
        assert "q1" in summary
        assert "q2" in summary

    def test_summarize_missing_session(self, repo):
        result = repo.summarize_session("nonexistent")
        assert "No conversations found" in result


class TestFacts:
    def test_save_and_search(self, repo):
        repo.save_fact("user_name", "Alex", "user")
        results = repo.search_facts("user_name")
        assert len(results) == 1
        assert results[0]["value"] == "Alex"

    def test_search_substring(self, repo):
        repo.save_fact("favorite_color", "blue", "user")
        results = repo.search_facts("color")
        assert len(results) == 1

    def test_search_no_match(self, repo):
        results = repo.search_facts("nonexistent")
        assert results == []


class TestRecordingStorage:
    def test_save_and_retrieve_chunks(self, repo):
        repo.save_chunk("rec1", 0, "first chunk")
        repo.save_chunk("rec1", 1, "second chunk")
        chunks = repo.get_session_transcript("rec1")
        assert len(chunks) == 2
        assert chunks[0]["transcript"] == "first chunk"
        assert chunks[1]["transcript"] == "second chunk"

    def test_chunks_ordered_by_index(self, repo):
        repo.save_chunk("rec1", 2, "third")
        repo.save_chunk("rec1", 0, "first")
        repo.save_chunk("rec1", 1, "second")
        chunks = repo.get_session_transcript("rec1")
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]

    def test_list_sessions(self, repo):
        repo.save_chunk("rec1", 0, "a")
        repo.save_chunk("rec1", 1, "b")
        repo.save_chunk("rec2", 0, "c")
        sessions = repo.list_sessions()
        assert len(sessions) == 2
        assert all("chunk_count" in s for s in sessions)


class TestSearchRecordingsSqlFallback:
    def test_finds_matching_transcript(self, repo):
        repo.save_chunk("rec1", 0, "meeting about project alpha")
        results = repo.search_recording_sessions("alpha")
        assert len(results) >= 1

    def test_returns_empty_on_no_match(self, repo):
        repo.save_chunk("rec1", 0, "some text")
        results = repo.search_recording_sessions("zzzznotfound")
        assert results == []


class TestIdMapVersioning:
    def test_save_creates_versioned_format(self, repo):
        import json
        repo._save_id_map(repo._id_map_path, [1, 2, 3])
        data = json.loads(repo._id_map_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["ids"] == [1, 2, 3]

    def test_load_migrates_legacy_bare_list(self, repo):
        import json
        repo._id_map_path.write_text(json.dumps([10, 20, 30]), encoding="utf-8")
        ids = repo._load_id_map(repo._id_map_path)
        assert ids == [10, 20, 30]
        reloaded = json.loads(repo._id_map_path.read_text(encoding="utf-8"))
        assert reloaded["version"] == 1

    def test_load_nonexistent_returns_empty(self, repo):
        ids = repo._load_id_map(repo._data_dir / "nonexistent.json")
        assert ids == []
