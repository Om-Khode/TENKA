"""Tests for P0 Tier 1 telemetry — interaction event logging."""

import sqlite3
import pytest
from assistant.storage.db import Database


@pytest.fixture
def db(tmp_path):
    """Fresh in-memory-like DB with all migrations applied."""
    db_path = tmp_path / "test_telemetry.db"
    return Database(db_path)


class TestSchema:
    def test_interaction_events_table_exists(self, db):
        row = db.fetchone(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='interaction_events'"
        )
        assert row is not None

    def test_interaction_events_columns(self, db):
        cursor = db._conn.execute("PRAGMA table_info(interaction_events)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id", "session_id", "timestamp", "input_modality", "transcript",
            "intent_detected", "intent_source",
            "action_dispatched", "action_outcome", "error_class",
            "latency_total_ms", "latency_stt_ms", "latency_intent_ms",
            "latency_action_ms", "latency_tts_ms",
            "llm_calls_count", "llm_tokens_in", "llm_tokens_out",
            "fallback_chain_depth", "vision_calls_count",
            "user_corrected_within_30s", "same_intent_repeated",
        }
        assert expected.issubset(columns)

    def test_schema_version_is_at_least_7(self, db):
        row = db.fetchone("SELECT version FROM _schema_version")
        assert row["version"] >= 7

    def test_interaction_events_has_provider_columns(self, db):
        """V11 migration adds JSON provider/model breakdown columns."""
        cursor = db._conn.execute("PRAGMA table_info(interaction_events)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "llm_providers_used" in columns
        assert "llm_models_used" in columns


from datetime import datetime, timedelta
from assistant.storage.repos.telemetry import TelemetryRepo


@pytest.fixture
def repo(db):
    return TelemetryRepo(db)


class TestTelemetryRepo:
    def test_create_returns_id(self, repo):
        row_id = repo.create(
            session_id="sess-1",
            timestamp=datetime.now().isoformat(),
            input_modality="text",
            transcript="hello",
            intent_detected="small_talk",
            intent_source="llm",
            action_dispatched="small_talk",
            action_outcome="success",
            error_class=None,
            latency_total_ms=500,
            latency_stt_ms=None,
            latency_intent_ms=100,
            latency_action_ms=300,
            latency_tts_ms=100,
            llm_calls_count=2,
            llm_tokens_in=150,
            llm_tokens_out=80,
            fallback_chain_depth=0,
            vision_calls_count=0,
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_get_last_event_returns_most_recent(self, repo):
        now = datetime.now()
        repo.create(
            session_id="sess-1",
            timestamp=(now - timedelta(seconds=60)).isoformat(),
            input_modality="text", transcript="first",
            intent_detected="get_time", intent_source="regex",
            action_dispatched="get_time", action_outcome="success",
            error_class=None,
            latency_total_ms=200, latency_stt_ms=None,
            latency_intent_ms=50, latency_action_ms=100, latency_tts_ms=50,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )
        repo.create(
            session_id="sess-1",
            timestamp=now.isoformat(),
            input_modality="text", transcript="second",
            intent_detected="small_talk", intent_source="llm",
            action_dispatched="small_talk", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=100, latency_action_ms=300, latency_tts_ms=100,
            llm_calls_count=1, llm_tokens_in=100, llm_tokens_out=50,
            fallback_chain_depth=0, vision_calls_count=0,
        )
        last = repo.get_last_event("sess-1")
        assert last is not None
        assert last["transcript"] == "second"

    def test_get_last_event_filters_by_session(self, repo):
        repo.create(
            session_id="sess-other",
            timestamp=datetime.now().isoformat(),
            input_modality="text", transcript="other session",
            intent_detected="get_time", intent_source="regex",
            action_dispatched="get_time", action_outcome="success",
            error_class=None,
            latency_total_ms=100, latency_stt_ms=None,
            latency_intent_ms=50, latency_action_ms=30, latency_tts_ms=20,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )
        assert repo.get_last_event("sess-1") is None

    def test_mark_correction(self, repo):
        row_id = repo.create(
            session_id="sess-1",
            timestamp=datetime.now().isoformat(),
            input_modality="text", transcript="test",
            intent_detected="small_talk", intent_source="llm",
            action_dispatched="small_talk", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=100, latency_action_ms=300, latency_tts_ms=100,
            llm_calls_count=1, llm_tokens_in=100, llm_tokens_out=50,
            fallback_chain_depth=0, vision_calls_count=0,
        )
        repo.mark_correction(row_id, user_corrected=True, same_intent=False)
        last = repo.get_last_event("sess-1")
        assert last["user_corrected_within_30s"] == 1
        assert last["same_intent_repeated"] == 0

    def test_cleanup_removes_old_rows(self, repo):
        old_time = (datetime.now() - timedelta(days=100)).isoformat()
        new_time = datetime.now().isoformat()
        repo.create(
            session_id="sess-1", timestamp=old_time,
            input_modality="text", transcript="old",
            intent_detected="get_time", intent_source="regex",
            action_dispatched="get_time", action_outcome="success",
            error_class=None,
            latency_total_ms=100, latency_stt_ms=None,
            latency_intent_ms=50, latency_action_ms=30, latency_tts_ms=20,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )
        repo.create(
            session_id="sess-1", timestamp=new_time,
            input_modality="text", transcript="new",
            intent_detected="small_talk", intent_source="llm",
            action_dispatched="small_talk", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=100, latency_action_ms=300, latency_tts_ms=100,
            llm_calls_count=1, llm_tokens_in=100, llm_tokens_out=50,
            fallback_chain_depth=0, vision_calls_count=0,
        )
        deleted = repo.cleanup(retention_days=90)
        assert deleted == 1
        remaining = repo.get_last_event("sess-1")
        assert remaining["transcript"] == "new"


from unittest.mock import patch
from dataclasses import dataclass


@dataclass(frozen=True)
class FakeLLMResult:
    text: str
    provider: str
    model: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: float
    fallback_depth: int


class TestTurnTracker:
    def test_record_llm_result_accumulates(self):
        from assistant.telemetry import TurnTracker
        tracker = TurnTracker("sess-1", "text", "hello")
        tracker.record_llm_result(FakeLLMResult(
            text="hi", provider="gemini", model="flash",
            tokens_in=100, tokens_out=50, latency_ms=200, fallback_depth=0,
        ))
        tracker.record_llm_result(FakeLLMResult(
            text="ok", provider="groq", model="70b",
            tokens_in=80, tokens_out=30, latency_ms=300, fallback_depth=1,
        ))
        assert tracker.llm_calls_count == 2
        assert tracker.llm_tokens_in == 180
        assert tracker.llm_tokens_out == 80
        assert tracker.fallback_chain_depth == 1

    def test_record_vision_call_increments_both(self):
        from assistant.telemetry import TurnTracker
        tracker = TurnTracker("sess-1", "text", "look")
        tracker.record_vision_call(FakeLLMResult(
            text="I see", provider="gemini", model="flash",
            tokens_in=500, tokens_out=100, latency_ms=1000, fallback_depth=0,
        ))
        assert tracker.vision_calls_count == 1
        assert tracker.llm_calls_count == 1
        assert tracker.llm_tokens_in == 500

    def test_record_llm_result_tracks_provider_and_model(self):
        from assistant.telemetry import TurnTracker
        tracker = TurnTracker("sess-1", "text", "hi")
        tracker.record_llm_result(FakeLLMResult(
            text="a", provider="gemini", model="gemini-2.5-flash",
            tokens_in=10, tokens_out=5, latency_ms=100, fallback_depth=0,
        ))
        tracker.record_llm_result(FakeLLMResult(
            text="b", provider="gemini", model="gemini-2.5-flash",
            tokens_in=20, tokens_out=8, latency_ms=120, fallback_depth=0,
        ))
        tracker.record_llm_result(FakeLLMResult(
            text="c", provider="groq", model="llama-3.3-70b",
            tokens_in=50, tokens_out=12, latency_ms=300, fallback_depth=1,
        ))
        assert dict(tracker.llm_providers_used) == {"gemini": 2, "groq": 1}
        assert dict(tracker.llm_models_used) == {
            "gemini-2.5-flash": 2, "llama-3.3-70b": 1,
        }

    def test_record_llm_result_handles_none_tokens(self):
        from assistant.telemetry import TurnTracker
        tracker = TurnTracker("sess-1", "text", "hi")
        tracker.record_llm_result(FakeLLMResult(
            text="hey", provider="ollama", model="llama",
            tokens_in=None, tokens_out=None, latency_ms=500, fallback_depth=3,
        ))
        assert tracker.llm_tokens_in == 0
        assert tracker.llm_tokens_out == 0
        assert tracker.fallback_chain_depth == 3

    def test_save_writes_to_repo(self, db):
        from assistant.telemetry import TurnTracker
        repo = TelemetryRepo(db)
        with patch("assistant.telemetry._get_repo", return_value=repo):
            tracker = TurnTracker("sess-1", "text", "hello")
            tracker.intent_detected = "small_talk"
            tracker.intent_source = "llm"
            tracker.action_dispatched = "small_talk"
            tracker.action_outcome = "success"
            tracker.latency_intent_ms = 100
            tracker.latency_action_ms = 300
            tracker.latency_tts_ms = 100
            tracker.save()

        last = repo.get_last_event("sess-1")
        assert last is not None
        assert last["intent_detected"] == "small_talk"
        assert last["action_outcome"] == "success"
        assert last["latency_total_ms"] is not None
        assert last["latency_total_ms"] >= 0

    def test_save_persists_provider_breakdown(self, db):
        """End-to-end: provider/model counters are JSON-encoded into the row."""
        import json as _json
        from assistant.telemetry import TurnTracker
        repo = TelemetryRepo(db)
        with patch("assistant.telemetry._get_repo", return_value=repo):
            tracker = TurnTracker("sess-1", "text", "hello")
            tracker.intent_detected = "small_talk"
            tracker.intent_source = "llm"
            tracker.action_dispatched = "small_talk"
            tracker.action_outcome = "success"
            tracker.record_llm_result(FakeLLMResult(
                text="a", provider="gemini", model="gemini-2.5-flash",
                tokens_in=10, tokens_out=5, latency_ms=100, fallback_depth=0,
            ))
            tracker.record_llm_result(FakeLLMResult(
                text="b", provider="groq", model="llama-3.3-70b",
                tokens_in=20, tokens_out=8, latency_ms=120, fallback_depth=1,
            ))
            tracker.save()

        last = repo.get_last_event("sess-1")
        assert last["llm_providers_used"] is not None
        assert _json.loads(last["llm_providers_used"]) == {"gemini": 1, "groq": 1}
        models = _json.loads(last["llm_models_used"])
        assert models == {"gemini-2.5-flash": 1, "llama-3.3-70b": 1}

    def test_save_empty_provider_breakdown_persists_null(self, db):
        """No LLM calls → providers/models columns are NULL, not empty JSON."""
        from assistant.telemetry import TurnTracker
        repo = TelemetryRepo(db)
        with patch("assistant.telemetry._get_repo", return_value=repo):
            tracker = TurnTracker("sess-1", "text", "open chrome")
            tracker.intent_detected = "open_browser"
            tracker.intent_source = "regex"
            tracker.action_dispatched = "open_browser"
            tracker.action_outcome = "success"
            tracker.save()

        last = repo.get_last_event("sess-1")
        assert last["llm_providers_used"] is None
        assert last["llm_models_used"] is None


class TestContextVar:
    def test_set_and_get_tracker(self):
        from assistant.telemetry import (
            TurnTracker, set_current_tracker, get_current_tracker,
            reset_current_tracker,
        )
        tracker = TurnTracker("sess-1", "text", "hi")
        token = set_current_tracker(tracker)
        assert get_current_tracker() is tracker
        reset_current_tracker(token)
        assert get_current_tracker() is None

    def test_default_is_none(self):
        from assistant.telemetry import get_current_tracker
        assert get_current_tracker() is None


class TestLLMResult:
    def test_llm_result_is_frozen(self):
        from assistant.llm.router import LLMResult
        result = LLMResult(
            text="hello", provider="gemini", model="flash",
            tokens_in=100, tokens_out=50, latency_ms=200.0, fallback_depth=0,
        )
        assert result.text == "hello"
        assert result.provider == "gemini"
        with pytest.raises(AttributeError):
            result.text = "changed"

    def test_streaming_result_accumulates_text(self):
        import asyncio
        from assistant.llm.router import StreamingLLMResult

        async def fake_stream():
            for chunk in ["hel", "lo ", "world"]:
                yield chunk

        async def run():
            stream = StreamingLLMResult(fake_stream(), "gemini", "flash", 0)
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            meta = stream.metadata
            assert "".join(chunks) == "hello world"
            assert meta.text == "hello world"
            assert meta.provider == "gemini"
            assert meta.fallback_depth == 0
            # On Windows, sub-microsecond synthetic streams round to 0.
            # >= 0 is the correct contract — latency must be a non-negative
            # measurement, not necessarily positive.
            assert meta.latency_ms >= 0

        asyncio.run(run())

    def test_streaming_result_records_on_tracker(self):
        import asyncio
        from assistant.llm.router import StreamingLLMResult
        from assistant.telemetry import (
            TurnTracker, set_current_tracker, reset_current_tracker,
        )

        async def fake_stream():
            for chunk in ["hello ", "world"]:
                yield chunk

        async def run():
            tracker = TurnTracker("sess-1", "text", "hi")
            token = set_current_tracker(tracker)
            try:
                stream = StreamingLLMResult(fake_stream(), "groq", "70b", 1)
                async for _ in stream:
                    pass
                assert tracker.llm_calls_count == 1
                assert tracker.fallback_chain_depth == 1
            finally:
                reset_current_tracker(token)

        asyncio.run(run())


class TestCorrectionDetection:
    def test_marks_correction_within_30s(self, db):
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        recent_time = (datetime.now() - timedelta(seconds=10)).isoformat()
        repo.create(
            session_id="sess-1", timestamp=recent_time,
            input_modality="text", transcript="open spotify",
            intent_detected="open_browser", intent_source="llm",
            action_dispatched="open_browser", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=100, latency_action_ms=300, latency_tts_ms=100,
            llm_calls_count=1, llm_tokens_in=100, llm_tokens_out=50,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        tracker = TurnTracker("sess-1", "text", "no open chrome")
        tracker.intent_detected = "open_browser"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 1
        assert prev["same_intent_repeated"] == 1

    def test_no_correction_after_30s(self, db):
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        old_time = (datetime.now() - timedelta(seconds=60)).isoformat()
        repo.create(
            session_id="sess-1", timestamp=old_time,
            input_modality="text", transcript="open spotify",
            intent_detected="open_browser", intent_source="llm",
            action_dispatched="open_browser", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=100, latency_action_ms=300, latency_tts_ms=100,
            llm_calls_count=1, llm_tokens_in=100, llm_tokens_out=50,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        tracker = TurnTracker("sess-1", "text", "open chrome")
        tracker.intent_detected = "open_browser"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 0

    def test_no_correction_when_prev_success_and_no_marker(self, db):
        """Different intent within 30s after a success → not a correction."""
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        recent_time = (datetime.now() - timedelta(seconds=5)).isoformat()
        repo.create(
            session_id="sess-1", timestamp=recent_time,
            input_modality="text", transcript="open spotify",
            intent_detected="open_browser", intent_source="llm",
            action_dispatched="open_browser", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=100, latency_action_ms=300, latency_tts_ms=100,
            llm_calls_count=1, llm_tokens_in=100, llm_tokens_out=50,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        tracker = TurnTracker("sess-1", "text", "what time is it")
        tracker.intent_detected = "get_time"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 0
        assert prev["same_intent_repeated"] == 0

    def test_correction_when_prev_failed(self, db):
        """Prev outcome was failure → always a correction, even without marker."""
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        recent_time = (datetime.now() - timedelta(seconds=5)).isoformat()
        repo.create(
            session_id="sess-1", timestamp=recent_time,
            input_modality="text", transcript="book tickets",
            intent_detected="planner", intent_source="llm",
            action_dispatched="planner", action_outcome="failure",
            error_class="TimeoutError",
            latency_total_ms=80000, latency_stt_ms=None,
            latency_intent_ms=200, latency_action_ms=78000, latency_tts_ms=1800,
            llm_calls_count=8, llm_tokens_in=4200, llm_tokens_out=180,
            fallback_chain_depth=0, vision_calls_count=2,
        )

        tracker = TurnTracker("sess-1", "text", "try booking on bookmyshow instead")
        tracker.intent_detected = "planner"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 1
        assert prev["same_intent_repeated"] == 1

    def test_correction_when_prev_skipped(self, db):
        """Prev outcome was skipped → counts as correction signal."""
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        recent_time = (datetime.now() - timedelta(seconds=2)).isoformat()
        repo.create(
            session_id="sess-1", timestamp=recent_time,
            input_modality="text", transcript="go to wikipedia",
            intent_detected="open_browser", intent_source="regex",
            action_dispatched=None, action_outcome="skipped",
            error_class=None,
            latency_total_ms=5200, latency_stt_ms=None,
            latency_intent_ms=10, latency_action_ms=None, latency_tts_ms=None,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        tracker = TurnTracker("sess-1", "text", "open wikipedia in chrome")
        tracker.intent_detected = "open_browser"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 1

    @pytest.mark.parametrize("phrase", [
        "no", "Nope", "wait", "actually", "I meant", "i mean",
        "stop", "cancel", "not that", "never mind", "nevermind",
        "let me", "Hold on", "  no ", "that's wrong",
    ])
    def test_correction_phrase_detection(self, db, phrase):
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        recent_time = (datetime.now() - timedelta(seconds=5)).isoformat()
        repo.create(
            session_id="sess-1", timestamp=recent_time,
            input_modality="text", transcript="open chrome",
            intent_detected="open_browser", intent_source="regex",
            action_dispatched="open_browser", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=10, latency_action_ms=400, latency_tts_ms=90,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        tracker = TurnTracker("sess-1", "text", f"{phrase} open firefox")
        tracker.intent_detected = "open_browser"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 1, f"phrase={phrase!r}"

    def test_same_intent_repeated_without_correction(self, db):
        """Same intent within 30s after success, no marker → same=1, corrected=0."""
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        recent_time = (datetime.now() - timedelta(seconds=10)).isoformat()
        repo.create(
            session_id="sess-1", timestamp=recent_time,
            input_modality="text", transcript="what time is it",
            intent_detected="get_time", intent_source="regex",
            action_dispatched="get_time", action_outcome="success",
            error_class=None,
            latency_total_ms=400, latency_stt_ms=None,
            latency_intent_ms=10, latency_action_ms=300, latency_tts_ms=90,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        tracker = TurnTracker("sess-1", "text", "what time is it again")
        tracker.intent_detected = "get_time"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["same_intent_repeated"] == 1
        assert prev["user_corrected_within_30s"] == 0

    def test_no_previous_event(self, db):
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        tracker = TurnTracker("sess-1", "text", "hello")
        tracker.intent_detected = "small_talk"

        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)
        # Should not raise

    def test_helper_phrase_detection_unit(self):
        from assistant.telemetry import _is_explicit_correction
        assert _is_explicit_correction("no open chrome")
        assert _is_explicit_correction("Wait, that's wrong")
        assert _is_explicit_correction("  actually use firefox")
        assert not _is_explicit_correction("open chrome please")
        assert not _is_explicit_correction("")
        assert not _is_explicit_correction(None)
        # "no" must be a word, not a prefix
        assert not _is_explicit_correction("note this down")

    def test_f11b_window_measured_from_utterance_not_check_time(self, db):
        """F11b regression: the 30s correction window measures from the
        user's utterance timestamp (tracker construction), not from the
        time check_correction runs.

        Scenario from live test: prev manifest_dispatch at T+0s, user speaks
        correction at T+9s (well within 30s), but the intent classifier
        stalls on a Gemini timeout for 32s, so check_correction runs at
        T+41s. Before F11b, datetime.now() - prev_time = 41s ≥ 30s and
        the correction was silently dropped. After F11b, the comparison
        uses tracker.utterance_wall_time, which was set at T+9s ≈ within
        window, so the correction fires.
        """
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        # Prev turn happened 9 seconds before the user's utterance.
        prev_time = datetime.now() - timedelta(seconds=9)
        repo.create(
            session_id="sess-1", timestamp=prev_time.isoformat(),
            input_modality="text", transcript="click play",
            intent_detected="manifest_dispatch", intent_source="regex",
            action_dispatched="manifest_dispatch", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=10, latency_action_ms=400, latency_tts_ms=90,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        # User speaks "no wrong" right now (within window).
        tracker = TurnTracker("sess-1", "voice", "no wrong")
        tracker.intent_detected = "computer_task"
        # SIMULATE A 32-SECOND CLASSIFIER STALL — pull the tracker's
        # utterance anchor back to "now" but advance check_correction's
        # wall clock by 32s. Before F11b, datetime.now() - prev = 41s
        # (> 30s) and the check would bail.
        from unittest.mock import patch as _patch
        future_now = datetime.now() + timedelta(seconds=32)
        with _patch("assistant.telemetry._get_repo", return_value=repo), \
             _patch("assistant.telemetry.datetime") as _dt:
            _dt.now.return_value = future_now
            _dt.fromisoformat = datetime.fromisoformat
            check_correction(tracker)

        # After F11b: the check uses tracker.utterance_wall_time (set when
        # the tracker was constructed, before the simulated stall), so the
        # elapsed math is ~9s, well within 30s, and the correction fires.
        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 1, (
            "F11b regression: classifier stall should NOT collapse the 30s "
            "correction window — user spoke within it"
        )

    def test_f11b_window_still_rejects_truly_late_utterances(self, db):
        """F11b counterpart: when the user really WAS late (utterance
        >30s after prev), the window still correctly rejects.

        Guards against an over-correction where every utterance counts
        regardless of how long ago the prev turn ended.
        """
        from assistant.telemetry import TurnTracker, check_correction
        repo = TelemetryRepo(db)

        prev_time = datetime.now() - timedelta(seconds=35)  # 35s ago — over window
        repo.create(
            session_id="sess-1", timestamp=prev_time.isoformat(),
            input_modality="text", transcript="click play",
            intent_detected="manifest_dispatch", intent_source="regex",
            action_dispatched="manifest_dispatch", action_outcome="success",
            error_class=None,
            latency_total_ms=500, latency_stt_ms=None,
            latency_intent_ms=10, latency_action_ms=400, latency_tts_ms=90,
            llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
            fallback_chain_depth=0, vision_calls_count=0,
        )

        tracker = TurnTracker("sess-1", "voice", "no wrong")
        tracker.intent_detected = "computer_task"
        with patch("assistant.telemetry._get_repo", return_value=repo):
            check_correction(tracker)

        prev = repo.get_last_event("sess-1")
        assert prev["user_corrected_within_30s"] == 0

    def test_f11a_expanded_correction_phrases(self):
        """F11a regression: phrases the live-test sheet promised would work.

        These were used or attempted during manifest-based Scenario-7 live testing
        and the original regex didn't recognize them, breaking the
        correction-signal path entirely. The fix extends the regex
        without dropping any of the originals.
        """
        from assistant.telemetry import _is_explicit_correction
        # Phrases the live-test sheet listed as valid corrections — now match.
        assert _is_explicit_correction("wrong button")
        assert _is_explicit_correction("Wrong button")  # case-insensitive
        assert _is_explicit_correction("wrong one")
        assert _is_explicit_correction("undo that")
        assert _is_explicit_correction("undo")
        assert _is_explicit_correction("the other one")
        assert _is_explicit_correction("other one")
        assert _is_explicit_correction("not it")
        assert _is_explicit_correction("that's not it")
        assert _is_explicit_correction("something else")

        # The previously-working phrases must STILL match (no regression).
        assert _is_explicit_correction("no open chrome")
        assert _is_explicit_correction("Wait, that's wrong")
        assert _is_explicit_correction("actually use firefox")
        assert _is_explicit_correction("never mind")
        assert _is_explicit_correction("cancel")

        # Word-boundary guard must STILL hold — "wrong" as a prefix in
        # an unrelated word doesn't trigger.
        assert not _is_explicit_correction("wrongful imprisonment")  # 'wrongful' boundary check
        # Mid-sentence "wrong" doesn't trigger (regex is anchored at ^).
        assert not _is_explicit_correction("I think this is the wrong file")
        # Empty / None still ignored.
        assert not _is_explicit_correction("")
        assert not _is_explicit_correction(None)


class TestMarkActionFailure:
    def test_mark_action_failure_sets_fields(self):
        from assistant.telemetry import (
            TurnTracker, set_current_tracker, reset_current_tracker,
            mark_action_failure,
        )
        tracker = TurnTracker("sess-1", "text", "do thing")
        token = set_current_tracker(tracker)
        try:
            mark_action_failure("PlannerStepFailed", "step 3 verification failed")
            assert tracker.action_outcome == "failure"
            assert tracker.error_class == "PlannerStepFailed"
        finally:
            reset_current_tracker(token)

    def test_mark_action_failure_noop_without_tracker(self):
        """If no current tracker, helper must not raise (background tasks)."""
        from assistant.telemetry import mark_action_failure
        # No tracker set in this ContextVar scope
        mark_action_failure("X", "y")  # should not raise

    def test_mark_action_failure_truncates_long_reason(self):
        from assistant.telemetry import (
            TurnTracker, set_current_tracker, reset_current_tracker,
            mark_action_failure,
        )
        tracker = TurnTracker("sess-1", "text", "do thing")
        token = set_current_tracker(tracker)
        try:
            long_reason = "x" * 5000
            mark_action_failure("E", long_reason)
            # error_class should be capped to a reasonable size
            assert len(tracker.error_class) <= 200
        finally:
            reset_current_tracker(token)

    def test_first_failure_wins(self):
        """Multiple mark_action_failure calls — first error_class is preserved."""
        from assistant.telemetry import (
            TurnTracker, set_current_tracker, reset_current_tracker,
            mark_action_failure,
        )
        tracker = TurnTracker("sess-1", "text", "do thing")
        token = set_current_tracker(tracker)
        try:
            mark_action_failure("FirstError", "first")
            mark_action_failure("SecondError", "second")
            assert tracker.action_outcome == "failure"
            assert tracker.error_class == "FirstError"
        finally:
            reset_current_tracker(token)

    def test_failure_survives_pipeline_success_set(self):
        """Pipeline guard semantics: once outcome=failure, success-setter is a no-op."""
        from assistant.telemetry import (
            TurnTracker, set_current_tracker, reset_current_tracker,
            mark_action_failure,
        )
        tracker = TurnTracker("sess-1", "text", "do thing")
        token = set_current_tracker(tracker)
        try:
            mark_action_failure("PlannerStepFailed", "step 2 failed")
            # Simulate pipeline guard at main.py:950
            if tracker.action_outcome != "failure":
                tracker.action_outcome = "success"
            assert tracker.action_outcome == "failure"
            assert tracker.error_class == "PlannerStepFailed"
        finally:
            reset_current_tracker(token)


class TestMe1CorrectionFeedback:
    """End-to-end: check_correction → manifest_runtime → dispatcher.record_correction."""

    def test_correction_signal_reaches_manifest_dispatcher_when_prev_was_manifest_dispatch(self, tmp_path):
        """T1 detects correction on manifest_dispatch turn → dispatcher's primary selector failure counter bumps."""
        from assistant.storage.db import Database, _reset_for_testing
        from assistant.storage.repos.telemetry import TelemetryRepo
        from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo
        from assistant.automation.manifest_schema import (
            AppManifest, Captured, Intent, Match, Selector, dump_manifest_to_yaml,
        )
        from assistant.automation.manifest_store import ManifestStore
        from assistant.automation.manifest_registry import ManifestRegistry
        from assistant.automation.manifest_dispatcher import ManifestDispatcher
        from assistant.automation import manifest_runtime
        from assistant import telemetry as _telemetry

        # Build a real manifest-based stack against tmp_path
        _reset_for_testing()
        manifest_runtime.reset_for_test()
        try:
            manifests_dir = tmp_path / "_manifests"
            manifests_dir.mkdir()
            db_dir = tmp_path / "_db"
            db_dir.mkdir()
            db = Database(db_dir / "test.db")

            am = AppManifest(
                schema_version=1, app_id="test_app.desktop", display_name="Test App",
                match=Match(process_names=["TestApp.exe"]),
                intents=[Intent(
                    id="play", display_name="Play",
                    phrases=["play music"],
                    handler_selectors=[
                        Selector(kind="hotkey", keys="Space"),
                    ],
                    captured=Captured(timestamp="2026-05-30T10:00:00Z"),
                )],
            )
            dump_manifest_to_yaml(am, manifests_dir / "test_app.desktop.yaml")
            repo = AppManifestIndexRepo(db._conn)
            store = ManifestStore(manifests_dir=manifests_dir, index_repo=repo)
            store.scan_and_index()
            registry = ManifestRegistry(store=store, index_repo=repo)

            class _FakeTerminator:
                last_call = None
                def send_key(self, key: str) -> None:
                    self.last_call = ("send_key", key)
                def find_element(self, **_):
                    raise LookupError("nope")
                def click(self, element):
                    pass

            disp = ManifestDispatcher(
                registry=registry, store=store,
                terminator_provider=lambda: _FakeTerminator(),
            )
            manifest_runtime.init_dispatcher(disp)

            # Dispatch once to populate _last_dispatch + capture baseline
            disp.dispatch(
                app_id="test_app.desktop", intent_id="play",
                slots={}, active_window="TestApp",
            )
            before = store.get("test_app.desktop").intents[0].handler_selectors[0].failures

            # Seed a prior manifest_dispatch event inside 30s
            from datetime import datetime
            trepo = TelemetryRepo(db)
            session_id = "wiring-test-session"
            trepo.create(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                input_modality="voice",
                transcript="play music",
                intent_detected="manifest_dispatch",
                intent_source="regex",
                action_dispatched="manifest_dispatch",
                action_outcome="success",
                error_class=None,
                latency_total_ms=120,
                latency_stt_ms=20, latency_intent_ms=5,
                latency_action_ms=80, latency_tts_ms=15,
                llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
                fallback_chain_depth=0, vision_calls_count=0,
            )

            # Construct a TurnTracker for the CURRENT turn with a correction phrase
            class _Tracker:
                pass
            tracker = _Tracker()
            tracker.session_id = session_id
            tracker.transcript = "no wait stop"  # explicit correction phrase
            tracker.intent_detected = "small_talk"
            # F11b: check_correction now reads utterance_wall_time (set
            # at TurnTracker construction = post-STT, pre-classifier).
            # Bare stubs must supply it or check_correction's try/except
            # swallows the AttributeError and silently drops the correction.
            tracker.utterance_wall_time = datetime.now()

            # Point telemetry's repo getter at this db
            import assistant.telemetry as _tmod
            original_repo_getter = _tmod._get_repo
            _tmod._get_repo = lambda: trepo
            try:
                _telemetry.check_correction(tracker)
            finally:
                _tmod._get_repo = original_repo_getter

            after = store.get("test_app.desktop").intents[0].handler_selectors[0].failures
            assert after == before + 1, (
                f"Expected failures to bump after correction; before={before}, after={after}"
            )
        finally:
            manifest_runtime.reset_for_test()
            try:
                db.close()
            except Exception:
                pass
            _reset_for_testing()


    def test_correction_signal_does_not_fire_when_prev_was_not_manifest_dispatch(self, tmp_path):
        """Prev intent != manifest_dispatch → no callback fires; dispatcher untouched."""
        from assistant.storage.db import Database, _reset_for_testing
        from assistant.storage.repos.telemetry import TelemetryRepo
        from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo
        from assistant.automation.manifest_schema import (
            AppManifest, Captured, Intent, Match, Selector, dump_manifest_to_yaml,
        )
        from assistant.automation.manifest_store import ManifestStore
        from assistant.automation.manifest_registry import ManifestRegistry
        from assistant.automation.manifest_dispatcher import ManifestDispatcher
        from assistant.automation import manifest_runtime
        from assistant import telemetry as _telemetry

        _reset_for_testing()
        manifest_runtime.reset_for_test()
        try:
            manifests_dir = tmp_path / "_manifests"
            manifests_dir.mkdir()
            db_dir = tmp_path / "_db"
            db_dir.mkdir()
            db = Database(db_dir / "test.db")

            am = AppManifest(
                schema_version=1, app_id="test_app.desktop", display_name="Test App",
                match=Match(process_names=["TestApp.exe"]),
                intents=[Intent(
                    id="play", display_name="Play", phrases=["play music"],
                    handler_selectors=[Selector(kind="hotkey", keys="Space")],
                    captured=Captured(timestamp="2026-05-30T10:00:00Z"),
                )],
            )
            dump_manifest_to_yaml(am, manifests_dir / "test_app.desktop.yaml")
            repo = AppManifestIndexRepo(db._conn)
            store = ManifestStore(manifests_dir=manifests_dir, index_repo=repo)
            store.scan_and_index()
            registry = ManifestRegistry(store=store, index_repo=repo)

            class _FakeTerminator:
                last_call = None
                def send_key(self, key: str) -> None:
                    self.last_call = ("send_key", key)
                def find_element(self, **_):
                    raise LookupError("nope")
                def click(self, element):
                    pass

            disp = ManifestDispatcher(
                registry=registry, store=store,
                terminator_provider=lambda: _FakeTerminator(),
            )
            manifest_runtime.init_dispatcher(disp)
            disp.dispatch(
                app_id="test_app.desktop", intent_id="play",
                slots={}, active_window="TestApp",
            )
            before = store.get("test_app.desktop").intents[0].handler_selectors[0].failures

            from datetime import datetime
            trepo = TelemetryRepo(db)
            session_id = "no-me1-session"
            trepo.create(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                input_modality="voice",
                transcript="what time is it",
                intent_detected="get_time",  # NOT manifest_dispatch
                intent_source="regex",
                action_dispatched="get_time",
                action_outcome="success",
                error_class=None,
                latency_total_ms=80,
                latency_stt_ms=20, latency_intent_ms=5,
                latency_action_ms=40, latency_tts_ms=15,
                llm_calls_count=0, llm_tokens_in=0, llm_tokens_out=0,
                fallback_chain_depth=0, vision_calls_count=0,
            )

            class _Tracker:
                pass
            tracker = _Tracker()
            tracker.session_id = session_id
            tracker.transcript = "no wait stop"
            tracker.intent_detected = "small_talk"
            # F11b: see sibling test — bare stub must supply this.
            tracker.utterance_wall_time = datetime.now()

            import assistant.telemetry as _tmod
            original_repo_getter = _tmod._get_repo
            _tmod._get_repo = lambda: trepo
            try:
                _telemetry.check_correction(tracker)
            finally:
                _tmod._get_repo = original_repo_getter

            after = store.get("test_app.desktop").intents[0].handler_selectors[0].failures
            assert after == before, (
                f"Failures should not change for non-manifest_dispatch prev; "
                f"before={before}, after={after}"
            )
        finally:
            manifest_runtime.reset_for_test()
            try:
                db.close()
            except Exception:
                pass
            _reset_for_testing()
