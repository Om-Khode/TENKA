"""Tests for Event-Driven Monitors."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Stub heavy modules before importing assistant packages
for mod_name in [
    "faster_whisper", "pyaudio", "sounddevice", "soundfile",
    "pyautogui", "pyperclip", "mss", "easyocr", "pygetwindow",
    "pynput", "openwakeword", "google", "google.genai", "groq",
    "cerebras.cloud.sdk", "kokoro", "pysbd", "sentence_transformers",
    "faiss", "speechbrain", "torchaudio", "psutil", "nest_asyncio",
    "PIL", "cv2", "dlib", "face_recognition", "send2trash",
    "comtypes", "numpy", "croniter", "winsdk",
    "winsdk.windows", "winsdk.windows.media",
    "winsdk.windows.media.control",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

import pytest
from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.monitor import MonitorRepo
from assistant.automation.event_bus import (
    compile_condition,
    eval_condition_code,
    make_dedup_key,
    render_payload,
    check_dispatch,
)


@pytest.fixture(autouse=True)
def reset_db():
    yield
    _reset_for_testing()


@pytest.fixture
def repo(tmp_path):
    db = Database(tmp_path / "test.db")
    return MonitorRepo(db)


class TestMonitorRepo:
    def test_create_and_get(self, repo):
        mid = repo.create(
            name="Skip Japanese songs",
            event_type="media_changed",
            source_filter="Spotify",
            condition_mode="code",
            condition_expr="any(ord(c) > 0x3000 for c in title)",
            condition_prompt="Is the song title in Japanese?",
            action_type="code_executor",
            action_payload="skip the current song on {source_app}",
            cooldown_secs=5,
            user_goal="skip Japanese songs on Spotify",
        )
        assert mid >= 1
        row = repo.get_by_id(mid)
        assert row is not None
        assert row["name"] == "Skip Japanese songs"
        assert row["event_type"] == "media_changed"
        assert row["source_filter"] == "Spotify"
        assert row["condition_mode"] == "code"
        assert row["enabled"] == 1
        assert row["fire_count"] == 0

    def test_get_active_excludes_disabled(self, repo):
        m1 = repo.create(
            name="Monitor A", event_type="media_changed",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="hello",
            cooldown_secs=5, user_goal="test A",
        )
        m2 = repo.create(
            name="Monitor B", event_type="window_focus",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="hello",
            cooldown_secs=5, user_goal="test B",
        )
        repo.toggle(m2, enabled=False)
        active = repo.get_active()
        assert len(active) == 1
        assert active[0]["name"] == "Monitor A"

    def test_toggle_and_delete(self, repo):
        mid = repo.create(
            name="To Delete", event_type="window_focus",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="hi",
            cooldown_secs=5, user_goal="test",
        )
        repo.toggle(mid, enabled=False)
        row = repo.get_by_id(mid)
        assert row["enabled"] == 0

        repo.toggle(mid, enabled=True)
        row = repo.get_by_id(mid)
        assert row["enabled"] == 1

        repo.delete(mid)
        assert repo.get_by_id(mid) is None

    def test_record_fire_updates_count(self, repo):
        mid = repo.create(
            name="Fire Test", event_type="media_changed",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="fired",
            cooldown_secs=5, user_goal="test",
        )
        repo.record_fire(mid, "2026-05-19T14:00:00")
        row = repo.get_by_id(mid)
        assert row["fire_count"] == 1
        assert row["last_fired_at"] == "2026-05-19T14:00:00"

        repo.record_fire(mid, "2026-05-19T14:05:00")
        row = repo.get_by_id(mid)
        assert row["fire_count"] == 2
        assert row["last_fired_at"] == "2026-05-19T14:05:00"


class TestConditionEval:
    def test_valid_code_condition(self):
        compiled = compile_condition("title == 'hello'")
        assert compiled is not None
        result = eval_condition_code(compiled, {"title": "hello", "artist": ""})
        assert result is True

    def test_japanese_detection_expression(self):
        expr = "any(ord(c) > 0x3000 for c in (title + artist))"
        compiled = compile_condition(expr)
        assert eval_condition_code(compiled, {"title": "夜に駆ける", "artist": "YOASOBI"}) is True
        assert eval_condition_code(compiled, {"title": "Blinding Lights", "artist": "The Weeknd"}) is False

    def test_invalid_condition_raises(self):
        assert compile_condition("if True: pass") is None

    def test_safe_builtins_available(self):
        compiled = compile_condition("any(ord(c) > 127 for c in title)")
        assert eval_condition_code(compiled, {"title": "テスト"}) is True
        assert eval_condition_code(compiled, {"title": "hello"}) is False

    def test_dangerous_builtins_blocked(self):
        compiled = compile_condition("__import__('os').system('echo hi')")
        assert compiled is not None
        assert eval_condition_code(compiled, {"title": "test"}) is False

    def test_event_dot_access_works(self):
        """LLM may generate `event.title` instead of `title` — both must work."""
        compiled = compile_condition("any(ord(c) > 127 for c in event.title)")
        assert compiled is not None
        assert eval_condition_code(compiled, {"title": "夜に駆ける", "artist": "YOASOBI"}) is True
        assert eval_condition_code(compiled, {"title": "Hello", "artist": "World"}) is False

    def test_missing_key_returns_false(self):
        compiled = compile_condition("title == 'x'")
        result = eval_condition_code(compiled, {"artist": "test"})
        assert result is False


class TestDispatchLogic:
    def test_event_type_mismatch_skipped(self):
        monitor = {"event_type": "window_focus", "source_filter": None, "cooldown_secs": 5}
        event = {"event_type": "media_changed"}
        assert check_dispatch(monitor, event, now=100.0) is False

    def test_source_filter_case_insensitive(self):
        monitor = {"event_type": "media_changed", "source_filter": "spotify", "cooldown_secs": 5}
        event = {"event_type": "media_changed", "source_app": "Spotify.exe"}
        assert check_dispatch(monitor, event, now=100.0) is True

    def test_source_filter_mismatch(self):
        monitor = {"event_type": "media_changed", "source_filter": "spotify", "cooldown_secs": 5}
        event = {"event_type": "media_changed", "source_app": "VLC.exe"}
        assert check_dispatch(monitor, event, now=100.0) is False

    def test_cooldown_same_dedup_key_blocked(self):
        monitor = {
            "event_type": "media_changed", "source_filter": None,
            "cooldown_secs": 5,
            "_last_dedup_key": "SongA|ArtistA",
            "_last_fire_time": 98.0,
        }
        event = {"event_type": "media_changed", "title": "SongA", "artist": "ArtistA"}
        assert check_dispatch(monitor, event, now=100.0) is False

    def test_cooldown_different_dedup_key_passes(self):
        monitor = {
            "event_type": "media_changed", "source_filter": None,
            "cooldown_secs": 5,
            "_last_dedup_key": "SongA|ArtistA",
            "_last_fire_time": 99.0,
        }
        event = {"event_type": "media_changed", "title": "SongB", "artist": "ArtistB"}
        assert check_dispatch(monitor, event, now=100.0) is True

    def test_dedup_key_media(self):
        event = {"event_type": "media_changed", "title": "Song", "artist": "Band"}
        assert make_dedup_key(event) == "Song|Band"

    def test_dedup_key_window(self):
        event = {"event_type": "window_focus", "source_app": "Discord", "window_title": "#general"}
        assert make_dedup_key(event) == "Discord|#general"

    def test_action_payload_template_renders(self):
        result = render_payload("Now playing: {title} by {artist}", {"title": "Song", "artist": "Band"})
        assert result == "Now playing: Song by Band"

    def test_action_payload_missing_key_fallback(self):
        result = render_payload("App: {missing_field}", {"title": "Song"})
        assert result == "App: {missing_field}"


class TestMonitorCreation:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock(return_value=SimpleNamespace(text="""{
            "name": "Skip Japanese songs",
            "event_type": "media_changed",
            "source_filter": "Spotify",
            "condition_code": "any(ord(c) > 0x3000 for c in (title + artist))",
            "condition_prompt": "Is the song in Japanese?",
            "action_type": "code_executor",
            "action_payload": "skip the current song on {source_app}",
            "cooldown_secs": 3
        }"""))

    @pytest.mark.asyncio
    async def test_llm_decomposition_accepted(self, mock_llm):
        with patch("assistant.llm.contracts.get_llm_response", mock_llm):
            from assistant.llm.contracts import ask_for_monitor_definition
            result = await ask_for_monitor_definition("skip Japanese songs on Spotify")
        assert result is not None
        assert result["name"] == "Skip Japanese songs"
        assert result["event_type"] == "media_changed"
        assert result["condition_code"] is not None

    @pytest.mark.asyncio
    async def test_bad_json_returns_none(self):
        bad_llm = AsyncMock(return_value=SimpleNamespace(text="not json at all"))
        with patch("assistant.llm.contracts.get_llm_response", bad_llm):
            from assistant.llm.contracts import ask_for_monitor_definition
            result = await ask_for_monitor_definition("skip Japanese songs")
        assert result is None


class TestDisambiguation:
    def test_duplicate_names_show_ids(self, repo):
        from assistant.event_monitoring import delete_monitor
        import assistant.actions as _act

        repo.create(
            name="Song Alert", event_type="media_changed",
            source_filter=None, condition_mode="code",
            condition_expr=None, condition_prompt=None,
            action_type="tts_notify", action_payload="hi",
            cooldown_secs=5, user_goal="test",
        )
        repo.create(
            name="Song Alert", event_type="media_changed",
            source_filter=None, condition_mode="code",
            condition_expr=None, condition_prompt=None,
            action_type="tts_notify", action_payload="hi",
            cooldown_secs=5, user_goal="test",
        )

        result = delete_monitor("delete Song Alert")
        assert "multiple matches" in result.lower()
        assert "#" in result
        assert _act.pending_monitor_disambig.active

        _act.pending_monitor_disambig.clear()

    def test_resolve_by_name(self, repo):
        from assistant.event_monitoring import pause_monitor, resolve_disambig
        import assistant.actions as _act

        repo.create(
            name="Alpha Monitor", event_type="media_changed",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="hi",
            cooldown_secs=5, user_goal="test",
        )
        repo.create(
            name="Beta Monitor", event_type="window_focus",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="hi",
            cooldown_secs=5, user_goal="test",
        )

        pause_monitor("pause monitor")
        assert _act.pending_monitor_disambig.active

        result = resolve_disambig("Alpha")
        assert result is not None
        assert "Paused" in result
        assert "Alpha Monitor" in result

    def test_resolve_cancel(self, repo):
        from assistant.event_monitoring import delete_monitor, resolve_disambig
        import assistant.actions as _act

        repo.create(
            name="Alpha Monitor", event_type="media_changed",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="hi",
            cooldown_secs=5, user_goal="test",
        )
        repo.create(
            name="Beta Monitor", event_type="window_focus",
            source_filter=None, condition_mode="code",
            condition_expr="True", condition_prompt=None,
            action_type="tts_notify", action_payload="hi",
            cooldown_secs=5, user_goal="test",
        )

        delete_monitor("delete monitor")
        assert _act.pending_monitor_disambig.active

        result = resolve_disambig("cancel")
        assert result == "Okay, cancelled."
        assert not _act.pending_monitor_disambig.active


class TestDebounce:
    """Debounce: only tts_notify on media_changed is debounced."""

    def test_tts_notify_media_debounced(self):
        """Rapid media tts_notify events cancel previous timers, only fire the last."""
        from assistant.automation.event_bus import EventBus

        bus = EventBus()
        bus._main_loop = MagicMock()
        bus._active_monitors = [{
            "id": 1,
            "event_type": "media_changed",
            "source_filter": None,
            "condition_mode": "code",
            "condition_expr": "True",
            "action_type": "tts_notify",
            "action_payload": "Now playing {title}",
            "cooldown_secs": 5,
        }]
        bus._compiled_conditions[1] = compile_condition("True")

        fired_payloads = []
        bus._fire_action = lambda m, e: fired_payloads.append(
            render_payload(m["action_payload"], e)
        )

        song1 = {"event_type": "media_changed", "title": "Song A", "artist": "X", "source_app": "Spotify"}
        song2 = {"event_type": "media_changed", "title": "Song B", "artist": "Y", "source_app": "Spotify"}
        song3 = {"event_type": "media_changed", "title": "Song C", "artist": "Z", "source_app": "Spotify"}

        bus._dispatch_event(song1)
        bus._dispatch_event(song2)
        bus._dispatch_event(song3)

        assert 1 in bus._debounce_timers
        assert bus._debounce_events[1][1]["title"] == "Song C"
        assert len(fired_payloads) == 0

        bus._debounce_callback(1)

        assert len(fired_payloads) == 1
        assert fired_payloads[0] == "Now playing Song C"

    def test_code_executor_media_fires_immediately(self):
        """code_executor actions on media_changed must NOT be debounced."""
        from assistant.automation.event_bus import EventBus

        bus = EventBus()
        bus._main_loop = MagicMock()
        bus._active_monitors = [{
            "id": 3,
            "event_type": "media_changed",
            "source_filter": None,
            "condition_mode": "code",
            "condition_expr": "True",
            "action_type": "code_executor",
            "action_payload": "skip the song",
            "cooldown_secs": 5,
        }]
        bus._compiled_conditions[3] = compile_condition("True")

        fired = []
        bus._fire_action = lambda m, e: fired.append(e["title"])

        song1 = {"event_type": "media_changed", "title": "Song A", "artist": "X", "source_app": "Spotify"}
        song2 = {"event_type": "media_changed", "title": "Song B", "artist": "Y", "source_app": "Spotify"}

        bus._dispatch_event(song1)
        bus._dispatch_event(song2)

        assert len(fired) == 2
        assert fired[0] == "Song A"
        assert fired[1] == "Song B"
        assert 3 not in bus._debounce_timers

    def test_window_events_fire_immediately(self):
        """Window events should NOT be debounced regardless of action_type."""
        from assistant.automation.event_bus import EventBus

        bus = EventBus()
        bus._main_loop = MagicMock()
        bus._active_monitors = [{
            "id": 2,
            "event_type": "window_focus",
            "source_filter": None,
            "condition_mode": "code",
            "condition_expr": "True",
            "action_type": "tts_notify",
            "action_payload": "Switched to {source_app}",
            "cooldown_secs": 5,
        }]
        bus._compiled_conditions[2] = compile_condition("True")

        fired = []
        bus._fire_action = lambda m, e: fired.append(e["source_app"])

        event = {"event_type": "window_focus", "source_app": "Discord", "window_title": "#general"}
        bus._dispatch_event(event)

        assert len(fired) == 1
        assert fired[0] == "Discord"
        assert 2 not in bus._debounce_timers

    def test_debounce_timers_cancelled_on_stop(self):
        """stop() must cancel pending debounce timers."""
        from assistant.automation.event_bus import EventBus
        import threading

        bus = EventBus()
        timer = MagicMock(spec=threading.Timer)
        bus._debounce_timers[1] = timer
        bus._debounce_events[1] = ({}, {})
        bus._thread = None

        bus.stop()

        timer.cancel.assert_called_once()
        assert len(bus._debounce_timers) == 0
        assert len(bus._debounce_events) == 0


class TestFlushOnDelete:
    """Deleting monitors should flush pending TTS notifications from the queue."""

    def test_flush_clears_proactive_queue(self):
        from assistant.automation.event_bus import EventBus
        from assistant import proactive
        import queue

        bus = EventBus()

        proactive._proactive_queue.put("You opened GitHub.")
        proactive._proactive_queue.put("You opened GitHub.")
        assert not proactive._proactive_queue.empty()

        bus._flush_pending_tts()
        assert proactive._proactive_queue.empty()

    def test_reload_with_flush_clears_queue(self, repo):
        from assistant.automation.event_bus import EventBus
        from assistant import proactive

        bus = EventBus()

        proactive._proactive_queue.put("stale notification")

        with patch("assistant.automation.event_bus.EventBus._load_monitors"):
            bus.reload_monitors(flush_pending=True)

        assert proactive._proactive_queue.empty()

    def test_reload_without_flush_preserves_queue(self, repo):
        from assistant.automation.event_bus import EventBus
        from assistant import proactive

        bus = EventBus()

        proactive._proactive_queue.put("keep this")

        with patch("assistant.automation.event_bus.EventBus._load_monitors"):
            bus.reload_monitors(flush_pending=False)

        assert not proactive._proactive_queue.empty()
        proactive._proactive_queue.get_nowait()


class TestNullCondition:
    """Monitors with no condition_code/condition_prompt should fire unconditionally."""

    def test_null_condition_fires(self):
        from assistant.automation.event_bus import EventBus

        bus = EventBus()
        bus._main_loop = MagicMock()
        bus._active_monitors = [{
            "id": 10,
            "event_type": "media_changed",
            "source_filter": None,
            "condition_mode": "code",
            "condition_expr": None,
            "condition_prompt": None,
            "action_type": "tts_notify",
            "action_payload": "Now playing {title}",
            "cooldown_secs": 5,
        }]

        fired = []
        bus._fire_action = lambda m, e: fired.append(e["title"])

        event = {"event_type": "media_changed", "title": "Song A", "artist": "X", "source_app": "Spotify"}
        bus._dispatch_event(event)

        bus._debounce_callback(10)
        assert len(fired) == 1
        assert fired[0] == "Song A"

    def test_failed_compile_does_not_fire(self):
        """condition_expr that fails to compile should NOT fire (unlike null)."""
        from assistant.automation.event_bus import EventBus

        bus = EventBus()
        bus._main_loop = MagicMock()
        bus._active_monitors = [{
            "id": 11,
            "event_type": "media_changed",
            "source_filter": None,
            "condition_mode": "code",
            "condition_expr": "if True: pass",
            "condition_prompt": None,
            "action_type": "tts_notify",
            "action_payload": "Now playing {title}",
            "cooldown_secs": 5,
        }]

        fired = []
        bus._fire_action = lambda m, e: fired.append(e["title"])

        event = {"event_type": "media_changed", "title": "Song A", "artist": "X", "source_app": "Spotify"}
        bus._dispatch_event(event)

        assert len(fired) == 0


class TestDbMigrationV6:
    def test_v5_to_v6_preserves_existing_tables(self, tmp_path):
        _reset_for_testing()
        db = Database(tmp_path / "migrate_test.db")
        tables = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {row["name"] for row in tables}
        assert "event_monitors" in table_names
        assert "schedules" in table_names
        assert "conversations" in table_names
        assert "user_preferences" in table_names

    def test_event_monitors_columns(self, tmp_path):
        _reset_for_testing()
        db = Database(tmp_path / "columns_test.db")
        cols = db.fetchall("PRAGMA table_info(event_monitors)")
        col_names = {c["name"] for c in cols}
        expected = {
            "id", "name", "event_type", "source_filter", "condition_mode",
            "condition_expr", "condition_prompt", "action_type", "action_payload",
            "enabled", "cooldown_secs", "last_fired_at", "fire_count",
            "created_at", "user_goal",
        }
        assert expected.issubset(col_names)
