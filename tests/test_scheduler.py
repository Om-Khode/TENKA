"""Tests for the scheduler — scheduled conditional tasks."""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ─── Module stubs ────────────────────────────────────────────────
# Prevent real side effects from assistant subpackages.
_STUBS: dict[str, types.ModuleType] = {}

def _ensure_stub(name: str) -> types.ModuleType:
    if name not in _STUBS:
        _STUBS[name] = types.ModuleType(name)
    return _STUBS[name]

_saved: dict[str, types.ModuleType] = {}

def _install_stubs():
    stub_names = [
        "sounddevice", "soundfile", "numpy", "kokoro", "pysbd",
        "pyaudio", "whisper", "faster_whisper", "speechbrain",
        "pygetwindow", "PIL", "PIL.Image", "PIL.ImageGrab",
        "google", "google.genai",
    ]
    for name in stub_names:
        if name in sys.modules:
            _saved[name] = sys.modules[name]
        sys.modules[name] = _ensure_stub(name)

def _restore_stubs():
    for name in list(_STUBS.keys()):
        if name in _saved:
            sys.modules[name] = _saved[name]
        else:
            sys.modules.pop(name, None)

_install_stubs()

from assistant.storage.db import Database, _reset_for_testing

# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Fresh in-memory-like DB with all migrations applied."""
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    yield database
    database.close()
    _reset_for_testing()


@pytest.fixture
def schedule_repo(db):
    from assistant.storage.repos.schedule import ScheduleRepo
    return ScheduleRepo(db)


# ─── ScheduleRepo Tests ─────────────────────────────────────────

class TestScheduleRepo:
    def test_create_and_list_all(self, schedule_repo):
        now = datetime.now().isoformat()
        next_fire = (datetime.now() + timedelta(hours=1)).isoformat()
        row_id = schedule_repo.create(
            name="morning AI check",
            cron_expr="0 9 * * *",
            task_type="web_search",
            task_goal="search GitHub for new desktop AI projects",
            notify_mode="on_match_only",
            condition_text="only if something new appeared",
            next_fire_at=next_fire,
        )
        assert row_id == 1
        all_schedules = schedule_repo.list_all()
        assert len(all_schedules) == 1
        assert all_schedules[0]["name"] == "morning AI check"
        assert all_schedules[0]["task_type"] == "web_search"
        assert all_schedules[0]["enabled"] == 1

    def test_get_due_returns_only_due_tasks(self, schedule_repo):
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        schedule_repo.create(
            name="due task", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="search AI",
            notify_mode="always", condition_text=None,
            next_fire_at=past,
        )
        schedule_repo.create(
            name="future task", cron_expr="0 18 * * *",
            task_type="web_search", task_goal="search LLMs",
            notify_mode="always", condition_text=None,
            next_fire_at=future,
        )
        due = schedule_repo.get_due(datetime.now().isoformat())
        assert len(due) == 1
        assert due[0]["name"] == "due task"

    def test_get_due_excludes_disabled(self, schedule_repo):
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        row_id = schedule_repo.create(
            name="disabled task", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="search AI",
            notify_mode="always", condition_text=None,
            next_fire_at=past,
        )
        schedule_repo.toggle(row_id, enabled=False)
        due = schedule_repo.get_due(datetime.now().isoformat())
        assert len(due) == 0

    def test_update_after_fire(self, schedule_repo):
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        next_fire = (datetime.now() + timedelta(hours=23)).isoformat()
        row_id = schedule_repo.create(
            name="fired task", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="search AI",
            notify_mode="on_change", condition_text=None,
            next_fire_at=past,
        )
        schedule_repo.update_after_fire(row_id, next_fire, "abc123hash")
        all_schedules = schedule_repo.list_all()
        task = all_schedules[0]
        assert task["next_fire_at"] == next_fire
        assert task["last_result_hash"] == "abc123hash"
        assert task["last_fired_at"] is not None

    def test_find_by_name_case_insensitive(self, schedule_repo):
        schedule_repo.create(
            name="Morning AI Check", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="search AI",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        result = schedule_repo.find_by_name("morning ai")
        assert result is not None
        assert result["name"] == "Morning AI Check"

    def test_find_by_name_no_match(self, schedule_repo):
        result = schedule_repo.find_by_name("nonexistent")
        assert result is None

    def test_toggle(self, schedule_repo):
        row_id = schedule_repo.create(
            name="toggle test", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="search AI",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        schedule_repo.toggle(row_id, enabled=False)
        all_schedules = schedule_repo.list_all()
        assert all_schedules[0]["enabled"] == 0

        schedule_repo.toggle(row_id, enabled=True)
        all_schedules = schedule_repo.list_all()
        assert all_schedules[0]["enabled"] == 1

    def test_delete(self, schedule_repo):
        row_id = schedule_repo.create(
            name="delete me", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="search AI",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        schedule_repo.delete(row_id)
        assert len(schedule_repo.list_all()) == 0

    def test_list_enabled_excludes_disabled(self, schedule_repo):
        schedule_repo.create(
            name="enabled one", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="search AI",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        row_id = schedule_repo.create(
            name="disabled one", cron_expr="0 18 * * *",
            task_type="web_search", task_goal="search LLMs",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        schedule_repo.toggle(row_id, enabled=False)
        enabled = schedule_repo.list_enabled()
        assert len(enabled) == 1
        assert enabled[0]["name"] == "enabled one"


# ─── LLM Contract Tests ─────────────────────────────────────────

class TestScheduleContracts:
    @pytest.mark.asyncio
    async def test_ask_for_schedule_parse_valid_json(self, monkeypatch):
        import json
        from assistant.llm import contracts

        mock_response = json.dumps({
            "name": "morning AI check",
            "cron_expr": "0 9 * * *",
            "task_type": "web_search",
            "goal": "search GitHub for new desktop AI projects",
            "notify_mode": "on_match_only",
            "condition_text": "only if something new appeared",
        })

        async def fake_llm(*args, **kwargs):
            return types.SimpleNamespace(text=mock_response)

        monkeypatch.setattr(contracts, "get_llm_response", fake_llm)

        result = await contracts.ask_for_schedule_parse(
            "schedule a web search for new AI projects every morning, only tell me if something new"
        )
        assert result["name"] == "morning AI check"
        assert result["cron_expr"] == "0 9 * * *"
        assert result["task_type"] == "web_search"
        assert result["notify_mode"] == "on_match_only"

    @pytest.mark.asyncio
    async def test_ask_for_schedule_parse_with_code_fences(self, monkeypatch):
        import json
        from assistant.llm import contracts

        inner = json.dumps({
            "name": "test", "cron_expr": "0 8 * * *",
            "task_type": "procedure", "goal": "morning routine",
            "notify_mode": "always", "condition_text": None,
        })
        mock_response = f"```json\n{inner}\n```"

        async def fake_llm(*args, **kwargs):
            return types.SimpleNamespace(text=mock_response)

        monkeypatch.setattr(contracts, "get_llm_response", fake_llm)

        result = await contracts.ask_for_schedule_parse("run morning routine every day at 8am")
        assert result["name"] == "test"
        assert result["task_type"] == "procedure"

    @pytest.mark.asyncio
    async def test_ask_for_schedule_parse_returns_none_on_bad_json(self, monkeypatch):
        from assistant.llm import contracts

        async def fake_llm(*args, **kwargs):
            return types.SimpleNamespace(text="I don't understand")

        monkeypatch.setattr(contracts, "get_llm_response", fake_llm)

        result = await contracts.ask_for_schedule_parse("something unparseable")
        assert result is None

    @pytest.mark.asyncio
    async def test_ask_for_condition_check_notify_true(self, monkeypatch):
        import json
        from assistant.llm import contracts

        mock_response = json.dumps({"notify": True, "summary": "Found 2 new AI projects"})

        async def fake_llm(*args, **kwargs):
            return types.SimpleNamespace(text=mock_response)

        monkeypatch.setattr(contracts, "get_llm_response", fake_llm)

        result = await contracts.ask_for_condition_check(
            "Found: ProjectA (new), ProjectB (new), ProjectC (old)",
            "only if something new appeared",
        )
        assert result["notify"] is True
        assert len(result["summary"]) <= 100

    @pytest.mark.asyncio
    async def test_ask_for_condition_check_notify_false(self, monkeypatch):
        import json
        from assistant.llm import contracts

        mock_response = json.dumps({"notify": False, "summary": "No new results"})

        async def fake_llm(*args, **kwargs):
            return types.SimpleNamespace(text=mock_response)

        monkeypatch.setattr(contracts, "get_llm_response", fake_llm)

        result = await contracts.ask_for_condition_check(
            "Same results as yesterday",
            "only if something new appeared",
        )
        assert result["notify"] is False

    @pytest.mark.asyncio
    async def test_ask_for_condition_check_fallback_on_bad_json(self, monkeypatch):
        from assistant.llm import contracts

        async def fake_llm(*args, **kwargs):
            return types.SimpleNamespace(text="yes notify")

        monkeypatch.setattr(contracts, "get_llm_response", fake_llm)

        result = await contracts.ask_for_condition_check("some result", "some condition")
        assert result["notify"] is False
        assert "summary" in result


# ─── Scheduler Module Tests ──────────────────────────────────────

import asyncio
import hashlib
import threading
import time


class TestSchedulerModule:
    def test_start_and_stop(self, db, monkeypatch):
        from assistant import scheduler

        monkeypatch.setattr(scheduler, "_thread", None)
        monkeypatch.setattr(scheduler, "_stop_event", threading.Event())
        monkeypatch.setattr("assistant.scheduler.get_db", lambda: db)

        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()

        scheduler.start(loop=loop)
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()

        scheduler.stop()
        # `stop()` joins the thread and sets `_thread = None`, so there is
        # nothing left to ask `is_alive()` -- the original assertion raised
        # AttributeError on the *success* path. Cleared is the stronger claim
        # anyway: a stopped scheduler that still holds a thread reference is
        # what a leak looks like.
        assert scheduler._thread is None

        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)

    def test_compute_next_fire(self):
        from assistant import scheduler
        from datetime import datetime

        now = datetime(2026, 5, 19, 10, 0, 0)
        next_fire = scheduler._compute_next_fire("0 9 * * *", now)
        parsed = datetime.fromisoformat(next_fire)
        assert parsed.hour == 9
        assert parsed.day == 20  # tomorrow, since 9am already passed today

    def test_compute_result_hash(self):
        from assistant import scheduler

        h1 = scheduler._compute_result_hash("hello world")
        h2 = scheduler._compute_result_hash("hello world")
        h3 = scheduler._compute_result_hash("different")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64  # SHA-256 hex digest

    def test_should_notify_always(self):
        from assistant import scheduler

        task = {"notify_mode": "always", "last_result_hash": None, "condition_text": None}
        notify, summary = scheduler._should_notify_sync(task, "some result")
        assert notify is True

    def test_should_notify_on_change_different(self):
        from assistant import scheduler

        result = "new result content"
        old_hash = "oldhash"
        task = {"notify_mode": "on_change", "last_result_hash": old_hash, "condition_text": None}
        notify, summary = scheduler._should_notify_sync(task, result)
        assert notify is True

    def test_should_notify_on_change_same(self):
        from assistant import scheduler

        result = "same result"
        current_hash = scheduler._compute_result_hash(result)
        task = {"notify_mode": "on_change", "last_result_hash": current_hash, "condition_text": None}
        notify, summary = scheduler._should_notify_sync(task, result)
        assert notify is False

    def test_should_notify_on_change_first_run(self):
        from assistant import scheduler

        task = {"notify_mode": "on_change", "last_result_hash": None, "condition_text": None}
        notify, summary = scheduler._should_notify_sync(task, "first result")
        assert notify is True

    def test_should_notify_on_match_only_no_condition(self):
        from assistant import scheduler

        task = {"notify_mode": "on_match_only", "last_result_hash": None, "condition_text": None}
        notify, summary = scheduler._should_notify_sync(task, "weather is sunny")
        assert notify is True
        assert summary == "weather is sunny"


# ─── Action Handler Tests ────────────────────────────────────────

class TestHandleManageSchedule:
    @pytest.fixture
    def mock_repo(self, db, monkeypatch):
        from assistant.storage.repos.schedule import ScheduleRepo
        repo = ScheduleRepo(db)
        monkeypatch.setattr(
            "assistant.actions.schedule._get_repo", lambda: repo
        )
        return repo

    @pytest.mark.asyncio
    async def test_create_schedule(self, mock_repo, monkeypatch):
        import json
        from assistant.actions.schedule import handle_manage_schedule
        from assistant.llm import contracts

        parse_result = {
            "name": "morning AI check",
            "cron_expr": "0 9 * * *",
            "task_type": "web_search",
            "goal": "search GitHub for new desktop AI projects",
            "notify_mode": "on_match_only",
            "condition_text": "only if something new appeared",
        }

        async def fake_parse(goal):
            return parse_result

        monkeypatch.setattr(contracts, "ask_for_schedule_parse", fake_parse)

        result = await handle_manage_schedule(
            {"goal": "schedule a web search for AI projects every morning", "action": "create"},
            "",
        )
        assert "morning AI check" in result.lower() or "scheduled" in result.lower()
        assert len(mock_repo.list_all()) == 1

    @pytest.mark.asyncio
    async def test_create_schedule_bad_parse(self, mock_repo, monkeypatch):
        from assistant.actions.schedule import handle_manage_schedule
        from assistant.llm import contracts

        async def fake_parse(goal):
            return None

        monkeypatch.setattr(contracts, "ask_for_schedule_parse", fake_parse)

        result = await handle_manage_schedule(
            {"goal": "blah blah", "action": "create"}, ""
        )
        assert "couldn't" in result.lower() or "sorry" in result.lower()
        assert len(mock_repo.list_all()) == 0

    @pytest.mark.asyncio
    async def test_list_schedules_empty(self, mock_repo):
        from assistant.actions.schedule import handle_manage_schedule

        result = await handle_manage_schedule({"action": "list"}, "")
        assert "don't have" in result.lower() or "no " in result.lower()

    @pytest.mark.asyncio
    async def test_list_schedules_with_items(self, mock_repo):
        from assistant.actions.schedule import handle_manage_schedule
        from datetime import datetime

        mock_repo.create(
            name="morning check", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="AI projects",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        result = await handle_manage_schedule({"action": "list"}, "")
        assert "morning check" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_schedule(self, mock_repo):
        from assistant.actions.schedule import handle_manage_schedule
        from datetime import datetime

        mock_repo.create(
            name="morning check", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="AI projects",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        result = await handle_manage_schedule(
            {"goal": "cancel morning check", "action": "cancel"}, ""
        )
        assert "cancelled" in result.lower() or "canceled" in result.lower()
        assert len(mock_repo.list_all()) == 0

    @pytest.mark.asyncio
    async def test_cancel_fuzzy_word_match(self, mock_repo):
        """Bug fix: 'delete LLM search schedule' should match 'Search LLM every 5 mins'."""
        from assistant.actions.schedule import handle_manage_schedule
        from datetime import datetime

        mock_repo.create(
            name="Search LLM every 5 mins", cron_expr="*/5 * * * *",
            task_type="web_search", task_goal="search for cheap LLMs",
            notify_mode="on_change", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        result = await handle_manage_schedule(
            {"goal": "delete LLM search schedule", "action": "cancel"}, ""
        )
        assert "cancelled" in result.lower() or "canceled" in result.lower()
        assert len(mock_repo.list_all()) == 0

    @pytest.mark.asyncio
    async def test_cancel_no_match(self, mock_repo):
        from assistant.actions.schedule import handle_manage_schedule

        result = await handle_manage_schedule(
            {"goal": "cancel nonexistent", "action": "cancel"}, ""
        )
        # Property, not phrasing. The reply is "You don't have any schedules
        # to cancel." -- which contains neither of the two strings this used to
        # look for, so the test failed on wording while the behaviour was right.
        # What matters is that it does not claim to have cancelled anything.
        lowered = result.lower()
        assert "cancelled" not in lowered, result
        assert any(p in lowered for p in
                   ("don't have", "couldn't find", "no schedule")), result

    @pytest.mark.asyncio
    async def test_toggle_pause(self, mock_repo):
        from assistant.actions.schedule import handle_manage_schedule
        from datetime import datetime

        mock_repo.create(
            name="morning check", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="AI projects",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        result = await handle_manage_schedule(
            {"goal": "pause morning check", "action": "toggle"}, ""
        )
        assert "paused" in result.lower()
        assert mock_repo.list_all()[0]["enabled"] == 0

    @pytest.mark.asyncio
    async def test_toggle_resume(self, mock_repo):
        from assistant.actions.schedule import handle_manage_schedule
        from datetime import datetime

        row_id = mock_repo.create(
            name="morning check", cron_expr="0 9 * * *",
            task_type="web_search", task_goal="AI projects",
            notify_mode="always", condition_text=None,
            next_fire_at=datetime.now().isoformat(),
        )
        mock_repo.toggle(row_id, enabled=False)
        result = await handle_manage_schedule(
            {"goal": "resume morning check", "action": "toggle"}, ""
        )
        assert "resumed" in result.lower()
        assert mock_repo.list_all()[0]["enabled"] == 1


# ─── Regex Router Tests ──────────────────────────────────────────

class TestScheduleRegexRouting:
    def test_schedule_create(self):
        from assistant.regex_router import pre_route

        result = pre_route("schedule a web search for AI projects every morning")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "create"

    # ── "monitor" belongs to event monitors now ────────────────────────────
    #
    # These five tests were written when a scheduled task was the only thing
    # called a monitor, and `_list_schedules()` said "You have 1 monitor" to
    # match. Event monitors then arrived and took the word: `manage_monitor` is
    # a real intent with its own `event_monitors` table, and "show my monitors"
    # is unambiguously about those.
    #
    # So the routing is right and the assertions were stale -- they encoded a
    # vocabulary that changed underneath them. Fixed on both sides: the tests
    # now expect `manage_monitor`, and `actions/schedule.py` stopped calling a
    # schedule "a monitor" in what it says out loud, which is what made these
    # assertions look reasonable for as long as they did.

    def test_monitor_create_has_no_regex_fast_path(self):
        """"monitor GitHub ... daily" is genuinely ambiguous -- daily makes it a
        schedule, "monitor" makes it sound like an event monitor, and event
        monitors only understand media/window events so it cannot be one. No
        regex claims it, and the classifier decides. This asserted a fast path
        that never existed."""
        from assistant.regex_router import pre_route

        assert pre_route("monitor GitHub for new AI projects daily") is None

    def test_list_schedules(self):
        from assistant.regex_router import pre_route

        result = pre_route("list my schedules")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "list"

    def test_show_monitors_lists_event_monitors(self):
        from assistant.regex_router import pre_route

        result = pre_route("show my monitors")
        assert result is not None
        assert result.intent == "manage_monitor"

    def test_list_schedule_singular(self):
        from assistant.regex_router import pre_route

        result = pre_route("list schedule")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "list"

    def test_show_schedule_singular(self):
        from assistant.regex_router import pre_route

        result = pre_route("show schedules")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "list"

    def test_cancelling_a_monitor_reaches_manage_monitor(self):
        from assistant.regex_router import pre_route

        result = pre_route("cancel the morning AI monitor")
        assert result is not None
        assert result.intent == "manage_monitor"

    def test_cancelling_a_schedule_still_reaches_manage_schedule(self):
        """The other half, and the one that matters: the word "schedule" routes
        to schedules, whatever else is in the sentence."""
        from assistant.regex_router import pre_route

        result = pre_route("cancel the morning AI schedule")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "cancel"

    def test_delete_schedule(self):
        from assistant.regex_router import pre_route

        result = pre_route("delete my morning schedule")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "cancel"

    def test_pause_monitor(self):
        from assistant.regex_router import pre_route

        result = pre_route("pause the GitHub monitor")
        assert result is not None
        assert result.intent == "manage_monitor"

    def test_pause_a_schedule_by_name(self):
        """The word "schedule" still routes to schedules."""
        from assistant.regex_router import pre_route

        result = pre_route("pause the GitHub schedule")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "toggle"

    def test_resume_monitor(self):
        from assistant.regex_router import pre_route

        result = pre_route("resume the GitHub monitor")
        assert result is not None
        assert result.intent == "manage_monitor"

    def test_resume_a_schedule_by_name(self):
        """The word "schedule" still routes to schedules."""
        from assistant.regex_router import pre_route

        result = pre_route("resume the GitHub schedule")
        assert result is not None
        assert result.intent == "manage_schedule"
        assert result.params["action"] == "toggle"

    def test_no_match_on_bare_schedule(self):
        from assistant.regex_router import pre_route

        result = pre_route("schedule")
        assert result is None or result.intent != "manage_schedule"

    def test_no_collision_with_reminder(self):
        from assistant.regex_router import pre_route

        result = pre_route("remind me in 5 minutes to check email")
        assert result is not None
        assert result.intent == "set_reminder"

    def test_no_collision_with_stop_recording(self):
        from assistant.regex_router import pre_route

        result = pre_route("stop recording")
        assert result is not None
        assert result.intent == "stop_recording"


# ─── HTTP Check Tests ───────────────────────────────────────────

class TestHttpCheck:
    @pytest.mark.asyncio
    async def test_http_check_success(self, monkeypatch):
        from assistant import scheduler
        import types

        mock_requests = types.ModuleType("requests")

        class FakeResp:
            text = "ALERT: server is on fire"

        mock_requests.get = lambda url, timeout=10: FakeResp()
        monkeypatch.setattr("assistant.scheduler.requests", mock_requests, raising=False)

        # Need to patch the import inside _http_check
        import importlib
        monkeypatch.setitem(sys.modules, "requests", mock_requests)

        result = await scheduler._http_check("http://localhost:9999/status")
        assert "ALERT" in result

    @pytest.mark.asyncio
    async def test_http_check_truncates_long_response(self, monkeypatch):
        from assistant import scheduler

        class FakeResp:
            text = "x" * 5000

        monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))
        sys.modules["requests"].get = lambda url, timeout=10: FakeResp()

        result = await scheduler._http_check("http://localhost:9999/long")
        assert len(result) <= 2000

    @pytest.mark.asyncio
    async def test_http_check_returns_empty_on_error(self, monkeypatch):
        from assistant import scheduler

        def raise_err(url, timeout=10):
            raise ConnectionError("refused")

        monkeypatch.setitem(sys.modules, "requests", types.ModuleType("requests"))
        sys.modules["requests"].get = raise_err

        result = await scheduler._http_check("http://localhost:9999/down")
        assert result == ""


# ─── Schema Migration Tests ──────────────────────────────────────

class TestSchemaV5:
    def test_schedules_table_exists(self, db):
        row = db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schedules'"
        )
        assert row is not None

    def test_schema_version_is_5(self, db):
        # `>=`, not `==`. This pinned an absolute number and went red the moment
        # the schema moved past v5 -- which it has, many times, for reasons that
        # have nothing to do with schedules. What the test is about is that
        # opening a database migrates it far enough to have the `schedules`
        # table, and the test below checks exactly that.
        row = db.fetchone("SELECT version FROM _schema_version WHERE id = 1")
        assert row["version"] >= 5

    def test_schedules_table_columns(self, db):
        cursor = db.execute("PRAGMA table_info(schedules)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id", "name", "cron_expr", "task_type", "task_goal",
            "notify_mode", "condition_text", "last_result_hash",
            "last_fired_at", "next_fire_at", "enabled", "created_at",
            # v21. A schedule fires later with LOCAL_GRANTS, so the row has to
            # say who asked for it -- see KI-30.
            "installed_by",
        }
        assert expected == columns
