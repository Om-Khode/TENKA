import sqlite3
import pytest
from unittest.mock import patch


def _make_db():
    """Create an in-memory Database instance for testing."""
    from assistant.storage.db import Database
    db = Database.__new__(Database)
    db._conn = sqlite3.connect(":memory:")
    db._conn.row_factory = sqlite3.Row
    db._conn.execute("PRAGMA journal_mode=WAL")
    db._migrate_v1()
    db._set_version(1)
    db._migrate_v2()
    db._set_version(2)
    db._migrate_v3()
    db._set_version(3)
    db._migrate_v4()
    db._set_version(4)
    db._migrate_v5()
    db._set_version(5)
    db._migrate_v6()
    db._set_version(6)
    db._migrate_v7()
    db._set_version(7)
    return db


# ─── Schema Tests ─────────────────────────────────────────────────────

def test_v8_migration_creates_automation_cache_table():
    db = _make_db()
    db._migrate_v8()
    db._set_version(8)
    row = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='automation_cache'"
    ).fetchone()
    assert row is not None, "automation_cache table should exist after v8 migration"


def test_v8_migration_creates_index():
    db = _make_db()
    db._migrate_v8()
    db._set_version(8)
    row = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_ac_lookup'"
    ).fetchone()
    assert row is not None, "idx_ac_lookup index should exist after v8 migration"


def test_v8_migration_is_idempotent():
    db = _make_db()
    db._migrate_v8()
    db._migrate_v8()  # should not raise


import json
from datetime import datetime, timedelta


def _make_repo():
    db = _make_db()
    db._migrate_v8()
    db._set_version(8)
    from assistant.storage.repos.automation_cache import AutomationCacheRepo
    return AutomationCacheRepo(db), db


# ─── Repo Tests ───────────────────────────────────────────────────────

def test_repo_save_and_get():
    repo, _ = _make_repo()
    steps = [{"action": "click", "params": {"selector": "name:Save"}}]
    repo.save("native", "notepad", "save_file", "save the file in notepad", steps)

    entry = repo.get("native", "notepad", "save_file")
    assert entry is not None
    assert entry["goal_text"] == "save the file in notepad"
    assert json.loads(entry["steps_json"]) == steps
    assert entry["hit_count"] == 0


def test_repo_get_returns_none_on_miss():
    repo, _ = _make_repo()
    assert repo.get("native", "notepad", "nonexistent") is None


def test_repo_save_upserts_on_duplicate():
    repo, _ = _make_repo()
    steps_v1 = [{"action": "click", "params": {"selector": "name:Save"}}]
    steps_v2 = [{"action": "press_key", "params": {"key": "ctrl+s"}}]
    repo.save("native", "notepad", "save_file", "save the file", steps_v1)
    repo.save("native", "notepad", "save_file", "save the file", steps_v2)

    entry = repo.get("native", "notepad", "save_file")
    assert json.loads(entry["steps_json"]) == steps_v2
    assert entry["hit_count"] == 0


def test_repo_record_hit_increments_count():
    repo, _ = _make_repo()
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "spotify", "play_pause", "play pause on spotify", steps)

    repo.record_hit("native", "spotify", "play_pause")
    entry = repo.get("native", "spotify", "play_pause")
    assert entry["hit_count"] == 1

    repo.record_hit("native", "spotify", "play_pause")
    entry = repo.get("native", "spotify", "play_pause")
    assert entry["hit_count"] == 2


def test_repo_delete():
    repo, _ = _make_repo()
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "spotify", "play_pause", "play pause", steps)

    deleted = repo.delete("native", "spotify", "play_pause")
    assert deleted is True
    assert repo.get("native", "spotify", "play_pause") is None


def test_repo_delete_returns_false_on_miss():
    repo, _ = _make_repo()
    assert repo.delete("native", "notepad", "nonexistent") is False


def test_repo_cleanup_expired():
    repo, db = _make_repo()
    steps = [{"action": "press_key", "params": {"key": "space"}}]

    old_date = (datetime.now() - timedelta(days=40)).isoformat()
    db.execute(
        """INSERT INTO automation_cache
           (backend, app_name, goal_slug, goal_text, steps_json,
            hit_count, created_at, last_hit_at, version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("native", "old_app", "old_task", "old goal",
         json.dumps(steps), 5, old_date, old_date, 1),
    )
    db.commit()

    repo.save("native", "new_app", "new_task", "new goal", steps)

    removed = repo.cleanup_expired(max_age_days=30)
    assert removed == 1
    assert repo.get("native", "old_app", "old_task") is None
    assert repo.get("native", "new_app", "new_task") is not None


def test_repo_list_all():
    repo, _ = _make_repo()
    steps = [{"action": "press_key", "params": {"key": "space"}}]
    repo.save("native", "spotify", "play", "play music", steps)
    repo.save("browser", "chrome", "search", "search google", steps)

    entries = repo.list_all()
    assert len(entries) == 2


# ─── Step Cache Facade Tests ──────────────────────────────────────────

def test_make_goal_slug_strips_stop_words():
    from assistant.automation.step_cache import _make_goal_slug
    slug = _make_goal_slug("play my liked songs on spotify")
    assert "my" not in slug
    assert "on" not in slug
    assert "play" in slug
    assert "liked" in slug
    assert "songs" in slug
    assert "spotify" in slug


def test_make_goal_slug_deterministic():
    from assistant.automation.step_cache import _make_goal_slug
    a = _make_goal_slug("save the file in notepad")
    b = _make_goal_slug("save the file in notepad")
    assert a == b


def test_make_goal_slug_order_independent():
    from assistant.automation.step_cache import _make_goal_slug
    a = _make_goal_slug("play songs on spotify")
    b = _make_goal_slug("on spotify play songs")
    assert a == b


def test_goal_matches_cached_similar():
    from assistant.automation.step_cache import _goal_matches_cached
    assert _goal_matches_cached(
        "play my liked songs on spotify",
        "play liked songs on spotify",
    ) is True


def test_goal_matches_cached_different():
    from assistant.automation.step_cache import _goal_matches_cached
    assert _goal_matches_cached(
        "play my liked songs on spotify",
        "save document in notepad",
    ) is False


def test_goal_matches_cached_empty_stored():
    from assistant.automation.step_cache import _goal_matches_cached
    assert _goal_matches_cached("play music", "") is True


# ─── Router Integration Tests (Native) ────────────────────────────────

@pytest.mark.asyncio
async def test_native_task_saves_to_cache_on_success():
    """After a successful LLM-planned native task, steps are cached."""
    from unittest.mock import AsyncMock, patch
    from assistant.automation.router import _execute_native_task

    planned_steps = [{"action": "click", "params": {"selector": "name:Save"}}]
    llm_response = json.dumps(planned_steps)

    with patch("assistant.automation.router._detect_running_app", return_value="Notepad"), \
         patch("assistant.automation.router._extract_target_app", return_value=(None, "save the document")), \
         patch("assistant.automation.step_cache.load_cached_steps", return_value=None), \
         patch("assistant.automation.step_cache.save_cached_steps") as mock_save, \
         patch("assistant.automation.router._maybe_await", new_callable=AsyncMock, return_value=llm_response), \
         patch("assistant.automation.router._extract_json_array", return_value=planned_steps), \
         patch("assistant.automation.native.list_elements", new_callable=AsyncMock, return_value=None), \
         patch("assistant.automation.native.run_app_steps", new_callable=AsyncMock, return_value="Done"), \
         patch("assistant.automation.native.focus_window", new_callable=AsyncMock, return_value="Focused"):

        result = await _execute_native_task("save the document", AsyncMock(return_value=llm_response))

        assert result != "__FALLBACK__"
        mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_native_task_uses_cache_on_hit():
    """On cache hit, skip LLM planner and use cached steps directly."""
    from unittest.mock import AsyncMock, patch
    from assistant.automation.router import _execute_native_task

    cached_steps = [{"action": "press_key", "params": {"key": "ctrl+s"}}]

    with patch("assistant.automation.router._detect_running_app", return_value="Notepad"), \
         patch("assistant.automation.router._extract_target_app", return_value=(None, "save the document")), \
         patch("assistant.automation.step_cache.load_cached_steps", return_value=cached_steps), \
         patch("assistant.automation.step_cache.delete_cached_steps") as mock_delete, \
         patch("assistant.automation.router._maybe_await", new_callable=AsyncMock) as mock_llm, \
         patch("assistant.automation.native.list_elements", new_callable=AsyncMock, return_value=None), \
         patch("assistant.automation.native.run_app_steps", new_callable=AsyncMock, return_value="Done"), \
         patch("assistant.automation.native.focus_window", new_callable=AsyncMock, return_value="Focused"):

        result = await _execute_native_task("save the document", AsyncMock())

        # LLM planner should NOT have been called — cache hit returned directly
        mock_llm.assert_not_called()
        mock_delete.assert_not_called()


@pytest.mark.asyncio
async def test_native_task_deletes_cache_on_failure():
    """On cached step execution failure, delete cache and fall back to LLM."""
    from unittest.mock import AsyncMock, patch
    from assistant.automation.router import _execute_native_task

    cached_steps = [{"action": "click", "params": {"selector": "name:Gone"}}]

    # run_app_steps will be called twice: once for cached steps (fails),
    # once for LLM steps (also returns error so we get __FALLBACK__).
    call_count = {"n": 0}

    async def _run_steps_side_effect(steps):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "Error: element not found"
        return "Error: still broken"

    with patch("assistant.automation.router._detect_running_app", return_value="Notepad"), \
         patch("assistant.automation.router._extract_target_app", return_value=(None, "click gone button")), \
         patch("assistant.automation.step_cache.load_cached_steps", return_value=cached_steps), \
         patch("assistant.automation.step_cache.save_cached_steps"), \
         patch("assistant.automation.step_cache.delete_cached_steps") as mock_delete, \
         patch("assistant.automation.router._maybe_await", new_callable=AsyncMock, return_value="[]"), \
         patch("assistant.automation.router._extract_json_array", return_value=None), \
         patch("assistant.automation.native.list_elements", new_callable=AsyncMock, return_value=None), \
         patch("assistant.automation.native.run_app_steps", new_callable=AsyncMock, side_effect=_run_steps_side_effect), \
         patch("assistant.automation.native.focus_window", new_callable=AsyncMock, return_value="Focused"):

        result = await _execute_native_task("click gone button", AsyncMock())

        mock_delete.assert_called_once()


@pytest.mark.asyncio
async def test_native_task_does_not_cache_on_verify_failed():
    """Steps must NOT be cached when verification fails."""
    from unittest.mock import AsyncMock, patch
    from assistant.automation.router import _execute_native_task

    planned_steps = [{"action": "type", "params": {"text": "hello", "window": "Notepad"}}]
    llm_response = json.dumps(planned_steps)
    verify_result = (
        "VERIFY_FAILED|step=1|tier=vision|obs=Typed into wrong window\n"
        "Typed text into focus"
    )

    with patch("assistant.automation.router._detect_running_app", return_value="Notepad"), \
         patch("assistant.automation.router._extract_target_app", return_value=(None, "type hello")), \
         patch("assistant.automation.step_cache.load_cached_steps", return_value=None), \
         patch("assistant.automation.step_cache.save_cached_steps") as mock_save, \
         patch("assistant.automation.router._maybe_await", new_callable=AsyncMock, return_value=llm_response), \
         patch("assistant.automation.router._extract_json_array", return_value=planned_steps), \
         patch("assistant.automation.native.list_elements", new_callable=AsyncMock, return_value=None), \
         patch("assistant.automation.native.run_app_steps", new_callable=AsyncMock, return_value=verify_result), \
         patch("assistant.automation.native.focus_window", new_callable=AsyncMock, return_value="Focused"):

        result = await _execute_native_task("type hello", AsyncMock(return_value=llm_response))

        assert "VERIFY_FAILED" in result
        mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_native_task_does_not_cache_on_abort():
    """Steps must NOT be cached when the user aborts via ESC."""
    from unittest.mock import AsyncMock, patch
    from assistant.automation.router import _execute_native_task

    planned_steps = [{"action": "type", "params": {"text": "hello", "window": "Notepad"}}]
    llm_response = json.dumps(planned_steps)
    abort_result = (
        "Focused window: Notepad\n"
        "Pressed key: ctrl+n\n"
        "[ABORT] Aborted by user."
    )

    with patch("assistant.automation.router._detect_running_app", return_value="Notepad"), \
         patch("assistant.automation.router._extract_target_app", return_value=(None, "type hello")), \
         patch("assistant.automation.step_cache.load_cached_steps", return_value=None), \
         patch("assistant.automation.step_cache.save_cached_steps") as mock_save, \
         patch("assistant.automation.router._maybe_await", new_callable=AsyncMock, return_value=llm_response), \
         patch("assistant.automation.router._extract_json_array", return_value=planned_steps), \
         patch("assistant.automation.native.list_elements", new_callable=AsyncMock, return_value=None), \
         patch("assistant.automation.native.run_app_steps", new_callable=AsyncMock, return_value=abort_result), \
         patch("assistant.automation.native.focus_window", new_callable=AsyncMock, return_value="Focused"):

        result = await _execute_native_task("type hello", AsyncMock(return_value=llm_response))

        assert "[ABORT]" in result
        mock_save.assert_not_called()


# ─── Router Integration Tests (Browser) ───────────────────────────────

@pytest.mark.asyncio
async def test_browser_task_saves_to_cache_on_success():
    from unittest.mock import AsyncMock, patch

    planned_steps = [{"action": "navigate", "params": {"url": "https://example.com"}}]

    with patch("assistant.automation.step_cache.load_cached_steps", return_value=None), \
         patch("assistant.automation.step_cache.save_cached_steps") as mock_save, \
         patch("assistant.automation.router._maybe_await", new_callable=AsyncMock, return_value=json.dumps(planned_steps)), \
         patch("assistant.automation.router._extract_json_array", return_value=planned_steps), \
         patch("assistant.automation.router.browser_automation", create=True) as mock_browser:

        mock_browser.run_browser_steps = AsyncMock(return_value="Page loaded")

        from assistant.automation.router import _execute_browser_task
        result = await _execute_browser_task("open example website and extract info", AsyncMock())

        assert result != "__FALLBACK__"
        mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_browser_task_uses_cache_on_hit():
    from unittest.mock import AsyncMock, patch

    cached_steps = [{"action": "navigate", "params": {"url": "https://example.com"}}]

    with patch("assistant.automation.step_cache.load_cached_steps", return_value=cached_steps), \
         patch("assistant.automation.router._maybe_await") as mock_llm, \
         patch("assistant.automation.router.browser_automation", create=True) as mock_browser:

        mock_browser.run_browser_steps = AsyncMock(return_value="Page loaded")

        from assistant.automation.router import _execute_browser_task
        result = await _execute_browser_task("open example website and extract info", AsyncMock())

        mock_llm.assert_not_called()


# ─── Edge Case Tests ──────────────────────────────────────────────────

def test_make_goal_slug_handles_empty():
    from assistant.automation.step_cache import _make_goal_slug
    assert _make_goal_slug("") == "unknown"
    assert _make_goal_slug("   ") == "unknown"


def test_make_goal_slug_handles_only_stop_words():
    from assistant.automation.step_cache import _make_goal_slug
    assert _make_goal_slug("do it for me") == "unknown"


def test_make_goal_slug_handles_punctuation():
    from assistant.automation.step_cache import _make_goal_slug
    slug = _make_goal_slug("play 'Bohemian Rhapsody' on Spotify!")
    assert "bohemian" in slug
    assert "rhapsody" in slug
    assert "play" in slug
    assert "spotify" in slug


def test_make_goal_slug_handles_special_chars():
    from assistant.automation.step_cache import _make_goal_slug
    slug = _make_goal_slug("open https://example.com/path?q=test")
    assert "open" in slug


def test_goal_matches_cached_partial_overlap():
    from assistant.automation.step_cache import _goal_matches_cached
    assert _goal_matches_cached(
        "type hello world in notepad",
        "type hello in notepad",
    ) is True


def test_goal_matches_cached_completely_different():
    from assistant.automation.step_cache import _goal_matches_cached
    assert _goal_matches_cached(
        "play music on spotify",
        "save file in notepad",
    ) is False


def test_cache_different_apps_same_slug():
    """Same goal slug for different apps should not collide."""
    repo, _ = _make_repo()
    steps_a = [{"action": "press_key", "params": {"key": "space"}}]
    steps_b = [{"action": "click", "params": {"selector": "name:Play"}}]

    repo.save("native", "spotify", "play_music", "play music", steps_a)
    repo.save("native", "vlc", "play_music", "play music", steps_b)

    entry_a = repo.get("native", "spotify", "play_music")
    entry_b = repo.get("native", "vlc", "play_music")
    assert json.loads(entry_a["steps_json"]) == steps_a
    assert json.loads(entry_b["steps_json"]) == steps_b


def test_cache_different_backends_same_app():
    """Same app on different backends should not collide."""
    repo, _ = _make_repo()
    steps_native = [{"action": "press_key", "params": {"key": "ctrl+s"}}]
    steps_browser = [{"action": "click", "params": {"selector": "#save"}}]

    repo.save("native", "app", "save", "save file", steps_native)
    repo.save("browser", "app", "save", "save file", steps_browser)

    assert json.loads(repo.get("native", "app", "save")["steps_json"]) == steps_native
    assert json.loads(repo.get("browser", "app", "save")["steps_json"]) == steps_browser


# ─── Robustness Tests (from code review) ──────────────────────────────

def test_load_returns_none_when_db_unavailable():
    """Facade gracefully returns None when DB is not initialized."""
    from unittest.mock import patch
    from assistant.automation.step_cache import load_cached_steps

    with patch("assistant.automation.step_cache.get_db", return_value=None):
        result = load_cached_steps("native", "app", "do something")
        assert result is None


def test_save_noop_when_db_unavailable():
    """Facade silently no-ops when DB is not initialized."""
    from unittest.mock import patch
    from assistant.automation.step_cache import save_cached_steps

    with patch("assistant.automation.step_cache.get_db", return_value=None):
        save_cached_steps("native", "app", "do something", [{"action": "click"}])


def test_load_deletes_corrupt_steps_json():
    """Corrupt steps_json is treated as cache miss and entry is deleted."""
    from unittest.mock import patch, MagicMock
    from assistant.automation.step_cache import load_cached_steps

    mock_repo = MagicMock()
    mock_repo.get.return_value = {
        "goal_text": "do something",
        "hit_count": 5,
        "steps_json": "{INVALID JSON",
        "version": 1,
    }

    with patch("assistant.automation.step_cache._get_repo", return_value=mock_repo):
        result = load_cached_steps("native", "app", "do something")
        assert result is None
        mock_repo.delete.assert_called_once()


def test_load_rejects_stale_version():
    """Cache entries with old version are deleted and treated as miss."""
    from unittest.mock import patch, MagicMock
    from assistant.automation.step_cache import load_cached_steps

    mock_repo = MagicMock()
    mock_repo.get.return_value = {
        "goal_text": "do something",
        "hit_count": 5,
        "steps_json": '[{"action": "click"}]',
        "version": 0,
    }

    with patch("assistant.automation.step_cache._get_repo", return_value=mock_repo):
        result = load_cached_steps("native", "app", "do something")
        assert result is None
        mock_repo.delete.assert_called_once()


# ─── Window-param stripping on cache save ────────────────────────────────────

@pytest.mark.asyncio
async def test_native_task_strips_window_from_cached_steps():
    """Cached steps must NOT contain window params (they go stale)."""
    from unittest.mock import AsyncMock, patch
    from assistant.automation.router import _execute_native_task

    planned_steps = [{"action": "type", "params": {"text": "hello", "window": "Untitled - Notepad"}}]
    llm_response = json.dumps(planned_steps)

    with patch("assistant.automation.router._detect_running_app", return_value="Untitled - Notepad"), \
         patch("assistant.automation.router._extract_target_app", return_value=(None, "type hello")), \
         patch("assistant.automation.step_cache.load_cached_steps", return_value=None), \
         patch("assistant.automation.step_cache.save_cached_steps") as mock_save, \
         patch("assistant.automation.router._maybe_await", new_callable=AsyncMock, return_value=llm_response), \
         patch("assistant.automation.router._extract_json_array", return_value=planned_steps), \
         patch("assistant.automation.native.list_elements", new_callable=AsyncMock, return_value=None), \
         patch("assistant.automation.native.run_app_steps", new_callable=AsyncMock, return_value="Typed text"), \
         patch("assistant.automation.native.focus_window", new_callable=AsyncMock, return_value="Focused"):

        await _execute_native_task("type hello", AsyncMock(return_value=llm_response))

        mock_save.assert_called_once()
        saved_steps = mock_save.call_args[0][3]
        for step in saved_steps:
            assert "window" not in step.get("params", {}), \
                f"window param should be stripped before caching: {step}"


# ─── VERIFY_FAILED interception in action handlers ───────────────────────────

@pytest.mark.asyncio
async def test_browser_action_intercepts_verify_failed():
    """handle_browser_action must NOT return raw VERIFY_FAILED strings."""
    from unittest.mock import AsyncMock, patch

    verify_result = (
        "VERIFY_FAILED|step=3|tier=pre_check|obs=button not visible\n"
        "Navigated OK\nFilled OK"
    )

    with patch("assistant.actions.da_handlers._llm_text", new_callable=AsyncMock), \
         patch("assistant.automation.router._execute_browser_task",
               new_callable=AsyncMock, return_value=verify_result), \
         patch("assistant.llm.contracts.ask_for_synthesis",
               new_callable=AsyncMock, return_value="That didn't work."):

        from assistant.actions.da_handlers import handle_browser_action
        result = await handle_browser_action(
            {"goal": "search wikipedia"}, "", None, _from_planner=False
        )

        assert not result.startswith("VERIFY_FAILED"), \
            f"Raw VERIFY_FAILED must not reach user: {result[:100]}"


@pytest.mark.asyncio
async def test_app_action_intercepts_verify_failed():
    """handle_app_action must NOT return raw VERIFY_FAILED strings."""
    from unittest.mock import AsyncMock, patch

    verify_result = (
        "VERIFY_FAILED|step=1|tier=vision|obs=wrong window focused\n"
        "Focused window: Edge"
    )

    with patch("assistant.actions.da_handlers._llm_text", new_callable=AsyncMock), \
         patch("assistant.automation.router._execute_native_task",
               new_callable=AsyncMock, return_value=verify_result), \
         patch("assistant.llm.contracts.ask_for_synthesis",
               new_callable=AsyncMock, return_value="That didn't work."):

        from assistant.actions.da_handlers import handle_app_action
        result = await handle_app_action(
            {"goal": "type hello in notepad"}, "", None, _from_planner=False
        )

        assert not result.startswith("VERIFY_FAILED"), \
            f"Raw VERIFY_FAILED must not reach user: {result[:100]}"
