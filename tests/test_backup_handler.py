"""Tests for the manage_backup intent handler."""
import tempfile
from pathlib import Path

import pytest

import assistant.actions as _act
from assistant.actions.backup import _classify_action
from assistant.io.backup import orchestrator


def test_classify_action_restore():
    assert _classify_action("restore my backup") == "restore"
    assert _classify_action("restore from cloud") == "restore"


def test_classify_action_status():
    assert _classify_action("when was my last backup") == "status"
    assert _classify_action("backup status") == "status"


def test_classify_action_disable():
    assert _classify_action("disable backup") == "disable"
    assert _classify_action("turn off cloud backup") == "disable"
    assert _classify_action("stop backing up") == "disable"


def test_classify_action_enable():
    assert _classify_action("enable backup") == "enable"
    assert _classify_action("set up cloud backup") == "enable"
    assert _classify_action("turn on backup") == "enable"


def test_classify_action_defaults_to_backup_now():
    assert _classify_action("back up now") == "backup_now"
    assert _classify_action("do a backup") == "backup_now"


@pytest.mark.asyncio
async def test_handler_status_fresh_db_reports_not_set_up(monkeypatch):
    from assistant.actions.backup import handle_manage_backup
    from assistant.storage.db import Database, _reset_for_testing
    import assistant.storage.db as db_module
    import tempfile
    from pathlib import Path

    _reset_for_testing()
    tmp = tempfile.mkdtemp()
    db_module.init_db(Path(tmp) / "test.db")

    # Fresh DB: backup_enabled defaults to False, so _status() reports the
    # "not set up" branch rather than a "never backed up" one — the
    # last_status="never" default in _status() only guards a branch that's
    # unreachable while last_at is falsy. Assert on the actual reachable
    # message for a never-configured install.
    result = await handle_manage_backup({"goal": "backup status"}, "")
    assert "isn't set up" in result.lower()

    _reset_for_testing()


# ─── Fixtures for the fix-review tests below ────────────────────────────────


@pytest.fixture
def db_session():
    """Fresh SQLite DB + settings repo, torn down after the test."""
    from assistant.storage.db import _reset_for_testing
    import assistant.storage.db as db_module
    from assistant.storage.repos.settings import SettingsRepo

    _reset_for_testing()
    tmp = tempfile.mkdtemp()
    db_module.init_db(Path(tmp) / "test.db")
    settings = SettingsRepo(db_module.get_db())
    yield settings
    _reset_for_testing()


@pytest.fixture(autouse=True)
def _clear_backup_state():
    yield
    _act.pending_backup_confirm_phrase.clear()
    _act.pending_backup_oauth.clear()
    orchestrator.set_unlocked_key(None)


# ─── Important #1: status after a failed backup ─────────────────────────────


@pytest.mark.asyncio
async def test_status_after_failed_backup_reports_failure(db_session):
    from assistant.actions.backup import handle_manage_backup

    db_session.set("backup_enabled", True, source="test")
    db_session.set("backup_last_backup_status", "failed", source="test")
    # No backup_last_backup_at — a failed attempt never reaches the success
    # path in orchestrator.run_backup(), which is exactly what previously
    # made _status() report "hasn't run yet" instead of surfacing the failure.

    result = await handle_manage_backup({"goal": "backup status"}, "")
    assert "failed" in result.lower()
    assert "hasn't run yet" not in result.lower()


@pytest.mark.asyncio
async def test_status_after_success_still_reports_last_backup(db_session):
    from assistant.actions.backup import handle_manage_backup

    db_session.set("backup_enabled", True, source="test")
    db_session.set("backup_last_backup_at", "2026-08-02T00:00:00+00:00", source="test")
    db_session.set("backup_last_backup_status", "success", source="test")

    result = await handle_manage_backup({"goal": "backup status"}, "")
    assert "2026-08-02" in result
    assert "success" in result.lower()


# ─── Important #2: re-enable guard against silently orphaning backups ───────


@pytest.mark.asyncio
async def test_enable_already_enabled_warns_instead_of_regenerating(db_session):
    from assistant.actions.backup import handle_manage_backup

    db_session.set("backup_enabled", True, source="test")

    result = await handle_manage_backup({"goal": "enable backup"}, "")

    assert not _act.pending_backup_confirm_phrase.active
    assert not orchestrator.is_unlocked()
    assert "already" in result.lower()


@pytest.mark.asyncio
async def test_enable_already_enabled_with_explicit_confirmation_proceeds(db_session):
    from assistant.actions.backup import handle_manage_backup

    db_session.set("backup_enabled", True, source="test")

    result = await handle_manage_backup({"goal": "enable backup, replace it"}, "")

    assert _act.pending_backup_confirm_phrase.active
    assert orchestrator.is_unlocked()
    assert "recovery phrase" in result.lower()


@pytest.mark.asyncio
async def test_enable_fresh_install_still_proceeds_without_confirmation(db_session):
    from assistant.actions.backup import handle_manage_backup

    result = await handle_manage_backup({"goal": "enable backup"}, "")

    assert _act.pending_backup_confirm_phrase.active
    assert orchestrator.is_unlocked()
    assert "recovery phrase" in result.lower()


# ─── Important #3: backup_now failure message must be TTS-safe ─────────────


@pytest.mark.asyncio
async def test_backup_now_failure_message_is_short_and_generic(db_session, monkeypatch, caplog):
    from assistant.actions.backup import handle_manage_backup

    orchestrator.set_unlocked_key(b"0" * 32)

    def _raise(provider_name):
        raise RuntimeError(
            "Connection to https://www.googleapis.com/upload/drive/v3/files "
            "failed with status code 503 at C:\\Users\\omkho\\AppData\\Local\\Temp\\tenka_backup.tar"
        )

    monkeypatch.setattr(orchestrator, "run_backup", _raise)

    with caplog.at_level("ERROR"):
        result = await handle_manage_backup({"goal": "back up now"}, "")

    assert len(result) < 120
    assert "https://" not in result
    assert "C:\\" not in result
    assert "503" not in result

    # The real error still gets logged server-side.
    assert any("503" in record.message for record in caplog.records)
