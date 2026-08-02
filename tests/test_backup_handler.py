"""Tests for the manage_backup intent handler."""
import pytest

from assistant.actions.backup import _classify_action


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
