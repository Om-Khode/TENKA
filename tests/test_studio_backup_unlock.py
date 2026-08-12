"""LiveSystemRuntime.unlock_backup against the real orchestrator.

The route tests in test_api_system.py drive a fake. This one proves the actual
implementation arms the real in-memory key, because the bug being fixed was a
whole-feature outage nobody could see: the key dies with every restart, the
scheduler skips silently, and `backup_enabled` stays true. Observed on
2026-08-10 with a last-backup date seven days stale.
"""
import asyncio

import pytest

from assistant.io.backup import orchestrator

PHRASE = "amber moss steel gold bone quiet signal drift"


@pytest.fixture(autouse=True)
def _locked():
    """Start locked and leave locked. This touches process-global state, so a
    leaked unlocked key would silently change what a later test observes."""
    orchestrator.set_unlocked_key(None)
    yield
    orchestrator.set_unlocked_key(None)


def _runtime():
    from assistant.actions.studio_runtime_system import LiveSystemRuntime
    return LiveSystemRuntime()


def test_a_valid_phrase_arms_the_real_key(monkeypatch):
    from assistant.io.backup import crypto

    monkeypatch.setattr(crypto, "is_valid_recovery_phrase", lambda phrase: True)
    monkeypatch.setattr(crypto, "derive_key", lambda phrase: b"k" * 32)

    assert orchestrator.is_unlocked() is False
    assert asyncio.run(_runtime().unlock_backup(PHRASE)) is True
    assert orchestrator.is_unlocked() is True
    assert orchestrator.get_unlocked_key() == b"k" * 32


def test_a_malformed_phrase_leaves_it_locked(monkeypatch):
    """False, not an exception -- the route turns this into a 400 whose body
    never contains the submitted phrase."""
    from assistant.io.backup import crypto

    monkeypatch.setattr(crypto, "is_valid_recovery_phrase", lambda phrase: False)

    def _must_not_derive(phrase):
        raise AssertionError("derive_key called for a phrase that failed validation")

    monkeypatch.setattr(crypto, "derive_key", _must_not_derive)

    assert asyncio.run(_runtime().unlock_backup("nope")) is False
    assert orchestrator.is_unlocked() is False


def test_run_backup_refuses_while_locked():
    """The reason unlock has to exist. run_backup raises RuntimeError, which
    errors.py maps to 409 -- the "precondition failed" a user saw with no way
    to resolve it."""
    with pytest.raises(RuntimeError, match="not unlocked"):
        orchestrator.run_backup()


def test_the_scheduler_warns_rather_than_skipping_silently(monkeypatch, caplog):
    """This is what let a week pass unnoticed. The skip was logged at DEBUG,
    invisible in a default install, while the panel showed a stale date."""
    import logging

    monkeypatch.setattr(orchestrator, "is_unlocked", lambda: False)

    class _Settings:
        def __init__(self, db):
            pass

        @staticmethod
        def get(key, default=None):
            return True if key == "backup_enabled" else default

    # _maybe_run_scheduled_backup imports these INSIDE the function, so the
    # names must be patched on their source modules -- patching them on
    # `orchestrator` would bind nothing and the test would pass vacuously.
    import assistant.storage.db as db_mod
    import assistant.storage.repos.settings as settings_mod

    monkeypatch.setattr(db_mod, "get_db", lambda: object())
    monkeypatch.setattr(settings_mod, "SettingsRepo", _Settings)

    with caplog.at_level(logging.WARNING, logger="backup.orchestrator"):
        orchestrator._maybe_run_scheduled_backup()

    assert any("NOT unlocked" in r.message for r in caplog.records), (
        "a skipped scheduled backup must be visible at WARNING -- at DEBUG the "
        "user has no way to learn backups stopped"
    )
