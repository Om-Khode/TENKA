"""A restore through the Studio API must request the same shutdown the voice
path does.

orchestrator.run_restore() closes the live SQLite connection before swapping
the file on disk and deliberately never reopens it: the 13 repos each cache
their own handle at startup, so a fresh singleton would not reach them. Only
a real process restart rebuilds those caches.

actions/backup_pending.py (the voice path) has always called
shutdown_signal.request() afterward. LiveSystemRuntime.restore_backup() did
not, so a restore triggered from Studio reported success and left the
assistant running against a closed database -- "Cannot operate on a closed
database" from the scheduler and a reminder poller failing every ten seconds,
until someone noticed and restarted her by hand. Observed live on 2026-08-10.
"""
import asyncio

import pytest

from assistant.core import shutdown_signal


@pytest.fixture(autouse=True)
def _reset_signal():
    shutdown_signal._reset_for_testing()
    yield
    shutdown_signal._reset_for_testing()


def _runtime():
    from assistant.actions.studio_runtime_system import LiveSystemRuntime
    return LiveSystemRuntime()


def test_successful_restore_requests_shutdown(monkeypatch):
    from assistant.io.backup import crypto, orchestrator

    monkeypatch.setattr(crypto, "is_valid_recovery_phrase", lambda phrase: True)
    monkeypatch.setattr(orchestrator, "run_restore", lambda phrase: None)

    assert shutdown_signal.is_requested() is False
    assert asyncio.run(_runtime().restore_backup("a b c d e f g h i j k l")) is True
    assert shutdown_signal.is_requested() is True, (
        "restore replaced the DB and closed the live connection; without a "
        "shutdown request the process keeps running against it"
    )


def test_failed_restore_does_not_request_shutdown(monkeypatch):
    """A restore that raised never swapped the database, so the process is
    still healthy -- killing her over a failure she recovered from would turn
    a refused restore into an outage."""
    from assistant.io.backup import crypto, orchestrator

    monkeypatch.setattr(crypto, "is_valid_recovery_phrase", lambda phrase: True)

    def _boom(phrase):
        raise RuntimeError("wrong phrase for this archive")

    monkeypatch.setattr(orchestrator, "run_restore", _boom)

    assert asyncio.run(_runtime().restore_backup("a b c d e f g h i j k l")) is False
    assert shutdown_signal.is_requested() is False


def test_malformed_phrase_never_reaches_the_orchestrator(monkeypatch):
    from assistant.io.backup import crypto, orchestrator

    monkeypatch.setattr(crypto, "is_valid_recovery_phrase", lambda phrase: False)

    def _must_not_run(phrase):
        raise AssertionError("run_restore called for a phrase that failed validation")

    monkeypatch.setattr(orchestrator, "run_restore", _must_not_run)

    assert asyncio.run(_runtime().restore_backup("nope")) is False
    assert shutdown_signal.is_requested() is False
