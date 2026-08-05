"""Tests for backup onboarding pending-states."""
import pytest

import assistant.actions as _act
from assistant.actions.backup_pending import (
    handle_pending_backup_confirm_phrase,
    handle_pending_backup_oauth,
    handle_pending_backup_unlock_phrase,
    handle_pending_backup_restore_phrase,
)
from assistant.io.backup import crypto, orchestrator


@pytest.fixture
def db_session():
    """Fresh SQLite DB + settings repo, torn down after the test."""
    import tempfile
    from pathlib import Path

    from assistant.storage.db import _reset_for_testing
    import assistant.storage.db as db_module
    from assistant.storage.repos.settings import SettingsRepo

    _reset_for_testing()
    tmp = tempfile.mkdtemp()
    db_module.init_db(Path(tmp) / "test.db")
    yield SettingsRepo(db_module.get_db())
    _reset_for_testing()


@pytest.fixture(autouse=True)
def _clear_pending():
    yield
    _act.pending_backup_confirm_phrase.clear()
    _act.pending_backup_oauth.clear()
    _act.pending_backup_unlock_phrase.clear()
    _act.pending_backup_restore_phrase.clear()
    orchestrator.set_unlocked_key(None)


@pytest.mark.asyncio
async def test_confirm_phrase_inactive_returns_none():
    result = await handle_pending_backup_confirm_phrase("yes")
    assert result is None


@pytest.mark.asyncio
async def test_confirm_phrase_accepts_yes_and_starts_oauth(monkeypatch):
    _act.pending_backup_confirm_phrase.set({"phrase": "abandon ability able about above absent absorb abstract absurd abuse access accident"})
    result = await handle_pending_backup_confirm_phrase("yes I saved it")
    assert "Google Drive" in result or "google" in result.lower()
    assert not _act.pending_backup_confirm_phrase.active
    assert _act.pending_backup_oauth.active


@pytest.mark.asyncio
async def test_confirm_phrase_caches_the_key_only_on_confirmation():
    """Important #6 — the key is armed here, not in _enable(), so an
    abandoned flow can't leave an orphan key that later backups upload
    under. And the confirmation response must not echo the phrase."""
    phrase = crypto.generate_recovery_phrase()
    _act.pending_backup_confirm_phrase.set({"phrase": phrase})

    assert not orchestrator.is_unlocked()
    result = await handle_pending_backup_confirm_phrase("saved it")

    assert orchestrator.is_unlocked()
    assert orchestrator.get_unlocked_key() == crypto.derive_key(phrase)
    assert phrase not in result


@pytest.mark.asyncio
async def test_confirm_phrase_declined_leaves_session_locked():
    phrase = crypto.generate_recovery_phrase()
    _act.pending_backup_confirm_phrase.set({"phrase": phrase})

    await handle_pending_backup_confirm_phrase("not yet")

    assert not orchestrator.is_unlocked()
    assert _act.pending_backup_confirm_phrase.active


@pytest.mark.asyncio
async def test_confirm_phrase_rejects_unclear_answer():
    _act.pending_backup_confirm_phrase.set({"phrase": "test phrase"})
    result = await handle_pending_backup_confirm_phrase("what do you mean")
    assert _act.pending_backup_confirm_phrase.active
    assert result is not None


@pytest.mark.asyncio
async def test_oauth_inactive_returns_none():
    result = await handle_pending_backup_oauth("some client id")
    assert result is None


@pytest.mark.asyncio
async def test_oauth_step_has_app_yes_prompts_for_client_id():
    _act.pending_backup_oauth.set({"step": "has_app"})
    result = await handle_pending_backup_oauth("yes I have one")
    assert _act.pending_backup_oauth.payload["step"] == "client_id"
    assert "client ID" in result


# ─── Regression: negated answers must not be misread as affirmative ────────


@pytest.mark.asyncio
async def test_confirm_phrase_not_done_yet_does_not_advance(monkeypatch):
    _act.pending_backup_confirm_phrase.set({"phrase": "test phrase"})
    result = await handle_pending_backup_confirm_phrase("I'm not done yet")
    assert _act.pending_backup_confirm_phrase.active
    assert not _act.pending_backup_oauth.active
    assert result is not None


@pytest.mark.asyncio
async def test_confirm_phrase_not_yet_still_writing_does_not_advance(monkeypatch):
    _act.pending_backup_confirm_phrase.set({"phrase": "test phrase"})
    result = await handle_pending_backup_confirm_phrase("not yet, still writing it down")
    assert _act.pending_backup_confirm_phrase.active
    assert not _act.pending_backup_oauth.active
    assert result is not None


@pytest.mark.asyncio
async def test_oauth_step_has_app_negated_no_takes_no_branch(monkeypatch):
    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    _act.pending_backup_oauth.set({"step": "has_app", "redirect_uri": "http://127.0.0.1:8888/callback"})
    result = await handle_pending_backup_oauth("No, I don't have one")

    assert _act.pending_backup_oauth.payload["step"] == "client_id"
    assert "Google Cloud credentials page" in result
    assert opened_urls == ["https://console.cloud.google.com/apis/credentials"]


# ─── Important #4: the OAuth flow's timeout is a per-step silence budget ────


def test_oauth_timeout_covers_creating_an_app():
    """The 'no, I don't have an app' branch sends the user off to create a
    Cloud project, enable the Drive API, and set a scope — minutes, not
    seconds."""
    assert _act.pending_backup_oauth.timeout >= 300.0


@pytest.mark.asyncio
async def test_oauth_each_step_refreshes_the_timeout(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: None)

    _act.pending_backup_oauth.set({"step": "has_app", "redirect_uri": "http://x/cb"})
    _act.pending_backup_oauth._ts -= 250.0  # simulate a long pause mid-flow
    aged_before = _act.pending_backup_oauth.age
    assert aged_before > 200.0

    await handle_pending_backup_oauth("yes")

    assert _act.pending_backup_oauth.age < 5.0


# ─── Important #3: raw OAuth error codes never reach TTS ────────────────────


def _oauth_at_auth_code():
    _act.pending_backup_oauth.set({
        "step": "auth_code",
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scopes": "https://www.googleapis.com/auth/drive.appdata",
        "redirect_uri": "http://127.0.0.1:8888/callback",
    })


@pytest.mark.asyncio
async def test_oauth_failure_never_speaks_the_raw_error_code(monkeypatch, caplog):
    from assistant import oauth_helper

    monkeypatch.setattr(
        oauth_helper, "exchange_code_for_tokens",
        lambda *a, **kw: (False, "invalid_grant"),
    )

    _oauth_at_auth_code()
    with caplog.at_level("WARNING"):
        result = await handle_pending_backup_oauth("4/some-code")

    assert "invalid_grant" not in result
    assert "http" not in result
    assert "enable backup" in result.lower()
    assert not _act.pending_backup_oauth.active
    # The real code still reaches the log.
    assert any("invalid_grant" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_oauth_redirect_uri_mismatch_retries_with_guidance(monkeypatch):
    from assistant import oauth_helper

    monkeypatch.setattr(
        oauth_helper, "exchange_code_for_tokens",
        lambda *a, **kw: (False, "redirect_uri_mismatch"),
    )

    _oauth_at_auth_code()
    result = await handle_pending_backup_oauth("4/some-code")

    assert "redirect_uri_mismatch" not in result
    assert "http://127.0.0.1:8888/callback" in result
    # Still on auth_code so the user can paste a fresh one after fixing it.
    assert _act.pending_backup_oauth.active
    assert _act.pending_backup_oauth.payload["step"] == "auth_code"


@pytest.mark.asyncio
async def test_oauth_invalid_client_clears_credentials(monkeypatch):
    from assistant import credentials, oauth_helper

    monkeypatch.setattr(
        oauth_helper, "exchange_code_for_tokens",
        lambda *a, **kw: (False, "invalid_client"),
    )
    deleted = []
    monkeypatch.setattr(credentials, "delete_credential", lambda s: deleted.append(s))

    _oauth_at_auth_code()
    result = await handle_pending_backup_oauth("4/some-code")

    from assistant.io.backup.google_drive import SERVICE_NAME
    assert deleted == [SERVICE_NAME]
    assert "invalid_client" not in result
    assert "enable backup" in result.lower()
    assert not _act.pending_backup_oauth.active


# ─── Critical #3: unlock phrase entry ───────────────────────────────────────


@pytest.mark.asyncio
async def test_unlock_phrase_inactive_returns_none():
    assert await handle_pending_backup_unlock_phrase("some words") is None


@pytest.mark.asyncio
async def test_unlock_phrase_valid_arms_the_session_key():
    phrase = crypto.generate_recovery_phrase()
    _act.pending_backup_unlock_phrase.set({})

    result = await handle_pending_backup_unlock_phrase(phrase)

    assert orchestrator.get_unlocked_key() == crypto.derive_key(phrase)
    assert not _act.pending_backup_unlock_phrase.active
    assert phrase not in result


@pytest.mark.asyncio
async def test_unlock_phrase_tolerates_dictated_punctuation_and_case():
    """Spoken input arrives capitalised and punctuated; it must still derive
    the same key as the phrase that was generated."""
    phrase = crypto.generate_recovery_phrase()
    spoken = phrase.replace(" ", ", ").title() + "."
    _act.pending_backup_unlock_phrase.set({})

    await handle_pending_backup_unlock_phrase(spoken)

    assert orchestrator.get_unlocked_key() == crypto.derive_key(phrase)


@pytest.mark.asyncio
async def test_unlock_phrase_invalid_reprompts_without_clearing():
    _act.pending_backup_unlock_phrase.set({})

    result = await handle_pending_backup_unlock_phrase("banana banana banana")

    assert _act.pending_backup_unlock_phrase.active  # retryable until timeout
    assert not orchestrator.is_unlocked()
    assert "12 words" in result


@pytest.mark.asyncio
async def test_unlock_phrase_cancel_clears():
    _act.pending_backup_unlock_phrase.set({})

    result = await handle_pending_backup_unlock_phrase("cancel")

    assert not _act.pending_backup_unlock_phrase.active
    assert not orchestrator.is_unlocked()
    assert "locked" in result.lower()


# ─── Critical #4: restore phrase entry ──────────────────────────────────────


@pytest.mark.asyncio
async def test_restore_phrase_inactive_returns_none():
    assert await handle_pending_backup_restore_phrase("some words") is None


@pytest.mark.asyncio
async def test_restore_phrase_cancel_does_not_restore(monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator, "run_restore", lambda *a, **kw: calls.append(a))

    _act.pending_backup_restore_phrase.set({})
    result = await handle_pending_backup_restore_phrase("never mind")

    assert calls == []
    assert not _act.pending_backup_restore_phrase.active
    assert "nothing was restored" in result.lower()


@pytest.mark.asyncio
async def test_restore_phrase_invalid_reprompts_without_restoring(monkeypatch):
    calls = []
    monkeypatch.setattr(orchestrator, "run_restore", lambda *a, **kw: calls.append(a))

    _act.pending_backup_restore_phrase.set({})
    result = await handle_pending_backup_restore_phrase("not really a phrase")

    assert calls == []
    assert _act.pending_backup_restore_phrase.active
    assert "12 words" in result


@pytest.mark.asyncio
async def test_restore_phrase_valid_runs_restore(monkeypatch, db_session):
    calls = []
    monkeypatch.setattr(orchestrator, "run_restore",
                        lambda phrase, provider: calls.append((phrase, provider)))
    db_session.set("backup_provider", "google_drive", source="test")

    phrase = crypto.generate_recovery_phrase()
    _act.pending_backup_restore_phrase.set({})
    result = await handle_pending_backup_restore_phrase(phrase)

    assert calls == [(phrase, "google_drive")]
    assert not _act.pending_backup_restore_phrase.active
    assert phrase not in result
    assert "closing" in result.lower()

    from assistant.core import shutdown_signal
    assert shutdown_signal.is_requested()
    shutdown_signal._reset_for_testing()


@pytest.mark.asyncio
async def test_restore_phrase_wrong_phrase_reports_generic_failure(monkeypatch, db_session, caplog):
    """run_restore raises RuntimeError on a wrong phrase / corrupt blob /
    empty provider. The user hears a short line; the log keeps the detail."""
    def _raise(phrase, provider):
        raise RuntimeError(
            "Recovery phrase is incorrect at "
            "https://www.googleapis.com/drive/v3/files C:\\TENKA\\restore.tar 403"
        )

    monkeypatch.setattr(orchestrator, "run_restore", _raise)

    phrase = crypto.generate_recovery_phrase()
    _act.pending_backup_restore_phrase.set({})
    with caplog.at_level("ERROR"):
        result = await handle_pending_backup_restore_phrase(phrase)

    assert len(result) < 120
    assert "https://" not in result
    assert "C:\\" not in result
    assert "403" not in result
    assert not _act.pending_backup_restore_phrase.active
    assert any("403" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_restore_phrase_provider_error_reports_generic_failure(monkeypatch, db_session, caplog):
    from assistant.io.backup.provider import BackupProviderError

    def _raise(phrase, provider):
        raise BackupProviderError("Drive is not connected")

    monkeypatch.setattr(orchestrator, "run_restore", _raise)

    phrase = crypto.generate_recovery_phrase()
    _act.pending_backup_restore_phrase.set({})
    with caplog.at_level("ERROR"):
        result = await handle_pending_backup_restore_phrase(phrase)

    assert "failed" in result.lower()
    assert any("not connected" in r.getMessage() for r in caplog.records)
