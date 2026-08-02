"""Tests for backup onboarding pending-states."""
import pytest

import assistant.actions as _act
from assistant.actions.backup_pending import (
    handle_pending_backup_confirm_phrase,
    handle_pending_backup_oauth,
)


@pytest.fixture(autouse=True)
def _clear_pending():
    yield
    _act.pending_backup_confirm_phrase.clear()
    _act.pending_backup_oauth.clear()


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
