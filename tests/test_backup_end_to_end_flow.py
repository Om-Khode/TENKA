"""End-to-end walk of the cloud-backup feature through main.py's dispatcher.

Every task in this feature was tested in isolation and still shipped an
unreachable flow: the pending handlers were never wired into main.py, so
"saved it" fell through to small talk. This test drives the whole chain the
way a real turn does — through main._PENDING_HANDLERS — so a break in any
seam fails here:

    enable -> phrase shown on screen only -> "saved it" -> OAuth walkthrough
    -> connected -> restart (key gone) -> "back up now" refuses -> unlock
    -> "back up now" succeeds -> restore.
"""
import tempfile
from pathlib import Path

import pytest

import assistant.actions as _act
import assistant.main as main_module
from assistant.actions.backup import handle_manage_backup
from assistant.io.backup import backup_provider_registry, orchestrator


class _FakeBridge:
    def __init__(self):
        self.unity_connected = True
        self.shown: list[str] = []

    async def send_command(self, action, _log_payload=True, **kwargs):
        if action == "show_thought":
            self.shown.append(kwargs.get("text", ""))


class _FakeProvider:
    name = "google_drive"

    def __init__(self):
        self.uploads: dict[str, bytes] = {}

    def is_connected(self): return True
    def upload(self, blob, label): self.uploads[label] = blob
    def list_versions(self): return sorted(self.uploads.keys(), reverse=True)
    def download(self, label): return self.uploads[label]
    def delete(self, label): del self.uploads[label]


async def _dispatch(text: str) -> str | None:
    """Mirror main.py's pending loop: first non-None response owns the turn."""
    for handler, _label, _mem_intent, needs_bridge in main_module._PENDING_HANDLERS:
        resp = await handler(text, None) if needs_bridge else await handler(text)
        if resp is not None:
            return resp
    return None


@pytest.fixture
def env(tmp_path, monkeypatch):
    from assistant import config, credentials, oauth_helper
    from assistant.storage.db import _reset_for_testing
    import assistant.storage.db as db_module

    _reset_for_testing()
    sandbox = tmp_path / "TENKA"
    (sandbox / "memory").mkdir(parents=True)
    (sandbox / "Notes").mkdir()
    (sandbox / "Notes" / "todo.md").write_text("- buy milk")
    db = db_module.init_db(sandbox / "memory" / "tenka.db")
    monkeypatch.setattr(config, "SANDBOX_DIR", sandbox)

    # Import before swapping the registry: google_drive.py self-registers at
    # import time, and the onboarding flow imports it mid-test.
    import assistant.io.backup.google_drive  # noqa: F401

    provider = _FakeProvider()
    monkeypatch.setattr(backup_provider_registry, "_entries", {"google_drive": provider})

    # Onboarding's OAuth leg, stubbed at the same seam the unit tests use.
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr(oauth_helper, "get_setup_url", lambda *a, **kw: "https://auth.example/x")
    monkeypatch.setattr(oauth_helper, "exchange_code_for_tokens", lambda *a, **kw: (True, ""))
    monkeypatch.setattr(credentials, "set_credential", lambda *a, **kw: None)

    yield sandbox, provider

    db.close()
    _reset_for_testing()
    for state in (_act.pending_backup_confirm_phrase, _act.pending_backup_oauth,
                  _act.pending_backup_unlock_phrase, _act.pending_backup_restore_phrase):
        state.clear()
    orchestrator.set_unlocked_key(None)


@pytest.mark.asyncio
async def test_full_backup_lifecycle_is_reachable(env):
    sandbox, provider = env
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo

    bridge = _FakeBridge()

    # ── enable ───────────────────────────────────────────────────────────
    said = await handle_manage_backup({"goal": "enable backup"}, "", bridge)
    assert bridge.shown, "recovery phrase never reached the screen"
    phrase = bridge.shown[0].splitlines()[-1].strip()
    assert len(phrase.split()) == 12
    assert phrase not in said
    assert not orchestrator.is_unlocked()   # not until the user confirms

    # ── "saved it" — the turn that used to fall through to small talk ────
    resp = await _dispatch("saved it")
    assert resp is not None, "'saved it' was not handled by any pending handler"
    assert "google drive" in resp.lower()
    assert orchestrator.is_unlocked()

    # ── OAuth walkthrough ────────────────────────────────────────────────
    assert await _dispatch("yes I have an app") is not None
    assert await _dispatch("client-id-1234567890") is not None
    assert await _dispatch("client-secret-1234567890") is not None
    done = await _dispatch("4/auth-code-here")
    assert "connected" in done.lower()
    assert SettingsRepo(get_db()).get("backup_enabled") is True

    # ── restart: the key is process-local, so it is simply gone ──────────
    orchestrator.set_unlocked_key(None)

    refused = await handle_manage_backup({"goal": "back up now"}, "", bridge)
    assert "unlock backup" in refused.lower()
    assert provider.uploads == {}

    # ── unlock with the phrase the user wrote down ───────────────────────
    assert await handle_manage_backup({"goal": "unlock backup"}, "", bridge) is not None
    unlocked = await _dispatch(phrase)
    assert "unlocked" in unlocked.lower()
    assert orchestrator.is_unlocked()

    # ── back up now, for real ────────────────────────────────────────────
    assert "complete" in (await handle_manage_backup({"goal": "back up now"}, "", bridge)).lower()
    assert len(provider.uploads) == 1

    # ── restore, into a clean sandbox ────────────────────────────────────
    from assistant import config
    restore_target = Path(tempfile.mkdtemp()) / "restored"
    config_sandbox = config.SANDBOX_DIR
    assert config_sandbox == sandbox

    warned = await handle_manage_backup({"goal": "restore my backup"}, "", bridge)
    assert "overwrit" in warned.lower()

    import assistant.config as cfg
    cfg.SANDBOX_DIR = restore_target
    try:
        restored = await _dispatch(phrase)
    finally:
        cfg.SANDBOX_DIR = config_sandbox

    assert "restart" in restored.lower()
    assert (restore_target / "memory" / "tenka.db").exists()
    assert (restore_target / "Notes" / "todo.md").read_text() == "- buy milk"
