"""Handler for the manage_backup intent.

Action is derived from the goal text via keyword matching (regex-first,
per ARCHITECTURE.md §2.2) rather than a pre-classified LLM field — same
convention as manage_schedule/manage_monitor, whose intent catalogue
entries also carry only a free-text "goal" param.
"""
from __future__ import annotations

import logging

from .registry import tool_registry

logger = logging.getLogger(__name__)

_RESTORE_WORDS = ("restore",)
_STATUS_WORDS = ("status", "when was", "last backup")
_DISABLE_WORDS = ("disable", "turn off", "stop backing up", "stop backup")
_ENABLE_WORDS = ("enable", "set up", "setup", "turn on", "connect")


def _classify_action(goal: str) -> str:
    goal_low = goal.lower()
    if any(w in goal_low for w in _RESTORE_WORDS):
        return "restore"
    if any(w in goal_low for w in _STATUS_WORDS):
        return "status"
    if any(w in goal_low for w in _DISABLE_WORDS):
        return "disable"
    if any(w in goal_low for w in _ENABLE_WORDS):
        return "enable"
    return "backup_now"


def _get_settings_repo():
    from ..storage.db import get_db
    from ..storage.repos.settings import SettingsRepo
    db = get_db()
    if db is None:
        raise RuntimeError("DB not initialized")
    return SettingsRepo(db)


@tool_registry.decorator("manage_backup")
async def handle_manage_backup(params: dict, llm_response: str, bridge=None) -> str:
    goal = params.get("goal", "")
    action = _classify_action(goal)

    if action == "status":
        return _status()
    if action == "enable":
        return await _enable()
    if action == "disable":
        return _disable()
    if action == "backup_now":
        return await _backup_now()
    if action == "restore":
        return await _restore()
    return "I'm not sure what to do with that backup command."


def _status() -> str:
    settings = _get_settings_repo()
    enabled = settings.get("backup_enabled", False)
    if not enabled:
        return "Cloud backup isn't set up yet. Say 'enable backup' to get started."

    last_at = settings.get("backup_last_backup_at")
    last_status = settings.get("backup_last_backup_status", "never")
    if not last_at:
        return "Backup is enabled but hasn't run yet."
    return f"Last backup was {last_at} — status: {last_status}."


async def _enable() -> str:
    import assistant.actions as _act
    from ..io.backup import crypto

    if _act.pending_backup_confirm_phrase.active or _act.pending_backup_oauth.active:
        return "Already in the middle of setting up backup — finish that first."

    phrase = crypto.generate_recovery_phrase()
    from ..io.backup import orchestrator
    orchestrator.set_unlocked_key(crypto.derive_key(phrase))

    _act.pending_backup_confirm_phrase.set({"phrase": phrase})
    return (
        f"Here's your recovery phrase — write it down somewhere safe, I won't "
        f"repeat it and I'm not saving it anywhere: {phrase}. "
        f"If you lose this, nobody, including me, can recover your backup. "
        f"Say 'saved it' once you've written it down."
    )


def _disable() -> str:
    settings = _get_settings_repo()
    settings.set("backup_enabled", False, source="user")
    from ..io.backup import orchestrator
    orchestrator.set_unlocked_key(None)
    return "Cloud backup disabled. Your existing backups are untouched."


async def _backup_now() -> str:
    from ..io.backup import orchestrator
    from ..io.backup.provider import BackupProviderError

    if not orchestrator.is_unlocked():
        return (
            "I need your recovery phrase to back up right now — "
            "say 'enable backup' if you haven't set it up, "
            "or restart hasn't unlocked it yet this session."
        )

    settings = _get_settings_repo()
    provider_name = settings.get("backup_provider", "google_drive")
    try:
        orchestrator.run_backup(provider_name)
    except (BackupProviderError, RuntimeError) as e:
        return f"Backup failed: {str(e)[:150]}"
    return "Backup complete."


async def _restore() -> str:
    return (
        "Restore needs your recovery phrase and isn't something to do by "
        "accident — run this from the setup wizard on a fresh install, "
        "or say 'enable backup' first if you're already set up and just "
        "want to check for a newer version."
    )
