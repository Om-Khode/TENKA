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
_STATUS_WORDS = (
    "status", "when was", "last backup", "is it working", "working fine",
    "working or not", "check if", "checking if", "is backup", "did it back up",
)
_DISABLE_WORDS = ("disable", "turn off", "stop backing up", "stop backup")
_UNLOCK_WORDS = ("unlock",)
_ENABLE_WORDS = ("enable", "set up", "setup", "turn on", "connect")
_BACKUP_NOW_WORDS = (
    "back up now", "backup now", "do a backup", "make a backup",
    "run a backup", "start a backup", "back it up",
)


def _classify_action(goal: str) -> str:
    goal_low = goal.lower()
    if any(w in goal_low for w in _RESTORE_WORDS):
        return "restore"
    if any(w in goal_low for w in _STATUS_WORDS):
        return "status"
    if any(w in goal_low for w in _DISABLE_WORDS):
        return "disable"
    # Before "enable": "unlock" shares no substring with the enable/connect
    # words, but it must not fall through to backup_now either — unlocking
    # is the non-destructive counterpart of enabling after a restart.
    if any(w in goal_low for w in _UNLOCK_WORDS):
        return "unlock"
    if any(w in goal_low for w in _ENABLE_WORDS):
        return "enable"
    if any(w in goal_low for w in _BACKUP_NOW_WORDS):
        return "backup_now"
    # An utterance that matched none of the above is ambiguous — reading it
    # as a status check (safe, read-only) instead of a command to actually
    # run a backup means unrecognized backup-related speech never silently
    # triggers a side-effecting action.
    return "status"


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
        return await _enable(goal, bridge)
    if action == "unlock":
        return _unlock()
    if action == "disable":
        return _disable()
    if action == "backup_now":
        return await _backup_now()
    if action == "restore":
        return _restore()
    return "I'm not sure what to do with that backup command."


def _status() -> str:
    settings = _get_settings_repo()
    enabled = settings.get("backup_enabled", False)
    if not enabled:
        return "Cloud backup isn't set up yet. Say 'enable backup' to get started."

    # Check status before last_at: a failed attempt (orchestrator.run_backup's
    # except branch) sets backup_last_backup_status="failed" but never touches
    # backup_last_backup_at — that only happens on the success path. Gating
    # solely on last_at would silently swallow the failure as "hasn't run yet".
    last_status = settings.get("backup_last_backup_status")
    if last_status == "failed":
        return "The last backup attempt failed. Say 'back up now' to try again."

    last_at = settings.get("backup_last_backup_at")
    if not last_at:
        return "Backup is enabled but hasn't run yet."
    from ..core.datetime_utils import humanize_relative
    return f"Last backup was {humanize_relative(last_at)} — status: {last_status}."


_REENABLE_CONFIRM_WORDS = ("replace it", "replace them", "start over", "yes, replace")


async def _show_phrase_privately(phrase: str, bridge=None) -> None:
    """Put the recovery phrase on screen without it entering any record.

    Everything a handler *returns* is spoken by TTS, written to debug.log
    (main.py's `Response: "…"`), saved into the conversations table, and
    replayed verbatim into later LLM prompts — so the phrase can never be
    part of a return value, and must never reach a logger.* call either.

    Two transient channels are used instead:
      - the Unity overlay's thought bubble ("show_thought"), which is a
        separate panel from the subtitle line — the subtitle would be
        overwritten a second later by TTS speaking this turn's response.
        _log_payload=False keeps unity_bridge's own debug trace from
        writing the payload out.
      - the console, which is always written to: it is the only channel
        that exists in terminal-only mode / the dev harness, and print()
        never passes through logging.
    """
    if bridge is not None and getattr(bridge, "unity_connected", False):
        try:
            await bridge.send_command(
                "show_thought", state="done", text=f"Recovery phrase:\n{phrase}",
                _log_payload=False,
            )
        except Exception:
            # Never let an overlay failure surface the phrase in an
            # exception message or abort the flow — the console copy below
            # is the durable one anyway.
            logger.warning("[BACKUP] Could not display recovery phrase on the overlay")

    print(f"\n=== TENKA backup recovery phrase (shown once, never stored) ===\n{phrase}\n")


async def _enable(goal: str = "", bridge=None) -> str:
    import assistant.actions as _act
    from ..io.backup import crypto

    if _act.pending_backup_confirm_phrase.active or _act.pending_backup_oauth.active:
        return "Already in the middle of setting up backup — finish that first."

    settings = _get_settings_repo()
    already_enabled = settings.get("backup_enabled", False)
    confirmed_replace = any(w in goal.lower() for w in _REENABLE_CONFIRM_WORDS)
    if already_enabled and not confirmed_replace:
        return (
            "Cloud backup is already set up. If you just need to unlock it "
            "after a restart, say 'unlock backup' instead. Enabling it again "
            "generates a brand-new recovery phrase and key — any backups made "
            "under the old one won't be recoverable without it. If you're sure, "
            "say 'enable backup, replace it' to confirm."
        )

    phrase = crypto.generate_recovery_phrase()

    # The key is NOT cached here: an abandoned or timed-out flow would
    # otherwise leave the session unlocked with a phrase nobody wrote
    # down, and the next backup would upload under that orphan key.
    # handle_pending_backup_confirm_phrase caches it on confirmation.
    _act.pending_backup_confirm_phrase.set({"phrase": phrase})
    await _show_phrase_privately(phrase, bridge)

    return (
        "Your recovery phrase is on screen now. Write it down somewhere safe — "
        "I won't repeat it, and it isn't saved anywhere. If you lose it, nobody, "
        "including me, can recover your backup. Say 'saved it' once it's written down."
    )


def _unlock() -> str:
    """Re-arm this session's backup key from a phrase the user already has.

    The key lives in process memory only, so every restart starts locked.
    This is the non-destructive counterpart of 'enable backup' — it never
    generates a new phrase and never orphans existing backups.
    """
    import assistant.actions as _act
    from ..io.backup import orchestrator

    if orchestrator.is_unlocked():
        return "Backup is already unlocked for this session."

    _act.pending_backup_unlock_phrase.set({})
    return (
        "Paste or say your 12-word recovery phrase and I'll unlock backup "
        "for this session. Say 'cancel' to stop."
    )


def _restore() -> str:
    """Start the restore flow — phrase entry happens in the pending handler."""
    import assistant.actions as _act
    from ..io.backup import backup_provider_registry

    settings = _get_settings_repo()
    provider_name = settings.get("backup_provider", "google_drive")
    provider = backup_provider_registry.require(provider_name)
    if not provider.is_connected():
        return "Google Drive isn't connected — say 'enable backup' to set it up first."

    _act.pending_backup_restore_phrase.set({})
    return (
        "Restoring overwrites what's on this machine with the latest backup, "
        "and I'll close myself when it's done — start me again after. "
        "Give me your 12-word recovery phrase to go ahead, or say 'cancel'."
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
            "I need your recovery phrase first — say 'unlock backup' if you "
            "already have one, or 'enable backup' to set it up."
        )

    settings = _get_settings_repo()
    provider_name = settings.get("backup_provider", "google_drive")
    try:
        orchestrator.run_backup(provider_name)
    except (BackupProviderError, RuntimeError) as e:
        logger.error(f"[BACKUP] backup_now failed: {e}")
        return "Backup failed — check the logs for details."
    return "Backup complete."
