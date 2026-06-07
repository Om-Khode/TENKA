"""
shortcuts.py — Shortcut storage facade over storage/repos/shortcut.py.
"""

from typing import Optional

from .storage.db import get_db, init_db
from . import config

_repo = None


def _get_repo():
    global _repo
    if _repo is None:
        from .storage.repos.shortcut import ShortcutRepo
        db = get_db()
        if db is None:
            raise RuntimeError(
                "shortcuts not initialized — call init_shortcut_db() first"
            )
        _repo = ShortcutRepo(
            db,
            assistant_name_lower=config.ASSISTANT_NAME_LOWER,
            intents=config.INTENTS,
        )
    return _repo


def init_shortcut_db() -> None:
    """Initialize the shared database if needed, then bind the repo."""
    global _repo
    db = get_db()
    if db is None:
        db_path = config.SANDBOX_DIR / "memory" / "tenka.db"
        db = init_db(db_path)
    from .storage.repos.shortcut import ShortcutRepo
    _repo = ShortcutRepo(
        db,
        assistant_name_lower=config.ASSISTANT_NAME_LOWER,
        intents=config.INTENTS,
    )


def match_shortcut(transcription: str) -> Optional[dict]:
    return _get_repo().match_shortcut(transcription)


def create_shortcut(trigger: str, intent: str, params: Optional[dict] = None,
                    description: str = "") -> bool:
    return _get_repo().create_shortcut(trigger, intent, params, description)


def delete_shortcut(trigger: str) -> bool:
    return _get_repo().delete_shortcut(trigger)


def get_shortcut(trigger: str) -> Optional[dict]:
    return _get_repo().get_shortcut(trigger)


def list_shortcuts() -> list[dict]:
    return _get_repo().list_shortcuts()


def reset_shortcuts() -> None:
    _get_repo().reset_shortcuts()
