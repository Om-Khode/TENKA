"""
settings.py — Settings facade over storage/repos/settings.py.

All callers continue to import this module unchanged. The actual logic
lives in SettingsRepo; this module just forwards calls.
"""

from typing import Any

from .storage.db import get_db, init_db
from .storage.repos.settings import SettingsRepo
from . import config

_repo: SettingsRepo | None = None


def _get_repo() -> SettingsRepo:
    global _repo
    if _repo is None:
        db = get_db()
        if db is None:
            raise RuntimeError(
                "settings not initialized — call init_settings_db() first"
            )
        _repo = SettingsRepo(db)
    return _repo


def init_settings_db() -> None:
    """Initialize the shared database if needed, then bind the repo."""
    global _repo
    db = get_db()
    if db is None:
        db_path = config.SANDBOX_DIR / "memory" / "tenka.db"
        db = init_db(db_path)
    _repo = SettingsRepo(db)


def get(key: str, default: Any = None) -> Any:
    if _repo is None:
        return default
    return _repo.get(key, default)


def set(key: str, value: Any, source: str = "user") -> None:
    _get_repo().set(key, value, source)


def delete(key: str) -> bool:
    return _get_repo().delete(key)


def list_all() -> dict:
    if _repo is None:
        return {}
    return _repo.list_all()
