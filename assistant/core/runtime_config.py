"""Runtime setting resolution.

Precedence: DB (if available) -> env var (uppercase key) -> hardcoded default.
DB access is lazy — returns None until main.py calls init_db().
"""

import logging
import os
from typing import Any

_logger = logging.getLogger("runtime_config")

REGISTRY: dict[str, dict] = {}


def _get_db_value(key: str) -> Any | None:
    """Read a setting value from the DB, or None if DB unavailable."""
    try:
        from assistant.storage.db import get_db
        db = get_db()
        if db is None:
            return None
        from assistant.storage.repos.settings import SettingsRepo
        repo = SettingsRepo(db)
        return repo.get(key)
    except Exception as e:
        _logger.debug(f"DB setting read failed for '{key}': {e}")
        return None


def setting(
    key: str,
    default: Any,
    cast=str,
    description: str = "",
    needs_restart: bool = False,
) -> Any:
    """Register and resolve a runtime-configurable setting.

    Precedence: DB value -> env var (uppercase key) -> hardcoded default.
    """
    REGISTRY[key] = {
        "default": default,
        "cast": cast,
        "description": description,
        "needs_restart": needs_restart,
    }

    db_value = _get_db_value(key)
    if db_value is not None:
        try:
            if cast is bool:
                return bool(db_value)
            return cast(db_value)
        except (ValueError, TypeError):
            pass

    env_value = os.getenv(key.upper())
    if env_value is not None:
        try:
            if cast is bool:
                return env_value.strip().lower() in ("true", "1", "yes", "on")
            return cast(env_value)
        except (ValueError, TypeError):
            pass

    return default


def reload_all() -> dict[str, Any]:
    """Re-resolve every registered setting. Returns {key: new_value}.

    Called from main.py after init_db(), and after /set or /reset commands.
    Callers must assign the returned values to their module globals.
    """
    results = {}
    for key, meta in REGISTRY.items():
        results[key] = setting(
            key, meta["default"], cast=meta["cast"],
            description=meta["description"],
            needs_restart=meta.get("needs_restart", False),
        )
    return results


def get_user_region() -> dict:
    """Return cached user region info. Call detect_region() first at startup."""
    from .geolocation import get_cached_region
    return get_cached_region()
