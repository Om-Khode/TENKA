"""
storage/repos/settings.py — Runtime settings repo.

Typed wrapper over the runtime_settings table. Takes a Database instance.
JSON-serializes values for round-trip through bool/int/float/str/list/dict.
"""

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("settings")


class SettingsRepo:
    def __init__(self, db) -> None:
        self._db = db

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.fetchone(
            "SELECT value FROM runtime_settings WHERE key = ?", (key,)
        )
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[SETTINGS] Corrupt value for '{key}': {e} — using default")
            return default

    def set(self, key: str, value: Any, source: str = "user") -> None:
        self._db.execute(
            "INSERT INTO runtime_settings (key, value, updated_at, updated_source) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = excluded.value, "
            "updated_at = excluded.updated_at, "
            "updated_source = excluded.updated_source",
            (key, json.dumps(value), datetime.utcnow().isoformat(), source),
        )
        self._db.commit()

    def delete(self, key: str) -> bool:
        cur = self._db.execute(
            "DELETE FROM runtime_settings WHERE key = ?", (key,)
        )
        self._db.commit()
        return cur.rowcount > 0

    def list_all(self) -> dict:
        rows = self._db.fetchall(
            "SELECT key, value FROM runtime_settings ORDER BY key"
        )
        out: dict = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except (json.JSONDecodeError, TypeError):
                continue
        return out
