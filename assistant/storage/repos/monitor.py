"""Repository for the event_monitors table ."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.storage.db import Database

logger = logging.getLogger(__name__)


class MonitorRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        name: str,
        event_type: str,
        source_filter: str | None,
        condition_mode: str,
        condition_expr: str | None,
        condition_prompt: str | None,
        action_type: str,
        action_payload: str,
        cooldown_secs: int,
        user_goal: str,
    ) -> int:
        cursor = self._db.execute(
            """INSERT INTO event_monitors
               (name, event_type, source_filter, condition_mode,
                condition_expr, condition_prompt, action_type,
                action_payload, cooldown_secs, user_goal, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, event_type, source_filter, condition_mode,
             condition_expr, condition_prompt, action_type,
             action_payload, cooldown_secs, user_goal,
             datetime.now().isoformat()),
        )
        self._db.commit()
        row_id = cursor.lastrowid
        logger.info(f"[MONITORS] Created #{row_id}: '{name}' ({event_type})")
        return row_id

    def get_active(self) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM event_monitors WHERE enabled = 1 ORDER BY created_at ASC",
        )
        return [dict(r) for r in rows]

    def get_all(self) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM event_monitors ORDER BY created_at ASC",
        )
        return [dict(r) for r in rows]

    def get_by_id(self, monitor_id: int) -> dict | None:
        rows = self._db.fetchall(
            "SELECT * FROM event_monitors WHERE id = ?", (monitor_id,),
        )
        return dict(rows[0]) if rows else None

    def toggle(self, monitor_id: int, enabled: bool) -> bool:
        self._db.execute(
            "UPDATE event_monitors SET enabled = ? WHERE id = ?",
            (1 if enabled else 0, monitor_id),
        )
        self._db.commit()
        return True

    def delete(self, monitor_id: int) -> bool:
        self._db.execute(
            "DELETE FROM event_monitors WHERE id = ?", (monitor_id,),
        )
        self._db.commit()
        logger.info(f"[MONITORS] Deleted #{monitor_id}")
        return True

    def record_fire(self, monitor_id: int, now_iso: str) -> None:
        self._db.execute(
            """UPDATE event_monitors
               SET last_fired_at = ?, fire_count = fire_count + 1
               WHERE id = ?""",
            (now_iso, monitor_id),
        )
        self._db.commit()
