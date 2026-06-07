"""Repository for the schedules table ."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.storage.db import Database

logger = logging.getLogger(__name__)


class ScheduleRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        name: str,
        cron_expr: str,
        task_type: str,
        task_goal: str,
        notify_mode: str,
        condition_text: str | None,
        next_fire_at: str,
    ) -> int:
        cursor = self._db.execute(
            """INSERT INTO schedules
               (name, cron_expr, task_type, task_goal, notify_mode,
                condition_text, next_fire_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, cron_expr, task_type, task_goal, notify_mode,
             condition_text, next_fire_at, datetime.now().isoformat()),
        )
        self._db.commit()
        row_id = cursor.lastrowid
        logger.info(f"[SCHEDULES] Created #{row_id}: '{name}' ({cron_expr}, {notify_mode})")
        return row_id

    def get_due(self, now_iso: str) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM schedules WHERE enabled = 1 AND next_fire_at <= ?",
            (now_iso,),
        )
        return [dict(r) for r in rows]

    def list_all(self) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM schedules ORDER BY created_at ASC",
        )
        return [dict(r) for r in rows]

    def list_enabled(self) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM schedules WHERE enabled = 1 ORDER BY created_at ASC",
        )
        return [dict(r) for r in rows]

    def find_by_name(self, name: str) -> dict | None:
        rows = self._db.fetchall(
            "SELECT * FROM schedules ORDER BY created_at ASC",
        )
        name_lower = name.lower()
        for row in rows:
            if name_lower in row["name"].lower():
                return dict(row)
        return None

    def update_after_fire(
        self, schedule_id: int, next_fire_at: str, last_result_hash: str | None
    ) -> None:
        self._db.execute(
            """UPDATE schedules
               SET last_fired_at = ?, next_fire_at = ?, last_result_hash = ?
               WHERE id = ?""",
            (datetime.now().isoformat(), next_fire_at, last_result_hash, schedule_id),
        )
        self._db.commit()

    def toggle(self, schedule_id: int, enabled: bool) -> None:
        self._db.execute(
            "UPDATE schedules SET enabled = ? WHERE id = ?",
            (1 if enabled else 0, schedule_id),
        )
        self._db.commit()

    def delete(self, schedule_id: int) -> None:
        self._db.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        self._db.commit()
        logger.info(f"[SCHEDULES] Deleted #{schedule_id}")
