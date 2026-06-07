"""Repository for the automation_cache table."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.storage.db import Database

logger = logging.getLogger(__name__)


class AutomationCacheRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    def save(
        self,
        backend: str,
        app_name: str,
        goal_slug: str,
        goal_text: str,
        steps: list[dict],
    ) -> None:
        now = datetime.now().isoformat()
        self._db.execute(
            """INSERT INTO automation_cache
               (backend, app_name, goal_slug, goal_text, steps_json,
                hit_count, created_at, last_hit_at, version)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?, 1)
               ON CONFLICT(backend, app_name, goal_slug) DO UPDATE SET
                   goal_text = excluded.goal_text,
                   steps_json = excluded.steps_json,
                   hit_count = 0,
                   last_hit_at = excluded.last_hit_at,
                   version = excluded.version""",
            (backend, app_name, goal_slug, goal_text,
             json.dumps(steps), now, now),
        )
        self._db.commit()
        logger.info(f"[AC] Saved cache: {backend}/{app_name}/{goal_slug}")

    def get(
        self, backend: str, app_name: str, goal_slug: str,
    ) -> dict | None:
        row = self._db.fetchone(
            """SELECT * FROM automation_cache
               WHERE backend = ? AND app_name = ? AND goal_slug = ?""",
            (backend, app_name, goal_slug),
        )
        return dict(row) if row else None

    def record_hit(
        self, backend: str, app_name: str, goal_slug: str,
    ) -> None:
        self._db.execute(
            """UPDATE automation_cache
               SET hit_count = hit_count + 1,
                   last_hit_at = ?
               WHERE backend = ? AND app_name = ? AND goal_slug = ?""",
            (datetime.now().isoformat(), backend, app_name, goal_slug),
        )
        self._db.commit()

    def delete(
        self, backend: str, app_name: str, goal_slug: str,
    ) -> bool:
        cursor = self._db.execute(
            """DELETE FROM automation_cache
               WHERE backend = ? AND app_name = ? AND goal_slug = ?""",
            (backend, app_name, goal_slug),
        )
        self._db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"[AC] Deleted stale cache: {backend}/{app_name}/{goal_slug}")
        return deleted

    def cleanup_expired(self, max_age_days: int = 30) -> int:
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        cursor = self._db.execute(
            "DELETE FROM automation_cache WHERE last_hit_at < ?",
            (cutoff,),
        )
        self._db.commit()
        removed = cursor.rowcount
        if removed:
            logger.info(f"[AC] Cleaned up {removed} expired cache entries")
        return removed

    def list_all(self) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM automation_cache ORDER BY last_hit_at DESC",
        )
        return [dict(r) for r in rows]

    # ─── Promotion (manifest) ────────────────────────────────────────────

    def find_unpromoted(self) -> list[dict]:
        """Return automation-cache entries where promoted_intent_id IS NULL.

        Used by the manifest-based promoter to discover successful, repeatedly-hit
        automation procedures that haven't yet been lifted into a typed
        app manifest. Each row has the minimal fields needed to synthesize
        a manifest intent.
        """
        rows = self._db.fetchall(
            "SELECT backend, app_name, goal_slug, goal_text, steps_json, "
            "created_at FROM automation_cache "
            "WHERE promoted_intent_id IS NULL"
        )
        return [dict(r) for r in rows]

    def mark_promoted(
        self,
        backend: str,
        app_name: str,
        goal_slug: str,
        intent_ref: str,
    ) -> None:
        """Claim an automation-cache entry as promoted into a manifest.

        `intent_ref` is the manifest reference string in the form
        "<app_id>:<intent_id>" (e.g. "test_app.desktop:play"). Once set,
        subsequent calls to `find_unpromoted()` will exclude this row.

        Logs a warning if no row matches the (backend, app_name, goal_slug)
        triple — silent no-ops are a debugging hazard for the promoter.
        """
        cursor = self._db.execute(
            "UPDATE automation_cache SET promoted_intent_id = ? "
            "WHERE backend = ? AND app_name = ? AND goal_slug = ?",
            (intent_ref, backend, app_name, goal_slug),
        )
        self._db.commit()
        if cursor.rowcount > 0:
            logger.info(
                f"[AC] Promoted {backend}/{app_name}/{goal_slug} -> {intent_ref}"
            )
        else:
            logger.warning(
                f"[AC] mark_promoted miss: no row for "
                f"{backend}/{app_name}/{goal_slug} -> {intent_ref}"
            )
