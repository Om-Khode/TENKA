"""
storage/repos/shortcut.py — Voice shortcut repo.

Maps trigger phrases to intent + params. Supports exact match and
filler-tolerant matching.
"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("shortcuts")


class ShortcutRepo:
    def __init__(self, db, *, assistant_name_lower: str, intents: list[str]) -> None:
        self._db = db
        self._assistant_name_lower = assistant_name_lower
        self._intents = intents

    @property
    def _reserved(self) -> set[str]:
        return {
            self._assistant_name_lower,
            "hey", "hi", "hello", "yes", "no", "okay", "ok",
            "stop", "cancel", "help", "please", "thanks", "thank you",
        }

    def match_shortcut(self, transcription: str) -> Optional[dict]:
        if not transcription or len(transcription.strip()) < 2:
            return None

        cleaned = transcription.strip().lower()

        rows = self._db.fetchall(
            "SELECT trigger, intent, params_json, description FROM user_shortcuts"
        )

        for row in rows:
            trigger = row["trigger"].lower()

            if cleaned == trigger:
                return self._build_match(row)

            fillers = ["please", "now", self._assistant_name_lower, "go", "do it", "run"]
            for filler in fillers:
                if cleaned == f"{trigger} {filler}" or cleaned == f"{filler} {trigger}":
                    return self._build_match(row)

        return None

    def _build_match(self, row) -> dict:
        now = datetime.now().isoformat()

        self._db.execute(
            "UPDATE user_shortcuts SET times_used = times_used + 1, updated_at = ? "
            "WHERE trigger = ?",
            (now, row["trigger"]),
        )
        self._db.commit()

        try:
            params = json.loads(row["params_json"])
        except (json.JSONDecodeError, TypeError):
            params = {}

        count = self._get_usage_count(row["trigger"])
        logger.info(
            f"[SHORTCUTS] Matched '{row['trigger']}' → {row['intent']} "
            f"(used {count} times)"
        )

        return {
            "intent": row["intent"],
            "params": params,
            "trigger": row["trigger"],
            "description": row["description"],
        }

    def _get_usage_count(self, trigger: str) -> int:
        row = self._db.fetchone(
            "SELECT times_used FROM user_shortcuts WHERE trigger = ?",
            (trigger,),
        )
        return row["times_used"] if row else 0

    def create_shortcut(
        self,
        trigger: str,
        intent: str,
        params: Optional[dict] = None,
        description: str = "",
    ) -> bool:
        now = datetime.now().isoformat()
        params_json = json.dumps(params or {})
        trigger_clean = trigger.strip().lower()

        if not trigger_clean or len(trigger_clean) < 2 or trigger_clean in self._reserved:
            logger.warning(f"[SHORTCUTS] Trigger '{trigger_clean}' rejected (too short or reserved)")
            return False

        if intent not in self._intents and intent not in ("shutdown",):
            logger.warning(f"[SHORTCUTS] Unknown intent '{intent}' — rejected")
            return False

        self._db.execute(
            "INSERT INTO user_shortcuts "
            "(trigger, intent, params_json, description, times_used, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 0, ?, ?) "
            "ON CONFLICT(trigger) DO UPDATE SET "
            "intent = excluded.intent, "
            "params_json = excluded.params_json, "
            "description = excluded.description, "
            "updated_at = excluded.updated_at",
            (trigger_clean, intent, params_json, description, now, now),
        )
        self._db.commit()

        logger.info(f"[SHORTCUTS] Created: '{trigger_clean}' → {intent} ({description})")
        return True

    def delete_shortcut(self, trigger: str) -> bool:
        trigger_clean = trigger.strip().lower()
        cursor = self._db.execute(
            "DELETE FROM user_shortcuts WHERE trigger = ?",
            (trigger_clean,),
        )
        self._db.commit()

        if cursor.rowcount > 0:
            logger.info(f"[SHORTCUTS] Deleted: '{trigger_clean}'")
            return True
        return False

    def get_shortcut(self, trigger: str) -> Optional[dict]:
        row = self._db.fetchone(
            "SELECT * FROM user_shortcuts WHERE trigger = ?",
            (trigger.strip().lower(),),
        )
        if not row:
            return None

        result = dict(row)
        try:
            result["params"] = json.loads(result.pop("params_json"))
        except (json.JSONDecodeError, TypeError):
            result["params"] = {}
        return result

    def list_shortcuts(self) -> list[dict]:
        rows = self._db.fetchall(
            "SELECT * FROM user_shortcuts ORDER BY updated_at DESC"
        )
        results = []
        for row in rows:
            r = dict(row)
            try:
                r["params"] = json.loads(r.pop("params_json"))
            except (json.JSONDecodeError, TypeError):
                r["params"] = {}
            results.append(r)
        return results

    def reset_shortcuts(self) -> None:
        self._db.execute("DELETE FROM user_shortcuts")
        self._db.commit()
        logger.info("[SHORTCUTS] All shortcuts cleared (reset)")
