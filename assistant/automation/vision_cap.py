"""vision_cap.py — daily Gemini-vision call counter for manifest-based tier-2 healer.

Persists in tenka.db so the cap survives restarts within the same day.
Counter rolls over at local midnight (caller invokes reset_for_new_day()
from the existing SC-1 scheduler pump or main.py daily tick).
"""
from __future__ import annotations

import sqlite3
from datetime import date

DEFAULT_DAILY_CAP = 100   # Per spec §3.3


class VisionCapTracker:
    def __init__(self, db: sqlite3.Connection, cap: int = DEFAULT_DAILY_CAP) -> None:
        self._db = db
        self._cap = cap

    def _today_key(self) -> str:
        return date.today().isoformat()

    def calls_today(self) -> int:
        cur = self._db.execute(
            "SELECT count FROM vision_calls WHERE day = ?",
            (self._today_key(),),
        )
        row = cur.fetchone()
        return row[0] if row else 0

    def try_increment(self) -> bool:
        """Atomic test-and-increment. Returns True if call is allowed, False if cap reached."""
        today = self._today_key()
        self._db.execute(
            "INSERT INTO vision_calls (day, count) VALUES (?, 0) "
            "ON CONFLICT(day) DO NOTHING",
            (today,),
        )
        cur = self._db.execute(
            "UPDATE vision_calls SET count = count + 1 "
            "WHERE day = ? AND count < ?",
            (today, self._cap),
        )
        self._db.commit()
        return cur.rowcount > 0

    def reset_for_new_day(self) -> None:
        """Purge all rows for days prior to today.

        Called from the midnight scheduler tick. Uses `< today` so all
        accumulated stale day-rows are cleaned in one sweep — today's
        own row (if any) is preserved, but at midnight it does not exist
        yet so this is also a fresh slate for the new day.
        """
        self._db.execute(
            "DELETE FROM vision_calls WHERE day < ?",
            (self._today_key(),),
        )
        self._db.commit()
