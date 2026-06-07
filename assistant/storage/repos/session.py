"""storage/repos/session.py — Session snapshot persistence."""

import logging
from datetime import datetime

from ..db import Database

logger = logging.getLogger("session")


class SessionRepo:
    """Session lifecycle and snapshot persistence."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def start_session(self, session_id: str) -> None:
        self._db.execute(
            "INSERT INTO session_snapshots (session_id, started_at) VALUES (?, ?)",
            (session_id, datetime.now().isoformat()),
        )
        self._db.commit()

    def end_session(self, session_id: str) -> None:
        self._db.execute(
            "UPDATE session_snapshots SET ended_at = ? WHERE session_id = ?",
            (datetime.now().isoformat(), session_id),
        )
        self._db.commit()

    def increment_turn_count(self, session_id: str) -> None:
        self._db.execute(
            "UPDATE session_snapshots SET turn_count = turn_count + 1 WHERE session_id = ?",
            (session_id,),
        )
        self._db.commit()

    def update_last_intent(self, session_id: str, intent: str) -> None:
        self._db.execute(
            "UPDATE session_snapshots SET last_intent = ? WHERE session_id = ?",
            (intent, session_id),
        )
        self._db.commit()

    def save_summary(
        self, session_id: str, last_intent: str,
        task_summary: str, blocker: str | None,
    ) -> None:
        self._db.execute(
            "UPDATE session_snapshots "
            "SET last_intent = ?, task_summary = ?, blocker = ?, summarized = 1 "
            "WHERE session_id = ?",
            (last_intent, task_summary, blocker, session_id),
        )
        self._db.commit()

    def get_last_snapshot(self) -> dict | None:
        row = self._db.fetchone(
            "SELECT * FROM session_snapshots "
            "WHERE summarized = 1 "
            "ORDER BY started_at DESC LIMIT 1"
        )
        return dict(row) if row else None

    def get_last_interaction_time(self) -> str | None:
        row = self._db.fetchone(
            "SELECT started_at FROM session_snapshots "
            "ORDER BY started_at DESC LIMIT 1"
        )
        return row["started_at"] if row else None

    def get_unsummarized_session(self) -> dict | None:
        row = self._db.fetchone(
            "SELECT * FROM session_snapshots "
            "WHERE summarized = 0 AND turn_count >= 2 "
            "ORDER BY started_at DESC LIMIT 1"
        )
        return dict(row) if row else None
