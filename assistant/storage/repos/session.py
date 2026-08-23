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
        """The most recent summarised session.

        `ORDER BY started_at DESC, id DESC` -- the `id` tie-break is not
        decoration. `started_at` is `datetime.now().isoformat()`, and the system
        clock's granularity on this platform is around 15ms, so two sessions
        started in quick succession get **identical** timestamps and the
        ordering between them is whatever SQLite happens to do. That made
        `test_get_last_snapshot_returns_most_recent_summarized` fail roughly one
        run in three, on `main`, for a reason that had nothing to do with
        sessions.
        A flaky test is worse than a red one: it teaches everyone to re-run
        rather than to look. Fixed in the query rather than by spacing out the
        timestamps in the test, because the ambiguity is real -- `id` is
        monotonic, so on a tie the later insert is genuinely the later session.
        """
        row = self._db.fetchone(
            "SELECT * FROM session_snapshots "
            "WHERE summarized = 1 "
            "ORDER BY started_at DESC, id DESC LIMIT 1"
        )
        return dict(row) if row else None

    def get_last_interaction_time(self) -> str | None:
        """Same tie-break, same reason -- see `get_last_snapshot`."""
        row = self._db.fetchone(
            "SELECT started_at FROM session_snapshots "
            "ORDER BY started_at DESC, id DESC LIMIT 1"
        )
        return row["started_at"] if row else None

    def get_unsummarized_session(self) -> dict | None:
        row = self._db.fetchone(
            "SELECT * FROM session_snapshots "
            "WHERE summarized = 0 AND turn_count >= 2 "
            "ORDER BY started_at DESC LIMIT 1"
        )
        return dict(row) if row else None
