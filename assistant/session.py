"""session.py — Session continuity facade.

Thin delegation layer over storage.repos.session.SessionRepo.
Manages session lifecycle, resume context, and crash recovery.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("session")

_repo: Optional["SessionRepo"] = None
_current_session_id: Optional[str] = None


def init_session_db() -> None:
    global _repo
    from .storage.db import get_db
    from .storage.repos.session import SessionRepo

    db = get_db()
    if db is None:
        raise RuntimeError(
            "session.init_session_db() called before storage.db.init_db(). "
            "Call init_db() first."
        )
    _repo = SessionRepo(db)
    logger.info("[SESSION] Initialized")


def _get_repo() -> "SessionRepo":
    if _repo is None:
        init_session_db()
    assert _repo is not None
    return _repo


def start_session() -> str:
    global _current_session_id
    _current_session_id = str(uuid.uuid4())
    _get_repo().start_session(_current_session_id)
    logger.info(f"[SESSION] Started: {_current_session_id}")
    return _current_session_id


def end_session() -> None:
    if _current_session_id:
        _get_repo().end_session(_current_session_id)
        logger.info(f"[SESSION] Ended: {_current_session_id}")


def record_turn(intent: str) -> None:
    if _current_session_id:
        _get_repo().increment_turn_count(_current_session_id)
        _get_repo().update_last_intent(_current_session_id, intent)


def get_current_session_id() -> str:
    return _current_session_id or ""


def get_resume_context() -> str:
    repo = _get_repo()
    snapshot = repo.get_last_snapshot()
    if not snapshot:
        return ""

    last_time_str = repo.get_last_interaction_time()
    if last_time_str:
        try:
            last_time = datetime.fromisoformat(last_time_str)
            gap = datetime.now() - last_time
            gap_str = _format_gap(gap)
        except (ValueError, TypeError):
            gap_str = "some time ago"
    else:
        gap_str = "some time ago"

    summary = snapshot["task_summary"]
    blocker = snapshot["blocker"]

    if blocker:
        return (
            f"SESSION CONTEXT: Last session ({gap_str}): {summary}. "
            f"Left unfinished: {blocker}."
        )
    return (
        f"SESSION CONTEXT: Last session ({gap_str}): {summary}. "
        f"Nothing was left unfinished."
    )


def _format_gap(gap: timedelta) -> str:
    total_seconds = int(gap.total_seconds())
    if total_seconds < 300:
        return "just now"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes} minutes ago"
    hours = total_seconds // 3600
    if hours < 24:
        return f"{hours} hours ago"
    days = total_seconds // 86400
    return f"{days} days ago"


async def save_snapshot(turns: list[dict]) -> None:
    repo = _get_repo()
    if not _current_session_id:
        return

    row = repo._db.fetchone(
        "SELECT turn_count FROM session_snapshots WHERE session_id = ?",
        (_current_session_id,),
    )
    if not row or row["turn_count"] < 2:
        return

    from .llm.contracts import ask_for_session_summary
    try:
        result = await ask_for_session_summary(turns)
        last_intent_row = repo._db.fetchone(
            "SELECT last_intent FROM session_snapshots WHERE session_id = ?",
            (_current_session_id,),
        )
        last_intent = last_intent_row["last_intent"] if last_intent_row else "unknown"
        repo.save_summary(
            _current_session_id,
            last_intent,
            result["task_summary"],
            result.get("blocker"),
        )
        logger.info(f"[SESSION] Snapshot saved: {result['task_summary']}")
    except Exception as e:
        logger.warning(f"[SESSION] Failed to save snapshot (will retry next startup): {e}")


async def recover_crashed_session() -> None:
    repo = _get_repo()
    crashed = repo.get_unsummarized_session()
    if not crashed:
        return

    crashed_id = crashed["session_id"]
    logger.info(f"[SESSION] Recovering crashed session: {crashed_id}")

    rows = repo._db.fetchall(
        "SELECT user_input, response, timestamp FROM conversations "
        "WHERE session_id = ? ORDER BY id ASC",
        (crashed_id,),
    )
    if not rows:
        logger.debug(f"[SESSION] No conversation turns found for {crashed_id}, skipping recovery")
        return

    turns = [{"user_input": r["user_input"], "response": r["response"]} for r in rows]
    last_timestamp = rows[-1]["timestamp"]

    from .llm.contracts import ask_for_session_summary
    try:
        result = await ask_for_session_summary(turns)
        last_intent = crashed.get("last_intent") or "unknown"
        repo.save_summary(crashed_id, last_intent, result["task_summary"], result.get("blocker"))
        repo._db.execute(
            "UPDATE session_snapshots SET ended_at = ? WHERE session_id = ?",
            (last_timestamp, crashed_id),
        )
        repo._db.commit()
        logger.info(f"[SESSION] Recovered crashed session: {result['task_summary']}")
    except Exception as e:
        logger.warning(f"[SESSION] Crash recovery failed (will retry next startup): {e}")
