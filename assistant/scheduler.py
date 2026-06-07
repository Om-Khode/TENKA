"""Scheduled Conditional Tasks — background scheduler.

Daemon thread polls every 30 seconds for due schedules.
Tasks execute via existing handlers, results are conditionally
pushed to the proactive TTS queue.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING

from croniter import croniter
from assistant.storage.db import get_db

if TYPE_CHECKING:
    from assistant.storage.repos.schedule import ScheduleRepo

logger = logging.getLogger(__name__)

_stop_event = threading.Event()
_thread: threading.Thread | None = None
_repo: ScheduleRepo | None = None
_loop: asyncio.AbstractEventLoop | None = None

POLL_INTERVAL = 30  # seconds


# ─── Public API ──────────────────────────────────────────────────

def start(loop: asyncio.AbstractEventLoop) -> None:
    global _thread, _repo, _loop
    from assistant.storage.repos.schedule import ScheduleRepo

    db = get_db()
    if db is None:
        logger.warning("[scheduler] DB not initialised — scheduler disabled")
        return

    _repo = ScheduleRepo(db)
    _loop = loop

    if _thread and _thread.is_alive():
        return

    _stop_event.clear()
    _thread = threading.Thread(target=_poll_loop, name="sc1-scheduler", daemon=True)
    _thread.start()
    logger.info(f"[scheduler] Scheduler started (polling every {POLL_INTERVAL}s)")


def stop() -> None:
    global _thread, _repo, _loop
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
        _thread = None
    _repo = None
    _loop = None
    logger.info("[scheduler] Scheduler stopped")


# ─── Poll Loop ───────────────────────────────────────────────────

def _poll_loop() -> None:
    logger.info("[scheduler] Poll loop thread alive")
    while not _stop_event.wait(timeout=POLL_INTERVAL):
        try:
            now = datetime.now()
            due = _repo.get_due(now.isoformat())
            for task in due:
                try:
                    _fire_task(task, now)
                except Exception as e:
                    logger.warning(f"[scheduler] Task #{task['id']} '{task['name']}' failed: {e}")
        except Exception as e:
            logger.warning(f"[scheduler] Poll error: {e}")
    logger.info("[scheduler] Poll loop exiting")


def _fire_task(task: dict, now: datetime) -> None:
    logger.info(f"[scheduler] Firing task #{task['id']} '{task['name']}'")

    result = _run_handler(task)

    notify, summary = _evaluate_condition(task, result)

    if notify:
        _push_notification(task["name"], summary)

    next_fire = _compute_next_fire(task["cron_expr"], now)
    result_hash = _compute_result_hash(result) if result else None
    _repo.update_after_fire(task["id"], next_fire, result_hash)

    logger.info(
        f"[scheduler] Task #{task['id']} done — notify={notify}, next={next_fire}"
    )


# ─── Handler Dispatch ────────────────────────────────────────────

def _run_handler(task: dict) -> str:
    if _loop is None:
        return ""

    future = asyncio.run_coroutine_threadsafe(
        _async_run_handler(task), _loop
    )
    try:
        return future.result(timeout=120)
    except Exception as e:
        logger.warning(f"[scheduler] Handler error for '{task['name']}': {e}")
        return ""


async def _async_run_handler(task: dict) -> str:
    from assistant.actions import execute

    task_type = task["task_type"]
    goal = task["task_goal"]

    if task_type == "web_search":
        return await execute("web_search", {"query": goal})
    elif task_type == "http_check":
        return await _http_check(goal)
    elif task_type == "procedure":
        from assistant import procedures
        proc = procedures.find_by_name_or_trigger(goal)
        if proc is None:
            logger.warning(f"[scheduler] Procedure not found: {goal}")
            return ""
        from assistant.procedure_executor import run_procedure
        return await run_procedure(proc, goal)
    else:
        logger.warning(f"[scheduler] Unknown task_type: {task_type}")
        return ""


async def _http_check(url: str) -> str:
    import asyncio
    import requests as _requests

    def _get():
        resp = _requests.get(url, timeout=10)
        return resp.text[:2000]

    try:
        return await asyncio.get_running_loop().run_in_executor(None, _get)
    except Exception as e:
        logger.warning(f"[scheduler] http_check failed for {url}: {e}")
        return ""


# ─── Notify Mode Logic ──────────────────────────────────────────

def _evaluate_condition(task: dict, result: str) -> tuple[bool, str]:
    notify_mode = task["notify_mode"]
    if notify_mode == "on_match_only" and task.get("condition_text"):
        return _evaluate_condition_llm(result, task["condition_text"])
    return _should_notify_sync(task, result)


def _should_notify_sync(task: dict, result: str) -> tuple[bool, str]:
    notify_mode = task["notify_mode"]

    if notify_mode == "on_change":
        new_hash = _compute_result_hash(result)
        old_hash = task.get("last_result_hash")
        if old_hash is None or new_hash != old_hash:
            return True, result or "Results changed"
        return False, ""

    # "always" and "on_match_only" without condition_text both notify
    return True, result or "Task completed"


def _evaluate_condition_llm(result: str, condition_text: str) -> tuple[bool, str]:
    if _loop is None:
        return False, ""

    future = asyncio.run_coroutine_threadsafe(
        _async_condition_check(result, condition_text), _loop
    )
    try:
        checked = future.result(timeout=30)
        return checked["notify"], checked["summary"]
    except Exception as e:
        logger.warning(f"[scheduler] Condition check failed: {e}")
        return False, ""


async def _async_condition_check(result: str, condition_text: str) -> dict:
    from assistant.llm.contracts import ask_for_condition_check
    return await ask_for_condition_check(result, condition_text)


# ─── Notification ────────────────────────────────────────────────

def _push_notification(name: str, summary: str) -> None:
    from assistant import proactive

    msg = summary if summary else f"Monitor '{name}' triggered"
    if len(msg) > 300:
        msg = msg[:297] + "..."
    proactive._proactive_queue.put(msg)
    logger.info(f"[scheduler] Notified: {msg}")


# ─── Utilities ───────────────────────────────────────────────────

def _compute_next_fire(cron_expr: str, start: datetime) -> str:
    return croniter(cron_expr, start).get_next(datetime).isoformat()


def _compute_result_hash(result: str) -> str:
    return hashlib.sha256(result.encode()).hexdigest()
