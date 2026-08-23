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

    ran, result = _run_handler(task)

    if ran:
        notify, summary = _evaluate_condition(task, result)
    else:
        # Did not run at all. `_should_notify_sync` used to turn an empty
        # result into the literal string "Task completed", so a schedule
        # pointing at a procedure that no longer exists announced success once
        # a minute -- the log said `Procedure not found` on the line above and
        # `Notified: Task completed` on the line below. Reporting a state change
        # that did not happen is KI-28's shape, arriving through the scheduler.
        #
        # Routed through the same notify_mode logic rather than always speaking:
        # `on_change` then dedupes a repeating failure by hash, which is what
        # stops a broken minute-schedule talking sixty times an hour.
        notify, summary = _should_notify_sync(task, result)

    if notify:
        _push_notification(task["name"], summary)

    next_fire = _compute_next_fire(task["cron_expr"], now)
    # The failure text is hashed too, deliberately: under `on_change` a task
    # that starts failing says so once, and a task that starts working again
    # says that once as well.
    result_hash = _compute_result_hash(result) if result else None
    _repo.update_after_fire(task["id"], next_fire, result_hash)

    logger.info(
        f"[scheduler] Task #{task['id']} "
        f"{'done' if ran else 'DID NOT RUN'} — notify={notify}, "
        f"next={next_fire}"
    )


# ─── Handler Dispatch ────────────────────────────────────────────

def _run_handler(task: dict) -> "tuple[bool, str]":
    """`(ran, text)`. `ran=False` means the work never happened.

    The distinction exists because every not-run path used to return `""`, and
    an empty string was then reported as "Task completed". A caller cannot tell
    "ran and had nothing to say" from "never ran" out of one empty string, so it
    is two values now.
    """
    if _loop is None:
        logger.warning(f"[scheduler] No event loop for '{task['name']}'")
        return False, "the assistant was not running"

    future = asyncio.run_coroutine_threadsafe(
        _async_run_handler(task), _loop
    )
    try:
        out = future.result(timeout=120)
    except Exception as e:
        logger.warning(f"[scheduler] Handler error for '{task['name']}': {e}")
        return False, f"it failed with {type(e).__name__}"

    if out is None:
        return False, "there was nothing to run"
    return True, out


async def _async_run_handler(task: dict) -> "str | None":
    from assistant.actions import (
        LOCAL_GRANTS, LOCAL_PRINCIPAL, LOCAL_RAISE_CONTEXT, execute,
    )
    # `run_turn` installs the three authority contextvars in one place, in the
    # order `main.py` argued for -- grants LAST, nothing between that line and
    # the `try`. These two branches installed them by hand with grants FIRST,
    # which is the arrangement `main.py` was fixed for: a raise between the
    # first install and the `try` left the grant set installed with no reset.
    # Neither setter plausibly raises, which is why it survived two reviews and
    # is the same argument that was rejected over there.
    #
    # `brain.turn` and not `brain` -- see `tests/test_brain_authority.py`'s A5
    # test. This file may reach the authority installer and must not reach
    # anything that can construct or resume a Task.
    from assistant.brain.turn import run_turn

    task_type = task["task_type"]
    goal = task["task_goal"]

    if task_type == "web_search":
        # A scheduled task has no requester attached to it, so
        # `current_grants` would be unset and `execute()` would refuse -- it
        # fails closed by design. Stated explicitly here: scheduling one
        # requires EXECUTE (`manage_schedule` in core/intent_capabilities.py),
        # so whoever installed this task already held it.
        #
        # The principal is the same argument with the same answer: whatever
        # this task arms is the operator's question to answer, and an unset
        # principal would arm it for nobody -- a confirmation she could not
        # answer at her own keyboard. See core/principal.py.
        # The raise context rides along for the same reason it always did: a
        # scheduled task never reaches a refusal in practice (LOCAL_GRANTS
        # holds everything), but every install site installs all three, so an
        # unset one is never mistaken for a forgotten one.
        return await run_turn(
            grants=LOCAL_GRANTS,
            principal=LOCAL_PRINCIPAL,
            raise_context=LOCAL_RAISE_CONTEXT,
            work=lambda: execute("web_search", {"query": goal}),
            label="schedule:web_search",
        )
    elif task_type == "http_check":
        return await _http_check(goal)
    elif task_type == "procedure":
        from assistant import procedures
        proc = procedures.find_by_name_or_trigger(goal)
        if proc is None:
            logger.warning(f"[scheduler] Procedure not found: {goal}")
            # None, not "" -- see `_run_handler`. A missing procedure is a task
            # that did not run, and it must not be reported as one that did.
            return None
        from assistant.procedure_executor import run_procedure
        # Same reasoning as the web_search branch above, and now load-bearing
        # rather than tidy: `run_procedure` checks EXECUTE itself, so without
        # this the scheduler would run every stored procedure with
        # `current_grants` unset and be refused. Installing the schedule
        # required EXECUTE (`manage_schedule`), so the grant being spent here
        # is the installer's, stated rather than inherited. The principal
        # rides along for the reason the web_search branch above gives.
        return await run_turn(
            grants=LOCAL_GRANTS,
            principal=LOCAL_PRINCIPAL,
            raise_context=LOCAL_RAISE_CONTEXT,
            work=lambda: run_procedure(proc, goal),
            label="schedule:procedure",
        )
    else:
        logger.warning(f"[scheduler] Unknown task_type: {task_type}")
        return None


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

    # "always" and "on_match_only" without condition_text both notify.
    #
    # No `or "Task completed"`. That fallback is where a not-run task acquired
    # a success message: every failure path returned `""`, and this line turned
    # it into an announcement that the work was done. `_fire_task` now passes a
    # reason in for the not-run case, and a task that genuinely ran with nothing
    # to say gets a sentence that claims nothing.
    return True, result or "it ran and returned nothing"


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
