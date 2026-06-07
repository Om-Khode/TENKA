"""Universal abort controller.

Singleton flag + subscriber pattern. Checked at loop boundaries inside
cursor-controlling handlers. Fired by io/esc_monitor.py (VK_ESCAPE hold)
or by in-proc callers.

Layering: this is the LOWEST layer. Must not import anything outside
stdlib + logging.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("abort")

_OFF_TASK_SUPPRESS_AFTER_SECS = 30.0


class UserAborted(Exception):
    """Raised by handlers after observing abort.is_aborted() is True."""


class AbortController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._aborted = False
        self._reason: str | None = None
        self._tasks: set[str] = set()
        self._last_reset_ts: float = time.time()
        self._subscribers: list[Callable[[str], None]] = []

    def is_aborted(self) -> bool:
        with self._lock:
            return self._aborted

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def request_abort(self, reason: str = "user") -> None:
        # Subscribers (e.g. stop_streaming + STOPPED pill) fire on EVERY call,
        # even when already aborted. The flag itself is idempotent (first reason
        # wins) so handler-loop semantics don't change — but a second ESC press
        # after a stale abort (e.g. user aborted a planner task, then said
        # something, then ESC'd again during TTS) must still kill audio. The
        # earlier "if self._aborted: return" was eating those repeat ESCs and
        # leaving small-talk / proactive-nudge TTS uninterruptible.
        with self._lock:
            already = self._aborted
            if not already:
                self._aborted = True
                self._reason = reason
            subs = list(self._subscribers)
        logger.info("[abort] requested: %s%s", reason, " (repeat)" if already else "")
        for cb in subs:
            try:
                cb(reason)
            except Exception as e:
                logger.warning("[abort] subscriber raised: %s", e)

    def reset(self) -> None:
        with self._lock:
            self._aborted = False
            self._reason = None
            self._last_reset_ts = time.time()
        logger.debug("[abort] reset")

    def register_task(self, task_id: str) -> None:
        with self._lock:
            self._tasks.add(task_id)

    def unregister_task(self, task_id: str) -> None:
        with self._lock:
            self._tasks.discard(task_id)

    def on_abort(self, callback: Callable[[str], None]) -> None:
        # NOTE: subscribers are permanent for the lifetime of this controller —
        # callers should register at startup and never call this from inside a
        # handler/loop. Tests that exercise abort flow should use AbortController()
        # directly rather than the module-level `abort` singleton.
        with self._lock:
            self._subscribers.append(callback)


abort = AbortController()
