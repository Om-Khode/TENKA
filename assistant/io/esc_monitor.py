# assistant/io/esc_monitor.py
"""ESC-hold monitor (universal abort trigger).

Single daemon thread polling VK_ESCAPE at 50ms. Hold >= 1s -> fires
abort.request_abort("esc_hold"). Lifted from automation/vision/agent.py;
hoisted so it's not tied to the vision loop.

Layering: io/ — may import core/abort and config only.
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from typing import Any

logger = logging.getLogger("esc_monitor")

# ─── Constants ────────────────────────────────────────────────────────────────

_HOLD_THRESHOLD_SECS = 1.0
_POLL_INTERVAL_SECS = 0.05
_VK_ESCAPE = 0x1B
_RESPAWN_BUDGET = 3
_RESPAWN_WINDOW_SECS = 60.0


# ─── Win32 polling helper ─────────────────────────────────────────────────────


def _is_esc_down() -> bool:
    """Windows-only. Returns True if ESC is currently pressed (any window)."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_ESCAPE) & 0x8000)
    except (AttributeError, OSError):
        return False  # non-Windows test env


# ─── EscMonitor class ────────────────────────────────────────────────────────


class EscMonitor:
    def __init__(self, abort_ref: Any = None, poll_interval: float = _POLL_INTERVAL_SECS) -> None:
        # abort_ref injected for test; defaults to the module singleton at start()
        self._abort = abort_ref
        self._poll = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pressed_at: float | None = None
        self._respawn_attempts: list[float] = []

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._abort is None:
            from assistant.core.abort import abort as _abort_singleton
            self._abort = _abort_singleton
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="cv1-esc-monitor", daemon=True)
        self._thread.start()
        logger.info("[esc_monitor] started")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            logger.info("[esc_monitor] stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except BaseException as e:
                logger.error("[esc_monitor] tick raised: %s", e)
                if not self._can_respawn():
                    logger.error("[esc_monitor] respawn budget exhausted, exiting")
                    return
            time.sleep(self._poll)

    def _tick_once(self) -> None:
        if _is_esc_down():
            if self._pressed_at is None:
                self._pressed_at = time.time()
            elif (time.time() - self._pressed_at) >= _HOLD_THRESHOLD_SECS:
                # request_abort is idempotent — safe to call unconditionally.
                # We reset _pressed_at so we fire exactly once per unbroken hold.
                self._abort.request_abort("esc_hold")
                self._pressed_at = None  # one fire per hold
        else:
            self._pressed_at = None

    def _tick_until(self, iterations: int) -> None:
        """Test helper: advance _tick_once N times without real sleeping.

        Patches the module-level ``time`` object so that each call to
        ``time.time()`` returns a value advanced by ``poll_interval`` per
        invocation.  Since _tick_once makes exactly one time.time() call per
        tick (to either set _pressed_at or to measure elapsed), each call
        increments the simulated clock by poll_interval, matching the
        semantics of the real polling loop.
        """
        import unittest.mock as _mock
        import assistant.io.esc_monitor as _mod

        base = time.time()
        counter = [0]

        def _simulated_time() -> float:
            val = base + counter[0] * self._poll
            counter[0] += 1
            return val

        with _mock.patch.object(_mod, "time") as mock_time_mod:
            mock_time_mod.time.side_effect = _simulated_time
            for _ in range(iterations):
                self._tick_once()

    def _can_respawn(self) -> bool:
        now = time.time()
        self._respawn_attempts = [t for t in self._respawn_attempts if now - t < _RESPAWN_WINDOW_SECS]
        if len(self._respawn_attempts) >= _RESPAWN_BUDGET:
            return False
        self._respawn_attempts.append(now)
        return True


# ─── Module-level singleton ───────────────────────────────────────────────────

esc_monitor = EscMonitor()
