# assistant/io/status_broadcaster.py
"""status broadcaster.

Singleton that handlers call to announce phase changes. Fans out to
in-proc subscribers (callbacks) and an optional IPC writer (overlay
subprocess stdin). Rate-limited (50ms per phase) and deduped to avoid
flooding the pipe during fast click loops.

Layering: io/ — may import core/abort + config only.
"""
from __future__ import annotations

import enum
import json
import logging
import select
import threading
import time
from typing import Callable, TextIO

logger = logging.getLogger("status_broadcaster")

_PROTOCOL_VERSION = 2
_RATE_LIMIT_SECS = 0.05
_DROP_LOG_INTERVAL_SECS = 60.0


# ─── Phase enum ───────────────────────────────────────────────────────────────

class StatusPhase(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    READING = "READING"
    CLICKING = "CLICKING"
    TYPING = "TYPING"
    BROWSING = "BROWSING"
    VISION = "VISION"
    HEALING = "HEALING"
    SPEAKING = "SPEAKING"
    DONE = "DONE"
    STOPPED = "STOPPED"  # user pressed ESC mid-task


# Valid `tier` literals — match design's TIER_META keys
TIERS = frozenset({"vision", "native", "browser"})


# ─── Broadcaster ──────────────────────────────────────────────────────────────

class StatusBroadcaster:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._writer: TextIO | None = None
        self._subscribers: list[Callable[[dict], None]] = []
        self._last_event: tuple[str, str, bool] | None = None
        self._last_ts_per_phase: dict[str, float] = {}
        self._last_drop_log_ts = 0.0
        self._on_overlay_dead: Callable[[], None] | None = None

    # ─── IPC attachment ───────────────────────────────────────────────────────

    def attach_ipc(self, writer: TextIO) -> None:
        with self._lock:
            self._writer = writer

    def detach_ipc(self) -> None:
        with self._lock:
            self._writer = None

    # ─── In-proc pub/sub ──────────────────────────────────────────────────────

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def set_on_overlay_dead(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._on_overlay_dead = callback

    # ─── Main publish method ──────────────────────────────────────────────────

    def set(self, phase: StatusPhase, *, detail: str = "",
            cursor_follows: bool = False,
            step: tuple[int, int] | None = None,
            tier: str | None = None) -> None:
        if not isinstance(phase, StatusPhase):
            raise TypeError(f"phase must be StatusPhase enum, got {type(phase).__name__}")
        if tier is not None and tier not in TIERS:
            raise ValueError(f"tier must be one of {sorted(TIERS)} or None, got {tier!r}")
        if step is not None:
            if not (isinstance(step, tuple) and len(step) == 2):
                raise TypeError(f"step must be (n, total) tuple or None, got {step!r}")
        now = time.time()
        key = (phase.value, detail, cursor_follows, step, tier)

        with self._lock:
            # Dedupe: identical full event tuple skipped
            if key == self._last_event:
                return
            # Rate-limit per phase (IDLE is exempt — always fires transitions)
            if phase is not StatusPhase.IDLE:
                last = self._last_ts_per_phase.get(phase.value, 0.0)
                if (now - last) < _RATE_LIMIT_SECS:
                    return
            self._last_event = key
            self._last_ts_per_phase[phase.value] = now
            subs = list(self._subscribers)
            writer = self._writer

        event: dict = {
            "v": _PROTOCOL_VERSION,
            "type": "status",
            "phase": phase.value,
            "detail": detail,
            "cursor_follows": cursor_follows,
            "step": [step[0], step[1]] if step else None,
            "tier": tier,
            "ts": now,
        }
        # Fire in-proc subscribers first (never blocked by IPC)
        for cb in subs:
            try:
                cb(event)
            except Exception as e:
                logger.warning("[status] subscriber raised: %s", e)
        # Then write to IPC pipe if attached
        if writer is not None:
            self._write_ipc(writer, event)

    # ─── IPC write helpers ────────────────────────────────────────────────────

    def _write_ipc(self, writer: TextIO, event: dict) -> None:
        """Non-blocking write probe — drop event if pipe not ready."""
        # Try a select() probe to avoid blocking on a full pipe buffer.
        # StringIO (used in tests) has no real fileno; the except clause skips
        # the probe so tests still get the write.
        try:
            fileno = writer.fileno()
            _, ready, _ = select.select([], [fileno], [], 0)
            if not ready:
                self._maybe_log_drop()
                return
        except (OSError, ValueError, AttributeError):
            pass  # not a real fd (e.g. StringIO in tests) — skip probe

        try:
            writer.write(json.dumps(event) + "\n")
            writer.flush()
        except (BrokenPipeError, OSError) as e:
            logger.warning("[status] IPC write failed: %s", e)
            self.detach_ipc()
            with self._lock:
                on_dead = self._on_overlay_dead
            if on_dead:
                try:
                    on_dead()
                except Exception as cb_e:
                    logger.warning("[status] on_overlay_dead raised: %s", cb_e)

    def _maybe_log_drop(self) -> None:
        now = time.time()
        if (now - self._last_drop_log_ts) > _DROP_LOG_INTERVAL_SECS:
            logger.warning("[status] dropped event — overlay stdin not ready")
            self._last_drop_log_ts = now


# ─── Module singleton ─────────────────────────────────────────────────────────

status = StatusBroadcaster()
