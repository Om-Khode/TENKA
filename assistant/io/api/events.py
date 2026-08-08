# assistant/io/api/events.py
"""One socket carrying every live frame.

Not one SSE stream per concern: EventSource cannot set an Authorization header,
abort needs a client-to-server channel, and one socket means one reconnect
story on the client.

Three frame types actually have a producer today:

  - `"status"` -- forwarded from `assistant.io.status_broadcaster` via
    `EventHub.publish_status()` (main.py subscribes that method, not
    `publish()` directly, once at startup: `status.subscribe(hub.publish_status)`).
    The connect-time frame `app.py` sends before any broadcaster event has
    fired is built by the same `build_status_frame()` used here, so it carries
    the identical key set -- a client never has to special-case "the first
    status frame" versus every later one. There is no separate frame type for
    task/step progress: `step` (`[n, total] | None`) and `tier` already ride
    as fields on `"status"` (`status_broadcaster.py`'s `set()`); a live task
    display should read those off `"status"`, not wait on a distinct frame
    type that nothing produces.
  - `"telemetry"` -- sampled from the runtime's `TelemetrySnapshot` every
    `_interval` seconds while >=1 socket is attached (`_sample_telemetry`
    below). Keys are `telemetry_body()`'s (`routes/system.py`, the same
    function `GET /v1/telemetry` uses) so the wire has one telemetry
    vocabulary, not one per transport.
  - `"error"` / `"ack"` -- built inline in `app.py`'s socket handler, in reply
    to a malformed/unrecognised client frame or a client's own
    `{"type": "abort"}`. Not hub-produced, so not covered by the frame
    builders below.

`"toast"` is reserved, not produced: a one-off notification is a different
shape of thing than a phase transition, so the type name is worth keeping for
whenever something actually emits one -- but nothing in `assistant/` does
today. Treat it as reserved-and-unproduced; do not build a client that waits
on it.

main.py subscribes the hub's `publish_status()` to status_broadcaster once,
at startup, so every socket the hub ever attaches receives its status frames;
the telemetry sampler runs only while at least one socket is attached.

`publish()` is the one method this module promises is safe to call from any
thread, not just the event loop's. `status_broadcaster` calls it from
synchronous code that may be running on a worker thread, and
`asyncio.Queue.put_nowait` is documented as safe only from the loop's own
thread -- calling it directly from elsewhere races the loop's internal
bookkeeping. `publish()` therefore hands the enqueue off to the loop captured
at `start()`/`attach()` time via `call_soon_threadsafe`, which is the one
asyncio primitive built for exactly this handoff. Before any loop has been
captured (nothing has started or attached yet -- exactly the window between
main.py wiring the subscription and the daemon's first socket, during which
status_broadcaster can already be firing from arbitrary worker threads), the
event is buffered rather than touched at all: there is no thread on which
enqueueing directly would be safe, so there is no such fallback. Once a loop
is captured, the buffer drains onto it the same way every later publish()
does.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any

from .routes.system import telemetry_body

logger = logging.getLogger(__name__)


# ─── frame builders ─────────────────────────────────────────────────────────
# Pure dict-in/dict-out functions, deliberately: this module may not import
# `assistant.io.status_broadcaster` (io/api sits one layer below `assistant.io`
# and may only reach `assistant.core`/`assistant.config`/third-party), so a
# broadcaster event is accepted here only as data -- a dict shaped the way
# `status_broadcaster.py`'s `set()` builds one -- never via the module that
# builds it. main.py is what actually bridges the two modules together.

def build_status_frame(*, phase: str, detail: str = "",
                        v: int | None = None,
                        cursor_follows: bool | None = None,
                        step: list[int] | None = None,
                        tier: str | None = None,
                        ts: float | None = None) -> dict:
    """The one `"status"` wire shape, camelCased. Shared by the connect-time
    synthetic frame (`app.py`, before any broadcaster event has ever fired)
    and every real one (`status_frame_from_broadcaster_event` below) so both
    carry exactly the same key set -- whatever the caller doesn't know yet is
    passed through as `None` (-> JSON `null`), never omitted.
    """
    return {
        "v": v,
        "type": "status",
        "phase": phase,
        "detail": detail,
        "cursorFollows": cursor_follows,
        "step": step,
        "tier": tier,
        "ts": ts,
    }


def status_frame_from_broadcaster_event(event: dict) -> dict:
    """Translate a `status_broadcaster.py`-shaped event dict (`v`, `type`,
    `phase`, `detail`, `cursor_follows`, `step`, `tier`, `ts`) into this
    socket's wire frame -- camelCasing `cursor_follows` and dropping nothing.
    Takes a plain dict, not a broadcaster import; see the module note above.
    """
    return build_status_frame(
        v=event.get("v"),
        phase=event.get("phase", ""),
        detail=event.get("detail", ""),
        cursor_follows=event.get("cursor_follows"),
        step=event.get("step"),
        tier=event.get("tier"),
        ts=event.get("ts"),
    )


def telemetry_frame(snapshot) -> dict:
    """The one `"telemetry"` wire shape -- `telemetry_body()`'s keys plus the
    frame envelope's `type`, so this can never drift from what
    `GET /v1/telemetry` reports for the same snapshot.

    `telemetry_body()` now returns a `TelemetryPayload` model (the typed-
    response rework), not a plain dict -- `model_dump(by_alias=True)` is the
    one place that has to change to keep spreading its keys; the camelCase
    key set itself is unaffected.
    """
    return {"type": "telemetry", **telemetry_body(snapshot).model_dump(by_alias=True)}


def build_error_frame(detail: str) -> dict:
    """The `"error"` reply app.py sends for a malformed or unrecognised
    client frame. Routed through a builder -- not a literal dict at the
    call site -- so the socket's one snake_case sweep (see the module
    docstring and `tests/test_api_events.py`) covers it the same way it
    covers every other frame the socket can emit, even though `EventHub`
    itself never produces this one.
    """
    return {"type": "error", "detail": detail}


def build_ack_frame(of: str) -> dict:
    """The `"ack"` reply app.py sends after acting on a client frame (only
    `{"type": "abort"}` today). Same reasoning as `build_error_frame`.
    """
    return {"type": "ack", "of": of}


class EventHub:
    def __init__(self) -> None:
        self._sockets: set[Any] = set()
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1_000)
        self._telemetry_task: asyncio.Task | None = None
        self._pump_task: asyncio.Task | None = None
        self._runtime = None
        self._interval = 2.0
        self._loop: asyncio.AbstractEventLoop | None = None
        # Guards `_loop` and `_pending` only -- both are touched from
        # publish()'s arbitrary caller threads as well as the loop thread
        # that runs attach()/start(). Nothing else in this class crosses
        # threads, so nothing else needs it.
        self._state_lock = threading.Lock()
        self._pending: list[dict] = []

    # ─── membership ─────────────────────────────────────────────────────
    def subscriber_count(self) -> int:
        return len(self._sockets)

    def telemetry_running(self) -> bool:
        return self._telemetry_task is not None and not self._telemetry_task.done()

    async def attach(self, socket) -> None:
        # Captured here too, not just in start(): a test harness (or a
        # deployment that never calls start() before the first connection)
        # still needs publish() to land on the loop actually running this
        # socket's pump, not on whatever loop happened to exist first.
        self._set_loop(asyncio.get_running_loop())
        self._sockets.add(socket)
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())
        if self._telemetry_task is None and self._runtime is not None:
            self._telemetry_task = asyncio.create_task(self._sample_telemetry())

    async def detach(self, socket) -> None:
        self._sockets.discard(socket)
        if not self._sockets and self._telemetry_task is not None:
            self._telemetry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._telemetry_task
            self._telemetry_task = None

    # ─── fan-out ────────────────────────────────────────────────────────
    def publish(self, event: dict) -> None:
        """Non-blocking. Safe from sync code on any thread (status_broadcaster).

        Sends `event` verbatim -- no translation happens here. Anything that
        needs shaping before it reaches a socket (a status_broadcaster event,
        today) goes through a dedicated method like `publish_status()`
        instead, which translates and then calls this.

        Buffers rather than enqueues while no loop is known yet -- see the
        module docstring for why a direct enqueue here is never safe.
        """
        with self._state_lock:
            loop = self._loop
            if loop is None:
                self._pending.append(event)
                return
        loop.call_soon_threadsafe(self._enqueue, event)

    def publish_status(self, event: dict) -> None:
        """The bridge point for `status_broadcaster`: main.py subscribes this
        method (`status.subscribe(hub.publish_status)`), not `publish()`
        directly, so every broadcaster event is translated to this socket's
        camelCase wire shape (`status_frame_from_broadcaster_event`) before
        anything reaches a browser. Safe from any thread, same as `publish()`.
        """
        self.publish(status_frame_from_broadcaster_event(event))

    def _set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the loop once, then flush anything published before it existed."""
        with self._state_lock:
            if self._loop is not None:
                return
            self._loop = loop
            pending, self._pending = self._pending, []
        for event in pending:
            loop.call_soon_threadsafe(self._enqueue, event)

    def _enqueue(self, event: dict) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("[API] event queue full; dropping a frame")

    async def _pump(self) -> None:
        while True:
            event = await self._queue.get()
            dead = []
            for socket in list(self._sockets):
                try:
                    await socket.send_json(event)
                except Exception:
                    dead.append(socket)
            for socket in dead:
                self._sockets.discard(socket)

    async def _sample_telemetry(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if not self._sockets or self._runtime is None:
                continue
            try:
                snapshot = await self._runtime.system.telemetry()
            except Exception as exc:
                logger.debug(f"[API] telemetry sample failed: {exc}")
                continue
            self.publish(telemetry_frame(snapshot))

    # ─── lifecycle ──────────────────────────────────────────────────────
    async def start(self, runtime, interval_seconds: float = 2.0) -> None:
        self._runtime = runtime
        self._interval = interval_seconds
        self._set_loop(asyncio.get_running_loop())
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        for task in (self._telemetry_task, self._pump_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._telemetry_task = None
        self._pump_task = None
        self._sockets.clear()
