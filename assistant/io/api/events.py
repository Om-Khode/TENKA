# assistant/io/api/events.py
"""One socket carrying every live frame.

Not one SSE stream per concern: EventSource cannot set an Authorization header,
abort needs a client-to-server channel, and one socket means one reconnect
story on the client.

main.py subscribes the hub's `publish()` to status_broadcaster once, at
startup, so every socket the hub ever attaches receives its status and
task_step frames; the telemetry sampler runs only while at least one socket
is attached.

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

logger = logging.getLogger(__name__)


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

        Buffers rather than enqueues while no loop is known yet -- see the
        module docstring for why a direct enqueue here is never safe.
        """
        with self._state_lock:
            loop = self._loop
            if loop is None:
                self._pending.append(event)
                return
        loop.call_soon_threadsafe(self._enqueue, event)

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
            self.publish({
                "type": "telemetry",
                "cpu": snapshot.cpu_percent,
                "ram": snapshot.ram_percent,
                "battery": snapshot.battery_percent,
            })

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
