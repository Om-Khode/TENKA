"""Milestone 6a.5, stream E -- availability and the dev-directory loader.

Four findings, all reachable the moment 6b binds a listener that is not
loopback:

* **G5 / lens 5 F3** -- one stalled socket wedges `EventHub._pump` for every
  other client. Task 17's `_drop(close=True)` added a second awaited call
  inside the same loop, so the surface that can hang it got slightly wider.
* **G6 / lens 5 F6** -- `EventHub._sockets` has no capacity cap and nothing
  ever closes an idle one.
* **G7 / lens 5 F4** -- the UI bundle decompresses synchronously on the event
  loop with no request coalescing, on a route that requires no credential.
* **G8 / lens 1 F7** -- `_from_dir` enumerates nothing, so the dev-directory
  loader publishes every file under its root.

Every fix here carries a control test as well as a reproduction: a merely slow
socket must not be evicted, an active socket must survive the idle sweep, a
cached member must stay nearly free, and the dev loader must still serve the
export it exists for.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from assistant.io.api import events as events_mod
from assistant.io.api.events import EventHub


# ─── socket doubles ──────────────────────────────────────────────────────
# Deliberately not `TestClient` sockets: task 17 recorded that publishing a hub
# frame while sending an inbound frame deadlocks the Starlette test portal, and
# a hung test reads as a slow one. Everything below drives `EventHub` directly.

class _HealthySocket:
    """Accepts every frame immediately."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.closed = False

    async def send_json(self, frame: dict) -> None:
        self.frames.append(frame)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class _SlowSocket:
    """A real client on a bad connection: late, but never stuck."""

    def __init__(self, delay: float = 0.02) -> None:
        self.frames: list[dict] = []
        self.closed = False
        self._delay = delay

    async def send_json(self, frame: dict) -> None:
        await asyncio.sleep(self._delay)
        self.frames.append(frame)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class _StalledSocket:
    """`send_json` never resolves -- a peer that stopped reading."""

    def __init__(self, *, stall_close: bool = False) -> None:
        self.frames: list[dict] = []
        self.closed = False
        self._stall_close = stall_close

    async def send_json(self, frame: dict) -> None:
        await asyncio.Event().wait()

    async def close(self, code: int = 1000) -> None:
        self.closed = True
        if self._stall_close:
            await asyncio.Event().wait()


async def _until(predicate, timeout: float = 2.0) -> bool:
    """Poll `predicate` on the loop until it holds or `timeout` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


# ─── E1 / G5: one stalled socket must not wedge the pump ─────────────────
@pytest.mark.asyncio
async def test_one_stalled_socket_does_not_wedge_the_pump_for_everyone(monkeypatch):
    """Attach a socket whose send_json never resolves, publish, then attach a
    healthy socket and publish again. The healthy one must receive.

    Lens 5 F3 proved the order-independent form of this: `_sockets` is not
    ordered by attach time, so the claim under test is not "later sockets are
    delayed" but "the pump never returns to the top of its loop at all".
    """
    monkeypatch.setattr(events_mod, "_SEND_TIMEOUT_SECONDS", 0.05, raising=False)
    hub = EventHub()
    try:
        stalled = _StalledSocket()
        await hub.attach(stalled)
        hub.publish({"type": "status", "phase": "FIRST"})
        await asyncio.sleep(0.02)

        healthy = _HealthySocket()
        await hub.attach(healthy)
        hub.publish({"type": "status", "phase": "SECOND"})

        assert await _until(lambda: bool(healthy.frames)), (
            "a healthy socket attached after the stall received nothing: "
            "the pump is wedged on the stalled socket's send"
        )
        assert healthy.frames[-1]["phase"] == "SECOND"
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_a_stalled_socket_is_dropped_not_retried_forever(monkeypatch):
    """The wedged socket must end up out of the set, not permanently retried."""
    monkeypatch.setattr(events_mod, "_SEND_TIMEOUT_SECONDS", 0.05, raising=False)
    hub = EventHub()
    try:
        stalled = _StalledSocket()
        await hub.attach(stalled)
        hub.publish({"type": "status", "phase": "FIRST"})
        assert await _until(lambda: stalled not in hub._sockets), (
            "the stalled socket is still attached and will be awaited again "
            "on the very next frame"
        )
        assert hub.subscriber_count() == 0
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_a_stalled_close_does_not_wedge_the_pump_either(monkeypatch):
    """Task 17's own memo: `_drop(close=True)` added a *second* awaited call
    inside the same loop, so `close()` needs the same bound the send does."""
    monkeypatch.setattr(events_mod, "_SEND_TIMEOUT_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(events_mod, "_CLOSE_TIMEOUT_SECONDS", 0.05, raising=False)
    hub = EventHub()
    try:
        stalled = _StalledSocket(stall_close=True)
        healthy = _HealthySocket()
        await hub.attach(stalled)
        await hub.attach(healthy)
        hub.publish({"type": "status", "phase": "ONLY"})
        assert await _until(lambda: bool(healthy.frames)), (
            "the healthy socket received nothing: close() wedged the pump"
        )
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_a_slow_but_working_socket_is_not_dropped():
    """Control: the timeout must not evict a merely slow client on a bad
    connection. Runs against the shipped timeout, not a patched one -- the
    number itself is what this pins."""
    hub = EventHub()
    try:
        slow = _SlowSocket(delay=0.05)
        await hub.attach(slow)
        hub.publish({"type": "status", "phase": "SLOW"})
        assert await _until(lambda: bool(slow.frames))
        assert slow in hub._sockets, "a slow but working socket was evicted"
        assert not slow.closed
    finally:
        await hub.stop()
