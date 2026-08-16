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
import contextlib
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


# ─── E2 / G6: cap and time out sockets ───────────────────────────────────
@pytest.mark.asyncio
async def test_a_single_device_cannot_hold_unlimited_sockets(monkeypatch):
    """Lens 5 F6: new handshakes are metered at ~120/min per device, but
    nothing capped how many one device *accumulates* over hours."""
    monkeypatch.setattr(events_mod, "_MAX_SOCKETS_PER_DEVICE", 3, raising=False)
    hub = EventHub()
    try:
        held = [_HealthySocket() for _ in range(3)]
        for socket in held:
            assert await hub.attach(socket, device_id="phone") is True
        refused = _HealthySocket()
        assert await hub.attach(refused, device_id="phone") is False
        assert refused not in hub._sockets
        assert refused.closed, "a refused socket must be closed, not left open"
        assert hub.subscriber_count() == 3
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_a_second_device_is_not_blocked_by_the_first_devices_cap(monkeypatch):
    """Control: the per-device cap must not collapse into a global one and
    lock the operator's own laptop out because a phone filled its share."""
    monkeypatch.setattr(events_mod, "_MAX_SOCKETS_PER_DEVICE", 2, raising=False)
    hub = EventHub()
    try:
        for _ in range(2):
            assert await hub.attach(_HealthySocket(), device_id="phone") is True
        assert await hub.attach(_HealthySocket(), device_id="phone") is False
        assert await hub.attach(_HealthySocket(), device_id="laptop") is True
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_a_global_socket_cap_exists(monkeypatch):
    """A per-device cap alone is multiplied by however many devices a vault
    holds; `_pump` iterates every one of them on every published frame."""
    monkeypatch.setattr(events_mod, "_MAX_SOCKETS_PER_DEVICE", 4, raising=False)
    monkeypatch.setattr(events_mod, "_MAX_SOCKETS_TOTAL", 4, raising=False)
    hub = EventHub()
    try:
        for index in range(4):
            assert await hub.attach(_HealthySocket(),
                                    device_id=f"device-{index}") is True
        refused = _HealthySocket()
        assert await hub.attach(refused, device_id="device-late") is False
        assert hub.subscriber_count() == 4
    finally:
        await hub.stop()


def test_the_per_device_cap_binds_before_the_rate_limit_does():
    """The spec's constraint on the number itself: one device must not be able
    to accumulate sockets faster than the handshake limiter lets it open them.

    The limiter allows `_MAX_PER_WINDOW` handshakes per `_WINDOW_SECONDS` per
    device. If the concurrent cap were the larger of the two, the limiter would
    still be what bounds accumulation and the cap would be decoration.
    """
    from assistant.io.api.security import _MAX_PER_WINDOW

    assert events_mod._MAX_SOCKETS_PER_DEVICE < _MAX_PER_WINDOW
    assert events_mod._MAX_SOCKETS_TOTAL >= events_mod._MAX_SOCKETS_PER_DEVICE


@pytest.mark.asyncio
async def test_an_idle_socket_is_closed_by_the_timeout(monkeypatch):
    """Nothing ever closed a socket that simply sat there. One that has neither
    accepted a frame nor said anything for the whole window is not a live view,
    it is an entry `_pump` pays for on every publish."""
    monkeypatch.setattr(events_mod, "_IDLE_TIMEOUT_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(events_mod, "_REVALIDATE_INTERVAL_SECONDS", 0.02)
    hub = EventHub()
    try:
        socket = _HealthySocket()
        await hub.attach(socket, device_id="phone")
        assert await _until(lambda: socket not in hub._sockets), (
            "an idle socket was never reaped"
        )
        assert socket.closed
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_an_active_socket_is_never_closed_by_the_idle_timeout(monkeypatch):
    """Control: the operator's own live view must survive a quiet period."""
    monkeypatch.setattr(events_mod, "_IDLE_TIMEOUT_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(events_mod, "_REVALIDATE_INTERVAL_SECONDS", 0.02)
    hub = EventHub()
    try:
        socket = _HealthySocket()
        await hub.attach(socket, device_id="phone")
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            hub.publish({"type": "status", "phase": "WORKING"})
            await asyncio.sleep(0.02)
        assert socket in hub._sockets, "a socket receiving frames was reaped"
        assert not socket.closed
        assert len(socket.frames) > 1
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_an_inbound_frame_also_counts_as_activity(monkeypatch):
    """The receive half lives in `app.py`'s socket handler, which this stream
    does not own. `note_activity()` is the hook it calls; a socket whose only
    traffic is inbound must not be reaped as idle."""
    monkeypatch.setattr(events_mod, "_IDLE_TIMEOUT_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(events_mod, "_REVALIDATE_INTERVAL_SECONDS", 0.02)
    hub = EventHub()
    try:
        socket = _HealthySocket()
        await hub.attach(socket, device_id="phone")
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            hub.note_activity(socket)
            await asyncio.sleep(0.02)
        assert socket in hub._sockets
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_a_detached_socket_leaves_no_bookkeeping_behind(monkeypatch):
    """The caps are only real if their counters shrink again -- a per-device
    tally that never decremented would lock a device out after eight reloads."""
    monkeypatch.setattr(events_mod, "_MAX_SOCKETS_PER_DEVICE", 2, raising=False)
    hub = EventHub()
    try:
        for _ in range(4):
            socket = _HealthySocket()
            assert await hub.attach(socket, device_id="phone") is True
            await hub.detach(socket)
        assert hub.subscriber_count() == 0
    finally:
        await hub.stop()


# ─── E3 / G7: coalesce and offload the UI bundle read ────────────────────
@pytest.fixture()
def bundle(tmp_path):
    """A real zip-backed bundle. The contract hash is irrelevant here: these
    tests drive `UiBundle` directly rather than through the route, so the
    stale-bundle 503 never enters into it."""
    from assistant.io.api.ui import UiBundle
    from tests.fakes.studio_ui import write_ui_zip

    built = UiBundle.open(
        zip_path=write_ui_zip(tmp_path / "studio-ui.zip", "any"), dir_path=None)
    assert built is not None
    return built


def _counting_read(bundle_obj, *, delay: float = 0.0):
    """Replace `_read_bytes` with a counting, deliberately slow stand-in.

    Slow *synchronously*, because that is exactly what a real decompression is:
    lens 5 F4 measured 47-97ms of blocking work for an 8 MiB member.
    """
    calls: list[str] = []
    original = bundle_obj._read_bytes

    def _read(safe: str):
        calls.append(safe)
        if delay:
            time.sleep(delay)
        return original(safe)

    bundle_obj._read_bytes = _read
    return calls


@pytest.mark.asyncio
async def test_concurrent_first_requests_decompress_once_not_n_times(bundle):
    """Ten threads racing the same cache miss produced ten decompressions.

    Nothing coalesced two callers racing to populate the cache: the lock is
    held only around the cache dict's get and set, and the read itself runs
    outside it.
    """
    calls = _counting_read(bundle, delay=0.05)
    results = await asyncio.gather(
        *[bundle.read_async("index.html") for _ in range(10)])
    assert len(calls) == 1, f"the same cache miss was read {len(calls)} times"
    assert all(result == results[0] for result in results)
    assert results[0] is not None


@pytest.mark.asyncio
async def test_decompression_does_not_block_the_event_loop(bundle):
    """An async route calling `bundle.read()` with no await blocks everything.

    The daemon shares its loop with TENKA herself, so tens of milliseconds of
    synchronous zip work is tens of milliseconds nothing else runs -- on a
    route that requires no credential at all.
    """
    _counting_read(bundle, delay=0.3)
    ticks = 0

    async def _tick() -> None:
        nonlocal ticks
        for _ in range(100):
            await asyncio.sleep(0.005)
            ticks += 1

    ticker = asyncio.create_task(_tick())
    await bundle.read_async("index.html")
    ticker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticker
    assert ticks > 5, (
        f"the loop advanced only {ticks} ticks during a 300ms read: the "
        f"decompression is still running on the event loop"
    )


@pytest.mark.asyncio
async def test_a_cached_member_is_still_nearly_free(bundle):
    """Control: the coalescing must not add cost to the warm path.

    A cached re-read measured ~0.01-0.02ms before any of this existed. The
    assertion that matters is not the microseconds but that the warm path
    reaches neither the archive nor a worker thread.
    """
    warm = await bundle.read_async("index.html")
    assert warm is not None
    calls = _counting_read(bundle, delay=0.5)
    started = time.perf_counter()
    for _ in range(50):
        assert await bundle.read_async("index.html") == warm
    elapsed = time.perf_counter() - started
    assert not calls, "a cached member went back to the archive"
    assert elapsed < 0.1, f"50 cached reads took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_a_member_that_does_not_exist_is_still_a_miss_not_a_cached_none(bundle):
    """`UiBundle`'s per-member cache never caches a miss -- lens 5 named that
    as one of the controls that already held. The in-flight map must not
    quietly become a negative cache."""
    assert await bundle.read_async("nope.js") is None
    assert await bundle.read_async("nope.js") is None
    assert bundle.read("index.html") is not None


@pytest.mark.asyncio
async def test_the_in_flight_map_does_not_leak_an_entry_per_member(bundle):
    """An in-flight map that is never emptied is the memory leak the cache was
    already careful not to be."""
    await asyncio.gather(*[bundle.read_async(name)
                           for name in ("index.html", "app.html", "nope.js")])
    assert bundle._inflight == {}


def test_the_ui_route_awaits_the_bundle_instead_of_blocking_on_it():
    """The route is `async def`, so a synchronous `bundle.read()` inside it is
    the blocking call this task exists to remove."""
    import inspect

    from assistant.io.api import ui as ui_mod

    source = inspect.getsource(ui_mod.mount_ui)
    assert "await bundle.read_async(" in source
    assert "bundle.read(" not in source.replace("bundle.read_async(", "")
