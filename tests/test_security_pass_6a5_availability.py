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
    nothing capped how many one device *accumulates* over hours.

    Amended on `fix/6a5-api-review`. This used to assert `refused.closed` --
    that the hub closes the socket it turns away. It no longer does, and the
    change is the fix for the review's P2-7: the hub's close sent code 1008,
    which flipped Starlette's application state to DISCONNECTED, so
    `app.py`'s own `close(code=1013)` on the next line raised
    `RuntimeError: Cannot call "send" once a close message has been sent` --
    out of the endpoint, because that handler's `try:` begins after the
    return. The client never received the 1013 and could not tell "at
    capacity, retry" from "you are not allowed here".

    The property being pinned is unchanged -- a refused socket must not be
    left open and must not be attached -- but closing it is the caller's, who
    is the only party that knows which close code is true. `app.py` does it
    under `suppress(Exception)` immediately after this returns `False`, and
    audits the refusal, which it also never used to do.
    """
    monkeypatch.setattr(events_mod, "_MAX_SOCKETS_PER_DEVICE", 3, raising=False)
    hub = EventHub()
    try:
        held = [_HealthySocket() for _ in range(3)]
        for socket in held:
            assert await hub.attach(socket, device_id="phone") is True
        refused = _HealthySocket()
        assert await hub.attach(refused, device_id="phone") is False
        assert refused not in hub._sockets
        assert not refused.closed, (
            "the hub must leave the close to its caller: closing here sends "
            "1008 and makes app.py's close(1013) raise on a dead socket")
        assert hub.subscriber_count() == 3
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_the_refused_socket_is_closed_exactly_once_with_1013(monkeypatch):
    """The other half of the ownership change above, on a socket that behaves
    the way Starlette's does.

    A real `WebSocket` refuses a second send after a close, so "the hub closes
    it AND app.py closes it" is not a harmless double-close -- it is a
    `RuntimeError` per refused attempt, at up to the 120 handshakes/min the
    limiter allows. This drives the exact two calls the handler makes and
    asserts the socket saw one close, carrying 1013.
    """
    monkeypatch.setattr(events_mod, "_MAX_SOCKETS_PER_DEVICE", 1, raising=False)

    class _Starletteish:
        def __init__(self):
            self.closes: list[int] = []

        async def send_json(self, frame):
            if self.closes:
                raise RuntimeError(
                    'Cannot call "send" once a close message has been sent.')

        async def close(self, code: int = 1000):
            if self.closes:
                raise RuntimeError(
                    'Cannot call "send" once a close message has been sent.')
            self.closes.append(code)

    hub = EventHub()
    try:
        assert await hub.attach(_Starletteish(), device_id="phone") is True
        refused = _Starletteish()
        if not await hub.attach(refused, device_id="phone"):
            await refused.close(code=1013)      # what app.py does
        assert refused.closes == [1013], refused.closes
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_the_hub_publishes_a_keepalive_of_its_own(monkeypatch):
    """Review P2-6: the idle timeout had nothing refreshing it.

    `_last_active` was written by `attach()`, by an inbound frame, and by a
    successful send -- and nothing in this module or in `app.py` ever sent
    anything periodically. uvicorn's protocol pings are handled below Starlette
    and never reach `note_activity()`, so the whole 120s reaper rested on the
    telemetry sampler, which needs a runtime and which one bad reading used to
    end permanently. A listen-only client -- a wall display, a backgrounded
    phone tab -- was evicted the moment that stopped.

    The heartbeat rides the revalidate sweep and goes through `publish()`, so
    `_pump` stays the only writer per socket and a successful send refreshes
    the clock by the same path every other frame uses. Driven here with both
    intervals collapsed so the test costs milliseconds.
    """
    monkeypatch.setattr(events_mod, "_REVALIDATE_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(events_mod, "_HEARTBEAT_INTERVAL_SECONDS", 0.0)
    hub = EventHub()
    socket = _HealthySocket()
    try:
        assert await hub.attach(socket, device_id="phone") is True
        for _ in range(50):
            await asyncio.sleep(0.01)
            if any(f.get("type") == "ping" for f in socket.frames):
                break
        assert any(f.get("type") == "ping" for f in socket.frames), (
            "no keepalive reached an attached socket, so the only thing "
            "refreshing its idle clock is the telemetry sampler")
    finally:
        await hub.stop()


@pytest.mark.asyncio
async def test_a_keepalive_refreshes_the_idle_clock(monkeypatch):
    """The point of the frame, not merely that it is sent: a socket nobody
    talks to must survive its own idle window because the hub keeps it
    alive."""
    monkeypatch.setattr(events_mod, "_REVALIDATE_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(events_mod, "_HEARTBEAT_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(events_mod, "_IDLE_TIMEOUT_SECONDS", 0.15)
    hub = EventHub()
    socket = _HealthySocket()
    try:
        assert await hub.attach(socket, device_id="phone") is True
        await asyncio.sleep(0.6)          # four idle windows
        assert socket in hub._sockets, (
            "a listen-only socket was idle-reaped despite the keepalive")
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


# ─── E4 / G8: the dev-directory loader must enumerate ────────────────────
def _reference_contract() -> str:
    """The contract hash of a daemon built the way `_ui_client` builds one.

    Computed rather than hardcoded: the hash is a function of the whole
    OpenAPI schema, so any route change would otherwise take these tests dark
    with a 503 that looks like the finding.
    """
    import tempfile
    from pathlib import Path

    from assistant.io.api.app import create_app
    from assistant.io.api.ui import contract_hash
    from assistant.io.api.vault import TokenVault
    from tests.fakes.studio_runtime import build_fake_runtime

    return contract_hash(create_app(
        build_fake_runtime(), TokenVault(Path(tempfile.mkdtemp())),
        origins=["http://localhost:3000"], ui_bundle=None))


def _ui_client(built):
    import tempfile
    from pathlib import Path

    from assistant.io.api.vault import TokenVault
    from tests.fakes.api_client import build_api_client
    from tests.fakes.studio_runtime import build_fake_runtime

    return build_api_client(build_fake_runtime(),
                            TokenVault(Path(tempfile.mkdtemp())),
                            ui_bundle=built)


def _dev_bundle(tmp_path, extra: dict[str, bytes] | None = None):
    from assistant.io.api.ui import UiBundle
    from tests.fakes.studio_ui import write_ui_dir

    root = write_ui_dir(tmp_path / "out", _reference_contract())
    for name, body in (extra or {}).items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    built = UiBundle.open(zip_path=None, dir_path=root)
    assert built is not None
    return root, built


def test_the_dev_directory_server_only_publishes_the_export(tmp_path):
    """Task 7 proved you cannot escape the root. It never asked whether the
    root is a safe thing to publish wholesale.

    `_from_zip` enumerates member names into `self._names` and `_read_from_zip`
    refuses anything outside that set. `_from_dir` passed `names=frozenset()`
    and enumerated nothing, so the only barrier between an unauthenticated
    request and any file under the root was `_PRIVATE_MEMBERS` -- a deny-list
    with exactly one entry. Observed: `GET /.env` returned 200 with the file's
    contents.
    """
    secret = b"GEMINI_API_KEY=sk-live-not-for-the-internet"
    _, built = _dev_bundle(tmp_path, {
        ".env": secret,
        ".env.local": secret,
        ".env.production.local": secret,
        ".DS_Store": secret,
        ".git/config": secret,
        ".vscode/settings.json": secret,
    })
    client = _ui_client(built)
    for path in ("/.env", "/.env.local", "/.env.production.local", "/.DS_Store",
                 "/.git/config", "/.vscode/settings.json"):
        response = client.get(path)
        assert secret not in response.content, path
    for member in (".env", ".env.local", ".git/config", ".vscode/settings.json"):
        assert built.read(member) is None, member


def test_the_dev_directory_server_still_serves_the_export(tmp_path):
    """Control: the dev loader must keep working for its actual purpose --
    it wins over the zip precisely so iterating on Studio needs no
    re-vendoring."""
    from tests.fakes.studio_ui import CHAT, INDEX

    _, built = _dev_bundle(tmp_path)
    client = _ui_client(built)
    assert client.get("/").content == INDEX
    assert client.get("/app/chat").content == CHAT
    assert client.get("/app/deep/unknown").content == INDEX   # bounded fallback
    assert built.read("_next/static/chunks/main-deadbeef.js") is not None


def test_an_edited_export_file_is_still_read_fresh_from_disk(tmp_path):
    """Control: enumerating names must not turn into caching bodies. A dev
    directory changes under the daemon -- that is the whole reason it wins
    over the zip."""
    root, built = _dev_bundle(tmp_path)
    (root / "index.html").write_bytes(b"<html>first</html>")
    assert b"first" in built.read("index.html")[0]
    (root / "index.html").write_bytes(b"<html>second</html>")
    assert b"second" in built.read("index.html")[0]


def test_a_file_dropped_into_the_root_after_mount_is_not_published(tmp_path):
    """The trade-off this fix makes, pinned rather than discovered.

    Membership is decided by the walk at mount time, so a file that appears
    afterwards is not served until the daemon restarts. That is the point: the
    alternative -- re-walking on a miss -- would let an unauthenticated caller
    trigger a directory walk by asking for names that do not exist, on the one
    route with no credential requirement.
    """
    root, built = _dev_bundle(tmp_path)
    (root / ".env").write_bytes(b"GEMINI_API_KEY=sk-live-later")
    assert built.read(".env") is None


def test_the_resolved_path_guard_is_still_there(tmp_path):
    """A name-level membership check cannot see a symlink or an 8.3 short
    name; only resolving the real path can. The enumeration is a second
    control, never a replacement for the first."""
    import inspect

    from assistant.io.api.ui import UiBundle

    source = inspect.getsource(UiBundle._read_from_dir)
    assert "resolve()" in source
    assert "is_relative_to" in source

    root, _ = _dev_bundle(tmp_path)
    outside = tmp_path / "devices.json"
    outside.write_text("stolen", encoding="utf-8")
    try:
        (root / "leak.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this machine will not create symlinks")
    # Mounted *after* the symlink exists, so the walk enumerates it and the
    # resolved-path guard is the only thing left refusing it.
    built = UiBundle.open(zip_path=None, dir_path=root)
    assert built.read("leak.json") is None
