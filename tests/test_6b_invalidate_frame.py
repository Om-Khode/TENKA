# tests/test_6b_invalidate_frame.py
"""`"invalidate"` frames: `{"type": "invalidate", "resource": ...}`, and
nothing else.

Three Studio surfaces went stale until a manual reload, all for the same
reason -- nothing pushed a state change to a socket that did not cause it.
The fix is one generic frame naming a resource, never carrying data about it,
so the client refetches through its normal authenticated route (which already
enforces every capability that route requires).

Fanout has two shapes, and both are pinned here against *false positives*
(a socket that should not receive one, connected at the same time as one that
should) rather than only against the happy path -- a fanout test that never
proves the wrong recipient stayed silent is exactly the vacuous-security-test
shape `CLAUDE.md` names, so every negative below is checked directly, in the
same test as the positive, never inferred from the positive alone.

Bounded waits throughout (`_receive_bounded`): a broken fanout must fail this
suite, not hang it.
"""
from __future__ import annotations

import asyncio
import queue as queue_module
import threading

import pytest

from assistant.io.api.app import create_app
from assistant.io.api.events import (
    EventHub,
    build_invalidate_frame,
    notify_invalidate,
)
from assistant.io.api.policy import ListenerPolicy
from assistant.io.api.raises import RaiseStore
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import BASE_URL, LOCAL_PORT, ApiTestClient, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime
from tests.test_6b_transport_routes import A_REAL_TRANSPORT, FakeTransportManager


def _receive_bounded(socket, timeout: float = 2.0):
    """`socket.receive_json()`, bounded.

    `WebSocketTestSession.receive()` blocks the calling thread on an anyio
    portal call with no timeout of its own -- fine while a frame is always
    expected, but a broken fanout under test is exactly a case where one
    never arrives. A daemon thread (never joined, never part of a `with`
    that would block process/test teardown) races a queue read with
    `timeout`, so a hang becomes a clean `TimeoutError` instead.
    """
    box: queue_module.Queue = queue_module.Queue(maxsize=1)

    def _run() -> None:
        try:
            box.put(("ok", socket.receive_json()))
        except Exception as exc:  # pragma: no cover - defensive
            box.put(("err", exc))

    threading.Thread(target=_run, daemon=True).start()
    try:
        kind, value = box.get(timeout=timeout)
    except queue_module.Empty:
        raise TimeoutError(f"no frame arrived within {timeout}s") from None
    if kind == "err":
        raise value
    return value


# ─── the wire shape ─────────────────────────────────────────────────────────
def test_build_invalidate_frame_carries_only_type_and_resource():
    frame = build_invalidate_frame("devices")
    assert frame == {"type": "invalidate", "resource": "devices"}
    assert set(frame) == {"type", "resource"}


# ─── classification is mandatory, and fails closed ──────────────────────────
@pytest.mark.asyncio
async def test_a_device_scoped_resource_with_no_device_id_is_dropped_not_broadcast(caplog):
    """`"session"` is device-scoped. A call site that forgot `device_id` must
    not fall back to telling every connected socket."""
    hub = EventHub()
    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload):
            sent.append(payload)

        async def close(self, code=1000):
            pass

    await hub.attach(_Socket(), lambda: frozenset(Capability), device_id="d1")
    hub.publish_invalidate("session")  # no device_id
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert sent == [], "a device-scoped resource broadcast with no target"


@pytest.mark.asyncio
async def test_an_unclassified_resource_is_dropped_not_broadcast():
    """A resource in neither table is refused, not delivered unfiltered --
    the guard against a future call site that forgets to classify one."""
    hub = EventHub()
    sent: list[dict] = []

    class _Socket:
        async def send_json(self, payload):
            sent.append(payload)

        async def close(self, code=1000):
            pass

    await hub.attach(_Socket(), lambda: frozenset(Capability), device_id="d1")
    hub.publish_invalidate("something-nobody-registered")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert sent == []


# ─── device-scoped fanout ───────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_device_scoped_delivery_reaches_only_the_named_device():
    """The negative is pinned directly: a second, different device connected
    at the same time must receive nothing -- and is proven to actually be
    listening by a follow-up broadcast frame it *does* receive, so "receives
    nothing" cannot pass because the socket was never live."""
    hub = EventHub()

    class _Socket:
        def __init__(self):
            self.sent: list[dict] = []

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self, code=1000):
            pass

    target, bystander = _Socket(), _Socket()
    await hub.attach(target, lambda: frozenset(Capability), device_id="target")
    await hub.attach(bystander, lambda: frozenset(Capability), device_id="bystander")

    hub.publish_invalidate("session", device_id="target")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert target.sent == [{"type": "invalidate", "resource": "session"}]
    assert bystander.sent == []

    hub.publish({"type": "marker"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert bystander.sent == [{"type": "marker"}], (
        "the bystander never actually received anything -- the negative "
        "above would have passed vacuously")


# ─── collection-scoped fanout: both halves of the gate ─────────────────────
@pytest.mark.asyncio
async def test_collection_fanout_requires_both_the_capability_and_the_admin_listener():
    """`admin_capability_satisfied` -- shared with `require_admin` -- is
    `capability in grants AND policy.admin`. Both conjuncts are exercised
    here as the only difference between two otherwise-identical sockets:
    - `no_capability` sits on the admin listener but lacks SYSTEM_CONTROL;
    - `wrong_listener` holds SYSTEM_CONTROL but sits on a listener that is
      not `policy.admin` (mirrors KI-18: a ceiling carrying the capability
      is not the same thing as an admin-trusted transport).
    Neither may receive a `"devices"` invalidate frame; both are proven live
    via the same marker trick as the device-scoped test above.
    """
    admin_policy = ListenerPolicy(
        name="local", admin=True, allow_bearer=True, secure_cookie=False,
        ceiling=frozenset(Capability), raisable=frozenset())
    non_admin_with_capability = ListenerPolicy(
        name="probe", admin=False, allow_bearer=False, secure_cookie=False,
        ceiling=frozenset({Capability.OBSERVE, Capability.SYSTEM_CONTROL}),
        raisable=frozenset())

    hub = EventHub()

    class _Socket:
        def __init__(self):
            self.sent: list[dict] = []

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self, code=1000):
            pass

    admin, no_capability, wrong_listener = _Socket(), _Socket(), _Socket()
    await hub.attach(admin, lambda: frozenset(Capability),
                     device_id="admin-dev", policy=admin_policy)
    await hub.attach(no_capability, lambda: frozenset({Capability.OBSERVE}),
                     device_id="watcher-dev", policy=admin_policy)
    await hub.attach(wrong_listener, lambda: frozenset({Capability.OBSERVE, Capability.SYSTEM_CONTROL}),
                     device_id="probe-dev", policy=non_admin_with_capability)

    hub.publish_invalidate("devices")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert admin.sent == [{"type": "invalidate", "resource": "devices"}]
    assert no_capability.sent == []
    assert wrong_listener.sent == []

    hub.publish({"type": "marker"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert no_capability.sent == [{"type": "marker"}], "no_capability was never actually live"
    assert wrong_listener.sent == [{"type": "marker"}], "wrong_listener was never actually live"


# ─── a hostile socket cannot deny delivery to a healthy one ────────────────
@pytest.mark.asyncio
async def test_a_socket_that_raises_on_send_does_not_stop_delivery_to_others():
    """`_pump`'s per-socket `try/except` is what makes one hostile or dead
    connection survivable for every *other* socket sharing the fan-out loop
    -- without it, an exception raised mid-iteration ends the pump task for
    the rest of this hub's life, and nothing (invalidate frames included)
    ever reaches anyone again."""
    hub = EventHub()

    class _HostileSocket:
        async def send_json(self, payload):
            raise RuntimeError("simulated dead socket")

        async def close(self, code=1000):
            raise RuntimeError("simulated dead socket, twice over")

    class _HealthySocket:
        def __init__(self):
            self.sent: list[dict] = []

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self, code=1000):
            pass

    healthy = _HealthySocket()
    # Attached in this order deliberately: `_sockets` is iterated in
    # insertion order, so the hostile one is the one `_pump` reaches
    # *first* -- the shape that actually tests the guard, since a hostile
    # socket iterated last would prove nothing about the others.
    await hub.attach(_HostileSocket(), lambda: frozenset(Capability), device_id="hostile")
    await hub.attach(healthy, lambda: frozenset(Capability), device_id="healthy")

    # A plain broadcast, not `"invalidate"` -- neither socket was attached
    # with a `policy`, and an invalidate resource requires one to pass the
    # collection gate; a marker sidesteps that entirely so this test is
    # purely about fan-out survival, not classification. Both sockets must
    # be in the same fan-out pass for the hostile one's failure to be able
    # to affect the healthy one at all.
    hub.publish({"type": "marker"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert healthy.sent == [{"type": "marker"}]


# ─── the route-facing wrapper never raises ─────────────────────────────────
def test_notify_invalidate_is_a_noop_with_no_hub():
    notify_invalidate(None, "devices")  # must not raise


def test_notify_invalidate_swallows_a_raising_hub(monkeypatch):
    hub = EventHub()

    def _boom(resource, *, device_id=None):
        raise RuntimeError("simulated failure inside the hub")

    monkeypatch.setattr(hub, "publish_invalidate", _boom)
    notify_invalidate(hub, "devices")  # must not raise


# ─── end-to-end: the real routes, the real hub, over real sockets ─────────
#
# One `TestClient`, entered via `with client:` so it holds exactly one
# ASGI portal (one background thread, one event loop) for its whole life.
# Left un-entered, `TestClient._portal_factory()` spins up a *fresh*,
# throwaway portal for every single `websocket_connect()`/HTTP call
# (`self.portal is None` -> its `else` branch) -- fine for one socket at a
# time, which is all the pre-existing suite ever opens, but fatal for two
# *concurrent* sockets on the same `EventHub`: the hub captures `_loop` from
# whichever socket attaches first (`_set_loop`, `events.py`) and never moves
# it, so a second socket living on a different portal's loop would have
# `_pump` awaiting `send_json()` on a WebSocket object that belongs to a
# foreign event loop entirely -- not a fanout bug, a test-harness one. One
# shared client/portal, with the device identity switched via `_as()`
# between connects (each socket reads the cookie jar once, at its own
# connect time, so this is safe), is what makes concurrent sockets on one
# hub actually representative of one daemon serving many devices on one
# loop -- which is what production is.
def _as(client: ApiTestClient, token: str) -> ApiTestClient:
    client.cookies.set(COOKIE_NAME, token)
    return client


def test_pairing_a_device_notifies_only_admin_viewers_of_the_devices_collection(tmp_path):
    vault = TokenVault(tmp_path)
    admin_token = vault.issue("laptop", frozenset(Capability))
    watcher_token = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    client = build_api_client(build_fake_runtime(), vault, policies={LOCAL_PORT: "local"})

    with client:
        with _as(client, admin_token).websocket_connect("/v1/events") as admin_socket:
            with _as(client, watcher_token).websocket_connect("/v1/events") as watcher_socket:
                _receive_bounded(admin_socket)  # hello
                _receive_bounded(watcher_socket)  # hello

                mint = _as(client, admin_token).post(
                    "/v1/pair/code", json={"label": "phone", "grants": ["observe"]},
                    headers={CSRF_HEADER: "1"})
                assert mint.status_code == 200, mint.text
                code = mint.json()["data"]["code"]

                # Redemption is the one unauthenticated write in the API --
                # no cookie needed, and its 204 sets a brand new one for the
                # device that just paired, which this test has no further
                # use for.
                client.cookies.delete(COOKIE_NAME)
                assert client.post("/v1/pair", json={"code": code}).status_code == 204

                frame = _receive_bounded(admin_socket)
                assert frame == {"type": "invalidate", "resource": "devices"}

                client.app.state.hub.publish({"type": "marker"})
                assert _receive_bounded(watcher_socket) == {"type": "marker"}, (
                    "the watcher was never actually live -- the negative "
                    "above would have passed vacuously")


def test_revoking_a_device_notifies_only_admin_viewers_of_the_devices_collection(tmp_path):
    vault = TokenVault(tmp_path)
    admin_token = vault.issue("laptop", frozenset(Capability))
    watcher_token = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    phone_id = [d for d in vault.devices() if d.label == "phone"][0].device_id
    client = build_api_client(build_fake_runtime(), vault, policies={LOCAL_PORT: "local"})

    with client:
        with _as(client, admin_token).websocket_connect("/v1/events") as admin_socket:
            with _as(client, watcher_token).websocket_connect("/v1/events") as watcher_socket:
                _receive_bounded(admin_socket)
                _receive_bounded(watcher_socket)

                r = _as(client, admin_token).delete(
                    f"/v1/devices/{phone_id}", headers={CSRF_HEADER: "1"})
                assert r.status_code == 200, r.text

                assert _receive_bounded(admin_socket) == {
                    "type": "invalidate", "resource": "devices"}

                client.app.state.hub.publish({"type": "marker"})
                assert _receive_bounded(watcher_socket) == {"type": "marker"}


def _raise_body(transport: str = "tailnet") -> dict:
    return {"transport": transport, "capabilities": ["execute"], "minutes": 5,
            "reason": "test"}


def test_raising_a_device_notifies_only_that_device_and_the_admin_devices_view(tmp_path):
    vault = TokenVault(tmp_path)
    admin_token = vault.issue("laptop", frozenset(Capability))
    target_token = vault.issue("phone", frozenset({Capability.OBSERVE, Capability.EXECUTE}))
    bystander_token = vault.issue("tablet", frozenset({Capability.OBSERVE}))
    target_id = [d for d in vault.devices() if d.label == "phone"][0].device_id

    client = build_api_client(build_fake_runtime(), vault, policies={LOCAL_PORT: "local"})
    client.app.state.raises = RaiseStore()
    client.app.state.transports = FakeTransportManager(running={"tailnet": "https://t.example"})

    with client:
        with _as(client, admin_token).websocket_connect("/v1/events") as admin_socket:
            with _as(client, target_token).websocket_connect("/v1/events") as target_socket:
                with _as(client, bystander_token).websocket_connect("/v1/events") as bystander_socket:
                    _receive_bounded(admin_socket)
                    _receive_bounded(target_socket)
                    _receive_bounded(bystander_socket)

                    r = _as(client, admin_token).post(
                        f"/v1/devices/{target_id}/raise", json=_raise_body(),
                        headers={CSRF_HEADER: "1"})
                    assert r.status_code == 200, r.text

                    assert _receive_bounded(target_socket) == {
                        "type": "invalidate", "resource": "session"}
                    assert _receive_bounded(admin_socket) == {
                        "type": "invalidate", "resource": "devices"}

                    client.app.state.hub.publish({"type": "marker"})
                    assert _receive_bounded(bystander_socket) == {"type": "marker"}


def test_revoking_a_raise_notifies_only_that_device_and_the_admin_devices_view(tmp_path):
    vault = TokenVault(tmp_path)
    admin_token = vault.issue("laptop", frozenset(Capability))
    target_token = vault.issue("phone", frozenset({Capability.OBSERVE, Capability.EXECUTE}))
    bystander_token = vault.issue("tablet", frozenset({Capability.OBSERVE}))
    target_id = [d for d in vault.devices() if d.label == "phone"][0].device_id

    client = build_api_client(build_fake_runtime(), vault, policies={LOCAL_PORT: "local"})
    client.app.state.raises = RaiseStore()
    client.app.state.transports = FakeTransportManager(running={"tailnet": "https://t.example"})

    with client:
        assert _as(client, admin_token).post(
            f"/v1/devices/{target_id}/raise", json=_raise_body(),
            headers={CSRF_HEADER: "1"}).status_code == 200

        with _as(client, admin_token).websocket_connect("/v1/events") as admin_socket:
            with _as(client, target_token).websocket_connect("/v1/events") as target_socket:
                with _as(client, bystander_token).websocket_connect("/v1/events") as bystander_socket:
                    _receive_bounded(admin_socket)
                    _receive_bounded(target_socket)
                    _receive_bounded(bystander_socket)

                    r = _as(client, admin_token).delete(
                        f"/v1/devices/{target_id}/raise", headers={CSRF_HEADER: "1"})
                    assert r.status_code == 200, r.text

                    assert _receive_bounded(target_socket) == {
                        "type": "invalidate", "resource": "session"}
                    assert _receive_bounded(admin_socket) == {
                        "type": "invalidate", "resource": "devices"}

                    client.app.state.hub.publish({"type": "marker"})
                    assert _receive_bounded(bystander_socket) == {"type": "marker"}


def test_starting_and_stopping_a_transport_notifies_only_admin_viewers(tmp_path):
    vault = TokenVault(tmp_path)
    admin_token = vault.issue("laptop", frozenset(Capability))
    watcher_token = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    client = build_api_client(build_fake_runtime(), vault, policies={LOCAL_PORT: "local"})
    client.app.state.transports = FakeTransportManager()

    with client:
        with _as(client, admin_token).websocket_connect("/v1/events") as admin_socket:
            with _as(client, watcher_token).websocket_connect("/v1/events") as watcher_socket:
                _receive_bounded(admin_socket)
                _receive_bounded(watcher_socket)

                start = _as(client, admin_token).post(
                    f"/v1/transports/{A_REAL_TRANSPORT}", headers={CSRF_HEADER: "1"})
                assert start.status_code == 200, start.text
                assert _receive_bounded(admin_socket) == {
                    "type": "invalidate", "resource": "transports"}
                # The marker is a plain broadcast -- it reaches admin_socket
                # too, not just watcher_socket, so it must be drained off
                # both queues before the next real frame is expected.
                client.app.state.hub.publish({"type": "marker", "seq": 1})
                assert _receive_bounded(watcher_socket) == {"type": "marker", "seq": 1}
                assert _receive_bounded(admin_socket) == {"type": "marker", "seq": 1}

                stop = _as(client, admin_token).delete(
                    f"/v1/transports/{A_REAL_TRANSPORT}", headers={CSRF_HEADER: "1"})
                assert stop.status_code == 200, stop.text
                assert _receive_bounded(admin_socket) == {
                    "type": "invalidate", "resource": "transports"}
                client.app.state.hub.publish({"type": "marker", "seq": 2})
                assert _receive_bounded(watcher_socket) == {"type": "marker", "seq": 2}


# ─── existing frame shapes are unchanged ────────────────────────────────────
def test_the_invalidate_frame_does_not_join_the_status_frame_shape():
    """`test_the_connect_frame_and_a_real_status_frame_share_one_shape`
    (test_api_events.py) pins that every `"status"` frame carries one
    identical key set. This frame sits beside it, not inside it."""
    from assistant.io.api.events import build_status_frame

    status_keys = set(build_status_frame(phase="connected"))
    invalidate_keys = set(build_invalidate_frame("devices"))
    assert invalidate_keys != status_keys
    assert invalidate_keys == {"type", "resource"}
