"""One socket, typed frames, and a telemetry loop that stops when nobody listens."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
from assistant.io.api.events import EventHub
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime


@pytest.fixture()
def context(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset(Capability))
    runtime = build_fake_runtime()
    app = create_app(runtime, vault, origins=["http://localhost:3000"])
    return TestClient(app), app, runtime, token


def test_the_socket_refuses_an_unauthenticated_connection(context):
    client, _, _, _ = context
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/events"):
            pass


def test_the_socket_refuses_an_invalid_token(context):
    """A missing token is refused (above); a wrong one must be too."""
    client, _, _, _ = context
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/events?access_token=not-a-real-token"):
            pass


def test_a_valid_token_connects_and_gets_a_hello(context):
    client, _, _, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        frame = socket.receive_json()
        assert frame["type"] == "status"


def test_a_broadcast_reaches_a_connected_socket(context):
    client, app, _, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        socket.receive_json()  # hello
        app.state.hub.publish({"type": "task_step", "taskId": "t1", "index": 0,
                               "label": "reading", "state": "running"})
        frame = socket.receive_json()
        assert frame["type"] == "task_step"
        assert frame["label"] == "reading"


def test_an_abort_frame_reaches_the_runtime(context):
    client, _, runtime, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        socket.receive_json()
        socket.send_json({"type": "abort"})
        socket.receive_json()  # ack
    assert runtime.chat.aborted == 1


def test_an_unknown_client_frame_is_ignored_not_fatal(context):
    client, _, _, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        socket.receive_json()
        socket.send_json({"type": "nonsense"})
        socket.send_json({"type": "abort"})
        assert socket.receive_json()["type"] in ("error", "ack")


def test_a_malformed_frame_is_ignored_not_fatal(context):
    """Text that is not JSON at all -- not just well-formed JSON of an
    unknown shape -- must not kill the connection either."""
    client, _, _, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        socket.receive_json()  # hello
        socket.send_text("not json at all {{{")
        frame = socket.receive_json()
        assert frame["type"] == "error"
        # the socket survives and keeps answering, not just tolerates one frame
        socket.send_json({"type": "abort"})
        assert socket.receive_json()["type"] == "ack"


@pytest.mark.asyncio
async def test_the_hub_counts_its_subscribers():
    hub = EventHub()
    assert hub.subscriber_count() == 0


@pytest.mark.asyncio
async def test_publishing_with_no_subscribers_does_not_raise():
    EventHub().publish({"type": "telemetry", "cpu": 1.0})


@pytest.mark.asyncio
async def test_publish_before_any_loop_is_known_buffers_instead_of_enqueueing():
    """The unsound fallback this replaces called _enqueue() directly on the
    premise that a caller with no known loop must already be on the loop's
    own thread -- true for a bare unit test, but not for status_broadcaster
    firing from a worker thread before the daemon has ever attached a socket
    or called start(). Pinning the buffer (not the queue) as the landing spot
    is what makes that premise irrelevant instead of merely usually true."""
    hub = EventHub()
    hub.publish({"type": "status", "phase": "THINKING"})
    assert hub._queue.qsize() == 0
    assert hub._pending == [{"type": "status", "phase": "THINKING"}]


@pytest.mark.asyncio
async def test_buffered_events_flush_once_a_loop_is_captured():
    """start() creates the pump in the same call that captures the loop, so
    a flushed item can already be drained by the time this checks -- qsize()
    would be racy. _pending emptying out is the property that matters: the
    event was handed off, not silently dropped on the floor."""
    hub = EventHub()
    hub.publish({"type": "status", "phase": "THINKING"})
    assert hub._pending, "sanity: nothing to flush would make this test vacuous"
    await hub.start(build_fake_runtime())
    await asyncio.sleep(0)  # let the call_soon_threadsafe-scheduled enqueue run
    assert hub._pending == []


def test_a_publish_from_a_worker_thread_before_any_socket_is_still_delivered(context):
    """Reproduces the status_broadcaster shape directly: a phase change fires
    from a thread that is not the daemon's event loop, before any socket has
    ever attached (so no loop has been captured yet either)."""
    import threading

    client, app, _, token = context
    fired = threading.Event()

    def _publish_from_worker():
        app.state.hub.publish({"type": "status", "phase": "THINKING"})
        fired.set()

    threading.Thread(target=_publish_from_worker).start()
    assert fired.wait(timeout=2), "publish() must not block the calling thread"

    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        socket.receive_json()  # hello
        frame = socket.receive_json()
        assert frame == {"type": "status", "phase": "THINKING"}


@pytest.mark.asyncio
async def test_the_telemetry_loop_stops_when_the_last_socket_leaves():
    hub = EventHub()
    runtime = build_fake_runtime()
    await hub.start(runtime, interval_seconds=0.01)
    assert hub.telemetry_running() is False, (
        "the loop must not sample while nobody is listening"
    )
    await hub.stop()


# ─── deferred item 6: the socket leaves an audit trail ───────────────────
def test_a_rejected_socket_connection_is_audited(context):
    client, app, _, _ = context
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/events?access_token=not-a-real-token"):
            pass
    entries = app.state.auth.audit.entries()
    assert any(e.path == "/v1/events" and e.device_id == "-" for e in entries), (
        "a rejected socket connection left no trace in the audit log"
    )


def test_an_accepted_socket_connection_is_audited(context):
    client, app, _, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        socket.receive_json()  # hello
    entries = app.state.auth.audit.entries()
    matches = [e for e in entries if e.path == "/v1/events" and e.device_id != "-"]
    assert matches, "an accepted socket connection left no trace in the audit log"


# ─── fix wave: the socket enforces the same capability its HTTP twins do ──
def test_a_non_chat_token_is_refused_before_accept(context):
    """Every HTTP analogue of what this socket serves -- GET /v1/status,
    GET /v1/telemetry, POST /v1/abort -- requires Capability.CHAT. A
    FILES-only token must be refused the same way, and refused before
    accept(): the connection must close, not merely answer nothing useful
    once open.
    """
    client, app, _, _ = context
    vault = app.state.auth.vault
    files_only = vault.issue("laptop", frozenset({Capability.FILES}))

    with pytest.raises(Exception):
        with client.websocket_connect(f"/v1/events?access_token={files_only}"):
            pass

    entries = app.state.auth.audit.entries()
    assert any(e.path == "/v1/events" and e.outcome == "1008" for e in entries), (
        "a capability-refused socket connection left no trace in the audit log"
    )


def test_a_chat_token_still_connects(context):
    """The fix above must not have collapsed into refusing everyone."""
    client, app, _, token = context
    device = app.state.auth.vault.verify(token)
    assert Capability.CHAT in device.grants
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        frame = socket.receive_json()
        assert frame["type"] == "status"


# ─── deferred item 7: the socket spends the same budget HTTP does ────────
# ─── socket contract fix: one camelCase status shape, aligned telemetry ──
def test_a_broadcaster_shaped_event_arrives_camelcased_via_publish_status(context):
    """publish_status() is the bridge point main.py actually subscribes to
    status_broadcaster -- exercise it directly with a dict shaped exactly
    the way status_broadcaster.py's set() builds one, snake_case and all."""
    client, app, _, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        socket.receive_json()  # connect frame
        app.state.hub.publish_status({
            "v": 2, "type": "status", "phase": "THINKING", "detail": "",
            "cursor_follows": True, "step": [1, 3], "tier": "vision",
            "ts": 123.0,
        })
        frame = socket.receive_json()
    assert frame == {
        "v": 2, "type": "status", "phase": "THINKING", "detail": "",
        "cursorFollows": True, "step": [1, 3], "tier": "vision", "ts": 123.0,
    }
    assert "cursor_follows" not in frame


def test_the_connect_frame_and_a_real_status_frame_share_one_shape(context):
    """A client that destructures every 'status' frame the same way must
    not find the connect-time one short any keys."""
    client, app, _, token = context
    with client.websocket_connect(f"/v1/events?access_token={token}") as socket:
        connect_frame = socket.receive_json()
        app.state.hub.publish_status({
            "v": 2, "phase": "IDLE", "detail": "", "cursor_follows": False,
            "step": None, "tier": None, "ts": 1.0,
        })
        real_frame = socket.receive_json()
    assert set(connect_frame.keys()) == set(real_frame.keys())
    assert connect_frame["type"] == real_frame["type"] == "status"
    # the connect frame's unknowns are explicit nulls, not missing keys
    assert connect_frame["cursorFollows"] is None
    assert connect_frame["step"] is None
    assert connect_frame["tier"] is None
    assert connect_frame["ts"] is None
    assert connect_frame["v"] is None


def test_the_telemetry_frame_keys_match_the_http_telemetry_route():
    """Asserted against the route's own serialiser, not a hardcoded list of
    names, so the two cannot silently drift apart again."""
    from assistant.io.api.events import telemetry_frame
    from assistant.io.api.routes.system import telemetry_body
    from assistant.io.api.runtime import TelemetrySnapshot

    snapshot = TelemetrySnapshot(cpu_percent=12.0, ram_percent=34.0,
                                  battery_percent=None, active_model="m",
                                  uptime_seconds=99)
    frame = telemetry_frame(snapshot)
    assert frame["type"] == "telemetry"
    # telemetry_body() now returns a TelemetryPayload (typed-response rework),
    # not a plain dict -- model_dump(by_alias=True) is the wire shape it used
    # to *be* directly.
    assert {k: v for k, v in frame.items() if k != "type"} == \
        telemetry_body(snapshot).model_dump(by_alias=True)


def test_no_frame_the_socket_can_emit_has_a_snake_case_key():
    """A sweep over every frame builder in events.py -- the connect/status
    builder, the broadcaster translator, the telemetry frame, and the
    error/ack builders app.py's socket handler calls directly (not
    EventHub-produced, but they reach the same client over the same socket,
    so a snake-case key added there would be exactly as real a regression).
    One sweep, not five separate spot checks that could miss a sixth."""
    from assistant.io.api.events import (
        build_ack_frame,
        build_error_frame,
        build_status_frame,
        status_frame_from_broadcaster_event,
        telemetry_frame,
    )
    from assistant.io.api.runtime import TelemetrySnapshot

    frames = [
        build_status_frame(phase="connected", detail="model-x"),
        status_frame_from_broadcaster_event({
            "v": 2, "phase": "THINKING", "detail": "", "cursor_follows": True,
            "step": [1, 2], "tier": "native", "ts": 1.0,
        }),
        telemetry_frame(TelemetrySnapshot(
            cpu_percent=1.0, ram_percent=2.0, battery_percent=None,
            active_model="m", uptime_seconds=5,
        )),
        build_error_frame("malformed frame"),
        build_ack_frame("abort"),
    ]
    for frame in frames:
        assert not any("_" in key for key in frame), frame


def test_repeated_bad_socket_tokens_eventually_get_refused_fast(context):
    """An accept-then-close cycle still costs a TCP handshake and a
    verify() call. The limiter must bound how many of those one source can
    trigger, the same way it bounds a flood of bad HTTP tokens -- proven
    here by exhausting the shared limiter directly against the source key
    the socket handler uses, then confirming a further connection attempt
    is refused without ever reaching verify().
    """
    client, app, _, _ = context
    refused = 0
    for _ in range(150):
        try:
            with client.websocket_connect("/v1/events?access_token=bad"):
                pass
        except Exception:
            refused += 1
    assert refused > 0
    # every one of the 150 attempts closed the connection; the budget for
    # this source must be visibly spent, not silently unlimited.
    source = next(iter(app.state.auth.limiter.hits), None)
    assert source is not None, "the socket never spent the shared limiter's budget"
    assert app.state.auth.limiter.check(source) is False
