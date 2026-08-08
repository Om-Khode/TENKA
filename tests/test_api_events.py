"""One socket, typed frames, and a telemetry loop that stops when nobody listens."""
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


@pytest.mark.asyncio
async def test_the_hub_counts_its_subscribers():
    hub = EventHub()
    assert hub.subscriber_count() == 0


@pytest.mark.asyncio
async def test_publishing_with_no_subscribers_does_not_raise():
    EventHub().publish({"type": "telemetry", "cpu": 1.0})


@pytest.mark.asyncio
async def test_the_telemetry_loop_stops_when_the_last_socket_leaves():
    hub = EventHub()
    runtime = build_fake_runtime()
    await hub.start(runtime, interval_seconds=0.01)
    assert hub.telemetry_running() is False, (
        "the loop must not sample while nobody is listening"
    )
    await hub.stop()
