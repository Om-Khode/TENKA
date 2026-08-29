"""
test_extension_ws.py — Drover Task 11: the socket, its handshake, and its calls.

The security-relevant part of a handshake is the *order* of its checks, and a
test that has to stand up a real WebSocket to assert an ordering gets written
once and then quietly not extended. So `evaluate_handshake` is a pure function
and this file drives it directly.

What is pinned, and the failure each describes:

  - **Origin is examined before the token is validated.** A client that is not a
    browser extension must learn nothing about whether the token it guessed was
    close. Without this ordering the socket is a token oracle for anything on
    the machine.
  - **A digest mismatch refuses.** Two copies of `dom_query.js` exist because
    MV3 forbids sending it over the wire; comparing them is the only thing that
    makes drift loud instead of silent.
  - **No minted credential refuses everything**, rather than accepting anything.
  - **A second client is refused and the first keeps serving.** Letting a
    newcomer displace the incumbent means anything that can reach the port can
    take over an in-flight task, and the browser goes quiet with no error.
  - **A timed-out call releases its pending slot**, and a disconnect fails every
    waiter rather than leaving it to hang.

Run: py -3.11 -m pytest tests/test_extension_ws.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from assistant.core import drover_protocol as proto  # noqa: E402
from assistant.io.api import extension_ws as ws  # noqa: E402

GOOD_DIGEST = "a" * 64
GOOD_TOKEN = "tok-correct-horse"
GOOD_ORIGIN = "chrome-extension://abcdefghijklmnop"


def hello(**overrides):
    frame = {
        "type": proto.Frame.HELLO,
        "protocolVersion": proto.PROTOCOL_VERSION,
        "domQuerySha256": GOOD_DIGEST,
        "token": GOOD_TOKEN,
        "browser": "firefox",
        "extensionVersion": "0.1.0",
    }
    frame.update(overrides)
    return frame


#: `None` is a frame a client can really send, so it cannot double as "use the
#: default" — the two meanings collided and made one case pass vacuously.
_DEFAULT = object()


def evaluate(frame=_DEFAULT, *, origin=GOOD_ORIGIN, token=GOOD_TOKEN,
             digest=GOOD_DIGEST, occupied=False):
    return ws.evaluate_handshake(
        hello() if frame is _DEFAULT else frame,
        origin=origin, expected_token=token,
        expected_digest=digest, occupied=occupied,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    ws.reset_state_for_test()
    yield
    ws.reset_state_for_test()


# ─── The permitted path, first ───────────────────────────────────────────


def test_a_good_handshake_is_accepted():
    # Stated before every refusal below. A handshake that refused everything
    # would satisfy all of them and ship a driver that never connects.
    assert evaluate().ok is True


@pytest.mark.parametrize("origin", [
    "chrome-extension://abcdefghijklmnop",
    "moz-extension://11111111-2222-3333-4444-555555555555",
])
def test_both_extension_schemes_are_accepted(origin):
    assert evaluate(origin=origin).ok is True


# ─── Origin ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("origin", [
    None, "", "http://localhost:3000", "https://example.invalid",
    "file://", "null", "ws://127.0.0.1:8790",
    "chrome-extension",                     # scheme-ish, not a scheme
    "https://chrome-extension://spoof",      # prefix buried, not leading
])
def test_a_non_extension_origin_is_refused(origin):
    verdict = evaluate(origin=origin)
    assert verdict.ok is False
    assert verdict.code == proto.Err.UNAUTHORIZED


def test_origin_is_judged_before_the_token_is_validated():
    """The oracle property.

    A caller with the wrong Origin must get the same answer whether its token
    was right or wrong. If the token were checked first, anything on the machine
    could use this socket to test guesses.
    """
    with_good_token = evaluate(origin="https://example.invalid")
    with_bad_token = evaluate(hello(token="wrong"), origin="https://example.invalid")
    assert with_good_token == with_bad_token, (
        "the refusal differs depending on whether the token was correct, so a "
        "non-extension client can use this socket to test guesses"
    )
    assert "origin" in with_good_token.reason.lower()


# ─── Version and digest ──────────────────────────────────────────────────


def test_a_protocol_mismatch_refuses_with_its_own_code():
    verdict = evaluate(hello(protocolVersion=proto.PROTOCOL_VERSION + 1))
    assert verdict.ok is False
    assert verdict.code == proto.Err.PROTOCOL_MISMATCH


def test_a_missing_protocol_version_refuses():
    frame = hello()
    del frame["protocolVersion"]
    assert evaluate(frame).code == proto.Err.PROTOCOL_MISMATCH


def test_a_digest_mismatch_refuses_with_its_own_code():
    verdict = evaluate(hello(domQuerySha256="b" * 64))
    assert verdict.ok is False
    assert verdict.code == proto.Err.HASH_MISMATCH


def test_the_digest_refusal_names_both_sides():
    # An operator seeing only "mismatch" cannot tell which copy drifted.
    reason = evaluate(hello(domQuerySha256="b" * 64)).reason
    assert "b" * 64 in reason
    assert GOOD_DIGEST in reason


def test_a_missing_digest_refuses_rather_than_skipping_the_check():
    frame = hello()
    del frame["domQuerySha256"]
    assert evaluate(frame).code == proto.Err.HASH_MISMATCH


# ─── Token ───────────────────────────────────────────────────────────────


def test_a_bad_token_refuses():
    assert evaluate(hello(token="nope")).code == proto.Err.UNAUTHORIZED


def test_no_minted_credential_refuses_everything():
    # Fail closed. The absence of a decision is never a decision to allow.
    verdict = evaluate(token=None)
    assert verdict.ok is False
    assert verdict.code == proto.Err.UNAUTHORIZED
    verdict_empty = evaluate(token="")
    assert verdict_empty.ok is False


def test_a_missing_token_field_refuses():
    frame = hello()
    del frame["token"]
    assert evaluate(frame).ok is False


# ─── First frame ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("frame", [
    {"type": proto.Frame.RESPONSE, "id": 1, "ok": True},
    {"type": "nonsense"},
    {}, [], "hello", None, 7,
])
def test_a_first_frame_that_is_not_hello_refuses(frame):
    assert evaluate(frame).ok is False


# ─── One client ──────────────────────────────────────────────────────────


def test_a_second_client_is_refused():
    assert evaluate(occupied=True).ok is False


def test_the_incumbent_is_not_displaced():
    """The half that matters.

    Asserting the second connection was refused says nothing about whether the
    first survived it. A registry that refused the newcomer *and* tore down the
    incumbent would pass the test above and leave the operator's browser
    disconnected.
    """
    first = ws.DroverConnection(send_json=_noop, browser_name="firefox")
    ws.register(first)
    assert ws.is_occupied() is True

    verdict = evaluate(occupied=ws.is_occupied())
    assert verdict.ok is False

    assert ws.current_connection() is first, "the incumbent was replaced"
    assert first.connected is True, "the incumbent was closed by a refused newcomer"


async def _noop(_frame):
    return None


# ─── The registry and the snapshot ───────────────────────────────────────


def test_the_snapshot_reports_disconnected_when_nothing_is_connected():
    snap = ws.drover_state_snapshot()
    assert snap.connected is False
    assert snap.browser_name == ""


def test_the_snapshot_reports_the_connected_browser():
    conn = ws.DroverConnection(send_json=_noop, browser_name="brave", extension_version="0.1.0")
    ws.register(conn)
    snap = ws.drover_state_snapshot()
    assert snap.connected is True
    assert snap.browser_name == "brave"
    assert snap.extension_version == "0.1.0"


def test_unregistering_closes_and_clears():
    conn = ws.DroverConnection(send_json=_noop)
    ws.register(conn)
    ws.unregister(conn, "test")
    assert ws.current_connection() is None
    assert ws.drover_state_snapshot().connected is False


# ─── Calls ───────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def test_a_call_sends_a_request_and_resolves_on_its_reply():
    sent: list[dict] = []

    async def send(frame):
        sent.append(frame)

    async def scenario():
        conn = ws.DroverConnection(send_json=send)
        task = asyncio.create_task(conn.call(proto.Rpc.TABS_LIST, {}))
        await asyncio.sleep(0)
        conn.handle_frame({
            "type": proto.Frame.RESPONSE, "id": sent[0]["id"],
            "ok": True, "result": {"tabs": []},
        })
        return await asyncio.wait_for(task, timeout=1)

    result = _run(scenario())
    assert result == {"tabs": []}
    assert sent[0]["method"] == proto.Rpc.TABS_LIST


def test_an_error_frame_raises_with_its_code_and_never_a_sentinel():
    """`.claude/rules/automation.md` records what a sentinel cost once:
    `router._execute_dom_task` turned an abort into `"__FALLBACK__"`, which is
    not a failure report but an instruction to escalate a tier."""
    sent: list[dict] = []

    async def send(frame):
        sent.append(frame)

    async def scenario():
        conn = ws.DroverConnection(send_json=send)
        task = asyncio.create_task(conn.call(proto.Rpc.ACT, {"idx": 3}))
        await asyncio.sleep(0)
        conn.handle_frame({
            "type": proto.Frame.RESPONSE, "id": sent[0]["id"],
            "ok": False, "code": proto.Err.BAD_SELECTOR, "message": "no element idx=3",
        })
        return await asyncio.wait_for(task, timeout=1)

    with pytest.raises(ws.DroverCallError) as excinfo:
        _run(scenario())
    assert excinfo.value.code == proto.Err.BAD_SELECTOR
    assert "no element idx=3" in str(excinfo.value)


def test_a_timed_out_call_raises_and_releases_its_slot():
    async def send(_frame):
        return None

    async def scenario():
        conn = ws.DroverConnection(send_json=send)
        for _ in range(50):
            with pytest.raises(TimeoutError):
                await conn.call(proto.Rpc.INFO, {}, timeout=0.001)
        return len(conn._pending)

    leaked = _run(scenario())
    assert leaked == 0, (
        f"{leaked} pending slots survived their timeouts. Fifty here stands in "
        f"for a long session; the leak is unbounded either way."
    )


def test_a_disconnect_fails_every_waiter_rather_than_hanging():
    sent: list[dict] = []

    async def send(frame):
        sent.append(frame)

    async def scenario():
        conn = ws.DroverConnection(send_json=send)
        task = asyncio.create_task(conn.call(proto.Rpc.QUERY, {}, timeout=5))
        await asyncio.sleep(0)
        conn.close("socket dropped")
        return await asyncio.wait_for(task, timeout=1)

    with pytest.raises(ws.DroverDisconnected):
        _run(scenario())


def test_calling_a_closed_connection_raises_immediately():
    conn = ws.DroverConnection(send_json=_noop)
    conn.close("gone")
    with pytest.raises(ws.DroverDisconnected):
        _run(conn.call(proto.Rpc.INFO, {}))


def test_events_reach_subscribers_and_a_throwing_one_does_not_break_the_rest():
    seen: list[str] = []
    conn = ws.DroverConnection(send_json=_noop)
    conn.on_event(lambda _f: (_ for _ in ()).throw(RuntimeError("bad subscriber")))
    conn.on_event(lambda f: seen.append(f["event"]))

    conn.handle_frame({"type": proto.Frame.EVENT, "event": "navigated", "tabId": 1})

    assert seen == ["navigated"], (
        "one throwing subscriber stopped the others. A misbehaving consumer "
        "must not take the transport's event path down with it."
    )


def test_a_removed_callback_stops_receiving():
    seen: list[str] = []

    def cb(frame):
        seen.append(frame["event"])

    conn = ws.DroverConnection(send_json=_noop)
    conn.on_event(cb)
    conn.handle_frame({"type": proto.Frame.EVENT, "event": "tab_opened"})
    conn.remove_event_callback(cb)
    conn.handle_frame({"type": proto.Frame.EVENT, "event": "tab_closed"})
    assert seen == ["tab_opened"], "a detached callback kept firing"


def test_a_reply_for_an_unknown_id_is_ignored_not_fatal():
    conn = ws.DroverConnection(send_json=_noop)
    conn.handle_frame({"type": proto.Frame.RESPONSE, "id": 999, "ok": True, "result": {}})


# ─── The token store ─────────────────────────────────────────────────────


def test_mint_then_read_round_trips(tmp_path):
    minted = ws.mint_token(tmp_path)
    assert ws.read_token(tmp_path) == minted
    assert len(minted) >= 32


def test_reading_an_absent_store_is_none_not_an_error(tmp_path):
    assert ws.read_token(tmp_path / "nothing-here") is None


def test_a_corrupt_store_reads_as_none(tmp_path):
    ws.token_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    ws.token_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert ws.read_token(tmp_path) is None


def test_an_older_schema_version_is_rejected_not_migrated(tmp_path):
    path = ws.token_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 0, "token": "old"}), encoding="utf-8")
    assert ws.read_token(tmp_path) is None, (
        "a marker from an older schema was accepted. Standing project rule: "
        "schema-version every on-disk marker and reject older ones."
    )


def test_minting_twice_replaces_rather_than_appends(tmp_path):
    first = ws.mint_token(tmp_path)
    second = ws.mint_token(tmp_path)
    assert first != second
    assert ws.read_token(tmp_path) == second


def test_clearing_removes_the_credential(tmp_path):
    ws.mint_token(tmp_path)
    assert ws.clear_token(tmp_path) is True
    assert ws.read_token(tmp_path) is None
    assert ws.clear_token(tmp_path) is False


# ─── The route, on a real app ────────────────────────────────────────────


def _app_with(tmp_path, digest: str, listeners: dict[int, str]):
    """An app on real listener ports.

    Built through `ApiTestClient`, not the stock `TestClient`: Starlette's
    `websocket_connect` hard-codes `ws://testserver`, so a socket opened through
    the stock client arrives on port 80 with a Host nothing allows — the policy
    resolves to `None` and every connection is refused for a reason that has
    nothing to do with the code under test.
    """
    from assistant.io.api.app import create_app
    from assistant.io.api.vault import TokenVault
    from tests.fakes.studio_runtime import build_fake_runtime

    vault = TokenVault(tmp_path / "vault")
    return create_app(
        build_fake_runtime(), vault,
        origins=["http://localhost:3000"],
        listener_policies=listeners,
        extension_digest=digest,
    )


def test_the_socket_is_refused_on_every_other_listener(tmp_path, monkeypatch):
    """`/drover` answers on the extension listener and nowhere else.

    The failure this prevents is the worst one available here: serving a driver
    for the user's browser on `local`, whose policy grants EXECUTE and admin.
    A socket registered on one app is reachable on every port that app is bound
    to, so "which listener" is a decision this route has to make for itself —
    nothing upstream makes it.
    """
    from tests.fakes.api_client import ApiTestClient
    from assistant.io.api import extension_ws as ews

    monkeypatch.setattr(ews, "read_token", lambda *a, **k: GOOD_TOKEN)
    listeners = {8787: "local", 8788: "tailnet", 8789: "funnel", 8790: "extension"}
    app = _app_with(tmp_path, GOOD_DIGEST, listeners)

    refused_on = []
    for port, name in listeners.items():
        client = ApiTestClient(app, base_url=f"http://127.0.0.1:{port}")
        try:
            with client.websocket_connect(
                "/drover", headers={"origin": GOOD_ORIGIN}
            ) as sock:
                sock.send_text(json.dumps(hello()))
                reply = json.loads(sock.receive_text())
                if reply.get("type") != proto.Frame.WELCOME:
                    refused_on.append(name)
        except Exception:
            refused_on.append(name)

    assert sorted(refused_on) == ["funnel", "local", "tailnet"], (
        f"expected the socket to be refused on every listener but `extension`; "
        f"it was refused on {sorted(refused_on)}. A driver for the user's "
        f"browser answering on `local` sits behind a policy that grants EXECUTE."
    )


def test_the_socket_accepts_on_its_own_listener(tmp_path, monkeypatch):
    # The permitted path, stated separately. A route that refused everywhere
    # would satisfy the test above and ship a tier that never connects.
    from tests.fakes.api_client import ApiTestClient
    from assistant.io.api import extension_ws as ews

    monkeypatch.setattr(ews, "read_token", lambda *a, **k: GOOD_TOKEN)
    app = _app_with(tmp_path, GOOD_DIGEST, {8790: "extension"})

    client = ApiTestClient(app, base_url="http://127.0.0.1:8790")
    with client.websocket_connect("/drover", headers={"origin": GOOD_ORIGIN}) as sock:
        sock.send_text(json.dumps(hello()))
        assert json.loads(sock.receive_text())["type"] == proto.Frame.WELCOME


def test_a_refusal_says_which_check_failed(tmp_path, monkeypatch):
    """The reject frame carries a code, not just a closed socket.

    A bare TCP close cannot say which of the four checks failed, so the
    extension would back off and retry an unfixable state forever — a protocol
    mismatch is not something waiting fixes.
    """
    from tests.fakes.api_client import ApiTestClient
    from assistant.io.api import extension_ws as ews

    monkeypatch.setattr(ews, "read_token", lambda *a, **k: GOOD_TOKEN)
    app = _app_with(tmp_path, GOOD_DIGEST, {8790: "extension"})

    client = ApiTestClient(app, base_url="http://127.0.0.1:8790")
    with client.websocket_connect("/drover", headers={"origin": GOOD_ORIGIN}) as sock:
        sock.send_text(json.dumps(hello(protocolVersion=99)))
        reply = json.loads(sock.receive_text())
    assert reply["type"] == proto.Frame.REJECT
    assert reply["code"] == proto.Err.PROTOCOL_MISMATCH


def test_the_slot_is_freed_when_the_socket_closes(tmp_path, monkeypatch):
    """A connection left registered after its socket died makes `is_occupied()`
    refuse the extension's own reconnect, and nothing ever clears it — the
    extension retries on its alarm forever against a slot held by a ghost."""
    from tests.fakes.api_client import ApiTestClient
    from assistant.io.api import extension_ws as ews

    monkeypatch.setattr(ews, "read_token", lambda *a, **k: GOOD_TOKEN)
    app = _app_with(tmp_path, GOOD_DIGEST, {8790: "extension"})
    client = ApiTestClient(app, base_url="http://127.0.0.1:8790")

    for _ in range(3):
        with client.websocket_connect("/drover", headers={"origin": GOOD_ORIGIN}) as sock:
            sock.send_text(json.dumps(hello()))
            assert json.loads(sock.receive_text())["type"] == proto.Frame.WELCOME
        assert ws.current_connection() is None, "the slot survived the socket"


def test_every_exit_path_from_the_socket_loop_frees_the_slot():
    """The teardown must run whatever ends the loop -- including a
    `BaseException`.

    `asyncio.CancelledError` inherits from `BaseException`, not `Exception`. The
    first version of this handler bound its `reason` in an `except Exception`
    and an `else`, so a cancellation matched neither: the `finally` read a name
    that was never bound and raised `UnboundLocalError` on the way out. The
    connection was then never unregistered, `is_occupied()` stayed true against
    a dead socket, and the extension's own reconnect was refused for as long as
    the process lived.

    Read from the source because the alternative is cancelling a real uvicorn
    task mid-receive, which is a slow and flaky way to assert a two-line
    property.
    """
    import ast
    import inspect

    from assistant.io.api import app as app_module

    source = inspect.getsource(app_module.create_app)
    tree = ast.parse(source.lstrip())

    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any("unregister" in ast.dump(h) for h in [node] + node.finalbody)
    ]
    assert loops, "the drover socket's try/finally was not found; this guard needs updating"
    block = loops[0]

    assert block.finalbody, "the teardown is not in a finally block"

    # `reason` must be bound before the try, not only inside its handlers.
    fn_source = source[: source.index("try:", source.index("while True:") - 400)]
    assert 'reason = "closed"' in fn_source, (
        "the teardown's `reason` is bound only inside exception handlers. A "
        "BaseException -- CancelledError is one -- matches none of them, and "
        "the finally then raises UnboundLocalError instead of unregistering."
    )

    handled = []
    for handler in block.handlers:
        if handler.type is None:
            handled.append("bare")
        else:
            handled.append(ast.unparse(handler.type))
    assert any("CancelledError" in h for h in handled), (
        f"cancellation is not handled explicitly: {handled}. It is not an "
        f"Exception, so `except Exception` does not catch it."
    )


# ─── A ghost must not hold the slot ──────────────────────────────────────
#
# "The first connection keeps the socket" is right, and it silently assumed the
# incumbent is alive. A browser that reloads its background page leaves a socket
# open to the OS and attached to nothing: `receive_text()` waits forever,
# `is_occupied()` stays true, and the extension is refused its own reconnect
# with "another extension is already connected" -- about itself, every few
# seconds, until the daemon restarts. Observed in exactly that shape.


def test_a_dead_incumbent_is_evicted():
    async def never_answers(_frame):
        return None

    async def scenario():
        conn = ws.DroverConnection(send_json=never_answers)
        ws.register(conn)
        assert ws.is_occupied() is True
        evicted = await ws.evict_if_dead(timeout=0.05)
        return evicted, ws.current_connection()

    evicted, remaining = _run(scenario())
    assert evicted is True, "a connection that never answers still holds the slot"
    assert remaining is None


def test_a_live_incumbent_is_not_evicted():
    """The half that protects a working session.

    A probe that dropped everyone would satisfy the test above and hand the
    socket to whoever asked last -- which is the exact behaviour the
    one-client rule exists to prevent.
    """
    sent: list[dict] = []

    async def send(frame):
        sent.append(frame)

    async def scenario():
        conn = ws.DroverConnection(send_json=send)
        ws.register(conn)

        async def answer_when_asked():
            for _ in range(100):
                await asyncio.sleep(0)
                if sent:
                    conn.handle_frame({
                        "type": proto.Frame.RESPONSE, "id": sent[0]["id"],
                        "ok": True, "result": {"tabs": []},
                    })
                    return

        asyncio.create_task(answer_when_asked())
        evicted = await ws.evict_if_dead(timeout=1.0)
        return evicted, ws.current_connection() is conn

    evicted, still_there = _run(scenario())
    assert evicted is False, "a live connection was evicted"
    assert still_there, "the live incumbent lost its slot"


def test_evicting_an_empty_slot_is_not_an_error():
    assert _run(ws.evict_if_dead(timeout=0.05)) is False


def test_the_route_probes_before_refusing_a_second_client():
    """The eviction has to be reachable from the handshake, not merely exist.

    Every unit test above passes against a route that never calls it, and the
    symptom of that is precisely what was observed: a correct rule, a correct
    helper, and an extension refused its own reconnect.
    """
    import inspect

    from assistant.io.api import app as app_module

    source = inspect.getsource(app_module.create_app)
    assert "evict_if_dead" in source, (
        "the drover route never probes the incumbent, so a ghost connection "
        "holds the slot until the daemon restarts"
    )
    probe_at = source.index("evict_if_dead(")
    verdict_at = source.index("verdict = evaluate_handshake(")
    assert probe_at < verdict_at, (
        "the probe runs after the handshake verdict, which has already refused "
        "the newcomer by then"
    )
