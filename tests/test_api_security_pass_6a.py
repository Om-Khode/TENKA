"""Regression tests for the Milestone 6a adversarial security pass.

Every test here is a probe copied from the lens worktrees that reproduced the
finding it belongs to. Each one failed against `main` @ 65a649a and passes
against the fix; the lens's own docstring is kept so the reasoning that found
the hole travels with the test that pins it closed.

Provenance, per section:

  * Finding 1 (revoked device keeps its socket) --
    `sec/lens-authsession`, `tests/test_api_lens3_findings_probe.py`, three
    tests that were `xfail(strict=True)`. The xfail markers are gone: the
    invariant holds now, so the tests assert it directly.
  * Finding 2 (the OBSERVE socket leaks the user's own words) --
    `sec/lens-accesscontrol`, `tests/test_api_access_control_probe.py`.
  * Finding 3 (the rate limiter's safety argument) --
    `sec/lens-accesscontrol` for the header-rotation half,
    `sec/lens-avail` (`tests/test_availability_lens5.py`) for the growth half.
  * Finding 4 (no request body limit) -- `sec/lens-avail`.
  * Finding 5 (`icacls` blocks the event loop) -- `sec/lens-avail`.
  * Finding 6 (`published_hosts` never removes a hostname) --
    `sec/lens-network`, `tests/test_lens4_dns_rebinding_and_origin_trust.py`.

Run per-file, per this repo's standing rule:
`py -3.11 -m pytest tests/test_api_security_pass_6a.py -v`
"""
from __future__ import annotations

import asyncio
import threading
import time
import tracemalloc

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from assistant.io.api import server
from assistant.io.api.app import MAX_BODY_BYTES, create_app
from assistant.io.api.pairing import PairCodeStore
from assistant.io.api.policy import POLICIES
from assistant.io.api.security import (
    _MAX_TRACKED_KEYS,
    COOKIE_NAME,
    PublishedHosts,
    RateLimiter,
    origin_is_known,
)
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import BASE_URL, LOCAL_PORT, ApiTestClient, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime


def _client(vault, *, policies, runtime=None, pair_store=None):
    # No `published=` parameter. It existed, defaulted to `()`, and no test in
    # this file ever passed one -- so when `PublishedHosts.add` grew a required
    # `listener=` in Milestone 6b, the dead loop inside it went on calling the
    # old signature and nothing failed. A helper nobody exercises is a helper
    # that quietly rots into a lie about the API it wraps; the two tests that
    # do publish call `app.state.published_hosts.publish(...)` directly, where
    # the listener they mean is visible at the call site.
    app = create_app(runtime or build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=policies,
                     pair_store=pair_store)
    return ApiTestClient(app, base_url=BASE_URL)


# ─── Finding 1: an accepted socket does not survive revocation ───────────
# Copied from sec/lens-authsession's findings probe. Each was
# `xfail(strict=True)` there; the reason strings are kept as docstrings.

def test_a_revoked_device_stops_receiving_frames(tmp_path):
    """app.py verified once, at the handshake, and the accepted socket was
    never re-checked. `_COOKIE_MAX_AGE_SECONDS`'s own docstring promises a
    revoked device is refused "on its very next request"; the socket was the
    exception."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)

    with client.websocket_connect("/v1/events") as socket:
        socket.receive_json()                       # hello
        assert vault.revoke(vault.devices()[0].device_id) is True
        assert client.get("/v1/status").status_code == 401   # HTTP is cut off

        client.app.state.hub.publish({"type": "status", "phase": "thinking",
                                      "detail": "after revocation"})
        # The socket should be gone, so the next read should disconnect rather
        # than hand back the frame that was just published.
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


def test_a_revoked_device_cannot_drive_the_runtime_over_its_socket(tmp_path):
    """The CHAT_SEND check read the `Device` captured before accept(), so the
    socket's one write verb kept working after revocation."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    runtime = build_fake_runtime()
    client = _client(vault, policies={LOCAL_PORT: "local"}, runtime=runtime)
    client.cookies.set(COOKIE_NAME, token)

    with client.websocket_connect("/v1/events") as socket:
        socket.receive_json()
        assert vault.revoke(vault.devices()[0].device_id) is True
        before = runtime.chat.aborted
        socket.send_json({"type": "abort"})
        socket.receive_json()
        assert runtime.chat.aborted == before, (
            "a revoked device reached runtime.chat.abort()")


def test_wiping_the_vault_closes_every_open_socket(tmp_path):
    """Same root cause: even `TokenVault.reset()` -- rotate the instance
    secret and delete every device record, the biggest hammer this daemon has
    -- left an accepted socket streaming."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)

    with client.websocket_connect("/v1/events") as socket:
        socket.receive_json()
        vault.reset()
        client.app.state.hub.publish({"type": "status", "phase": "x",
                                      "detail": "still here"})
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


def test_a_silent_listener_is_closed_by_the_revalidation_sweep(tmp_path):
    """The half the per-frame check cannot reach.

    A device that only listens sends no frame to be checked on, and a quiet
    assistant publishes none either -- so a fan-out-only check would cut off
    the busiest sockets first and a silent one never. The hub's revalidate
    task is what closes that; this drives it with no traffic at all.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)

    with client.websocket_connect("/v1/events") as socket:
        socket.receive_json()
        assert vault.revoke(vault.devices()[0].device_id) is True
        # Nothing published, nothing sent. Only the timer can find this one.
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


# ─── Finding 2: the OBSERVE socket must not carry the user's own words ───
# Copied from sec/lens-accesscontrol.

def test_an_observe_only_socket_never_receives_what_the_user_asked_for(tmp_path):
    """`app.py`'s socket handler gates the handshake on `Capability.OBSERVE`,
    justified in its own comment: "nothing this socket emits is stored data --
    it carries status and telemetry frames -- so gating the handshake on the
    stored-data grant would deny a live view to a device that holds no
    stored-data grant and needs none."

    The `"status"` frame's `detail` field falsified that. It is
    `status_broadcaster.set(..., detail=...)` passed through verbatim by
    `events.status_frame_from_broadcaster_event`, and its producers put the
    user's own words in it:

      * assistant/actions/da_handlers.py:35  detail=str(params["goal"])[:40]
      * assistant/actions/web.py:58          detail=query[:40]
      * assistant/actions/web.py:216         detail=url[:40]
      * assistant/actions/da_handlers.py:231 detail=target[:40]

    This test wires the real bridge main.py wires (`status.subscribe(
    hub.publish_status)`) and drives the real `StatusBroadcaster.set()`.
    """
    from assistant.io.api.events import EventHub
    from assistant.io.status_broadcaster import StatusBroadcaster, StatusPhase

    secret = "call mum about the biopsy results"

    vault = TokenVault(tmp_path)
    # OBSERVE alone: no RECALL, no CHAT_SEND. This device was never issued
    # any grant that lets it read what she was told.
    token = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    hub = EventHub()
    client = build_api_client(build_fake_runtime(), vault, hub=hub)
    client.cookies.set(COOKIE_NAME, token)

    status = StatusBroadcaster()
    status.subscribe(hub.publish_status)          # exactly main.py's wiring

    with client.websocket_connect("/v1/events") as socket:
        socket.receive_json()                      # the connect-time hello
        # What handle_computer_task does with the user's goal, verbatim.
        status.set(StatusPhase.THINKING, detail=secret[:40])
        frame = socket.receive_json()

    assert secret[:40] not in frame.get("detail", ""), (
        "an OBSERVE-only device read the user's own instruction off the "
        f"status socket: {frame!r}")


def test_the_quick_ceiling_keeps_user_content_off_the_cloudflare_tunnel(tmp_path):
    """The transport half of the same hole.

    `policy.py` sets `quick`'s ceiling to `{OBSERVE}` and argues at length
    that RECALL is excluded because "it carries the entire knowledge graph
    and every transcript ... Excluding SCREEN while admitting stored data
    withheld the photograph and shipped the description of it."

    The event socket is reachable on `quick` -- its handshake asks for
    OBSERVE and OBSERVE is the whole ceiling -- so the description shipped
    anyway, to the one transport whose threat model names the intermediary
    as the adversary.
    """
    from assistant.io.api.events import EventHub
    from assistant.io.status_broadcaster import StatusBroadcaster, StatusPhase

    query = "divorce lawyer near me"

    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    hub = EventHub()
    app = create_app(build_fake_runtime(), vault, origins=[],
                     hub=hub, listener_policies={LOCAL_PORT: "quick"})
    client = ApiTestClient(app, base_url=BASE_URL)
    client.cookies.set(COOKIE_NAME, token)

    status = StatusBroadcaster()
    status.subscribe(hub.publish_status)

    # A tunnelled handshake with no `Origin` at all is refused (`allowed =
    # policy.name == "local"`). That gate costs an attacker one header: the
    # loopback origin for the accepting port is in `endpoint_origins()` on
    # *every* policy, not just `local`, and a non-browser client picks its own
    # `Origin`. So this is what the request looks like from a tunnel.
    with client.websocket_connect(
            "/v1/events",
            headers={"Origin": f"http://127.0.0.1:{LOCAL_PORT}"}) as socket:
        socket.receive_json()
        status.set(StatusPhase.BROWSING, detail=query[:40])   # web.py:58
        frame = socket.receive_json()

    assert query[:40] not in frame.get("detail", ""), (
        "the user's search query crossed the Cloudflare quick tunnel on an "
        f"OBSERVE-only ceiling: {frame!r}")


def test_a_recall_holding_device_still_sees_the_detail_it_is_entitled_to(tmp_path):
    """The fix must not have collapsed into blanking `detail` for everyone.

    A device holding RECALL is one the owner trusted with transcripts and the
    knowledge graph; withholding a forty-character live echo of the same
    material from it would be theatre, and would gut the dashboard the
    laptop's own session runs.
    """
    from assistant.io.api.events import EventHub
    from assistant.io.status_broadcaster import StatusBroadcaster, StatusPhase

    goal = "book the dentist for thursday"

    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    hub = EventHub()
    client = build_api_client(build_fake_runtime(), vault, hub=hub)
    client.cookies.set(COOKIE_NAME, token)

    status = StatusBroadcaster()
    status.subscribe(hub.publish_status)

    with client.websocket_connect("/v1/events") as socket:
        socket.receive_json()
        status.set(StatusPhase.THINKING, detail=goal)
        frame = socket.receive_json()

    assert frame["detail"] == goal


def test_a_withheld_detail_is_blanked_not_dropped(tmp_path):
    """The frame's key set is load-bearing: every `"status"` frame carries
    the same keys so a client never special-cases one. A withheld field stays
    present and empty."""
    from assistant.io.api.events import EventHub
    from assistant.io.status_broadcaster import StatusBroadcaster, StatusPhase

    vault = TokenVault(tmp_path)
    watcher = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    hub = EventHub()
    client = build_api_client(build_fake_runtime(), vault, hub=hub)
    client.cookies.set(COOKIE_NAME, watcher)

    status = StatusBroadcaster()
    status.subscribe(hub.publish_status)

    with client.websocket_connect("/v1/events") as socket:
        connect_frame = socket.receive_json()
        status.set(StatusPhase.THINKING, detail="something she said",
                   step=(1, 3), tier="vision")
        frame = socket.receive_json()

    assert set(connect_frame.keys()) == set(frame.keys())
    assert frame["detail"] == ""
    # Everything that is about *her* survives -- this is the observation the
    # `quick` ceiling was approved for.
    assert frame["phase"] == "THINKING"
    assert frame["step"] == [1, 3]
    assert frame["tier"] == "vision"


def test_every_frame_the_socket_sends_goes_through_the_classifier(tmp_path, monkeypatch):
    """`_safe_send` is the socket's other send path -- the connect frame and
    the `error`/`ack` replies `app.py` builds itself -- and `build_error_frame`
    emits a field also named `detail`. Its four values are protocol literals
    today, so this is not a live leak; the point is that the path must not be
    the one place that escapes the classifier, because "classify the field
    once, not per call site" is the rule the original leak came from breaking.

    Proved structurally: add `error` to the table and an error frame's
    `detail` must come back blanked for a device without RECALL. If
    `_safe_send` ever stops going through `visible_frame`, this fails.
    """
    from assistant.io.api import events as events_module

    monkeypatch.setattr(events_module, "_USER_CONTENT_FIELDS",
                        {"status": ("detail",), "error": ("detail",)})

    vault = TokenVault(tmp_path)
    watcher = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    client = build_api_client(build_fake_runtime(), vault)
    client.cookies.set(COOKIE_NAME, watcher)

    with client.websocket_connect("/v1/events") as socket:
        socket.receive_json()                       # connect frame
        socket.send_json({"type": "not-a-verb"})
        frame = socket.receive_json()

    assert frame["type"] == "error"
    assert frame["detail"] == "", (
        "an error frame reached the client without passing through "
        "visible_frame -- _safe_send is bypassing the classifier")


# ─── Finding 3: the rate limiter's documented safety argument ────────────
# The header-rotation half is copied from sec/lens-accesscontrol; the growth
# half from sec/lens-avail.

def _production_wrapped(app):
    """The app as `server.serve()` actually runs it, if uvicorn's proxy-header
    default were still in force: `ProxyHeadersMiddleware` rewriting
    `scope["client"]` from `X-Forwarded-For` whenever the peer is 127.0.0.1 --
    which is every request that arrives through `cloudflared` or
    `tailscale funnel`."""
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    return ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1")


def test_the_daemon_does_not_run_with_uvicorn_proxy_headers_enabled():
    """The precondition for the finding below, asserted on the real config."""
    import inspect

    source = inspect.getsource(server.serve)
    assert "proxy_headers=False" in source, (
        "server.serve() builds uvicorn.Config without proxy_headers=False, so "
        "uvicorn's default (True, trusting 127.0.0.1) is in force and every "
        "tunnelled client picks its own scope['client'] via X-Forwarded-For")


def test_the_anonymous_lockout_cannot_be_reset_by_a_client_header(tmp_path):
    """`authenticate()` keys the anonymous budget and the exponential lockout
    on `request.client.host`, and its docstring promises "an anonymous flood
    that reaches the vault is still bounded".

    Under uvicorn's default wrapping, `request.client.host` is
    `X-Forwarded-For`. Rotating that header gave a fresh window and a fresh
    failure count per value, so neither bound existed. Kept as the wrapped
    case rather than the plain one: this asserts the *limiter itself* still
    throttles a rotating header, so the property holds even if somebody puts a
    real proxy in front of the daemon later.
    """
    vault = TokenVault(tmp_path)
    vault.issue("owner", frozenset(Capability))
    app = _client(vault, policies={LOCAL_PORT: "local"}).app
    from fastapi.testclient import TestClient
    client = TestClient(_production_wrapped(app), base_url=BASE_URL,
                        client=("127.0.0.1", 51234))

    # Baseline: one fixed source really is locked out after _MAX_FAILURES.
    fixed = {"Host": f"127.0.0.1:{LOCAL_PORT}", "X-Forwarded-For": "203.0.113.7"}
    codes = {client.get("/v1/session", headers=fixed,
                        cookies={COOKIE_NAME: f"wrong-{i}"}).status_code
             for i in range(40)}
    assert 429 in codes, "the lockout never fired even for one fixed source"

    # Now the same 40 guesses, each announcing a different address.
    app.state.auth.limiter = RateLimiter()
    rotated = [
        client.get("/v1/session",
                   headers={"Host": f"127.0.0.1:{LOCAL_PORT}",
                            "X-Forwarded-For": f"203.0.113.{i}"},
                   cookies={COOKIE_NAME: f"wrong-{i}"}).status_code
        for i in range(40)
    ]
    assert 429 in rotated, (
        "40 wrong-token guesses, each with a different X-Forwarded-For, were "
        "never throttled: the anonymous budget and the exponential lockout "
        "are both keyed on a value the client chooses")


@pytest.mark.asyncio
async def test_x_forwarded_for_does_not_buy_a_second_budget(tmp_path):
    """The wire half, against a real uvicorn server.

    The lens proved the inverse of this: ten wrong-token requests carrying
    `X-Forwarded-For: 203.0.113.9` locked that key out (429), and the very
    next request, identical but for `X-Forwarded-For: 198.51.100.7`, was
    answered 401 -- a fresh, unmetered budget. With `proxy_headers=False` both
    requests are metered as the one peer this daemon can actually see, so the
    second value inherits the first's lockout instead of escaping it.
    """
    port = 8963
    vault = TokenVault(tmp_path)
    task = server.serve(build_fake_runtime(), vault, host="127.0.0.1", port=port,
                        origins=["http://localhost:3000"])
    await asyncio.sleep(0.3)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            headers_a = {"Authorization": "Bearer nope", "X-Forwarded-For": "203.0.113.9"}
            for _ in range(10):
                await client.get("/v1/status", headers=headers_a)
            locked = await client.get("/v1/status", headers=headers_a)

            headers_b = {"Authorization": "Bearer nope", "X-Forwarded-For": "198.51.100.7"}
            fresh = await client.get("/v1/status", headers=headers_b)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert locked.status_code == 429, "sanity: A should be locked out by now"
    assert fresh.status_code == 429, (
        f"a second X-Forwarded-For value was answered {fresh.status_code}, not "
        "429 -- it bought a fresh RateLimiter budget instead of sharing the "
        "one key the real peer address collapses to")


def test_rate_limiter_prunes_a_key_whose_window_has_fully_expired():
    """Every distinct key `check()` ever saw kept a permanent `defaultdict`
    entry -- an empty `deque` once its window had fully slid, but the dict
    entry itself was never deleted. ~862 bytes per key, measured, for the life
    of the process."""
    limiter = RateLimiter()
    n = 20_000
    for i in range(n):
        limiter.check(f"key-{i}", now=0.0)
    assert len(limiter.hits) == n

    # A week later, every window is long expired and not one of those keys has
    # been seen since. One unrelated caller arrives.
    later = 7 * 24 * 3600.0
    assert limiter.check("a-fresh-visitor", now=later) is True
    assert len(limiter.hits) < n, (
        "an expired window's dict entry is still held: nothing evicts a key "
        "whose window has fully slid and whose lockout has expired")
    assert len(limiter.hits) <= _MAX_TRACKED_KEYS
    # All three dicts, not just `hits` -- `record_failure` writes to the other
    # two without going through `hits` at all.
    assert not any(key.startswith("key-") for key in limiter.failures)
    assert not any(key.startswith("key-") for key in limiter.locked_until)


def test_a_locked_out_key_is_never_evicted():
    """Eviction must only ever move in the direction of costing an attacker
    more, never less: forgetting a live lockout would hand back exactly the
    budget it exists to withhold."""
    limiter = RateLimiter()
    for _ in range(20):
        limiter.record_failure("guesser", now=0.0)
    for i in range(2_000):          # enough keys that the sweep runs at all
        limiter.check(f"noise-{i}", now=0.0)

    later = 120.0     # every noise window expired; the lockout has not
    limiter.locked_until["guesser"] = later + 60.0
    limiter.check("someone", now=later)
    assert "guesser" in limiter.locked_until
    assert limiter.check("guesser", now=later) is False


def test_rate_limiter_memory_cost_per_distinct_key():
    """A concrete bytes/key number, via tracemalloc rather than a guess --
    the measurement that made the growth a finding (~862 bytes/key, never
    reclaimed).

    The lens's version printed the number and asserted nothing, so it could
    not fail. It asserts two things now. The per-key cost is bounded, so a
    future `RateLimiter` that starts remembering per-key history (a list of
    outcomes, a source label, a timestamp ring) trips a test rather than
    quietly multiplying the ceiling `_MAX_TRACKED_KEYS` was sized against.
    And the total live footprint at that ceiling stays inside a budget worth
    naming out loud, since the whole point of the eviction work is that this
    number has a roof.
    """
    limiter = RateLimiter()
    n = 50_000
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for i in range(n):
        limiter.check(f"probe-key-{i}", now=0.0)
    after = tracemalloc.take_snapshot()
    stats = after.compare_to(before, "lineno")
    total = sum(s.size_diff for s in stats if s.size_diff > 0)
    tracemalloc.stop()
    per_key = total / n
    print(f"\n[6a] {n} distinct RateLimiter keys cost ~{total/1024:.0f} KiB total, "
          f"~{per_key:.0f} bytes/key while live")

    assert 0 < per_key < 2_048, (
        f"~{per_key:.0f} bytes per RateLimiter key -- the measured cost when "
        "this was a finding was ~862, and _MAX_TRACKED_KEYS was sized against "
        "that. Something started storing per-key history.")
    # The roof the eviction work buys: worst case is every tracked key live at
    # once, and that has to stay a number nobody minds a daemon holding.
    assert per_key * _MAX_TRACKED_KEYS < 128 * 1024 * 1024


# ─── Finding 4: request bodies are bounded, before authentication ────────
# Copied from sec/lens-avail's measurement, restated as the invariant.

@pytest.mark.asyncio
async def test_an_oversized_unauthenticated_body_is_refused_before_it_is_parsed(tmp_path):
    """No middleware anywhere under `assistant/io/api` bounded request body
    size: `create_app` registered `CORSMiddleware` and `HostGate` only,
    neither of which touches the body, and uvicorn's `Config` has no
    body-size cap to set. `ChatRequest.text` is bounded to 8,000 characters,
    but that is a Pydantic *field* constraint -- it only applies after the
    whole body has already been read off the socket and parsed as JSON.

    Measured before the fix: a `POST /v1/chat` carrying a 20 MiB body and no
    credential of any kind was answered 401 in 323-653ms with a traced Python
    heap delta of ~40 MiB. The credential check happened after the body had
    been fully received and parsed.
    """
    port = 8961
    vault = TokenVault(tmp_path)
    task = server.serve(build_fake_runtime(), vault, host="127.0.0.1", port=port,
                        origins=["http://localhost:3000"])
    await asyncio.sleep(0.3)

    size_mb = 20
    body = ('{"text": "' + ("a" * (size_mb * 1024 * 1024)) + '"}').encode("utf-8")

    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            start = time.perf_counter()
            response = await client.post(
                "/v1/chat", content=body,
                headers={"Content-Type": "application/json", "X-TENKA-Request": "1"},
            )
            elapsed = time.perf_counter() - start
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    print(f"\n[6a] POST with a {size_mb}MiB body, NO credential at all, to "
          f"/v1/chat -> status={response.status_code} in {elapsed*1000:.1f}ms")
    assert response.status_code == 413, (
        f"a {size_mb}MiB uncredentialed body was answered "
        f"{response.status_code}, which means it was read and parsed first")


@pytest.mark.asyncio
async def test_the_one_unauthenticated_write_is_bounded_too(tmp_path):
    """`POST /v1/pair` is the only unauthenticated write in the API and it
    becomes publicly reachable in 6b. It reads no credential at all, so the
    body limit is the only thing standing in front of it."""
    port = 8962
    vault = TokenVault(tmp_path)
    task = server.serve(build_fake_runtime(), vault, host="127.0.0.1", port=port,
                        origins=["http://localhost:3000"])
    await asyncio.sleep(0.3)
    body = ('{"code": "' + ("a" * (4 * 1024 * 1024)) + '"}').encode("utf-8")
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            response = await client.post(
                "/v1/pair", content=body,
                headers={"Content-Type": "application/json"})
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert response.status_code == 413


def test_a_body_under_the_cap_is_untouched(tmp_path):
    """The limit must not have become a limit on ordinary use."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = build_api_client(build_fake_runtime(), vault)
    client.cookies.set(COOKIE_NAME, token)
    response = client.post("/v1/chat", json={"text": "hello"},
                           headers={"X-TENKA-Request": "1"})
    assert response.status_code == 202
    assert MAX_BODY_BYTES >= 1024 * 1024


@pytest.mark.asyncio
async def test_a_declared_oversized_body_that_never_arrives_is_answered_anyway(tmp_path):
    """The slowloris primitive the first draft of `BodyLimit` introduced.

    Its `_refuse()` drained unconditionally, including on the declared-
    `Content-Length` path where nothing had been read yet -- so headers
    announcing a two-megabyte body followed by *nothing* left the server
    sitting in `await receive()` for a body that never came. uvicorn has no
    body-read timeout, so a ~150-byte unauthenticated request pinned a server
    task indefinitely: measured at 12+ seconds with no response at all, on the
    middleware added for availability, reachable without a credential.

    Raw sockets rather than httpx, because the whole point is to send the
    headers and then stop.
    """
    port = 8964
    vault = TokenVault(tmp_path)
    task = server.serve(build_fake_runtime(), vault, host="127.0.0.1", port=port,
                        origins=["http://localhost:3000"])
    await asyncio.sleep(0.3)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"POST /v1/chat HTTP/1.1\r\n"
            b"Host: 127.0.0.1:%d\r\n"
            b"Content-Type: application/json\r\n"
            b"X-TENKA-Request: 1\r\n"
            b"Content-Length: %d\r\n"
            b"\r\n" % (port, 2 * 1024 * 1024)
        )
        await writer.drain()
        # Not one byte of body follows. The answer must not wait on it.
        start = time.perf_counter()
        try:
            head = await asyncio.wait_for(reader.readline(), timeout=5.0)
        except asyncio.TimeoutError:
            head = b""
        elapsed = time.perf_counter() - start
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    print(f"\n[6a] headers declaring 2MiB then silence -> {head!r} "
          f"after {elapsed*1000:.0f}ms")
    assert head.startswith(b"HTTP/1.1 413"), (
        "a request that declared an oversized body and then sent nothing was "
        f"not answered ({head!r}) -- the server is waiting on a body that "
        "will never arrive, which is a slowloris primitive on an "
        "unauthenticated route")


def test_a_cross_origin_413_is_readable_by_the_browser_that_asked(tmp_path):
    """`BodyLimit` was registered between CORS and `HostGate`, and the comment
    there claimed it sat *inside* CORS so a 413 would carry the CORS headers a
    browser needs. `add_middleware` inserts at index 0, so it sat outside: a
    cross-origin 413 carried no `Access-Control-Allow-Origin` while a 401
    under the same cap did, and the browser reports that as "could not reach
    her" rather than as a refusal -- the exact trap the `BackupProviderError`
    handler was added for. The ordering moved rather than the comment.
    """
    vault = TokenVault(tmp_path)
    client = build_api_client(build_fake_runtime(), vault)
    dev_origin = "http://localhost:3000"

    over = client.post("/v1/chat", content=b"x" * (MAX_BODY_BYTES + 1),
                       headers={"Content-Type": "application/json",
                                "X-TENKA-Request": "1",
                                "Origin": dev_origin})
    under = client.post("/v1/chat", json={"text": "hi"},
                        headers={"X-TENKA-Request": "1", "Origin": dev_origin})

    assert over.status_code == 413
    assert under.status_code == 401
    assert over.headers.get("access-control-allow-origin") == dev_origin, (
        "the 413 skipped CORS entirely, so the page that asked cannot read it")
    assert under.headers.get("access-control-allow-origin") == dev_origin


def test_a_lying_content_length_does_not_get_past_the_cap(tmp_path):
    """A header cannot be trusted to be the truth, so it is used only to
    refuse early, never to permit: an oversized body arriving with no
    `Content-Length` at all (chunked) is counted as it streams."""
    vault = TokenVault(tmp_path)
    client = build_api_client(build_fake_runtime(), vault)

    def _chunks():
        chunk = b"a" * (256 * 1024)
        for _ in range(12):          # 3 MiB, no Content-Length
            yield chunk

    response = client.post("/v1/chat", content=_chunks(),
                           headers={"Content-Type": "application/json",
                                    "X-TENKA-Request": "1"})
    assert response.status_code == 413


# ─── Finding 5: icacls must not run on the event loop ────────────────────
# Copied from sec/lens-avail, with the thread-identity assertion the
# measurement implied but never made.

@pytest.mark.asyncio
async def test_a_vault_write_never_spawns_icacls_on_the_event_loop(tmp_path):
    """`touch()` ran inside `authenticate()`, synchronously, on every
    authenticated request whose device has no recent `last_seen_at` -- and
    `_save()` inside it spawned `icacls` synchronously too, 13-90ms per call
    with nothing wrapping either in a thread. Against a real uvicorn server,
    eight unrelated, unauthenticated requests to a path this daemon has no
    route for measured 1.5x-7.7x slower while one such write was in flight,
    because the loop's single thread was inside `subprocess.run()`'s wait.

    This drives the same REAL uvicorn server the lens did and asserts the
    property the measurement implied: the ACL subprocess does not run on the
    thread the event loop is running on.
    """
    from assistant.io.api import vault as vault_module

    port = 8960
    loop_thread = threading.get_ident()
    ran_on: list[int] = []
    real = vault_module._restrict_to_current_user

    def recording(path):
        ran_on.append(threading.get_ident())
        return real(path)

    vault_module._restrict_to_current_user = recording
    try:
        vault = TokenVault(tmp_path)
        task = server.serve(build_fake_runtime(), vault, host="127.0.0.1", port=port,
                            origins=["http://localhost:3000"])
        await asyncio.sleep(0.3)
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
                baseline = []
                for _ in range(5):
                    start = time.perf_counter()
                    await client.get("/definitely-not-a-route")
                    baseline.append(time.perf_counter() - start)
                baseline_avg = sum(baseline) / len(baseline)

                # A fresh device's first authenticate() call always writes: no
                # last_seen_at yet means the throttle never suppresses it.
                token = vault.issue("victim", frozenset({Capability.OBSERVE}))
                ran_on.clear()          # issue() itself is not on the loop

                async def bystander():
                    start = time.perf_counter()
                    await client.get("/definitely-not-a-route")
                    return time.perf_counter() - start

                async def writer():
                    start = time.perf_counter()
                    await client.get("/v1/session",
                                     headers={"Authorization": f"Bearer {token}"})
                    return time.perf_counter() - start

                bystander_times, write_time = await asyncio.gather(
                    asyncio.gather(*(bystander() for _ in range(8))),
                    writer(),
                )
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        vault_module._restrict_to_current_user = real

    bystander_avg = sum(bystander_times) / len(bystander_times)
    print(f"\n[6a] baseline unrelated-404 latency avg={baseline_avg*1000:.2f}ms")
    print(f"[6a] touch()-writing request latency={write_time*1000:.2f}ms")
    print(f"[6a] 8 unrelated bystander 404s fired DURING that write: "
          f"avg={bystander_avg*1000:.2f}ms")
    print(f"[6a] bystander slowdown vs baseline: "
          f"{bystander_avg/baseline_avg if baseline_avg else float('inf'):.1f}x")

    assert ran_on, (
        "the request did not write at all -- a fresh device's first "
        "authenticated call must touch(), or this test proves nothing")
    assert loop_thread not in ran_on, (
        "icacls ran on the event loop thread: every other request this daemon "
        "was serving, and the assistant sharing this loop, stopped for the "
        "duration of the subprocess wait")


# ─── Finding 6: a published hostname can be un-published ─────────────────
# Copied from sec/lens-network.

def test_the_security_property_6b_needs_holds(tmp_path):
    """The finding, stated as the property that must hold.

    The contract 6b's transports need is: when a transport that published a
    hostname stops, that hostname must stop being trusted by both the Host
    gate and the origin gate -- otherwise `*.trycloudflare.com` names being
    ephemeral and reassignable by Cloudflare turns a merely-stale entry into
    one now pointed at a stranger. The victim's browser is still holding the
    host-only, httpOnly, one-year device cookie for that exact hostname and
    attaches it to whatever answers there next; `httpOnly` blocks JavaScript
    from reading the cookie, not the browser from sending it.

    The lens's version of this test looked for such a function under the only
    reasonable names it could have and failed because none existed.
    """
    import assistant.io.api.app as app_module
    import assistant.io.api.security as security_module

    candidates = [
        getattr(security_module, name, None)
        for name in ("unpublish_host", "remove_published_host",
                     "revoke_published_host", "expire_published_host")
    ] + [
        getattr(app_module, name, None)
        for name in ("unpublish_host", "remove_published_host",
                     "revoke_published_host", "expire_published_host")
    ]
    unpublish = next((fn for fn in candidates if fn is not None), None)

    assert unpublish is not None, (
        "no function exists anywhere in assistant/io/api to remove a "
        "hostname from published_hosts once the transport that published it "
        "stops -- HostGate and origin_is_known() both treat every entry as "
        "trusted forever.")

    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    # A `quick` listener, not `local`, since Milestone 6b: a published name is
    # only ever accepted on a *transport* listener (`host_is_allowed` argues
    # why -- it is KI-17's layer 3). The property under test is unchanged; the
    # listener is simply the one on which a Cloudflare hostname is a name this
    # daemon could have published in the first place.
    client = _client(vault, policies={LOCAL_PORT: "quick"})
    client.cookies.set(COOKIE_NAME, token)

    stale_host = "abc-def.trycloudflare.com"
    client.app.state.published_hosts.publish(stale_host, owner="tunnel-1",
                                             listener=LOCAL_PORT)
    assert client.get("/v1/status", headers={"Host": stale_host}).status_code == 200

    withdrawn = unpublish(client.app.state, "tunnel-1")
    assert stale_host in withdrawn
    assert client.get("/v1/status", headers={"Host": stale_host}).status_code == 421, (
        "a hostname published by a since-stopped transport is still accepted")


def test_a_withdrawn_host_is_no_longer_a_trusted_origin(tmp_path):
    """`endpoint_origins()` folds every entry in `published_hosts` into the
    trusted-origin set as `https://<host>`. Once stale, it was not just an
    accepted `Host` -- it was an accepted `Origin` for the cross-site
    read/CSRF checks too, which is the set `_refuse_cross_site()` consults
    before any credential is read.

    On a `quick` listener since Milestone 6b, for the same reason two other
    tests in this file moved: `endpoint_origins()` now withholds published
    names from `local`, matching the `Host` gate, so `local` is no longer a
    listener on which a Cloudflare hostname is a trusted origin to withdraw.
    The proposition is untouched -- withdrawing a host withdraws its origin --
    and both assertions below are the originals; only the listener the
    question is asked of moved to one where a published name means anything.
    """
    from assistant.io.api.security import unpublish_host

    class _State:
        pass

    state = _State()
    state.published_hosts = PublishedHosts()
    state.cors_origins = []
    state.listener_policies = {LOCAL_PORT: "quick"}
    state.published_hosts.publish("abc-def.trycloudflare.com", owner="tunnel-1",
                                  listener=LOCAL_PORT)

    assert origin_is_known("https://abc-def.trycloudflare.com", state,
                           LOCAL_PORT, POLICIES["quick"])
    unpublish_host(state, "tunnel-1")
    assert not origin_is_known("https://abc-def.trycloudflare.com", state,
                               LOCAL_PORT, POLICIES["quick"])


def test_a_withdrawn_host_no_longer_grants_the_event_socket(tmp_path):
    """Same withdrawal, proved on the WebSocket handshake -- the one surface
    carrying the most valuable read behind the weakest origin story, since no
    CORS applies to a WS handshake at all."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    # `quick`, for the same reason as the test above: since 6b a published
    # name is accepted on a transport listener only.
    client = _client(vault, policies={LOCAL_PORT: "quick"})
    client.cookies.set(COOKIE_NAME, token)

    stale_host = "xyz-tunnel.trycloudflare.com"
    client.app.state.published_hosts.publish(stale_host, owner="tunnel-2",
                                             listener=LOCAL_PORT)
    with client.websocket_connect(
        "/v1/events",
        headers={"Host": stale_host,
                 "Origin": f"http://127.0.0.1:{LOCAL_PORT}"},
    ) as ws:
        assert ws is not None

    client.app.state.published_hosts.unpublish("tunnel-2")
    # `WebSocketDisconnect`, not a bare `Exception`: `HostGate` answers the
    # handshake with `websocket.close(1008)` before `accept()`, and that is
    # the refusal this test is about. A bare `Exception` would be satisfied by
    # an `AttributeError` from a typo in the lines above it.
    with pytest.raises(WebSocketDisconnect) as refusal:
        with client.websocket_connect(
            "/v1/events",
            headers={"Host": stale_host,
                     "Origin": f"http://127.0.0.1:{LOCAL_PORT}"},
        ):
            pass
    assert refusal.value.code == 1008


def test_one_transports_withdrawal_does_not_take_anothers_hostname(tmp_path):
    """Ownership is per transport *session*, so stopping one tunnel must not
    un-publish a second one that is still running."""
    hosts = PublishedHosts()
    hosts.publish("first.trycloudflare.com", owner="tunnel-1",
                  listener=LOCAL_PORT)
    hosts.publish("second.ts.net", owner="tunnel-2", listener=LOCAL_PORT)

    hosts.unpublish("tunnel-1")
    # `hosts_for(port)`, not `in hosts`: 6b deleted the unscoped read, because
    # "is this one of ours?" is the wrong question once four listeners share
    # one app. Same assertion, asked of the listener both names were published
    # on.
    assert "first.trycloudflare.com" not in hosts.hosts_for(LOCAL_PORT)
    assert "second.ts.net" in hosts.hosts_for(LOCAL_PORT)
    # Idempotent: a crash handler and an orderly stop both call it.
    assert hosts.unpublish("tunnel-1") == frozenset()


def test_pairing_endpoints_still_ignore_published_hosts(tmp_path):
    """Carried control, re-pinned: `_endpoints()` is hard-coded to the
    loopback origin and does not read `published_hosts`, so minting a code
    while a tunnel hostname is published does not leak it.

    Both halves are asserted, because the endpoint list and the QR are
    rendered separately by the same route and **the QR is the thing a phone
    actually scans** -- a tunnel host leaking into the SVG while `endpoints`
    stayed clean would otherwise go unnoticed. Precondition flagged for 6b:
    the day `_endpoints()` does start reading `published_hosts`, it must not
    keep offering the bare `http://127.0.0.1:<port>` candidate alongside (or
    ahead of) an `https://<tunnel-host>` one to a phone that is not on this
    LAN segment.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    store = PairCodeStore()
    client = _client(vault, policies={LOCAL_PORT: "local"}, pair_store=store)
    client.cookies.set(COOKIE_NAME, token)
    client.app.state.published_hosts.publish("abc.trycloudflare.com",
                                             owner="tunnel-1",
                                             listener=LOCAL_PORT)

    response = client.post("/v1/pair/code", json={"label": "phone",
                                                  "grants": ["observe"]},
                           headers={"X-TENKA-Request": "1"})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["endpoints"] == [f"http://127.0.0.1:{LOCAL_PORT}"]
    assert "trycloudflare.com" not in str(data["endpoints"])
    assert "trycloudflare.com" not in data["qrSvg"], (
        "the QR now encodes the published tunnel host -- re-check it is "
        "https:// and that the loopback http:// candidate was not left "
        "alongside it for an off-LAN phone to fall back to"
    )
