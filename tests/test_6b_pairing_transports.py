"""Pairing over a transport, not just loopback.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §5.3-§5.5.

Three properties, each with its own failure mode if lost:

- **`_endpoints()` is listener-scoped.** A phone off-LAN cannot reach
  `127.0.0.1`, so a QR built from the loopback origin on any listener but
  `local` is a scannable code with no destination.
- **Minting for a named transport is a deliberate, explicit cross-listener
  read.** `POST /v1/pair/code` only ever runs on `local` (`require_admin`), so
  without a `transport` field it could never build anything but a loopback
  URL -- useless to a phone off-LAN. Naming a transport that is not running,
  or one that has not yet announced a hostname, is refused rather than
  silently falling back to loopback.
- **Redemption is refused on `quick`, before the code is consulted.**
  Cloudflare terminates TLS and reads the plaintext of the code and the
  resulting `Set-Cookie` alike, so a wrong code and a right one must be
  indistinguishable on that listener -- and, unlike a wrong code, a right one
  must not be burned by an attempt that could never have succeeded.
"""
from __future__ import annotations

from assistant.io.api.app import create_app
from assistant.io.api.listeners import port_for
from assistant.io.api.pairing import PairCodeStore
from assistant.io.api.routes import pairing as pairing_module
from assistant.io.api.policy import POLICIES, effective
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER, PublishedHosts
from assistant.io.api.transports import transport_registry
from assistant.io.api.transports.base import TransportSession
from assistant.io.api.transports.manager import is_serving
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import LOCAL_PORT, ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime

# The four listeners at their real offsets, exactly as a fully-started 6b
# daemon has them -- the same shape `test_6b_raise_routes.py` and
# `test_6b_listener_matrix.py` already use.
PORTS: dict[str, int] = {name: port_for(name, LOCAL_PORT)
                         for name in ("local", "tailnet", "funnel", "quick")}
POLICY_REGISTRY: dict[int, str] = {port: name for name, port in PORTS.items()}


class FakeTransports:
    """Stands in for Task 9's `TransportManager`. Only `running()` is read.

    Filters through the real `is_serving` (never a hand-rolled hostname
    check), so this fake carries the identical "a session with no announced
    hostname does not serve" contract the real manager enforces -- a session
    with `hostname=None` is present in `_sessions` but absent from
    `running()`, exactly as it would be on a transport that started but never
    announced.
    """

    def __init__(self, sessions: dict[str, TransportSession] | None = None) -> None:
        self._sessions = dict(sessions or {})

    def running(self) -> dict[str, TransportSession]:
        return {name: session for name, session in self._sessions.items()
                if is_serving(session)}


def _session(name: str, *, hostname: str | None) -> TransportSession:
    """A `TransportSession` with the bookkeeping fields this route never
    reads (`process`, `sock`, `serve_task`) left as `None` -- `is_serving` and
    `mint_pair_code` only ever look at `hostname`, `port` and `policy_name`.

    `adapter` is looked up from the real, module-level `transport_registry`
    by *name* -- the same source `TransportManager.start` reads it from --
    rather than left `None`, so `.url` here carries the same public port a
    real session would (fix for the live-test defect: a bare `None` adapter
    silently fell back to a portless URL, which is exactly the bug on
    `tailnet`).
    """
    return TransportSession(
        policy_name=name, port=PORTS[name], owner=f"owner-{name}",
        process=None, sock=None, serve_task=None, hostname=hostname,
        adapter=transport_registry.get(name),
    )


def build(vault: TokenVault, *, store: PairCodeStore | None = None,
          transports: dict[str, TransportSession] | None = None):
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=dict(POLICY_REGISTRY),
                     pair_store=store)
    app.state.transports = FakeTransports(transports)
    return app


def client_on(app, listener: str, token: str | None = None) -> ApiTestClient:
    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{PORTS[listener]}")
    if token is not None:
        client.cookies.set(COOKIE_NAME, token)
    return client


def mint(client: ApiTestClient, *, transport: str = "local",
         grants: list[str] | None = None) -> dict:
    body = {"label": "phone", "grants": grants or ["observe"]}
    if transport is not None:
        body["transport"] = transport
    r = client.post("/v1/pair/code", json=body, headers={CSRF_HEADER: "1"})
    return r


# ─── §5.3/§5.4: minting builds a URL a phone can actually reach ───────────

def test_minting_defaults_to_the_local_loopback_endpoint(tmp_path):
    """No `transport` field at all -- the 6a shape -- still mints the
    loopback origin, unchanged."""
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    app = build(vault)
    client = client_on(app, "local", admin)
    r = mint(client, transport=None)
    assert r.status_code == 200, r.text
    assert r.json()["data"]["endpoints"] == [f"http://127.0.0.1:{PORTS['local']}"]


def test_minting_for_a_running_transport_encodes_its_https_url(tmp_path, monkeypatch):
    """`transport="tailnet"`, with tailnet actually running and announced,
    builds the QR from that transport's published host -- never loopback.

    Carries `:8443`, tailnet's own public port -- the live-test defect this
    guards: a phone scanning this QR must be pointed at the port
    `tailscale serve` actually publishes on, not the HTTPS default it fell
    back to before the fix (`ERR_CONNECTION_TIMED_OUT`, since nothing
    listens on 443 for a tailnet mapping). Asserted on the JSON `endpoints`
    field AND on the literal payload handed to `qr_svg` -- a QR encodes its
    input as a scannable bitmap, not as recoverable text (`qr.py`'s own
    docstring: that opacity is deliberate, so the code cannot be read back
    off a screenshot), so proving the *QR itself* carries the port means
    spying on the string `qr_svg` was called with, the same shape
    `test_the_pair_code_still_travels_in_the_url_fragment` already uses for
    the fragment half of this same payload.
    """
    captured: list[str] = []
    original = pairing_module.qr_svg

    def spy(payload: str) -> str:
        captured.append(payload)
        return original(payload)

    monkeypatch.setattr(pairing_module, "qr_svg", spy)

    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    session = _session("tailnet", hostname="laptop.tailxyz.ts.net")
    app = build(vault, transports={"tailnet": session})
    client = client_on(app, "local", admin)
    r = mint(client, transport="tailnet")
    assert r.status_code == 200, r.text
    endpoints = r.json()["data"]["endpoints"]
    assert endpoints == ["https://laptop.tailxyz.ts.net:8443"]
    assert not any(e.startswith("http://127.0.0.1") for e in endpoints)

    assert len(captured) == 1
    assert captured[0].startswith("https://laptop.tailxyz.ts.net:8443/pair#")


def test_minting_for_a_transport_that_is_not_running_is_refused(tmp_path):
    """`transport="tailnet"` named, but nothing is running under that name at
    all -- a scannable code with no destination, refused rather than built."""
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    app = build(vault, transports={})
    client = client_on(app, "local", admin)
    r = mint(client, transport="tailnet")
    assert r.status_code not in (200, 201)
    assert r.status_code != 500


def test_minting_for_a_transport_with_no_published_host_is_refused(tmp_path):
    """A session that exists but never announced a hostname (`hostname=None`)
    is filtered out of `running()` by `is_serving`, so this must refuse
    exactly like "not running" -- the route must not reach past `running()`
    for a hostname of its own.
    """
    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))
    unannounced = _session("tailnet", hostname=None)
    app = build(vault, transports={"tailnet": unannounced})
    client = client_on(app, "local", admin)
    r = mint(client, transport="tailnet")
    assert r.status_code not in (200, 201)
    assert r.status_code != 500


def test_the_pair_code_still_travels_in_the_url_fragment(tmp_path, monkeypatch):
    """A fragment is stripped by the browser before the request leaves it, so
    no server, proxy or tunnel between the phone and this daemon ever sees the
    code. Re-pinned for BOTH URL shapes: the loopback origin (6a) and a
    transport's published `https://` host (6b) -- a refactor of the URL
    builder that moved the code into the path or the query for one shape and
    not the other would pass every status-code assertion in this file and
    still leak the code to `cloudflared`.
    """
    captured: list[str] = []
    original = pairing_module.qr_svg

    def spy(payload: str) -> str:
        captured.append(payload)
        return original(payload)

    monkeypatch.setattr(pairing_module, "qr_svg", spy)

    vault = TokenVault(tmp_path)
    admin = vault.issue("laptop", frozenset(Capability))

    # Loopback shape.
    app = build(vault)
    client = client_on(app, "local", admin)
    r = mint(client, transport="local")
    assert r.status_code == 200, r.text
    code = r.json()["data"]["code"]

    # Transport shape.
    session = _session("funnel", hostname="laptop.example.ts.net")
    app2 = build(vault, transports={"funnel": session})
    client2 = client_on(app2, "local", admin)
    r2 = mint(client2, transport="funnel")
    assert r2.status_code == 200, r2.text
    code2 = r2.json()["data"]["code"]

    assert len(captured) == 2
    for payload, minted_code in zip(captured, (code, code2)):
        assert "#" in payload
        before, _, after = payload.partition("#")
        assert after == minted_code
        # What a browser actually sends: everything before "#". The code
        # must be absent from it, in the query string and in the path alike.
        assert minted_code not in before


def test_a_tunnel_listener_never_offers_a_loopback_endpoint():
    """`_endpoints()` itself, unit-tested directly: `mint_pair_code`'s only
    caller (`require_admin`) means it always runs on `local` today, so this is
    the one place the transport branch is exercised at all. Accepted on a
    transport listener's own port, it must answer that listener's published
    `https://` hosts, each carrying that transport's own public port
    (`:8443` on tailnet -- the live-test defect this pins), and offer no
    loopback candidate whatsoever.
    """
    published = PublishedHosts()
    published.publish("laptop.tailxyz.ts.net", owner="o1",
                      listener=PORTS["tailnet"], public_port=8443)

    class _State:
        listener_policies = POLICY_REGISTRY
        published_hosts = published

    class _App:
        state = _State()

    class _Request:
        app = _App()
        scope = {"server": ("127.0.0.1", PORTS["tailnet"])}

    endpoints = pairing_module._endpoints(_Request())
    assert endpoints == ["https://laptop.tailxyz.ts.net:8443"]
    assert not any(e.startswith("http://127.0.0.1") for e in endpoints)


# ─── §5.5: redemption refused on `quick`, before the code is consulted ────

def test_redemption_succeeds_on_tailnet_and_funnel(tmp_path):
    """Sanity companion to the `quick` refusal below: pairing over the other
    two remote transports is the ordinary, expected path and must keep
    working."""
    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    app = build(vault, store=store)

    code_a = store.mint("phone-a", frozenset({Capability.OBSERVE})).code
    r_a = client_on(app, "tailnet").post("/v1/pair", json={"code": code_a})
    assert r_a.status_code == 204

    code_b = store.mint("phone-b", frozenset({Capability.OBSERVE})).code
    r_b = client_on(app, "funnel").post("/v1/pair", json={"code": code_b})
    assert r_b.status_code == 204

    assert {d.label for d in vault.devices()} == {"phone-a", "phone-b"}


def test_redemption_is_refused_on_quick(tmp_path):
    """Cloudflare would observe the pair code AND the resulting `Set-Cookie`
    device token in plaintext -- so this must refuse a CORRECT, live code,
    not merely fail to redeem a wrong one (that would pass even with no
    quick-specific guard at all, since `consume()` already refuses wrong
    codes on every listener). The refusal also must not burn the code: the
    same one succeeds afterward on `tailnet`, proving the attempt on `quick`
    never reached `store.consume()`.
    """
    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    app = build(vault, store=store)
    code = store.mint("phone", frozenset(Capability)).code

    r = client_on(app, "quick").post("/v1/pair", json={"code": code})
    assert r.status_code != 204
    assert vault.devices() == []

    # Still live: the attempt on `quick` did not consume it.
    r2 = client_on(app, "tailnet").post("/v1/pair", json={"code": code})
    assert r2.status_code == 204
    assert {d.label for d in vault.devices()} == {"phone"}


def test_redeemed_grants_are_narrowed_to_ceiling_plus_raisable_not_ceiling_alone(tmp_path):
    """Issue-time narrowing (`pair_device`) uses `effective(pair_code.grants,
    policy, raised=policy.raisable)`, never the bare-ceiling two-argument
    form. `tailnet`'s `raisable` is `{EXECUTE, SYSTEM_CONTROL}`, so a code
    minted with every capability, redeemed over `tailnet`, DOES mint a device
    carrying both -- narrowing to the everyday ceiling alone used to discard
    them from the device's own issued set permanently, making `tailnet`'s
    whole `raisable` configuration unreachable (the live-test defect this
    fixes). This is not a widening of what the device may do on an ordinary
    request: `test_a_tailnet_device_holding_execute_is_still_refused_it_
    without_a_live_raise` below pins that `authenticate()`'s own per-request
    narrowing is untouched.
    """
    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    app = build(vault, store=store)
    code = store.mint("phone", frozenset(Capability)).code

    r = client_on(app, "tailnet").post("/v1/pair", json={"code": code})
    assert r.status_code == 204

    paired = [d for d in vault.devices() if d.label == "phone"][0]
    assert paired.grants == effective(frozenset(Capability), POLICIES["tailnet"],
                                      raised=POLICIES["tailnet"].raisable)
    # Not vacuous: the stored set must actually carry what the bare ceiling
    # alone would have discarded.
    assert Capability.EXECUTE in paired.grants
    assert Capability.SYSTEM_CONTROL in paired.grants
    assert paired.grants == frozenset(Capability)  # tailnet's ceiling | raisable is everything


def test_redeemed_grants_over_funnel_still_exclude_execute_and_system_control(tmp_path):
    """`funnel`'s `raisable` is empty, so `ceiling | raisable` there is just
    `ceiling` -- identical to the pre-fix behaviour, and the property the
    operator confirmed is deliberately unaffected: a public URL with no
    second credential gating it must never mint a device carrying EXECUTE or
    SYSTEM_CONTROL, raise or no raise.
    """
    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    app = build(vault, store=store)
    code = store.mint("phone", frozenset(Capability)).code

    r = client_on(app, "funnel").post("/v1/pair", json={"code": code})
    assert r.status_code == 204

    paired = [d for d in vault.devices() if d.label == "phone"][0]
    assert Capability.EXECUTE not in paired.grants
    assert Capability.SYSTEM_CONTROL not in paired.grants
    assert paired.grants == frozenset({
        Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
        Capability.SCREEN, Capability.FILES,
    })


def test_a_tailnet_device_holding_execute_is_still_refused_it_without_a_live_raise(tmp_path):
    """The property the whole fix rests on: storing EXECUTE in the vault is
    not the same as being allowed to use it. `authenticate()`'s per-request
    narrowing (`effective(issued, policy, raised)` with whatever the *live*
    raise store holds -- empty here) is untouched by the issue-time change
    above, so a device that paired over `tailnet` and now has EXECUTE on file
    is still refused it on an ordinary request with no raise live.

    Redeems and re-authenticates through the *same* client on purpose: the
    pairing response's `Set-Cookie` is copied into this client's own jar (the
    `Secure` attribute `tailnet` sets, per `HOST_COOKIE_NAME`'s docstring,
    means the test client's own cookie jar will not re-attach it
    automatically over the plain-`http` test transport -- the same reason
    `test_pairing_over_funnel_permanently_limits_the_device_even_on_a_wider_
    listener` above copies it by hand rather than relying on the jar), so the
    follow-up requests below carry the exact credential `pair_device` just
    issued, with no need to read the token back out of the vault (which is
    not exposed after issuance by design).
    """
    from assistant.io.api.raises import RaiseStore
    from assistant.io.api.security import CSRF_HEADER, HOST_COOKIE_NAME

    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    app = build(vault, store=store)
    app.state.raises = RaiseStore()
    code = store.mint("phone", frozenset(Capability)).code

    remote = client_on(app, "tailnet")
    paired = remote.post("/v1/pair", json={"code": code})
    assert paired.status_code == 204
    remote.cookies.set(HOST_COOKIE_NAME, paired.cookies[HOST_COOKIE_NAME])

    # The vault DOES hold EXECUTE now (the fix), which `GET /v1/session`'s
    # `grants` field reports -- but `effective` (what this connection may
    # actually carry right now) must not.
    session = remote.get("/v1/session").json()["data"]
    assert "execute" in session["grants"]
    assert "system_control" in session["grants"]
    assert "execute" not in session["effective"]
    assert "system_control" not in session["effective"]

    # And on a real capability-gated route, not just the session summary:
    # PATCH /v1/settings needs SYSTEM_CONTROL, and the ceiling alone (no
    # raise live) must still refuse it.
    patch = {"changes": {"tts_speed": 1.25}}
    refused = remote.patch("/v1/settings", json=patch, headers={CSRF_HEADER: "1"})
    assert refused.status_code == 403


def test_a_raise_lets_a_tailnet_paired_device_actually_reach_execute(tmp_path):
    """The end-to-end property the whole milestone rests on, run through the
    *real* pairing route rather than `vault.issue()` directly (which is what
    `test_6b_raise_routes.py`'s own version of this test does, and is not
    enough on its own to prove the pairing fix and the raise mechanism
    actually compose): a device paired over tailnet holds SYSTEM_CONTROL in
    the vault only because of the issue-time fix above; refused without a
    live raise; and the identical request succeeds once one is minted.
    """
    from assistant.io.api.raises import RaiseStore
    from assistant.io.api.security import CSRF_HEADER, HOST_COOKIE_NAME

    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    session = _session("tailnet", hostname="laptop.tailxyz.ts.net")
    app = build(vault, store=store, transports={"tailnet": session})
    app.state.raises = RaiseStore()
    code = store.mint("phone", frozenset(Capability)).code

    remote = client_on(app, "tailnet")
    paired = remote.post("/v1/pair", json={"code": code})
    assert paired.status_code == 204
    # `tailnet` sets `secure_cookie=True`, so `HOST_COOKIE_NAME`'s `Secure`
    # attribute means the test client's own jar will not re-attach it over
    # the plain-`http` test transport -- copied by hand, the same shape the
    # pre-existing funnel-permanence test above already uses.
    remote.cookies.set(HOST_COOKIE_NAME, paired.cookies[HOST_COOKIE_NAME])
    device_id = [d for d in vault.devices() if d.label == "phone"][0].device_id

    patch = {"changes": {"tts_speed": 1.25}}
    before = remote.patch("/v1/settings", json=patch, headers={CSRF_HEADER: "1"})
    assert before.status_code == 403, "no raise yet: must be refused"

    admin = vault.issue("laptop", frozenset(Capability))
    local = client_on(app, "local", admin)
    minted = local.post(
        f"/v1/devices/{device_id}/raise",
        json={"transport": "tailnet", "capabilities": ["system_control"],
              "minutes": 30, "reason": "live-test follow-up"},
        headers={CSRF_HEADER: "1"})
    assert minted.status_code == 200, minted.text

    after = remote.patch("/v1/settings", json=patch, headers={CSRF_HEADER: "1"})
    assert after.status_code == 200, "the raise must actually let this through"
    assert "tts_speed" in after.json()["data"]["saved"]
