"""Per-listener `Host` and `Origin` scoping — KI-17's third layer.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.3 L3,
§5.1, §5.2.

6b binds three listeners on three ports serving one ASGI app. Two of
KI-17's three defences are procedural: TENKA builds every tunnel's argv
(L1), and a preflight refuses a stale `tailscale serve` mapping (L2). Both
assume the tunnel is one TENKA launched, or at least one it can see.

This file pins the layer that assumes nothing. `tailscale serve` and
`cloudflared` both forward the *public* authority in `Host`, so if the `local`
listener accepts loopback names only, a tunnel pointed at the local port --
by a stale persisted config, or by a tunnel TENKA never launched and knows
nothing about -- arrives carrying a non-loopback name and is refused with 421
before authentication, before policy lookup, before any route runs.

Written before the feature, per spec §2.4.
"""
import pytest

from assistant.io.api.app import create_app
from assistant.io.api.policy import POLICIES
from assistant.io.api.security import (
    COOKIE_NAME,
    PublishedHosts,
    endpoint_origins,
    origin_is_known,
)
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime

# The three fixed ports of spec §2.2, spelled out rather than derived from
# `listeners.port_for`: this suite is about what each *socket* is willing to
# be called, so the numbers it sends to have to be visible in the test.
LOCAL_PORT = 8787
TAILNET_PORT = 8788
FUNNEL_PORT = 8789

ALL_LISTENERS: dict[int, str] = {
    LOCAL_PORT: "local",
    TAILNET_PORT: "tailnet",
    FUNNEL_PORT: "funnel",
}

DEV_ORIGINS = ["http://localhost:3000"]

# Names a real provider hands out, kept as constants so a test that asserts a
# refusal and a test that asserts an acceptance cannot drift onto different
# spellings of "the tunnel's public name".
#
# There is deliberately no separate `FUNNEL_NAME` for the tailnet/funnel
# *sameness* tests below, and the absence is a fact about the deployment
# rather than an omission: `tailscale serve` and `tailscale funnel` publish
# the **same** MagicDNS name, so those tests deliberately publish
# `TAILNET_NAME` on `FUNNEL_PORT` too, rather than inventing a distinct
# funnel hostname that would assert a separation the real deployment does
# not have.
#
# `SECOND_NAME`, below, is a different thing: the generic cross-listener
# scoping tests (`PublishedHosts` keying on `(listener, hostname)`) need
# only *some* second hostname that differs from `TAILNET_NAME`, published on
# a different port -- `PublishedHosts` never parses domain shape, so nothing
# about *which* real transport would publish it matters to those tests.
# Milestone 6b's `quick` transport (a Cloudflare tunnel) used to supply that
# second name for free, because Cloudflare genuinely hands out its own
# `*.trycloudflare.com` domain; it was removed outright (no device could
# ever authenticate over it), so `SECOND_NAME` is published on `FUNNEL_PORT`
# instead, purely as test data, and the tailnet/funnel sameness tests below
# are unaffected since they use `TAILNET_NAME` on both ports, never this one.
TAILNET_NAME = "laptop.tail1234.ts.net"
SECOND_NAME = "second-tunnel.example.net"


def _app(tmp_path, *, policies: dict[int, str] | None = None):
    """One app behind four listeners, plus a token holding every capability.

    Maximal grants deliberately: whatever a request in this file cannot reach,
    the *listener* refused -- never the token.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    app = create_app(
        build_fake_runtime(), vault,
        origins=list(DEV_ORIGINS),
        listener_policies=dict(ALL_LISTENERS if policies is None else policies),
    )
    return app, token


def _client_on(app, token: str, port: int) -> ApiTestClient:
    """A client whose requests really do arrive on `port`.

    The same pattern `tests/test_api_cookie_auth.py` and
    `tests/test_api_security_pass_6a.py` already use -- an absolute loopback
    base URL, so `scope["server"]` carries the port policy resolution reads and
    the default `Host` is a loopback name. Only the port differs here, because
    the whole point is that four sockets answer differently.

    **HTTP only. Do not open a WebSocket through a client from this helper
    with a relative URL.** `ApiTestClient.websocket_connect` in
    `tests/fakes/api_client.py` rewrites a leading `/` to the hard-coded
    `ws://127.0.0.1:8787` -- the *local* port -- regardless of the client's
    own `base_url`. A socket test written against `_client_on(app, token,
    FUNNEL_PORT)` would therefore arrive on the local listener and assert
    nothing about the funnel one, silently and while passing. Pass an
    absolute `ws://127.0.0.1:<port>/v1/events`, which `urljoin` leaves
    untouched, or fix the fake to honour `base_url`. Every test in this file
    uses HTTP, so none of them is currently exposed to it.
    """
    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{port}")
    client.cookies.set(COOKIE_NAME, token)
    return client


@pytest.fixture()
def client_on_local(tmp_path):
    app, token = _app(tmp_path)
    return _client_on(app, token, LOCAL_PORT)


# ─── KI-17 containment ───────────────────────────────────────────────────

def test_a_tunnel_hostname_on_the_local_listener_is_refused(client_on_local):
    """KI-17, layer 3. `tailscale serve` and `cloudflared` both forward the
    PUBLIC authority in Host. So a tunnel pointed at the local port -- by a
    stale persisted config, or by a tunnel TENKA never launched and knows
    nothing about -- arrives here carrying a non-loopback name, and is
    refused before authentication, before policy lookup, before any route.

    This is the one test in the milestone that holds when every other layer
    has been bypassed. If it is ever relaxed, KI-17 is live again.
    """
    response = client_on_local.get("/v1/status",
                                   headers={"Host": "laptop.tail1234.ts.net"})
    assert response.status_code == 421

    # ── DO NOT DELETE THE THREE ASSERTIONS BELOW AS REDUNDANT. ──
    # They are the only ones in this test that discriminate. The assertion
    # above passes against the *pre-6b* code too, because nothing had
    # published `laptop.tail1234.ts.net` and an unpublished name was already
    # refused for an unrelated reason -- so on its own it reads as coverage of
    # KI-17 while proving nothing about it.
    #
    # A tunnel reaches this port precisely *because* a session of it is
    # running and has published its public authority somewhere on this app.
    # So the real question is what `local` does with a name that IS published:
    # once against another listener (the cross-listener case), and once
    # against this very socket (a stale publish on the port under test, the
    # worst case). Both must stay 421. Relax either and KI-17 is live again.
    client_on_local.app.state.published_hosts.publish(
        "laptop.tail1234.ts.net", owner="ts-1", listener=TAILNET_PORT)
    assert client_on_local.get(
        "/v1/status",
        headers={"Host": "laptop.tail1234.ts.net"}).status_code == 421
    client_on_local.app.state.published_hosts.publish(
        "laptop.tail1234.ts.net", owner="ts-2", listener=LOCAL_PORT)
    assert client_on_local.get(
        "/v1/status",
        headers={"Host": "laptop.tail1234.ts.net"}).status_code == 421


def test_the_local_listener_accepts_loopback_names_only(tmp_path):
    """The other half of the containment, and the stronger statement.

    A loopback name is accepted here because it is genuinely this machine.
    Every other name is refused -- *including one published against this very
    port*, which is what makes layer 3 hold against a tunnel nobody declared:
    the refusal does not depend on anyone having noticed the tunnel.
    """
    app, token = _app(tmp_path)
    client = _client_on(app, token, LOCAL_PORT)

    for name in (f"127.0.0.1:{LOCAL_PORT}", "127.0.0.1", "localhost",
                 f"localhost:{LOCAL_PORT}", "[::1]", f"[::1]:{LOCAL_PORT}"):
        assert client.get("/v1/status",
                          headers={"Host": name}).status_code == 200, name

    # A published name on `local` is still refused. `local` is the one policy
    # for which "published" means nothing at all.
    app.state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                      listener=LOCAL_PORT)
    assert client.get("/v1/status",
                      headers={"Host": TAILNET_NAME}).status_code == 421
    assert client.get("/v1/status",
                      headers={"Host": "evil.example"}).status_code == 421


# ─── §5.1 / §5.2: a name belongs to one listener ─────────────────────────

def test_a_transport_listener_accepts_the_names_its_own_session_published(tmp_path):
    """A tunnel's public name is not knowable when the app is built, so the
    running session publishes it -- against the port it is serving."""
    app, token = _app(tmp_path)
    tailnet = _client_on(app, token, TAILNET_PORT)

    assert tailnet.get("/v1/status",
                       headers={"Host": TAILNET_NAME}).status_code == 421
    app.state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                      listener=TAILNET_PORT)
    assert tailnet.get("/v1/status",
                       headers={"Host": TAILNET_NAME}).status_code == 200
    # The port in `Host` still carries no meaning -- the real port is the one
    # the socket was accepted on, and a tunnel forwards a public authority
    # whose port is not this daemon's.
    assert tailnet.get("/v1/status",
                       headers={"Host": f"{TAILNET_NAME}:443"}).status_code == 200


def test_a_name_published_by_one_listener_is_refused_on_another(tmp_path):
    """Spec §5.1. Owner scoping alone made a name published by one
    transport a trusted `Host` -- and a trusted `Origin` -- on a different
    listener's port, one with a different ceiling and a different threat
    model. `SECOND_NAME` published on `FUNNEL_PORT` is the module-level
    stand-in for "some transport's own distinct public name" -- see the
    module docstring for why the real `funnel`/`tailnet` pair cannot supply
    one honestly."""
    app, token = _app(tmp_path)
    app.state.published_hosts.publish(SECOND_NAME, owner="cf-1",
                                      listener=FUNNEL_PORT)

    funnel = _client_on(app, token, FUNNEL_PORT)
    assert funnel.get("/v1/status",
                      headers={"Host": SECOND_NAME}).status_code == 200

    for port in (TAILNET_PORT, LOCAL_PORT):
        other = _client_on(app, token, port)
        assert other.get("/v1/status",
                         headers={"Host": SECOND_NAME}).status_code == 421, port

    assert app.state.published_hosts.hosts_for(FUNNEL_PORT) == frozenset({SECOND_NAME})
    assert app.state.published_hosts.hosts_for(TAILNET_PORT) == frozenset()


def test_unpublishing_a_session_stops_its_names_immediately(tmp_path):
    """`HostGate` reads the live collection, not a copy taken when the app was
    built, so a transport's stop path takes its name back on the very next
    request."""
    app, token = _app(tmp_path)
    funnel = _client_on(app, token, FUNNEL_PORT)

    app.state.published_hosts.publish(SECOND_NAME, owner="cf-1",
                                      listener=FUNNEL_PORT)
    assert funnel.get("/v1/status",
                      headers={"Host": SECOND_NAME}).status_code == 200

    withdrawn = app.state.published_hosts.unpublish("cf-1")
    assert SECOND_NAME in withdrawn
    assert funnel.get("/v1/status",
                      headers={"Host": SECOND_NAME}).status_code == 421
    assert app.state.published_hosts.hosts_for(FUNNEL_PORT) == frozenset()


def test_dropping_a_listener_from_the_registry_stops_its_names_immediately(tmp_path):
    """`HostGate` holds `app.state.listener_policies` itself, not a snapshot.

    The sibling of `test_unpublishing_a_session_stops_its_names_immediately`,
    on the other axis, and it needs its own pin because reading the call site
    is not a test: `registry=dict(app.state.listener_policies)` would pass
    every other assertion in this file and every adjacent suite, while
    quietly breaking this.

    It matters because a transport's stop sequence drops its registry entry.
    Snapshotted, a stopped transport's port keeps its policy name and keeps
    accepting the names published against it -- stale trust surviving the
    thing that earned it, which is the exact class `PublishedHosts` exists to
    prevent, arrived at from the registry side instead and with no published
    entry left to look wrong.
    """
    app, token = _app(tmp_path)
    funnel = _client_on(app, token, FUNNEL_PORT)
    app.state.published_hosts.publish(SECOND_NAME, owner="cf-1",
                                      listener=FUNNEL_PORT)
    assert funnel.get("/v1/status",
                      headers={"Host": SECOND_NAME}).status_code == 200

    # The name is still published; only the listener stopped being declared.
    app.state.listener_policies.pop(FUNNEL_PORT)
    assert funnel.get("/v1/status",
                      headers={"Host": SECOND_NAME}).status_code == 421
    assert app.state.published_hosts.hosts_for(FUNNEL_PORT) == frozenset({SECOND_NAME})

    # The ratified half of "an unknown port refuses", pinned here beside it: a
    # loopback name still passes this gate on a port nobody declares, and is
    # answered 401 by `authenticate()` rather than 421 by the gate. Spec §2.4
    # item 4 and `test_api_cookie_auth.py::test_a_request_on_an_unregistered_
    # port_is_refused` both depend on that being the shape of the refusal.
    assert funnel.get("/v1/status").status_code == 401


def test_a_restarted_tunnel_does_not_leave_its_previous_name_trusted(tmp_path):
    """The hostname-reuse class, spec §8.

    Cloudflare assigns a `*.trycloudflare.com` name for the lifetime of one
    tunnel and hands it to somebody else afterwards. The device cookie is
    host-only and lives a year, so the browser keeps attaching it to that name
    no matter who answers there now -- `httpOnly` stops script from reading
    the cookie, not the browser from sending it. A second run of the same
    tunnel is a different session, and the first session's name must not
    survive it.
    """
    app, _token = _app(tmp_path)
    published = app.state.published_hosts

    first = "old-tunnel.trycloudflare.com"
    second = "new-tunnel.trycloudflare.com"

    published.publish(first, owner="cf-session-1", listener=FUNNEL_PORT)
    published.unpublish("cf-session-1")
    published.publish(second, owner="cf-session-2", listener=FUNNEL_PORT)

    assert published.hosts_for(FUNNEL_PORT) == frozenset({second})
    origins = endpoint_origins(app.state, FUNNEL_PORT, POLICIES["funnel"])
    assert f"https://{second}" in origins
    assert f"https://{first}" not in origins


# ─── §5.1: the origin set is scoped the same way ─────────────────────────

class _State:
    """The three attributes `endpoint_origins` reads, and nothing else."""


def test_endpoint_origins_only_lists_hosts_published_by_the_accepting_port(tmp_path):
    """A second listener cannot lend its origins to the first -- which is what
    the `port` argument has been there for since 6a.5, applied now to the half
    that ignored it."""
    state = _State()
    state.published_hosts = PublishedHosts()
    state.cors_origins = list(DEV_ORIGINS)
    state.listener_policies = dict(ALL_LISTENERS)
    state.published_hosts.publish(SECOND_NAME, owner="cf-1", listener=FUNNEL_PORT)
    state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                  listener=TAILNET_PORT)

    funnel_origins = endpoint_origins(state, FUNNEL_PORT, POLICIES["funnel"])
    assert f"https://{SECOND_NAME}" in funnel_origins
    assert f"https://{TAILNET_NAME}" not in funnel_origins

    tailnet_origins = endpoint_origins(state, TAILNET_PORT, POLICIES["tailnet"])
    assert f"https://{TAILNET_NAME}" in tailnet_origins
    assert f"https://{SECOND_NAME}" not in tailnet_origins

    # No port means no listener, so there is nothing to scope a published name
    # to and none is offered.
    unknown = endpoint_origins(state, None, POLICIES["funnel"])
    assert not any(origin.startswith("https://") for origin in unknown), unknown


def test_endpoint_origins_withholds_published_hosts_from_the_local_listener(tmp_path):
    """The two gates must agree about `local`, not merely both be defensible.

    `host_is_allowed` refuses a published name on `local` -- that is KI-17's
    layer 3. If `endpoint_origins` kept trusting one, the local listener would
    refuse `https://tunnel.ts.net` as a `Host` while vouching for it as an
    `Origin`, and a page served there could drive `http://127.0.0.1:<local>`
    cross-origin: that request carries a loopback `Host` the gate allows and
    an `Origin` this set would have approved. Nothing legitimate needs it --
    Studio over a tunnel talks to the tunnel listener, never to loopback -- so
    the rule is one rule on both sides: a name is trusted only where it was
    published, and `local` publishes nothing.
    """
    state = _State()
    state.published_hosts = PublishedHosts()
    state.cors_origins = list(DEV_ORIGINS)
    state.listener_policies = dict(ALL_LISTENERS)
    # Published against the local port itself -- the strongest form of the
    # question, and the one a stale or hand-made tunnel configuration creates.
    state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                  listener=LOCAL_PORT)

    local_origins = endpoint_origins(state, LOCAL_PORT, POLICIES["local"])
    assert f"https://{TAILNET_NAME}" not in local_origins
    # The loopback pair and the dev origins are untouched: this withholds the
    # published half only, never the front doors `local` actually serves.
    assert f"http://127.0.0.1:{LOCAL_PORT}" in local_origins
    assert DEV_ORIGINS[0] in local_origins

    # Same collection, same port, a transport policy: still trusted there.
    # The listener is what changed, which is the whole claim.
    assert f"https://{TAILNET_NAME}" in endpoint_origins(
        state, LOCAL_PORT, POLICIES["funnel"])


def test_endpoint_origins_still_withholds_dev_origins_from_a_tunnel(tmp_path):
    """Carried control. Studio's Next.js dev server only ever existed on a
    developer's own loopback; a public URL trusting it as an origin would be a
    laptop's dev server driving a tunnelled daemon."""
    state = _State()
    state.published_hosts = PublishedHosts()
    state.cors_origins = list(DEV_ORIGINS)
    state.listener_policies = dict(ALL_LISTENERS)

    local_origins = endpoint_origins(state, LOCAL_PORT, POLICIES["local"])
    assert DEV_ORIGINS[0] in local_origins

    for name, port in (("tailnet", TAILNET_PORT), ("funnel", FUNNEL_PORT)):
        origins = endpoint_origins(state, port, POLICIES[name])
        assert DEV_ORIGINS[0] not in origins, name


# ─── the live-test defect: a tunnel's public port belongs in the Origin too ──
#
# `tailnet` publishes on 8443, not the HTTPS default -- a browser's own
# `Origin` header for a page served there is `https://<host>:8443`, never
# bare. `endpoint_origins` trusting only the bare form (the pre-fix shape)
# refused every genuine one with 403 before a route ever ran, which broke
# pairing and the event socket alike over a real tailnet tunnel.

def test_a_tailnet_origin_must_carry_its_public_port(tmp_path):
    """Exactly one of the two spellings is a front door that exists. Trusting
    the bare form only is the live-test defect; trusting both would accept
    an origin nothing actually serves."""
    state = _State()
    state.published_hosts = PublishedHosts()
    state.cors_origins = []
    state.listener_policies = dict(ALL_LISTENERS)
    state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                  listener=TAILNET_PORT, public_port=8443)

    assert origin_is_known(f"https://{TAILNET_NAME}:8443", state,
                           TAILNET_PORT, POLICIES["tailnet"])
    assert not origin_is_known(f"https://{TAILNET_NAME}", state,
                               TAILNET_PORT, POLICIES["tailnet"])


def test_a_funnel_origin_must_not_carry_the_default_port(tmp_path):
    """The companion property, inverted: `funnel` publishes on 443, the
    HTTPS default, which a browser's `Origin` header never states
    explicitly. Trusting an explicit `:443` form would accept an origin no
    real browser sends -- get the port-omission rule backwards and this is
    the test that catches it."""
    state = _State()
    state.published_hosts = PublishedHosts()
    state.cors_origins = []
    state.listener_policies = dict(ALL_LISTENERS)
    # Tailscale publishes the same MagicDNS name on `serve` and `funnel`
    # alike (module docstring's note on `TAILNET_NAME`/`FUNNEL_NAME`) -- only
    # the port and the listener differ, which is exactly what this test is
    # about.
    state.published_hosts.publish(TAILNET_NAME, owner="fn-1",
                                  listener=FUNNEL_PORT, public_port=443)

    assert origin_is_known(f"https://{TAILNET_NAME}", state,
                           FUNNEL_PORT, POLICIES["funnel"])
    assert not origin_is_known(f"https://{TAILNET_NAME}:443", state,
                               FUNNEL_PORT, POLICIES["funnel"])


def test_hosts_for_stays_bare_even_when_a_public_port_is_published(tmp_path):
    """`origins_for` is the new, port-carrying read; `hosts_for` -- and the
    `Host` gate it feeds -- must be completely unaffected by a published
    `public_port`. The property most likely to be broken by that change, so
    it is pinned directly rather than assumed from the origin tests above.
    """
    app, token = _app(tmp_path)
    tailnet = _client_on(app, token, TAILNET_PORT)
    app.state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                      listener=TAILNET_PORT, public_port=8443)

    assert app.state.published_hosts.hosts_for(TAILNET_PORT) == frozenset(
        {TAILNET_NAME})
    assert tailnet.get("/v1/status",
                       headers={"Host": TAILNET_NAME}).status_code == 200
    # The port-carrying form must never appear as a trusted `Host` -- that
    # would be layer 3 accepting something a tunnel never actually forwards.
    assert f"{TAILNET_NAME}:8443" not in app.state.published_hosts.hosts_for(
        TAILNET_PORT)


def test_event_socket_over_tailnet_is_not_refused_as_cross_site(tmp_path):
    """`app.py`'s event-socket handshake reads the identical `origin_is_known`
    that `refuse_unknown_origin` does, so it inherits this fix automatically
    -- pinned directly rather than merely assumed, since a WebSocket
    handshake has no CORS preflight to catch a regression before it reaches
    the socket. Absolute `ws://` URL, per this file's own module note on
    `websocket_connect`: a relative one silently lands on `local` instead.
    """
    app, token = _app(tmp_path)
    app.state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                      listener=TAILNET_PORT, public_port=8443)
    client = _client_on(app, token, TAILNET_PORT)

    with client.websocket_connect(
        f"ws://127.0.0.1:{TAILNET_PORT}/v1/events",
        headers={"Host": TAILNET_NAME,
                 "Origin": f"https://{TAILNET_NAME}:8443"},
    ) as ws:
        assert ws is not None


def test_pairing_over_tailnet_is_not_refused_as_cross_site(tmp_path):
    """The fully-broken flow the operator's live tunnel actually hit:
    `POST /v1/pair` over `tailnet`, from a phone whose browser sends
    `Origin: https://<host>:8443` -- the real `Origin` a page served over
    that listener carries. Before this fix, `refuse_unknown_origin` compared
    it against a bare, portless trusted set and refused every attempt with
    403 regardless of whether the code was right, which made pairing over
    `tailnet` impossible.
    """
    vault = TokenVault(tmp_path)
    app = create_app(
        build_fake_runtime(), vault,
        origins=list(DEV_ORIGINS),
        listener_policies=dict(ALL_LISTENERS),
    )
    app.state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                      listener=TAILNET_PORT, public_port=8443)

    store = app.state.pair_store
    code = store.mint("phone", frozenset({Capability.OBSERVE})).code

    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{TAILNET_PORT}")
    r = client.post("/v1/pair", json={"code": code},
                    headers={"Origin": f"https://{TAILNET_NAME}:8443"})
    assert r.status_code == 204, r.text


# ─── the unscoped read is gone, not merely unused ────────────────────────

def test_published_hosts_has_no_unscoped_read():
    """`hosts_for(listener)` is the only way to read this collection.

    Leaving `__contains__` and `__iter__` in place would keep the bug one
    `if host in published` away: every existing reader was updated, and the
    next one added would have reached for the surface that ignores the
    listener because it is the one that reads most naturally.
    """
    assert not hasattr(PublishedHosts, "__contains__")
    assert not hasattr(PublishedHosts, "__iter__")
    assert hasattr(PublishedHosts, "hosts_for")
