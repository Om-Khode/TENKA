"""Per-listener `Host` and `Origin` scoping — KI-17's third layer.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.3 L3,
§5.1, §5.2.

6b binds four listeners on four ports serving one ASGI app. Two of KI-17's
three defences are procedural: TENKA builds every tunnel's argv (L1), and a
preflight refuses a stale `tailscale serve` mapping (L2). Both assume the
tunnel is one TENKA launched, or at least one it can see.

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
)
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime

# The four fixed ports of spec §2.2, spelled out rather than derived from
# `listeners.port_for`: this suite is about what each *socket* is willing to
# be called, so the numbers it sends to have to be visible in the test.
LOCAL_PORT = 8787
TAILNET_PORT = 8788
FUNNEL_PORT = 8789
QUICK_PORT = 8790

ALL_LISTENERS: dict[int, str] = {
    LOCAL_PORT: "local",
    TAILNET_PORT: "tailnet",
    FUNNEL_PORT: "funnel",
    QUICK_PORT: "quick",
}

DEV_ORIGINS = ["http://localhost:3000"]

# Names a real provider hands out, kept as constants so a test that asserts a
# refusal and a test that asserts an acceptance cannot drift onto different
# spellings of "the tunnel's public name".
TAILNET_NAME = "laptop.tail1234.ts.net"
QUICK_NAME = "abc-def.trycloudflare.com"
FUNNEL_NAME = "laptop.tail1234.ts.net"


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

    # And it stays 421 once that name is a *published* one -- which is the
    # case that actually discriminates. Before this task, the assertion above
    # passed for the wrong reason (nothing had published the name yet), and a
    # tunnel arrives here precisely because a session of it is running and has
    # published its public authority somewhere on this app.
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
    """Spec §5.1. Owner scoping alone made a name published by the quick
    tunnel a trusted `Host` -- and a trusted `Origin` -- on the funnel port,
    which is a different transport with a different ceiling and a different
    threat model."""
    app, token = _app(tmp_path)
    app.state.published_hosts.publish(QUICK_NAME, owner="cf-1",
                                      listener=QUICK_PORT)

    quick = _client_on(app, token, QUICK_PORT)
    assert quick.get("/v1/status",
                     headers={"Host": QUICK_NAME}).status_code == 200

    for port in (FUNNEL_PORT, TAILNET_PORT, LOCAL_PORT):
        other = _client_on(app, token, port)
        assert other.get("/v1/status",
                         headers={"Host": QUICK_NAME}).status_code == 421, port

    assert app.state.published_hosts.hosts_for(QUICK_PORT) == frozenset({QUICK_NAME})
    assert app.state.published_hosts.hosts_for(FUNNEL_PORT) == frozenset()


def test_unpublishing_a_session_stops_its_names_immediately(tmp_path):
    """`HostGate` reads the live collection, not a copy taken when the app was
    built, so a transport's stop path takes its name back on the very next
    request."""
    app, token = _app(tmp_path)
    quick = _client_on(app, token, QUICK_PORT)

    app.state.published_hosts.publish(QUICK_NAME, owner="cf-1",
                                      listener=QUICK_PORT)
    assert quick.get("/v1/status",
                     headers={"Host": QUICK_NAME}).status_code == 200

    withdrawn = app.state.published_hosts.unpublish("cf-1")
    assert QUICK_NAME in withdrawn
    assert quick.get("/v1/status",
                     headers={"Host": QUICK_NAME}).status_code == 421
    assert app.state.published_hosts.hosts_for(QUICK_PORT) == frozenset()


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

    published.publish(first, owner="cf-session-1", listener=QUICK_PORT)
    published.unpublish("cf-session-1")
    published.publish(second, owner="cf-session-2", listener=QUICK_PORT)

    assert published.hosts_for(QUICK_PORT) == frozenset({second})
    origins = endpoint_origins(app.state, QUICK_PORT, POLICIES["quick"])
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
    state.published_hosts.publish(QUICK_NAME, owner="cf-1", listener=QUICK_PORT)
    state.published_hosts.publish(TAILNET_NAME, owner="ts-1",
                                  listener=TAILNET_PORT)

    quick_origins = endpoint_origins(state, QUICK_PORT, POLICIES["quick"])
    assert f"https://{QUICK_NAME}" in quick_origins
    assert f"https://{TAILNET_NAME}" not in quick_origins

    tailnet_origins = endpoint_origins(state, TAILNET_PORT, POLICIES["tailnet"])
    assert f"https://{TAILNET_NAME}" in tailnet_origins
    assert f"https://{QUICK_NAME}" not in tailnet_origins

    # No port means no listener, so there is nothing to scope a published name
    # to and none is offered.
    unknown = endpoint_origins(state, None, POLICIES["quick"])
    assert not any(origin.startswith("https://") for origin in unknown), unknown


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

    for name, port in (("tailnet", TAILNET_PORT), ("funnel", FUNNEL_PORT),
                       ("quick", QUICK_PORT)):
        origins = endpoint_origins(state, port, POLICIES[name])
        assert DEV_ORIGINS[0] not in origins, name


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
