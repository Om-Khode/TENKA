"""The credential is a cookie, and the listener decides what it is worth.

Four independent mechanisms are pinned here, because Milestone 6 puts this
daemon on a public URL and `POST /v1/chat` reaches every intent including
`code_executor`:

1. the cookie itself -- httpOnly, host-only, flags chosen by the listener;
2. the policy gate -- resolved from the accepting port, never from the client
   address (tunnels connect from 127.0.0.1) and never from `Host`;
3. the CSWSH gate -- a cookie is attached to a WebSocket handshake from any
   origin, which the old query-string token accidentally prevented;
4. the DNS-rebinding gate -- an unknown `Host` is refused with 421.
"""
import pathlib

import pytest
from starlette.websockets import WebSocketDisconnect

from assistant.io.api.app import create_app
from assistant.io.api.policy import POLICIES
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER, cookie_kwargs
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import BASE_URL, LOCAL_PORT, ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime


def _client(vault: TokenVault, *, policies: dict[int, str]) -> ApiTestClient:
    """A client whose requests really do arrive on `LOCAL_PORT`.

    `policies` names what that one port is, so a single test can move the same
    token between a loopback listener and a Cloudflare quick tunnel without
    anything about the request itself changing.
    """
    app = create_app(build_fake_runtime(), vault,
                     origins=["http://localhost:3000"],
                     listener_policies=policies)
    return ApiTestClient(app, base_url=BASE_URL)


def test_a_cookie_authenticates(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/status").status_code == 200


def test_bearer_is_refused_on_a_remote_listener(tmp_path):
    """The weaker path must be unreachable from anywhere remote."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "funnel"})
    r = client.get("/v1/status", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_a_request_on_an_unregistered_port_is_refused(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={})          # nothing registered
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/status").status_code == 401


def test_the_ceiling_narrows_a_full_token(tmp_path):
    """Same token, same route, different listener: the quick policy has no
    FILES in its ceiling, so the file route is refused even though the device
    holds the grant."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    quick = _client(vault, policies={LOCAL_PORT: "quick"})
    quick.cookies.set(COOKIE_NAME, token)
    assert quick.get("/v1/files/roots").status_code == 403
    assert quick.post("/v1/chat", json={"text": "hi"}).status_code == 403
    assert quick.get("/v1/status").status_code == 200      # CHAT survives


def test_a_forged_host_header_cannot_change_policy(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "quick"})
    client.cookies.set(COOKIE_NAME, token)
    r = client.get("/v1/files/roots", headers={"Host": "127.0.0.1",
                                              "X-Forwarded-For": "127.0.0.1"})
    assert r.status_code == 403


def test_a_cookie_write_needs_the_csrf_header(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.post("/v1/chat", json={"text": "hi"}).status_code == 403
    assert client.post("/v1/chat", json={"text": "hi"},
                       headers={CSRF_HEADER: "1"}).status_code == 202


def test_reads_do_not_need_the_csrf_header(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/status").status_code == 200


def test_a_bearer_write_does_not_need_the_csrf_header(tmp_path):
    """A header credential is never attached by a browser on its own, so a
    cross-site page cannot produce one -- there is no request to forge. The
    CSRF header is a defence against *ambient* authority, and bearer is not
    ambient. Requiring it here would only break `curl` on loopback, which is
    the one place bearer is allowed at all."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    r = client.post("/v1/chat", json={"text": "hi"},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 202


def test_the_socket_authenticates_by_cookie_and_no_query_string(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    with client.websocket_connect("/v1/events") as ws:
        assert ws is not None


def test_a_query_string_token_no_longer_authenticates_the_socket(tmp_path):
    """Credentials in URLs land in every intermediary's access log. This is the
    line that must be dead before any transport exists."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/v1/events?access_token={token}"):
            pass


def test_no_source_file_reads_access_token():
    """Replaces test_the_query_string_exception_is_only_the_socket: the
    exception is gone, so the assertion inverts."""
    for path in pathlib.Path("assistant/io/api").rglob("*.py"):
        assert "access_token" not in path.read_text(encoding="utf-8"), path


def test_cookie_flags_follow_the_policy():
    assert cookie_kwargs(POLICIES["funnel"]) == {
        "httponly": True, "samesite": "strict", "secure": True, "path": "/",
    }
    assert cookie_kwargs(POLICIES["local"])["secure"] is False
    assert cookie_kwargs(POLICIES["local"])["httponly"] is True


def test_the_cookie_is_never_given_a_domain():
    """Host-only, always. `*.ts.net` and `*.trycloudflare.com` are shared
    parents: your machine's neighbours under them are OTHER PEOPLE'S machines.
    A Domain attribute would send this credential to strangers."""
    for policy in POLICIES.values():
        assert "domain" not in cookie_kwargs(policy)


def test_the_socket_refuses_a_cross_site_origin(tmp_path):
    """Cross-site WebSocket hijacking. WebSockets are NOT subject to CORS, and
    the browser attaches the cookie automatically -- so any page the user
    visits could open this socket and stream her screen and status. The
    query-string token accidentally prevented this; a cookie does not."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/events",
                                      headers={"Origin": "https://evil.example"}):
            pass


def test_the_socket_accepts_its_own_origin(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    with client.websocket_connect("/v1/events",
                                  headers={"Origin": f"http://127.0.0.1:{LOCAL_PORT}"}) as ws:
        assert ws is not None


def test_a_missing_origin_is_refused_on_a_remote_listener(tmp_path):
    """A browser always sends `Origin` on a socket handshake, so an absent one
    means a non-browser client. On loopback that is `curl` or a script, which
    already has this machine -- refusing it buys nothing. Arriving over a
    tunnel it is anomalous: a cookie is a browser artefact, and a remote
    non-browser holding one is more likely a replay than a user."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "tailnet"})
    client.cookies.set(COOKIE_NAME, token)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/events"):
            pass


def test_an_unknown_host_header_is_refused(tmp_path):
    """DNS rebinding: a page on evil.example re-resolves its own name to
    127.0.0.1, then speaks to us as same-origin. Host allow-listing is the
    standard defence. Note this REJECTS an unknown Host -- it never lets Host
    select a policy, which would be attacker-controlled input choosing its own
    permissions."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    r = client.get("/v1/status", headers={"Host": "evil.example"})
    assert r.status_code == 421


def test_the_expected_hosts_are_accepted(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    for host in (f"127.0.0.1:{LOCAL_PORT}", f"localhost:{LOCAL_PORT}"):
        assert client.get("/v1/status", headers={"Host": host}).status_code == 200


def test_an_unknown_host_is_refused_on_the_socket_too(tmp_path):
    """The rebinding gate is middleware, not an auth dependency, precisely so
    it also covers the WebSocket route and the unauthenticated static and
    pairing paths Tasks 7 and 10 add."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/events", headers={"Host": "evil.example"}):
            pass


def test_a_published_transport_hostname_is_accepted(tmp_path):
    """A tunnel's public hostname is not knowable at build time, so a running
    transport publishes it onto the live set the gate reads."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/status",
                      headers={"Host": "abc-def.trycloudflare.com"}).status_code == 421
    client.app.state.published_hosts.add("abc-def.trycloudflare.com")
    assert client.get("/v1/status",
                      headers={"Host": "abc-def.trycloudflare.com"}).status_code == 200


def test_a_cross_site_origin_is_refused_on_an_ordinary_read(tmp_path):
    """`Origin` is checked on every method, not only on writes: a cross-site
    *read* of /v1/files or /v1/memory is a disclosure even though it changes
    nothing, and the cookie is attached to it just the same."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    r = client.get("/v1/status", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_the_csrf_header_does_not_rescue_a_cross_site_origin(tmp_path):
    """The two gates are independent, not alternatives."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    r = client.post("/v1/chat", json={"text": "hi"},
                    headers={"Origin": "https://evil.example", CSRF_HEADER: "1"})
    assert r.status_code == 403


def test_the_host_gate_sits_outside_cors(tmp_path):
    """Middleware order, pinned. CORSMiddleware answers a preflight itself and
    returns without ever reaching the router, so if it sat outside the host
    gate a rebinding page would get its preflight approved by a daemon that
    never looked at where the request claimed to be going."""
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    r = client.request("OPTIONS", "/v1/chat",
                       headers={"Host": "evil.example",
                                "Origin": "http://localhost:3000",
                                "Access-Control-Request-Method": "POST"})
    assert r.status_code == 421
    assert "access-control-allow-origin" not in r.headers


def test_the_ceiling_narrows_a_route_that_checks_grants_itself(tmp_path):
    """`run_command` authenticates with `Depends(authenticate)` and then
    compares the catalogue's declared grant against `device.grants` itself,
    without going through `require()`. The narrowing therefore has to happen
    in `authenticate()` -- if it lived in `require()`, this route would keep
    the device's full issued grants on a listener that carries none of them."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    quick = _client(vault, policies={LOCAL_PORT: "quick"})
    quick.cookies.set(COOKIE_NAME, token)
    r = quick.post("/v1/commands/lock_workstation/run", headers={CSRF_HEADER: "1"})
    assert r.status_code == 403


# ─── the commands catalogue may only name capabilities that mean "may act" ──
def _catalogue():
    from assistant.actions.studio_runtime_system import LiveCommandRuntime
    return LiveCommandRuntime._CATALOGUE


def test_no_command_declares_a_read_only_grant():
    """`run_command` trusts the catalogue's `required_grant`. A command that
    declared a read capability would be executable by a device deliberately
    issued read-only -- the CHAT split undone through a different door. The
    catalogue may only name capabilities that mean 'may act'."""
    actionable = {Capability.CHAT_SEND, Capability.FILES,
                  Capability.SYSTEM_CONTROL, Capability.SCREEN}
    for command in _catalogue():
        grant = Capability(command.required_grant)
        assert grant in actionable, f"{command.command_id} declares {grant}"
        assert grant is not Capability.CHAT, command.command_id
