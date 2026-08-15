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
from fastapi import APIRouter, Depends
from starlette.websockets import WebSocketDisconnect

from assistant.io.api.app import create_app
from assistant.io.api.pairing import PairCodeStore
from assistant.io.api.policy import POLICIES
from assistant.io.api.security import (
    COOKIE_NAME,
    CSRF_HEADER,
    RateLimiter,
    _COOKIE_MAX_AGE_SECONDS,
    cookie_kwargs,
    require,
)
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import BASE_URL, LOCAL_PORT, ApiTestClient, build_api_client
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


@pytest.fixture()
def quick_client(tmp_path):
    """A device issued every capability, arriving on the Cloudflare quick
    tunnel. The token is deliberately maximal: whatever this client cannot
    reach, the *ceiling* refused, not the grant."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "quick"})
    client.cookies.set(COOKIE_NAME, token)
    return client


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
    assert quick.get("/v1/status").status_code == 200      # OBSERVE survives


# ─── OBSERVE is watching her; RECALL is reading what she stored ──────────
def test_an_observe_only_device_sees_status_but_not_history(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("watcher", frozenset({Capability.OBSERVE}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.get("/v1/status").status_code == 200
    assert client.get("/v1/telemetry").status_code == 200
    assert client.get("/v1/chat/conversations").status_code == 403
    assert client.get("/v1/memory/knowledge").status_code == 403


def test_the_socket_streams_for_an_observe_only_device(tmp_path):
    """The stream carries status frames, so it follows OBSERVE -- not RECALL,
    which would deny a live view to a device holding no stored-data grant."""
    vault = TokenVault(tmp_path)
    token = vault.issue("watcher", frozenset({Capability.OBSERVE}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    with client.websocket_connect("/v1/events") as ws:
        assert ws.receive_json()["type"] == "status"


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
    exception is gone, so the assertion inverts.

    Anchored on `__file__`, not on the working directory. A relative
    `Path("assistant/io/api")` yields an empty `rglob` when pytest is invoked
    from anywhere but the repo root -- and an empty sweep passes, silently,
    asserting nothing. Its predecessor read one fixed file and would have
    raised `FileNotFoundError` instead; a sweep has to earn that back by
    proving it found something.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "assistant" / "io" / "api"
    sources = list(root.rglob("*.py"))
    assert len(sources) > 5, f"the sweep found almost nothing under {root}"
    for path in sources:
        assert "access_token" not in path.read_text(encoding="utf-8"), path


def test_cookie_flags_follow_the_policy():
    assert cookie_kwargs(POLICIES["funnel"]) == {
        "httponly": True, "samesite": "strict", "secure": True, "path": "/",
        "max_age": _COOKIE_MAX_AGE_SECONDS,
    }
    assert cookie_kwargs(POLICIES["local"])["secure"] is False
    assert cookie_kwargs(POLICIES["local"])["httponly"] is True


def test_the_cookie_is_never_given_a_domain():
    """Host-only, always. `*.ts.net` and `*.trycloudflare.com` are shared
    parents: your machine's neighbours under them are OTHER PEOPLE'S machines.
    A Domain attribute would send this credential to strangers."""
    for policy in POLICIES.values():
        assert "domain" not in cookie_kwargs(policy)


def test_the_cookie_survives_closing_the_browser():
    """A paired device STAYS paired (Milestone 6a spec): without an explicit
    `max_age` this is a session cookie, and closing the browser would un-pair
    the phone while its `Device` record lives on in the vault -- the
    credential and the record disagree, and she re-pairs for no security gain.
    Every policy gets the same long lifetime; the constant, not a magic
    number, is what a change to the lifetime should have to touch."""
    for policy in POLICIES.values():
        assert cookie_kwargs(policy)["max_age"] == _COOKIE_MAX_AGE_SECONDS


def test_a_real_pair_sets_a_long_lived_cookie(tmp_path):
    """The helper being right is not the same as the route using it: this
    exercises `POST /v1/pair` end to end and reads the `Set-Cookie` header
    the client actually received, rather than only asserting on
    `cookie_kwargs`'s return value."""
    vault = TokenVault(tmp_path)
    store = PairCodeStore()
    code = store.mint("p", frozenset({Capability.CHAT_SEND})).code
    client = build_api_client(build_fake_runtime(), vault,
                              policies={LOCAL_PORT: "local"}, pair_store=store)
    r = client.post("/v1/pair", json={"code": code})
    assert r.status_code == 204
    set_cookie = r.headers.get("set-cookie", "")
    assert f"Max-Age={_COOKIE_MAX_AGE_SECONDS}" in set_cookie, set_cookie


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


def test_a_cross_site_refusal_does_not_reveal_whether_the_cookie_is_valid(tmp_path):
    """The cross-site checks run before `verify()`, so they cannot double as a
    credential oracle.

    Run afterwards, a cross-site request holding a *valid* cookie answered 403
    while the identical request holding a junk one answered 401 -- which any
    page could have used to ask "does this browser have a working session for
    that daemon?" without ever being able to use it. Both are now 403.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    evil = {"Origin": "https://evil.example"}

    client.cookies.set(COOKIE_NAME, token)
    with_valid = client.get("/v1/status", headers=evil)
    client.cookies.set(COOKIE_NAME, "not-a-real-token")
    with_junk = client.get("/v1/status", headers=evil)

    assert with_valid.status_code == with_junk.status_code == 403
    assert with_valid.json() == with_junk.json()


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


# ─── nothing that acts is reachable on a read-only ceiling ───────────────
_PATH_PARAMS = {
    "{scope}": "knowledge",
    "{conversation_id}": "c1",
    "{item_id}": "k1",
    "{command_id}": "volume_up",
    "{kind}": "voice",
    "{device_id}": "d1",
}

# The one mutating operation that is deliberately not gated on a capability
# at all, so the ceiling has nothing to narrow and the sweep below has
# nothing to assert. `POST /v1/pair` is how a device gets a credential --
# demanding one would make it unable to issue one -- so it is reachable on
# every listener by construction, including a read-only ceiling. That is not
# a hole in the ceiling: pairing does not act on the machine, and the grants
# the new device receives come from a code only a loopback admin could mint,
# then pass through `effective()` on every later request exactly like any
# other device's. Its own coverage is tests/test_api_pairing_routes.py.
_ANONYMOUS_OPERATIONS = frozenset({
    ("POST", "/v1/pair"),
})


# Every mutating operation this daemon serves, by name. Asserted as an exact
# set rather than as a count or a floor, because the sweep's entire value is
# completeness: a filter that quietly stops matching real routes would still
# pass a `> 0` check on whatever it happened to keep. Adding a route here is
# meant to be a conscious act -- the failure message is where somebody is
# forced to ask whether the new one is gated at the right tier.
_MUTATING_OPERATIONS = frozenset({
    ("DELETE", "/v1/enrollment/{kind}/{item_id}"),
    ("DELETE", "/v1/files"),
    ("DELETE", "/v1/memory"),
    ("DELETE", "/v1/memory/{scope}/{item_id}"),
    ("PATCH", "/v1/personality"),
    ("PATCH", "/v1/settings"),
    ("POST", "/v1/abort"),
    ("POST", "/v1/backup/restore"),
    ("POST", "/v1/backup/run"),
    ("POST", "/v1/backup/unlock"),
    ("POST", "/v1/chat"),
    ("POST", "/v1/commands/{command_id}/run"),
    ("POST", "/v1/files/rename"),
    ("POST", "/v1/personality/reset"),
    # Enrollment and revocation. Both are `require_admin(SYSTEM_CONTROL)`, so
    # they are refused here twice over -- the ceiling carries no
    # SYSTEM_CONTROL, and `quick` is not an admin listener.
    ("POST", "/v1/pair/code"),
    ("DELETE", "/v1/devices/{device_id}"),
})


def _sweep_mutations_are_refused(client) -> set[tuple[str, str]]:
    """Call every mutating operation the schema knows about, and report all of
    the ones that are not refused -- not just the first.

    Returns the set of `(verb, path)` it actually swept, so a caller can pin
    the coverage as well as the outcome.

    Enumerated from `app.openapi()` for the same reason
    `_sweep_v1_routes_require_auth` in test_api_auth.py is: it walks the
    *effective* routes regardless of how they were registered, so it cannot be
    dodged by choice of registration call.

    Violations are collected and asserted once at the end. Asserting inside
    the loop reports whichever offender happens to sort first and hides the
    rest, which for a sweep whose job is completeness means somebody
    re-running it once per bug -- the four real violations would have been
    found one at a time.

    No request body is sent. A capability dependency raises before FastAPI
    ever validates a body, so a route that refuses correctly answers 403
    whether or not one is supplied -- and a route that does *not* refuse
    betrays itself by answering anything else (the four real violations
    answered 200, 200, 422 and 404).
    """
    swept: set[tuple[str, str]] = set()
    violations: list[str] = []
    for path, operations in client.app.openapi().get("paths", {}).items():
        if not path.startswith("/v1"):
            continue
        concrete = path
        for placeholder, value in _PATH_PARAMS.items():
            concrete = concrete.replace(placeholder, value)
        for method in operations:
            verb = method.upper()
            if verb not in ("POST", "PUT", "PATCH", "DELETE"):
                continue
            if (verb, path) in _ANONYMOUS_OPERATIONS:
                # Nothing to refuse: this one never reads a credential, so
                # there is no grant for the ceiling to narrow. See the
                # comment on `_ANONYMOUS_OPERATIONS` above.
                continue
            # A fresh limiter per probe: several of these routes carry their
            # own tighter budget, and this sweep must fail on authorization,
            # never on a 429 that happens to look like a refusal.
            client.app.state.auth.limiter = RateLimiter()
            response = client.request(verb, concrete, headers={CSRF_HEADER: "1"})
            swept.add((verb, path))
            if response.status_code != 403:
                violations.append(
                    f"{verb} {path} answered {response.status_code} on a "
                    "read-only listener"
                )
    assert not violations, "\n".join(violations)
    return swept


def test_no_mutation_is_reachable_on_a_read_only_ceiling(tmp_path):
    """The `quick` ceiling claims to be read-only *on the write side*. This is
    that half of the claim, executed.

    `policy.py` says a device on a Cloudflare quick tunnel is "limited to
    reading history and status through this transport, never to acting". The
    "never to acting" half lives entirely in which capability each route
    happens to demand -- data spread across five route modules, readable from
    no single place -- and four routes were quietly violating it: a real cloud
    upload, two personality writes and a memory delete, all gated on `CHAT`.

    **What this does not check.** It sweeps the 14 mutating operations of 30,
    and says nothing about what the surviving `{OBSERVE}` ceiling lets a
    caller *read*. The ceiling's other stated purpose is limiting disclosure
    over the one listener a third party can observe -- `SCREEN` is excluded
    from `quick` for what Cloudflare could see, not for what an attacker could
    do -- and a read route that discloses too much would pass this sweep
    untouched. Method-based sweeping cannot express that;
    `test_no_stored_data_route_is_reachable_on_a_read_only_ceiling` below is
    that separate thing, added once the `CHAT` split proved the gap was real.

    `OBSERVE` must mean "may watch": a device holding every capability in
    existence, arriving on a listener whose ceiling is `{OBSERVE}`, must not
    be able to change anything.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "quick"})
    client.cookies.set(COOKIE_NAME, token)
    assert _sweep_mutations_are_refused(client) == _MUTATING_OPERATIONS


def test_the_read_only_sweep_catches_a_mutation_gated_on_a_read(tmp_path):
    """Guard for the sweep itself, not for the app: a new route gated on a
    read capability -- the exact mistake the four real ones made -- must be
    caught."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "quick"})
    client.cookies.set(COOKIE_NAME, token)

    leaky = APIRouter()

    @leaky.post("/leaky")
    async def leaky_handler(_=Depends(require(Capability.OBSERVE))):
        return {"ok": True}

    client.app.include_router(leaky, prefix="/v1")

    with pytest.raises(AssertionError, match="answered 200"):
        _sweep_mutations_are_refused(client)


# ─── nothing she stored is readable on a read-only ceiling ──────────────
# The disclosure-side companion to the sweep above, which says in its own
# docstring that it checks nothing about what `{OBSERVE}` lets a caller
# *read*. Same shape and same reason: three exact sets rather than a spot
# check, so a GET route added later cannot slip through unclassified --
# whoever adds it is forced to say whether it is observation of her, a read
# of what she stored, or something this transport carries at all.
_OBSERVATION_OPERATIONS = frozenset({
    ("GET", "/v1/backup"),        # whether backups run, not what is in one
    ("GET", "/v1/commands"),      # what she can be asked to do
    ("GET", "/v1/personality"),   # how she is configured to answer
    ("GET", "/v1/settings"),      # the same, in registry form
    ("GET", "/v1/status"),
    ("GET", "/v1/telemetry"),
    ("GET", "/v1/session"),       # who the caller is, not anything she stored
})

_STORED_DATA_OPERATIONS = frozenset({
    ("GET", "/v1/chat/conversations"),
    ("GET", "/v1/chat/conversations/{conversation_id}"),
    ("GET", "/v1/enrollment"),    # the names of the people she recognises
    ("GET", "/v1/memory/knowledge"),
    ("GET", "/v1/memory/preferences"),
    ("GET", "/v1/memory/procedures"),
})

# Refused on `quick` for a reason that predates this split: its ceiling
# carries neither FILES nor SYSTEM_CONTROL. Listed so the completeness
# assertion below can be an equality rather than a subset.
_OFF_THIS_TRANSPORT_ENTIRELY = frozenset({
    ("GET", "/v1/audit"),
    # Who holds a credential to this machine: the security configuration, the
    # same class of fact as the audit log, and refused here for the same two
    # reasons -- SYSTEM_CONTROL is not in this ceiling, and `quick` is not an
    # admin listener.
    ("GET", "/v1/devices"),
    ("GET", "/v1/files"),
    ("GET", "/v1/files/content"),
    ("GET", "/v1/files/roots"),
})


def _sweep_reads(client) -> dict[tuple[str, str], int]:
    """Call every GET operation the schema knows about and report each status.

    Enumerated from `app.openapi()` for the same reason the mutation sweep is:
    it walks the effective routes regardless of how they were registered.

    No query string is sent to the routes that require one. A capability
    dependency raises before FastAPI validates request params -- sub
    dependencies are solved before the current dependant's own arguments --
    so a route that refuses correctly answers 403 whether or not its `path=`
    is supplied, and one that does not betrays itself with a 422.
    """
    statuses: dict[tuple[str, str], int] = {}
    for path, operations in client.app.openapi().get("paths", {}).items():
        if not path.startswith("/v1"):
            continue
        concrete = path
        for placeholder, value in _PATH_PARAMS.items():
            concrete = concrete.replace(placeholder, value)
        for method in operations:
            if method.upper() != "GET":
                continue
            # Fresh limiter per probe, as in the mutation sweep: this must
            # fail on authorization, never on a 429 that resembles a refusal.
            client.app.state.auth.limiter = RateLimiter()
            statuses[("GET", path)] = client.get(concrete).status_code
    return statuses


def test_no_stored_data_route_is_reachable_on_a_read_only_ceiling(quick_client):
    """The companion to Task 5's mutation sweep, for the disclosure side that
    sweep explicitly does not cover. Enumerate the GET operations and assert
    the transcript and knowledge routes refuse on `quick`."""
    for path in ("/v1/chat/conversations", "/v1/memory/knowledge"):
        assert quick_client.get(path).status_code == 403

    statuses = _sweep_reads(quick_client)
    assert set(statuses) == (_OBSERVATION_OPERATIONS | _STORED_DATA_OPERATIONS
                             | _OFF_THIS_TRANSPORT_ENTIRELY), (
        "a GET operation appeared or vanished; classify it before this passes"
    )
    stored_leaks = {op: code for op, code in statuses.items()
                    if op in _STORED_DATA_OPERATIONS and code != 403}
    assert not stored_leaks, f"stored data readable on quick: {stored_leaks}"
    assert {op for op, code in statuses.items() if code == 200} == _OBSERVATION_OPERATIONS


# ─── the commands catalogue may only name capabilities that mean "may act" ──
def _catalogue():
    from assistant.actions.studio_runtime_system import LiveCommandRuntime
    return LiveCommandRuntime._CATALOGUE


def test_no_command_declares_a_read_only_grant():
    """`run_command` trusts the catalogue's `required_grant`. A command that
    declared a read capability would be executable by a device deliberately
    issued read-only -- the read/write split undone through a different door.
    The catalogue may only name capabilities that mean 'may act'."""
    actionable = {Capability.CHAT_SEND, Capability.FILES,
                  Capability.SYSTEM_CONTROL, Capability.SCREEN}
    readonly = {Capability.OBSERVE, Capability.RECALL}
    for command in _catalogue():
        grant = Capability(command.required_grant)
        assert grant in actionable, f"{command.command_id} declares {grant}"
        assert grant not in readonly, command.command_id
