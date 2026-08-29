"""Every route, on every listener, with a maximal token: the reachability matrix.

Spec `.superpowers/sdd/2026-08-16-milestone6b-transports/spec.md` §2.1, §2.2.

Milestone 6b serves **one** ASGI app on three sockets and resolves what a
request may do from the local port it was accepted on. That makes "which
routes does each listener actually answer?" a table -- and a table nobody
wrote down is a table nobody can audit. This file is that table, and it is
enforced from both ends:

* every row is exercised against a real client on that listener's port, so a
  ceiling that stops narrowing shows up as a route that suddenly answers 200;
* every route the app declares must HAVE a row, so a route added by a later
  task and never classified is a **failure**, not a silent gap.

The token used everywhere holds `frozenset(Capability)` -- every capability
there is. That is deliberate and it is what makes the table mean something:
whatever a request in this file cannot reach, the **listener** refused. The
device never did.

Deliberately **not** derived from `security.py`'s dependencies by reflection.
A matrix computed from the same code it is checking agrees with that code by
construction, including when both are wrong; these rows are written out by
hand so that changing a route's gate has to be a deliberate edit here too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from assistant.io.api.app import create_app
from assistant.io.api.policy import POLICIES, effective
from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import ApiTestClient
from tests.fakes.studio_runtime import build_fake_runtime

# The three fixed ports of spec §2.2, spelled out rather than derived from
# `listeners.port_for`: this file is about what each *socket* answers, so the
# number a request is sent to has to be visible in the test.
LISTENER_PORTS: dict[str, int] = {
    "local": 8787,
    "tailnet": 8788,
    "funnel": 8789,
    # The browser extension's WebSocket. It appears here for the same reason
    # every other listener does: this file is the reachability table, and a
    # socket the daemon binds but the table omits is a socket nobody audits.
    # Its ceiling is empty, so every capability-gated row below is REFUSED --
    # which is the point. What it does still answer (the rows gated on
    # nothing) is now written down rather than assumed.
    "extension": 8790,
}

# The registry every app in this file is built with -- all three listeners
# declared, exactly as a fully-started 6b daemon has them. Only the port a
# given client talks to changes.
ALL_LISTENERS: dict[int, str] = {port: name for name, port in LISTENER_PORTS.items()}

DEV_ORIGINS = ["http://localhost:3000"]

# The one refusal a listener ceiling produces. `require()`, `require_admin()`
# and `routes/commands.py`'s own per-command check all raise the identical
# 403 "capability not granted" -- deliberately indistinguishable, so a remote
# session cannot map out which of its capabilities are transport-limited and
# which are grant-limited (see `security.py`'s `_NOT_ADMIN`).
REFUSED = 403

# `/v1/events` is a WebSocket handshake, which has no HTTP status to put in a
# table. It is covered by `tests/test_api_events.py` (unauthenticated connect,
# invalid token, capability at the handshake) and by the per-listener policy
# lookup in `app.py`'s own handler. Listed here rather than merely skipped so
# that a *second* socket route added later fails
# `test_the_event_socket_is_still_the_only_websocket_route` and has to be
# classified by whoever adds it.
# `/drover` is the browser extension's socket, added by the Drover tier. Its
# per-listener behaviour is decided and pinned in
# `tests/test_extension_ws.py::test_the_socket_is_refused_on_every_other_listener`:
# it answers only on the `extension` listener and closes on every other one
# before accepting. It is not in MATRIX because a socket has no HTTP status,
# which is the same reason `/v1/events` is not.
WEBSOCKET_ROUTES = frozenset({"/v1/events", "/drover"})

# The Studio front-end's catch-all (`ui.py`'s `mount_ui`), recorded rather
# than covered, because it is absent from every app in this file: these apps
# are built with no `ui_bundle`, so no `/{ui_path:path}` route is registered
# at all, and `include_in_schema=False` keeps it out of the sweep even when
# one is.
#
# **Intended answer, stated so it is a decision rather than a silence:** the
# UI is deliberately unauthenticated and reachable on **every** listener,
# including `funnel`. A page is not a capability -- it carries no
# credential of its own, and the credential a browser then presents is
# narrowed by that listener's ceiling on every request the page makes, which
# is what the rest of this table is about. It is not ungated either: it sits
# *inside* `HostGate`, so DNS rebinding is refused there, and it answers an
# API-shaped 404 for any unrouted `/v1` path rather than an HTML shell.
# `tests/test_api_ui_serving.py` is the compensating coverage, and
# `test_api_auth.py`'s `test_no_route_hides_from_the_schema_sweep` already
# carries it as a named exemption.
UNCLASSIFIED_BY_DESIGN = frozenset({"GET|HEAD /{ui_path:path}"})


# ─── the table ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Row:
    """One route, and what decides whether a listener carries it.

    `capability` is the grant the route is gated on; `None` means the route is
    gated on `authenticate` alone (or on nothing at all, for `POST /v1/pair`),
    so no ceiling can withhold it. `admin` marks the routes behind
    `require_admin`, which loopback alone satisfies.

    `allowed` is the status the route answers when the listener does carry it
    -- against `tests/fakes/studio_runtime.py`, so it is that fixture's answer
    and not a claim about production data.
    """

    method: str
    path: str                       # the OpenAPI path template
    url: str                        # a concrete URL matching that template
    capability: Capability | None
    admin: bool = False
    allowed: int = 200
    body: dict | None = field(default=None)
    # A third axis, and the matrix is what found it: `POST /v1/session/cookie`
    # is gated on no capability at all and is still refused off loopback,
    # because a listener that reads the cookie and nothing else has no second
    # channel for the exchange to move a credential *from* -- and offering it
    # anyway would put a `Set-Cookie` route on a tunnel for no gain. Refused
    # with `UNAUTHORIZED`, not a 403: the route is gated on no capability, so
    # "capability not granted" would be a lie.
    bearer_only: bool = False
    # A fourth axis, and the extension listener is what found it: a route that
    # consults no credential at all. Until a listener existed whose ceiling
    # narrows every device to nothing, "gated on no capability" and "reached
    # without authenticating" behaved identically, so nothing distinguished
    # them. On such a listener they diverge: the authenticated-but-ungated
    # rows are refused with 401 at authentication, and only the genuinely
    # credential-free route still answers.
    unauthenticated: bool = False


MATRIX: tuple[Row, ...] = (
    # ── no ceiling can withhold these: `authenticate` alone, or nothing ──
    Row("GET", "/v1/session", "/v1/session", None),
    Row("POST", "/v1/session/cookie", "/v1/session/cookie", None, allowed=204,
        bearer_only=True),
    # The one unauthenticated write in the API. 401 is the *reachable*
    # answer -- there is no live pair code, and `pair_device` refuses a
    # wrong, expired, used or malformed one identically. What matters here
    # is that it answers the same on all three listeners: pairing over a
    # tunnel is the expected path for a phone with no Tailscale.
    Row("POST", "/v1/pair", "/v1/pair", None, allowed=401,
        body={"code": "AAAA-BBBB"}),
    # `GET /v1/listener` needs no credential at all -- it is what a caller
    # consults *before* it has one -- so no ceiling can withhold it either;
    # it answers 200 on every listener.
    #
    # `unauthenticated=True` is what separates it from a row like
    # `GET /v1/session`, which also carries no capability but does run through
    # `authenticate`. Until the extension listener arrived nothing distinguished
    # the two, because every listener authenticated somebody; a listener whose
    # ceiling narrows to nothing refuses even the capability-free authenticated
    # rows, and only a route that never authenticates at all still answers.
    Row("GET", "/v1/listener", "/v1/listener", None, unauthenticated=True),

    # ── OBSERVE: watching her work ──────────────────────────────────────
    Row("GET", "/v1/status", "/v1/status", Capability.OBSERVE),
    Row("GET", "/v1/telemetry", "/v1/telemetry", Capability.OBSERVE),
    Row("GET", "/v1/settings", "/v1/settings", Capability.OBSERVE),
    Row("GET", "/v1/personality", "/v1/personality", Capability.OBSERVE),
    Row("GET", "/v1/commands", "/v1/commands", Capability.OBSERVE),
    Row("GET", "/v1/backup", "/v1/backup", Capability.OBSERVE),

    # ── RECALL: what she has stored ─────────────────────────────────────
    Row("GET", "/v1/memory/knowledge", "/v1/memory/knowledge", Capability.RECALL),
    Row("GET", "/v1/memory/preferences", "/v1/memory/preferences", Capability.RECALL),
    Row("GET", "/v1/memory/procedures", "/v1/memory/procedures", Capability.RECALL),
    Row("GET", "/v1/chat/conversations", "/v1/chat/conversations", Capability.RECALL),
    Row("GET", "/v1/chat/conversations/{conversation_id}",
        "/v1/chat/conversations/c1", Capability.RECALL),
    Row("GET", "/v1/enrollment", "/v1/enrollment", Capability.RECALL),

    # ── CHAT_SEND: driving a turn ───────────────────────────────────────
    Row("POST", "/v1/chat", "/v1/chat", Capability.CHAT_SEND, allowed=202,
        body={"text": "hello"}),
    Row("POST", "/v1/abort", "/v1/abort", Capability.CHAT_SEND),
    Row("DELETE", "/v1/memory/{scope}/{item_id}",
        "/v1/memory/preferences/reading_pace", Capability.CHAT_SEND),

    # ── FILES ───────────────────────────────────────────────────────────
    Row("GET", "/v1/files/roots", "/v1/files/roots", Capability.FILES),
    Row("GET", "/v1/files", "/v1/files?path=desktop", Capability.FILES),
    Row("GET", "/v1/files/content", "/v1/files/content?path=desktop/notes.md",
        Capability.FILES),
    Row("POST", "/v1/files/rename", "/v1/files/rename", Capability.FILES,
        body={"path": "desktop/notes.md", "newName": "renamed.md"}),
    Row("DELETE", "/v1/files", "/v1/files", Capability.FILES,
        body={"path": "desktop/notes.md"}),

    # ── SCREEN, reached through the command catalogue ───────────────────
    # `run_command` is gated on `authenticate` alone and then on the
    # *command's own* `required_grant`, so the capability in this row is a
    # property of `screenshot` (SCREEN), not of the route. That is exactly
    # why the row names it: the ceiling still decides, one indirection later.
    Row("POST", "/v1/commands/{command_id}/run", "/v1/commands/screenshot/run",
        Capability.SCREEN),

    # ── SYSTEM_CONTROL: changing the machine ────────────────────────────
    Row("DELETE", "/v1/memory", "/v1/memory", Capability.SYSTEM_CONTROL),
    Row("PATCH", "/v1/settings", "/v1/settings", Capability.SYSTEM_CONTROL,
        body={"changes": {"tts_speed": 1.2}}),
    Row("PATCH", "/v1/personality", "/v1/personality", Capability.SYSTEM_CONTROL,
        body={"base": "dry"}),
    Row("POST", "/v1/personality/reset", "/v1/personality/reset",
        Capability.SYSTEM_CONTROL),
    Row("POST", "/v1/backup/run", "/v1/backup/run", Capability.SYSTEM_CONTROL),
    Row("POST", "/v1/backup/restore", "/v1/backup/restore",
        Capability.SYSTEM_CONTROL,
        body={"recoveryPhrase": "one two three four five six seven eight"}),
    Row("POST", "/v1/backup/unlock", "/v1/backup/unlock",
        Capability.SYSTEM_CONTROL,
        body={"recoveryPhrase": "one two three four five six seven eight"}),
    Row("DELETE", "/v1/enrollment/{kind}/{item_id}", "/v1/enrollment/voice/v1",
        Capability.SYSTEM_CONTROL),

    # ── admin: this daemon's own credentials, loopback alone ────────────
    Row("GET", "/v1/audit", "/v1/audit", Capability.SYSTEM_CONTROL, admin=True),
    Row("GET", "/v1/devices", "/v1/devices", Capability.SYSTEM_CONTROL, admin=True),
    Row("DELETE", "/v1/devices/{device_id}", "/v1/devices/{victim}",
        Capability.SYSTEM_CONTROL, admin=True),
    Row("POST", "/v1/pair/code", "/v1/pair/code", Capability.SYSTEM_CONTROL,
        admin=True, body={"label": "phone", "grants": ["observe"]}),

    # A raise is this milestone's one privilege-widening route, minted at the
    # keyboard -- same `admin=True` as the two rows above, for the identical
    # reason: `policy.admin` is `local` alone, so no other listener even
    # reaches the check that would answer either 409 or 404. `allowed` here
    # is this harness's own answer, not production's: `create_app` in
    # `_client_on` wires no `transports` manager, so `raise_device_ceiling`
    # always finds nothing running and refuses with 409 before it can mint a
    # raise; the fresh `RaiseStore` behind `revoke_device_raise` never holds
    # one for the victim, so that one answers 404. Both are "the admin gate
    # let the request through", not a claim that a raise ever really lands
    # in this fixture.
    Row("POST", "/v1/devices/{device_id}/raise", "/v1/devices/{victim}/raise",
        Capability.SYSTEM_CONTROL, admin=True, allowed=409,
        body={"transport": "tailnet", "capabilities": ["execute"],
              "minutes": 30, "reason": "fixing the build"}),
    Row("DELETE", "/v1/devices/{device_id}/raise", "/v1/devices/{victim}/raise",
        Capability.SYSTEM_CONTROL, admin=True, allowed=404),

    # Task 13: which doors exist, opening one, closing one. Same admin gate as
    # the two device rows above, for the identical reason -- managing
    # transports is the same class of thing as managing devices, and both
    # stay at the keyboard. `_client_on` builds every app in this file with no
    # `transports` manager wired (a sibling task's job), so the `allowed`
    # values below are this harness's own fail-closed answers, exactly as the
    # raise route's two rows above are: the listing always has something to
    # report (empty, with no manager), starting has nothing to start against
    # (503), and stopping finds nothing running to stop (404) -- none of that
    # is a claim about what a wired manager would do.
    Row("GET", "/v1/transports", "/v1/transports",
        Capability.SYSTEM_CONTROL, admin=True),
    Row("POST", "/v1/transports/{name}", "/v1/transports/tailnet",
        Capability.SYSTEM_CONTROL, admin=True, allowed=503),
    Row("DELETE", "/v1/transports/{name}", "/v1/transports/tailnet",
        Capability.SYSTEM_CONTROL, admin=True, allowed=404),
)


def expected_status(row: Row, policy_name: str) -> int:
    """What this listener must answer for this route.

    Two ways to be refused with a 403, one answer. A capability outside the
    listener's ceiling is withheld by `effective()`; an admin route on a
    non-admin listener is withheld by `require_admin`. Both are 403 with the
    identical body, on purpose -- a remote session must not be able to tell
    which of the two gates stopped it.

    The bearer check runs last because that is where it runs in the request:
    it is the route's own refusal, after `authenticate()` and after any
    capability gate had its say.
    """
    policy = POLICIES[policy_name]

    # A listener whose ceiling narrows a maximal device to nothing refuses at
    # authentication, with 401, before any capability gate is consulted --
    # `security.py`'s `if not grants` and the comment above it. That is not an
    # implementation detail to route around here: a device that authenticated
    # with an empty grant set would still reach every route gated on
    # `authenticate` alone, and the 404-vs-403 split in `run_command` would
    # become an oracle for which command ids exist.
    #
    # Derived from `effective()` rather than restated as `not policy.ceiling`,
    # so a future policy that carries something only under a raise is judged by
    # the same arithmetic the request is.
    if row.unauthenticated:
        # No credential is consulted, so no ceiling can withhold it.
        return row.allowed

    if not effective(frozenset(Capability), policy):
        return 401

    if row.capability is not None and row.capability not in policy.ceiling:
        return REFUSED
    if row.admin and not policy.admin:
        return REFUSED
    if row.bearer_only and not policy.allow_bearer:
        return 401
    return row.allowed


# ─── driving one listener ────────────────────────────────────────────────

def _client_on(vault: TokenVault, token: str, policy_name: str) -> ApiTestClient:
    """A client whose requests really do arrive on that listener's port.

    A **fresh app per policy**, sharing only the vault. The apps would
    otherwise share one `FakeFileRuntime`/`FakeMemoryRuntime`, and the
    mutating rows (rename, delete, forget) would then answer 200 on the first
    listener that carries them and 404 on the next -- an ordering artefact
    dressed up as a policy difference. Every app still declares all four
    listeners in its registry, exactly as a running daemon does.
    """
    app = create_app(
        build_fake_runtime(), vault,
        origins=list(DEV_ORIGINS),
        listener_policies=dict(ALL_LISTENERS),
    )
    client = ApiTestClient(app, base_url=f"http://127.0.0.1:{LISTENER_PORTS[policy_name]}")
    client.cookies.set(COOKIE_NAME, token)
    return client


def _send(client: ApiTestClient, row: Row, victim_id: str):
    # The CSRF header on every request, not only the writes: it is required on
    # a cookie-authenticated write (`_AMBIENT_METHODS`) and ignored elsewhere,
    # so sending it always keeps this helper from having to re-derive the rule
    # that `security.py` already owns.
    return client.request(row.method, row.url.replace("{victim}", victim_id),
                          json=row.body, headers={CSRF_HEADER: "1"})


@pytest.fixture()
def vault_and_token(tmp_path) -> tuple[TokenVault, str, str]:
    """One vault, a maximal token, and a second device to revoke.

    The victim exists so `DELETE /v1/devices/{device_id}` has a real row to
    cut -- revoking the caller's own credential mid-matrix would make the
    next request in the same test 401 for a reason that has nothing to do
    with any listener.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    vault.issue("victim", frozenset(Capability))
    victim = next(d for d in vault.devices() if d.label == "victim")
    return vault, token, victim.device_id


# ─── the matrix itself ───────────────────────────────────────────────────

@pytest.mark.parametrize("row", MATRIX, ids=lambda r: f"{r.method} {r.path}")
def test_the_listener_matrix_holds(row: Row, vault_and_token):
    vault, token, victim_id = vault_and_token
    for policy_name in LISTENER_PORTS:
        client = _client_on(vault, token, policy_name)
        response = _send(client, row, victim_id)
        assert response.status_code == expected_status(row, policy_name), (
            f"{row.method} {row.path} on the {policy_name} listener answered "
            f"{response.status_code}, expected "
            f"{expected_status(row, policy_name)}. Either the route's gate "
            f"changed or this row is stale -- fix whichever is wrong, but do "
            f"not relax the row to match the code without deciding that the "
            f"new answer is the one you want on a socket the internet can "
            f"reach."
        )


# ─── completeness, from both ends ────────────────────────────────────────

def _declared_http_routes(app) -> set[tuple[str, str]]:
    """Every `(method, path)` this app actually serves over HTTP.

    Read from `app.openapi()`, not from `app.routes`. FastAPI 0.141.1 wraps an
    included router in an internal `_IncludedRouter` node whose `path` is
    `None` and whose `methods` are empty, so a flat sweep of `app.routes` sees
    **zero** API routes here and every completeness assertion below would pass
    vacuously. `tests/test_api_auth.py` already walks the resolved schema for
    exactly this reason; the socket route, which has no schema representation
    at all, is checked separately.
    """
    return {(method.upper(), path)
            for path, operations in app.openapi()["paths"].items()
            for method in operations}


@pytest.fixture()
def app_for_introspection(tmp_path):
    return create_app(build_fake_runtime(), TokenVault(tmp_path),
                      origins=list(DEV_ORIGINS),
                      listener_policies=dict(ALL_LISTENERS))


def test_every_route_has_a_row_in_the_matrix(app_for_introspection):
    """A route added later with no row is a failure, not a silent gap.

    Three more tasks in this milestone add routes to this app. Each one adds
    its rows here, and decides -- explicitly, in the table above -- what every
    listener answers for it. That decision is the point: a route that reaches
    the machine and is never classified would ship reachable over a URL the
    open internet holds, and nothing else in the suite would notice.
    """
    declared = {(row.method, row.path) for row in MATRIX}
    actual = {pair for pair in _declared_http_routes(app_for_introspection)
              if pair[1].startswith("/v1")}
    missing = actual - declared
    assert not missing, (
        "these routes have no row in MATRIX (tests/test_6b_listener_matrix.py): "
        f"{sorted(missing)}. Add one per route, naming the capability it is "
        "gated on, whether it is admin-only, and the status it answers when "
        "the listener carries it -- then check what that makes it answer on "
        "`funnel`, which anyone holding the URL can reach."
    )


def test_no_matrix_row_names_a_route_that_does_not_exist(app_for_introspection):
    """The other direction. A row for a route that was renamed or removed
    would keep passing -- 404 is a status like any other -- while quietly
    covering nothing."""
    declared = {(row.method, row.path) for row in MATRIX}
    actual = _declared_http_routes(app_for_introspection)
    stale = declared - actual
    assert not stale, f"MATRIX rows for routes this app does not serve: {sorted(stale)}"


def _websocket_paths(routes, prefix: str = "") -> set[str]:
    """Every WebSocket path reachable from `routes`, prefixes applied.

    Recursive, and that is the whole point. A socket registered directly on
    the app *is* a flat `WebSocketRoute` in `app.routes` -- which is what
    `/v1/events` happens to be -- but one registered on an included router is
    reachable only through that router's `_IncludedRouter` wrapper, at
    `original_router.routes`, with the prefix carried on `include_context`.
    Verified against this checkout: a websocket added to any of the ten
    included routers is invisible to a flat sweep, which would leave the guard
    below as green and as vacuous as the `app.routes` sweep this file already
    corrects for HTTP routes. The same mistake, one function apart.
    """
    from starlette.routing import WebSocketRoute

    found: set[str] = set()
    for route in routes:
        if isinstance(route, WebSocketRoute):
            found.add(f"{prefix}{route.path}")
            continue
        included = getattr(route, "original_router", None)
        if included is None:
            continue
        context = getattr(route, "include_context", None)
        found |= _websocket_paths(getattr(included, "routes", ()),
                                  prefix + getattr(context, "prefix", ""))
    return found


def test_the_event_socket_is_still_the_only_websocket_route(app_for_introspection):
    """`app.openapi()` cannot see a WebSocket route, so a second one added
    later would slip past `test_every_route_has_a_row_in_the_matrix` entirely.
    Socket routes are swept off `app.routes` instead -- recursively, through
    the included routers as well as the app itself -- and pinned by name."""
    sockets = _websocket_paths(app_for_introspection.routes)
    assert sockets == set(WEBSOCKET_ROUTES), (
        f"the WebSocket routes changed: {sorted(sockets)}. A socket has no "
        "HTTP status to put in MATRIX, so decide what each listener does with "
        "it at the handshake (app.py resolves a policy there) and pin it in "
        "tests/test_api_events.py."
    )


def test_the_websocket_sweep_would_actually_see_a_route_added_on_a_router(
    app_for_introspection,
):
    """The guard on the guard above. `/v1/events` is registered directly on
    the app, so a flat sweep of `app.routes` finds it and the test passes --
    while a socket added the way every *other* route in this app is added
    (on a router, then `include_router`) would be invisible. Proved by
    registering exactly that and requiring the sweep to see it, at its
    prefixed path."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.websocket("/probe")
    async def probe(websocket) -> None:  # pragma: no cover - never connected
        pass

    app_for_introspection.include_router(router, prefix="/v1")
    assert "/v1/probe" in _websocket_paths(app_for_introspection.routes)


# ─── the table must not be vacuous ───────────────────────────────────────

def test_the_port_table_matches_the_one_the_daemon_actually_uses():
    """The literals above are duplicated from `listeners.py` on purpose -- this
    file is about what each *socket* answers, so the numbers a request is sent
    to have to be visible in the test rather than computed out of sight. That
    is only safe while the duplicate cannot drift, which is what this pins:
    every offset against `port_for`, and the base against the one setting the
    whole map is derived from."""
    import assistant.config as config
    from assistant.io.api.listeners import LISTENER_OFFSETS, port_for

    base = LISTENER_PORTS["local"]
    assert base == config.STUDIO_API_PORT
    assert LISTENER_PORTS == {name: port_for(name, base) for name in LISTENER_OFFSETS}


def test_the_matrix_is_not_all_permitted():
    """A guard on the guard. If every cell said "allowed", this file would be
    a large, green, expensive assertion that routing works -- and the ceilings
    it exists to hold could all be deleted without a single failure.

    "Refused" means any status other than the row's allowed one, not 403
    specifically: a listener that narrows to nothing refuses at authentication
    with 401, and counting only 403s would read that as refusing nothing at
    all -- the most closed listener in the table scoring as the most open.
    """
    for policy_name, policy in POLICIES.items():
        if policy_name == "local":
            continue
        refused = [row for row in MATRIX
                   if expected_status(row, policy_name) != row.allowed]
        assert refused, f"the {policy_name} listener refuses nothing in MATRIX"


def test_an_observe_only_listener_carries_nothing_but_observation(monkeypatch):
    """Spelled out separately from the ceiling itself, because this is the
    shape of listener whose plaintext a third party could read -- Milestone
    6b's `quick` transport (a Cloudflare tunnel, where Cloudflare terminates
    TLS) was the one real listener this narrow, and it was removed outright
    (no device could ever authenticate over it -- `policy.py`'s module
    docstring has the full argument). `tailnet` and `funnel` both carry more
    than OBSERVE, so neither can demonstrate "every row that is not
    OBSERVE-gated, or gated on nothing at all, is refused" on its own; this
    synthetic policy is what still proves the property against the real
    MATRIX rows.
    """
    from assistant.io.api.policy import ListenerPolicy
    monkeypatch.setitem(POLICIES, "observe_only", ListenerPolicy(
        name="observe_only", admin=False, allow_bearer=False,
        secure_cookie=True, ceiling=frozenset({Capability.OBSERVE}),
        raisable=frozenset(), pairable=True,
    ))
    for row in MATRIX:
        expected = expected_status(row, "observe_only")
        if row.capability not in (None, Capability.OBSERVE) or row.admin:
            assert expected == REFUSED, (
                f"{row.method} {row.path} is reachable on an OBSERVE-only "
                "listener, the shape a third party could read the plaintext of"
            )
        elif row.bearer_only:
            # Refused here too, but by the route rather than by the ceiling,
            # and with a different status. Named as its own case so the
            # reachable branch below can compare against the row's own allowed
            # status instead of merely "is not 403" -- which this row would
            # have satisfied while in fact being turned away.
            assert expected == 401, (
                f"{row.method} {row.path} is bearer-only and must be refused "
                f"off loopback, got {expected}"
            )
        else:
            assert expected == row.allowed, (
                f"{row.method} {row.path} is observation (or gated on "
                f"nothing) and must answer {row.allowed} on an OBSERVE-only "
                f"listener, not {expected}"
            )
