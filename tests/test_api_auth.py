"""Nothing is reachable without a token. Not even status."""
import pytest
from fastapi import APIRouter, Depends

from assistant.io.api.app import create_app
from assistant.io.api.security import COOKIE_NAME, RateLimiter, require
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import ApiTestClient, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime

ALL = frozenset(Capability)


@pytest.fixture()
def vault(tmp_path):
    return TokenVault(tmp_path)


@pytest.fixture()
def token(vault):
    return vault.issue("studio", ALL)


def _client(vault: TokenVault) -> ApiTestClient:
    """Build a client against a caller-supplied vault.

    The `client` fixture below covers the common case of one vault per test;
    the CHAT/CHAT_SEND split tests need to issue several tokens off vaults
    they construct themselves, so the app-building step is pulled out here
    rather than duplicated per test.
    """
    return build_api_client(build_fake_runtime(), vault)


@pytest.fixture()
def client(vault):
    return _client(vault)


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_status_requires_a_token(client):
    assert client.get("/v1/status").status_code == 401


def test_status_with_a_token_answers(client, token):
    response = client.get("/v1/status", headers=auth(token))
    assert response.status_code == 200
    assert response.json()["data"]["assistantName"] == "TENKA"


def test_unauthenticated_status_leaks_nothing(client):
    body = client.get("/v1/status").text.lower()
    for leak in ("tenka", "1.0.0", "gemini", "warm", "windows"):
        assert leak not in body


def test_a_wrong_token_is_rejected(client):
    assert client.get("/v1/status", headers=auth("wrong")).status_code == 401


def test_a_revoked_token_is_rejected(client, vault, token):
    device = vault.verify(token)
    vault.revoke(device.device_id)
    assert client.get("/v1/status", headers=auth(token)).status_code == 401


def test_wrong_and_revoked_are_indistinguishable(client, vault, token):
    device = vault.verify(token)
    vault.revoke(device.device_id)
    revoked = client.get("/v1/status", headers=auth(token))
    unknown = client.get("/v1/status", headers=auth("never-issued"))
    assert revoked.status_code == unknown.status_code
    assert revoked.json() == unknown.json()


def test_a_malformed_header_is_rejected(client, token):
    for header in ({"Authorization": token},
                   {"Authorization": "Basic " + token},
                   {"Authorization": "Bearer"},
                   {"Authorization": "Bearer  "}):
        assert client.get("/v1/status", headers=header).status_code == 401


def test_a_token_in_the_query_string_does_not_work(client, token):
    """Query strings are logged by every proxy. The header is the only channel."""
    assert client.get(f"/v1/status?token={token}").status_code == 401


# The operations that are deliberately reachable without a credential. This
# list is the record of what this daemon answers to a stranger, so it is kept
# short enough to read in one glance and every entry states why.
#
# ("POST", "/v1/pair") -- how a device gets a credential in the first place,
# and therefore the only unauthenticated write in the API. It cannot demand
# one without being unable to issue one. What stands in for a credential is
# the pair code in its body: ~40 bits, live for 180 seconds, single-use, at
# most one outstanding at a time, and mintable only from the loopback
# listener by a device holding SYSTEM_CONTROL. Its compensating coverage is
# tests/test_api_pairing_routes.py, which pins every one of those properties
# plus the global attempt budget that burns the outstanding code when it is
# spent. Anything added here needs the same: a named test file, not silent
# trust.
_ANONYMOUS_OPERATIONS = frozenset({
    ("POST", "/v1/pair"),
})


def _sweep_v1_routes_require_auth(client) -> int:
    """Call every /v1 route with no token; fail loudly if any answers publicly.

    Enumerated via `app.openapi()` rather than `app.routes`: FastAPI's
    `include_router` does not always append flat routes to `app.routes` --
    0.141.1 wraps an included router in an internal `_IncludedRouter` node
    that is a `BaseRoute` but not a `Route`/`APIRoute`, so a sweep that does
    `isinstance(route, Route)` over `app.routes` silently finds nothing for
    routes registered that way, no matter how they're authenticated. The
    OpenAPI schema is built by walking the *effective* routes regardless of
    how they were registered -- it is what /docs would render -- so it can't
    be dodged by choice of registration call. `openapi_url=None` only
    disables the public `/openapi.json` endpoint; building the schema
    in-process via `app.openapi()` still works and exposes nothing over the
    wire.
    """
    schema = client.app.openapi()
    checked = 0
    for path, operations in schema.get("paths", {}).items():
        if not path.startswith("/v1"):
            continue
        concrete = (path.replace("{scope}", "knowledge")
                        .replace("{conversation_id}", "c1")
                        .replace("{item_id}", "k1")
                        .replace("{command_id}", "volume_up")
                        .replace("{kind}", "voice")
                        .replace("{device_id}", "d1"))
        for method in operations:
            if method.upper() in ("HEAD", "OPTIONS"):
                continue
            if (method.upper(), path) in _ANONYMOUS_OPERATIONS:
                continue
            # A fresh limiter before every probe, not once before the loop:
            # the sweep's whole point is "every route answers 401/403 with no
            # token," and that guarantee must hold no matter how many routes
            # exist. Sharing one TestClient's source identity across N
            # sequential anonymous calls would otherwise make the Nth+1
            # (once N crosses RateLimiter._MAX_FAILURES) collide with the
            # limiter's own lockout and answer 429 instead -- a growing
            # route count breaking the auth guarantee's own test, not the
            # guarantee itself. Resetting per call keeps this sweep correct
            # at 13 routes today and however many Tasks 11-14 add, without
            # this test needing to know the limiter's internal thresholds or
            # the exact source key `authenticate()` computes.
            client.app.state.auth.limiter = RateLimiter()
            response = client.request(method.upper(), concrete)
            assert response.status_code in (401, 403), (
                f"{method.upper()} {path} answered {response.status_code} with no token"
            )
            checked += 1
    return checked


def test_every_registered_route_rejects_an_anonymous_call(client):
    checked = _sweep_v1_routes_require_auth(client)
    assert checked > 0, "no /v1 routes were checked — the sweep found nothing"


def test_the_anonymous_allow_list_names_only_real_operations(client):
    """A stale entry is a standing exemption for an operation that no longer
    exists -- or, worse, for a path a later task reuses for something else.
    The list only stays readable if it cannot accumulate ghosts."""
    schema = client.app.openapi()
    for verb, path in _ANONYMOUS_OPERATIONS:
        assert verb.lower() in schema["paths"].get(path, {}), (
            f"{verb} {path} is exempted from the auth sweep but is not a "
            "route this app serves"
        )


def test_the_one_anonymous_route_really_is_reachable_without_a_credential(client):
    """The other direction: the exemption must describe something true. A
    422 (a body it could not parse) proves the request reached the route
    rather than being turned away by an auth dependency."""
    assert client.post("/v1/pair", json={}).status_code == 422


def test_the_sweep_catches_a_route_registered_without_auth(vault):
    """Guard for the sweep itself, not for the app.

    A route added the ordinary `app.include_router()` way -- the exact call
    every later route module's brief instructs -- with no auth dependency at
    all, must still be caught. If this test ever fails, the sweep has
    regressed to something that can be dodged by registration style, which
    is exactly the gap `_sweep_v1_routes_require_auth` exists to close.
    """
    leaky_client = build_api_client(build_fake_runtime(), vault)

    leaky = APIRouter()

    @leaky.get("/leaky")
    async def leaky_handler():
        return {"ok": True}

    leaky_client.app.include_router(leaky, prefix="/v1")

    with pytest.raises(AssertionError, match="answered 200"):
        _sweep_v1_routes_require_auth(leaky_client)


def test_a_capability_it_lacks_is_refused(client, vault):
    """A token that lacks the capability a route requires must be refused.

    `/v1/status` requires OBSERVE and is the only shipped route, so an
    all-grants or OBSERVE-holding token can never exercise `require()`'s 403
    branch -- the original version of this test issued a read token against
    the read-gated route and asserted 200, which would pass identically if
    the capability check were deleted outright. A second, FILES-gated route
    is mounted here to force the refusal, with a mirror case proving the
    same route accepts a token that does hold the capability it demands.
    """
    probe = APIRouter()

    @probe.get("/probe")
    async def probe_handler(_=Depends(require(Capability.FILES))):
        return {"ok": True}

    client.app.include_router(probe, prefix="/v1")

    watcher = vault.issue("phone", frozenset({Capability.OBSERVE}))
    refused = client.get("/v1/probe", headers=auth(watcher))
    assert refused.status_code == 403

    files_holder = vault.issue("laptop", frozenset({Capability.FILES}))
    allowed = client.get("/v1/probe", headers=auth(files_holder))
    assert allowed.status_code == 200


# ─── RECALL vs CHAT_SEND: reading her conversations is not driving her ────
def test_a_read_only_device_cannot_send(tmp_path):
    """The whole point of the split: POST /v1/chat reaches all 38 intents, so
    reading history must not imply driving her."""
    vault = TokenVault(tmp_path)
    token = vault.issue("reader", frozenset({Capability.RECALL}))
    client = _client(vault)
    assert client.get("/v1/chat/conversations",
                      headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.post("/v1/chat", json={"text": "hi"},
                       headers={"Authorization": f"Bearer {token}"}).status_code == 403
    assert client.post("/v1/abort",
                       headers={"Authorization": f"Bearer {token}"}).status_code == 403


def test_a_sending_device_still_reads(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.RECALL, Capability.CHAT_SEND}))
    client = _client(vault)
    assert client.post("/v1/chat", json={"text": "hi"},
                       headers={"Authorization": f"Bearer {token}"}).status_code == 202


def test_the_events_socket_stays_a_read_gate(tmp_path):
    """app.py checks OBSERVE before accept(). The socket only streams, so it
    must NOT start demanding CHAT_SEND -- nor RECALL, which a watching device
    need not hold at all -- and a reader device keeps its live view."""
    vault = TokenVault(tmp_path)
    token = vault.issue("reader", frozenset({Capability.OBSERVE}))
    client = _client(vault)
    client.cookies.set(COOKIE_NAME, token)
    with client.websocket_connect("/v1/events") as ws:
        assert ws is not None


def test_openapi_is_not_public(client):
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code in (401, 404)


# `test_the_query_string_exception_is_only_the_socket` lived here: it pinned
# that exactly one route -- the event socket -- read its token from the query
# string. That exception no longer exists; the socket authenticates by cookie
# like everything else, so the property to pin inverted with it. Its successor
# is `test_no_source_file_reads_access_token` in test_api_cookie_auth.py,
# which asserts the string appears in *no* source file under io/api. Replaced
# rather than deleted, because a query-string credential reappearing is
# exactly the regression the original test existed to catch.


def test_a_bad_tenka_secret_fails_at_app_build_not_per_request(tmp_path, monkeypatch):
    """instance_secret() is uncached on the env-override path, so a wrong-length
    TENKA_SECRET would otherwise surface as a 500 on the first authenticated
    request instead of at startup. create_app() must resolve it eagerly."""
    monkeypatch.setenv("TENKA_SECRET", "ab" * 10)  # valid hex, only 10 bytes
    with pytest.raises(ValueError, match="TENKA_SECRET"):
        create_app(build_fake_runtime(), TokenVault(tmp_path),
                   origins=["http://localhost:3000"])


def _walk_every_registered_route(routes):
    """Recurse through FastAPI 0.141.1's `_IncludedRouter` wrapping.

    `include_router` in this version does not append flat routes to
    `app.routes` -- each call adds one `_IncludedRouter` node whose real
    routes live on `.original_router.routes` (see `app.py`'s own comment on
    why `_sweep_v1_routes_require_auth` walks `app.openapi()` instead of
    `app.routes` for auth). `app.openapi()` cannot serve this test's
    purpose, though: it only ever lists schema-*visible* routes, so a
    hidden route -- the exact thing this guard exists to catch -- would
    never appear there to be flagged. Recursing through
    `original_router.routes` sees every route regardless of whether it
    opted out of the schema.
    """
    for route in routes:
        yield route
        router = getattr(route, "original_router", None)
        if router is not None:
            yield from _walk_every_registered_route(router.routes)


# The one route allowed to be invisible to the sweep, named by its endpoint
# function rather than by its path so a change of prefix cannot silently widen
# the exemption.
#
# `serve_studio_ui` (assistant/io/api/ui.py) is the catch-all that serves the
# Studio front-end. It is out of schema for two reasons that cut the same way:
# a `/{ui_path:path}` catch-all belongs in nobody's generated TypeScript
# client, and it is *deliberately* unauthenticated -- the page it serves is
# what bootstraps pairing, so it cannot demand a credential -- which means the
# sweep would flag it as a leak on every run. Its compensating coverage is
# tests/test_api_ui_serving.py, where every property this sweep would
# otherwise have checked is pinned instead: traversal refusal, that /v1 is not
# shadowed, and that the host gate still covers it. Anything added here needs
# the same: an explicit test, named in this comment, not silent trust.
_SCHEMA_EXEMPT_ENDPOINTS = frozenset({"serve_studio_ui"})


def _is_schema_exempt(route) -> bool:
    endpoint = getattr(route, "endpoint", None)
    return getattr(endpoint, "__name__", "") in _SCHEMA_EXEMPT_ENDPOINTS


def _assert_nothing_hides_from_the_sweep(app) -> int:
    """Returns how many exempt routes it saw, so a caller can pin that too."""
    from starlette.routing import Mount

    exempt = 0
    for route in _walk_every_registered_route(app.routes):
        assert not isinstance(route, Mount), (
            f"a Mount route ({getattr(route, 'path', '?')!r}) is invisible "
            "to the openapi-based sweep"
        )
        if _is_schema_exempt(route):
            exempt += 1
            continue
        if hasattr(route, "include_in_schema"):
            assert route.include_in_schema, (
                f"{route.path!r} hides from the schema the auth sweep walks"
            )
    return exempt


def test_no_route_hides_from_the_schema_sweep(client):
    """`_sweep_v1_routes_require_auth` walks `app.openapi()`'s resolved
    paths -- which is exactly what a route registered with
    `include_in_schema=False` or mounted via `Mount` never appears in. The
    sweep cannot catch what it cannot see, so this is a cheap standing
    guard: as long as nothing in this app opts out of the schema except by
    the named exemption above, the sweep's blind spot stays accounted for.
    If this ever fails, a route was added that the auth sweep cannot check
    at all, and it needs its own explicit auth test, not silent trust.
    """
    assert _assert_nothing_hides_from_the_sweep(client.app) == 0


def test_the_exemption_holds_when_a_ui_bundle_is_actually_mounted(vault, tmp_path):
    """The guard above is satisfied on the default client for the wrong
    reason: no fixture mounts a UI bundle, so the exempt route does not exist
    yet. Once the packaging step wires a real bundle in, it will -- and an
    invariant that breaks in production while the suite stays green is exactly
    what this guard exists to prevent. So mount one here and assert the
    exemption is real, singular, and still the only thing out of schema."""
    from tests.fakes.studio_ui import build_ui_bundle

    app = build_api_client(build_fake_runtime(), vault,
                           ui_bundle=build_ui_bundle(tmp_path)).app
    assert _assert_nothing_hides_from_the_sweep(app) == 1


def test_the_exemption_does_not_excuse_anything_else(vault, tmp_path):
    """An exemption list is only worth having if it still catches the next
    route. A second hidden route, mounted alongside the UI one, must fail."""
    from tests.fakes.studio_ui import build_ui_bundle

    app = build_api_client(build_fake_runtime(), vault,
                           ui_bundle=build_ui_bundle(tmp_path)).app
    hidden = APIRouter()

    @hidden.get("/hidden", include_in_schema=False)
    async def hidden_handler():
        return {"ok": True}

    app.include_router(hidden, prefix="/v1")

    with pytest.raises(AssertionError, match="hides from the schema"):
        _assert_nothing_hides_from_the_sweep(app)


# ─── fix wave: the wire is camelCase, pinned against the schema itself ────
def test_no_schema_property_is_snake_case(client):
    """Studio generates its TypeScript types from app.openapi(). Before the
    fix wave, chat.py/commands.py/status.py/telemetry/_backup_body/audit's
    hand-built response dicts and settings.py's own restart_required were
    snake_case while memory.py/files.py were already camelCase -- and every
    Pydantic-modelled field (Meta, RenameRequest, RestoreRequest) still is,
    since `data: Any` never round-trips those hand-built dicts through a
    schema Pydantic would otherwise normalise. Walking `components.schemas`
    catches every *modelled* property; it cannot see inside `data: Any`,
    which is exactly why the individual route tests (test_api_chat.py,
    test_api_commands.py, test_api_system.py, ...) pin the hand-built dicts
    directly -- this test and those together are the actual coverage.
    """
    schema = client.app.openapi()

    def _walk(node) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for name in props:
                    assert "_" not in name, f"snake_case wire property: {name!r}"
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    schemas = schema.get("components", {}).get("schemas", {})
    assert schemas, "sanity: no component schemas found to check"
    _walk(schemas)


def test_the_hiding_guard_catches_a_route_that_opts_out_of_the_schema(vault):
    """Regression guard for the guard itself: without the recursive walk
    through `original_router`, a hidden route registered the ordinary way
    is invisible to a naive `app.routes` scan in this FastAPI version --
    proven by using that naive scan here and showing it finds nothing.
    """
    app = build_api_client(build_fake_runtime(), vault).app
    hidden = APIRouter()

    @hidden.get("/hidden", include_in_schema=False)
    async def hidden_handler():
        return {"ok": True}

    app.include_router(hidden, prefix="/v1")

    found_via_walk = any(
        getattr(r, "include_in_schema", True) is False
        for r in _walk_every_registered_route(app.routes)
    )
    found_via_naive_scan = any(
        getattr(r, "include_in_schema", True) is False for r in app.routes
    )
    assert found_via_walk, "the recursive walk failed to see the hidden route"
    assert not found_via_naive_scan, (
        "app.routes alone already saw the hidden route -- this FastAPI "
        "version's _IncludedRouter wrapping must have changed; the walk "
        "helper above may no longer be needed"
    )
