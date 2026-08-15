"""Nothing is reachable without a token. Not even status."""
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
from assistant.io.api.security import RateLimiter, require
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime

ALL = frozenset(Capability)


@pytest.fixture()
def vault(tmp_path):
    return TokenVault(tmp_path)


@pytest.fixture()
def token(vault):
    return vault.issue("studio", ALL)


def _client(vault: TokenVault) -> TestClient:
    """Build a client against a caller-supplied vault.

    The `client` fixture below covers the common case of one vault per test;
    the CHAT/CHAT_SEND split tests need to issue several tokens off vaults
    they construct themselves, so the app-building step is pulled out here
    rather than duplicated per test.
    """
    app = create_app(build_fake_runtime(), vault, origins=["http://localhost:3000"])
    return TestClient(app)


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
                        .replace("{kind}", "voice"))
        for method in operations:
            if method.upper() in ("HEAD", "OPTIONS"):
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


def test_the_sweep_catches_a_route_registered_without_auth(vault):
    """Guard for the sweep itself, not for the app.

    A route added the ordinary `app.include_router()` way -- the exact call
    every later route module's brief instructs -- with no auth dependency at
    all, must still be caught. If this test ever fails, the sweep has
    regressed to something that can be dodged by registration style, which
    is exactly the gap `_sweep_v1_routes_require_auth` exists to close.
    """
    app = create_app(build_fake_runtime(), vault, origins=["http://localhost:3000"])

    leaky = APIRouter()

    @leaky.get("/leaky")
    async def leaky_handler():
        return {"ok": True}

    app.include_router(leaky, prefix="/v1")
    leaky_client = TestClient(app)

    with pytest.raises(AssertionError, match="answered 200"):
        _sweep_v1_routes_require_auth(leaky_client)


def test_a_capability_it_lacks_is_refused(client, vault):
    """A token that lacks the capability a route requires must be refused.

    `/v1/status` requires CHAT and is the only shipped route, so an
    all-grants or CHAT-holding token can never exercise `require()`'s 403
    branch -- the original version of this test issued a CHAT token against
    the CHAT-gated route and asserted 200, which would pass identically if
    the capability check were deleted outright. A second, FILES-gated route
    is mounted here to force the refusal, with a mirror case proving the
    same route accepts a token that does hold the capability it demands.
    """
    probe = APIRouter()

    @probe.get("/probe")
    async def probe_handler(_=Depends(require(Capability.FILES))):
        return {"ok": True}

    client.app.include_router(probe, prefix="/v1")

    chat_only = vault.issue("phone", frozenset({Capability.CHAT}))
    refused = client.get("/v1/probe", headers=auth(chat_only))
    assert refused.status_code == 403

    files_holder = vault.issue("laptop", frozenset({Capability.FILES}))
    allowed = client.get("/v1/probe", headers=auth(files_holder))
    assert allowed.status_code == 200


# ─── CHAT vs CHAT_SEND: reading her conversations is not driving her ──────
def test_a_read_only_chat_device_cannot_send(tmp_path):
    """The whole point of the split: CHAT reaches all 38 intents through
    POST /v1/chat, so reading history must not imply driving her."""
    vault = TokenVault(tmp_path)
    token = vault.issue("reader", frozenset({Capability.CHAT}))
    client = _client(vault)
    assert client.get("/v1/chat/conversations",
                      headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.post("/v1/chat", json={"text": "hi"},
                       headers={"Authorization": f"Bearer {token}"}).status_code == 403
    assert client.post("/v1/abort",
                       headers={"Authorization": f"Bearer {token}"}).status_code == 403


def test_a_sending_device_still_reads(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.CHAT, Capability.CHAT_SEND}))
    client = _client(vault)
    assert client.post("/v1/chat", json={"text": "hi"},
                       headers={"Authorization": f"Bearer {token}"}).status_code == 202


def test_the_events_socket_stays_a_read_gate(tmp_path):
    """app.py checks CHAT before accept(). The socket only streams, so it must
    NOT start demanding CHAT_SEND -- a reader device keeps its live view."""
    vault = TokenVault(tmp_path)
    token = vault.issue("reader", frozenset({Capability.CHAT}))
    with _client(vault).websocket_connect(f"/v1/events?access_token={token}") as ws:
        assert ws is not None


def test_openapi_is_not_public(client):
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code in (401, 404)


def test_the_query_string_exception_is_only_the_socket():
    """One route may take a token from the query string. Exactly one."""
    import pathlib
    source = pathlib.Path("assistant/io/api/app.py").read_text(encoding="utf-8")
    assert source.count("query_params.get(\"access_token\"") == 1


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


def test_no_route_hides_from_the_schema_sweep(client):
    """`_sweep_v1_routes_require_auth` walks `app.openapi()`'s resolved
    paths -- which is exactly what a route registered with
    `include_in_schema=False` or mounted via `Mount` never appears in. The
    sweep cannot catch what it cannot see, so this is a cheap standing
    guard: as long as nothing in this app opts out of the schema or mounts
    a sub-application, the sweep's blind spot stays theoretical. If this
    ever fails, a route was added that the auth sweep cannot check at all,
    and it needs its own explicit auth test, not silent trust.
    """
    from starlette.routing import Mount

    for route in _walk_every_registered_route(client.app.routes):
        assert not isinstance(route, Mount), (
            f"a Mount route ({getattr(route, 'path', '?')!r}) is invisible "
            "to the openapi-based sweep"
        )
        if hasattr(route, "include_in_schema"):
            assert route.include_in_schema, (
                f"{route.path!r} hides from the schema the auth sweep walks"
            )


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
    app = create_app(build_fake_runtime(), vault, origins=["http://localhost:3000"])
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
