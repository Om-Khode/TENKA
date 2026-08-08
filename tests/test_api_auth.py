"""Nothing is reachable without a token. Not even status."""
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
from assistant.io.api.security import require
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.studio_runtime import build_fake_runtime

ALL = frozenset(Capability)


@pytest.fixture()
def vault(tmp_path):
    return TokenVault(tmp_path)


@pytest.fixture()
def token(vault):
    return vault.issue("studio", ALL)


@pytest.fixture()
def client(vault):
    app = create_app(build_fake_runtime(), vault, origins=["http://localhost:3000"])
    return TestClient(app)


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_status_requires_a_token(client):
    assert client.get("/v1/status").status_code == 401


def test_status_with_a_token_answers(client, token):
    response = client.get("/v1/status", headers=auth(token))
    assert response.status_code == 200
    assert response.json()["data"]["assistant_name"] == "TENKA"


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


def test_openapi_is_not_public(client):
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code in (401, 404)


def test_a_bad_tenka_secret_fails_at_app_build_not_per_request(tmp_path, monkeypatch):
    """instance_secret() is uncached on the env-override path, so a wrong-length
    TENKA_SECRET would otherwise surface as a 500 on the first authenticated
    request instead of at startup. create_app() must resolve it eagerly."""
    monkeypatch.setenv("TENKA_SECRET", "ab" * 10)  # valid hex, only 10 bytes
    with pytest.raises(ValueError, match="TENKA_SECRET"):
        create_app(build_fake_runtime(), TokenVault(tmp_path),
                   origins=["http://localhost:3000"])
