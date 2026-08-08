"""Nothing is reachable without a token. Not even status."""
import pytest
from fastapi.testclient import TestClient

from assistant.io.api.app import create_app
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


def test_every_registered_route_rejects_an_anonymous_call(client):
    from starlette.routing import Route

    checked = 0
    for route in client.app.routes:
        if not isinstance(route, Route) or not route.path.startswith("/v1"):
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            response = client.request(method, route.path.replace("{scope}", "knowledge")
                                              .replace("{conversation_id}", "c1")
                                              .replace("{item_id}", "k1")
                                              .replace("{command_id}", "volume_up")
                                              .replace("{kind}", "voice"))
            assert response.status_code in (401, 403), (
                f"{method} {route.path} answered {response.status_code} with no token"
            )
            checked += 1
    assert checked > 0, "no /v1 routes were checked — the sweep found nothing"


def test_a_capability_it_lacks_is_refused(client, vault):
    chat_only = vault.issue("phone", frozenset({Capability.CHAT}))
    response = client.get("/v1/status", headers=auth(chat_only))
    assert response.status_code == 200


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
