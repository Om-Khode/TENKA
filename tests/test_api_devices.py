"""Listing paired devices, and revoking one.

Both are loopback-only, for the same reason minting a pair code is: a
compromised remote session must not be able to enumerate the other devices or
to revoke them. Revocation is immediate -- it is the one control that has to
work under duress, so it must not depend on anything expiring.
"""
from __future__ import annotations

from assistant.io.api.security import COOKIE_NAME, CSRF_HEADER
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import LOCAL_PORT, ApiTestClient, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime


def _client(vault: TokenVault, *, policies: dict[int, str]) -> ApiTestClient:
    return build_api_client(build_fake_runtime(), vault, policies=policies)


# ─── the gate ────────────────────────────────────────────────────────────
def test_devices_list_and_revoke_are_loopback_only(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    remote = _client(vault, policies={LOCAL_PORT: "funnel"})
    remote.cookies.set(COOKIE_NAME, token)
    assert remote.get("/v1/devices").status_code == 403
    device_id = vault.devices()[0].device_id
    assert remote.delete(f"/v1/devices/{device_id}",
                         headers={CSRF_HEADER: "1"}).status_code == 403


def test_listing_devices_requires_authentication(tmp_path):
    client = _client(TokenVault(tmp_path), policies={LOCAL_PORT: "local"})
    assert client.get("/v1/devices").status_code == 401


def test_a_watching_device_cannot_read_or_revoke_the_device_list(tmp_path):
    """The device list is the security configuration -- who holds a credential
    to this machine -- so it sits behind SYSTEM_CONTROL like /v1/audit, not
    behind the grant that lets a wall display watch her work."""
    vault = TokenVault(tmp_path)
    watcher = vault.issue("wall display", frozenset({Capability.OBSERVE}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, watcher)
    assert client.get("/v1/devices").status_code == 403
    device_id = vault.devices()[0].device_id
    assert client.delete(f"/v1/devices/{device_id}",
                         headers={CSRF_HEADER: "1"}).status_code == 403


# ─── the listing ─────────────────────────────────────────────────────────
def test_the_listing_carries_only_what_the_revoke_list_needs(tmp_path):
    """Deliberately narrow. This response describes every credential to this
    machine, so it carries the four facts a person needs to decide what to
    kill -- which row, what it is called, what it can do, when it was last
    used -- and nothing else."""
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    data = client.get("/v1/devices").json()["data"]
    assert set(data) == {"devices"}
    assert set(data["devices"][0]) == {
        "deviceId", "label", "grants", "createdAt", "lastSeenAt",
    }


def test_the_listing_never_carries_a_token_or_its_hash(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    body = client.get("/v1/devices").text
    assert token not in body
    assert "hmac" not in body.lower()


def test_every_paired_device_is_listed(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    devices = client.get("/v1/devices").json()["data"]["devices"]
    assert {d["label"] for d in devices} == {"laptop", "phone"}
    phone = [d for d in devices if d["label"] == "phone"][0]
    assert phone["grants"] == ["observe"]


# ─── revocation ──────────────────────────────────────────────────────────
def test_revoke_kills_the_credential_immediately(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    device_id = vault.devices()[0].device_id
    assert client.delete(f"/v1/devices/{device_id}",
                         headers={CSRF_HEADER: "1"}).status_code == 200
    assert client.get("/v1/status").status_code == 401


def test_revoking_an_unknown_device_is_404(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    assert client.delete("/v1/devices/nope",
                         headers={CSRF_HEADER: "1"}).status_code == 404


def test_revoking_leaves_the_other_devices_alone(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("laptop", frozenset(Capability))
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    phone_id = [d for d in vault.devices() if d.label == "phone"][0].device_id
    response = client.delete(f"/v1/devices/{phone_id}", headers={CSRF_HEADER: "1"})
    assert response.json()["data"] == {"revoked": phone_id}
    assert [d.label for d in vault.devices()] == ["laptop"]


def test_a_cookie_revoke_still_needs_the_csrf_header(tmp_path):
    """Revocation is a write reached with ambient authority. A page the user
    happens to visit must not be able to unpair her devices."""
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    device_id = vault.devices()[0].device_id
    assert client.delete(f"/v1/devices/{device_id}").status_code == 403
    assert len(vault.devices()) == 1


# ─── last seen ───────────────────────────────────────────────────────────
def test_a_successful_call_updates_last_seen(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, token)
    client.get("/v1/status")
    assert vault.devices()[0].last_seen_at is not None


def test_a_refused_call_does_not_update_last_seen(tmp_path):
    """`last_seen_at` is a fact about the device, not about the socket. A
    wrong token has no device to attribute anything to, and a request refused
    for lacking a capability was still that device -- so the split is
    authenticated-or-not, not allowed-or-not."""
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset(Capability))
    client = _client(vault, policies={LOCAL_PORT: "local"})
    client.cookies.set(COOKIE_NAME, "not-a-real-token")
    client.get("/v1/status")
    assert vault.devices()[0].last_seen_at is None
