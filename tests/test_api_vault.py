"""Token vault — the only thing that decides whether a caller is real."""
import json
import pytest

from assistant.io.api.vault import Capability, Device, TokenVault


@pytest.fixture()
def vault(tmp_path):
    return TokenVault(tmp_path)


def test_instance_secret_is_256_bits_and_stable(vault):
    first = vault.instance_secret()
    assert len(first) == 32
    assert vault.instance_secret() == first


def test_two_installations_get_different_secrets(tmp_path):
    a = TokenVault(tmp_path / "a").instance_secret()
    b = TokenVault(tmp_path / "b").instance_secret()
    assert a != b


def test_env_var_overrides_the_stored_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("TENKA_SECRET", "f" * 64)
    assert TokenVault(tmp_path).instance_secret() == bytes.fromhex("f" * 64)


def test_issued_token_verifies_and_carries_its_grants(vault):
    token = vault.issue("studio", frozenset({Capability.CHAT, Capability.FILES}))
    device = vault.verify(token)
    assert isinstance(device, Device)
    assert device.label == "studio"
    assert device.grants == frozenset({Capability.CHAT, Capability.FILES})


def test_token_is_at_least_256_bits_of_entropy(vault):
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    assert len(token) >= 43  # 32 bytes, url-safe base64, unpadded


def test_two_issues_never_collide(vault):
    a = vault.issue("one", frozenset({Capability.CHAT}))
    b = vault.issue("two", frozenset({Capability.CHAT}))
    assert a != b


def test_plaintext_token_is_never_written_to_disk(vault, tmp_path):
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert token not in path.read_text(encoding="utf-8", errors="ignore")


def test_unknown_token_is_rejected(vault):
    vault.issue("studio", frozenset({Capability.CHAT}))
    assert vault.verify("not-a-real-token") is None


def test_garbage_token_is_rejected_without_raising(vault):
    for junk in ("", "   ", "\x00", "a" * 5000, "===="):
        assert vault.verify(junk) is None


def test_revoked_token_stops_verifying(vault):
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    device = vault.verify(token)
    assert vault.revoke(device.device_id) is True
    assert vault.verify(token) is None


def test_revoking_an_unknown_device_reports_false(vault):
    assert vault.revoke("nope") is False


def test_rotating_the_instance_secret_revokes_everything(vault):
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    vault.reset()
    assert vault.verify(token) is None


def test_devices_lists_what_was_issued(vault):
    vault.issue("studio", frozenset({Capability.CHAT}))
    vault.issue("phone", frozenset({Capability.CHAT, Capability.SCREEN}))
    labels = sorted(d.label for d in vault.devices())
    assert labels == ["phone", "studio"]


def test_device_record_persists_across_instances(tmp_path):
    token = TokenVault(tmp_path).issue("studio", frozenset({Capability.CHAT}))
    assert TokenVault(tmp_path).verify(token) is not None


def test_stored_record_holds_a_hash_not_the_token(vault, tmp_path):
    vault.issue("studio", frozenset({Capability.CHAT}))
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    assert raw["devices"], "no device recorded"
    entry = raw["devices"][0]
    assert "token_hmac" in entry
    assert len(entry["token_hmac"]) == 64  # sha256 hex
    assert "token" not in entry
