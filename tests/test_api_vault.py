"""Token vault — the only thing that decides whether a caller is real."""
import json
import logging
from hashlib import sha256

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


def test_issuing_a_device_with_no_grants_is_refused(vault):
    """A device with no capabilities can still authenticate -- just not do
    anything, except distinguish 404 from 403 on a route gated by
    `authenticate` alone rather than `require(capability)`, which leaks
    membership it was never granted CHAT to read. Refusing at issuance
    closes that for every such route, not just the ones known today.
    """
    with pytest.raises(ValueError):
        vault.issue("ghost", frozenset())


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


# ─── Corruption: the vault must fail closed, never raise, never fail open ──


def test_corrupt_secret_file_regenerates_and_revokes_everything(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    (tmp_path / "instance_secret").write_text("not-hex-garbage", encoding="utf-8")

    fresh = TokenVault(tmp_path)  # no in-memory cache -- forces a file read
    secret = fresh.instance_secret()
    assert len(secret) == 32
    assert fresh.verify(token) is None


def test_devices_json_as_bare_array_is_rejected_wholesale(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    device_id = vault.verify(token).device_id
    (tmp_path / "devices.json").write_text("[]", encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []
    assert vault.revoke(device_id) is False


def test_devices_field_as_string_is_rejected_wholesale(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    device_id = vault.verify(token).device_id
    (tmp_path / "devices.json").write_text(
        json.dumps({"version": 1, "devices": "oops"}), encoding="utf-8"
    )

    assert vault.verify(token) is None
    assert vault.devices() == []
    assert vault.revoke(device_id) is False


def test_entry_that_is_not_a_dict_is_skipped(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    device_id = vault.verify(token).device_id
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    raw["devices"][0] = "not-a-dict"
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []
    assert vault.revoke(device_id) is False


def test_entry_with_non_string_token_hmac_fails_closed_but_stays_administrable(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    device_id = vault.verify(token).device_id
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    raw["devices"][0]["token_hmac"] = 12345
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    # A non-string hash can never compare equal to a real hex digest, so the
    # token that used to work now correctly fails closed.
    assert vault.verify(token) is None
    # device_id/label/grants/created_at are untouched, so admin listing and
    # revocation -- which never look at token_hmac -- are unaffected.
    devices = vault.devices()
    assert len(devices) == 1
    assert vault.revoke(device_id) is True


def test_entry_with_unknown_capability_fails_closed_but_stays_revocable(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    device_id = vault.verify(token).device_id
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    raw["devices"][0]["grants"] = ["not-a-real-capability"]
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []
    # device_id is untouched, so an operator can still revoke a device whose
    # grants got hand-edited into garbage.
    assert vault.revoke(device_id) is True


def test_entry_with_empty_grants_fails_closed_but_a_normal_entry_still_verifies(tmp_path):
    """`issue()` refuses an empty grant set, but a hand-edited devices.json
    can still produce one: a valid token_hmac paired with `"grants": []`.
    That combination used to parse into `Device(grants=frozenset())`, which
    `authenticate()` passes straight through -- reproducing, on any route
    gated by `authenticate` alone, the 404-vs-403 oracle Finding 2 was meant
    to close. A second, untouched device in the same file proves the check
    is discriminating: it rejects the empty entry without taking down every
    entry in the store.
    """
    vault = TokenVault(tmp_path)
    ghost_token = vault.issue("ghost", frozenset({Capability.CHAT}))
    ghost_id = vault.verify(ghost_token).device_id
    normal_token = vault.issue("normal", frozenset({Capability.CHAT}))

    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    for entry in raw["devices"]:
        if entry["device_id"] == ghost_id:
            entry["grants"] = []
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    assert vault.verify(ghost_token) is None
    assert vault.verify(normal_token) is not None

    labels = sorted(d.label for d in vault.devices())
    assert labels == ["normal"]  # the ghost entry drops out of listing too
    assert vault.revoke(ghost_id) is True  # still administrable by raw id


def test_entry_missing_device_id_fails_closed_everywhere(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    device_id = vault.verify(token).device_id
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    del raw["devices"][0]["device_id"]
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []
    assert vault.revoke(device_id) is False


# ─── A stored/overridden secret that decodes to the wrong size ────────────
# bytes.fromhex("") == b"" with no exception, so an empty or whitespace-only
# secret file reads back as a "valid" zero-length key unless length is
# checked explicitly. Same shape for hex of the wrong length (e.g. a 4-byte
# secret instead of 32) -- decodes cleanly, just isn't a 256-bit secret.


def test_empty_secret_file_regenerates_and_revokes_everything(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    (tmp_path / "instance_secret").write_text("", encoding="utf-8")

    fresh = TokenVault(tmp_path)  # no in-memory cache -- forces a file read
    secret = fresh.instance_secret()
    assert len(secret) == 32
    assert fresh.verify(token) is None


def test_whitespace_only_secret_file_regenerates_and_revokes_everything(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    (tmp_path / "instance_secret").write_text("   \n\t  ", encoding="utf-8")

    fresh = TokenVault(tmp_path)
    secret = fresh.instance_secret()
    assert len(secret) == 32
    assert fresh.verify(token) is None


def test_short_but_valid_hex_secret_file_regenerates_and_revokes_everything(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.CHAT}))
    (tmp_path / "instance_secret").write_text("ab" * 4, encoding="utf-8")  # 4 bytes, not 32

    fresh = TokenVault(tmp_path)
    secret = fresh.instance_secret()
    assert len(secret) == 32
    assert fresh.verify(token) is None


def test_env_var_empty_string_is_treated_as_unset(tmp_path, monkeypatch):
    # A caller-supplied empty override must not silently become a weak key.
    # Decision: an empty TENKA_SECRET is treated as if the variable were
    # unset at all, so the secret still comes from -- and is persisted to --
    # the on-disk file, never derived from the empty string itself.
    monkeypatch.setenv("TENKA_SECRET", "")
    vault = TokenVault(tmp_path)

    secret = vault.instance_secret()

    assert len(secret) == 32
    assert secret != b""
    assert secret != sha256(b"").digest()
    assert (tmp_path / "instance_secret").exists()


def test_env_var_with_wrong_length_hex_raises_loudly(tmp_path, monkeypatch):
    # Decision: unlike a corrupt on-disk file, a bad TENKA_SECRET is not
    # something the vault can recover from by regenerating -- there is
    # nothing to regenerate, the operator asked for this exact secret and
    # got the length wrong. Silently accepting a weak key or silently
    # substituting a different secret would both hide the mistake, so this
    # raises instead of returning anything.
    monkeypatch.setenv("TENKA_SECRET", "ab" * 4)  # valid hex, 4 bytes, not 32
    vault = TokenVault(tmp_path)

    with pytest.raises(ValueError):
        vault.instance_secret()


def test_env_var_that_is_not_hex_is_hashed_into_a_key(tmp_path, monkeypatch):
    # A passphrase is not a 256-bit key, but it is unambiguously deliberate, so
    # it is stretched into one rather than rejected. Nothing is persisted: the
    # override lives in the environment, and writing it to disk would outlive
    # the variable that set it.
    monkeypatch.setenv("TENKA_SECRET", "  correct horse battery staple  ")
    vault = TokenVault(tmp_path)

    assert vault.instance_secret() == sha256(b"correct horse battery staple").digest()
    assert not (tmp_path / "instance_secret").exists()


def test_env_var_wins_over_both_the_stored_and_the_cached_secret(tmp_path, monkeypatch):
    # The file read added below must not become a second source of truth that
    # outranks an explicit operator override -- precedence stays env, file, new.
    vault = TokenVault(tmp_path)
    stored = vault.instance_secret()  # populates the cache and the file

    monkeypatch.setenv("TENKA_SECRET", "ab" * 32)

    assert vault.instance_secret() == bytes.fromhex("ab" * 32)
    assert vault.instance_secret() != stored


# ─── Disk is the truth; the cached secret is only a fallback ──────────────
# `instance_secret()` re-reads the secret file on every call. The cost is one
# ~64-byte read per HMAC, which is cheaper than the devices.json read and JSON
# parse `verify()` already does on every call via `_load()`. What it buys: a
# rotation performed by any other vault instance -- or any other process -- is
# seen immediately, instead of a stale cache minting tokens nobody can verify.


def test_issue_after_another_instance_rotated_the_secret_still_verifies(tmp_path):
    """The blocker: a long-running vault must not mint tokens against a secret
    that disk has already moved past.

    A daemon vault caches the secret at startup; a rotation then happens
    through a *different* vault instance on the same root (a slash command, an
    admin route). Hashing the next issued token against the cached, superseded
    secret produces a token that verifies for the rest of that process's life
    and then never again -- the device works today and is silently dead after
    the next restart, with no error anywhere.
    """
    daemon = TokenVault(tmp_path)
    daemon.instance_secret()  # long-running instance caches the pre-rotation secret

    TokenVault(tmp_path).reset()  # rotated out from under it

    token = daemon.issue("phone", frozenset({Capability.CHAT}))

    assert daemon.verify(token) is not None, "unusable the moment it was issued"
    fresh = TokenVault(tmp_path)  # what the next daemon start sees: disk only
    device = fresh.verify(token)
    assert device is not None, "verified in-process, dead after a restart"
    assert device.label == "phone"


def test_rotation_is_visible_even_when_the_device_list_survives(tmp_path):
    """`reset()` deletes devices.json, so revocation used to appear to work
    cross-process even with a stale cached secret -- an empty device list
    matches nothing regardless of which secret hashed the token. Rotate the
    secret alone and that cover is gone: revocation now holds because the
    secret itself is re-read, not because there is nothing left to compare
    against.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.CHAT}))
    assert vault.verify(token) is not None

    (tmp_path / "instance_secret").write_text("ab" * 32, encoding="utf-8")

    assert vault.verify(token) is None


def test_deleted_secret_file_mid_run_does_not_revoke_issued_devices(tmp_path, caplog):
    """Disk losing the secret is not authority to revoke every device.

    Regenerating here would be a robustness regression: today's cache
    accidentally prevents it, and a vanished file is far more likely to be a
    backup tool, a sync client, or a stray delete than an instruction to
    invalidate every paired device mid-run. Keep the secret already in memory,
    say so loudly, and leave the decision to the operator.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.CHAT}))
    (tmp_path / "instance_secret").unlink()

    with caplog.at_level(logging.WARNING, logger="assistant.io.api.vault"):
        assert vault.verify(token) is not None

    assert "keeping the secret already in memory" in caplog.text
    # No silent re-persist either: rewriting the file would hide the loss and
    # bake whichever process noticed first in as the winner.
    assert not (tmp_path / "instance_secret").exists()


def test_corrupt_secret_file_mid_run_does_not_revoke_issued_devices(tmp_path, caplog):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.CHAT}))
    (tmp_path / "instance_secret").write_text("not-hex-garbage", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="assistant.io.api.vault"):
        assert vault.verify(token) is not None

    assert "keeping the secret already in memory" in caplog.text
    # The bad file is left exactly as found, for the operator to look at.
    assert (tmp_path / "instance_secret").read_text(encoding="utf-8") == "not-hex-garbage"


def test_corrupt_secret_file_with_no_cached_secret_regenerates_and_persists(tmp_path):
    """The documented recovery path, unchanged: with nothing cached there is
    no secret worth preserving, and raising would take the whole daemon down
    at startup with no way back. Regenerate, persist, and revoke.
    """
    (tmp_path / "instance_secret").write_text("not-hex-garbage", encoding="utf-8")
    vault = TokenVault(tmp_path)

    secret = vault.instance_secret()

    assert len(secret) == 32
    on_disk = (tmp_path / "instance_secret").read_text(encoding="utf-8").strip()
    assert bytes.fromhex(on_disk) == secret, "regenerated secret was not persisted"
    assert vault.instance_secret() == secret  # and it is stable from then on
