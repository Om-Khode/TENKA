"""Token vault — the only thing that decides whether a caller is real."""
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from assistant.io.api.vault import Capability, Device, TokenVault, VaultReadError


@pytest.fixture()
def vault(tmp_path):
    return TokenVault(tmp_path)


def test_instance_secret_is_256_bits_and_stable(vault):
    first = vault.instance_secret()
    assert len(first) == 32
    assert vault.instance_secret() == first


def test_chat_send_is_a_distinct_capability():
    assert Capability.CHAT_SEND.value == "chat_send"
    assert Capability.CHAT_SEND is not Capability.OBSERVE


def test_two_installations_get_different_secrets(tmp_path):
    a = TokenVault(tmp_path / "a").instance_secret()
    b = TokenVault(tmp_path / "b").instance_secret()
    assert a != b


def test_env_var_overrides_the_stored_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("TENKA_SECRET", "f" * 64)
    assert TokenVault(tmp_path).instance_secret() == bytes.fromhex("f" * 64)


def test_issued_token_verifies_and_carries_its_grants(vault):
    token = vault.issue("studio", frozenset({Capability.OBSERVE, Capability.FILES}))
    device = vault.verify(token)
    assert isinstance(device, Device)
    assert device.label == "studio"
    assert device.grants == frozenset({Capability.OBSERVE, Capability.FILES})


def test_issuing_a_device_with_no_grants_is_refused(vault):
    """A device with no capabilities can still authenticate -- just not do
    anything, except distinguish 404 from 403 on a route gated by
    `authenticate` alone rather than `require(capability)`, which leaks
    membership it was never granted OBSERVE to read. Refusing at issuance
    closes that for every such route, not just the ones known today.
    """
    with pytest.raises(ValueError):
        vault.issue("ghost", frozenset())


def test_token_is_at_least_256_bits_of_entropy(vault):
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    assert len(token) >= 43  # 32 bytes, url-safe base64, unpadded


def test_two_issues_never_collide(vault):
    a = vault.issue("one", frozenset({Capability.OBSERVE}))
    b = vault.issue("two", frozenset({Capability.OBSERVE}))
    assert a != b


def test_plaintext_token_is_never_written_to_disk(vault, tmp_path):
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert token not in path.read_text(encoding="utf-8", errors="ignore")


def test_unknown_token_is_rejected(vault):
    vault.issue("studio", frozenset({Capability.OBSERVE}))
    assert vault.verify("not-a-real-token") is None


def test_garbage_token_is_rejected_without_raising(vault):
    for junk in ("", "   ", "\x00", "a" * 5000, "===="):
        assert vault.verify(junk) is None


def test_revoked_token_stops_verifying(vault):
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    device = vault.verify(token)
    assert vault.revoke(device.device_id) is True
    assert vault.verify(token) is None


def test_revoking_an_unknown_device_reports_false(vault):
    assert vault.revoke("nope") is False


def test_rotating_the_instance_secret_revokes_everything(vault):
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    vault.reset()
    assert vault.verify(token) is None


def test_devices_lists_what_was_issued(vault):
    vault.issue("studio", frozenset({Capability.OBSERVE}))
    vault.issue("phone", frozenset({Capability.OBSERVE, Capability.SCREEN}))
    labels = sorted(d.label for d in vault.devices())
    assert labels == ["phone", "studio"]


def test_device_record_persists_across_instances(tmp_path):
    token = TokenVault(tmp_path).issue("studio", frozenset({Capability.OBSERVE}))
    assert TokenVault(tmp_path).verify(token) is not None


def test_stored_record_holds_a_hash_not_the_token(vault, tmp_path):
    vault.issue("studio", frozenset({Capability.OBSERVE}))
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    assert raw["devices"], "no device recorded"
    entry = raw["devices"][0]
    assert "token_hmac" in entry
    assert len(entry["token_hmac"]) == 64  # sha256 hex
    assert "token" not in entry


# ─── Corruption: the vault must fail closed, never raise, never fail open ──


def test_corrupt_secret_file_regenerates_and_revokes_everything(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    (tmp_path / "instance_secret").write_text("not-hex-garbage", encoding="utf-8")

    fresh = TokenVault(tmp_path)  # no in-memory cache -- forces a file read
    secret = fresh.instance_secret()
    assert len(secret) == 32
    assert fresh.verify(token) is None


def test_devices_json_as_bare_array_is_rejected_wholesale(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    device_id = vault.verify(token).device_id
    (tmp_path / "devices.json").write_text("[]", encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []
    assert vault.revoke(device_id) is False


def test_devices_field_as_string_is_rejected_wholesale(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    device_id = vault.verify(token).device_id
    (tmp_path / "devices.json").write_text(
        json.dumps({"version": 1, "devices": "oops"}), encoding="utf-8"
    )

    assert vault.verify(token) is None
    assert vault.devices() == []
    assert vault.revoke(device_id) is False


def test_entry_that_is_not_a_dict_is_skipped(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    device_id = vault.verify(token).device_id
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    raw["devices"][0] = "not-a-dict"
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []
    assert vault.revoke(device_id) is False


def test_entry_with_non_string_token_hmac_fails_closed_but_stays_administrable(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
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
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    device_id = vault.verify(token).device_id
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    raw["devices"][0]["grants"] = ["not-a-real-capability"]
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []
    # device_id is untouched, so an operator can still revoke a device whose
    # grants got hand-edited into garbage.
    assert vault.revoke(device_id) is True


def test_a_device_paired_before_the_observe_recall_split_fails_closed(tmp_path):
    """`"chat"` is not a capability any more, and there is deliberately no
    migration for it.

    A record written before the split says `"chat"`, which meant both "watch
    her work" and "read everything she stored". Upgrading it to `RECALL` would
    hand a device paired under the old, ambiguous grant exactly the stored-data
    access the split exists to withhold; upgrading it to `OBSERVE` would be a
    silent downgrade nobody asked for. So the record simply stops parsing --
    `_parse_device` drops it, exactly as it drops any unknown capability
    string, rather than raising and taking the whole store down with it. The
    device re-pairs, which is what this milestone is for.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    device_id = vault.verify(token).device_id
    raw = json.loads((tmp_path / "devices.json").read_text(encoding="utf-8"))
    raw["devices"][0]["grants"] = ["chat"]
    (tmp_path / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    assert vault.verify(token) is None
    assert vault.devices() == []            # dropped, not raised
    assert vault.revoke(device_id) is True  # and still administrable by id


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
    ghost_token = vault.issue("ghost", frozenset({Capability.OBSERVE}))
    ghost_id = vault.verify(ghost_token).device_id
    normal_token = vault.issue("normal", frozenset({Capability.OBSERVE}))

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
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
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
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    (tmp_path / "instance_secret").write_text("", encoding="utf-8")

    fresh = TokenVault(tmp_path)  # no in-memory cache -- forces a file read
    secret = fresh.instance_secret()
    assert len(secret) == 32
    assert fresh.verify(token) is None


def test_whitespace_only_secret_file_regenerates_and_revokes_everything(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
    (tmp_path / "instance_secret").write_text("   \n\t  ", encoding="utf-8")

    fresh = TokenVault(tmp_path)
    secret = fresh.instance_secret()
    assert len(secret) == 32
    assert fresh.verify(token) is None


def test_short_but_valid_hex_secret_file_regenerates_and_revokes_everything(tmp_path):
    vault = TokenVault(tmp_path)
    token = vault.issue("studio", frozenset({Capability.OBSERVE}))
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

    token = daemon.issue("phone", frozenset({Capability.OBSERVE}))

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
    token = vault.issue("phone", frozenset({Capability.OBSERVE}))
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
    token = vault.issue("phone", frozenset({Capability.OBSERVE}))
    (tmp_path / "instance_secret").unlink()

    with caplog.at_level(logging.WARNING, logger="assistant.io.api.vault"):
        assert vault.verify(token) is not None

    assert "keeping the secret already in memory" in caplog.text
    # No silent re-persist either: rewriting the file would hide the loss and
    # bake whichever process noticed first in as the winner.
    assert not (tmp_path / "instance_secret").exists()


def test_corrupt_secret_file_mid_run_does_not_revoke_issued_devices(tmp_path, caplog):
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.OBSERVE}))
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


# ─── last_seen_at ──────────────────────────────────────────────────────────


def test_a_new_device_has_no_last_seen(tmp_path):
    vault = TokenVault(tmp_path)
    # `Capability.CHAT` (the brief's literal value) predates the Task 5b
    # OBSERVE/RECALL split and no longer exists; OBSERVE is the equivalent
    # "just issued, still valid" grant for this test's purpose.
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    assert vault.devices()[0].last_seen_at is None


def test_touch_records_a_timestamp(tmp_path):
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id
    vault.touch(device_id)
    assert vault.devices()[0].last_seen_at is not None


def test_touching_an_unknown_device_is_silent(tmp_path):
    TokenVault(tmp_path).touch("nope")      # must not raise


def test_a_record_without_last_seen_still_parses(tmp_path):
    """Records written before this task exist on disk. A missing key is not a
    malformed record.

    Grants use `"observe"` rather than the brief's literal `"chat"`: that
    string was retired by the Task 5b OBSERVE/RECALL split and is *supposed*
    to fail closed and drop (tested elsewhere) -- using it here would make
    this test pass for the wrong reason, on an empty device list, instead of
    proving a missing `last_seen_at` key parses as `None`.
    """
    (tmp_path / "devices.json").write_text(json.dumps({
        "version": 1,
        "devices": [{"device_id": "a", "label": "old", "grants": ["observe"],
                     "created_at": "2026-01-01T00:00:00+00:00", "token_hmac": "x"}],
    }), encoding="utf-8")
    assert TokenVault(tmp_path).devices()[0].last_seen_at is None


# ─── touch() throttling ────────────────────────────────────────────────────
# `touch()` sits on the authenticated request path (Task 10). Without a
# floor, every call pays a full devices.json rewrite plus an `icacls`
# subprocess -- a per-request cost on the one API that will be reachable
# from a public URL. These assert on the file itself, not on `devices()`,
# because the point being tested is the *absence* of I/O, which a read-back
# through the vault's own re-parsing could not distinguish from "wrote the
# same value back".


def test_first_touch_on_a_device_with_no_stored_timestamp_always_writes(tmp_path):
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id
    path = tmp_path / "devices.json"
    before = path.read_bytes()

    vault.touch(device_id)

    assert path.read_bytes() != before
    assert vault.devices()[0].last_seen_at is not None


def test_touch_within_the_throttle_window_does_not_write_to_disk(tmp_path):
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id
    vault.touch(device_id)  # first touch: no stored timestamp yet, always writes

    path = tmp_path / "devices.json"
    before = path.read_bytes()
    vault.touch(device_id)  # second touch, well inside the 60s window

    assert path.read_bytes() == before, "a touch within the window must not write at all"


def test_touch_after_the_throttle_window_does_write(tmp_path):
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    path = tmp_path / "devices.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    stale = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
    raw["devices"][0]["last_seen_at"] = stale
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()

    vault.touch(device_id)

    assert path.read_bytes() != before
    assert vault.devices()[0].last_seen_at != stale


def test_a_future_stored_timestamp_does_not_wedge_the_write_permanently(tmp_path):
    """A clock change or a hand-edited file could leave `last_seen_at` in the
    future. That must not disable touching forever -- age computed against a
    future timestamp is negative, which the throttle window (0 <= age <
    threshold) excludes, so the write proceeds and overwrites the bogus value.
    """
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    path = tmp_path / "devices.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    raw["devices"][0]["last_seen_at"] = future
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()

    vault.touch(device_id)

    assert path.read_bytes() != before
    assert vault.devices()[0].last_seen_at != future


def test_a_naive_stored_timestamp_does_not_raise_and_self_heals(tmp_path):
    """`datetime.fromisoformat("2026-01-01T00:00:00")` does not raise -- it
    returns a naive datetime -- so subtracting it from an aware `now` is what
    actually blows up if unguarded. Reproduces the round-2 Critical: this
    must come back clean, and the stored value must end up aware.
    """
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    path = tmp_path / "devices.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["devices"][0]["last_seen_at"] = "2026-01-01T00:00:00"  # naive, no offset
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()

    vault.touch(device_id)  # must not raise TypeError

    assert path.read_bytes() != before
    healed = vault.devices()[0].last_seen_at
    assert healed != "2026-01-01T00:00:00"
    assert datetime.fromisoformat(healed).tzinfo is not None


def test_a_garbage_stored_timestamp_does_not_raise_and_overwrites(tmp_path):
    """Untested in round 1 even though the branch existed: a stored value
    that isn't ISO-8601 at all must not raise, and must be overwritten same
    as the naive case above.
    """
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    path = tmp_path / "devices.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["devices"][0]["last_seen_at"] = "not-a-timestamp"
    path.write_text(json.dumps(raw), encoding="utf-8")
    before = path.read_bytes()

    vault.touch(device_id)  # must not raise

    assert path.read_bytes() != before
    assert vault.devices()[0].last_seen_at != "not-a-timestamp"


# ─── I-3: a read failure is not the same as an empty vault ────────────────
# `_load()` used to swallow `OSError`/`json.JSONDecodeError` into a
# synthetic empty document -- indistinguishable from a fresh install. That
# made every mutator unsafe: `issue()` appended to the fake-empty list and
# saved, destroying every other device the moment a lock (a scanner, a
# backup tool, a second TENKA process) coincided with a write. Proved on
# Windows under real contention (`PermissionError [WinError 5]`); simulated
# here deterministically by breaking reads of exactly this vault's
# `devices.json`, nothing else -- not the instance secret file, which lives
# in the same directory and must stay readable throughout.


def _break_devices_json_reads(monkeypatch, tmp_path):
    target = tmp_path / "devices.json"
    original_read_text = Path.read_text

    def broken(self, *args, **kwargs):
        if self == target:
            raise PermissionError("simulated lock")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken)
    return original_read_text


def test_verify_fails_closed_on_a_genuine_read_failure(tmp_path, monkeypatch, caplog):
    """Decision: `verify()` treats an unreadable devices.json the same way it
    treats an unknown or revoked token -- refuse. This is the call
    `authenticate()` makes on every request specifically so a revocation is
    never stale, so the alternative (verify whatever was last cached) is not
    a thing this vault does anywhere else. The cost is real -- every device
    is refused for as long as the lock holds -- and accepted on purpose.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.OBSERVE}))
    original = _break_devices_json_reads(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING, logger="assistant.io.api.vault"):
        assert vault.verify(token) is None

    monkeypatch.setattr(Path, "read_text", original)
    assert "could not read devices.json" in caplog.text
    assert vault.verify(token) is not None, (
        "the device must still be there once the read recovers -- verify() "
        "must not have treated the failure as a revocation"
    )


def test_devices_reports_none_on_a_genuine_read_failure(tmp_path, monkeypatch):
    """Decision: `devices()` also fails closed to an empty listing. This is an
    admin-only read (`GET /v1/devices`, `/studio devices`), not a security
    decision -- a stale "none issued" while a lock clears costs a confusing
    UI, not a privilege handed to someone who should not have had it.
    """
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    original = _break_devices_json_reads(monkeypatch, tmp_path)

    assert vault.devices() == []

    monkeypatch.setattr(Path, "read_text", original)
    assert len(vault.devices()) == 1, "the device must still be there once the read recovers"


def test_issue_raises_rather_than_destroying_every_other_device(tmp_path, monkeypatch):
    """The finding, reproduced directly: `issue()` used to overwrite
    devices.json with a document containing only the new device, silently
    destroying everything else, whenever the read that should have loaded
    the real document failed instead. It must now raise and write nothing.
    """
    vault = TokenVault(tmp_path)
    vault.issue("existing", frozenset({Capability.OBSERVE}))
    original = _break_devices_json_reads(monkeypatch, tmp_path)

    with pytest.raises(VaultReadError):
        vault.issue("new", frozenset({Capability.OBSERVE}))

    monkeypatch.setattr(Path, "read_text", original)
    assert {d.label for d in vault.devices()} == {"existing"}, (
        "issue() must not have written anything while the read was failing"
    )


def test_revoke_raises_rather_than_reporting_a_false_not_found(tmp_path, monkeypatch):
    """`revoke()` returning `False` under a read failure would tell the one
    person trying to cut a device off that it is already gone, when the
    truth is "the vault could not be read." That is worse than an error.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.verify(token).device_id
    original = _break_devices_json_reads(monkeypatch, tmp_path)

    with pytest.raises(VaultReadError):
        vault.revoke(device_id)

    monkeypatch.setattr(Path, "read_text", original)
    assert vault.verify(token) is not None, "not actually revoked -- the write never happened"


def test_touch_raises_rather_than_silently_skipping(tmp_path, monkeypatch):
    """`touch()`'s existing silence covers "the device is not in the
    document" -- a stale or revoked credential on the request path is
    ordinary. "The document could not be read at all" is a different fact
    and must not collapse into that same silence, or a target that is
    genuinely present in a stale-but-readable snapshot could be skipped as if
    it were merely unknown.
    """
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id
    original = _break_devices_json_reads(monkeypatch, tmp_path)

    with pytest.raises(VaultReadError):
        vault.touch(device_id)

    monkeypatch.setattr(Path, "read_text", original)
    assert vault.devices()[0].last_seen_at is None, "no write happened while the read was failing"


@pytest.mark.parametrize("bad_value", ["2026-01-01T00:00:00", "not-a-timestamp"])
def test_a_healed_value_is_well_formed_and_throttles_the_next_touch(tmp_path, bad_value):
    """The point of healing is that the written value is a real, aware
    timestamp usable by the throttle itself -- not merely "some string that
    replaced the bad one". Prove it by touching again right after the heal
    and confirming the second touch is suppressed, exactly like the
    well-formed case in `test_touch_within_the_throttle_window_does_not_write_to_disk`.
    """
    vault = TokenVault(tmp_path)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    path = tmp_path / "devices.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["devices"][0]["last_seen_at"] = bad_value
    path.write_text(json.dumps(raw), encoding="utf-8")

    vault.touch(device_id)  # heals the bad value
    healed_bytes = path.read_bytes()

    vault.touch(device_id)  # immediately after -- must be throttled

    assert path.read_bytes() == healed_bytes


# ─── Write races: _load-through-_save must be one critical section ────────
# A barrier alone does not prove anything here -- it only synchronises thread
# *start*, and CPython's GIL serialises these short critical sections tightly
# enough that a racing pair can pass by accident even with no lock at all.
# Both tests below force the interleaving explicitly: the first `_load()` call
# is made to pause -- after capturing its snapshot, before returning it -- so
# the racing caller is provably still running when the second call is allowed
# to proceed. `_load` is patched on the *instance*, not the class, so other
# tests are unaffected.


def test_touch_cannot_resurrect_a_device_revoked_mid_sequence(tmp_path):
    """Reproduces the bug from the brief directly: touch() loads a snapshot
    that still has the device, is held there while revoke() runs to
    completion, and then must not save that stale snapshot back over the
    revocation. Without a lock spanning `_load()` through `_save()`, it does
    exactly that -- the device comes back.
    """
    vault = TokenVault(tmp_path)
    token = vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.verify(token).device_id

    touch_loaded = threading.Event()   # touch() has captured its snapshot
    let_touch_save = threading.Event()  # revoke() has finished; touch() may resume
    original_load = vault._load
    call_count = {"n": 0}

    def blocking_load():
        # `_load` is patched on the instance, so *every* caller through this
        # vault goes through here, including revoke()'s own `_load()` call.
        # Only the first call (touch's) pauses; revoke's call must read the
        # live document and return immediately, or the two threads would
        # deadlock waiting on each other's events instead of racing.
        call_count["n"] += 1
        data = original_load()
        if call_count["n"] == 1:
            touch_loaded.set()
            assert let_touch_save.wait(timeout=5), "revoke() never let touch() resume"
        return data

    vault._load = blocking_load
    touch_thread = threading.Thread(target=vault.touch, args=(device_id,))
    touch_thread.start()
    assert touch_loaded.wait(timeout=5), "touch() never reached _load()"

    # touch() is now holding a snapshot that still contains the device (with
    # the lock in place, it is also still holding `self._lock`). Run revoke()
    # to completion -- or, with the fix, until it blocks on that lock -- on
    # its own thread; the bounded join just keeps this from hanging either way.
    revoke_thread = threading.Thread(target=vault.revoke, args=(device_id,))
    revoke_thread.start()
    revoke_thread.join(timeout=1.0)

    let_touch_save.set()
    touch_thread.join(timeout=5)
    revoke_thread.join(timeout=5)
    vault._load = original_load

    assert not touch_thread.is_alive() and not revoke_thread.is_alive()
    assert vault.verify(token) is None, "touch() resurrected a device revoked mid-sequence"


def test_issue_racing_revoke_loses_neither_the_new_device_nor_the_revocation(tmp_path):
    """Same class of bug, the other direction: issue() appending a new device
    and revoke() removing an existing one are both load-mutate-save. If they
    interleave, whichever saves last wins outright and the other caller's
    write is silently gone -- either the new device never really got added,
    or the revoked one comes back.
    """
    vault = TokenVault(tmp_path)
    existing_token = vault.issue("existing", frozenset({Capability.OBSERVE}))
    existing_id = vault.verify(existing_token).device_id

    revoke_loaded = threading.Event()   # revoke() has captured its snapshot
    let_revoke_save = threading.Event()  # issue() has finished; revoke() may resume
    original_load = vault._load
    call_count = {"n": 0}

    def blocking_load():
        # Only the first call (revoke's) pauses. A second call through this
        # same patched attribute -- issue()'s, if it gets that far before the
        # lock releases -- must read live, not repeat the pause.
        call_count["n"] += 1
        data = original_load()
        if call_count["n"] == 1:
            revoke_loaded.set()
            assert let_revoke_save.wait(timeout=5), "issue() never let revoke() resume"
        return data

    vault._load = blocking_load
    revoke_thread = threading.Thread(target=vault.revoke, args=(existing_id,))
    revoke_thread.start()
    assert revoke_loaded.wait(timeout=5), "revoke() never reached _load()"

    # revoke() is now holding a snapshot with the existing device still in it
    # (with the lock in place, also still holding `self._lock`). Run issue()
    # to completion -- or, with the fix, until it blocks on that lock.
    issue_thread = threading.Thread(
        target=vault.issue, args=("new", frozenset({Capability.OBSERVE}))
    )
    issue_thread.start()
    issue_thread.join(timeout=1.0)

    let_revoke_save.set()
    revoke_thread.join(timeout=5)
    issue_thread.join(timeout=5)
    vault._load = original_load

    assert not revoke_thread.is_alive() and not issue_thread.is_alive()
    labels = sorted(d.label for d in vault.devices())
    assert labels == ["new"], (
        f"expected only the new device to survive the race, got {labels}"
    )
