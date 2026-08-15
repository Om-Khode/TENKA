"""/studio slash command — local revocation path for Studio device tokens.

Revocation is deliberately a slash command, not an HTTP route: a route
behind `system_control` would be reachable by the very token being revoked,
and pairing's trust anchor ("you are physically at the desktop") should be
the same anchor revocation rests on. See ARCHITECTURE.md / the milestone
design doc for the full reasoning.

Every test here points the vault at a tmp_path via `config.SANDBOX_DIR` --
never at the real `~/TENKA/`, which would revoke the developer's actual
Studio pairings.
"""
from __future__ import annotations

import pytest

from assistant import config, slash_commands
from assistant.io.api.vault import Capability, TokenVault


@pytest.fixture()
def vault_root(tmp_path, monkeypatch):
    """Redirect the vault the command reaches for to an isolated tmp dir."""
    monkeypatch.setattr(config, "SANDBOX_DIR", tmp_path)
    return tmp_path


# ─── RESERVED ───────────────────────────────────────────────────────────────


def test_studio_is_reserved():
    """`studio` must not be treated as a runtime-setting shortcut key."""
    assert "studio" in slash_commands.RESERVED


# ─── /studio devices ────────────────────────────────────────────────────────


def test_devices_with_none_issued_says_so_plainly(vault_root):
    result = slash_commands.handle("/studio devices")
    assert result == "No Studio devices issued."


def test_devices_lists_id_label_grants_and_created_at(vault_root):
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    vault.issue("browser", frozenset({Capability.OBSERVE, Capability.SCREEN}))
    ids = {d.label: d.device_id for d in vault.devices()}

    result = slash_commands.handle("/studio devices")

    assert ids["phone"] in result
    assert ids["browser"] in result
    assert "phone" in result
    assert "browser" in result
    assert "observe" in result.lower()
    assert "screen" in result.lower()
    # created_at is an ISO timestamp -- year at minimum should show up.
    assert "20" in result


def test_devices_never_prints_a_plaintext_token(vault_root):
    vault = TokenVault(vault_root)
    token = vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    result = slash_commands.handle("/studio devices")

    # Assert the device actually shows up -- otherwise "token not in result"
    # would pass just as well against "No Studio devices issued.".
    assert device_id in result
    assert token not in result


def test_devices_survives_a_corrupt_devices_json(vault_root):
    """A hand-edited/corrupt devices.json fails closed to an empty list
    (TokenVault._load already guarantees this) -- the command must not
    raise, and must report the same "none issued" message rather than a
    stack trace or a misleading count.
    """
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    (vault_root / "devices.json").write_text("not json at all", encoding="utf-8")

    result = slash_commands.handle("/studio devices")

    assert result == "No Studio devices issued."


def test_devices_survives_one_malformed_entry_among_good_ones(vault_root):
    """A single hand-edited bad entry drops out of the listing; the rest
    still show up -- matches TokenVault._parse_device's fail-closed-per-entry
    behavior (see test_api_vault.py's malformed-entry tests).
    """
    import json

    vault = TokenVault(vault_root)
    vault.issue("good", frozenset({Capability.OBSERVE}))
    vault.issue("bad", frozenset({Capability.OBSERVE}))
    raw = json.loads((vault_root / "devices.json").read_text(encoding="utf-8"))
    for entry in raw["devices"]:
        if entry["label"] == "bad":
            entry["grants"] = ["not-a-real-capability"]
    (vault_root / "devices.json").write_text(json.dumps(raw), encoding="utf-8")

    result = slash_commands.handle("/studio devices")

    assert "good" in result
    assert "bad" not in result


# ─── /studio revoke <device_id> ─────────────────────────────────────────────


def test_revoke_known_device_succeeds_and_removes_it(vault_root):
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    result = slash_commands.handle(f"/studio revoke {device_id}")

    assert result == f"Revoked Studio device {device_id}."
    assert TokenVault(vault_root).devices() == []


def test_revoke_unknown_device_id_reports_no_match(vault_root):
    result = slash_commands.handle("/studio revoke not-a-real-id")
    assert result == "No Studio device found with id 'not-a-real-id' -- nothing was revoked."


def test_revoke_success_and_failure_read_differently(vault_root):
    """A wrong id must not read like success -- the two messages must not
    share a distinguishing prefix a skimming user could confuse.
    """
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    success = slash_commands.handle(f"/studio revoke {device_id}")
    failure = slash_commands.handle(f"/studio revoke {device_id}")  # already gone

    assert success.startswith("Revoked")
    assert failure.startswith("No Studio device found")
    assert success != failure


def test_revoke_with_no_id_returns_usage(vault_root):
    result = slash_commands.handle("/studio revoke")
    assert "Usage" in result


def test_revoke_with_extra_token_after_device_id_is_rejected(vault_root):
    """Ambiguous input (two tokens that aren't the 'all confirm' pair) must
    not be guessed at -- it should read as a usage error, not silently
    revoke based on the first token.
    """
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    device_id = vault.devices()[0].device_id

    result = slash_commands.handle(f"/studio revoke {device_id} extra")

    assert "Usage" in result
    assert len(TokenVault(vault_root).devices()) == 1


# ─── /studio revoke all [confirm] ───────────────────────────────────────────


def test_revoke_all_without_confirm_does_not_revoke_anything(vault_root):
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))
    vault.issue("browser", frozenset({Capability.OBSERVE}))

    result = slash_commands.handle("/studio revoke all")

    assert "confirm" in result.lower()
    assert len(TokenVault(vault_root).devices()) == 2


def test_revoke_all_without_confirm_mentions_it_is_destructive(vault_root):
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))

    result = slash_commands.handle("/studio revoke all")

    assert "irreversible" in result.lower() or "re-pair" in result.lower()


def test_revoke_all_confirm_revokes_every_device(vault_root):
    vault = TokenVault(vault_root)
    token_a = vault.issue("phone", frozenset({Capability.OBSERVE}))
    token_b = vault.issue("browser", frozenset({Capability.OBSERVE}))

    result = slash_commands.handle("/studio revoke all confirm")

    assert "revoked" in result.lower()
    assert "re-pair" in result.lower()
    fresh = TokenVault(vault_root)
    assert fresh.devices() == []
    assert fresh.verify(token_a) is None
    assert fresh.verify(token_b) is None


def test_revoke_all_near_miss_confirmation_does_not_revoke(vault_root):
    """Only the exact 'confirm' token authorizes the destructive path --
    a near-miss like 'confirmed' or 'confirm please' must not slip through.
    """
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))

    result_a = slash_commands.handle("/studio revoke all confirmed")
    result_b = slash_commands.handle("/studio revoke all confirm please")

    assert len(TokenVault(vault_root).devices()) == 1
    assert "revoked" not in result_a.lower().split("all")[0]
    assert "revoked" not in result_b.lower().split("all")[0]


# ─── /studio with no subcommand ─────────────────────────────────────────────


def test_studio_with_no_subcommand_returns_usage(vault_root):
    result = slash_commands.handle("/studio")
    assert "Usage" in result


def test_studio_with_unknown_subcommand_returns_usage(vault_root):
    result = slash_commands.handle("/studio frobnicate")
    assert "Usage" in result
