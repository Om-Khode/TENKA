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

import re

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


def test_devices_shows_last_seen_column(vault_root):
    """Task 11: `Device.last_seen_at` (Task 9) gains a column in the listing."""
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))

    result = slash_commands.handle("/studio devices")

    assert "last seen" in result.lower()


def test_devices_shows_never_for_a_device_not_yet_seen(vault_root):
    """A freshly issued device has `last_seen_at is None` -- the column must
    say so plainly rather than print the literal string 'None' or omit the
    device from the listing.
    """
    vault = TokenVault(vault_root)
    vault.issue("phone", frozenset({Capability.OBSERVE}))

    result = slash_commands.handle("/studio devices")

    assert "none" not in result.lower()
    assert "never" in result.lower()


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


# ─── /studio pair ────────────────────────────────────────────────────────────
# `PairCodeStore` is deliberately in-memory only (see pairing.py's docstring),
# so it is not reachable through a fresh construction the way the file-backed
# vault is via `_studio_vault()`. It only exists as `assistant.main`'s
# module-level `_studio_pair_store`, set once by `_start_studio_daemon()`.
# Every test below drives that global directly rather than actually starting
# the daemon, the same way test_api_server_lifecycle.py pins the vault
# equivalent.


@pytest.fixture()
def running_pair_store(monkeypatch):
    """Simulate a running Studio daemon by installing a live store on the
    module global `/studio pair` has to read -- without booting uvicorn."""
    import assistant.main as main_mod
    from assistant.io.api.pairing import PairCodeStore

    store = PairCodeStore()
    monkeypatch.setattr(main_mod, "_studio_pair_store", store)
    return store


def test_pair_prints_a_code_and_expiry(running_pair_store):
    out = slash_commands.handle("/studio pair phone")
    assert "expires" in out.lower()
    assert re.search(r"[0-9A-Z]{4}-[0-9A-Z]{4}", out)


def test_pair_defaults_the_label(running_pair_store):
    assert slash_commands.handle("/studio pair").strip() != ""


def test_pair_prints_an_ascii_qr_not_svg(running_pair_store):
    """`qr_svg()` renders `<svg>...</svg>`, which is useless in a terminal --
    the console command must use `qrcode`'s ASCII renderer instead."""
    out = slash_commands.handle("/studio pair phone")
    assert "<svg" not in out.lower()
    # An ASCII QR is built from block characters over many lines -- a real
    # render is always more than a couple of lines long.
    assert len(out.splitlines()) > 5


def test_pair_is_reserved_and_not_a_setting_key():
    assert "studio" in slash_commands.RESERVED


def test_pair_refuses_plainly_when_the_daemon_is_not_running(monkeypatch):
    """No daemon means no store at all -- printing a code nobody could ever
    redeem would be worse than an honest refusal."""
    import assistant.main as main_mod
    monkeypatch.setattr(main_mod, "_studio_pair_store", None)

    result = slash_commands.handle("/studio pair phone")

    assert "not running" in result.lower()
    assert not re.search(r"[0-9A-Z]{4}-[0-9A-Z]{4}", result)


def test_pair_never_grants_system_control_by_default(running_pair_store):
    """The equivalent of `POST /v1/pair/code`'s intersection-with-the-minting-
    device's-own-grants restraint: a console command has no device to
    intersect against, so the restraint here is refusing the capabilities
    that let a paired device act on the machine rather than talk to it.

    Both exclusions are named, not derived. This assertion used to read
    `frozenset(Capability) - {SYSTEM_CONTROL}`, which meant it kept passing
    when EXECUTE joined the enum and quietly widened the default to include
    the strongest capability in the model. A test written as a subtraction
    from the enum cannot notice the enum growing -- that is the whole failure
    mode, and it is why the ceilings in policy.py are literals now too.
    """
    slash_commands.handle("/studio pair phone")
    pair_code = running_pair_store.current()
    assert Capability.SYSTEM_CONTROL not in pair_code.grants
    assert Capability.EXECUTE not in pair_code.grants
    # Still useful as a remote: watch, recall, chat, read files, see the screen.
    assert pair_code.grants == frozenset({
        Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
        Capability.SCREEN, Capability.FILES,
    })


def test_pair_mints_into_the_store_the_pair_route_can_redeem(vault_root, running_pair_store):
    """The property that matters most: /studio pair must mint into the exact
    store `POST /v1/pair` consults, not a look-alike of its own. Proved by
    minting through the slash command, then redeeming that code through a
    real app wired to the same store, and confirming a device actually lands
    in the vault -- an output-only assertion would pass even if the command
    minted into a store nobody could ever reach.
    """
    from tests.fakes.api_client import build_api_client
    from tests.fakes.studio_runtime import build_fake_runtime

    vault = TokenVault(vault_root)

    out = slash_commands.handle("/studio pair phone")
    match = re.search(r"[0-9A-Z]{4}-[0-9A-Z]{4}", out)
    assert match, f"no code found in: {out!r}"
    code = match.group(0)

    client = build_api_client(build_fake_runtime(), vault, pair_store=running_pair_store)
    response = client.post("/v1/pair", json={"code": code})

    assert response.status_code == 204
    assert any(d.label == "phone" for d in vault.devices())


# ─── Fix round: the daemon stop path must invalidate the store ─────────────
# C1's review found the store was never cleared on stop, so `/studio pair`
# kept reporting success after the daemon it belonged to had already gone
# away -- the exact "prints a code nobody can redeem" failure this task
# exists to prevent, reached through a stale global instead of a fresh
# construction.


@pytest.mark.asyncio
async def test_pair_refuses_after_the_daemon_is_stopped(running_pair_store):
    import assistant.main as main_mod

    # A no-task stop is exactly what happens when the daemon never actually
    # got a task back from serve() (see _start_studio_daemon's failure
    # path) -- and it is also the simplest way to drive the real
    # _stop_studio_daemon() code path without booting uvicorn.
    await main_mod._stop_studio_daemon(None)

    result = slash_commands.handle("/studio pair phone")

    assert "not running" in result.lower()
    assert not re.search(r"[0-9A-Z]{4}-[0-9A-Z]{4}", result)
    assert main_mod._studio_pair_store is None


# ─── C2: the code must never reach a log line ──────────────────────────────
# main.py speaks (and, before this fix, would have logged) only
# `response.split("\n", 1)[0]` for every non-"chat" source. A code sitting
# on the first line of /studio pair's response would ride straight into
# tts.speak()'s log line the moment this command is ever reached by voice
# (structurally possible even though "voice rarely produces a leading /").


def test_pair_response_keeps_the_code_off_the_first_line(running_pair_store):
    """The property main.py's `spoken = response.split("\\n", 1)[0][:200]`
    actually depends on. `redact_secrets()` cannot be trusted to catch a
    9-character pair code on its own (see test_speak_redacts.. below for
    what it *can* catch) -- so the code must never be on line one at all.
    """
    out = slash_commands.handle("/studio pair phone")
    first_line = out.split("\n", 1)[0]
    assert not re.search(r"[0-9A-Z]{4}-[0-9A-Z]{4}", first_line), (
        f"the pair code must not be on the first line -- got: {first_line!r}"
    )
    # Sanity: the code is still somewhere in the full response (a human at
    # the console still needs to read it).
    assert re.search(r"[0-9A-Z]{4}-[0-9A-Z]{4}", out)


@pytest.mark.asyncio
async def test_speak_redacts_a_secret_shaped_string_before_logging(monkeypatch, caplog):
    """Generic defence-in-depth on `tts.speak()`'s own log line, for the
    class C2 named ("nothing upstream of speak() guarantees secret-free
    text"), not scoped to pair codes -- a 9-character code is below every
    threshold `redact_secrets()` checks (see the test above for the fix
    that actually protects it: keeping it off line one in the first
    place). What this function *can* catch -- a labelled secret like
    "password is ..." -- must actually be caught, proving the import and
    the call are real and not merely present in a docstring.

    Driven with `_pipeline` forced `None` and `init_tts()` stubbed to fail,
    so `speak()` bails out immediately after the log line under test --
    no Kokoro synthesis, no sounddevice, matching this suite's existing
    rule against real audio/model calls.
    """
    import logging

    from assistant.io.audio import tts as tts_mod

    monkeypatch.setattr(tts_mod, "_pipeline", None)
    monkeypatch.setattr(tts_mod, "init_tts", lambda: False)

    secret_text = "my password is Sup3rSecretPassw0rd"
    with caplog.at_level(logging.INFO, logger="tts"):
        result = await tts_mod.speak(secret_text, bridge=None)

    assert result is False, "sanity: speak() must have bailed out at the stubbed init_tts(), not run real synthesis"
    assert any("Speaking:" in r.message for r in caplog.records), (
        "sanity: the log line under test must actually have fired"
    )
    for record in caplog.records:
        assert "Sup3rSecretPassw0rd" not in record.message
    assert any("[REDACTED]" in r.message for r in caplog.records), (
        "redact_secrets() must actually have replaced the value, not just "
        "have been imported"
    )
