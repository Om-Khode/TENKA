"""Milestone 6a.5, stream F — redaction, the audit gate, and log framing.

Three findings, one file each side of them:

  * G2 (lens 2, F2) — `GET /v1/audit` enumerates every device id from a
    non-admin listener, while `GET /v1/devices` — the same class of data,
    by `devices.py`'s own docstring — is `require_admin`.
  * G10 (lens 7, HIGH) — `redact_secrets_strict` has four proven bypasses,
    and it is what the `FILES` file-preview route runs on a whole file.
  * G12 (lens 1, F6) — `intent.py` and `tts.py` interpolate user text into a
    log line with no `!r`, so a newline forges a whole fabricated log line
    in the file an operator greps after an incident.

`main.py:839,841` is the other half of G12 and belongs to stream A; nothing
here touches it.
"""
from __future__ import annotations

import inspect
import logging
import pathlib

import pytest

from assistant.core.redact import redact_secrets, redact_secrets_strict
from assistant.io.api.security import COOKIE_NAME
from assistant.io.api.vault import Capability, TokenVault
from tests.fakes.api_client import LOCAL_PORT, build_api_client
from tests.fakes.studio_runtime import build_fake_runtime

REDACTED = "[REDACTED]"


def _async_return(value):
    async def _call(*_args, **_kwargs):
        return value
    return _call


# ─── F1 / G2: /v1/audit becomes admin-only ───────────────────────────────

@pytest.fixture()
def audit_context(tmp_path):
    vault = TokenVault(tmp_path)
    runtime = build_fake_runtime()
    client = build_api_client(runtime, vault)
    token = vault.issue("studio", frozenset(Capability))
    return client, token


def test_audit_is_admin_only_like_devices():
    """devices.py's own docstring names /v1/audit as the same class of thing:
    "this is the security configuration, the same class of thing as
    /v1/audit". It was not in the same tier."""
    from assistant.io.api.routes import system
    src = inspect.getsource(system.audit)
    assert "require_admin" in src, src


def test_audit_still_works_from_the_local_listener(audit_context):
    """Control: the operator must still be able to read her own audit log.
    An admin gate that also locks out the keyboard has broken the feature."""
    client, token = audit_context
    response = client.get("/v1/audit", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    assert "entries" in response.json()["data"]


def test_a_non_admin_listener_cannot_enumerate_devices_out_of_the_audit_log(tmp_path):
    """Lens 2 F2's own scenario, reduced to one listener: a session that is
    refused /v1/devices must not read the same device ids out of /v1/audit."""
    vault = TokenVault(tmp_path)
    runtime = build_fake_runtime()
    client = build_api_client(runtime, vault, policies={LOCAL_PORT: "tailnet"})
    # A cookie, not a bearer header: a tunnel listener only accepts the
    # browser credential, which is what the cookie-auth suite uses too. The
    # grant set is deliberately maximal -- whatever this client cannot reach,
    # the listener refused, not the grant.
    token = vault.issue("phone", frozenset(Capability))
    client.cookies.set(COOKIE_NAME, token)

    devices = client.get("/v1/devices")
    assert devices.status_code == 403, devices.text

    audit = client.get("/v1/audit")
    assert audit.status_code == 403, (
        "a non-admin listener read the audit log after being refused "
        f"/v1/devices: {audit.text}"
    )


def test_the_audit_route_docstring_is_untouched():
    """Route docstrings are published as OpenAPI `description` and
    `ui.contract_hash()` fingerprints the schema, so editing one takes the
    vendored Studio bundle dark with a stale-contract 503. The reasoning for
    this change belongs in a body comment."""
    from assistant.io.api.routes import system
    assert system.audit.__doc__ is None


# ─── F2 / G10: the four proven redaction bypasses ────────────────────────

def test_pretty_printed_json_with_a_strong_label_is_redacted():
    """Bypass 1. The filler class excludes '"', so the regex captures the
    two-character '":' as the value, fails the length floor, and leaves the
    real secret untouched -- on a label the redactor explicitly calls
    strong.

    The value here is deliberately one the bare high-entropy rule cannot
    save: lowercase hex, no separator, so nothing but the label can catch
    it. A value like `sk-live-4f9a...` is already redacted today by accident
    of its own shape, which is what made this bypass easy to miss."""
    out = redact_secrets_strict('{\n  "api_key": "0123456789abcdef01234"\n}')
    assert "0123456789abcdef01234" not in out, out


def test_a_json_password_label_is_redacted():
    """A real passphrase is short and shapeless, which is exactly why the
    strong-label tier exists."""
    out = redact_secrets_strict('{\n  "password": "hunter2"\n}')
    assert "hunter2" not in out, out


def test_minified_json_is_redacted_too():
    """The same shape with no pretty-printing. A line-anchored rule alone
    would miss it, and one `json.dumps` with no indent is not obfuscation."""
    out = redact_secrets_strict('{"client_secret":"swordfish"}')
    assert "swordfish" not in out, out


def test_a_json_key_keeps_its_name_so_the_preview_stays_readable():
    """A preview whose keys survive still tells the reader which values are
    set -- the whole point of previewing a config file."""
    out = redact_secrets_strict('{\n  "api_key": "0123456789abcdef01234"\n}')
    assert '"api_key"' in out, out
    assert REDACTED in out, out


def test_lowercase_snake_case_labels_are_redacted():
    """Bypass 2. More common in real .env/YAML than the UPPER_SNAKE shape the
    assignment rule expects, and `\\b` never fires inside `db_pass`."""
    out = redact_secrets_strict("db_pass=hunter2")
    assert "hunter2" not in out, out
    assert "db_pass=" in out, out

    out = redact_secrets_strict("client_secret: swordfish")
    assert "swordfish" not in out, out
    assert "client_secret:" in out, out


def test_a_camel_case_label_is_redacted():
    """`apiKey` lowercases to a label the redactor already knows."""
    out = redact_secrets_strict("apiKey = sk-abc123def456ghi789")
    assert "sk-abc123def456ghi789" not in out, out


def test_a_pem_private_key_body_does_not_survive_in_fragments():
    """Bypass 3. '+' and '/' are outside the bare charset, so the base64 body
    splits into runs, some under the 24-char floor, printing in the clear
    next to [REDACTED]."""
    pem = ("-----BEGIN PRIVATE KEY-----\n"
           "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC+abc/def\n"
           "-----END PRIVATE KEY-----")
    out = redact_secrets_strict(pem)
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC" not in out, out
    assert "abc/def" not in out, out
    assert "-----BEGIN PRIVATE KEY-----" in out, out


def test_a_pem_body_is_redacted_on_the_log_path_too():
    """A PEM body is never a diagnostic, so the usual "the log path must not
    get stricter" trade does not apply to this one shape."""
    pem = ("-----BEGIN RSA PRIVATE KEY-----\n"
           "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKc+abc/def\n"
           "-----END RSA PRIVATE KEY-----")
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKc" not in redact_secrets(pem)


def test_an_unterminated_pem_block_still_loses_its_body():
    """A truncated preview cuts the footer off; the body is still a key."""
    pem = ("-----BEGIN OPENSSH PRIVATE KEY-----\n"
           "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ+xyz/123\n")
    out = redact_secrets_strict(pem)
    assert "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ" not in out, out


def test_the_block_rule_is_scoped_to_private_keys():
    """Control: the new rule keys off "PRIVATE KEY", not off PEM framing, so
    a certificate -- the public half -- keeps its markers and is left to the
    ordinary rules. (Its base64 body is redacted by the pre-existing bare
    high-entropy rule; that is today's behaviour, unchanged here.)"""
    cert = ("-----BEGIN CERTIFICATE-----\n"
            "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA\n"
            "-----END CERTIFICATE-----")
    out = redact_secrets_strict(cert)
    assert "-----BEGIN CERTIFICATE-----" in out, out
    assert "-----END CERTIFICATE-----" in out, out


def test_a_hard_line_wrapped_secret_does_not_leak_its_second_half():
    r"""Bypass 4. `\S+` stops at the newline, so only the first half is
    redacted; the second half sits in plaintext directly below."""
    out = redact_secrets_strict("api_key=abcdefghijklmnop\nqrstuvwxyz0123456789")
    assert "qrstuvwxyz0123456789" not in out, out


def test_a_secret_wrapped_over_several_lines_loses_every_line():
    out = redact_secrets_strict(
        "api_key=abcdefghijklmnop\nqrstuvwxyz0123456789\nZZ99aabbccddeeff0011\n")
    assert "qrstuvwxyz0123456789" not in out, out
    assert "ZZ99aabbccddeeff0011" not in out, out


def test_the_line_after_a_redaction_is_not_eaten_when_it_is_prose():
    """Control for bypass 4's fix: a wrapped *secret* is an unbroken run of
    secret-alphabet characters. An English sentence is not, and swallowing
    the line below every redaction would shred a previewed note."""
    out = redact_secrets_strict("password: hunter2\nThen click submit to continue")
    assert "Then click submit to continue" in out, out


# ─── F2: the two control tests that must stay green ──────────────────────

def test_ordinary_prose_is_not_destroyed():
    """Over-redaction makes the file preview useless and is its own bug."""
    text = "The quick brown fox jumps over the lazy dog near the riverbank today."
    assert redact_secrets_strict(text) == text


def test_a_git_hash_is_not_redacted():
    """The existing hex/entropy trade-off is deliberate -- do not regress it."""
    out = redact_secrets_strict("commit 228602a4f9b1c3d5e7a9b2c4d6e8f0a1b3c5d7e9")
    assert "228602a" in out, out


def test_ordinary_python_source_survives_the_snake_case_rule():
    """The lowercase-identifier rule fires on the exact names ordinary source
    uses for ordinary things. Without the configuration-vs-code test in
    `_is_configuration_value`, previewing this repo's own modules blanks 90
    lines of working code -- measured, not guessed."""
    # `key = hashlib.sha256(...)` is deliberately absent: the bare word `key`
    # already trips the pre-existing weak-label rule, with or without this
    # change, and pinning it here would claim a regression that is not one.
    for line in ("client_secret = text.strip()",
                 "auth_url = parts[2]",
                 'auth_url = parsed["auth_url"].split("?")[0]',
                 "goal_tokens = {t for t in goal.lower().split() if len(t) >= 3}",
                 "key = 0xAF if command_id == 'volume_up' else 0xAE",
                 "is_secret = _looks_secret(value, min_len=8)"):
        assert redact_secrets_strict(line) == line, line


def test_a_switch_is_not_a_credential():
    """`enabled: true` is the most common configuration line there is, and a
    boolean has nothing to hide. Excluding it is a three-word exact list, not
    an entropy test creeping back in."""
    for line in ("allow_bearer=False,", "auth_enabled: true", "token_cache: none"):
        assert redact_secrets_strict(line) == line, line


def test_the_new_rules_name_no_brand():
    """THE rule, restated for the mechanisms this task adds."""
    source = pathlib.Path("assistant/core/redact.py").read_text(encoding="utf-8")
    for brand in ("github", "openai", "google", "aws", "slack", "stripe"):
        assert brand not in source.lower()


# ─── F3 / G12: user text cannot forge a line in debug.log ────────────────

def test_chat_text_cannot_forge_a_line_in_the_debug_log(caplog, monkeypatch):
    """One POST /v1/chat writes as many fabricated log lines as it likes into
    the file an operator greps after an incident. `ChatRequest.text` allows
    8,000 characters, and `redact_secrets` is about secrets, not framing --
    it passes newlines straight through."""
    import asyncio
    import types

    import assistant.intent as intent_mod

    stub = types.SimpleNamespace(
        build_intent_prompt=lambda **_kwargs: "system",
        ask_for_intent=_async_return('{"intent": "unknown", "response": ""}'),
    )
    monkeypatch.setattr(intent_mod, "llm", stub)

    forged = ("hello\n2026-08-16 12:00:00 [INFO] [API] "
              "device revoked (device=deadbeef)")
    with caplog.at_level(logging.INFO, logger="intent"):
        asyncio.run(intent_mod.detect_intent(forged))

    classifying = [r for r in caplog.records if "Classifying" in r.getMessage()]
    assert classifying, [r.getMessage() for r in caplog.records]
    message = classifying[0].getMessage()
    assert "\n" not in message, message
    assert "\\n" in message, message


def test_spoken_text_cannot_forge_a_line_in_the_debug_log(caplog, monkeypatch):
    """tts.py:275 has the identical shape for spoken text, and a slash
    command's response reaches it."""
    import asyncio

    import assistant.io.audio.tts as tts_mod

    # The log line sits above every synthesis call, so refusing to initialise
    # the pipeline returns False before anything reaches the speakers.
    monkeypatch.setattr(tts_mod, "_pipeline", None)
    monkeypatch.setattr(tts_mod, "init_tts", lambda: False)

    forged = "hi\n2026-08-16 12:00:00 [INFO] [API] device revoked (device=deadbeef)"
    with caplog.at_level(logging.INFO, logger=tts_mod.logger.name):
        asyncio.run(tts_mod.speak(forged))

    speaking = [r for r in caplog.records if "Speaking" in r.getMessage()]
    assert speaking, [r.getMessage() for r in caplog.records]
    message = speaking[0].getMessage()
    assert "\n" not in message, message
    assert "\\n" in message, message


def test_both_log_sites_still_redact_secrets():
    """`!r` is about framing and `redact_secrets` is about secrets. Adding
    the first must not drop the second -- they do different jobs and both
    are needed."""
    for path, marker in (("assistant/intent.py", "Classifying:"),
                         ("assistant/io/audio/tts.py", "Speaking:")):
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if marker in line and "logger.info" in line:
                assert "redact_secrets(" in line, f"{path}: {line.strip()}"
                assert "!r" in line, f"{path}: {line.strip()}"
                break
        else:
            raise AssertionError(f"no {marker!r} log site found in {path}")
