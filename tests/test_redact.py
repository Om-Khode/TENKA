"""Secret redaction — generic, brand-agnostic, applied before logging."""
from assistant.core.redact import redact_secrets

REDACTED = "[REDACTED]"


def test_leaves_ordinary_prose_untouched():
    text = "open the third file and read the second paragraph to me"
    assert redact_secrets(text) == text


def test_redacts_a_labelled_secret_but_keeps_the_label():
    out = redact_secrets("my api key is sk-abc123def456ghi789jkl")
    assert "sk-abc123def456ghi789jkl" not in out
    assert "api key" in out
    assert REDACTED in out


def test_redacts_a_labelled_password_mid_sentence():
    out = redact_secrets("set the password to hunter2please and then log in")
    assert "hunter2please" not in out
    assert "and then log in" in out


def test_redacts_a_long_high_entropy_token_with_no_label():
    token = "ghp_0Xk2LmQ9zRt4Ws7Yc1Vb3Nd6Fh8Jp5Ka2Le"
    out = redact_secrets(f"use {token} for that")
    assert token not in out
    assert out.startswith("use ")
    assert out.endswith(" for that")


def test_does_not_redact_a_long_ordinary_word():
    text = "she recommended internationalization immediately afterwards"
    assert redact_secrets(text) == text


def test_redacts_a_bearer_header_value():
    out = redact_secrets("Authorization: Bearer aaaabbbbccccddddeeeeffff00001111")
    assert "aaaabbbbccccddddeeeeffff00001111" not in out
    assert "Bearer" in out


def test_handles_empty_and_none_shaped_input():
    assert redact_secrets("") == ""


def test_is_idempotent():
    once = redact_secrets("token: ghp_0Xk2LmQ9zRt4Ws7Yc1Vb3Nd6Fh8Jp5Ka2Le")
    assert redact_secrets(once) == once


# ─── Regression: strong labels redact on the label alone ──────────────────
# "password" (and its close kin) is essentially never followed by a
# non-secret. A real passphrase can be a plain lowercase word or pure
# digits -- shape alone can never tell "letmein" apart from "admirer" -- so
# for these labels the label itself has to be enough evidence, with no
# digit/letter-mixing requirement on the value.

def test_redacts_a_plain_lowercase_password_with_no_digits():
    out = redact_secrets("my password is letmein")
    assert "letmein" not in out
    assert REDACTED in out


def test_redacts_a_digits_only_password_with_no_letters():
    out = redact_secrets("my password is 13579246")
    assert "13579246" not in out
    assert REDACTED in out


# ─── Regression: label words in ordinary conversation ─────────────────────
# A voice assistant logs conversation, and "key", "secret", "auth",
# "credential" and "token" are ordinary English words. A labelled value must
# still look secret-shaped before it is masked, or normal speech gets eaten.

def test_does_not_redact_key_used_as_an_ordinary_word():
    text = "the key thing to remember is to stay calm"
    assert redact_secrets(text) == text


def test_does_not_redact_secret_used_as_an_ordinary_word():
    text = "she has a secret admirer at the office"
    assert redact_secrets(text) == text


def test_does_not_redact_credentials_used_as_an_ordinary_word():
    text = "credentials matter in this job"
    assert redact_secrets(text) == text


def test_does_not_redact_auth_used_as_an_ordinary_word():
    text = "you need auth before you can enter the building"
    assert redact_secrets(text) == text


def test_does_not_redact_token_used_as_an_ordinary_word():
    text = "her token collection was impressive"
    assert redact_secrets(text) == text


# ─── Regression: unlabelled hex identifiers are not secrets ───────────────
# A bare run of hex digits (a git commit SHA, for instance) has no case
# mixing and no separator, so it must survive the bare path even though it
# is long and alphanumeric.

def test_does_not_redact_an_unlabelled_git_style_hex_sha():
    text = "see commit d68e4d2fc73a1b9e2c3d4e5f6a7b8c9d0e1f2a3 for the fix"
    assert redact_secrets(text) == text


def test_names_no_brand():
    """THE rule: the redactor must not know any vendor by name."""
    import pathlib
    source = pathlib.Path("assistant/core/redact.py").read_text(encoding="utf-8")
    for brand in ("github", "openai", "google", "aws", "slack", "stripe"):
        assert brand not in source.lower()


def test_transcription_log_sites_are_redacted():
    import pathlib
    main_src = pathlib.Path("assistant/main.py").read_text(encoding="utf-8")
    for line in main_src.splitlines():
        if "Transcription (" in line and "logger.info" in line:
            assert "redact_secrets(" in line, f"unredacted log site: {line.strip()}"

    intent_src = pathlib.Path("assistant/intent.py").read_text(encoding="utf-8")
    for line in intent_src.splitlines():
        if "Classifying:" in line and "logger.info" in line:
            assert "redact_secrets(" in line, f"unredacted log site: {line.strip()}"
