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
