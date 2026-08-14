"""Secret redaction — generic, brand-agnostic, applied before logging."""
from assistant.core.redact import redact_secrets, redact_secrets_strict

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


# ─── The log path must not get stricter ───────────────────────────────────
# `redact_secrets` is what main.py, intent.py and the API's audit log call on
# every turn. The assignment rule below is deliberately blunt, so it stays
# opt-in: these four assertions are the contract that adding it changed
# nothing for the log callers. If one of them starts failing, a log line
# somewhere just lost content it used to carry.

def test_the_log_path_leaves_an_assignment_line_alone():
    assert redact_secrets("DB_PASS=hunter2") == "DB_PASS=hunter2"


def test_the_log_path_leaves_a_connection_string_alone():
    text = "DATABASE_URL=postgres://user:pw@localhost:5432/tenka"
    assert redact_secrets(text) == text


def test_the_log_path_leaves_an_upper_snake_constant_alone():
    text = "MAX_PREVIEW_BYTES = 512000"
    assert redact_secrets(text) == text


def test_the_log_path_still_redacts_what_it_always_did():
    """Spot-check of each existing mechanism, in one string, unchanged."""
    out = redact_secrets("password is letmein, token ghp_0Xk2LmQ9zRt4Ws7Yc1Vb3Nd6Fh8")
    assert "letmein" not in out
    assert "ghp_0Xk2LmQ9zRt4Ws7Yc1Vb3Nd6Fh8" not in out
    assert out.count(REDACTED) == 2


# ─── Strict mode: assignment-shaped lines ─────────────────────────────────
# For text that is *disclosed* rather than logged — a file preview served
# over the API. On an assignment-shaped line the value is the payload and
# the label is the only part worth keeping, whatever the value looks like:
# "hunter2" carries no entropy signal and no role noun, and is still a
# password. Shape tests cannot save this case, so strict mode stops asking.

def test_strict_redacts_an_assignment_value_of_any_shape():
    assert redact_secrets_strict("DB_PASS=hunter2") == "DB_PASS=[REDACTED]"


def test_strict_redacts_a_labelled_assignment_value():
    assert redact_secrets_strict("API_KEY=sk-abc123def456") == "API_KEY=[REDACTED]"


def test_strict_redacts_a_connection_string_the_log_path_would_keep():
    text = "DATABASE_URL=postgres://user:pw@localhost:5432/tenka"
    assert redact_secrets(text) == text          # the log path sees nothing
    assert redact_secrets_strict(text) == "DATABASE_URL=[REDACTED]"


def test_strict_keeps_the_separator_and_the_spacing_verbatim():
    assert redact_secrets_strict("SMTP_PASS = hunter2") == "SMTP_PASS = [REDACTED]"
    assert redact_secrets_strict("SMTP_PASS: hunter2") == "SMTP_PASS: [REDACTED]"


def test_strict_matches_an_indented_key_because_yaml_and_ini_indent():
    assert redact_secrets_strict("  CLIENT_ID: abc123") == "  CLIENT_ID: [REDACTED]"


def test_strict_keeps_an_export_prefix():
    out = redact_secrets_strict("export SESSION_KEY=abc123")
    assert out == "export SESSION_KEY=[REDACTED]"


def test_strict_leaves_a_comment_line_alone():
    text = "# a comment line, DB_PASS is set below"
    assert redact_secrets_strict(text) == text


def test_strict_leaves_an_ordinary_lowercase_assignment_alone():
    for line in ("x = 1", "count = compute()", "self.total = 0", "path: str = ''"):
        assert redact_secrets_strict(line) == line, line


def test_strict_leaves_a_mid_line_assignment_alone():
    """Line-start anchoring is what saves `print(x = 1)` and its kin."""
    for line in ("print(x = 1)", "call(TIMEOUT=30)", "the FOO=bar setting"):
        assert redact_secrets_strict(line) == line, line


def test_strict_redacts_every_value_in_an_env_file_and_keeps_the_shape():
    preview = (
        "# service credentials\n"
        "DB_PASS=hunter2\n"
        "API_KEY=sk-abc123def456\n"
        "\n"
        "EMPTY_ON_PURPOSE=\n"
        "DATABASE_URL=postgres://user:pw@localhost:5432/tenka\n"
    )
    out = redact_secrets_strict(preview)
    for leaked in ("hunter2", "sk-abc123def456", "postgres://"):
        assert leaked not in out, leaked
    for kept in ("# service credentials", "DB_PASS=", "API_KEY=", "DATABASE_URL="):
        assert kept in out, kept
    # An empty value has nothing to hide, so the line is left as it is
    # rather than gaining a [REDACTED] that stands for nothing.
    assert "EMPTY_ON_PURPOSE=\n" in out
    assert out.count("\n") == preview.count("\n")


def test_strict_also_applies_the_labelled_and_bare_mechanisms():
    """Strict is additive: a preview gets every mechanism, not just the new
    one. Neither of these lines is assignment-shaped."""
    out = redact_secrets_strict("my password is letmein\nuse ghp_0Xk2LmQ9zRt4Ws7Yc1Vb3Nd6Fh8Jp5\n")
    assert "letmein" not in out
    assert "ghp_0Xk2LmQ9zRt4Ws7Yc1Vb3Nd6Fh8Jp5" not in out


def test_strict_is_idempotent():
    once = redact_secrets_strict("DB_PASS=hunter2\nmy api key is sk-abc123def456ghi\n")
    assert redact_secrets_strict(once) == once


def test_strict_handles_empty_input():
    assert redact_secrets_strict("") == ""


def test_strict_preserves_crlf_line_endings():
    assert redact_secrets_strict("DB_PASS=hunter2\r\nB=2\r\n").count("\r\n") == 2


# ─── Strict mode: the accepted costs, pinned ──────────────────────────────
# These are not bugs waiting to be fixed, they are the price of the rule, and
# they are asserted so that nobody "repairs" them by making the rule ask what
# the value looks like again -- which is exactly the question that cannot tell
# "hunter2" from a word. Over-redacting a previewed constant costs a reader
# one line of a file they can open locally; under-redacting costs a
# credential over a transport. If one of these ever has to change, the fix is
# a narrower *shape* for the identifier, never a shape test on the value.

def test_strict_also_redacts_a_public_upper_snake_constant():
    """A module constant in a source preview loses its value too."""
    assert redact_secrets_strict("MAX_PREVIEW_BYTES = 512_000") == \
        "MAX_PREVIEW_BYTES = [REDACTED]"


def test_strict_leaves_a_private_upper_snake_constant_alone():
    """Only because `_` is not `[A-Z]` -- an accident of the identifier
    pattern, not a policy. Worth pinning: most of this codebase's own
    constants are underscore-prefixed, so most of its source previews are
    unaffected by the line above."""
    text = "_MAX_PREVIEW_BYTES = 512_000"
    assert redact_secrets_strict(text) == text


def test_strict_also_redacts_an_all_caps_prose_marker():
    """"TODO: buy cable" in a previewed .md is assignment-shaped by this
    rule's definition. Accepted: same trade, same direction."""
    assert redact_secrets_strict("TODO: buy cable") == "TODO: [REDACTED]"


def test_strict_leaves_a_comparison_at_line_start_alone():
    """A comparison carries no value to hide. `==` is the only comparison
    operator that starts with a separator character, so it is the only one
    that needed excluding; the rest never match in the first place."""
    for line in ('MODE == "prod"', "COUNT >= 3", "LIMIT != 0", "DEPTH <= 9"):
        assert redact_secrets_strict(line) == line


def test_strict_names_no_brand():
    """THE rule, restated for the new mechanism."""
    import pathlib
    source = pathlib.Path("assistant/core/redact.py").read_text(encoding="utf-8")
    for brand in ("github", "openai", "google", "aws", "slack", "stripe"):
        assert brand not in source.lower()
