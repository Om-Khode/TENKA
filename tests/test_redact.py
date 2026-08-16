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

def test_strict_still_redacts_an_upper_snake_constant_by_position():
    """A module constant in a source preview still loses its value, because on
    a `.env`/INI/YAML key the value is the payload by position and no shape
    test can tell `hunter2` from a word.

    Narrowed on `fix/6a5-api-review`, and this test used to read
    `MAX_PREVIEW_BYTES = 512_000 -> [REDACTED]`. The review (P2-4) measured
    what that cost: `_is_configuration_value` was short-circuited for every
    UPPER_SNAKE name, so previewing this project's own `config.py` through the
    FILES route came back as *broken source* -- `INTENTS = [...]` lost its
    list, and `TASK_MODEL_MAP = {` lost its opening brace, orphaning the dict
    body and the closing brace under it. Over-redaction that makes a preview
    useless is its own bug, and it had its own failing probe.

    The section's own rule -- "if one of these ever has to change, the fix is a
    narrower *shape* for the identifier, never a shape test on the value" --
    is honoured on the identifier side: UPPER_SNAKE **with** a role noun is
    still the strong tier, and UPPER_SNAKE **without** one is a new,
    lower-evidence tier. Two value shapes are then declined there, and both are
    shapes no credential takes: a bracketed literal, and a bare number. That is
    a much smaller concession than it looks -- see the two tests below.
    """
    assert redact_secrets_strict("RELEASE_TAG = v2-abc123") == \
        "RELEASE_TAG = [REDACTED]"
    assert redact_secrets_strict("ARGS=--foo --bar") == "ARGS=[REDACTED]"
    assert redact_secrets_strict("  CLIENT_ID: abc123") == \
        "  CLIENT_ID: [REDACTED]"


def test_strict_leaves_a_numeric_constant_alone():
    """The narrowing, half one. A number is not a credential, and this
    identifier carries no role noun claiming otherwise."""
    for line in ("MAX_PREVIEW_BYTES = 512_000", "TIMEOUT_SECONDS = 30",
                 "PORT: 8787", "MASK = 0xFF", "RATIO = 1.5"):
        assert redact_secrets_strict(line) == line, line


def test_strict_leaves_a_public_constant_that_holds_a_literal_alone():
    """The narrowing, half two, and the one the review actually measured: a
    preview of this project's own `config.py` must come back as source
    somebody can read, not as source with its brackets orphaned."""
    snippet = ('INTENTS = ["small_talk", "web_search"]\n'
               'MAX_PREVIEW_BYTES = 65536\n'
               'TASK_MODEL_MAP = {\n'
               '    "intent": "flash",\n'
               '}\n')
    assert redact_secrets_strict(snippet) == snippet


def test_strict_still_redacts_a_numeric_secret_under_a_role_noun():
    """The narrowing must not become "numbers are safe". A role noun in the
    identifier puts the line back in the strong tier, where the value's shape
    is not asked about at all -- which is the whole point of that tier."""
    assert redact_secrets_strict("TOKEN: 918273645509") == "TOKEN: [REDACTED]"
    assert redact_secrets_strict("API_KEY = 4815162342") == "API_KEY = [REDACTED]"
    assert redact_secrets_strict("DB_PASS=13579246") == "DB_PASS=[REDACTED]"


def test_strict_leaves_a_private_upper_snake_constant_alone():
    """Only because `_` is not `[A-Z]` -- an accident of the identifier
    pattern, not a policy. Worth pinning: most of this codebase's own
    constants are underscore-prefixed, so most of its source previews are
    unaffected by the line above."""
    text = "_MAX_PREVIEW_BYTES = 512_000"
    assert redact_secrets_strict(text) == text


def test_strict_leaves_an_all_caps_prose_marker_alone():
    """A colon is the one separator English also writes, so under `:` the
    value has to be a single unbroken token to count. A sentence is not one,
    and blanking these lines would make a previewed note useless while
    protecting nothing."""
    for line in ("TODO: buy cable", "WARNING: do not run this", "NOTE: see below",
                 "IMPORTANT: read first"):
        assert redact_secrets_strict(line) == line


def test_strict_still_redacts_a_colon_separated_secret():
    """The relaxation above must not open the shape that motivated the rule:
    a config value under a colon is one token, so it still goes -- including
    the short unlabelled one no entropy test would ever catch."""
    assert redact_secrets_strict("API_KEY: sk-abc123def456") == "API_KEY: [REDACTED]"
    assert redact_secrets_strict("DB_PASS: hunter2") == "DB_PASS: [REDACTED]"
    assert redact_secrets_strict("TOKEN: hunter2") == "TOKEN: [REDACTED]"
    assert redact_secrets_strict("  DATABASE_URL: postgres://u:p@h/db") == \
        "  DATABASE_URL: [REDACTED]"


def test_strict_ignores_whitespace_around_a_colon_value():
    """The single-token test runs on the stripped value. Trailing spaces are
    invisible in a file and must not decide whether a secret is redacted."""
    assert redact_secrets_strict("API_KEY:   sk-abc123   ") == "API_KEY:   [REDACTED]"


def test_strict_holds_the_equals_form_to_no_such_test():
    """`=` is machine syntax prose does not reach for, so a spaced value
    still loses itself there. The asymmetry with `:` is deliberate."""
    assert redact_secrets_strict("ARGS=--foo --bar") == "ARGS=[REDACTED]"


def test_strict_gives_up_a_quoted_multi_word_colon_value():
    """The knowing cost of the prose exemption. Pinned so it is a decision
    and not a surprise: the labelled mechanisms are what still cover the
    cases where the identifier carries a role noun they recognise."""
    text = 'SOME_PHRASE: "two words"'
    assert redact_secrets_strict(text) == text


def test_strict_leaves_a_comparison_at_line_start_alone():
    """A comparison carries no value to hide. `==` is the only comparison
    operator that starts with a separator character, so it is the only one
    that needed excluding; the rest never match in the first place."""
    for line in ('MODE == "prod"', "COUNT >= 3", "LIMIT != 0", "DEPTH <= 9"):
        assert redact_secrets_strict(line) == line


# ─── fix/6a5-api-review: the mechanisms the adversarial pass added ────────
# Each of these closed a verified leak or a verified over-redaction. The
# reviewer's own probes live in tests/test_6a5_api_fixes.py; these pin the
# behaviour at the level of the mechanism, including the cases the probes did
# not name.

def test_strict_redacts_a_credential_in_a_url_and_keeps_the_rest():
    """`scheme://user:password@host` had no rule at all, and none of the
    general ones reach it -- `:`, `@` and `/` fragment every run below the
    bare rule's 24-character floor. Only the password goes, so the preview
    still says which host this is and who connects to it."""
    out = redact_secrets_strict(
        "clone from https://admin:hunter2@git.internal.example.com/repo.git")
    assert "hunter2" not in out, out
    assert "admin" in out and "git.internal.example.com" in out, out

    out = redact_secrets_strict("database_url: postgres://user:p4ssw0rd@host:5432/db")
    assert "p4ssw0rd" not in out, out
    assert "database_url:" in out and "host:5432" in out, out


def test_the_log_path_keeps_a_url_credential():
    """The URL rule is strict-only. A connection string in a traceback is a
    real diagnostic, and the log path's standing contract is that it does not
    get stricter."""
    text = "connect failed: postgres://user:pw@localhost:5432/tenka"
    assert redact_secrets(text) == text


def test_strict_reads_a_camel_case_identifier():
    """`_IDENT_SPLIT` split on `[_-]` only, so `clientSecret` was one token in
    neither part set and `\\bsecret\\b` could not see inside it -- while
    camelCase is what JSON and JavaScript config actually use."""
    out = redact_secrets_strict('const cfg = { clientSecret: "hunter2plain" };')
    assert "hunter2plain" not in out, out
    assert "clientSecret" in out, out


def test_strict_does_not_read_a_camel_case_hump_as_a_role_noun_on_its_own():
    """Control: splitting camelCase must not turn every capital into a label.
    `sortKey` and `nextToken` are weak parts and still need a value that
    looks the part; `monkey` is one word and is not `key` at all."""
    for line in ("sortKey = 3", "nextToken: abc", "monkey = banana"):
        assert redact_secrets_strict(line) == line, line


def test_strict_redacts_a_yaml_sequence_item():
    """`^[ \\t]*` allowed indentation but not a `- ` list marker, and a
    docker-compose `environment:` block writes its keys as list items."""
    out = redact_secrets_strict(
        "environment:\n      - POSTGRES_PASSWORD=hunter2\n")
    assert "hunter2" not in out, out
    assert "- POSTGRES_PASSWORD=" in out, out


def test_strict_redacts_a_walrus_assignment():
    """`(?!=)` was added to protect `==` and took `:=` with it."""
    assert redact_secrets_strict("db_pass := hunter2") == "db_pass := [REDACTED]"


def test_strict_redacts_a_punctuation_rich_password():
    """The bracket exemption treated any bracket as evidence of code, which
    exempted exactly the passwords that are strongest."""
    out = redact_secrets_strict("db_pass=P@ssw(rd!1")
    assert "P@ssw(rd!1" not in out, out


def test_a_multiword_passphrase_loses_all_of_itself():
    """The labelled rule's `\\S+` stopped at the first space, so three quarters
    of a diceware phrase shipped *underneath* a `[REDACTED]` claiming the line
    was handled -- worse than not redacting, because the marker says safe."""
    out = redact_secrets_strict("passphrase: correct horse battery staple")
    for word in ("correct", "horse", "battery", "staple"):
        assert word not in out, out


def test_a_strong_label_in_prose_still_only_takes_one_word():
    """Control for the rule above: the rest-of-line behaviour is scoped to an
    explicit `:` or `=`. Prose does not write `password:` before a sentence it
    wants read, so a sentence keeps its words."""
    out = redact_secrets("set the password to hunter2please and then log in")
    assert "and then log in" in out, out
    assert "hunter2please" not in out, out


def test_a_pgp_private_key_block_is_redacted():
    """The PEM rule required `-----` directly after `KEY`, and every exported
    PGP secret key writes ` BLOCK` there."""
    out = redact_secrets_strict(
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBGa1\n"
        "-----END PGP PRIVATE KEY BLOCK-----")
    assert "lQOYBGa1" not in out, out


def test_an_unterminated_pem_marker_only_eats_a_base64_body():
    """The largest destructive rule in the module. `(.*?)…\\Z` under `(?is)`
    meant a lowercase prose *mention* of the header erased every line after
    it -- in the preview path and the log path both."""
    doc = ("# Setup\n"
           "Paste the -----BEGIN RSA PRIVATE KEY----- header, then the body.\n"
           "\n## Step two\nRun the installer.\n")
    assert redact_secrets_strict(doc) == doc

    clipped = ("-----BEGIN OPENSSH PRIVATE KEY-----\n"
               "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ+xyz/123\n")
    out = redact_secrets_strict(clipped)
    assert "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ" not in out, out


def test_a_uuid_is_an_identifier_not_a_secret():
    """A hyphen counted as the bare rule's entropy signal, so every UUID in a
    fixture, a migration or a log tripped it."""
    out = redact_secrets_strict("run id 550e8400-e29b-41d4-a716-446655440000")
    assert out == "run id 550e8400-e29b-41d4-a716-446655440000", out


def test_a_uuid_under_a_role_noun_is_still_redacted():
    """Control: the exemption is on the *bare* path only. A label still makes
    it a token."""
    out = redact_secrets_strict("api_key: 550e8400-e29b-41d4-a716-446655440000")
    assert "550e8400" not in out, out


def test_strict_redacts_a_numeric_token_under_a_role_noun():
    """`_looks_secret` required both a digit and a letter, so a numeric token
    was exempt from every labelled path."""
    out = redact_secrets_strict("token: 918273645509")
    assert "918273645509" not in out, out


def test_a_short_number_under_a_role_noun_survives():
    """Control for the rule above: the numeric floor is well past every port,
    size, year and small count a configuration file writes."""
    for line in ("token: 42", "retry_key: 2024"):
        assert redact_secrets_strict(line) == line, line


def test_strict_names_no_brand():
    """THE rule, restated for the new mechanism."""
    import pathlib
    source = pathlib.Path("assistant/core/redact.py").read_text(encoding="utf-8")
    for brand in ("github", "openai", "google", "aws", "slack", "stripe"):
        assert brand not in source.lower()
