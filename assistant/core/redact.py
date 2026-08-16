"""Redact secret-shaped substrings before text reaches a log or a reader.

Generic by construction: no vendor prefixes, no brand names, no per-service
patterns. Three mechanisms, the third opt-in —

  1. A labelled assignment ("api key is X", "password: X", "Bearer X") keeps
     its label and loses its value. How much the value itself has to prove
     depends on how trustworthy the label is (see the two label tiers
     below) — a plain word can never distinguish "letmein" (a real
     passphrase) from "admirer" (not one), so for the strongest labels the
     label alone is the evidence.
  2. An unlabelled run of >= 24 secret-alphabet characters loses itself,
     provided it also carries an actual entropy signal (mixed case or a
     separator) — otherwise a plain lowercase-hex identifier such as a git
     commit hash would be mistaken for a secret.
  3. `redact_secrets_strict` only: an assignment-shaped *line* loses its
     value whatever that value looks like. Blunt on purpose, which is why
     it is not in the default path — see its own section below.

Two entry points, and the difference between them is the audience.
`redact_secrets` is for text on its way to a log, where over-redaction
costs a diagnostic. `redact_secrets_strict` is for text on its way to a
reader over a transport, where under-redaction costs a credential.

Layering: core/ — imports nothing from the assistant.
"""
from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# ─── Labelled secrets ────────────────────────────────────────────────────
# Role nouns that introduce a secret, split into two tiers by how reliably
# they mean "a secret follows" versus "ordinary conversation." Generic role
# nouns only — never product or vendor names.
#
# Strong: essentially never followed by a non-secret in practice. The label
# alone is the evidence, because the value that follows one of these carries
# no independent signal — "my password is letmein" and "my password is
# 13579246" are both real leaks, and neither value mixes letters and digits
# or looks statistically unusual. Over-redacting "the password field" is an
# acceptable cost; leaking a passphrase is not.
_STRONG_LABELS = (
    "password", "passwd", "pwd", "passphrase", "bearer", "apikey", "api_key",
)

# Weak: ordinary English words that also happen to be role nouns ("the key
# thing to remember", "she has a secret admirer", "credentials matter in
# this job", "her token collection"). A value following one of these is
# redacted only when it independently looks secret-shaped (see
# `_looks_secret` below) — the label is a hint, not proof.
_WEAK_LABELS = (
    "key", "keys", "token", "tokens", "secret", "secrets", "credential",
    "credentials", "auth",
)

_STRONG_LABEL_SET = frozenset(_STRONG_LABELS)
_WEAK_LABEL_SET = frozenset(_WEAK_LABELS)
_LABEL_ALT = "|".join(_STRONG_LABELS + _WEAK_LABELS)

# ─── Compound identifiers ────────────────────────────────────────────────
# The two lists above are matched as whole words, which is right for prose
# and wrong for configuration: `\b` never fires inside `db_pass`, because
# `_` is a word character. So `client_secret`, `db_password` and
# `access_token` -- the shapes a real `.env`, YAML or JSON file actually
# uses -- match nothing at all, and they are not UPPER_SNAKE either, so the
# assignment rule below misses them too.
#
# The fix is to split an identifier on `_`/`-` and look at its parts. Same
# two tiers, same reasoning, one addition: `pass` counts as a strong part.
# It is deliberately NOT in `_STRONG_LABELS`, because as a bare word English
# writes it constantly ("pass the salt", "a boarding pass") and the labelled
# rule would eat the next word every time. As one component of a
# configuration identifier it has no such second meaning.
_STRONG_IDENT_PARTS = frozenset(_STRONG_LABELS) | {"pass", "secret", "secrets"}
_WEAK_IDENT_PARTS = frozenset(_WEAK_LABELS) - {"secret", "secrets"}

# Split on `_`/`-` **and** on a camelCase hump. The hump alternatives are
# zero-width: `clientSecret` -> `client`, `Secret`, and `HTTPToken` ->
# `HTTP`, `Token`. Without them `clientSecret` is one token that belongs to
# neither part set, and `\bsecret\b` cannot see inside it either -- so a
# JavaScript or JSON config using the spelling JavaScript and JSON actually
# use was invisible to every mechanism in this module.
#
# The split runs on the identifier as written and the *parts* are lowercased
# afterwards; lowercasing first would erase the humps this depends on.
_IDENT_SPLIT = re.compile(
    r"[_\-]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_UPPER_SNAKE = re.compile(r"[A-Z][A-Z0-9_]*\Z")

# Values that are switches rather than credentials. See
# `_is_configuration_value`.
_NON_SECRET_LITERALS = frozenset({"true", "false", "none", "null", "nil", ""})

# Filler between the label and the value: whitespace, and the connective
# words/punctuation that commonly sit between a label and its value ("is",
# "to", ":", "="). Matched as a run so any mix of these is skipped without
# ever reaching into the value itself — word-boundary anchors on "is"/"to"
# stop the filler from eating the front of a value that happens to start
# with those letters (e.g. "isabelle...").
_FILLER = r"(?:\s|is\b|to\b|[:=])*"

# label, filler, then the value up to the next whitespace.
_LABELLED = re.compile(
    rf"(?i)\b({_LABEL_ALT})\b({_FILLER})(\S+)"
)

# ─── A strong label under an explicit separator ──────────────────────────
# `\S+` above stops at the first space, so `passphrase: correct horse battery
# staple` lost the word "correct" and shipped the other three -- *underneath* a
# `[REDACTED]` claiming the job was done. That is worse than not redacting at
# all: the marker tells the reader the line is safe. A diceware or BIP39 phrase
# is the normal shape of a passphrase, not an exotic one.
#
# Multi-word values cannot be recovered by widening `_LABELLED` itself, for two
# reasons. Widening the group to the rest of the line makes the *match* cover
# the rest of the line, so `re.sub` never rescans it and a second label on the
# same line stops being redacted -- `password is letmein, token ghp_...` would
# lose one of its two. And in prose ("the password is fine, thanks") the rest
# of the line is a sentence, not a secret.
#
# So this is a separate, narrower rule: a **strong** label, an explicit `:` or
# `=` -- machine syntax, not the connective words `_FILLER` also tolerates --
# and then everything to the end of the line. Eating the whole tail is correct
# here precisely because the separator says the tail *is* the value. Prose
# keeps `_LABELLED`'s first-token behaviour, unchanged, because prose does not
# write `password:` before a sentence it wants read.
_LABELLED_ASSIGNED = re.compile(
    rf"(?i)\b({'|'.join(_STRONG_LABELS)})\b([ \t]*[:=][ \t]*)([^\r\n]+)"
)

# ─── A credential in a URL ───────────────────────────────────────────────
# `scheme://user:password@host` had no rule anywhere in this module, and none
# of the general ones reach it: `:`, `@` and `/` all break a run, so every
# fragment lands under `_BARE`'s 24-character floor, and the identifier on the
# left of the assignment (`database_url`, `clone from`) carries no role noun.
# A connection string is one of the two or three most common ways a real
# credential sits in a real config file.
#
# Only the password is replaced. The scheme, the user and the host survive, so
# a preview still says which database this is and who connects to it -- the
# same "keep the shape, lose the payload" trade every other rule here makes.
#
# Strict only, deliberately. The log path has a pinned contract that it must
# not get stricter (`test_the_log_path_leaves_a_connection_string_alone`), and
# a connection string in a traceback is a genuine diagnostic. Text on its way
# to a reader over a transport is the case where the credential half costs
# more than the diagnostic half.
_URL_USERINFO = re.compile(
    r"(?i)\b([a-z][a-z0-9+.\-]*://)([^\s:/@]+):([^\s/@]+)@"
)

# ─── Unlabelled high-entropy tokens ──────────────────────────────────────
# A bare run of secret-alphabet characters, long enough that prose does not
# reach it, containing at least one digit and one letter so that ordinary long
# words are left alone.
_BARE = re.compile(r"(?<![\w-])[A-Za-z0-9_\-]{24,}(?![\w-])")

# ─── Private-key blocks ──────────────────────────────────────────────────
# A PEM body is base64, and base64 uses `+` and `/` — neither of which is in
# the bare charset above. So a key body is not one long run to the bare rule,
# it is a handful of shorter ones, and every run that lands under the 24-char
# floor prints in the clear directly beside a `[REDACTED]`. Widening the bare
# charset is the wrong repair: `/` is in every path and URL, and the rule has
# no label to corroborate it.
#
# The framing is the evidence instead. A `-----BEGIN … PRIVATE KEY-----`
# marker says what follows is a key with no shape test required, so the whole
# body goes as one block, before any character-level rule gets to fragment it.
#
# Scoped to `PRIVATE KEY` rather than to PEM framing in general: a
# certificate is the public half and redacting it protects nothing. The
# footer is optional because a truncated preview cuts it off, and a key whose
# end was clipped is still a key.
#
# This one lives in the shared path rather than in strict mode. The usual
# trade -- over-redaction costs a diagnostic -- does not apply, because a
# base64 key body has never been a useful thing to read in a log.
# Two patterns rather than one with an optional footer, and that is the fix
# for the largest destructive rule this module had.
#
# It used to be `(BEGIN…)(.*?)(END…|\Z)` under `(?is)`. With no END marker
# `.*?` expands to the end of the string, and `(?i)` means a lowercase prose
# *mention* triggers it -- so a setup guide that merely names the header, or a
# traceback that quotes it, lost every line that followed. Measured on a
# `# Setup` document: everything after "Paste the -----BEGIN RSA PRIVATE
# KEY----- header" was gone, in the preview path and the log path both.
#
# A terminated block still goes as a block: the footer is the evidence that
# everything between the two markers is body, whatever it looks like.
#
# An unterminated one -- a preview that clipped the footer, and a clipped key
# is still a key -- consumes only what a PEM body actually is: whole lines of
# base64, starting on the line *after* the header. A sentence continuing on
# the header's own line is not that, and neither is a `##` heading two lines
# down, so a document that mentions the marker keeps its text. The 16-character
# floor keeps a one-word line from qualifying.
#
# `[A-Z0-9 ]*` now appears on *both* sides of `PRIVATE KEY`. The old pattern
# required `-----` immediately after `KEY`, so `-----BEGIN PGP PRIVATE KEY
# BLOCK-----` -- the spelling every exported PGP secret key uses -- matched
# nothing and its body printed in the clear.
_PEM_MARKER = r"-----{verb} [A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*-----"
_PEM_PRIVATE_KEY = re.compile(
    "(?is)({begin})(.*?)({end})".format(
        begin=_PEM_MARKER.format(verb="BEGIN"),
        end=_PEM_MARKER.format(verb="END"))
)
_PEM_PRIVATE_KEY_UNTERMINATED = re.compile(
    "(?i)({begin})((?:\\r?\\n[ \\t]*[A-Za-z0-9+/=]{{16,}}[ \\t]*)+)".format(
        begin=_PEM_MARKER.format(verb="BEGIN"))
)

# ─── Quoted labels (strict only) ─────────────────────────────────────────
# `{"api_key": "sk-…"}` — the shape every JSON config file on disk has, and
# the one the labelled rule is blindest to. Its filler class excludes `"`, so
# after the label it captures the two-character `":` as the value, that fails
# the length floor, and the real secret sitting in its own quotes is never
# looked at. A strong label the redactor explicitly trusts ends up protecting
# nothing.
#
# Rather than teach the prose rule about quotes -- which would let it eat the
# word after any quoted role noun in ordinary text -- this is a separate rule
# for the structured shape: a quoted key, a colon, a quoted value. Not
# line-anchored, so minified JSON is covered as well as pretty-printed;
# `json.dumps` without `indent` is not obfuscation and must not be a bypass.
# The quotes and the key survive so the preview still says which values are
# set.
# The key may be unquoted as well as quoted, because the shape this rule is
# for is not only JSON. `const cfg = { clientSecret: "hunter2plain" }` is the
# same key-colon-quoted-value structure with JavaScript's object syntax, and it
# reaches nothing else here: `_ASSIGNMENT` is line-anchored and this line
# starts with `const`, and `\bsecret\b` cannot see inside `clientSecret`. It is
# also the exact shape a `.ts`/`.js` config file previewed over the FILES route
# has.
#
# Widening the key is safe because the key is not what decides anything --
# `_identifier_tier` is, and it refuses `{ name: "Bob" }` and `{ key: "abc" }`
# exactly as it refuses them everywhere else.
_QUOTED_LABELLED = re.compile(
    r'(?:"([A-Za-z][A-Za-z0-9_\-]*)"|\b([A-Za-z][A-Za-z0-9_\-]*))'
    r'(\s*:\s*)"([^"\r\n]*)"'
)

# ─── Hard-wrapped values (strict only) ───────────────────────────────────
# `\S+` stops at a newline, and so does the bare rule, so a secret broken
# across a line boundary loses only its first half; the second sits in
# plaintext on the line below, directly under a `[REDACTED]` that claims the
# job is done.
#
# A continuation is recognised structurally, not guessed at: the line above
# ended in a redaction, and this line is one unbroken run of
# secret-alphabet characters (base64's `+/=` included) that clears a length
# floor and mixes letters with digits. An English sentence has spaces, so it
# is never a continuation -- which matters, because swallowing the line under
# every redaction would shred a previewed note.
_WRAP_MIN_LEN = 16
_WRAPPED_CONTINUATION = re.compile(
    r"(?m)(" + re.escape(REDACTED) + r"[ \t]*\r?\n)([A-Za-z0-9+/=_\-]+)(?=\r?\n|\Z)"
)

# Minimum lengths for the shape checks below. The bare path has no label to
# corroborate it, so it needs a long run before it is even considered (this
# also matches the regex's own {24,}). The weak-label path already has a
# role noun as a hint, so a much shorter value still counts — "api key is
# sk-abc..." must redact even though the value alone would never trip the
# bare path. The strong-label path trusts the label alone, so its floor is
# only a sanity check against degenerate matches (stray punctuation), not a
# secret-shape requirement.
_BARE_MIN_LEN = 24
_WEAK_LABEL_MIN_LEN = 8
_STRONG_LABEL_MIN_LEN = 3

# How long an all-digit value must be, under a role noun, to count as secret
# rather than as a number. Ten digits is past every port, size, year, HTTP
# status and small count a configuration file writes, and short of nothing a
# numeric credential realistically is.
_NUMERIC_MIN_LEN = 10

# ─── Assignment-shaped lines (strict only) ───────────────────────────────
# The two mechanisms above both ask "does this value look like a secret?".
# On an assignment-shaped line that question is the wrong one: the value is
# the payload by position, so `DB_PASS=hunter2` must lose "hunter2" even
# though "hunter2" has no entropy signal, and `DATABASE_URL=postgres://...`
# must lose a URL that no shape test would ever flag. So this rule stops
# asking. It keeps the identifier and the separator verbatim — a preview
# whose keys survive still tells the reader which values are set, which is
# the whole point of previewing a config file — and replaces only what sits
# to the right of the separator.
#
# Four constraints keep it from eating ordinary source code and prose:
#   * the identifier must be UPPER_SNAKE *or* carry a role noun as one of
#     its `_`/`-` separated parts, so `x = 1`, `count = compute()` and
#     `self.total = 0` never match while `db_pass=` and `client_secret:` do
#     — see `_identifier_tier`, which is where that second alternative is
#     decided, and which is the whole of the lowercase-snake_case fix;
#   * it must start the line (leading whitespace allowed, because YAML and
#     INI indent their keys), so `print(x = 1)` and every mid-line
#     assignment inside prose survive;
#   * a comment line starts with its comment marker, not an identifier;
#   * under a `:` separator the value must be a single unbroken token —
#     see `_mask_assignment`, which is where the prose markers English
#     writes with a colon ("TODO: buy cable") are let through.
#
# The accepted cost is a *public* UPPER_SNAKE constant in a source preview:
# `MAX_PREVIEW_BYTES = 512_000` loses its value. That is the same trade the
# strong-label tier already makes — over-redacting a constant is cheap,
# disclosing a credential is not — and it is why this rule stays out of the
# log path, where a redacted constant is a lost diagnostic instead.
#
# `[^\r\n]*` rather than `.*$`: it stops the value group short of a CRLF's
# carriage return, so a Windows file's line endings come back unchanged.
#
# `(?!=)` after the separator keeps a comparison from being read as an
# assignment: `MODE == "prod"` would otherwise keep its leading `=` and lose
# `= "prod"`, mangling a line that never carried a value. Every other
# comparison operator is excluded by construction — `!`, `<` and `>` are not
# in `[:=]`, so `COUNT >= 3` never starts a match at all — and `==` was the
# one shape that slipped through, because `=` leads it.
#
# The identifier group accepts any identifier, not only UPPER_SNAKE; which of
# them actually carries a value worth hiding is decided in
# `_identifier_tier`. Keeping that judgement in Python rather than in the
# pattern is what lets the two tiers -- redact on the label alone, versus
# redact only a secret-shaped value -- stay the same two tiers the labelled
# rule already uses, instead of becoming a second, differently-argued policy.
# `(?:-[ \t]+)?` in the lead is the YAML sequence marker. `^[ \t]*` allowed
# spaces and tabs before the identifier -- which is indentation, and covers a
# YAML *mapping* -- but a docker-compose `environment:` block writes its keys
# as list items:
#
#     environment:
#       - POSTGRES_PASSWORD=hunter2
#
# and that `- ` made the line unrecognisable. It is one of the two or three
# most common places a real password sits in a real repository.
#
# The separator alternates `:=` ahead of the single-character class. `(?!=)`
# was added to stop `MODE == "prod"` being read as an assignment, and it also
# killed `db_pass := hunter2` -- Go's and Pascal's assignment operator, and
# what a `.tf`/`.ini`-adjacent config or a pasted snippet may well use. Trying
# `:=` first matches the operator; falling through to `[:=](?!=)` still refuses
# `==`, and every other comparison (`!=`, `<=`, `>=`) never starts a match
# because `!`, `<` and `>` are not in the class.
_ASSIGNMENT = re.compile(
    r"(?m)^([ \t]*(?:-[ \t]+)?(?:export[ \t]+)?)([A-Za-z][A-Za-z0-9_\-]*)"
    r"([ \t]*(?::=|[:=](?!=))[ \t]*)([^\r\n]*)"
)

# Values that are code rather than configuration: a call, a subscript, or a
# literal collection. This replaced a blanket "contains any bracket" test,
# which exempted `db_pass=P@ssw(rd!1` -- so the punctuation-rich passwords,
# which are the strong ones, were the ones the guard let through. A password is
# not shaped like a call: the shapes below are an identifier (dotted paths
# included) immediately followed by `(` or `[`, or a value that opens with a
# bracket. `P@ssw(rd!1` is neither, because `@` is not part of an identifier.
_CODE_SHAPE = re.compile(r"\A(?:[\[{(]|[A-Za-z_][A-Za-z0-9_.]*[ \t]*[\[(])")

# A numeric literal, in the spellings source and configuration actually use.
# Deliberately not "anything made of hex characters": `abc123` is all hex
# characters and is a plausible short credential, and `CLIENT_ID: abc123` is
# pinned as redacted. Only decimal (with `_` digit separators) and an explicit
# `0x`/`0b`/`0o` prefix count.
_BARE_NUMBER = re.compile(
    r"\A[+-]?(?:0[xX][0-9A-Fa-f_]+|0[bB][01_]+|0[oO][0-7_]+"
    r"|[0-9][0-9_]*(?:\.[0-9_]+)?(?:[eE][+-]?[0-9]+)?)\Z"
)

# The canonical 8-4-4-4-12 identifier. `_looks_secret`'s entropy signal counts
# a hyphen, so every UUID in a fixture, a migration, a log line or a config
# tripped the bare rule -- the same false positive the git-hash carve-out
# exists to prevent, arrived at from the other direction. A UUID is an
# identifier by definition: it is generated to be *shared*, and it carries no
# authority on its own. Exempted on the bare path only -- `api_key:
# 550e8400-...` is a token because the label says so.
_UUID = re.compile(
    r"\A[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}"
    r"-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)


def _looks_secret(candidate: str, *, min_len: int, require_entropy: bool) -> bool:
    """Shape check for the weak-label and bare paths.

    A candidate has to mix letters and digits and clear a minimum length to
    be considered secret-shaped at all — that alone is enough for the
    weak-label path, where the label is only a hint. The bare path has no
    label at all, so it additionally requires an actual entropy signal
    (mixed case or a separator); without that, an ordinary lowercase-hex
    identifier such as a git commit hash would be flagged just for being
    long and alphanumeric.

    Not used by the strong-label path: a real password or bearer token can
    be short, all-digits, or all-lowercase, so no shape test can separate
    it from an ordinary word of the same shape ("letmein" vs. "admirer").
    For those labels the label itself is the evidence — see
    `_is_plausible_value` and `_mask_labelled` below.
    """
    if len(candidate) < min_len:
        return False
    has_digit = any(c.isdigit() for c in candidate)
    has_alpha = any(c.isalpha() for c in candidate)
    if not (has_digit and has_alpha):
        # An all-digit run under a role noun is a secret too. Requiring *both*
        # a digit and a letter exempted `token: 918273645509` outright -- and
        # numeric tokens, PINs and account identifiers are common enough that
        # "it has no letters" is not evidence of anything.
        #
        # Only on the labelled paths (`require_entropy=False`), never on the
        # bare one: a long unlabelled digit string is a timestamp, an id or a
        # phone number far more often than a credential, and there is no label
        # to corroborate it. `_NUMERIC_MIN_LEN` is well above the four or five
        # digits a port, a size or a year takes.
        if require_entropy or not has_digit or has_alpha:
            return False
        return len(candidate) >= _NUMERIC_MIN_LEN
    if not require_entropy:
        return True
    has_case_mix = candidate != candidate.lower() and candidate != candidate.upper()
    has_separator = "_" in candidate or "-" in candidate
    return has_case_mix or has_separator


def _is_plausible_value(candidate: str, *, min_len: int) -> bool:
    """Minimal sanity floor for the strong-label path: long enough to be a
    value at all, and not just stray punctuation the filler failed to eat.
    """
    return len(candidate) >= min_len and any(c.isalnum() for c in candidate)


def _is_configuration_value(stripped: str) -> bool:
    """Does the right-hand side read as configuration rather than as code?

    Only the lowercase-identifier branch asks this, and it is the difference
    between a fix and a new bug. `secret`, `pass`, `auth`, `token` and `key`
    are role nouns in a `.env` file and ordinary variable names in a Python
    file -- `client_secret = text.strip()`, `auth_url = parts[2]`,
    `key = hashlib.sha256(x).digest()` -- and a source preview that blanks
    every one of those lines is useless.

    UPPER_SNAKE needs no such test: the identifier shape is itself the signal
    that this is configuration, which is why that branch keeps redacting
    whatever follows, spaces and all (`ARGS=--foo --bar`). A lowercase name
    carries no such signal, so the *value* has to supply it. Two structural
    marks, neither of them a guess about entropy:

      * a bracket of any kind means a call, a subscript or a literal
        collection -- syntax a config value has no use for;
      * whitespace means an expression, because neither `.env` nor unquoted
        YAML can carry a space in a value without quoting it.

    A bare boolean or null literal is excluded as well. This is not an
    entropy test creeping back in -- it is a three-word exact list, and a
    switch is not a credential. `enabled: true` and `allow_bearer=False` are
    the most common configuration lines there are, and blanking them says
    nothing was there to protect while destroying the one bit that mattered.
    The UPPER_SNAKE branch deliberately does not get this exclusion: its
    behaviour is pinned by tests that argue it line by line, and it is not
    the branch that fires on ordinary source.

    What this knowingly gives up is a quoted config value with spaces under a
    lowercase key -- unless a *strong* role noun names it, which
    `_mask_assignment` now handles separately, because `client_secret:
    "hunter two"` is not a sentence and the label says so.
    """
    return not any(c.isspace() for c in stripped)


def _looks_like_code(stripped: str) -> bool:
    """A call, a subscript, or a literal collection -- syntax, not a value.

    Hoisted out of `_is_configuration_value` and applied to **every** tier,
    including UPPER_SNAKE, which used to skip that function entirely. That
    short-circuit is what destroyed a preview of this project's own
    `config.py`: `INTENTS = [...]` lost its list and `TASK_MODEL_MAP = {` lost
    its opening brace, orphaning the dict body and the closing brace below it.
    A source preview that returns structurally broken source is its own bug.

    Narrower than the "any bracket anywhere" test it replaces -- see
    `_CODE_SHAPE` for why that one exempted exactly the punctuation-rich
    passwords it should have caught.
    """
    return _CODE_SHAPE.match(stripped) is not None


def _is_switch_literal(stripped: str) -> bool:
    """A boolean or a null, in the punctuation a real line wraps it in.

    Quotes and a trailing comma or semicolon are punctuation around the
    literal rather than part of it -- `allow_bearer=False,` inside a
    multi-line call is still a boolean. Applied to every tier now, for the
    same reason `_looks_like_code` is: `DEBUG=true` and `debug=true` are the
    same line and the identifier's casing was never the thing that made the
    value a switch.
    """
    return stripped.strip("\"',;").lower() in _NON_SECRET_LITERALS


def _is_quoted(stripped: str) -> bool:
    """Whether the value is wholly enclosed in one pair of quotes.

    A quoted value is a value by syntax rather than by guess, which is what
    lets the colon rule below admit `client_secret: "hunter two"` without
    also admitting `TODO: buy cable`.
    """
    return (len(stripped) >= 2 and stripped[0] == stripped[-1]
            and stripped[0] in "\"'")


def _identifier_tier(name: str) -> str | None:
    """How much evidence an identifier is, on a line whose value is the payload.

    Four answers now, and the fourth is the fix for the over-redaction half of
    the review:

      * `"strong"` — the name alone is enough; redact whatever follows.
      * `"weak"` — redact only a value that independently looks secret-shaped.
      * `"constant"` — UPPER_SNAKE with **no** role noun in it. Still redacts
        by position, because on a `.env`/INI/YAML key the value *is* the
        payload, but no longer redacts a bare numeric literal.
      * `None` — not a secret-carrying identifier at all.

    Three ways to be strong, in order of how they were arrived at:

      * the whole name is a strong label (`password`, `api_key`, `apiKey`
        after lowercasing);
      * one part is a strong part. This is the snake_case fix: `db_pass`,
        `client_secret`, `db_password` are the shapes real configuration uses,
        and `\\b` cannot see inside them because `_` is a word character.
        `_IDENT_SPLIT` now splits camelCase too, so `clientSecret` lands here
        as well;
      * UPPER_SNAKE **and** a role noun of either tier. `SESSION_KEY` and
        `TOKEN` are configuration keys naming a credential twice over, so the
        weak tier's shape test does not apply to them -- `export
        SESSION_KEY=abc123` and `TOKEN: hunter2` are both pinned as redacted
        and neither value would clear it.

    UPPER_SNAKE **without** a role noun is the `"constant"` tier, and
    separating it out is the whole of the change. It used to be plain
    `"strong"`, and `config.py` is essentially nothing but public UPPER_SNAKE
    constants, so previewing this project's own configuration through the
    FILES route blanked it. The value shapes that tier now declines to redact
    -- a bare number, a bracketed literal -- are shapes no credential takes.
    Everything else it still redacts by position, which is what keeps
    `DATABASE_URL=postgres://…`, `CLIENT_ID: abc123` and `ARGS=--foo --bar`
    behaving exactly as their tests pin them.

    Weak parts (`key`, `token`, `auth`, `credential`) are the same ordinary
    English words the weak-label tier already distrusts, and they are common
    in source code that is not configuration at all -- `sort_key`,
    `primary_key`, `next_token`. So on a *lowercase* name they get the same
    treatment they get in prose: a hint, not proof.
    """
    lowered = name.lower()
    parts = {p.lower() for p in _IDENT_SPLIT.split(name) if p}
    if lowered in _STRONG_LABEL_SET or parts & _STRONG_IDENT_PARTS:
        return "strong"
    has_weak_noun = lowered in _WEAK_LABEL_SET or bool(parts & _WEAK_IDENT_PARTS)
    if _UPPER_SNAKE.match(name):
        return "strong" if has_weak_noun else "constant"
    return "weak" if has_weak_noun else None


def _mask_assignment(match: re.Match[str]) -> str:
    """Keep the lead, the identifier and the separator; drop the value.

    An empty value is left alone: `EMPTY_ON_PURPOSE=` has nothing to hide,
    and a `[REDACTED]` standing for nothing would read as a secret that is
    not there.

    The colon form additionally requires a value with no internal
    whitespace, because `:` is the one separator English also uses: an
    ALL-CAPS prose marker at the start of a previewed note ("TODO: buy
    cable", "WARNING: do not run this") is assignment-shaped by every
    structural test this rule can apply, and blanking those lines makes a
    `.md` preview useless while protecting nothing. A sentence is what tells
    the two apart -- a secret in a config file is a single unbroken token by
    necessity (`sk-abc123`, `postgres://u:p@h/db`), since neither `.env` nor
    unquoted YAML can carry a space without quoting it.

    The `=` form is deliberately not held to the same test: `=` is machine
    syntax that prose does not reach for, so `ARGS=--foo --bar` should still
    lose its value even though it has spaces in it. That asymmetry is the
    point, not an oversight.

    What this gives up, knowingly: a quoted multi-word value under a colon
    (`SOME_PASSPHRASE: "two words"`) survives this rule -- though the
    labelled mechanisms still sweep it afterwards whenever the identifier
    contains a role noun they know, which is most of the cases that matter.
    """
    lead, name, separator, value = match.groups()
    tier = _identifier_tier(name)
    if tier is None:
        return match.group(0)
    stripped = value.strip()
    if not stripped:
        return match.group(0)
    # Already handled, by this rule on an earlier pass or by the quoted rule
    # above it. Bailing keeps both entry points idempotent without depending on
    # `[REDACTED]` happening to fail some other test.
    if REDACTED in stripped:
        return match.group(0)
    # Code and switches are not credentials, on any tier. Both used to be
    # skipped for UPPER_SNAKE, which is what destroyed `INTENTS = [...]`,
    # `TASK_MODEL_MAP = {` and `DEBUG=true` in a source preview.
    if _looks_like_code(stripped) or _is_switch_literal(stripped):
        return match.group(0)
    if ":" in separator and any(c.isspace() for c in stripped):
        # `:` is the one separator English also writes, so an ALL-CAPS prose
        # marker ("TODO: buy cable") is assignment-shaped by every structural
        # test available. A sentence is what tells the two apart.
        #
        # Unless the value is quoted *and* a strong role noun named it:
        # `client_secret: "hunter two"` is a value by syntax and a credential
        # by label, and it was leaking. The exemption is scoped to the strong
        # tier so that `SOME_PHRASE: "two words"` -- pinned as surviving -- is
        # untouched, since `SOME_PHRASE` carries no role noun.
        if not (tier == "strong" and _is_quoted(stripped)):
            return match.group(0)
    # The whitespace half of the configuration-vs-code test stays exclusive to
    # lowercase names, and that asymmetry is deliberate rather than an
    # oversight: `=` is machine syntax prose does not reach for, so
    # `ARGS=--foo --bar` should still lose its value, while `key = 0xAF if
    # command_id == 'volume_up' else 0xAE` is an expression and must not.
    if not _UPPER_SNAKE.match(name) and not _is_configuration_value(stripped):
        return match.group(0)
    if tier == "constant" and _BARE_NUMBER.match(stripped):
        # A number is not a credential, and this tier's identifier carries no
        # role noun saying otherwise. `MAX_PREVIEW_BYTES = 65536` survives;
        # `TOKEN: 918273645509` does not, because that name reaches the strong
        # tier and never arrives here.
        return match.group(0)
    if tier == "weak" and not _looks_secret(
        stripped, min_len=_WEAK_LABEL_MIN_LEN, require_entropy=False
    ):
        return match.group(0)
    return f"{lead}{name}{separator}{REDACTED}"


def _mask_quoted(match: re.Match[str]) -> str:
    """Blank a JSON string value whose key names a secret, keeping the shape.

    Same two tiers as everywhere else, resolved by `_identifier_tier`, so a
    `"password"` key loses a short shapeless passphrase while a `"key"` key
    only loses something that looks the part.
    """
    quoted_name, bare_name, separator, value = match.groups()
    name = quoted_name if quoted_name is not None else bare_name
    if value == REDACTED or value.lower() in _NON_SECRET_LITERALS:
        return match.group(0)
    tier = _identifier_tier(name)
    if tier is None:
        return match.group(0)
    if tier == "constant" and quoted_name is None:
        # An *unquoted* key is source syntax -- a JavaScript object field, a
        # YAML mapping in a `.ts` config -- not a `.env` line, so the
        # "the value is the payload by position" argument that carries the
        # constant tier on an assignment line does not apply. Only a real role
        # noun makes an unquoted key evidence here, which is what keeps
        # `SOME_PHRASE: "two words"` intact while `clientSecret: "…"` goes.
        return match.group(0)
    if tier == "weak":
        keep = not _looks_secret(
            value, min_len=_WEAK_LABEL_MIN_LEN, require_entropy=False
        )
    else:
        # `strong` and `constant` alike: inside a quoted structure the key is
        # a field name, not a module constant, so there is no public-constant
        # cost to weigh here the way there is on an assignment line.
        keep = not _is_plausible_value(value, min_len=_STRONG_LABEL_MIN_LEN)
    if keep:
        return match.group(0)
    # The key comes back the way it went in: quoting it when it was not quoted
    # would rewrite the JavaScript this rule now also reads.
    rendered = f'"{name}"' if quoted_name is not None else name
    return f'{rendered}{separator}"{REDACTED}"'


def _mask_wrapped_continuations(text: str) -> str:
    """Extend a redaction over the lines a hard wrap split it across.

    Applied repeatedly rather than once, because a value wrapped over three
    lines needs the second line redacted before the third one can be seen as
    following a redaction. It converges: every pass either replaces a run
    with `[REDACTED]`, which contains characters the continuation charset
    excludes, or changes nothing.
    """

    def _mask(match: re.Match[str]) -> str:
        head, run = match.group(1), match.group(2)
        if not _looks_secret(run, min_len=_WRAP_MIN_LEN, require_entropy=False):
            return match.group(0)
        return f"{head}{REDACTED}"

    while True:
        swept = _WRAPPED_CONTINUATION.sub(_mask, text)
        if swept == text:
            return text
        text = swept


def redact_secrets(text: str) -> str:
    """Return `text` with secret-shaped substrings replaced by `[REDACTED]`."""
    if not text:
        return text

    # Before anything character-level: a private-key block goes as a block,
    # or the bare rule fragments its base64 on `+`/`/` and prints the pieces
    # that fall under the length floor. Terminated first, so that a block with
    # a footer is consumed as a block and the unterminated pattern -- which
    # only ever eats whole base64 lines -- has nothing left to see.
    text = _PEM_PRIVATE_KEY.sub(
        lambda m: f"{m.group(1)}\n{REDACTED}\n{m.group(3)}", text
    )
    text = _PEM_PRIVATE_KEY_UNTERMINATED.sub(
        lambda m: f"{m.group(1)}\n{REDACTED}", text
    )

    def _mask_labelled_assigned(match: re.Match[str]) -> str:
        label, separator, tail = match.groups()
        value = tail.rstrip()
        if not value or value == REDACTED:
            return match.group(0)
        if not _is_plausible_value(value, min_len=_STRONG_LABEL_MIN_LEN):
            return match.group(0)
        # The trailing whitespace the value did not include is put back, so a
        # file's own spacing survives a preview.
        return f"{label}{separator}{REDACTED}{tail[len(value):]}"

    text = _LABELLED_ASSIGNED.sub(_mask_labelled_assigned, text)

    def _mask_labelled(match: re.Match[str]) -> str:
        label, _filler, value = match.group(1), match.group(2), match.group(3)
        if value == REDACTED:
            return match.group(0)
        if label.lower() in _STRONG_LABEL_SET:
            is_secret = _is_plausible_value(value, min_len=_STRONG_LABEL_MIN_LEN)
        else:
            is_secret = _looks_secret(
                value, min_len=_WEAK_LABEL_MIN_LEN, require_entropy=False
            )
        if not is_secret:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}{REDACTED}"

    out = _LABELLED.sub(_mask_labelled, text)

    def _mask_bare(match: re.Match[str]) -> str:
        candidate = match.group(0)
        if _UUID.match(candidate):
            # An identifier, not a credential. Without this every UUID trips
            # the bare rule, because the hyphens read as the entropy signal
            # this path requires -- see `_UUID`.
            return candidate
        if not _looks_secret(candidate, min_len=_BARE_MIN_LEN, require_entropy=True):
            return candidate
        return REDACTED

    return _BARE.sub(_mask_bare, out)


def redact_secrets_strict(text: str) -> str:
    """Every mechanism at once, for text disclosed to a reader.

    Additive, never a replacement: the assignment rule runs first so a
    config file's values are gone before the labelled and bare rules sweep
    whatever prose is left around them. Callers that log must keep using
    `redact_secrets` — this variant would eat an UPPER_SNAKE constant out of
    a traceback, a cost only worth paying when the text is leaving the
    machine.

    Callers are responsible for one thing this function cannot judge: what
    the text *is*. Base64 in a `data:` URI is one long high-entropy run, so
    the bare rule would shred an image preview into `[REDACTED]`. Text of
    that shape must not be passed here at all.

    Order matters and is not arbitrary. The quoted rule runs before the
    assignment rule so a JSON line is handled once, as JSON, keeping its
    quotes -- the assignment rule cannot start on a `"` and so leaves the
    result alone rather than mangling it a second time. The wrap sweep runs
    last of all, because it works from the `[REDACTED]` markers every other
    rule has by then written.
    """
    if not text:
        return text
    out = _QUOTED_LABELLED.sub(_mask_quoted, text)
    out = _ASSIGNMENT.sub(_mask_assignment, out)

    def _mask_url_userinfo(match: re.Match[str]) -> str:
        scheme, user, password = match.groups()
        if password == REDACTED:
            return match.group(0)
        return f"{scheme}{user}:{REDACTED}@"

    # After the assignment rule, which will already have taken the whole value
    # on a line like `DATABASE_URL=postgres://…` -- so this one is what catches
    # a connection string in prose, in a command line, or under a lowercase key
    # with no role noun in it.
    out = _URL_USERINFO.sub(_mask_url_userinfo, out)
    return _mask_wrapped_continuations(redact_secrets(out))
