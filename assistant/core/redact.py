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
_LABEL_ALT = "|".join(_STRONG_LABELS + _WEAK_LABELS)

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

# ─── Unlabelled high-entropy tokens ──────────────────────────────────────
# A bare run of secret-alphabet characters, long enough that prose does not
# reach it, containing at least one digit and one letter so that ordinary long
# words are left alone.
_BARE = re.compile(r"(?<![\w-])[A-Za-z0-9_\-]{24,}(?![\w-])")

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
# Three constraints keep it from eating ordinary source code:
#   * the identifier must be UPPER_SNAKE, so `x = 1`, `count = compute()`
#     and `self.total = 0` never match;
#   * it must start the line (leading whitespace allowed, because YAML and
#     INI indent their keys), so `print(x = 1)` and every mid-line
#     assignment inside prose survive;
#   * a comment line starts with its comment marker, not an identifier.
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
_ASSIGNMENT = re.compile(
    r"(?m)^([ \t]*(?:export[ \t]+)?)([A-Z][A-Z0-9_]*)([ \t]*[:=](?!=)[ \t]*)([^\r\n]*)"
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
        return False
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


def _mask_assignment(match: re.Match[str]) -> str:
    """Keep the lead, the identifier and the separator; drop the value.

    An empty value is left alone: `EMPTY_ON_PURPOSE=` has nothing to hide,
    and a `[REDACTED]` standing for nothing would read as a secret that is
    not there.
    """
    lead, name, separator, value = match.groups()
    if not value.strip():
        return match.group(0)
    return f"{lead}{name}{separator}{REDACTED}"


def redact_secrets(text: str) -> str:
    """Return `text` with secret-shaped substrings replaced by `[REDACTED]`."""
    if not text:
        return text

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
    """
    if not text:
        return text
    return redact_secrets(_ASSIGNMENT.sub(_mask_assignment, text))
