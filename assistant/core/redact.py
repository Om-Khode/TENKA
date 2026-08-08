"""Redact secret-shaped substrings before text reaches a log.

Generic by construction: no vendor prefixes, no brand names, no per-service
patterns. Two mechanisms only —

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
