"""Redact secret-shaped substrings before text reaches a log.

Generic by construction: no vendor prefixes, no brand names, no per-service
patterns. Two mechanisms only —

  1. A labelled assignment ("api key is X", "password: X", "Bearer X") keeps
     its label and loses its value.
  2. An unlabelled run of >= 24 secret-alphabet characters that does not look
     like a word loses itself.

Layering: core/ — imports nothing from the assistant.
"""
from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# ─── Labelled secrets ────────────────────────────────────────────────────
# Words that introduce a secret. Generic role nouns, never product names.
_LABELS = (
    "key", "keys", "token", "tokens", "secret", "secrets", "password",
    "passwd", "pwd", "passphrase", "credential", "credentials", "auth",
    "bearer", "apikey", "api_key",
)

_LABEL_ALT = "|".join(_LABELS)

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


def _looks_random(candidate: str) -> bool:
    has_digit = any(c.isdigit() for c in candidate)
    has_alpha = any(c.isalpha() for c in candidate)
    has_case_mix = candidate != candidate.lower() and candidate != candidate.upper()
    has_separator = "_" in candidate or "-" in candidate
    return has_digit and has_alpha and (has_case_mix or has_separator or has_digit)


def redact_secrets(text: str) -> str:
    """Return `text` with secret-shaped substrings replaced by `[REDACTED]`."""
    if not text:
        return text

    def _mask_labelled(match: re.Match[str]) -> str:
        value = match.group(3)
        if value == REDACTED:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}{REDACTED}"

    out = _LABELLED.sub(_mask_labelled, text)

    def _mask_bare(match: re.Match[str]) -> str:
        candidate = match.group(0)
        if not _looks_random(candidate):
            return candidate
        return REDACTED

    return _BARE.sub(_mask_bare, out)
