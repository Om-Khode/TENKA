"""
policy.py — Policy enforcement layer for the Voice Assistant.

Mirrors the C# PolicyEngine.cs:
  - Whitelist of allowed intents (everything else is denied)
  - Blacklist of dangerous patterns in text and parameters
  - File path sandboxing (all file ops restricted to SANDBOX_DIR)
  - URL scheme validation (http/https only)
"""

import os
import re
import logging
from dataclasses import dataclass

from . import config
from .intent import IntentResult

logger = logging.getLogger("policy")

# Compile DANGEROUS_PATTERNS once — they are regexes (with word boundaries on
# bare command names) so they don't substring-match inside benign words like
# "form with" matching "rm ".
_DANGEROUS_REGEXES = [
    (pat, re.compile(pat, re.IGNORECASE))
    for pat in config.DANGEROUS_PATTERNS
]


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""
    allowed: bool
    reason: str
    safe_response: str = ""

    @staticmethod
    def allow() -> "PolicyResult":
        return PolicyResult(allowed=True, reason="Intent is whitelisted")

    @staticmethod
    def deny(reason: str) -> "PolicyResult":
        return PolicyResult(
            allowed=False,
            reason=reason,
            safe_response="I'm sorry, I can't do that for safety reasons.",
        )


def evaluate(intent_result: IntentResult) -> PolicyResult:
    """
    Check whether an intent should be allowed to execute.

    Args:
        intent_result: The detected intent with parameters.

    Returns:
        A PolicyResult indicating whether execution is allowed.
    """
    if intent_result is None:
        logger.warning("Null intent received")
        return PolicyResult.deny("Null intent")

    intent = intent_result.intent

    # ── Check whitelist ─────────────────────────────────────────────
    if intent not in config.ALLOWED_INTENTS:
        logger.warning(f"DENIED: Intent '{intent}' not in whitelist")
        return PolicyResult.deny(f"Intent '{intent}' is not allowed")

    # ── Check for dangerous patterns in intent name ─────────────────
    intent_lower = intent.lower()
    for pattern, regex in _DANGEROUS_REGEXES:
        if regex.search(intent_lower):
            logger.warning(f"DENIED: Dangerous pattern '{pattern}' in intent")
            return PolicyResult.deny(f"Dangerous pattern detected: {pattern}")

    # ── Check for dangerous patterns in parameters ──────────────────
    for key, value in intent_result.params.items():
        value_str = str(value)
        for pattern, regex in _DANGEROUS_REGEXES:
            if regex.search(value_str):
                logger.warning(
                    f"DENIED: Dangerous pattern '{pattern}' in parameter '{key}'"
                )
                return PolicyResult.deny(
                    f"Dangerous pattern detected in parameter: {pattern}"
                )

    # ── Path validation for file-related intents ────────────────────
    if intent in ("create_note", "file_task"):
        file_path = (
            intent_result.get_param("filename")
            or intent_result.get_param("title")
            or ""
        )
        if file_path and not _validate_path(file_path):
            logger.warning(f"DENIED: Path '{file_path}' outside sandbox")
            return PolicyResult.deny("File path is outside the allowed directory")

    # ── URL validation for browser intents ──────────────────────────
    if intent == "open_browser":
        url = intent_result.get_param("url", "")
        if url and not _validate_url(url):
            logger.warning(f"DENIED: URL '{url}' uses disallowed scheme")
            return PolicyResult.deny("Only http:// and https:// URLs are allowed")

    logger.info(f"ALLOWED: {intent}")
    return PolicyResult.allow()


def _validate_path(path: str) -> bool:
    """
    Check that a file path stays within the sandbox directory.
    Simple filenames (no slashes, no ..) are always allowed.
    """
    # Simple filename with no directory traversal
    if "/" not in path and "\\" not in path and ".." not in path:
        return True

    # Resolve the full path and check it's within sandbox
    try:
        sandbox = str(config.SANDBOX_DIR.resolve())
        full_path = str((config.SANDBOX_DIR / path).resolve())
        return full_path.startswith(sandbox)
    except Exception:
        return False


def _validate_url(url: str) -> bool:
    """Check that a URL uses only http or https schemes."""
    url_lower = url.lower().strip()
    return url_lower.startswith("http://") or url_lower.startswith("https://")
