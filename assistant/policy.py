"""
policy.py — Policy enforcement layer for the Voice Assistant.

What survives here is positive: an intent must be on the whitelist, a file
path must resolve inside the sandbox, a URL must carry a scheme we allow.
Each answers "is this one of the things we permit", which is a question a
caller cannot argue with.

  - Whitelist of allowed intents (everything else is denied)
  - File path sandboxing (all file ops restricted to SANDBOX_DIR)
  - URL scheme validation (http/https only)

**The `DANGEROUS_PATTERNS` deny-list was removed in milestone 6a.5.** It is
worth recording why, because "add a pattern" is a tempting answer to the
next scare and it is the wrong one.

It scanned intent parameters for regexes like `\brm\b`, `\bshell\b`,
`\bformat\b`, `\bkill\b`, `\badmin\b`, `\broot\b`. Three independent
problems, each fatal on its own:

1. **It judged a string that was then thrown away.** `main.py` evaluated
   policy and afterwards overwrote `params["goal"]` with the raw
   transcription for six intents, so the text approved was not the text
   that ran. Any pattern was bypassed by getting the classifier to return a
   short goal.
2. **It was evadable by spelling.** No normalisation anywhere on the path,
   so a Cyrillic `ѕhutdown`, a zero-width space, a soft hyphen, fullwidth
   characters or a trailing digit all read as the word to the model that
   picks the intent and as something else to the regex. `rm/` carried no
   word boundary at all and fired inside `platform/` and `confirm/`, under
   a comment claiming the opposite.
3. **It refused ordinary English.** "format this as a table", "restart the
   music", "kill the timer", "what command should I run", "the root of the
   problem". Three live tests during 6a.5 were eaten by it before reaching
   what they meant to test.

A refusal a user hits daily teaches them to rephrase until something works,
which is the opposite of a control. And a deny-list over text an LLM
influenced is the same class of thing this project already rejected for the
code sandbox: the boundary has to be what a caller *may do*, not which
words they may say. That boundary is `Capability.EXECUTE`, enforced in
`actions/execute()` and at every pre-dispatch branch.
"""

import logging
from dataclasses import dataclass

from . import config
from .intent import IntentResult

logger = logging.getLogger("policy")


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

    # No pattern scan of the intent name or the parameters -- see the module
    # docstring for why it was removed rather than repaired. Note that the
    # name scan only ever "worked" by accident: `shutdown` matches
    # `\bshutdown\b`, so the intent was refused by its own name, and the
    # feature survived solely because `main.py` handles it ~100 lines above
    # this call. Deleting the scan is what makes that intent's real gate --
    # EXECUTE -- the thing actually deciding it.

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
    """Check that a file path stays within the sandbox directory.

    Three bugs an adversarial review found here, all fixed:

    - **`startswith` is not ancestry.** `str(full).startswith(str(sandbox))`
      accepts a *sibling* whose name merely shares the prefix, so
      `../sandbox_evil/loot.txt` escaped a sandbox at `.../sandbox`. Path
      containment is a question about path components, and
      `Path.is_relative_to` is the operation that asks it.
    - **"Simple filename" was defined as "contains no separator",** which on
      Windows admits two shapes that are not simple at all: `C:evil.txt` is
      drive-relative and resolves against that drive's current directory,
      and `note.txt:hidden` names an NTFS alternate data stream. Both are
      refused by name now, before any resolution.
    - **A bare `..` check is redundant once containment is real**, and was
      never sufficient on its own.

    Returns True only for a name that stays inside the sandbox. Anything it
    cannot resolve is False -- an unreadable path is not a permitted one.
    """
    if not path:
        return False

    # Refused by name, ahead of resolution: a colon is either a drive letter
    # or an ADS marker, and neither is a filename this assistant should write.
    # Checked on the whole string rather than the last component, since
    # `sub/C:evil.txt` is the same trick one level down.
    if ":" in path:
        return False

    # NUL and the other C0 controls are not filename characters; Windows
    # silently truncates at NUL, which turns `a.txt\x00.png` into `a.txt`.
    if any(ord(ch) < 32 for ch in path):
        return False

    try:
        sandbox = config.SANDBOX_DIR.resolve()
        full_path = (config.SANDBOX_DIR / path).resolve()
    except (OSError, ValueError):
        return False

    # Component-wise containment, not a string prefix. `is_relative_to`
    # returns True for the sandbox itself, which is correct -- writing *to*
    # the directory is not writing outside it.
    return full_path.is_relative_to(sandbox)


def _validate_url(url: str) -> bool:
    """Check that a URL uses only http or https schemes."""
    url_lower = url.lower().strip()
    return url_lower.startswith("http://") or url_lower.startswith("https://")
