"""
preferences.py — User preference storage + explicit correction learning.

Top half: Thin facade over storage/repos/preference.py (was preference_store.py).
Bottom half: Regex-based correction detection (was preference_corrections.py).
"""

import logging
import re
from typing import Optional

from .storage.db import get_db, init_db
from .storage.repos.preference import (
    PreferenceRepo,
    CONFIDENCE_SILENT, CONFIDENCE_ASK, CONFIDENCE_IGNORE,
    CONFIDENCE_FIRST_OBSERVATION, CONFIDENCE_REOBSERVED,
    CONFIDENCE_USER_CONFIRMS, CONFIDENCE_USER_CORRECTS,
    CONFIDENCE_APPLIED_NO_COMPLAINT, CONFIDENCE_APPLIED_OVERRIDDEN,
    DECAY_THRESHOLD_DAYS, DECAY_AMOUNT, MIN_CONFIDENCE_BEFORE_PRUNE,
)
from .core.known_apps import get_category, resolve_app
from . import config

logger = logging.getLogger("preferences")

# ─── Repo Facade ────────────────────────────────────────────────────────────

_repo: PreferenceRepo | None = None


def _get_repo() -> PreferenceRepo:
    global _repo
    if _repo is None:
        db = get_db()
        if db is None:
            raise RuntimeError(
                "preferences not initialized — call init_preference_db() first"
            )
        _repo = PreferenceRepo(db)
    return _repo


def init_preference_db() -> None:
    """Initialize the shared database if needed, then bind the repo."""
    global _repo
    db = get_db()
    if db is None:
        db_path = config.SANDBOX_DIR / "memory" / "tenka.db"
        db = init_db(db_path)
    _repo = PreferenceRepo(db)


def get_preference(key: str, category: Optional[str] = None) -> Optional[dict]:
    return _get_repo().get_preference(key, category)


def get_preferences_by_category(category: str) -> list[dict]:
    return _get_repo().get_preferences_by_category(category)


def get_active_preferences(min_confidence: float = CONFIDENCE_SILENT) -> list[dict]:
    return _get_repo().get_active_preferences(min_confidence)


def get_all_preferences() -> list[dict]:
    return _get_repo().get_all_preferences()


def set_preference(key: str, value: str, category: str,
                   confidence: float, source: str, reason: str) -> None:
    _get_repo().set_preference(key, value, category, confidence, source, reason)


def bump_confidence(key: str, delta: float = 0.1) -> Optional[float]:
    return _get_repo().bump_confidence(key, delta)


def decay_preference(key: str, delta: float = DECAY_AMOUNT) -> Optional[float]:
    return _get_repo().decay_preference(key, delta)


def record_preference_used(key: str) -> None:
    _get_repo().record_preference_used(key)


def record_preference_overridden(key: str) -> None:
    _get_repo().record_preference_overridden(key)


def get_preference_context_block() -> str:
    return _get_repo().get_preference_context_block()


def get_preference_history(days: int = 30) -> list[dict]:
    return _get_repo().get_preference_history(days)


def decay_unused_preferences() -> int:
    return _get_repo().decay_unused_preferences()


def reset_preferences() -> None:
    _get_repo().reset_preferences()


# ─── Explicit Correction Learning ────────────────────────────────────────────
#
# Detects when the user explicitly corrects behavior or states a preference,
# and immediately writes it to the preference store — no waiting for the
# next reflection cycle. Regex-first, no LLM calls.

# ─── Correction Patterns ────────────────────────────────────────────────────

_CORRECTION_PATTERNS = [
    re.compile(
        r"(?:no|nah|nope|nuh\s*uh),?\s*(?:use|try|open|play\s+(?:on|with|in))\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"not\s+\w+,?\s*(?:use|try|open|play\s+(?:on|with|in))\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:use|try|open|play\s+(?:on|with|in))\s+(.+?)\s+instead",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i\s+prefer|always\s+use|default\s+to)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:switch|change|move)\s+(?:to|over\s+to)\s+(.+)",
        re.IGNORECASE,
    ),
]

_STYLE_PATTERNS = [
    re.compile(
        r"(?:keep\s+it|be\s+more|make\s+it)\s+(short|brief|concise)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:keep\s+it|be\s+more|make\s+it)\s+(detailed|verbose|longer|in[- ]?depth)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i\s+(?:like|want|prefer))\s+(short|brief|concise)\s+(?:answers?|responses?|replies?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i\s+(?:like|want|prefer))\s+(detailed|long|verbose)\s+(?:answers?|responses?|replies?)",
        re.IGNORECASE,
    ),
]

_ENVIRONMENT_PATTERNS = [
    re.compile(
        r"my\s+project\s+(?:is\s+(?:at|in)|folder\s+is|path\s+is|directory\s+is)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"my\s+downloads?\s+(?:are\s+in|folder\s+is|is\s+(?:at|in))\s+(.+)",
        re.IGNORECASE,
    ),
]

_CONTACT_ROUTING_PATTERNS = [
    re.compile(
        r"(?:use|always\s+use)\s+(\w+)\s+for\s+(\w+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:message|text|contact)\s+(\w+)\s+(?:on|via|through)\s+(\w+)\s+always",
        re.IGNORECASE,
    ),
]


# ─── Corrections Public API ─────────────────────────────────────────────────


def check_for_corrections(
    transcription: str,
    last_intent: str = "",
    last_params: Optional[dict] = None,
) -> bool:
    """
    Check if the user's transcription contains an explicit preference
    correction or instruction. If detected, immediately writes to the
    preference store.

    Call from main.py after each conversation turn.
    """
    if not transcription or len(transcription.strip()) < 5:
        return False

    text = transcription.strip()

    if _check_contact_routing(text):
        return True
    if _check_environment(text):
        return True
    if _check_style(text):
        return True
    if _check_app_correction(text, last_intent, last_params):
        return True

    return False


# ─── Detection Handlers ─────────────────────────────────────────────────────


def _check_app_correction(text: str, last_intent: str, last_params: Optional[dict]) -> bool:
    for pattern in _CORRECTION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        raw_value = match.group(1).strip().rstrip(".,!?")
        normalized = _normalize_app_name(raw_value)

        if not normalized:
            continue

        pref_key = get_category(normalized)
        if not pref_key:
            pref_key = _infer_key_from_context(normalized, last_intent, last_params)

        if not pref_key:
            logger.debug(
                f"Correction detected but can't infer key for: '{raw_value}'"
            )
            continue

        category = _infer_category(pref_key)
        _apply_correction(pref_key, normalized, category, text)
        return True

    return False


def _check_style(text: str) -> bool:
    for pattern in _STYLE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        raw_value = match.group(1).strip().lower()

        if raw_value in ("short", "brief", "concise"):
            _apply_correction("verbosity", "brief", "response_style", text)
            return True
        elif raw_value in ("detailed", "verbose", "longer", "in-depth", "indepth", "in depth"):
            _apply_correction("verbosity", "detailed", "response_style", text)
            return True

    return False


def _check_environment(text: str) -> bool:
    for pattern in _ENVIRONMENT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        path_value = match.group(1).strip().rstrip(".,!?\"'")

        if not path_value or len(path_value) < 3:
            continue

        if "download" in pattern.pattern:
            key = "downloads_folder"
        else:
            key = "project_path"

        _apply_correction(key, path_value, "environment", text)
        return True

    return False


def _check_contact_routing(text: str) -> bool:
    for pattern in _CONTACT_ROUTING_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        groups = match.groups()
        if len(groups) != 2:
            continue

        if "for" in pattern.pattern:
            app_raw, contact_raw = groups
        else:
            contact_raw, app_raw = groups

        app = _normalize_app_name(app_raw.strip())
        contact = contact_raw.strip().lower()

        if not app or not contact:
            continue

        if contact in ("me", "my", "the", "a", "it", "that", "this", "everyone"):
            continue

        key = f"contact_{contact}_app"
        _apply_correction(key, app, "contact_routing", text)
        return True

    return False


# ─── Helpers ────────────────────────────────────────────────────────────────


_AUX_PREFIX_RE = re.compile(
    r"^(?:using|use|with|on|in|the|my)\s+", re.IGNORECASE,
)


def _normalize_app_name(raw: str) -> str:
    lowered = raw.lower().strip()

    lowered = _AUX_PREFIX_RE.sub("", lowered).strip()
    if not lowered:
        return ""

    result = resolve_app(lowered)
    if result:
        return result[0]

    for suffix in (" app", " application", " client"):
        if lowered.endswith(suffix):
            trimmed = lowered[:-len(suffix)].strip()
            result = resolve_app(trimmed)
            if result:
                return result[0]

    words = lowered.split()
    if 1 <= len(words) <= 3:
        return lowered

    return ""


def _infer_key_from_context(app_name: str, last_intent: str, last_params: Optional[dict]) -> str:
    if not last_intent or not last_params:
        return ""

    goal = (last_params or {}).get("goal", "").lower()

    if any(kw in goal for kw in ("play", "music", "song", "playlist", "lofi", "lo-fi")):
        return "music_app"

    from .core.known_apps import get_apps_by_category as _get_cat
    _msg_names = frozenset(_get_cat("messaging_default"))
    if any(kw in goal for kw in ({"message", "send", "text"} | _msg_names)):
        return "messaging_default"

    if any(kw in goal for kw in ("email", "mail", "inbox", "draft")):
        return "email_app"

    if any(kw in goal for kw in ("open", "browse", "search", "url", "website")):
        return "browser"

    return ""


def _infer_category(key: str) -> str:
    if key.startswith("contact_") and key.endswith("_app"):
        return "contact_routing"
    if key in ("music_app", "messaging_default", "email_app", "browser"):
        return "app_routing"
    if key in ("verbosity", "email_format", "explanation_depth", "tone"):
        return "response_style"
    if key in ("project_path", "downloads_folder"):
        return "environment"
    if key.endswith("_default") or key.endswith("_format"):
        return "task_defaults"
    return "app_routing"


def _apply_correction(key: str, value: str, category: str, user_input: str) -> None:
    try:
        existing = get_preference(key)

        if existing and existing["value"] == value:
            logger.debug(f"Skipping duplicate correction: {key}={value}")
            return

        if existing:
            record_preference_overridden(key)

        set_preference(
            key=key,
            value=value,
            category=category,
            confidence=0.85,
            source="correction",
            reason=f"User explicitly said: '{user_input}'",
        )

        logger.info(
            f"Correction captured: {key}={value} "
            f"(category={category}, confidence=0.85)"
        )

    except Exception as e:
        logger.warning(f"Failed to apply correction: {e}")
