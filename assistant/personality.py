"""
personality.py — Personality trait state + event-driven bumps.

Top half: Thin facade over storage/repos/personality.py (was personality_state.py).
Bottom half: Event-driven trait bumps via regex (was personality_events.py).
"""

import logging
import re
from datetime import datetime
from typing import Optional

from .storage.db import get_db, init_db
from .storage.repos.personality import (
    PersonalityRepo,
    TRAIT_DEFAULTS, MAX_DELTA_PER_CYCLE, MAX_DELTA_PER_EVENT,
)
from . import config

logger = logging.getLogger("personality")

# ─── Repo Facade ────────────────────────────────────────────────────────────

_repo: PersonalityRepo | None = None


def _get_repo() -> PersonalityRepo:
    global _repo
    if _repo is None:
        db = get_db()
        if db is None:
            raise RuntimeError(
                "personality not initialized — call init_personality_db() first"
            )
        from .personalities import get_active_personality_id, PersonalityLoader
        pid = get_active_personality_id()
        defaults = PersonalityLoader(pid).get_trait_defaults()
        _repo = PersonalityRepo(db, personality_id=pid, trait_defaults=defaults)
    return _repo


def init_personality_db() -> None:
    """Initialize the shared database if needed, then bind the repo.

    Detects pre-P1 installs (existing trait rows → keep tsundere) vs
    fresh installs (no rows → warm_honest default).
    """
    global _repo
    db = get_db()
    if db is None:
        db_path = config.SANDBOX_DIR / "memory" / "tenka.db"
        db = init_db(db_path)

    from .personalities import PersonalityLoader, set_active_personality

    active = db.fetchone("SELECT value FROM metadata WHERE key = 'active_personality'")

    if active is None:
        # First run after P1 migration — detect if this is an existing or new user
        row = db.fetchone(
            "SELECT COUNT(*) AS cnt FROM personality_state WHERE personality_id = 'tsundere'"
        )
        has_existing_rows = row and row["cnt"] > 0
        if has_existing_rows:
            default_id = "tsundere"
        else:
            default_id = PersonalityLoader.DEFAULT  # "warm_honest"

        from datetime import datetime
        now = datetime.now().isoformat()
        db.execute(
            "INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
            ("active_personality", default_id, now),
        )
        db.commit()
        personality_id = default_id
        logger.info(f"[PERSONALITY] First P1 startup — active personality set to '{personality_id}'")
    else:
        personality_id = active["value"]

    set_active_personality(personality_id)
    defaults = PersonalityLoader(personality_id).get_trait_defaults()
    _repo = PersonalityRepo(db, personality_id=personality_id, trait_defaults=defaults)

    _migrate_wrongly_seeded_traits(db)
    _tune_warm_honest_tiers(db)


def reload_for_personality(personality_id: str) -> None:
    """Re-bind repo to a different personality_id (called on personality switch)."""
    global _repo
    db = get_db()
    if db is None:
        return
    from .personalities import PersonalityLoader
    defaults = PersonalityLoader(personality_id).get_trait_defaults()
    _repo = PersonalityRepo(db, personality_id=personality_id, trait_defaults=defaults)


def get_current_traits() -> dict[str, float]:
    return _get_repo().get_current_traits()


def get_full_trait_info() -> dict[str, dict[str, float]]:
    return _get_repo().get_full_trait_info()


def get_trait_history(days: int = 30) -> list[dict]:
    return _get_repo().get_trait_history(days)


def update_traits(deltas: dict[str, float], reason: str,
                  trigger: str = "reflection_cycle") -> dict[str, float]:
    return _get_repo().update_traits(deltas, reason, trigger)


def reset_traits() -> None:
    _get_repo().reset_traits()


def get_conversation_count() -> int:
    return _get_repo().get_conversation_count()


def increment_conversation_count() -> int:
    return _get_repo().increment_conversation_count()


def reset_conversation_count() -> None:
    _get_repo().reset_conversation_count()


def get_metadata(key: str) -> Optional[str]:
    return _get_repo().get_metadata(key)


def set_metadata(key: str, value: str) -> None:
    _get_repo().set_metadata(key, value)


def switch_personality(name: str) -> str:
    """Switch to a different personality base. Called from /set personality."""
    from .personalities import PersonalityLoader, set_active_personality

    if name not in PersonalityLoader.BUILTIN:
        return f"Unknown personality '{name}'. Options: {', '.join(PersonalityLoader.BUILTIN)}"

    set_active_personality(name)
    reload_for_personality(name)

    db = get_db()
    if db is not None:
        now = datetime.now().isoformat()
        db.execute(
            "INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            ("active_personality", name, now),
        )
        db.commit()

    reset_conversation_count()

    display_name = name.replace("_", "-")
    return f"Switched to {display_name} personality."


# ─── One-Time Migration ────────────────────────────────────────────────────


def _migrate_wrongly_seeded_traits(db) -> None:
    """Re-seed non-tsundere personalities that were wrongly seeded with TRAIT_DEFAULTS."""
    flag = db.fetchone("SELECT value FROM metadata WHERE key = 'trait_seeding_v2'")
    if flag is not None:
        return

    from .personalities import PersonalityLoader

    for pid in PersonalityLoader.BUILTIN:
        if pid == "tsundere":
            continue
        rows = db.fetchall(
            "SELECT trait, value FROM personality_state WHERE personality_id = ?",
            (pid,),
        )
        if not rows:
            continue

        current = {row["trait"]: row["value"] for row in rows}
        wrongly_seeded = all(
            abs(current.get(trait, -1) - vals["initial"]) < 0.001
            for trait, vals in TRAIT_DEFAULTS.items()
        )
        if wrongly_seeded:
            correct = PersonalityLoader(pid).get_trait_defaults()
            now = datetime.now().isoformat()
            for trait, vals in correct.items():
                db.execute(
                    "UPDATE personality_state SET value = ?, updated_at = ? "
                    "WHERE personality_id = ? AND trait = ?",
                    (vals["initial"], now, pid, trait),
                )
            db.commit()
            logger.info(f"[PERSONALITY] Migrated wrongly-seeded traits for '{pid}'")

    now = datetime.now().isoformat()
    db.execute(
        "INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        ("trait_seeding_v2", "done", now),
    )
    db.commit()


def _tune_warm_honest_tiers(db) -> None:
    """Bump warm_honest warmth into 'high' tier, drop sass into 'low' tier."""
    flag = db.fetchone("SELECT value FROM metadata WHERE key = 'wh_tuning_v1'")
    if flag is not None:
        return

    row = db.fetchone(
        "SELECT value FROM personality_state "
        "WHERE personality_id = 'warm_honest' AND trait = 'warmth'"
    )
    if row and abs(row["value"] - 0.65) < 0.001:
        now = datetime.now().isoformat()
        db.execute(
            "UPDATE personality_state SET value = 0.70, updated_at = ? "
            "WHERE personality_id = 'warm_honest' AND trait = 'warmth'",
            (now,),
        )
        db.execute(
            "UPDATE personality_state SET value = 0.30, updated_at = ? "
            "WHERE personality_id = 'warm_honest' AND trait = 'sass'",
            (now,),
        )
        db.commit()
        logger.info("[PERSONALITY] Tuned warm_honest: warmth→high, sass→low")

    now = datetime.now().isoformat()
    db.execute(
        "INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        ("wh_tuning_v1", "done", now),
    )
    db.commit()


# ─── Event-Driven Trait Bumps ────────────────────────────────────────────────
#
# Detects specific interaction patterns via regex and applies immediate
# micro-adjustments to personality traits. No LLM calls — pure pattern matching.

_NAME_RE = re.escape(config.ASSISTANT_NAME_LOWER)

# ─── Rate Limiting ───────────────────────────────────────────────────────────

_MAX_BUMPS_PER_SESSION = 3
_bumps_this_session: int = 0
_events_fired_this_session: set[str] = set()

_consecutive_small_talk: int = 0
_BANTER_THRESHOLD = 5


# ─── Pattern Definitions ────────────────────────────────────────────────────

_GREETING_PATTERNS = re.compile(
    r"\b(good\s*morning|goodnight|good\s*night|good\s*evening|"
    rf"hey\s+{_NAME_RE}|hi\s+{_NAME_RE}|hello\s+{_NAME_RE}|yo\s+{_NAME_RE}|"
    rf"morning\s+{_NAME_RE}|night\s+{_NAME_RE})\b",
    re.IGNORECASE,
)

_CHECK_ON_PATTERNS = re.compile(
    r"\b(how\s+are\s+you|how\s+do\s+you\s+feel|you\s+okay|you\s+alright|"
    r"are\s+you\s+okay|are\s+you\s+alright|how\s+have\s+you\s+been|"
    rf"how\'s\s+it\s+going\s+{_NAME_RE}|you\s+doing\s+okay)\b",
    re.IGNORECASE,
)

_FRUSTRATION_PATTERNS = re.compile(
    r"\b(never\s*mind|forget\s+it|nvm|ugh+|forget\s+about\s+it|"
    r"this\s+is\s+stupid|doesn\'t\s+work|not\s+working|"
    r"what\s+the\s+hell|are\s+you\s+dumb|you\'re\s+useless)\b",
    re.IGNORECASE,
)

_COMPLIMENT_PATTERNS = re.compile(
    r"\b(you\'re\s+awesome|you\'re\s+amazing|you\'re\s+the\s+best|"
    rf"good\s+job|nice\s+work|well\s+done|thanks?\s+{_NAME_RE}|thank\s+you\s+{_NAME_RE}|"
    r"i\s+love\s+you|you\'re\s+great|you\'re\s+cool|you\'re\s+smart|"
    r"i\s+appreciate\s+you|that\s+was\s+perfect|you\s+rock)\b",
    re.IGNORECASE,
)


# ─── Events Public API ──────────────────────────────────────────────────────


def process_turn(
    transcription: str,
    intent: str,
    facts_extracted: bool = False,
) -> None:
    """
    Analyze a completed conversation turn for event-driven trait bumps.
    Call from main.py after save_turn() and fact extraction.
    """
    global _consecutive_small_talk

    if intent in ("small_talk", "unknown"):
        _consecutive_small_talk += 1
    else:
        _consecutive_small_talk = 0

    if _bumps_this_session >= _MAX_BUMPS_PER_SESSION:
        return

    lowered = transcription.lower().strip()

    if _check_greeting(lowered):
        return
    if _check_on_assistant(lowered):
        return
    if _check_personal_sharing(facts_extracted):
        return
    if _check_frustration(lowered):
        return
    if _check_compliment(lowered):
        return
    if _check_banter():
        return


def check_absence() -> None:
    """
    Check if the user has been absent for 48+ hours.
    Call from the reflection loop in proactive.py.
    """
    try:
        from . import memory

        recent = memory.get_recent(1)
        if not recent:
            return

        last_turn = recent[0]
        last_time = datetime.fromisoformat(last_turn["timestamp"])
        hours_elapsed = (datetime.now() - last_time).total_seconds() / 3600

        if hours_elapsed >= 48:
            _apply_bump(
                "absence",
                {"warmth": -0.01},
                f"User absent for {int(hours_elapsed)} hours",
            )
    except Exception as e:
        logger.debug(f"[PERSONALITY_EVENT] Absence check failed: {e}")


# ─── Event Detectors ────────────────────────────────────────────────────────


def _check_greeting(lowered: str) -> bool:
    if _GREETING_PATTERNS.search(lowered):
        return _apply_bump(
            "greeting",
            {"warmth": 0.01},
            f"User greeted {config.ASSISTANT_NAME_DISPLAY} (social bonding ritual)",
        )
    return False


def _check_on_assistant(lowered: str) -> bool:
    if _CHECK_ON_PATTERNS.search(lowered):
        return _apply_bump(
            "check_on",
            {"openness": 0.01},
            f"User asked how {config.ASSISTANT_NAME_DISPLAY} is doing (cares about her state)",
        )
    return False


def _check_personal_sharing(facts_extracted: bool) -> bool:
    if facts_extracted:
        return _apply_bump(
            "personal",
            {"trust": 0.02},
            "User shared personal information (vulnerability = trust)",
        )
    return False


def _check_frustration(lowered: str) -> bool:
    if _FRUSTRATION_PATTERNS.search(lowered):
        return _apply_bump(
            "frustration",
            {"patience": 0.01},
            f"User expressed frustration ({config.ASSISTANT_NAME_DISPLAY} adapts with more patience)",
        )
    return False


def _check_compliment(lowered: str) -> bool:
    if _COMPLIMENT_PATTERNS.search(lowered):
        return _apply_bump(
            "compliment",
            {"warmth": 0.01, "trust": 0.01},
            f"User complimented {config.ASSISTANT_NAME_DISPLAY} (positive feedback loop)",
        )
    return False


def _check_banter() -> bool:
    if _consecutive_small_talk >= _BANTER_THRESHOLD:
        return _apply_bump(
            "banter",
            {"sass": 0.01, "playfulness": 0.01},
            f"Extended banter ({_consecutive_small_talk} consecutive small_talk turns)",
        )
    return False


# ─── Bump Applicator ────────────────────────────────────────────────────────


def _apply_bump(
    event_type: str,
    deltas: dict[str, float],
    reason: str,
) -> bool:
    """
    Apply a trait bump if the event hasn't fired this session and we're
    under the rate limit. Returns True if bump was applied.
    """
    global _bumps_this_session

    if event_type in _events_fired_this_session:
        return False

    if _bumps_this_session >= _MAX_BUMPS_PER_SESSION:
        return False

    try:
        changed = update_traits(deltas, reason, trigger="event")

        if changed:
            _bumps_this_session += 1
            _events_fired_this_session.add(event_type)
            logger.info(
                f"[PERSONALITY_EVENT] {event_type} → {changed} "
                f"(bump {_bumps_this_session}/{_MAX_BUMPS_PER_SESSION})"
            )
            return True

    except Exception as e:
        logger.debug(f"[PERSONALITY_EVENT] Failed to apply {event_type} bump: {e}")

    return False
