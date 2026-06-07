"""
storage/repos/preference.py — Adaptive preference repo.

Manages user preferences with confidence scoring, decay, and audit logging.
Takes a Database instance; all SQL goes through self._db.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("preferences")

# --- Confidence Thresholds ---

CONFIDENCE_SILENT = 0.7       # at or above → applied silently
CONFIDENCE_ASK = 0.4          # at or above → applied but mentioned
CONFIDENCE_IGNORE = 0.3       # below this → preference is ignored

# --- Confidence Deltas ---

CONFIDENCE_FIRST_OBSERVATION = 0.4    # reflection discovers a new pattern
CONFIDENCE_REOBSERVED = 0.15          # same pattern seen again in reflection
CONFIDENCE_USER_CONFIRMS = 0.9        # user says "yes, use that"
CONFIDENCE_USER_CORRECTS = 0.85       # user provides a correction (new pref)
CONFIDENCE_APPLIED_NO_COMPLAINT = 0.05  # preference used, user didn't object
CONFIDENCE_APPLIED_OVERRIDDEN = -0.2    # preference used, user overrode it

# --- Decay ---

DECAY_THRESHOLD_DAYS = 30
DECAY_AMOUNT = 0.05
MIN_CONFIDENCE_BEFORE_PRUNE = 0.15    # never decay below this (don't delete)


class PreferenceRepo:
    def __init__(self, db) -> None:
        self._db = db

    # --- Read Operations ---

    def get_preference(self, key: str, category: Optional[str] = None) -> Optional[dict]:
        """
        Get a single preference by key.
        Optional category filter for disambiguation.
        Returns dict with all preference fields, or None if not found.
        """
        if category:
            row = self._db.fetchone(
                "SELECT * FROM user_preferences WHERE key = ? AND category = ?",
                (key, category),
            )
        else:
            row = self._db.fetchone(
                "SELECT * FROM user_preferences WHERE key = ?",
                (key,),
            )
        return dict(row) if row else None

    def get_preferences_by_category(self, category: str) -> list[dict]:
        """Get all preferences in a category, ordered by confidence descending."""
        rows = self._db.fetchall(
            "SELECT * FROM user_preferences WHERE category = ? ORDER BY confidence DESC",
            (category,),
        )
        return [dict(row) for row in rows]

    def get_active_preferences(self, min_confidence: float = CONFIDENCE_SILENT) -> list[dict]:
        """Get all preferences at or above the given confidence threshold."""
        rows = self._db.fetchall(
            "SELECT * FROM user_preferences WHERE confidence >= ? "
            "ORDER BY category, confidence DESC",
            (min_confidence,),
        )
        return [dict(row) for row in rows]

    def get_all_preferences(self) -> list[dict]:
        """Get every stored preference regardless of confidence."""
        rows = self._db.fetchall(
            "SELECT * FROM user_preferences ORDER BY category, key"
        )
        return [dict(row) for row in rows]

    # --- Write Operations ---

    def set_preference(
        self,
        key: str,
        value: str,
        category: str,
        confidence: float,
        source: str,
        reason: str,
    ) -> None:
        """
        Create or update a preference with full logging.
        Clamps confidence to [0.0, 1.0]. Logs old value before overwriting.
        """
        now = datetime.now().isoformat()

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Check for existing preference (for logging)
        existing = self.get_preference(key)
        old_value = existing["value"] if existing else None
        old_confidence = existing["confidence"] if existing else None

        # Upsert the preference
        self._db.execute(
            "INSERT INTO user_preferences "
            "(key, value, category, confidence, source, times_used, times_overridden, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = excluded.value, "
            "category = excluded.category, "
            "confidence = excluded.confidence, "
            "source = excluded.source, "
            "updated_at = excluded.updated_at",
            (key, value, category, confidence, source, now, now),
        )

        # Log the change
        self._db.execute(
            "INSERT INTO preference_log "
            "(timestamp, key, old_value, new_value, old_confidence, new_confidence, source, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, key, old_value, value, old_confidence, confidence, source, reason),
        )

        self._db.commit()

        if existing:
            logger.info(
                f"[PREFERENCES] Updated {key}: '{old_value}' → '{value}' "
                f"(confidence {old_confidence:.2f} → {confidence:.2f}, {source})"
            )
        else:
            logger.info(
                f"[PREFERENCES] New preference {key}='{value}' "
                f"(category={category}, confidence={confidence:.2f}, {source})"
            )

    def bump_confidence(self, key: str, delta: float = 0.1) -> Optional[float]:
        """
        Increase a preference's confidence score.
        Capped at 1.0. Skips if no actual change.
        Returns new confidence value, or None if preference doesn't exist.
        """
        now = datetime.now().isoformat()

        existing = self.get_preference(key)
        if not existing:
            logger.debug(f"[PREFERENCES] bump_confidence: key '{key}' not found")
            return None

        old_confidence = existing["confidence"]
        new_confidence = min(1.0, old_confidence + delta)

        # Skip if no change
        if round(new_confidence, 4) == round(old_confidence, 4):
            return old_confidence

        self._db.execute(
            "UPDATE user_preferences SET confidence = ?, updated_at = ? WHERE key = ?",
            (new_confidence, now, key),
        )

        self._db.execute(
            "INSERT INTO preference_log "
            "(timestamp, key, old_value, new_value, old_confidence, new_confidence, source, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, key, existing["value"], existing["value"],
             old_confidence, new_confidence, "confidence_bump",
             f"Confidence bumped by {delta:+.2f}"),
        )

        self._db.commit()

        logger.info(
            f"[PREFERENCES] {key} confidence: {old_confidence:.2f} → {new_confidence:.2f} "
            f"(Δ{delta:+.2f})"
        )
        return new_confidence

    def decay_preference(self, key: str, delta: float = DECAY_AMOUNT) -> Optional[float]:
        """
        Decrease a preference's confidence score due to non-use.
        Never decays below MIN_CONFIDENCE_BEFORE_PRUNE (0.15).
        Returns new confidence value, or None if preference doesn't exist.
        """
        now = datetime.now().isoformat()

        existing = self.get_preference(key)
        if not existing:
            return None

        old_confidence = existing["confidence"]
        new_confidence = max(MIN_CONFIDENCE_BEFORE_PRUNE, old_confidence - delta)

        # Skip if no change
        if round(new_confidence, 4) == round(old_confidence, 4):
            return old_confidence

        self._db.execute(
            "UPDATE user_preferences SET confidence = ?, updated_at = ? WHERE key = ?",
            (new_confidence, now, key),
        )

        self._db.execute(
            "INSERT INTO preference_log "
            "(timestamp, key, old_value, new_value, old_confidence, new_confidence, source, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, key, existing["value"], existing["value"],
             old_confidence, new_confidence, "decay",
             f"Unused for {DECAY_THRESHOLD_DAYS}+ days, decayed by {delta:.2f}"),
        )

        self._db.commit()

        logger.info(
            f"[PREFERENCES] {key} decayed: {old_confidence:.2f} → {new_confidence:.2f}"
        )
        return new_confidence

    def record_preference_used(self, key: str) -> None:
        """
        Record that a preference was applied successfully.
        Increments times_used and bumps confidence by +0.05.
        """
        now = datetime.now().isoformat()

        existing = self.get_preference(key)
        if not existing:
            return

        self._db.execute(
            "UPDATE user_preferences SET times_used = times_used + 1, updated_at = ? "
            "WHERE key = ?",
            (now, key),
        )
        self._db.commit()

        # Small confidence bump for successful use
        self.bump_confidence(key, delta=CONFIDENCE_APPLIED_NO_COMPLAINT)

        logger.debug(
            f"[PREFERENCES] {key} used successfully "
            f"(total uses: {existing['times_used'] + 1})"
        )

    def record_preference_overridden(self, key: str) -> None:
        """
        Record that a preference was applied but the user overrode it.
        Increments times_overridden and drops confidence by -0.2.
        """
        now = datetime.now().isoformat()

        existing = self.get_preference(key)
        if not existing:
            return

        self._db.execute(
            "UPDATE user_preferences SET times_overridden = times_overridden + 1, updated_at = ? "
            "WHERE key = ?",
            (now, key),
        )
        self._db.commit()

        # Significant confidence drop
        self.bump_confidence(key, delta=CONFIDENCE_APPLIED_OVERRIDDEN)  # negative delta

        logger.info(
            f"[PREFERENCES] {key} OVERRIDDEN by user "
            f"(total overrides: {existing['times_overridden'] + 1})"
        )

    # --- Context Block for Prompt Injection ---

    def get_preference_context_block(self) -> str:
        """
        Generate a formatted text block of active preferences for prompt injection.
        Only includes preferences at or above CONFIDENCE_ASK (0.4).
        Returns multi-line string or empty string if none qualify.
        """
        rows = self._db.fetchall(
            "SELECT key, value, category, confidence FROM user_preferences "
            "WHERE confidence >= ? ORDER BY category, confidence DESC",
            (CONFIDENCE_ASK,),
        )

        if not rows:
            return ""

        lines = ["--- User Preferences ---"]

        # Group by category for readability
        current_category = None
        for row in rows:
            r = dict(row)
            if r["category"] != current_category:
                current_category = r["category"]
                lines.append(f"  [{current_category}]")

            certainty = "confirmed" if r["confidence"] >= CONFIDENCE_SILENT else "probable"
            lines.append(f"    {r['key']}: {r['value']} ({certainty})")

        return "\n".join(lines)

    # --- Preference History ---

    def get_preference_history(self, days: int = 30) -> list[dict]:
        """
        Return recent preference change log entries.
        Ordered newest first.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self._db.fetchall(
            "SELECT timestamp, key, old_value, new_value, old_confidence, "
            "new_confidence, source, reason "
            "FROM preference_log WHERE timestamp >= ? ORDER BY id DESC",
            (cutoff,),
        )
        return [dict(row) for row in rows]

    # --- Decay Engine ---

    def decay_unused_preferences(self) -> int:
        """
        Reduce confidence of preferences not used in DECAY_THRESHOLD_DAYS.
        Never deletes — just decays to MIN_CONFIDENCE_BEFORE_PRUNE.
        Returns number of preferences that were decayed.
        """
        all_prefs = self.get_all_preferences()
        now = datetime.now()
        decayed_count = 0

        for pref in all_prefs:
            # Skip preferences already at minimum
            if pref["confidence"] <= MIN_CONFIDENCE_BEFORE_PRUNE:
                continue

            # Check how long since last update
            try:
                last_updated = datetime.fromisoformat(pref["updated_at"])
            except (ValueError, TypeError):
                continue

            days_since = (now - last_updated).days
            if days_since >= DECAY_THRESHOLD_DAYS:
                self.decay_preference(pref["key"], DECAY_AMOUNT)
                decayed_count += 1

        if decayed_count:
            logger.info(
                f"[PREFERENCES] Decay cycle complete: {decayed_count} preferences decayed"
            )

        return decayed_count

    # --- Reset (Dev/Debug) ---

    def reset_preferences(self) -> None:
        """Delete all preferences and logs. Dev/debug use only."""
        self._db.execute("DELETE FROM user_preferences")
        self._db.execute("DELETE FROM preference_log")
        self._db.commit()
        logger.info("[PREFERENCES] All preferences and logs cleared (reset)")
