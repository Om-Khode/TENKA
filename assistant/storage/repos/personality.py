"""
storage/repos/personality.py — Dynamic personality state repo.

Manages evolving personality traits with floor/ceiling bounds, audit logging,
conversation counter, and metadata. Takes a Database instance; all SQL goes
through self._db.

Tables (created by db.py migration):
  - personality_state : current trait values + floor/ceiling bounds
  - personality_log   : full audit trail of every trait change
  - metadata          : key-value store for counters (conversation_count, etc.)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("personality")

# --- Trait Definitions ---
# Each trait: (initial_value, floor, ceiling)
# Initial values match tsundere baseline — high sass, low warmth/trust.

TRAIT_DEFAULTS: dict[str, dict[str, float]] = {
    "trust":       {"initial": 0.30, "floor": 0.10, "ceiling": 0.95},
    "warmth":      {"initial": 0.25, "floor": 0.10, "ceiling": 0.90},
    "sass":        {"initial": 0.75, "floor": 0.30, "ceiling": 0.95},
    "openness":    {"initial": 0.20, "floor": 0.05, "ceiling": 0.85},
    "patience":    {"initial": 0.50, "floor": 0.20, "ceiling": 0.90},
    "playfulness": {"initial": 0.60, "floor": 0.20, "ceiling": 0.95},
}

# --- Clamping Constants ---

MAX_DELTA_PER_CYCLE = 0.05   # reflection cycle cap (±)
MAX_DELTA_PER_EVENT = 0.02   # event bump cap (±)


class PersonalityRepo:
    def __init__(
        self, db, personality_id: str = "tsundere",
        *, trait_defaults: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._db = db
        self._personality_id = personality_id
        self._trait_defaults = trait_defaults
        self._seed_if_empty()

    # --- Seeding ---

    def _seed_if_empty(self) -> None:
        """Seed default trait values and metadata on first run."""
        # Seed default traits if table is empty for this personality
        row = self._db.fetchone(
            "SELECT COUNT(*) AS cnt FROM personality_state WHERE personality_id = ?",
            (self._personality_id,),
        )
        if row["cnt"] == 0:
            self._seed_defaults(overrides=self._trait_defaults)
            logger.info(f"[PERSONALITY] Seeded defaults for '{self._personality_id}'")

        # Seed conversation counter if not present
        counter = self._db.fetchone(
            "SELECT value FROM metadata WHERE key = 'conversation_count'"
        )
        if counter is None:
            now = datetime.now().isoformat()
            self._db.execute(
                "INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
                ("conversation_count", "0", now),
            )
            # Also seed last_reflection_at so the first reflection knows when it started
            self._db.execute(
                "INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
                ("last_reflection_at", now, now),
            )
            self._db.commit()

    def _seed_defaults(self, overrides: dict | None = None) -> None:
        """Insert default trait values into personality_state table."""
        now = datetime.now().isoformat()
        defaults = overrides or TRAIT_DEFAULTS
        for trait, vals in defaults.items():
            self._db.execute(
                "INSERT OR IGNORE INTO personality_state "
                "(personality_id, trait, value, floor_val, ceiling_val, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self._personality_id, trait, vals["initial"], vals["floor"], vals["ceiling"], now),
            )
        self._db.commit()

    def seed_defaults(self, defaults: dict[str, dict[str, float]]) -> None:
        """Seed trait defaults with custom values (used when switching to a new personality)."""
        self._seed_defaults(overrides=defaults)

    # --- Read Operations ---

    def get_current_traits(self) -> dict[str, float]:
        """
        Return current trait values as a simple dict.

        Returns:
            {"trust": 0.30, "warmth": 0.25, "sass": 0.75, ...}
        """
        rows = self._db.fetchall(
            "SELECT trait, value FROM personality_state WHERE personality_id = ?",
            (self._personality_id,),
        )
        return {row["trait"]: round(row["value"], 4) for row in rows}

    def get_full_trait_info(self) -> dict[str, dict[str, float]]:
        """
        Return full trait info including floor and ceiling.

        Returns:
            {"trust": {"value": 0.30, "floor": 0.10, "ceiling": 0.95}, ...}
        """
        rows = self._db.fetchall(
            "SELECT trait, value, floor_val, ceiling_val FROM personality_state WHERE personality_id = ?",
            (self._personality_id,),
        )
        return {
            row["trait"]: {
                "value": round(row["value"], 4),
                "floor": row["floor_val"],
                "ceiling": row["ceiling_val"],
            }
            for row in rows
        }

    def get_trait_history(self, days: int = 30) -> list[dict]:
        """
        Return recent trait change log entries.

        Args:
            days: How far back to look (default 30 days).

        Returns:
            List of dicts with: timestamp, trait, old_value, new_value,
            delta, reason, trigger. Ordered newest first.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self._db.fetchall(
            "SELECT timestamp, trait, old_value, new_value, delta, reason, trigger "
            "FROM personality_log WHERE personality_id = ? AND timestamp >= ? ORDER BY id DESC",
            (self._personality_id, cutoff),
        )
        return [dict(row) for row in rows]

    # --- Write Operations ---

    def update_traits(
        self,
        deltas: dict[str, float],
        reason: str,
        trigger: str = "reflection_cycle",
    ) -> dict[str, float]:
        """
        Apply trait deltas with clamping and log every change.

        Each delta is clamped to ±MAX_DELTA_PER_CYCLE (for reflection) or
        ±MAX_DELTA_PER_EVENT (for events). The resulting value is then clamped
        to the trait's [floor, ceiling] range.

        Args:
            deltas:  Dict of {trait_name: delta_value}. Traits not in the dict
                     are left unchanged. Unknown trait names are silently ignored.
            reason:  Human-readable reason for the change (logged).
            trigger: One of "reflection_cycle", "event", "manual".

        Returns:
            Dict of updated trait values (only traits that actually changed).
        """
        now = datetime.now().isoformat()

        # Pick the delta cap based on trigger type
        if trigger == "event":
            max_delta = MAX_DELTA_PER_EVENT
        else:
            max_delta = MAX_DELTA_PER_CYCLE

        # Load current state in one query
        rows = self._db.fetchall(
            "SELECT trait, value, floor_val, ceiling_val FROM personality_state WHERE personality_id = ?",
            (self._personality_id,),
        )
        current = {row["trait"]: dict(row) for row in rows}

        changed = {}

        for trait, raw_delta in deltas.items():
            if trait not in current:
                logger.debug(f"[PERSONALITY] Ignoring unknown trait '{trait}'")
                continue

            info = current[trait]
            old_value = info["value"]

            # Clamp delta magnitude
            clamped_delta = max(-max_delta, min(max_delta, raw_delta))

            # Skip zero deltas
            if clamped_delta == 0.0:
                continue

            # Apply delta and clamp to floor/ceiling
            new_value = old_value + clamped_delta
            new_value = max(info["floor_val"], min(info["ceiling_val"], new_value))
            new_value = round(new_value, 4)

            # Skip if no actual change (e.g. already at floor/ceiling)
            if new_value == round(old_value, 4):
                continue

            actual_delta = round(new_value - old_value, 4)

            # Update state
            self._db.execute(
                "UPDATE personality_state SET value = ?, updated_at = ? "
                "WHERE personality_id = ? AND trait = ?",
                (new_value, now, self._personality_id, trait),
            )

            # Log the change
            self._db.execute(
                "INSERT INTO personality_log "
                "(timestamp, trait, old_value, new_value, delta, reason, trigger, personality_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now, trait, round(old_value, 4), new_value, actual_delta, reason, trigger, self._personality_id),
            )

            changed[trait] = new_value
            logger.info(
                f"[PERSONALITY] {trait}: {old_value:.4f} → {new_value:.4f} "
                f"(Δ{actual_delta:+.4f}, {trigger})"
            )

        if changed:
            self._db.commit()
        else:
            logger.debug(
                "[PERSONALITY] No trait changes applied "
                "(all deltas zero or at bounds)"
            )

        return changed

    def reset_traits(self) -> None:
        """
        Reset all traits to their initial default values.
        Logs the reset as a 'manual' trigger. Dev/debug use only.
        """
        from assistant.personalities import PersonalityLoader
        defaults = PersonalityLoader(self._personality_id).get_trait_defaults()

        now = datetime.now().isoformat()

        rows = self._db.fetchall(
            "SELECT trait, value FROM personality_state WHERE personality_id = ?",
            (self._personality_id,),
        )
        old_values = {row["trait"]: row["value"] for row in rows}

        for trait, vals in defaults.items():
            old = old_values.get(trait, vals["initial"])
            new = vals["initial"]

            self._db.execute(
                "UPDATE personality_state SET value = ?, updated_at = ? "
                "WHERE personality_id = ? AND trait = ?",
                (new, now, self._personality_id, trait),
            )

            if round(old, 4) != round(new, 4):
                self._db.execute(
                    "INSERT INTO personality_log "
                    "(timestamp, trait, old_value, new_value, delta, reason, trigger, personality_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (now, trait, round(old, 4), round(new, 4),
                     round(new - old, 4), "Manual reset to defaults", "manual", self._personality_id),
                )

        self._db.commit()
        logger.info(f"[PERSONALITY] All traits reset to defaults for '{self._personality_id}'")

    # --- Conversation Counter ---

    def get_conversation_count(self) -> int:
        """Return the current conversation count since last reflection."""
        row = self._db.fetchone(
            "SELECT value FROM metadata WHERE key = 'conversation_count'"
        )
        return int(row["value"]) if row else 0

    def increment_conversation_count(self) -> int:
        """
        Increment the conversation counter by 1. Returns the new count.
        Called from main.py after each completed pipeline turn.
        """
        now = datetime.now().isoformat()
        current = self.get_conversation_count()
        new_count = current + 1

        self._db.execute(
            "UPDATE metadata SET value = ?, updated_at = ? "
            "WHERE key = 'conversation_count'",
            (str(new_count), now),
        )
        self._db.commit()
        return new_count

    def reset_conversation_count(self) -> None:
        """Reset counter to 0. Called after a reflection cycle runs."""
        now = datetime.now().isoformat()
        self._db.execute(
            "UPDATE metadata SET value = ?, updated_at = ? "
            "WHERE key = 'conversation_count'",
            ("0", now),
        )
        self._db.commit()

    # --- Metadata Helpers ---

    def get_metadata(self, key: str) -> Optional[str]:
        """Read a metadata value by key. Returns None if not found."""
        row = self._db.fetchone(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        )
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Upsert a metadata value."""
        now = datetime.now().isoformat()
        self._db.execute(
            "INSERT INTO metadata (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = excluded.value, updated_at = excluded.updated_at",
            (key, value, now),
        )
        self._db.commit()
