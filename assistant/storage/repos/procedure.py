"""
storage/repos/procedure.py — Teachable procedure repo.

Maps trigger phrases to ordered step lists. Supports exact match,
filler-tolerant matching, prefix/contained/subsequence trigger
resolution, conflict detection against both procedures and shortcuts.
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger("procedures")

_MAX_STEPS = 20
_WARN_STEPS = 10


class ProcedureRepo:
    def __init__(self, db, *, assistant_name_lower: str) -> None:
        self._db = db
        self._assistant_name_lower = assistant_name_lower

        self._reserved: set[str] = {
            assistant_name_lower,
            "hey", "hi", "hello", "yes", "no", "okay", "ok",
            "stop", "cancel", "help", "please", "thanks", "thank you",
        }

        # Filler words stripped before matching
        fillers = frozenset([
            "please", "now", assistant_name_lower, "go", "do it", "run",
            "hey", "hi", "can you", "could you", "would you", "just",
        ])
        self._filler_re = re.compile(
            r"^(?:(?:"
            + "|".join(re.escape(f) for f in sorted(fillers, key=len, reverse=True))
            + r")\s+)+|(?:\s+(?:"
            + "|".join(re.escape(f) for f in sorted(fillers, key=len, reverse=True))
            + r"))+$",
            re.IGNORECASE,
        )

    # --- CRUD ---

    def create_procedure(
        self,
        trigger: str,
        name: str,
        steps: list[dict],
        backend: str = "auto",
        description: str = "",
    ) -> int:
        """
        Store a new procedure. Returns the new procedure ID.

        Raises ValueError if:
          - trigger is reserved or too short
          - trigger conflicts with an existing procedure trigger
          - trigger conflicts with an existing shortcut trigger
          - steps list is empty or exceeds _MAX_STEPS
        """
        trigger_clean = trigger.strip().lower()

        if not trigger_clean or len(trigger_clean) < 3:
            raise ValueError(f"Trigger '{trigger_clean}' is too short (minimum 3 chars)")

        if trigger_clean in self._reserved:
            raise ValueError(f"Trigger '{trigger_clean}' is a reserved word")

        if not steps:
            raise ValueError("A procedure must have at least one step")

        if len(steps) > _MAX_STEPS:
            raise ValueError(f"Procedure exceeds maximum of {_MAX_STEPS} steps")

        now = datetime.now().isoformat()

        # Check for conflict with existing procedures
        existing = self._db.fetchone(
            "SELECT id FROM user_procedures WHERE trigger = ? AND enabled = 1",
            (trigger_clean,),
        )
        if existing:
            raise ValueError(
                f"A procedure with trigger '{trigger_clean}' already exists "
                f"(id={existing['id']}). Delete or update it first."
            )

        # Check for conflict with shortcuts
        shortcut_conflict = self._db.fetchone(
            "SELECT trigger FROM user_shortcuts WHERE trigger = ?",
            (trigger_clean,),
        )
        if shortcut_conflict:
            raise ValueError(
                f"Trigger '{trigger_clean}' conflicts with an existing shortcut. "
                "The procedure would take priority — delete the shortcut first if intended."
            )

        steps_json = json.dumps(steps, ensure_ascii=False)

        cursor = self._db.execute(
            "INSERT INTO user_procedures "
            "(trigger, name, description, steps, backend, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trigger_clean, name.strip(), description.strip(), steps_json, backend, now, now),
        )
        self._db.commit()

        proc_id = cursor.lastrowid
        logger.info(
            f"[PROCEDURES] Created id={proc_id} trigger='{trigger_clean}' "
            f"steps={len(steps)}"
        )
        return proc_id

    def get_procedure(self, trigger: str) -> Optional[dict]:
        """Lookup a procedure by exact trigger phrase. Case-insensitive."""
        row = self._db.fetchone(
            "SELECT * FROM user_procedures WHERE trigger = ? AND enabled = 1",
            (trigger.strip().lower(),),
        )
        return self._row_to_dict(row) if row else None

    def get_procedure_by_id(self, proc_id: int) -> Optional[dict]:
        """Lookup a procedure by id regardless of enabled state."""
        row = self._db.fetchone(
            "SELECT * FROM user_procedures WHERE id = ?",
            (proc_id,),
        )
        return self._row_to_dict(row) if row else None

    def update_procedure(
        self,
        proc_id: int,
        steps: Optional[list[dict]] = None,
        name: Optional[str] = None,
        trigger: Optional[str] = None,
        description: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> bool:
        """
        Update one or more fields on an existing procedure.
        Only provided (non-None) fields are changed.
        Returns True if found and updated, False if not found.
        """
        now = datetime.now().isoformat()

        sets = ["updated_at = ?"]
        values: list = [now]

        if steps is not None:
            if len(steps) > _MAX_STEPS:
                raise ValueError(f"Procedure exceeds maximum of {_MAX_STEPS} steps")
            sets.append("steps = ?")
            values.append(json.dumps(steps, ensure_ascii=False))

        if name is not None:
            sets.append("name = ?")
            values.append(name.strip())

        if trigger is not None:
            new_trigger = trigger.strip().lower()
            if new_trigger in self._reserved:
                raise ValueError(f"Trigger '{new_trigger}' is a reserved word")
            # Conflict check excluding self
            conflict = self._db.fetchone(
                "SELECT id FROM user_procedures WHERE trigger = ? AND id != ? AND enabled = 1",
                (new_trigger, proc_id),
            )
            if conflict:
                raise ValueError(
                    f"Trigger '{new_trigger}' is already used by procedure "
                    f"id={conflict['id']}"
                )
            sets.append("trigger = ?")
            values.append(new_trigger)

        if description is not None:
            sets.append("description = ?")
            values.append(description.strip())

        if backend is not None:
            sets.append("backend = ?")
            values.append(backend)

        values.append(proc_id)

        cursor = self._db.execute(
            f"UPDATE user_procedures SET {', '.join(sets)} WHERE id = ?",
            tuple(values),
        )
        self._db.commit()

        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"[PROCEDURES] Updated id={proc_id}")
        return updated

    def delete_procedure(self, proc_id: int) -> bool:
        """Soft delete a procedure (sets enabled=0)."""
        now = datetime.now().isoformat()

        cursor = self._db.execute(
            "UPDATE user_procedures SET enabled = 0, updated_at = ? WHERE id = ?",
            (now, proc_id),
        )
        self._db.commit()

        if cursor.rowcount > 0:
            logger.info(f"[PROCEDURES] Soft-deleted id={proc_id}")
            return True
        return False

    def list_procedures(self, enabled_only: bool = True) -> list[dict]:
        """List all procedures, ordered by use_count DESC."""
        query = "SELECT * FROM user_procedures"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY use_count DESC, updated_at DESC"

        rows = self._db.fetchall(query)
        return [self._row_to_dict(r) for r in rows]

    # --- Trigger Matching ---

    def _normalize(self, text: str) -> str:
        """Lowercase, strip, and remove leading/trailing filler words."""
        return self._filler_re.sub("", text.strip().lower()).strip()

    def match_trigger(self, text: str) -> Optional[dict]:
        """
        Match input text against all stored (enabled) procedure triggers.

        Priority:
          1. Exact match (after normalization)
          2. Text starts with the trigger (trigger is a prefix)
          3. Trigger is contained in text as a substring
          4. Trigger words appear in order within input (subsequence, 2+ words)

        Returns the best-matching procedure dict, or None if no match.
        """
        if not text or len(text.strip()) < 3:
            return None

        rows = self._db.fetchall(
            "SELECT * FROM user_procedures WHERE enabled = 1"
        )

        if not rows:
            return None

        normalized_input = self._normalize(text)
        input_words = normalized_input.split()

        # Sort by trigger length descending so longest match wins
        rows = sorted(rows, key=lambda r: len(r["trigger"]), reverse=True)

        exact = None
        prefix = None
        contained = None
        subsequence = None

        for row in rows:
            trigger = row["trigger"].lower()

            if normalized_input == trigger:
                exact = row
                break  # Can't do better than exact

            if normalized_input.startswith(trigger + " ") or normalized_input.startswith(trigger):
                if prefix is None:
                    prefix = row

            if trigger in normalized_input:
                if contained is None:
                    contained = row

            trigger_words = trigger.split()
            if len(trigger_words) >= 2 and self._subsequence_match(trigger_words, input_words) is not None:
                if subsequence is None:
                    subsequence = row

        best = exact or prefix or contained or subsequence
        if best is None:
            return None

        result = self._row_to_dict(best)
        logger.info(
            f"[PROCEDURES] Matched trigger='{best['trigger']}' → id={best['id']} "
            f"name='{best['name']}'"
        )
        return result

    # --- Usage Tracking ---

    def record_usage(self, proc_id: int) -> None:
        """Increment use_count and update last_used timestamp."""
        now = datetime.now().isoformat()
        self._db.execute(
            "UPDATE user_procedures SET use_count = use_count + 1, "
            "last_used = ?, updated_at = ? WHERE id = ?",
            (now, now, proc_id),
        )
        self._db.commit()

    # --- Conflict Check ---

    def check_trigger_conflict(self, trigger: str) -> Optional[str]:
        """
        Check if a trigger phrase conflicts with existing procedures or shortcuts.
        Returns a human-readable conflict message, or None if clean.
        """
        trigger_clean = trigger.strip().lower()

        if trigger_clean in self._reserved:
            return f"'{trigger_clean}' is a reserved word and can't be used as a trigger."

        existing_proc = self._db.fetchone(
            "SELECT name FROM user_procedures WHERE trigger = ? AND enabled = 1",
            (trigger_clean,),
        )
        if existing_proc:
            return (
                f"You already have a procedure called '{existing_proc['name']}' "
                f"triggered by '{trigger_clean}'."
            )

        existing_shortcut = self._db.fetchone(
            "SELECT intent FROM user_shortcuts WHERE trigger = ?",
            (trigger_clean,),
        )
        if existing_shortcut:
            return (
                f"You already have a shortcut triggered by '{trigger_clean}' "
                f"(runs {existing_shortcut['intent']}). "
                "The procedure would take priority over it."
            )

        return None

    # --- Lookup ---

    def find_by_name_or_trigger(
        self, text: str, enabled_only: bool = True
    ) -> Optional[dict]:
        """
        Fuzzy-find a procedure by name or trigger phrase.
        Uses word-based scoring to avoid false substring matches.
        """
        if not text:
            return None

        text_lower = text.strip().lower()
        text_words = set(text_lower.split())

        where = " AND enabled = 1" if enabled_only else ""
        rows = self._db.fetchall(
            f"SELECT * FROM user_procedures WHERE 1=1{where}"
        )

        if not rows:
            return None

        best = None
        best_score = 0

        for row in rows:
            score = self._match_score(
                text_lower, text_words,
                row["trigger"].lower(),
                row["name"].lower(),
            )
            if score > best_score:
                best_score = score
                best = row

        return self._row_to_dict(best) if best and best_score > 0 else None

    # --- Static / Class Helpers ---

    @staticmethod
    def _subsequence_match(
        trigger_words: list[str], input_words: list[str]
    ) -> list[int] | None:
        """
        Check if trigger words appear in order within input words.
        Returns the positions of matched words, or None if no match.
        Requires first trigger word to match first input word.
        """
        if not trigger_words or len(trigger_words) < 2:
            return None
        if not input_words or input_words[0] != trigger_words[0]:
            return None
        ti = 0
        positions = []
        for ii, word in enumerate(input_words):
            if ti < len(trigger_words) and word == trigger_words[ti]:
                positions.append(ii)
                ti += 1
        if ti == len(trigger_words):
            return positions
        return None

    @staticmethod
    def subsequence_remainder(trigger: str, text: str) -> str:
        """Extract the words between trigger words in a subsequence match."""
        trigger_words = trigger.strip().lower().split()
        input_words = text.strip().lower().split()
        positions = ProcedureRepo._subsequence_match(trigger_words, input_words)
        if positions is None:
            return text.strip()
        original_words = text.strip().split()
        remainder = [w for i, w in enumerate(original_words) if i not in set(positions)]
        return " ".join(remainder).strip()

    @staticmethod
    def _match_score(
        text: str, text_words: set, trigger: str, name: str
    ) -> int:
        """Score how well search text matches a procedure's trigger/name."""
        trigger_words = set(trigger.split())
        name_words = set(name.split())

        if text == trigger:
            return 1000
        if text == name:
            return 900

        if text_words and trigger_words:
            if text_words <= trigger_words:
                return 800 + len(text_words) * 10
            if trigger_words <= text_words:
                return 700 + len(trigger_words) * 10

        if text_words and name_words:
            if text_words <= name_words:
                return 600 + len(text_words) * 10
            if name_words <= text_words:
                return 500 + len(name_words) * 10

        best_overlap = 0
        for ref_words in (trigger_words, name_words):
            if not ref_words:
                continue
            overlap = len(text_words & ref_words)
            total = max(len(text_words), len(ref_words))
            if total > 0 and overlap / total >= 0.5:
                best_overlap = max(best_overlap, overlap)

        if best_overlap > 0:
            return 100 + best_overlap * 10

        return 0

    @staticmethod
    def _row_to_dict(row) -> dict:
        """Convert a sqlite3.Row to a plain dict with steps parsed from JSON."""
        result = dict(row)
        try:
            result["steps"] = json.loads(result["steps"])
        except (json.JSONDecodeError, TypeError):
            result["steps"] = []
        return result

    @staticmethod
    def step_count_warning(steps: list[dict]) -> Optional[str]:
        """Return a warning string if step count hits the soft cap, else None."""
        if len(steps) >= _MAX_STEPS:
            return (
                f"That's {len(steps)} steps — at the maximum of {_MAX_STEPS}. "
                "I can't add more."
            )
        if len(steps) >= _WARN_STEPS:
            return (
                f"That's {len(steps)} steps — longer procedures are less reliable. "
                "Want to continue adding steps or save what we have?"
            )
        return None
