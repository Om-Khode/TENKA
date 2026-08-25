"""
storage/db.py — Single Database connection for all SQLite stores.

One Database instance, one connection, one schema-version table.
Repos call db.execute(...) — no raw connections leave this module.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("storage")

_instance: "Database | None" = None


class Database:
    """Single SQLite connection with schema versioning."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()
        logger.info(f"[STORAGE] Database initialized at {path}")

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: list[tuple]) -> sqlite3.Cursor:
        return self._conn.executemany(sql, rows)

    def executescript(self, sql: str) -> None:
        self._conn.executescript(sql)

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def backup_to(self, dest_path: Path) -> None:
        """Write a consistent on-disk snapshot of this database to dest_path.

        Uses SQLite's own online backup API rather than copying the .db
        file's bytes directly — safe under WAL mode and concurrent
        writers, where a raw file copy can capture a mid-write state.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_conn = sqlite3.connect(str(dest_path))
        try:
            self._conn.backup(dest_conn)
        finally:
            dest_conn.close()

    @property
    def path(self) -> Path:
        return self._path

    # --- Schema versioning ---

    _LATEST_VERSION = 23

    def _get_version(self) -> int:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='_schema_version'"
        ).fetchone()
        if row is None:
            return 0
        row = self._conn.execute(
            "SELECT version FROM _schema_version"
        ).fetchone()
        return int(row["version"]) if row else 0

    def _set_version(self, version: int) -> None:
        self._conn.execute(
            "INSERT INTO _schema_version (id, version) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET version = excluded.version",
            (version,),
        )
        self._conn.commit()

    def _migrate(self) -> None:
        current = self._get_version()
        if current >= self._LATEST_VERSION:
            return

        migrations = {
            1: self._migrate_v1,
            2: self._migrate_v2,
            3: self._migrate_v3,
            4: self._migrate_v4,
            5: self._migrate_v5,
            6: self._migrate_v6,
            7: self._migrate_v7,
            8: self._migrate_v8,
            9: self._migrate_v9,
            10: self._migrate_v10,
            11: self._migrate_v11,
            12: self._migrate_v12,
            13: self._migrate_v13,
            14: self._migrate_v14,
            15: self._migrate_v15,
            16: self._migrate_v16,
            17: self._migrate_v17,
            18: self._migrate_v18,
            19: self._migrate_v19,
            20: self._migrate_v20,
            21: self._migrate_v21,
            22: self._migrate_v22,
            23: self._migrate_v23,
        }

        for v in range(current + 1, self._LATEST_VERSION + 1):
            logger.info(f"[STORAGE] Running migration v{v}")
            migrations[v]()
            self._set_version(v)

        logger.info(
            f"[STORAGE] Schema at v{self._LATEST_VERSION}"
        )

    def _migrate_v1(self) -> None:
        """V1: create schema_version table + all existing store tables."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS _schema_version (
                id      INTEGER PRIMARY KEY,
                version INTEGER NOT NULL
            );

            -- personality_state tables
            CREATE TABLE IF NOT EXISTS personality_state (
                trait       TEXT PRIMARY KEY,
                value       REAL NOT NULL,
                floor_val   REAL NOT NULL,
                ceiling_val REAL NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS personality_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                trait       TEXT NOT NULL,
                old_value   REAL NOT NULL,
                new_value   REAL NOT NULL,
                delta       REAL NOT NULL,
                reason      TEXT NOT NULL,
                trigger     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            -- preference_store tables
            CREATE TABLE IF NOT EXISTS user_preferences (
                key             TEXT PRIMARY KEY,
                value           TEXT NOT NULL,
                category        TEXT NOT NULL,
                confidence      REAL NOT NULL DEFAULT 0.5,
                source          TEXT NOT NULL,
                times_used      INTEGER DEFAULT 0,
                times_overridden INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preference_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                key             TEXT NOT NULL,
                old_value       TEXT,
                new_value       TEXT NOT NULL,
                old_confidence  REAL,
                new_confidence  REAL NOT NULL,
                source          TEXT NOT NULL,
                reason          TEXT NOT NULL
            );

            -- procedure_store table
            CREATE TABLE IF NOT EXISTS user_procedures (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger     TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                steps       TEXT NOT NULL,
                backend     TEXT NOT NULL DEFAULT 'auto',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                use_count   INTEGER NOT NULL DEFAULT 0,
                last_used   TEXT DEFAULT NULL,
                enabled     INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_procedures_trigger
                ON user_procedures(trigger);

            -- settings_store table
            CREATE TABLE IF NOT EXISTS runtime_settings (
                key             TEXT PRIMARY KEY,
                value           TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                updated_source  TEXT NOT NULL DEFAULT 'user'
            );

            -- shortcut_store table
            CREATE TABLE IF NOT EXISTS user_shortcuts (
                trigger         TEXT PRIMARY KEY,
                intent          TEXT NOT NULL,
                params_json     TEXT NOT NULL DEFAULT '{}',
                description     TEXT NOT NULL DEFAULT '',
                times_used      INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
        """)
        self._conn.commit()

    _V2_COLUMNS = {
        "conversations": (
            "id", "timestamp", "user_input", "intent", "response", "session_id",
        ),
        "facts": ("id", "timestamp", "key", "value", "source"),
        "recording_sessions": (
            "id", "session_id", "chunk_index", "timestamp", "transcript",
        ),
    }

    def _migrate_v2(self) -> None:
        """V2: add memory tables (conversations, facts, recording_sessions).

        Migrates data from legacy assistant_memory.db if it exists in the
        same directory. Uses INSERT OR IGNORE so partial retries are safe.
        """
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                user_input  TEXT,
                intent      TEXT,
                response    TEXT,
                session_id  TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT,
                key         TEXT,
                value       TEXT,
                source      TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS recording_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                timestamp   TEXT NOT NULL,
                transcript  TEXT NOT NULL
            )
        """)
        self._conn.commit()

        legacy_path = self._path.parent / "assistant_memory.db"
        if legacy_path.exists():
            logger.info(f"[STORAGE] Migrating data from legacy {legacy_path}")
            try:
                self._conn.execute(
                    "ATTACH DATABASE ? AS legacy", (str(legacy_path),)
                )
                try:
                    for table, cols in self._V2_COLUMNS.items():
                        has_table = self._conn.execute(
                            "SELECT name FROM legacy.sqlite_master "
                            "WHERE type='table' AND name=?",
                            (table,),
                        ).fetchone()
                        if has_table:
                            col_list = ", ".join(cols)
                            self._conn.execute(
                                f"INSERT OR IGNORE INTO main.{table} ({col_list}) "
                                f"SELECT {col_list} FROM legacy.{table}"
                            )
                    self._conn.commit()
                    logger.info("[STORAGE] Legacy memory data migrated successfully")
                finally:
                    self._conn.execute("DETACH DATABASE legacy")
            except Exception as e:
                logger.warning(f"[STORAGE] Legacy migration failed: {e}")

    def _migrate_v3(self) -> None:
        """V3: add memory governance columns (memory_type, expires_at) to facts."""
        from datetime import datetime, timedelta

        self._conn.execute(
            "ALTER TABLE facts ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'fact'"
        )
        self._conn.execute(
            "ALTER TABLE facts ADD COLUMN expires_at TEXT"
        )
        expires_at = (datetime.now() + timedelta(days=30)).isoformat()
        self._conn.execute(
            "UPDATE facts SET expires_at = ? WHERE expires_at IS NULL",
            (expires_at,),
        )
        self._conn.commit()

    def _migrate_v4(self) -> None:
        """V4: add session_snapshots table for session continuity."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS session_snapshots (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL UNIQUE,
                started_at   TEXT NOT NULL,
                ended_at     TEXT,
                turn_count   INTEGER DEFAULT 0,
                last_intent  TEXT,
                task_summary TEXT,
                blocker      TEXT,
                summarized   INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

    def _migrate_v5(self) -> None:
        """V5: add schedules table for scheduler (scheduled conditional tasks)."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                cron_expr       TEXT NOT NULL,
                task_type       TEXT NOT NULL,
                task_goal       TEXT NOT NULL,
                notify_mode     TEXT NOT NULL DEFAULT 'on_match_only',
                condition_text  TEXT,
                last_result_hash TEXT,
                last_fired_at   TEXT,
                next_fire_at    TEXT NOT NULL,
                enabled         INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _migrate_v6(self) -> None:
        """V6: add event_monitors table for event-monitor (event-driven monitors)."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS event_monitors (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                event_type      TEXT    NOT NULL,
                source_filter   TEXT,
                condition_mode  TEXT    NOT NULL DEFAULT 'code',
                condition_expr  TEXT,
                condition_prompt TEXT,
                action_type     TEXT    NOT NULL,
                action_payload  TEXT    NOT NULL,
                enabled         INTEGER NOT NULL DEFAULT 1,
                cooldown_secs   INTEGER NOT NULL DEFAULT 5,
                last_fired_at   TEXT,
                fire_count      INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    NOT NULL,
                user_goal       TEXT    NOT NULL
            )
        """)
        self._conn.commit()

    def _migrate_v7(self) -> None:
        """V7: add interaction_events table for telemetry."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS interaction_events (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id                  TEXT NOT NULL,
                timestamp                   TEXT NOT NULL,
                input_modality              TEXT NOT NULL,
                transcript                  TEXT,
                intent_detected             TEXT,
                intent_source               TEXT,
                action_dispatched           TEXT,
                action_outcome              TEXT,
                error_class                 TEXT,
                latency_total_ms            INTEGER,
                latency_stt_ms              INTEGER,
                latency_intent_ms           INTEGER,
                latency_action_ms           INTEGER,
                latency_tts_ms              INTEGER,
                llm_calls_count             INTEGER DEFAULT 0,
                llm_tokens_in               INTEGER DEFAULT 0,
                llm_tokens_out              INTEGER DEFAULT 0,
                fallback_chain_depth        INTEGER DEFAULT 0,
                vision_calls_count          INTEGER DEFAULT 0,
                user_corrected_within_30s   INTEGER DEFAULT 0,
                same_intent_repeated        INTEGER DEFAULT 0
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ie_session ON interaction_events(session_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ie_timestamp ON interaction_events(timestamp)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ie_intent ON interaction_events(intent_detected)"
        )
        self._conn.commit()

    def _migrate_v8(self) -> None:
        """V8: add automation_cache table for step caching."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS automation_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                backend         TEXT NOT NULL,
                app_name        TEXT NOT NULL,
                goal_slug       TEXT NOT NULL,
                goal_text       TEXT NOT NULL,
                steps_json      TEXT NOT NULL,
                hit_count       INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                last_hit_at     TEXT NOT NULL,
                version         INTEGER NOT NULL DEFAULT 1
            )
        """)
        self._conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ac_lookup
            ON automation_cache (backend, app_name, goal_slug)
        """)
        self._conn.commit()

    def _migrate_v9(self) -> None:
        """V9: Per-personality trait rows — composite PK on personality_state."""
        self._conn.executescript("""
            CREATE TABLE personality_state_new (
                personality_id TEXT NOT NULL,
                trait          TEXT NOT NULL,
                value          REAL NOT NULL,
                floor_val      REAL NOT NULL,
                ceiling_val    REAL NOT NULL,
                updated_at     TEXT NOT NULL,
                PRIMARY KEY (personality_id, trait)
            );
            INSERT INTO personality_state_new
                SELECT 'tsundere', trait, value, floor_val, ceiling_val, updated_at
                FROM personality_state;
            DROP TABLE personality_state;
            ALTER TABLE personality_state_new RENAME TO personality_state;

            ALTER TABLE personality_log
                ADD COLUMN personality_id TEXT NOT NULL DEFAULT 'tsundere';
        """)

    def _migrate_v10(self) -> None:
        """V10: FTS5 virtual tables + sync triggers for facts and conversations."""
        self._conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(key, value);

            -- facts_fts and conversations_fts are regular (non-external-content)
            -- FTS5 tables, so delete-via-command-form ("INSERT INTO ftsname(ftsname,
            -- rowid, ...) VALUES('delete', ...)") does not apply — that syntax is
            -- only valid for external-content tables. Use a plain DELETE here.
            -- v19 retroactively repairs DBs that landed at v10 with the broken
            -- syntax; this block keeps fresh DBs correct from creation.
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, key, value)
                    VALUES (new.id, new.key, new.value);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                DELETE FROM facts_fts WHERE rowid = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                DELETE FROM facts_fts WHERE rowid = old.id;
                INSERT INTO facts_fts(rowid, key, value)
                    VALUES (new.id, new.key, new.value);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(user_input, response);

            CREATE TRIGGER IF NOT EXISTS conv_ai AFTER INSERT ON conversations BEGIN
                INSERT INTO conversations_fts(rowid, user_input, response)
                    VALUES (new.id, new.user_input, new.response);
            END;
            CREATE TRIGGER IF NOT EXISTS conv_ad AFTER DELETE ON conversations BEGIN
                DELETE FROM conversations_fts WHERE rowid = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS conv_au AFTER UPDATE ON conversations BEGIN
                DELETE FROM conversations_fts WHERE rowid = old.id;
                INSERT INTO conversations_fts(rowid, user_input, response)
                    VALUES (new.id, new.user_input, new.response);
            END;

            INSERT INTO facts_fts(rowid, key, value)
                SELECT id, key, value FROM facts;
            INSERT INTO conversations_fts(rowid, user_input, response)
                SELECT id, user_input, response FROM conversations;
        """)

    def _migrate_v11(self) -> None:
        """V11: per-turn LLM provider/model breakdown columns.

        Adds JSON-encoded counter columns to interaction_events so we can
        see provider mix per turn (e.g. {"gemini": 3, "groq": 1}) without
        a separate llm_calls table.
        """
        self._conn.execute(
            "ALTER TABLE interaction_events ADD COLUMN llm_providers_used TEXT"
        )
        self._conn.execute(
            "ALTER TABLE interaction_events ADD COLUMN llm_models_used TEXT"
        )
        self._conn.commit()

    def _migrate_v12(self) -> None:
        """V12 (manifest): app manifest index + phrases.

        Stores a denormalized index over <sandbox>/manifests/*.yaml so the
        regex pre-router can look up (app, intent) by phrase in O(1) and
        the active-app matcher can scan candidate manifests by process
        without re-reading every YAML.
        """
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS app_manifest_index (
                app_id            TEXT PRIMARY KEY,
                file_path         TEXT NOT NULL,
                file_mtime        REAL NOT NULL,
                process_names     TEXT NOT NULL,
                window_patterns   TEXT NOT NULL,
                intent_count      INTEGER NOT NULL,
                last_dispatched   REAL,
                indexed_at        REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_manifest_phrases (
                phrase            TEXT NOT NULL,
                app_id            TEXT NOT NULL,
                intent_id         TEXT NOT NULL,
                is_synthesized    INTEGER NOT NULL,
                PRIMARY KEY (phrase, app_id, intent_id)
            );
            CREATE INDEX IF NOT EXISTS idx_phrases_phrase
                ON app_manifest_phrases(phrase);
        """)
        self._conn.commit()

    def _migrate_v13(self) -> None:
        """V13 (manifest): automation-cache promotion bookkeeping.

        Adds a nullable promoted_intent_id column to automation_cache so the
        manifest-based promoter can claim entries it has lifted into a manifest. NULL
        means "not yet promoted"; a string like "<app_id>:<intent_id>"
        identifies the manifest entry that absorbed this cached procedure.
        """
        self._conn.execute(
            "ALTER TABLE automation_cache ADD COLUMN promoted_intent_id TEXT"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ac_promoted "
            "ON automation_cache(promoted_intent_id)"
        )
        self._conn.commit()

    def _migrate_v14(self) -> None:
        """V14 (manifest): daily vision-call cap counter table.

        Tier-2 healer (Gemini Flash-Lite vision) is gated by a daily call
        cap (default 100/day). The (day, count) row is checked + atomically
        incremented per call so the cap survives TENKA restarts within the
        same calendar day. Rolled over by the SC-1 midnight tick.
        """
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vision_calls ("
            "  day TEXT PRIMARY KEY, "
            "  count INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._conn.commit()

    def _migrate_v15(self) -> None:
        """V15 : knowledge graph layer — entities, facts, relationships.

        Adds three tables for typed entity storage that sits in parallel to
        the flat `facts` table. hybrid-retrieval retrieval is untouched. See
        docs/superpowers/specs/2026-05-31-kg-1-knowledge-graph-design.md.
        """
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                type            TEXT NOT NULL,
                canonical_name  TEXT NOT NULL,
                display_name    TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                source          TEXT NOT NULL,
                confidence      REAL NOT NULL DEFAULT 1.0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                UNIQUE(type, canonical_name)
            );
            CREATE INDEX IF NOT EXISTS idx_kg_entities_canon
                ON kg_entities(canonical_name);
            CREATE INDEX IF NOT EXISTS idx_kg_entities_type
                ON kg_entities(type);

            CREATE TABLE IF NOT EXISTS kg_facts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id   INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                predicate    TEXT NOT NULL,
                object       TEXT NOT NULL,
                confidence   REAL NOT NULL DEFAULT 1.0,
                source       TEXT NOT NULL,
                verified_at  TEXT,
                expires_at   TEXT,
                created_at   TEXT NOT NULL,
                UNIQUE(subject_id, predicate, object)
            );
            CREATE INDEX IF NOT EXISTS idx_kg_facts_subj ON kg_facts(subject_id);
            CREATE INDEX IF NOT EXISTS idx_kg_facts_pred ON kg_facts(predicate);

            CREATE TABLE IF NOT EXISTS kg_relationships (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id         INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                to_id           INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                type            TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                confidence      REAL NOT NULL DEFAULT 1.0,
                source          TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                UNIQUE(from_id, to_id, type)
            );
            CREATE INDEX IF NOT EXISTS idx_kg_rel_from ON kg_relationships(from_id);
            CREATE INDEX IF NOT EXISTS idx_kg_rel_to   ON kg_relationships(to_id);
        """)
        self._conn.commit()

    def _migrate_v16(self) -> None:
        """V16 (A+B): add event_at + invalid_at nullable TEXT columns
        to kg_facts. event_at = when the fact happened (ISO string, distinct
        from created_at). invalid_at = soft-delete marker set when a newer
        fact for the same (subject_id, predicate) but different object
        supersedes the old one. Both NULL on legacy rows.
        """
        self._conn.executescript("""
            ALTER TABLE kg_facts ADD COLUMN event_at   TEXT;
            ALTER TABLE kg_facts ADD COLUMN invalid_at TEXT;
        """)
        self._conn.commit()

    def _migrate_v17(self) -> None:
        """V17 (H): source_turn_id provenance backlink on kg_entities,
        kg_facts, kg_relationships. NULL on legacy rows. Lets queries answer
        'why do you think that?' by tracing a KG row back to the originating
        conversation turn — the Zep-style episodic backlink.
        Format is opaque TEXT so callers can pick their own scheme (e.g.
        f"{session_id}:{conv_row_id}").
        """
        self._conn.executescript("""
            ALTER TABLE kg_entities      ADD COLUMN source_turn_id TEXT;
            ALTER TABLE kg_facts         ADD COLUMN source_turn_id TEXT;
            ALTER TABLE kg_relationships ADD COLUMN source_turn_id TEXT;
        """)
        self._conn.commit()

    def _migrate_v18(self) -> None:
        """V18 (E): kg_commitments — first-class promises.

        Ontology call (Session 4): commitments and reminders are DISTINCT
        concepts that may co-occur. Reminders are time-anchored
        notifications fired by a background poller (assistant/reminders.py).
        Commitments are promises the user made (or made TO the user) that
        may or may not carry a deadline. A commitment that does carry a
        deadline can optionally link to a reminder row via reminder_id —
        but the commitment is the source-of-truth promise; the reminder is
        just the notification trigger.

        owner_id FKs kg_entities so commitments hang off the person who
        made (or received) them. when_due is a free-form TEXT (ISO when
        parseable, otherwise the original phrase) — matches event_at's
        permissive shape. fulfilled_at NULL = still open. reminder_id is
        nullable; on reminder removal it stays as a dangling pointer that
        history queries can still surface ("you DID promise X, the alert
        just fired") — no ON DELETE CASCADE.

        source_turn_id mirrors v17's provenance pattern.
        """
        self._conn.executescript("""
            CREATE TABLE kg_commitments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id        INTEGER NOT NULL,
                promise_text    TEXT NOT NULL,
                when_due        TEXT,
                created_at      TEXT NOT NULL,
                fulfilled_at    TEXT,
                source          TEXT NOT NULL,
                source_turn_id  TEXT,
                reminder_id     INTEGER,
                FOREIGN KEY (owner_id) REFERENCES kg_entities(id)
            );
            CREATE INDEX idx_kg_commitments_owner    ON kg_commitments(owner_id);
            CREATE INDEX idx_kg_commitments_open     ON kg_commitments(fulfilled_at)
                WHERE fulfilled_at IS NULL;
            CREATE INDEX idx_kg_commitments_due      ON kg_commitments(when_due)
                WHERE when_due IS NOT NULL;
        """)

    def _migrate_v19(self) -> None:
        """V19: fix facts/conversations FTS sync triggers.

        v10 created facts_ad/facts_au/conv_ad/conv_au with the FTS5
        external-content "delete" command-form, which is invalid for regular
        FTS5 tables and raises 'SQL logic error' on every DELETE/UPDATE.
        Symptom in production: cleanup_expired() and any other delete on the
        facts or conversations table silently failed. Drop and recreate the
        four triggers with the correct plain-DELETE syntax.
        """
        self._conn.executescript("""
            DROP TRIGGER IF EXISTS facts_ad;
            DROP TRIGGER IF EXISTS facts_au;
            DROP TRIGGER IF EXISTS conv_ad;
            DROP TRIGGER IF EXISTS conv_au;

            CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
                DELETE FROM facts_fts WHERE rowid = old.id;
            END;
            CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
                DELETE FROM facts_fts WHERE rowid = old.id;
                INSERT INTO facts_fts(rowid, key, value)
                    VALUES (new.id, new.key, new.value);
            END;
            CREATE TRIGGER conv_ad AFTER DELETE ON conversations BEGIN
                DELETE FROM conversations_fts WHERE rowid = old.id;
            END;
            CREATE TRIGGER conv_au AFTER UPDATE ON conversations BEGIN
                DELETE FROM conversations_fts WHERE rowid = old.id;
                INSERT INTO conversations_fts(rowid, user_input, response)
                    VALUES (new.id, new.user_input, new.response);
            END;
        """)
        self._conn.commit()
        self._conn.commit()

    def _migrate_v20(self) -> None:
        """V20: security_skip marker on conversations.

        A turn a security control skipped this turn -- KI-13's foreign-answer
        skip, or a capability shortfall in the pending-handler chain -- still
        gets an honest deterministic reply from main.py's turn loop, but that
        is only the first half of the fix. The reply still lands in the
        `conversations` table the same way every turn's does, and
        `session.save_snapshot()` folds recent turns into an LLM-written
        session summary that gets replayed verbatim as fact at the start of
        the next session (`get_resume_context`). This column is the
        deterministic backstop for that second half: a turn flagged here is
        excluded before it ever reaches the summarizer, regardless of what
        its own reply text says -- no model judges its own output to decide.

        NOT NULL DEFAULT 0 so every legacy row and every ordinary turn reads
        as "not a skip" with no backfill required; `main.py` sets it to 1 only
        on the turns its own skip-tracking flags.
        """
        self._conn.execute(
            "ALTER TABLE conversations ADD COLUMN security_skip "
            "INTEGER NOT NULL DEFAULT 0"
        )
        self._conn.commit()


    def _migrate_v21(self) -> None:
        """V21: who installed this, on everything that runs later.

        `event_monitors`, `schedules`, `user_procedures` and `user_shortcuts`
        all store something that fires after the turn that created it, and
        `automation/event_bus.py` and `scheduler.py` run it with
        `LOCAL_GRANTS`. Until now the row recorded nothing about who asked
        for it, so the audit trail said the machine had done it to itself.

        The gate that stops a temporary raise installing one of these is at
        dispatch (`actions.durable_capability_refusal`), not here. This column
        is the *record*, deliberately: a fire-time check would need a live
        policy for a device that may not be connected, and inventing one is
        worse than knowing who to ask.

        NOT NULL DEFAULT 'local' so every existing row reads as
        keyboard-installed with no backfill. That is honest rather than
        convenient: before this migration a remote device could not reach any
        of these intents at all without a raise, and the raise mechanism is
        newer than every row in the table.
        """
        for table in ("event_monitors", "schedules",
                      "user_procedures", "user_shortcuts"):
            cols = {
                r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")
            }
            if not cols:
                # A table this schema has not created yet. Nothing to alter,
                # and its own CREATE will carry the column.
                continue
            if "installed_by" in cols:
                continue
            self._conn.execute(
                f"ALTER TABLE {table} ADD COLUMN installed_by "
                "TEXT NOT NULL DEFAULT 'local'"
            )
        self._conn.commit()


    def _migrate_v23(self) -> None:
        """V23: four telemetry columns that answer §15's O2 questions.

        Added because the six questions O2 names have to be answerable *by
        query*, without reading source. Three of them were not:

            why did she call a model      -> `llm_purposes`
            why did planning happen       -> `replan_count`
            why did execution fail        -> `recovery_count`
            why did she report success    -> `verification_tiers`

        **Four columns, not the ten §15 lists**, and the omission is the
        phase's own rule rather than an oversight: *a field that is always null
        is not observability*. `task_id`, `step_id`, `affordance`, `operation`
        and `final_task_status` all need a Task to exist per turn, and nothing
        creates one -- `brain/authority.py:create_task` has no caller in
        `assistant/` (P5). `context_bytes_by_profile` needs the Context Builder
        of section 12, which is not built. Adding six columns that can only
        ever be NULL would make the schema *look* like it answers questions it
        cannot, which is worse than a schema that visibly does not.

        Counters are stored as JSON objects rather than as separate columns for
        the same reason `llm_providers_used` is: the key set is open. A new
        verification tier or a new `task_type` must not need a migration.
        """
        for column, sql_type in (
            ("llm_purposes", "TEXT"),
            ("replan_count", "INTEGER NOT NULL DEFAULT 0"),
            ("recovery_count", "INTEGER NOT NULL DEFAULT 0"),
            ("verification_tiers", "TEXT"),
        ):
            try:
                self._conn.execute(
                    f"ALTER TABLE interaction_events ADD COLUMN "
                    f"{column} {sql_type}"
                )
            except Exception as e:
                # An existing column is the expected outcome of a re-run, and a
                # migration that cannot be re-run is a migration that cannot be
                # recovered from a half-applied state.
                if "duplicate column" not in str(e).lower():
                    raise
        self._conn.commit()

    def _migrate_v22(self) -> None:
        """V22: tasks and task_steps.

        A Task that survives a restart is the point of persisting one, and the
        two columns that make that safe rather than dangerous are `principal`
        and `granted` -- see `brain/authority.py`. Without them a resumed Task
        is either refused by every step (an unset grant set refuses) or run
        with LOCAL_GRANTS by whatever picked it up, which is a phone's banked
        work laundered into keyboard privilege.

        `granted` stores a **sorted comma-joined list of capability values**,
        never an integer bitmask. A bitmask silently re-maps if the enum's
        member order changes, which is exactly the kind of quiet re-grant the
        explicit-literal discipline in `io/api/policy.py` exists to prevent. A
        name that no longer exists in the enum must be a load error, not a
        dropped member.

        No foreign key from `task_steps` to `tasks`: `PRAGMA foreign_keys` is
        ON for this connection, and a cascade delete on a task would silently
        discard the step history that explains what it did. Steps are cleaned
        up explicitly or not at all.

        `status` is a TEXT enum value rather than an integer for the same
        reason `intent` is a string: a database someone opens by hand should
        say `suspended_needs_authority`, not `10`.
        """
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id      TEXT PRIMARY KEY,
                intent       TEXT NOT NULL,
                principal    TEXT NOT NULL,
                granted      TEXT NOT NULL,
                affordance   TEXT NOT NULL DEFAULT '',
                operation    TEXT NOT NULL DEFAULT '',
                parameters   TEXT NOT NULL DEFAULT '{}',
                constraints  TEXT NOT NULL DEFAULT '{}',
                source       TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL DEFAULT '',
                expected     TEXT NOT NULL DEFAULT '',
                context_ref  TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                parent       TEXT,
                updated_at   TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_steps (
                task_id      TEXT NOT NULL,
                step_id      INTEGER NOT NULL,
                intent       TEXT NOT NULL,
                affordance   TEXT NOT NULL DEFAULT '',
                operation    TEXT NOT NULL DEFAULT '',
                parameters   TEXT NOT NULL DEFAULT '{}',
                depends_on   TEXT NOT NULL DEFAULT '',
                condition    TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                goal         TEXT NOT NULL DEFAULT '',
                outcome      TEXT,
                observation  TEXT,
                tier         TEXT,
                escalated    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (task_id, step_id)
            )
            """
        )
        # Resumption asks two questions -- "whose is this?" and "what is still
        # open?" -- so those are the two indexes. `status` alone would scan
        # every task a device ever created.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_principal_status "
            "ON tasks (principal, status)"
        )
        self._conn.commit()


def init_db(path: Path) -> "Database":
    """Initialize the singleton Database. Safe to call multiple times."""
    global _instance
    if _instance is not None:
        return _instance
    _instance = Database(path)
    return _instance


def get_db() -> "Database | None":
    """Return the singleton Database, or None if not yet initialized."""
    return _instance


def close_for_restore() -> None:
    """Close the live connection and clear the singleton before a backup
    restore overwrites tenka.db on disk.

    A still-open connection has its own page cache and WAL bookkeeping for
    the file it originally opened — if the file's bytes get replaced out
    from under it (as backup restore does) and anything then writes
    through the stale connection, SQLite can write new WAL frames against
    a file whose physical structure no longer matches what the connection
    last read, corrupting it. Closing first prevents that.

    Callers are expected to call init_db() again immediately after the
    swap completes (io/backup/orchestrator.py's run_restore() does this)
    — a get_db() left returning None until some later manual restart
    turned out to be far more fragile than the corruption this guards
    against: dozens of call sites across the app assume a live connection
    always exists once init_db() has run once, and each one crashes the
    moment it doesn't.
    """
    global _instance
    if _instance is not None:
        _instance.close()
    _instance = None


def _reset_for_testing() -> None:
    """Reset singleton — test use only."""
    global _instance
    if _instance is not None:
        _instance.close()
    _instance = None
