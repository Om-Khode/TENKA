"""app_manifest_index.py — SQLite index over <sandbox>/manifests/*.yaml.

Source of truth is the YAML files on disk. This repo only indexes for
fast lookup (active-app match, phrase → (app, intent) lookup). Rebuilt
from disk on startup; mtime mismatch triggers re-scan per file.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any


class AppManifestIndexRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._db = conn

    # ─── App index ────────────────────────────────────────────────────────

    def upsert_manifest(
        self, *, app_id: str, file_path: str, file_mtime: float,
        process_names: list[str], window_patterns: list[str], intent_count: int,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO app_manifest_index
                (app_id, file_path, file_mtime, process_names, window_patterns,
                 intent_count, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))
            ON CONFLICT(app_id) DO UPDATE SET
                file_path = excluded.file_path,
                file_mtime = excluded.file_mtime,
                process_names = excluded.process_names,
                window_patterns = excluded.window_patterns,
                intent_count = excluded.intent_count,
                indexed_at = excluded.indexed_at
            """,
            (app_id, file_path, file_mtime,
             json.dumps(process_names), json.dumps(window_patterns), intent_count),
        )
        self._db.commit()

    def get(self, app_id: str) -> dict[str, Any] | None:
        cur = self._db.execute(
            "SELECT app_id, file_path, file_mtime, process_names, window_patterns, "
            "intent_count, last_dispatched, indexed_at "
            "FROM app_manifest_index WHERE app_id = ?",
            (app_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "app_id": row[0], "file_path": row[1], "file_mtime": row[2],
            "process_names": json.loads(row[3]),
            "window_patterns": json.loads(row[4]),
            "intent_count": row[5], "last_dispatched": row[6], "indexed_at": row[7],
        }

    def all_apps(self) -> list[dict[str, Any]]:
        cur = self._db.execute(
            "SELECT app_id, file_path, file_mtime, process_names, window_patterns, "
            "intent_count, last_dispatched, indexed_at FROM app_manifest_index"
        )
        return [
            {
                "app_id": r[0], "file_path": r[1], "file_mtime": r[2],
                "process_names": json.loads(r[3]),
                "window_patterns": json.loads(r[4]),
                "intent_count": r[5], "last_dispatched": r[6], "indexed_at": r[7],
            }
            for r in cur.fetchall()
        ]

    def delete(self, app_id: str) -> None:
        self._db.execute("DELETE FROM app_manifest_phrases WHERE app_id = ?", (app_id,))
        self._db.execute("DELETE FROM app_manifest_index WHERE app_id = ?", (app_id,))
        self._db.commit()

    def touch_last_dispatched(self, app_id: str) -> None:
        self._db.execute(
            "UPDATE app_manifest_index SET last_dispatched = strftime('%s','now') "
            "WHERE app_id = ?",
            (app_id,),
        )
        self._db.commit()

    # ─── Phrase index ─────────────────────────────────────────────────────

    def replace_phrases(
        self, app_id: str, phrases: list[tuple[str, str, bool]],
    ) -> None:
        """Replace ALL phrases for an app with the given list.

        Each entry is (phrase, intent_id, is_synthesized).
        """
        self._db.execute("DELETE FROM app_manifest_phrases WHERE app_id = ?", (app_id,))
        self._db.executemany(
            "INSERT OR IGNORE INTO app_manifest_phrases "
            "(phrase, app_id, intent_id, is_synthesized) VALUES (?, ?, ?, ?)",
            [(p, app_id, i, int(s)) for (p, i, s) in phrases],
        )
        self._db.commit()

    def find_phrase(self, phrase: str) -> list[dict[str, Any]]:
        cur = self._db.execute(
            "SELECT phrase, app_id, intent_id, is_synthesized "
            "FROM app_manifest_phrases WHERE phrase = ?",
            (phrase,),
        )
        return [
            {"phrase": r[0], "app_id": r[1], "intent_id": r[2], "is_synthesized": bool(r[3])}
            for r in cur.fetchall()
        ]
