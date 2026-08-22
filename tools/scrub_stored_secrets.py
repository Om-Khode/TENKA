"""Scrub credential-shaped strings out of an existing tenka.db.

The write-side fix (`storage/repos/`) stops *new* secrets from being stored.
It does nothing about rows already there, and every backup carries them until
they are gone -- `io/backup/orchestrator.py` snapshots the whole database.

Dry-run by default. It prints how many rows each column would change and
nothing else, so you can see the blast radius before anything is written:

    py -3.11 tools/scrub_stored_secrets.py
    py -3.11 tools/scrub_stored_secrets.py --apply

`--apply` takes a timestamped copy of the database first and prints where it
put it. Redaction is not reversible, so that copy is the only way back.

Uses the same `core.redact.redact_secrets` the write path uses, so a row this
tool leaves alone is a row the write path would also have left alone -- there
is one definition of "looks like a secret" and this does not add a second.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sqlite3
import sys
from datetime import datetime

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.redact import redact_secrets  # noqa: E402

# (table, primary key, column) for every column that stores free text a user
# or a model produced. Keep this list in step with the write sites wired in
# storage/repos/ -- a column added there and missed here scrubs clean and
# stays dirty.
TARGETS = [
    ("conversations", "id", "user_input"),
    ("conversations", "id", "response"),
    ("facts", "id", "value"),
    ("interaction_events", "id", "transcript"),
    ("recording_sessions", "id", "transcript"),
    ("kg_facts", "id", "object"),
    ("session_snapshots", "id", "task_summary"),
    ("session_snapshots", "id", "blocker"),
    # Learned/taught state: a credential pasted during a setup flow can end up
    # in a procedure's steps, a monitor's goal, or a schedule's task text.
    ("user_procedures", "id", "steps"),
    ("user_procedures", "id", "description"),
    ("user_shortcuts", "trigger", "params_json"),
    ("event_monitors", "id", "user_goal"),
    ("event_monitors", "id", "action_payload"),
    ("schedules", "id", "task_goal"),
    ("kg_entities", "id", "properties_json"),
]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def scrub(db_path: pathlib.Path, apply: bool) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    total = 0

    for table, pk, col in TARGETS:
        cols = _columns(conn, table)
        if not cols:
            print(f"  {table:22} (no such table, skipped)")
            continue
        if col not in cols:
            print(f"  {table}.{col:18} (no such column, skipped)")
            continue

        changed = []
        for row in conn.execute(
            f"SELECT {pk} AS pk, {col} AS val FROM {table} "
            f"WHERE {col} IS NOT NULL AND {col} <> ''"
        ):
            clean = redact_secrets(row["val"])
            if clean != row["val"]:
                changed.append((row["pk"], clean))

        print(f"  {table}.{col:18} {len(changed):5} row(s) would change")
        total += len(changed)

        if apply and changed:
            conn.executemany(
                f"UPDATE {table} SET {col} = ? WHERE {pk} = ?",
                [(clean, pk_val) for pk_val, clean in changed],
            )

    if apply:
        # The FTS tables are populated by INSERT triggers on their base tables,
        # so an UPDATE does not refresh them. Rebuild rather than trust the
        # trigger: a secret left in the index is as readable as one left in the
        # table, and `rebuild` is the documented way to resynchronise fts5.
        for fts in ("conversations_fts", "facts_fts"):
            try:
                conn.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")
                print(f"  rebuilt {fts}")
            except sqlite3.Error as e:
                print(f"  WARNING: could not rebuild {fts}: {e}")
        conn.commit()

    conn.close()
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=pathlib.Path,
                    default=pathlib.Path.home() / "TENKA" / "memory" / "tenka.db")
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without this, nothing is modified.")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"no database at {args.db}")
        return 1

    print(f"{'SCRUBBING' if args.apply else 'DRY RUN'}: {args.db}")

    if args.apply:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = args.db.with_name(f"{args.db.stem}.pre-scrub-{stamp}.db")
        shutil.copy2(args.db, backup)
        print(f"backup: {backup}")
        print("Redaction is not reversible. That copy is the only way back.\n")

    total = scrub(args.db, args.apply)

    print()
    if not total:
        print("nothing credential-shaped found.")
    elif args.apply:
        print(f"scrubbed {total} row(s).")
        print("Next: rotate anything that was in there. Redacting the copy in")
        print("this database does not un-leak a secret that reached a backup.")
    else:
        print(f"{total} row(s) would change. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
