"""Wipe knowledge-graph tables only (keeps facts/memories/personality untouched).

Run: python kg_wipe.py
Make sure the assistant is NOT running when you execute this.
"""
import sqlite3
from pathlib import Path

DB = Path.home() / "TENKA" / "memory" / "tenka.db"

if not DB.exists():
    print(f"[wipe] DB not found: {DB}")
    raise SystemExit(1)

c = sqlite3.connect(DB)
try:
    # Order matters: relationships + facts reference entities (FK cascade is on,
    # but explicit order is safer if PRAGMA is ever flipped).
    for tbl in ("kg_relationships", "kg_facts", "kg_entities"):
        before = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        c.execute(f"DELETE FROM {tbl}")
        print(f"[wipe] {tbl}: {before} -> 0")
    # Reset autoincrement so the next IDs start at 1
    c.execute(
        "DELETE FROM sqlite_sequence WHERE name IN "
        "('kg_entities', 'kg_facts', 'kg_relationships')"
    )
    c.commit()
    print("[wipe] DONE")
finally:
    c.close()
