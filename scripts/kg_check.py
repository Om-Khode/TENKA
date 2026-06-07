"""Quick KG inspection helper. Run: python kg_check.py"""
import sqlite3
from pathlib import Path

DB = Path.home() / "TENKA" / "memory" / "tenka.db"

c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print(f"DB: {DB}")
print()

tables = [r[0] for r in c.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kg_%' ORDER BY name"
)]
print(f"KG tables: {tables}")
print()

if not tables:
    print("No KG tables found — schema migration may not have run on this DB.")
    raise SystemExit(0)

print("ENTITIES:")
for r in c.execute(
    "SELECT id, type, display_name, confidence, source FROM kg_entities ORDER BY id"
):
    print(" ", dict(r))

print()
print("FACTS:")
for r in c.execute(
    "SELECT id, subject_id, predicate, object, confidence, source FROM kg_facts ORDER BY id"
):
    print(" ", dict(r))

print()
print("RELATIONSHIPS:")
for r in c.execute(
    "SELECT id, from_id, to_id, type, confidence, source FROM kg_relationships ORDER BY id"
):
    print(" ", dict(r))
