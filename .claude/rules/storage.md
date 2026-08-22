---
paths:
  - "assistant/storage/**"
  - "assistant/memory.py"
  - "assistant/knowledge.py"
  - "assistant/knowledge_graph.py"
  - "assistant/preferences.py"
  - "assistant/session.py"
  - "assistant/telemetry.py"
  - "assistant/procedures.py"
  - "assistant/shortcuts.py"
  - "assistant/reminders.py"
  - "assistant/settings.py"
  - "assistant/core/runtime_config.py"
  - "assistant/io/backup/**"
---

# Storage and durable state

Where data lives: `ARCHITECTURE.md` §18.

## One connection

**One SQLite connection, 13 repos**, all sharing the `storage/db.py` singleton at
`~/TENKA/memory/tenka.db`. **Never open a second connection.** The memory repo additionally
keeps its own FAISS index + ID-map files.

Schema is at **v20**. Migrations are a numbered dict in `storage/db.py:_migrate()` — add
`_migrate_vN`, register it, and bump `_LATEST_VERSION`. Recent: v18 `kg_commitments`, v19 an
FTS trigger fix, v20 `conversations.security_skip`.

`core/runtime_config.py` reads SQLite **lazily** via `get_db()`. `config.py` itself never
imports `sqlite3` or `storage` — it is a single module, not a package, and a test asserts
this.

## Durable state is more than memory

Eleven stores survive the process **and** can influence a later turn:

conversations · facts (+FAISS) · knowledge graph · **preferences** · personality traits ·
procedures · schedules · event monitors · shortcuts · per-service knowledge
(`~/TENKA/knowledge/{service}.json`) · app manifests (`~/TENKA/manifests/*.yaml`)

Anything that steers a later turn is durable state, whatever table it lives in. `preferences`
is the one most often forgotten and the one with the most reach — `automation/router.py`
reads it as routing **priority 1**, above URL detection.

## Provenance

**False memory is worse than no memory.** She should prefer "I don't remember that" over
inventing a personal fact.

- Record **how** a value was obtained, and **consult it on read**. Recording provenance and
  then ignoring it buys nothing: `set_preference` requires `source`, and
  `automation/router.py:208` never reads it (KI-31).
- `memory.save_fact(key, value, source="user")` **defaults to the highest trust tier** — a
  caller that forgets the argument manufactures an explicit user statement. Do not rely on
  the default; it is scheduled for removal.
- LLM inference is not fact. A single inference stays ephemeral; promotion needs repetition
  **counted by TENKA**, not claimed by a model, or explicit user confirmation.
- A correction supersedes; it does not accumulate a second contradictory row.
- Fields worth carrying: `created_at`, `updated_at`, `valid_from`, `valid_until`,
  `confidence`, `source`.

`main.py` currently writes LLM-extracted facts straight to durable memory with no candidate
stage, on two duplicated paths (streaming and non-streaming). Do not add a third.

## Tests

**Integration tests hit real SQLite in tmp dirs.** Mocked DBs have masked migration failures
here before. A migration test that never opens a real file is not testing the migration.
