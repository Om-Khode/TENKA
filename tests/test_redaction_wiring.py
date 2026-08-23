"""A redactor that cannot be resolved redacts nothing, loudly enough to ignore.

`storage/repos/knowledge_graph.py` called `redact_secrets(object)` and never
imported it. Every `add_fact` raised `NameError`, the caller caught it, and the
log said:

    [KG] persist failed (non-critical): name 'redact_secrets' is not defined

So the knowledge graph persisted nothing at all from `5bf3f65` -- the commit that
wired redaction into five write sites and missed one import -- until this one. A
whole subsystem dead behind the word "non-critical".

**The tests to catch it already existed and were not run.** Removing the import
again turns **21** tests red across `test_knowledge_graph.py`,
`test_kg_invalidation.py`, `test_kg_provenance.py` and `test_kg_temporal.py` --
measured, not assumed. So this was never a coverage gap. `5bf3f65` shipped
without running the KG files, and nothing between then and now ran them either;
the only signal in between was a log line with the word "non-critical" in it.
That is the argument for `tests/BASELINE.md` (P0.5, still deferred) more than
for anything in this file.

What the two tests here add is different. The behavioural one pins the fix
where the fix is, so a reader of this commit can see what broke. The structural
one is *cheap* -- it catches an unbound redactor anywhere in the tree without
running the subsystem that would raise, which matters because these call sites
all sit inside a `try` whose caller logs and continues. `lint-imports` checks
boundaries between packages, not whether a name resolves.

Run with:  py -3.11 -m pytest tests/test_redaction_wiring.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_KEY = "AIzaSyC8Ur4kFakeKeyForTestingOnly123"


# ─── every caller can actually reach the redactor ────────────────────────────

def test_every_module_that_redacts_can_resolve_the_name():
    """The guard that was missing.

    A name used and not bound is a `NameError` on the one path that reaches it,
    which is exactly the kind of defect that survives a test suite: the call
    site is inside a `try` whose caller logs and continues. Static, so it does
    not depend on reaching the line.

    Definitions count as bindings -- `core/redact.py` uses the name it defines,
    and flagging that would make this test noise nobody reads.
    """
    offenders = []
    walked = 0
    for path in sorted((_ROOT / "assistant").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "redact_secrets" not in src:
            continue
        walked += 1
        tree = ast.parse(src)

        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Assign):
                bound.update(t.id for t in node.targets
                             if isinstance(t, ast.Name))

        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id.startswith("redact_secrets")}
        for name in sorted(used - bound):
            offenders.append(f"{path.relative_to(_ROOT)} uses {name}")

    assert walked >= 5, f"only walked {walked} modules -- the sweep found nothing"
    assert not offenders, (
        f"a redactor is called but never bound: {offenders}. This raises "
        f"NameError on whichever path reaches it, and every such call site in "
        f"this tree sits inside a try/except that logs 'non-critical'."
    )


# ─── the knowledge graph actually writes again ───────────────────────────────

@pytest.fixture()
def kg(tmp_path):
    """Real SQLite, real repo. A mock would have been perfectly happy with the
    broken version, since the failure was a missing name rather than a wrong
    call."""
    from assistant.storage.db import Database
    from assistant.storage.repos.knowledge_graph import KnowledgeGraphRepo

    db = Database(tmp_path / "kg.db")
    try:
        yield KnowledgeGraphRepo(db, lambda: None), db
    finally:
        db._conn.close()


def _subject(repo, name="Om"):
    """A fact is addressed by subject id, so an entity comes first.

    `upsert_entity(entity_type, name, source)` returns `(id, was_created)` --
    called explicitly rather than probed for, because guessing at a signature is
    how the first version of this helper passed the arguments in the wrong
    order.
    """
    entity_id, _created = repo.upsert_entity("person", name, "user")
    return entity_id


def test_add_fact_persists_at_all(kg):
    """**The regression.** It raised `NameError` every single call, so the row
    count stayed at zero however many facts were extracted."""
    repo, db = kg
    subject_id = _subject(repo)

    fact_id = repo.add_fact(subject_id, "lives_in", "Pune", source="user")
    assert fact_id, "add_fact returned no row id"

    rows = db.fetchall("SELECT object FROM kg_facts")
    assert len(rows) == 1, (
        f"the knowledge graph stored {len(rows)} facts -- it stored none at all "
        f"between 5bf3f65 and this fix"
    )
    assert rows[0]["object"] == "Pune"


def test_add_fact_redacts_the_object(kg):
    """What the missing import was there to do. This table is replayed into
    prompts and snapshotted to cloud backup, which is why the redaction was
    added -- and why the `NameError` was hiding a second problem behind the
    first."""
    repo, db = kg
    subject_id = _subject(repo)

    repo.add_fact(subject_id, "api_key_is", f"my key is {_KEY}", source="user")
    stored = db.fetchone("SELECT object FROM kg_facts")["object"]
    assert _KEY not in stored, f"a credential was stored verbatim: {stored!r}"
    assert "REDACTED" in stored


def test_an_ordinary_fact_survives_redaction(kg):
    """Both directions. A redactor that ate every object would satisfy the test
    above while making the knowledge graph store nothing usable -- the same
    invisible outcome as the NameError, one layer along."""
    repo, db = kg
    subject_id = _subject(repo)

    repo.add_fact(subject_id, "works_on", "the TENKA assistant", source="user")
    assert db.fetchone("SELECT object FROM kg_facts")["object"] == (
        "the TENKA assistant")
