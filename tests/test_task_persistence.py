"""A persisted Task keeps its authority, exactly.

Persisting a Task is what makes it survive a restart, and the two columns that
make that safe rather than dangerous are `principal` and `granted`. A stored
Task whose authority cannot be reconstructed *exactly* is the whole hazard:
load it narrower and it looks like working software while a security record has
silently shrunk; load it wider and the restart is a privilege escalation.

Real SQLite in a tmp dir throughout, never a mock. Mocked databases have masked
migration failures in this tree before, and two of these tests are about what
the schema does rather than what the repo intends.

Run with:  py -3.11 -m pytest tests/test_task_persistence.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.brain.task import (  # noqa: E402
    Observation, ObservationKind, Outcome, Task, TaskStatus, TaskStep, Verdict,
)
from assistant.core.capabilities import Capability  # noqa: E402
from assistant.storage.repos.task import (  # noqa: E402
    TaskRepo, UnknownCapabilityStored, _load_grants,
)


@pytest.fixture()
def repo(tmp_path):
    from assistant.storage.db import Database
    db = Database(tmp_path / "t.db")
    try:
        yield TaskRepo(db), db
    finally:
        db._conn.close()


def _task(**kw):
    base = dict(
        task_id="t1", intent="file_task", principal="device:phone",
        granted=frozenset({Capability.CHAT_SEND, Capability.FILES}),
        source="studio", created_at="2026-08-23T10:00:00",
    )
    base.update(kw)
    return Task(**base)


# ─── the schema ──────────────────────────────────────────────────────────────

def test_the_migration_creates_both_tables_at_v22(repo):
    _, db = repo
    assert db._get_version() >= 22
    for table in ("tasks", "task_steps"):
        assert db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ) is not None, f"{table} was not created"


def test_authority_columns_are_not_nullable(repo):
    """A Task with a null principal or grant set is the state this whole
    design exists to make impossible. The schema says so, not just the code."""
    _, db = repo
    cols = {r[1]: r for r in db._conn.execute("PRAGMA table_info(tasks)")}
    for name in ("principal", "granted", "intent"):
        assert cols[name][3] == 1, f"tasks.{name} is nullable"


def test_the_migration_is_idempotent(repo):
    _, db = repo
    db._migrate_v22()   # must not raise on an already-migrated database


# ─── authority round-trips exactly ───────────────────────────────────────────

def test_granted_round_trips(repo):
    r, _ = repo
    g = frozenset({Capability.CHAT_SEND, Capability.FILES, Capability.SCREEN})
    r.save(_task(granted=g))
    assert r.load("t1").granted == g


def test_granted_is_stored_as_names_not_a_bitmask(repo):
    """A bitmask silently re-maps if the enum's member order changes, which is
    the quiet re-grant the explicit-literal discipline exists to prevent. A
    database someone opens by hand should also say what it means."""
    r, db = repo
    r.save(_task(granted=frozenset({Capability.FILES, Capability.CHAT_SEND})))
    raw = db.fetchone("SELECT granted FROM tasks WHERE task_id='t1'")["granted"]
    assert raw == "chat_send,files", (
        f"granted stored as {raw!r}. Sorted names, so the column is stable "
        f"across enum reordering and readable without this code."
    )


def test_an_unknown_stored_capability_is_a_load_error(repo):
    """**The important one.** Skipping an unrecognised name would load the Task
    with LESS authority than it was created with -- software that looks fine
    while a security record has shrunk. Refuse instead."""
    r, db = repo
    r.save(_task())
    db.execute("UPDATE tasks SET granted = ? WHERE task_id = 't1'",
               ("chat_send,a_capability_from_the_future",))
    db.commit()

    with pytest.raises(UnknownCapabilityStored):
        r.load("t1")


def test_an_empty_grant_set_loads_as_empty_not_as_an_error(repo):
    """Distinct from the case above. Empty is a legitimate recorded state --
    it refuses everything, which is the fail-closed direction -- whereas an
    unknown name means the record cannot be reconstructed."""
    assert _load_grants("") == frozenset()


def test_saving_again_cannot_rewrite_who_owns_the_task(repo):
    """`principal` and `granted` are absent from the upsert's update clause.
    Authority is recorded once, at creation; a later save updates status and
    parameters and must not be able to launder ownership."""
    r, _ = repo
    r.save(_task())
    r.save(_task(principal="device:attacker",
                 granted=frozenset(Capability),
                 status=TaskStatus.RUNNING))

    back = r.load("t1")
    assert back.principal == "device:phone", "the owner was rewritten by a save"
    assert Capability.EXECUTE not in back.granted, (
        "a save widened the recorded authority"
    )
    assert back.status is TaskStatus.RUNNING, "the status did not update"


# ─── the rest of the Task ────────────────────────────────────────────────────

def test_constraints_survive_byte_for_byte(repo):
    """User-pinned values are HARD constraints -- "mobile as 99999" is never
    silently substituted, and a round trip through storage is a place it could
    be."""
    r, _ = repo
    r.save(_task(constraints={"mobile": "99999", "name": "  spaced  "}))
    assert r.load("t1").constraints == {"mobile": "99999", "name": "  spaced  "}


def test_a_steps_verdict_survives(repo):
    """Including `UNVERIFIED`, which is the member that distinguishes "chose
    not to look" from "looked and could not tell". Losing it in storage would
    collapse them again."""
    r, _ = repo
    r.save(_task(steps=(
        TaskStep(step_id=1, intent="file_task", goal="delete x",
                 verdict=Verdict(
                     outcome=Outcome.UNVERIFIED,
                     observation=Observation(
                         kind=ObservationKind.NOTHING_CHANGED,
                         detail="verification disabled"),
                     tier="skipped")),
    )))
    step = r.load("t1").steps[0]
    assert step.verdict.outcome is Outcome.UNVERIFIED
    assert step.verdict.tier == "skipped"
    assert step.verdict.observation.detail == "verification disabled"


def test_steps_are_replaced_not_merged(repo):
    """A step list is a plan. A plan that half-updates is worse than one that
    is replaced wholesale."""
    r, _ = repo
    r.save(_task(steps=(
        TaskStep(step_id=1, intent="a"), TaskStep(step_id=2, intent="b"),
    )))
    r.save(_task(steps=(TaskStep(step_id=1, intent="c"),)))

    steps = r.load("t1").steps
    assert len(steps) == 1 and steps[0].intent == "c", (
        f"stale steps survived the rewrite: {[s.intent for s in steps]}"
    )


def test_a_missing_task_loads_as_none(repo):
    r, _ = repo
    assert r.load("never-existed") is None


# ─── finding what is still open ──────────────────────────────────────────────

def test_open_for_lists_only_unfinished_tasks_of_that_principal(repo):
    r, _ = repo
    r.save(_task(task_id="open1", status=TaskStatus.PENDING))
    r.save(_task(task_id="open2", status=TaskStatus.SUSPENDED_NEEDS_AUTHORITY))
    r.save(_task(task_id="done", status=TaskStatus.SUCCEEDED))
    r.save(_task(task_id="cancelled", status=TaskStatus.CANCELLED))
    r.save(_task(task_id="theirs", principal="device:other"))

    got = set(r.open_for("device:phone"))
    assert got == {"open1", "open2"}, (
        f"open_for returned {got}. Terminal tasks and other principals' tasks "
        f"must not appear -- a resume sweep over them is wasted work at best."
    )


def test_open_for_does_not_filter_by_authority(repo):
    """The repo returns what was stored; `brain/authority.may_resume` decides
    whether any of it may run. One place decides, and it is not this one --
    a repo that also judged would be a second source of truth about
    resumption."""
    r, _ = repo
    r.save(_task(task_id="x", granted=frozenset()))   # authority nobody can use
    assert r.open_for("device:phone") == ["x"]


def test_delete_removes_the_steps_too(repo):
    """There is deliberately no foreign key: a cascade on a task would discard
    the step history that explains what it did. So deletion is explicit, and
    has to actually clean up."""
    r, db = repo
    r.save(_task(steps=(TaskStep(step_id=1, intent="a"),)))
    r.delete("t1")
    assert r.load("t1") is None
    left = db.fetchall("SELECT * FROM task_steps WHERE task_id = 't1'")
    assert not left, "steps were orphaned by the delete"
