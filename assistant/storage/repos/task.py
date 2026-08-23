"""Repository for the tasks and task_steps tables (schema v22).

Storage only. Every authority decision lives in `brain/authority.py`; this
module's one opinion is about *serialisation*, and it is deliberate:

**`granted` round-trips as sorted capability names, never as an integer
bitmask.** A bitmask re-maps silently if the enum's member order ever changes,
which is the same class of quiet re-grant that `io/api/policy.py`'s
explicit-literal discipline exists to prevent. And a stored name the enum no
longer has is a **load error**, not a dropped member: a Task whose recorded
authority cannot be reconstructed exactly must refuse to load rather than load
narrower and look fine.

The repo does not filter by authority. `load` returns what was stored, and the
caller asks `brain/authority.may_resume` — one place decides, and it is not
this one.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from ...core.capabilities import Capability

if TYPE_CHECKING:
    from assistant.storage.db import Database

logger = logging.getLogger("task_repo")


class UnknownCapabilityStored(ValueError):
    """A stored `granted` names a capability this build does not have.

    Loud on purpose. The alternative -- skipping the unknown name -- loads the
    Task with *less* authority than it was created with, which looks like
    working software and is a silent narrowing of a security record. If the
    enum shrank deliberately, the migration that shrank it owns the rewrite.
    """


def _dump_grants(granted: "frozenset[Capability]") -> str:
    return ",".join(sorted(c.value for c in granted))


def _load_grants(raw: str) -> "frozenset[Capability]":
    if not raw:
        return frozenset()
    known = {c.value: c for c in Capability}
    out, unknown = set(), []
    for name in raw.split(","):
        name = name.strip()
        if not name:
            continue
        cap = known.get(name)
        if cap is None:
            unknown.append(name)
        else:
            out.add(cap)
    if unknown:
        raise UnknownCapabilityStored(
            f"stored task authority names capabilities this build does not "
            f"have: {unknown}. Refusing to load it with less authority than it "
            f"was created with."
        )
    return frozenset(out)


class TaskRepo:
    def __init__(self, db: "Database") -> None:
        self._db = db

    # ── write ──

    def save(self, task) -> None:
        """Upsert a task and replace its steps.

        Steps are deleted and rewritten rather than merged: a step list is a
        plan, and a plan that half-updates is worse than one that is replaced.
        """
        now = datetime.now().isoformat()
        self._db.execute(
            """INSERT INTO tasks
               (task_id, intent, principal, granted, affordance, operation,
                parameters, constraints, source, created_at, expected,
                context_ref, status, parent, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                 status = excluded.status,
                 parameters = excluded.parameters,
                 constraints = excluded.constraints,
                 expected = excluded.expected,
                 context_ref = excluded.context_ref,
                 updated_at = excluded.updated_at""",
            (task.task_id, task.intent, task.principal,
             _dump_grants(task.granted), task.affordance, task.operation,
             json.dumps(task.parameters), json.dumps(task.constraints),
             task.source, task.created_at, task.expected, task.context_ref,
             task.status.value, task.parent, now),
        )
        # `principal` and `granted` are deliberately NOT in the update clause.
        # Authority is recorded once, at creation, and a later save must not be
        # able to rewrite whose task it is or what it was allowed to do.
        self._db.execute("DELETE FROM task_steps WHERE task_id = ?", (task.task_id,))
        for st in task.steps:
            v = st.verdict
            o = st.observation or (v.observation if v else None)
            self._db.execute(
                """INSERT INTO task_steps
                   (task_id, step_id, intent, affordance, operation, parameters,
                    depends_on, condition, status, goal, outcome, observation,
                    tier, escalated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (task.task_id, st.step_id, st.intent, st.affordance,
                 st.operation, json.dumps(st.parameters),
                 ",".join(str(d) for d in st.depends_on), st.condition,
                 st.status.value, st.goal,
                 v.outcome.value if v else None,
                 o.detail if o else None,
                 v.tier if v else None,
                 int(bool(v.escalated)) if v else 0),
            )
        self._db.commit()
        logger.debug(f"[TASKS] saved {task.task_id} ({task.status.value})")

    # ── read ──

    def load(self, task_id: str):
        """Return the Task, or None. Raises `UnknownCapabilityStored` if its
        recorded authority cannot be reconstructed exactly."""
        from ...brain.task import Observation, ObservationKind, Outcome
        from ...brain.task import Task, TaskStatus, TaskStep, Verdict

        row = self._db.fetchone("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if row is None:
            return None

        steps = []
        for s in self._db.fetchall(
            "SELECT * FROM task_steps WHERE task_id = ? ORDER BY step_id",
            (task_id,),
        ):
            verdict = None
            if s["outcome"]:
                verdict = Verdict(
                    outcome=Outcome(s["outcome"]),
                    observation=Observation(
                        kind=ObservationKind.STATE_CHANGED,
                        detail=s["observation"] or "",
                    ),
                    tier=s["tier"] or "code",
                    escalated=bool(s["escalated"]),
                )
            steps.append(TaskStep(
                step_id=s["step_id"], intent=s["intent"],
                affordance=s["affordance"], operation=s["operation"],
                parameters=json.loads(s["parameters"] or "{}"),
                depends_on=tuple(
                    int(d) for d in (s["depends_on"] or "").split(",") if d
                ),
                condition=s["condition"],
                status=TaskStatus(s["status"]),
                goal=s["goal"], verdict=verdict,
            ))

        return Task(
            task_id=row["task_id"], intent=row["intent"],
            principal=row["principal"], granted=_load_grants(row["granted"]),
            affordance=row["affordance"], operation=row["operation"],
            parameters=json.loads(row["parameters"] or "{}"),
            constraints=json.loads(row["constraints"] or "{}"),
            source=row["source"], created_at=row["created_at"],
            expected=row["expected"], context_ref=row["context_ref"],
            status=TaskStatus(row["status"]), parent=row["parent"],
            steps=tuple(steps),
        )

    def open_for(self, principal: str) -> list[str]:
        """Task ids this principal has left unfinished, newest first.

        Ids rather than Tasks: whether any of them may actually resume is
        `brain/authority.may_resume`'s question, and loading a hundred Tasks to
        discard ninety-nine is the wrong shape for the answer.
        """
        rows = self._db.fetchall(
            "SELECT task_id FROM tasks WHERE principal = ? "
            "AND status NOT IN ('succeeded','failed','unsupported','cancelled') "
            "ORDER BY updated_at DESC",
            (principal,),
        )
        return [r["task_id"] for r in rows]

    def delete(self, task_id: str) -> None:
        self._db.execute("DELETE FROM task_steps WHERE task_id = ?", (task_id,))
        self._db.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        self._db.commit()
