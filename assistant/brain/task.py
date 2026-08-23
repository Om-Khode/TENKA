"""What TENKA has committed to doing, and what came of it.

The contracts, and nothing else. No coordinator, no dispatch, no I/O — those
are later phases. This module imports `core/` and `config` only, so a test can
exercise a Task without a database or a running assistant.

Three things here are deliberately not what the source documents proposed, and
the reasons are load-bearing:

**A Task carries its own authority.** `principal` and `granted` are required
fields, not metadata. The plan those documents describe has Tasks surviving
restart with neither, which in this tree is either a dead feature (an unset
grant set refuses everything) or a privilege-escalation path (a background
runner resuming a remote device's work with `LOCAL_GRANTS`). See
`brain/authority.py`.

**A Task carries an `intent`.** The Brain reasons in affordances and dispatches
through intents, because `core/intent_capabilities.py` is keyed by intent and is
the only working security control in the tree. Changing the key domain of that
control during an architecture refactor is the failure this project has already
paid for. The intent is the execution ABI; the affordance is the vocabulary.

**`Outcome` has five members, not four.** `UNVERIFIED` is the difference
between "the operator chose not to look" and "we looked and could not tell".
Collapsing them makes `VERIFY_ENABLED=False` turn every task uncertain, which
is a rule nobody adopts.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, replace

from ..core.capabilities import Capability


# ─── What came of a step ─────────────────────────────────────────────────────

class Outcome(str, enum.Enum):
    """A step's verdict. Replaces the boolean that made three states look like
    one (`VerifyResult.ok` is True for success, ambiguity and skip alike).

    The distinction between `UNCERTAIN` and `UNVERIFIED` is the whole reason
    this is an enum: one is a failure of knowledge, the other is a choice not
    to acquire it, and only the first is a problem.
    """

    SUCCEEDED = "succeeded"      # positive evidence the effect happened
    FAILED = "failed"            # positive evidence it did not
    UNCERTAIN = "uncertain"      # verification ran and could not decide
    UNVERIFIED = "unverified"    # verification did not run: policy, or nothing to verify
    UNSUPPORTED = "unsupported"  # no route exists; never attempted at all

    @property
    def is_evidence_of_success(self) -> bool:
        """Only `SUCCEEDED` is. Absence of an exception is not evidence.

        A property rather than a bare comparison so there is one place that
        answers it — the bug this type replaces was six call sites each
        deciding for themselves what `ok` meant.
        """
        return self is Outcome.SUCCEEDED


class ObservationKind(str, enum.Enum):
    STATE_CHANGED = "state_changed"
    EXPECTED_PRESENT = "expected_present"
    EXPECTED_ABSENT = "expected_absent"
    NOTHING_CHANGED = "nothing_changed"
    ERROR = "error"


@dataclass(frozen=True)
class Observation:
    """What was seen, by what, when, and how sure.

    Freshness and provenance are fields rather than hopes: an observation of
    the desktop is stale the moment it is taken, and a caller that cannot tell
    a code-tier check from a vision guess cannot weigh them differently. The
    thing this replaces is a bare string.
    """

    kind: ObservationKind
    detail: str = ""
    source: str = "code"          # code | dom | uia | vision | process | llm
    confidence: float = 1.0
    at: str = ""                  # ISO 8601; set by the observer, never inferred


@dataclass(frozen=True)
class Verdict:
    """An outcome plus the evidence for it. **There is no `ok` field.**

    `escalated` records that a cheaper tier was inconclusive and a dearer one
    ran, which is what makes the cost of a verification legible after the fact.
    """

    outcome: Outcome
    observation: Observation
    tier: str = "code"            # pre | code | vision | skipped
    escalated: bool = False


# ─── Where a Task is in its life ─────────────────────────────────────────────

class TaskStatus(str, enum.Enum):
    """Eleven states.

    `UNSUPPORTED` and `NEEDS_CLARIFICATION` come from the source documents' own
    failure semantics, which their lifecycle list omitted.
    `SUSPENDED_NEEDS_AUTHORITY` is what a Task becomes when it outlives the
    authority that created it — see `brain/authority.py`.

    Deliberately *not* the same type as `Outcome`. A step that was `UNVERIFIED`
    by operator policy does not make the Task unverified: the operator chose
    not to look, the telemetry records that, and the Task may still have
    succeeded. Collapsing the two types would force one of them to lie.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    UNSUPPORTED = "unsupported"
    NEEDS_CLARIFICATION = "needs_clarification"
    RECOVERING = "recovering"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    SUSPENDED_NEEDS_AUTHORITY = "suspended_needs_authority"


_TERMINAL = frozenset({
    TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.UNSUPPORTED,
    TaskStatus.CANCELLED,
})

# Legal transitions, as data. A dict rather than a chain of `if`s so the shape
# is readable and testable without executing anything.
#
# `SUCCEEDED` is reachable only from `RUNNING` or `RECOVERING`: a Task cannot
# succeed without having run. `CANCELLED` is reachable from every non-terminal
# state, because abort is (`core/abort.py` fires at loop boundaries, and the
# state it interrupts is whatever happened to be current).
# `SUSPENDED_NEEDS_AUTHORITY` leads only to `PENDING`, and only a qualifying
# turn may make that move.
_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({
        TaskStatus.RUNNING, TaskStatus.UNSUPPORTED,
        TaskStatus.NEEDS_CLARIFICATION, TaskStatus.PAUSED,
        TaskStatus.SUSPENDED_NEEDS_AUTHORITY, TaskStatus.CANCELLED,
    }),
    TaskStatus.RUNNING: frozenset({
        TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.UNCERTAIN,
        TaskStatus.RECOVERING, TaskStatus.NEEDS_CLARIFICATION,
        TaskStatus.PAUSED, TaskStatus.SUSPENDED_NEEDS_AUTHORITY,
        TaskStatus.CANCELLED, TaskStatus.UNSUPPORTED,
    }),
    TaskStatus.RECOVERING: frozenset({
        TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.UNCERTAIN,
        TaskStatus.RUNNING, TaskStatus.CANCELLED,
    }),
    TaskStatus.PAUSED: frozenset({
        TaskStatus.RUNNING, TaskStatus.CANCELLED,
        TaskStatus.SUSPENDED_NEEDS_AUTHORITY,
    }),
    TaskStatus.NEEDS_CLARIFICATION: frozenset({
        TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.CANCELLED,
    }),
    TaskStatus.UNCERTAIN: frozenset({
        TaskStatus.RUNNING, TaskStatus.RECOVERING, TaskStatus.CANCELLED,
    }),
    # Only a qualifying turn moves this, and only back to PENDING.
    TaskStatus.SUSPENDED_NEEDS_AUTHORITY: frozenset({
        TaskStatus.PENDING, TaskStatus.CANCELLED,
    }),
    # Terminal.
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.UNSUPPORTED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


def is_terminal(status: TaskStatus) -> bool:
    return status in _TERMINAL


def may_transition(current: TaskStatus, target: TaskStatus) -> bool:
    """May a Task move from `current` to `target`?

    Unknown states answer False. A status added to the enum without a row in
    `_TRANSITIONS` therefore goes nowhere rather than everywhere, and a test
    turns the omission into a failure — the same shape as
    `core/intent_capabilities.py`'s missing-row default.
    """
    return target in _TRANSITIONS.get(current, frozenset())


class AuthorityMissing(RuntimeError):
    """Raised when a Task is built without a caller's authority in scope.

    Not a refusal string: a refusal is something a caller is told, and this is
    a bug in the code that forgot to install the turn's grants. Loud, and at
    construction, so it cannot be mistaken for a permission decision.
    """


# ─── The work ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskStep:
    """One executable unit. Structured enough that an executor need not
    re-read the user's original sentence.

    `goal` survives as a payload field, not as the step's meaning. Deleting it
    would rewrite roughly forty files and six loops at once; what changes here
    is that the fields above it are what decide, and the string is data an
    adapter may read.

    `verdict` is per-step because the vision agent already tracks per-step
    confirmation state (`_TaskState.todo_list` carries `pending_visual_confirm`
    and `confirm_strikes`). A contract designed only against the planner's
    coarser `PlanStep` would not survive contact with it.
    """

    step_id: int
    intent: str
    affordance: str = ""
    operation: str = ""
    parameters: dict = field(default_factory=dict)
    depends_on: tuple[int, ...] = ()
    condition: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    goal: str = ""
    observation: Observation | None = None
    verdict: Verdict | None = None


@dataclass(frozen=True)
class Task:
    """What TENKA has committed to accomplishing, and on whose authority.

    Build with `Task.create()` — the bare constructor cannot check anything,
    and every field that must be checked is checked there.
    """

    task_id: str
    intent: str
    principal: str
    granted: frozenset[Capability]
    affordance: str = ""
    operation: str = ""
    parameters: dict = field(default_factory=dict)
    # User-pinned values are HARD constraints: "mobile as 99999" is never
    # silently substituted. Kept separate from `parameters` so a planner or an
    # adapter cannot quietly widen one.
    constraints: dict = field(default_factory=dict)
    source: str = ""              # stt | console | studio | monitor | schedule
    created_at: str = ""
    expected: str = ""
    context_ref: str | None = None  # a key, never an inlined context blob
    status: TaskStatus = TaskStatus.PENDING
    parent: str | None = None
    steps: tuple[TaskStep, ...] = ()

    # ── construction ──

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def with_status(self, target: TaskStatus) -> "Task":
        """Return a copy in `target`, or raise if the move is illegal.

        Frozen dataclass, so a transition is a new object. That is deliberate:
        an illegal move cannot be half-applied, and a caller holding the old
        Task still holds a truthful one.
        """
        if not may_transition(self.status, target):
            raise ValueError(
                f"illegal task transition {self.status.value} -> {target.value}"
            )
        return replace(self, status=target)

    def requires(self) -> Capability:
        """What this Task costs, from the intent table.

        Read here rather than stored, so a Task cannot carry a stale answer if
        the classification changes. An unlisted intent costs `EXECUTE`, the
        same fail-closed default dispatch uses.
        """
        from ..core.intent_capabilities import DEFAULT_REQUIRED, REQUIRED_CAPABILITY
        return REQUIRED_CAPABILITY.get(self.intent, DEFAULT_REQUIRED)
