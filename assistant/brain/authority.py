"""On whose authority a Task was created, and whether it may still run.

The source documents say *"meaningful Tasks should survive restart"* and list a
Task's fields without a principal or a grant set. In this tree that is not an
omission, it is a fork:

* resumed with `current_grants` unset, every step is refused — the feature is
  dead on arrival; or
* resumed with `LOCAL_GRANTS` by a background runner, on the same argument
  `scheduler.py` and `event_bus.py` already use — and a tunnel-paired phone can
  bank work that later executes with full keyboard privilege.

The second is KI-24's shape (a re-arm losing its principal on a fresh thread)
and KI-30's shape (a temporary raise outliving itself), reintroduced as a
headline feature. So authority is recorded on the Task, and resumption is
gated on a live turn.

Five rules. Each is a function here, and each has a test that reds when it is
deleted — a rule with no test is a paragraph.
"""
from __future__ import annotations

from ..core.capabilities import Capability
from ..core.principal import current_principal
from .task import AuthorityMissing, Task, TaskStatus


def _live_grants() -> "frozenset[Capability] | None":
    """This turn's effective grant set, or `None` if nobody said.

    Imported lazily from `actions/` rather than at module scope. `actions/`
    imports the whole handler tree, and `brain/task.py` deliberately depends on
    `core/` alone so a Task can be built in a test with no assistant running.
    Keeping the heavy import inside the function preserves that.
    """
    from ..actions import current_grants
    return current_grants.get()


# ─── A1 + A2 — creation ──────────────────────────────────────────────────────

def create_task(
    *,
    intent: str,
    affordance: str = "",
    operation: str = "",
    parameters: dict | None = None,
    constraints: dict | None = None,
    source: str = "",
    expected: str = "",
    context_ref: str | None = None,
    parent: str | None = None,
    created_at: str = "",
) -> Task:
    """Build a Task, recording who asked and what they were allowed to ask.

    **A1 — authority is recorded, never inferred.** `granted` is
    `current_grants.get()` — the already-narrowed effective set, not the
    device's vault grants and not the transport ceiling. `principal` is
    `current_principal.get()`. Neither is a parameter, because a parameter is
    something a caller can get wrong and this is the one fact the whole rule
    rests on.

    Raises `AuthorityMissing` when either is unset. Not a refusal string: a
    refusal is something a caller is told, and an unset contextvar is a bug in
    the code that forgot to install the turn's authority. Loud, at
    construction, so it cannot be mistaken for a permission decision.

    **A2 — creation is gated as tightly as execution.** A Task may not be
    created for an intent the creating turn could not execute right now, using
    the same predicate dispatch uses. Banking work you are not allowed to do is
    not allowed, and without this a caller could create Tasks freely and rely
    on a later, wider turn to run them.

    Returns the Task. Raises `PermissionError` carrying the refusal sentence
    when A2 refuses — the sentence is the one a caller would have been told at
    dispatch, so the wording stays in one place.
    """
    from ..actions import capability_refusal

    granted = _live_grants()
    principal = current_principal.get()

    if granted is None:
        raise AuthorityMissing(
            "cannot create a Task: no grant set is installed for this turn. "
            "An unset grant set refuses everything at dispatch, so a Task "
            "built here could never run -- install the turn's authority first."
        )
    if not principal:
        raise AuthorityMissing(
            "cannot create a Task: no principal is installed for this turn. "
            "A Task with no owner can be resumed by nobody, which reads as a "
            "timeout rather than as the bug it is."
        )

    task = Task(
        task_id=Task.new_id(),
        intent=intent,
        principal=principal,
        granted=frozenset(granted),
        affordance=affordance,
        operation=operation,
        parameters=dict(parameters or {}),
        constraints=dict(constraints or {}),
        source=source,
        created_at=created_at,
        expected=expected,
        context_ref=context_ref,
        parent=parent,
    )

    refusal = capability_refusal(task.requires())
    if refusal is not None:
        raise PermissionError(refusal)

    return task


# ─── A3 + A4 — resumption ────────────────────────────────────────────────────

def resume_grants(task: Task) -> frozenset[Capability]:
    """**A3 — stored authority is a ceiling, never a source.**

    `task.granted & current_grants.get()`. The same shape as
    `io/api/policy.py:effective()`: an intersection that can only narrow. A
    stored `EXECUTE` grants nothing unless the *live* turn also holds it, so a
    Task cannot be a saved copy of a privilege that has since been revoked or
    has expired with a raise.

    Empty when no turn is in scope, which refuses everything downstream.
    """
    live = _live_grants()
    if live is None:
        return frozenset()
    return task.granted & live


def may_resume(task: Task) -> str | None:
    """**A4 — resumption needs a live, authorised, principal-matching turn.**

    Returns `None` when the Task may resume, or a short reason when it may not.
    A reason rather than a bool because the four conditions fail for different
    causes and an operator staring at a stalled Task needs to know which.

    All four fail closed:

    1. a grant set is installed. `None` means nobody said, and nobody said
       refuses.
    2. the principal matches. A Task belongs to whoever asked for it — the
       same rule `PendingState.owned_by` applies to a confirmation, and for the
       same reason (KI-13: a device holding `FILES` answering a file
       confirmation the operator armed).
    3. a device principal still exists. A revoked device's banked work does not
       resume because the credential is gone; the check is deferred to the
       caller through `device_is_live`, since `brain/` may not import
       `io/api/`.
    4. the live intersection still covers what the intent costs. This is what
       makes a raise's expiry actually bite: the stored set said `EXECUTE`, the
       live one does not, so the work stops.
    """
    live = _live_grants()
    if live is None:
        return "no grant set is installed for this turn"

    who = current_principal.get()
    if not who:
        return "no principal is installed for this turn"
    if who != task.principal:
        return f"this turn belongs to a different caller than the task does"

    effective = resume_grants(task)
    required = task.requires()
    if required not in effective:
        return (
            f"{required.value} is no longer held: the task recorded it, this "
            f"turn does not carry it"
        )
    return None


def device_is_live(task: Task, known_devices: "set[str] | None") -> bool:
    """Condition 3, asked of a caller that can see the vault.

    `brain/` may not import `io/api/`, so the device list arrives as data. A
    `None` list means the caller could not answer — treated as *not live*,
    because a resumption whose authorisation cannot be confirmed is exactly the
    case to refuse.

    A local task is always live: `LOCAL_PRINCIPAL` is not a credential that can
    be revoked.
    """
    if not task.principal.startswith("device:"):
        return True
    if known_devices is None:
        return False
    return task.principal.removeprefix("device:") in known_devices


# ─── A5 — no background resumption ───────────────────────────────────────────

def assert_resumable_by_a_turn(task: Task) -> Task:
    """**A5 — a background runner may not resume a Task.**

    The scheduler, the event bus and the notification flusher may *enqueue* a
    turn; they may not resume. This is the rule KI-30 exists to protect: those
    three install `LOCAL_GRANTS` on the argument that whoever installed the
    trigger held `EXECUTE`, and a resumable Task would let that argument launder
    a remote caller's banked work into full local privilege.

    Enforced by source, not convention: `tests/test_brain_authority.py` walks
    `scheduler.py` and `automation/event_bus.py` and fails if either reaches
    this module at all. Nothing here can detect its own caller reliably, so the
    structural test is the real control and this function is the documented
    door it guards.

    Returns the Task moved to `PENDING`, or moves it to
    `SUSPENDED_NEEDS_AUTHORITY` and returns that when `may_resume` refuses.
    Never raises: a Task that cannot resume is a state, not an error.
    """
    reason = may_resume(task)
    if reason is None:
        if task.status is TaskStatus.SUSPENDED_NEEDS_AUTHORITY:
            return task.with_status(TaskStatus.PENDING)
        return task
    if task.status is TaskStatus.SUSPENDED_NEEDS_AUTHORITY:
        return task
    return task.with_status(TaskStatus.SUSPENDED_NEEDS_AUTHORITY)
