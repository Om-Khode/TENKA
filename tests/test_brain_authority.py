"""A Task carries its own authority, and cannot outlive it.

The source documents say *"meaningful Tasks should survive restart"* and give a
Task no principal and no grant set. In this tree that is a fork, not an
omission: resumed with `current_grants` unset every step is refused, and
resumed with `LOCAL_GRANTS` by a background runner a tunnel-paired phone can
bank work that later executes with full keyboard privilege. The second is
KI-24's shape and KI-30's shape at once.

So `brain/authority.py` has five rules. Each is tested here, and each test reds
when its rule is deleted — the mutation list is in the commit that added them.

Both directions throughout. A module that refused every Task would satisfy
every "does not escalate" assertion while making the whole feature useless, so
the permitted path is pinned as hard as the refused one.

Run with:  py -3.11 -m pytest tests/test_brain_authority.py -v
"""
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.brain import authority  # noqa: E402
from assistant.brain.task import (  # noqa: E402
    AuthorityMissing, Task, TaskStatus, may_transition,
)
from assistant.core.capabilities import Capability  # noqa: E402

# What a tunnel ceiling actually carries (io/api/policy.py). EXECUTE and
# SYSTEM_CONTROL are the two it omits.
TUNNEL = frozenset({
    Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
    Capability.SCREEN, Capability.FILES,
})


@pytest.fixture()
def turn():
    """Install a turn's authority the way main.py does, and clear it after.

    Reset by assignment rather than by token: a token minted in a sync fixture
    cannot be reset from inside a coroutine, and the fail-closed defaults are
    what a leak should land on anyway.
    """
    import assistant.actions as actions

    def _install(grants, principal="device:phone"):
        actions.set_principal(principal)
        actions.set_grants(grants) if grants is not None else \
            actions.current_grants.set(None)

    yield _install

    actions.current_grants.set(None)
    actions.set_principal(None)


def _task(intent="get_time", granted=None, principal="device:phone"):
    """A Task built directly, bypassing the creation gate — for testing the
    resume rules against a Task that already exists."""
    return Task(
        task_id=Task.new_id(), intent=intent, principal=principal,
        granted=frozenset(granted if granted is not None else {Capability.CHAT_SEND}),
    )


# ─── A1 — authority is recorded, never inferred ──────────────────────────────

def test_a_task_records_the_turns_grants_and_principal(turn):
    turn(TUNNEL, principal="device:phone")
    t = authority.create_task(intent="get_time")

    assert t.principal == "device:phone"
    assert t.granted == TUNNEL, (
        "the Task did not record the effective grant set. Without it there is "
        "nothing for the resume rule to intersect."
    )


def test_creation_without_a_grant_set_raises_rather_than_defaulting(turn):
    """`None` means nobody said. A Task built then could never run, so this is
    a bug in the caller and must be loud at construction — not a refusal, which
    is something a caller is *told*."""
    turn(None)
    with pytest.raises(AuthorityMissing):
        authority.create_task(intent="get_time")


def test_creation_without_a_principal_raises(turn):
    """A Task with no owner can be resumed by nobody, which reads as a timeout
    rather than as the bug it is — the same failure `PendingState.owned_by`
    exists to prevent."""
    turn(TUNNEL, principal=None)
    with pytest.raises(AuthorityMissing):
        authority.create_task(intent="get_time")


def test_the_recorded_grants_are_a_snapshot_not_a_live_view(turn):
    """`frozenset(granted)`, so a later change to the turn's grants cannot
    retroactively widen a Task that was already created."""
    import assistant.actions as actions
    turn(TUNNEL)
    t = authority.create_task(intent="get_time")
    actions.set_grants(frozenset(Capability))
    assert t.granted == TUNNEL, "the Task's authority followed the turn's"


# ─── A2 — creation is gated as tightly as execution ──────────────────────────

def test_a_task_cannot_be_created_for_work_the_turn_cannot_do(turn):
    """Banking work you are not allowed to do is not allowed. Without this a
    caller could create Tasks freely and rely on a later, wider turn."""
    turn(TUNNEL)   # no EXECUTE
    with pytest.raises(PermissionError):
        authority.create_task(intent="code_executor")


def test_a_task_can_be_created_for_work_the_turn_can_do(turn):
    """The other half. A gate that refused everything would pass the test
    above and stop the Brain creating any Task at all."""
    turn(TUNNEL)
    t = authority.create_task(intent="get_time")   # costs CHAT_SEND
    assert t.intent == "get_time"


def test_an_unclassified_intent_costs_execute(turn):
    """The fail-closed default, one layer up from dispatch. If the two layers
    disagreed about a new intent, the Brain would create Tasks dispatch then
    refuses."""
    turn(TUNNEL)
    t = _task(intent="an_intent_nobody_classified")
    assert t.requires() is Capability.EXECUTE


# ─── A3 — stored authority is a ceiling, never a source ──────────────────────

def test_resume_grants_can_only_narrow(turn):
    """Same shape as `io/api/policy.py:effective()`. A stored EXECUTE grants
    nothing unless the live turn also holds it — which is what makes a raise's
    expiry actually bite."""
    turn(TUNNEL)
    t = _task(granted=TUNNEL | {Capability.EXECUTE})

    got = authority.resume_grants(t)
    assert Capability.EXECUTE not in got, (
        "a stored EXECUTE survived into a turn that does not hold it. The Task "
        "would be a saved copy of an expired privilege."
    )
    assert got == TUNNEL


def test_resume_grants_is_empty_with_no_turn_in_scope(turn):
    turn(None)
    assert authority.resume_grants(_task(granted=TUNNEL)) == frozenset()


def test_resume_grants_never_exceeds_either_input(turn):
    turn(frozenset({Capability.CHAT_SEND, Capability.FILES}))
    t = _task(granted=frozenset({Capability.CHAT_SEND, Capability.SCREEN}))
    got = authority.resume_grants(t)
    assert got <= t.granted and got <= frozenset(
        {Capability.CHAT_SEND, Capability.FILES})
    assert got == frozenset({Capability.CHAT_SEND})


# ─── A4 — resumption needs a live, matching turn ─────────────────────────────

def test_a_task_resumes_for_its_own_principal(turn):
    turn(TUNNEL, principal="device:phone")
    assert authority.may_resume(_task(principal="device:phone")) is None


def test_a_task_does_not_resume_for_a_different_principal(turn):
    """The rule `PendingState.owned_by` applies to a confirmation, applied to
    banked work. KI-13 is what happens without it."""
    turn(TUNNEL, principal="device:other")
    assert authority.may_resume(_task(principal="device:phone")) is not None


def test_a_task_does_not_resume_for_the_local_caller_either(turn):
    """`local` is one principal among several, not a superuser. A Task a phone
    created is not the operator's to resume."""
    turn(frozenset(Capability), principal="local")
    assert authority.may_resume(_task(principal="device:phone")) is not None


def test_a_task_does_not_resume_with_no_grants(turn):
    turn(None, principal="device:phone")
    assert authority.may_resume(_task(principal="device:phone")) is not None


def test_a_task_does_not_resume_when_the_capability_is_gone(turn):
    """The raise-expiry case, stated as a test. The Task recorded EXECUTE; this
    turn does not carry it; the work stops."""
    turn(TUNNEL, principal="device:phone")
    t = _task(intent="code_executor",
              granted=TUNNEL | {Capability.EXECUTE},
              principal="device:phone")
    reason = authority.may_resume(t)
    assert reason and "execute" in reason.lower(), (
        f"the reason does not name what is missing: {reason!r}"
    )


@pytest.mark.parametrize("known,expected", [
    ({"phone"}, True),
    ({"other"}, False),
    (set(), False),
    (None, False),
])
def test_a_revoked_device_does_not_resume(known, expected):
    """Condition 3. `None` means the caller could not answer, which is treated
    as not-live: a resumption whose authorisation cannot be confirmed is
    exactly the case to refuse."""
    t = _task(principal="device:phone")
    assert authority.device_is_live(t, known) is expected


def test_a_local_task_is_always_live():
    """`LOCAL_PRINCIPAL` is not a credential that can be revoked."""
    assert authority.device_is_live(_task(principal="local"), None) is True


# ─── A5 — no background resumption ───────────────────────────────────────────

def test_the_background_runners_cannot_reach_the_resume_path():
    """**Structural, and the real control.** Nothing can reliably detect its own
    caller, so the rule is enforced by what the background runners cannot
    reach: `scheduler.py` and `automation/event_bus.py` install `LOCAL_GRANTS`
    on the argument that whoever installed the trigger held `EXECUTE`, and a
    resumable Task would launder a remote caller's banked work into full local
    privilege through exactly that argument.

    **This check was narrowed, and here is the accounting.** It used to forbid
    importing *anything* from `brain` -- blunt, and effective precisely because
    it needed no judgement. P4a moved authority installation into
    `brain/turn.py` so that one implementation of the install order exists
    instead of two that disagreed, which means `scheduler.py` now imports from
    `brain` and the blunt version is no longer available.

    What the blunt version was incidentally holding up, enumerated rather than
    waved at (`CLAUDE.md` process rule 10):

    1. no reach to `authority.may_resume` -- **still held**, forbidden below by
       name;
    2. no reach to `Task` construction, so a background runner could not create
       something that later resumes -- **still held**: `brain/task.py` is
       forbidden below too, and `brain/__init__.py` imports neither module, so
       `from .brain.turn import run_turn` binds no path to either;
    3. no reach to anything else in `brain` -- **given up**. `brain/turn.py` is
       reachable, by design. It installs contextvars and awaits a callable; it
       decides nothing and imports neither of the two modules above.

    So the property is unchanged and its enforcement is one step less blunt. A
    new module in `brain/` that can construct or resume a Task has to be added
    to `_FORBIDDEN` here, and the test's own docstring is the reminder.
    """
    _FORBIDDEN = ("authority", "task")
    _ALLOWED = ("turn",)

    for name in ("scheduler.py", "automation/event_bus.py"):
        src = (_ROOT / "assistant" / name).read_text(encoding="utf-8")
        hits = [
            line.strip() for line in src.splitlines()
            if re.search(
                r"brain[.\s]*(?:import\s+|\.)\s*(?:" + "|".join(_FORBIDDEN) + r")\b|"
                r"from\s+\.*brain\s+import\s+(?:" + "|".join(_FORBIDDEN) + r")\b",
                line)
        ]
        assert not hits, (
            f"{name} reaches brain/{{{','.join(_FORBIDDEN)}}}: {hits}. A "
            f"background runner may enqueue a turn; it may not construct or "
            f"resume a Task."
        )

    # The narrowing is only sound while the two reachable modules stay inert.
    # If either imports the machinery it is allowed past, the distinction above
    # is decorative -- so both are checked by AST rather than by reading the
    # text. The first version of this grepped the source and failed on
    # `turn.py`'s own docstring, which says the words "resume a Task" while
    # importing neither.
    import ast

    def _imported_names(rel: str) -> "set[str]":
        src = (_ROOT / "assistant" / rel).read_text(encoding="utf-8")
        names: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[-1] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module.split(".")[-1])
                names.update(a.name for a in node.names)
        return names

    for rel in ("brain/turn.py", "brain/__init__.py"):
        reached = _imported_names(rel) & set(_FORBIDDEN)
        assert not reached, (
            f"{rel} imports brain/{reached}, which hands the background "
            f"runners the path this test just narrowed"
        )

    assert _ALLOWED, "the allow-list is empty -- nothing is being permitted"


def test_an_unresumable_task_becomes_suspended_rather_than_failing(turn):
    """A Task that cannot resume is a state, not an error. Failing it would
    lose work the operator could still authorise by speaking."""
    turn(TUNNEL, principal="device:other")
    t = _task(principal="device:phone")
    out = authority.assert_resumable_by_a_turn(t)
    assert out.status is TaskStatus.SUSPENDED_NEEDS_AUTHORITY


def test_a_suspended_task_returns_to_pending_for_a_qualifying_turn(turn):
    """And the way back. Suspension is not a grave."""
    turn(TUNNEL, principal="device:phone")
    t = _task(principal="device:phone").with_status(
        TaskStatus.SUSPENDED_NEEDS_AUTHORITY)
    out = authority.assert_resumable_by_a_turn(t)
    assert out.status is TaskStatus.PENDING


def test_a_suspended_task_stays_suspended_for_a_foreign_turn(turn):
    turn(TUNNEL, principal="device:other")
    t = _task(principal="device:phone").with_status(
        TaskStatus.SUSPENDED_NEEDS_AUTHORITY)
    assert authority.assert_resumable_by_a_turn(t).status is \
        TaskStatus.SUSPENDED_NEEDS_AUTHORITY


# ─── the state machine ───────────────────────────────────────────────────────

def test_success_requires_having_run():
    """`SUCCEEDED` is reachable only from `RUNNING` or `RECOVERING`."""
    assert may_transition(TaskStatus.RUNNING, TaskStatus.SUCCEEDED)
    assert may_transition(TaskStatus.RECOVERING, TaskStatus.SUCCEEDED)
    assert not may_transition(TaskStatus.PENDING, TaskStatus.SUCCEEDED)


def test_abort_reaches_cancelled_from_every_non_terminal_state():
    """`core/abort.py` fires at loop boundaries and the state it interrupts is
    whatever happened to be current."""
    from assistant.brain.task import is_terminal
    for s in TaskStatus:
        if is_terminal(s) or s is TaskStatus.CANCELLED:
            continue
        assert may_transition(s, TaskStatus.CANCELLED), (
            f"abort cannot cancel a task in {s.value}"
        )


def test_a_terminal_state_goes_nowhere():
    from assistant.brain.task import is_terminal
    for s in TaskStatus:
        if not is_terminal(s):
            continue
        for target in TaskStatus:
            assert not may_transition(s, target), (
                f"{s.value} is terminal but transitions to {target.value}"
            )


def test_an_illegal_transition_raises_rather_than_being_applied():
    t = _task()
    with pytest.raises(ValueError):
        t.with_status(TaskStatus.SUCCEEDED)
    assert t.status is TaskStatus.PENDING, "the task was mutated by a failed move"


def test_every_status_has_a_transition_row():
    """Anti-vacuity for `may_transition`: a status added to the enum without a
    row answers False for everything, which is safe but silent. This makes the
    omission loud, the same way a missing `REQUIRED_CAPABILITY` row is."""
    from assistant.brain.task import _TRANSITIONS
    missing = [s.value for s in TaskStatus if s not in _TRANSITIONS]
    assert not missing, f"statuses with no transition row: {missing}"


# ─── vocabulary ──────────────────────────────────────────────────────────────

# Names in `brain/` that legitimately mean the security enum or something
# built directly on it. Anything else containing "capabilit" is the collision
# the affordance vocabulary exists to prevent.
_SECURITY_VOCABULARY = frozenset({
    "Capability",                    # core/capabilities.py, the enum
    "capabilities",                  # the module
    "intent_capabilities",           # the intent -> capability table's module
    "REQUIRED_CAPABILITY",           # that table
    "DEFAULT_REQUIRED",              # its fail-closed default
    "capability_refusal",            # the single predicate
    "durable_capability_refusal",    # its durability sibling (KI-30)
})


def test_the_brain_package_does_not_say_capability_for_affordance():
    """The word collision that made the source documents look implementable.
    `Capability` is the security enum; what TENKA can do is an affordance.

    Checked over **code tokens only** -- docstrings and comments discuss the
    collision at length and must be allowed to. The first version of this test
    read raw lines and failed on `brain/__init__.py`'s own explanation of the
    rule, which is a check pointing one step to the side of its property.

    The only permitted code use is the security enum itself.
    """
    import io
    import tokenize

    for f in sorted((_ROOT / "assistant" / "brain").glob("*.py")):
        src = f.read_text(encoding="utf-8")
        offenders = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            if tok.type != tokenize.NAME:
                continue
            if "capabilit" not in tok.string.lower():
                continue
            # The security surface, enumerated. Anything referring to
            # `core/capabilities.py`'s enum or the tables and predicates built
            # on it is the legitimate meaning of the word here. Listing them
            # rather than pattern-matching means a NEW loose use is flagged
            # while the five existing security references are not -- the first
            # version allowed only `Capability` itself and flagged
            # `capability_refusal`, which is the predicate that enforces it.
            if tok.string in _SECURITY_VOCABULARY:
                continue
            offenders.append(f"{f.name}:{tok.start[0]} {tok.string}")

        assert not offenders, (
            f"code in brain/ uses 'capability' for something other than the "
            f"security enum: {offenders}. What TENKA can do is an affordance; "
            f"the collision is what let the source documents propose re-keying "
            f"the only working security control without anyone noticing."
        )
