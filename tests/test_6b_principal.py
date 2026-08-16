"""KI-13 — pending state gets an owner.

Milestone 6a.5 charged every pending row its own capability. That stops a
device that *lacks* the capability; it cannot stop a confused deputy that
*holds* it. A tunnel device legitimately holding FILES can answer a file
confirmation the person at the keyboard armed, because `PendingState` is
process-global and remembers only *what* was asked, never *who* was asked.

6b opens three internet transports, which widens "the device that answers"
from "a local process" to "any paired device, anywhere", so this closes here.

The mechanism is the pair `current_grants` already established:

1. `current_principal` -- a contextvar set alongside `current_grants`, `None`
   by default, and `None` owns nothing. The absence of a decision is not a
   decision to allow.
2. `PendingState.set(payload, principal=...)` records the owner at arm time,
   and `owned_by()` compares at answer time. Both sides must be set and equal.
3. `test_every_arming_site_records_a_principal` below, which walks `main.py`'s
   AST rather than trusting a hand-maintained list -- a list is exactly what
   the next arming site forgets, and a forgotten arming site is how KI-13's
   sibling bug was written in the first place.

Seven of these tests assert a refusal and exactly one --
`test_a_device_can_answer_its_own_confirmation` -- asserts that the ordinary
flow still works. That one is the point: a principal check that broke every
legitimate confirmation would pass the other seven.

Run with:  py -3.11 -m pytest tests/test_6b_principal.py -v
"""
import ast
import asyncio
import contextvars
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.capabilities import Capability

_MAIN_PY = _ROOT / "assistant" / "main.py"

_A_DEVICE = "device:phone"
_ANOTHER_DEVICE = "device:laptop"


def _remote_grants() -> "frozenset[Capability]":
    """Built from policy.py rather than hand-written, so this file tracks the
    ceiling if it moves."""
    from assistant.io.api.policy import POLICIES
    return frozenset(POLICIES["funnel"].ceiling)


def _local_grants() -> "frozenset[Capability]":
    from assistant.actions import LOCAL_GRANTS
    return LOCAL_GRANTS


# ─────────────────────────────────────────────────────────────────────────
# Harness: one real turn through main.py, with a real grant set AND a real
# principal. Only the edges are stubbed -- TTS, the bridge, the LLM, SQLite,
# telemetry. The dispatcher, the pending chain and the owner check are the
# real code.
#
# Deliberately a copy of `test_6a5_predispatch_gate.py`'s fixture rather than
# an import of it: that one takes no principal, and widening it would make a
# security file's harness depend on this one's needs.
# ─────────────────────────────────────────────────────────────────────────

class _FakeBridge:
    def __init__(self):
        self.commands = []

    async def send_command(self, name, **kw):
        self.commands.append((name, kw))


class _FakeTracker:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return None

    def save(self):
        pass


@pytest.fixture(autouse=True)
def _clean_pending():
    """Pending states are process-global -- that is the whole defect. Clear
    every one of them around each test so a leftover armed state cannot make
    a later test pass or fail for the wrong reason."""
    from assistant.pending import pending_registry

    def _clear_all():
        for name in pending_registry.names():
            pending_registry.get(name).clear()

    _clear_all()
    yield
    _clear_all()


@pytest.fixture
def turn(monkeypatch):
    from assistant import main as main_mod
    from assistant.intent import IntentResult

    spoken: list = []
    saved: list = []
    tracker_box: dict = {}

    def _make_tracker(**kw):
        t = _FakeTracker(**kw)
        tracker_box["t"] = t
        return t

    monkeypatch.setattr(main_mod._telemetry, "TurnTracker", _make_tracker)
    monkeypatch.setattr(main_mod._telemetry, "check_correction", lambda *a, **k: None)
    monkeypatch.setattr(main_mod._telemetry, "set_current_tracker",
                        lambda t: contextvars.ContextVar("x", default=None).set(None))
    monkeypatch.setattr(main_mod._telemetry, "reset_current_tracker", lambda t: None)

    import assistant.session as session_mod
    monkeypatch.setattr(session_mod, "get_current_session_id", lambda: "probe-session")
    monkeypatch.setattr(session_mod, "record_turn", lambda *a, **k: None)
    monkeypatch.setattr(main_mod.memory, "save_turn",
                        lambda *a, **k: saved.append((a, k)))
    # Needed by `_build_conversation_messages`, which a turn that falls all
    # the way through the pending chain does reach. Without it the ordinary
    # path dies on "init_memory() called before init_db()" and a skipped
    # non-owner would look identical to a refused one -- which is exactly the
    # distinction this file now has to be able to make.
    monkeypatch.setattr(main_mod.memory, "get_recent", lambda *a, **k: [])

    async def _speak(text, *a, **k):
        spoken.append(text)
    monkeypatch.setattr(main_mod.tts, "speak", _speak)

    async def _finish(*a, **k):
        return None
    monkeypatch.setattr(main_mod, "_finish_turn", _finish)
    monkeypatch.setattr(main_mod, "_publish_turn_status", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_wake_listener", None)

    async def _chat(*a, **k):
        return "ok"
    monkeypatch.setattr(main_mod.llm, "chat", _chat)

    class _Topic:
        def push_turn(self, *a, **k): pass
        def resolve_query(self, q): return q
        def get_topic_hint(self): return None
    monkeypatch.setattr(main_mod, "_get_topic_tracker", lambda: _Topic())

    monkeypatch.setattr(main_mod.procedures, "match_trigger", lambda t: None)
    monkeypatch.setattr(main_mod.procedures, "find_by_name_or_trigger",
                        lambda *a, **k: None)
    monkeypatch.setattr(main_mod.shortcuts, "match_shortcut", lambda t: None)
    monkeypatch.setattr(main_mod.regex_router, "match_procedure_command",
                        lambda t: None)
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    # `get_time`, not `small_talk`. A turn the pending chain SKIPS falls all
    # the way through to dispatch, and `small_talk` takes the streaming branch
    # -- which, with no stub in front of it, reaches the real providers over
    # the network. A tool intent ends at `execute_action`, which is stubbed
    # just below, so a skipped turn finishes locally and leaves a record of
    # having got there.
    async def _detect(*a, **k):
        return IntentResult(intent="get_time", response="ok", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)

    dispatched: list = []

    async def _execute_action(intent, params, llm_response="", bridge=None, **k):
        dispatched.append(intent)
        return "It is noon."
    monkeypatch.setattr(main_mod, "execute_action", _execute_action)

    class _Run:
        def __init__(self):
            self.spoken = spoken
            self.saved = saved
            self.dispatched = dispatched
            self.tracker = None

        async def __call__(self, text, grants, principal,
                           source="studio", bridge=None):
            bridge = bridge or _FakeBridge()
            await main_mod.process_text_from_queue(
                source, text, bridge, grants=grants, principal=principal)
            self.tracker = tracker_box.get("t")
            return self

    yield _Run()


def _arm(state_name: str, principal):
    """Arm one registered pending state as `principal` and return it."""
    from assistant.pending import pending_registry
    state = pending_registry.get(state_name)
    state.set({"op": "delete", "path": "downloads"}, principal=principal)
    return state


def _one_row(spy, state, capability=Capability.FILES):
    """A one-row `_PENDING_HANDLERS` table pointing at `state`.

    The real table's row shape is what the dispatch loop unpacks, so a change
    to it fails here rather than silently walking the wrong column.
    """
    return [(spy, "DESTRUCTIVE", "file_task", False, capability, state)]


def _spy():
    answered: list = []

    async def _handler(text):
        answered.append(text)
        return "Deleted."
    return answered, _handler


# ═════════════════════════════════════════════════════════════════════════
# The four ownership cases
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_locally_armed_confirmation_cannot_be_answered_by_a_device(
        turn, monkeypatch):
    """KI-13's headline case. The operator says "delete my downloads folder"
    at the keyboard, TENKA arms `pending_destructive` and waits for "yes",
    and a paired device that happens to hold FILES says it first.

    The capability check cannot see this: the device is *allowed* to delete
    files. It simply was not the one that was asked."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    answered, spy = _spy()
    state = _arm("destructive", LOCAL_PRINCIPAL)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    await turn("yes", _remote_grants(), _A_DEVICE)
    assert answered == [], (
        "a device answered a confirmation the person at the keyboard armed")


@pytest.mark.asyncio
async def test_a_device_armed_confirmation_cannot_be_answered_locally(
        turn, monkeypatch):
    """The other direction, and it must hold too. Ownership is not a ranking:
    `LOCAL_PRINCIPAL` is not a master key that can answer anyone's question,
    it is one principal among several. A confirmation armed from the phone is
    the phone's to answer, and a "yes" typed at the console is as much a
    different voice as a remote one is."""
    from assistant import main as main_mod

    answered, spy = _spy()
    state = _arm("destructive", _A_DEVICE)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    from assistant.actions import LOCAL_PRINCIPAL
    await turn("yes", _local_grants(), LOCAL_PRINCIPAL, source="stt")
    assert answered == [], (
        "a local 'yes' drove a confirmation a device armed")


@pytest.mark.asyncio
async def test_a_device_cannot_answer_another_devices_confirmation(
        turn, monkeypatch):
    """Two paired devices, both holding the same ceiling. Nothing about the
    capability set distinguishes them, which is exactly why the capability
    check cannot close this."""
    from assistant import main as main_mod

    answered, spy = _spy()
    state = _arm("destructive", _A_DEVICE)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    await turn("yes", _remote_grants(), _ANOTHER_DEVICE)
    assert answered == [], (
        "one device answered another device's confirmation")


@pytest.mark.asyncio
async def test_a_device_can_answer_its_own_confirmation(turn, monkeypatch):
    """THE test in this file. Seven others assert a refusal, and a check that
    refused every confirmation ever armed would pass all seven of them. This
    is the one that says the feature still works."""
    from assistant import main as main_mod

    answered, spy = _spy()
    state = _arm("destructive", _A_DEVICE)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    await turn("yes", _remote_grants(), _A_DEVICE)
    assert answered == ["yes"], (
        "the device that armed the confirmation could not answer it -- "
        "the owner check broke the ordinary flow")


# ═════════════════════════════════════════════════════════════════════════
# The refusal itself
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_mismatch_is_not_silently_ignored_the_owner_is_told(
        turn, monkeypatch):
    """A dropped confirmation reads as a timeout to whoever armed it and as
    nothing at all to whoever tried to answer. KI-13 asks for the opposite,
    and it names the audience: loud "so the operator sees that something else
    tried to answer".

    So the loudness is aimed at the OWNER, not the intruder. A sentence sent
    back to whoever tried is loud only in the conversation that already knows
    what it did, and silent in the one that needed to hear it; the operator
    would have been left with a line in `debug.log`. The attempt is parked on
    the state and collected when she next answers -- which is the ordinary
    course of answering a confirmation, so she cannot miss it.

    The notice is spoken, so it carries the constraints every spoken string in
    this tree carries -- under 120 characters, no paths, no error codes --
    plus one of its own: it must not name the device, because the thing that
    tried to answer is exactly what must not be disclosed onward."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    answered, spy = _spy()
    state = _arm("destructive", LOCAL_PRINCIPAL)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    # A device reaches for the operator's confirmation and is skipped...
    await turn("yes", _remote_grants(), _A_DEVICE)
    assert answered == []

    # ...and then the operator answers her own question and is told.
    out = await turn("yes", _local_grants(), LOCAL_PRINCIPAL, source="stt")
    assert answered == ["yes"], "the owner's own answer stopped working"

    said = [a[2] for a, _ in out.saved if len(a) > 2]
    assert said, f"nothing was saved for the owner's turn: {out.saved}"
    told = said[-1]
    assert main_mod._FOREIGN_ATTEMPT_NOTICE in told, (
        "the owner was never told that something else tried to answer -- "
        f"she got: {told!r}")

    notice = main_mod._FOREIGN_ATTEMPT_NOTICE
    assert notice.strip(), "an empty notice is a silent one"
    assert len(notice) < 120, f"{len(notice)} chars, spoken: {notice!r}"
    assert _A_DEVICE not in notice, notice
    assert "phone" not in notice.lower(), notice
    assert "\\" not in notice and "/" not in notice, notice


@pytest.mark.asyncio
async def test_a_non_owners_message_still_takes_an_ordinary_turn(
        turn, monkeypatch):
    """The non-owner is SKIPPED, not refused, and this is the test that says
    so. Refusing the turn was the first shape and it denied traffic that had
    nothing to do with the confirmation -- including, in the direction nobody
    writes down, the person at the keyboard being denied by a *phone's* open
    question. There is no attacker in that story and the cure was worse than
    the disease.

    It also matches the capability skip immediately above it in the loop,
    whose own comment argues that refusing there discloses that a
    confirmation is waiting. Two answer sites with two behaviours is how the
    next reader gets one of them wrong."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    answered, spy = _spy()
    state = _arm("destructive", LOCAL_PRINCIPAL)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    out = await turn("what is the weather", _remote_grants(), _A_DEVICE)

    assert answered == [], "the non-owner drove the confirmation"
    assert state.active, (
        "the non-owner's turn consumed the owner's confirmation")
    assert out.dispatched == ["get_time"], (
        "the non-owner's turn never reached dispatch -- it was refused "
        f"rather than passed through to ordinary classification: {out.dispatched}")
    said = [a[2] for a, _ in out.saved if len(a) > 2]
    assert main_mod._FOREIGN_ATTEMPT_NOTICE not in " ".join(said), (
        "the notice about the attempt was delivered to the one that made it")


@pytest.mark.asyncio
async def test_the_owner_is_told_once_not_on_every_later_answer(
        turn, monkeypatch):
    """Read-and-clear, so a batch of attempts is reported exactly once. A
    notice that repeated on every subsequent answer would be noise, and noise
    is how a real warning gets ignored."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    answered, spy = _spy()
    state = _arm("destructive", LOCAL_PRINCIPAL)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    # Two foreign attempts, then two answers by the owner.
    await turn("yes", _remote_grants(), _ANOTHER_DEVICE)
    await turn("yes", _remote_grants(), _ANOTHER_DEVICE)
    assert answered == []

    first = await turn("yes", _local_grants(), LOCAL_PRINCIPAL, source="stt")
    said = [a[2] for a, _ in first.saved if len(a) > 2]
    assert main_mod._FOREIGN_ATTEMPT_NOTICE in said[-1], said[-1]

    second = await turn("yes", _local_grants(), LOCAL_PRINCIPAL, source="stt")
    said = [a[2] for a, _ in second.saved if len(a) > 2]
    assert main_mod._FOREIGN_ATTEMPT_NOTICE not in said[-1], (
        "the owner was told again about attempts she has already heard "
        f"about: {said[-1]!r}")


@pytest.mark.asyncio
async def test_a_device_owner_is_never_told_who_else_was_talking(
        turn, monkeypatch):
    """`owned_by` is symmetric, so a *device* can own a pending state -- and a
    device owner that could be told "something else tried to answer" would
    hold a presence oracle about the operator: arm a long confirmation
    (`pending_backup_oauth` runs 300 seconds), wait, answer, and learn whether
    anyone else spoke to TENKA in that window. Repeatable on demand by anyone
    holding a pairing.

    That is the same disclosure the capability skip in the dispatch loop
    refuses to make, pointed the other way. The notice goes to
    `LOCAL_PRINCIPAL` and nobody else; the WARNING log is the record for every
    owner."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    answered, spy = _spy()
    state = _arm("destructive", _A_DEVICE)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    # The operator speaks while the device's confirmation is open...
    await turn("what is the weather", _local_grants(), LOCAL_PRINCIPAL,
               source="stt")
    assert answered == [], "the operator drove the device's confirmation"

    # ...and the device answers its own question and learns nothing about it.
    out = await turn("yes", _remote_grants(), _A_DEVICE)
    assert answered == ["yes"], "the device could not answer its own question"
    said = [a[2] for a, _ in out.saved if len(a) > 2]
    assert main_mod._FOREIGN_ATTEMPT_NOTICE not in said[-1], (
        "a device owner was told whether the operator was at the machine: "
        f"{said[-1]!r}")


@pytest.mark.asyncio
async def test_the_teaching_session_answer_site_follows_the_same_rule(
        turn, monkeypatch):
    """There are two answer sites, and they must behave identically. The
    teaching session is the other one: `teaching_session` is process-global
    too, so a session opened at the keyboard would otherwise eat the next
    message from any paired device and write its words into the operator's
    procedure.

    Same rule as the chain, end to end: a caller that holds EXECUTE but did
    not open the session is skipped (its message takes an ordinary turn, the
    session stays open), and the operator is told on her next step. Two
    answer sites with two behaviours is how the next reader gets one of them
    wrong."""
    from assistant import main as main_mod
    import assistant.actions as actions_pkg
    from assistant.actions import LOCAL_PRINCIPAL

    fed: list = []

    async def _spy_teaching(text):
        fed.append(text)
        return "Got it. Next step?"

    monkeypatch.setattr(actions_pkg, "handle_pending_teaching", _spy_teaching)
    actions_pkg.teaching_session.set(
        {"trigger": "sync files", "steps": []}, principal=LOCAL_PRINCIPAL)

    # A device that DOES hold EXECUTE, so the capability skip cannot be what
    # stops it -- only the identity can.
    foreign = await turn("open cmd", _local_grants(), _A_DEVICE)
    assert fed == [], "a device fed a step into the operator's session"
    assert actions_pkg.teaching_session.active, (
        "the foreign turn closed the operator's session")
    assert foreign.dispatched == ["get_time"], (
        "the foreign turn was refused rather than passed through")

    owner = await turn("open cmd", _local_grants(), LOCAL_PRINCIPAL,
                       source="stt")
    assert fed == ["open cmd"], "the owner's own step stopped working"
    said = [a[2] for a, _ in owner.saved if len(a) > 2]
    assert main_mod._FOREIGN_ATTEMPT_NOTICE in said[-1], (
        f"the owner was never told her session was reached for: {said[-1]!r}")


def test_a_bare_arm_inherits_the_turns_principal():
    """The load-bearing default of the entire design, and it had no test.

    Fifteen of the ~17 arming sites in the tree are bare `state.set(payload)`
    calls inside `actions/` handlers, several frames below the turn that
    authorised them. They record an owner only because `set()` consumes
    `current_principal` when none is passed. Change that default to `None` and
    every confirmation in the product silently stops working; change it to
    `LOCAL_PRINCIPAL` and KI-13 silently reopens with a device inheriting the
    operator's identity. Neither turned anything red before this test.
    """
    from assistant.actions import LOCAL_PRINCIPAL, current_principal, set_principal
    from assistant.pending import PendingState

    state = PendingState("probe_ambient", timeout=60.0)

    token = set_principal(_A_DEVICE)
    try:
        state.set({"op": "delete"})
    finally:
        current_principal.reset(token)
    assert state.principal == _A_DEVICE, (
        "a bare arm did not consume the turn's principal -- every handler in "
        "actions/ arms this way, so nothing they arm would have an owner")
    assert state.owned_by(_A_DEVICE) is True
    assert state.owned_by(LOCAL_PRINCIPAL) is False, (
        "a device-armed state was answerable from the console")

    # Nothing installed: the fallback must not invent an owner.
    token = set_principal(None)
    try:
        state.set({"op": "delete"})
    finally:
        current_principal.reset(token)
    assert state.principal is None, (
        f"an arm outside any turn invented an owner: {state.principal!r}")
    assert state.owned_by(LOCAL_PRINCIPAL) is False


def test_an_explicit_nobody_is_not_the_same_as_an_unnamed_owner():
    """`set(payload)` and `set(payload, principal=None)` are opposites.

    Before the `AMBIENT_PRINCIPAL` sentinel, `set()` collapsed them: both
    consulted `current_principal`. That made passing a *carried* principal
    through -- `set(entry, principal=entry.get("principal"))`, which is
    exactly what the lazily-arming knowledge row does -- invert whenever the
    carried value was None. The entry meant "the run that learned this had no
    identity"; the state recorded "owned by whoever is speaking right now",
    and the `owned_by` check one line below then said yes to them. Fail-open,
    in the one branch KI-13 exists to close.

    So the two meanings need two spellings, and this pins both. The bare-arm
    half is `test_a_bare_arm_inherits_the_turns_principal`; this is the half
    that says an explicit None owns nothing and is owned by nobody.
    """
    from assistant.actions import LOCAL_PRINCIPAL, current_principal, set_principal
    from assistant.pending import AMBIENT_PRINCIPAL, PendingState

    state = PendingState("probe_explicit_nobody", timeout=60.0)

    token = set_principal(_A_DEVICE)
    try:
        state.set({"op": "delete"}, principal=None)

        assert state.principal is None, (
            f"an explicit `principal=None` took the ambient principal instead "
            f"of meaning nobody: {state.principal!r}")
        # The caller doing the arming is the one a fail-open default hands it
        # to, so it is the first thing to refuse.
        assert state.owned_by(current_principal.get()) is False, (
            "the caller that armed an unowned state became its owner")
        assert state.owned_by(_A_DEVICE) is False
        assert state.owned_by(_ANOTHER_DEVICE) is False
        assert state.owned_by(LOCAL_PRINCIPAL) is False
        assert state.owned_by(None) is False, (
            "an unowned state matched an unknown caller -- two absences of a "
            "decision agreed with each other")

        # And the sentinel keeps meaning what the ~18 bare arming sites need,
        # including when it is written out longhand.
        state.set({"op": "delete"}, principal=AMBIENT_PRINCIPAL)
        assert state.principal == _A_DEVICE
        assert state.owned_by(_A_DEVICE) is True
    finally:
        current_principal.reset(token)


def test_an_unset_principal_owns_nothing():
    """The fail-closed default, and it matches `current_grants`' exactly: the
    absence of a decision is not a decision to allow. A state armed with no
    principal is answerable by nobody -- not by a device, and not by the
    local console either."""
    from assistant.actions import LOCAL_PRINCIPAL
    from assistant.pending import PendingState

    state = PendingState("probe_unowned", timeout=60.0)
    state.set({"op": "delete"})

    assert state.principal is None
    assert state.owned_by(LOCAL_PRINCIPAL) is False
    assert state.owned_by(_A_DEVICE) is False
    assert state.owned_by(None) is False, (
        "two unset principals matched each other -- an unknown caller "
        "answered an unowned state")

    # And the ordinary direction, so the assertion above is not passing by
    # `owned_by` simply always returning False.
    state.set({"op": "delete"}, principal=_A_DEVICE)
    assert state.principal == _A_DEVICE
    assert state.owned_by(_A_DEVICE) is True
    assert state.owned_by(_ANOTHER_DEVICE) is False


# ═════════════════════════════════════════════════════════════════════════
# Structural: nobody can arm without saying who
# ═════════════════════════════════════════════════════════════════════════

def _pending_state_attribute_names() -> "set[str]":
    """Every name bound to a PendingState in `assistant.actions` OR in
    `assistant.main`'s own namespace.

    Derived from the live modules, not typed out here. A hand-written list of
    state names is the same failure mode as a hand-written list of arming
    sites, one level down.

    `main` is unioned in and that is not belt-and-braces -- it is the hole
    this walk had. `main.py` binds fifteen module-level aliases for the table
    (`from .actions import pending_destructive as _s_destructive, ...`), so
    `_s_destructive.set(payload)` is the spelling closest to hand for anyone
    editing that file, and it resolves to a name `vars(actions)` has never
    heard of. The walk skipped it in silence and passed. Both spellings now
    resolve, and the injection check in this file's docstring was re-run with
    the `_s_*` one specifically.
    """
    from assistant import actions, main
    from assistant.pending import PendingState
    names = {name for name, obj in vars(actions).items()
             if isinstance(obj, PendingState)}
    names |= {name for name, obj in vars(main).items()
              if isinstance(obj, PendingState)}
    return names


def _arming_calls_in(path: pathlib.Path):
    """(lineno, source_line, has_principal_kwarg) for every `<state>.set(...)`.

    Matches both spellings main.py uses: a bare name (`teaching_session.set`)
    and an attribute chain (`_actions.pending_incoming_messages.set`).
    """
    names = _pending_state_attribute_names()
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    found = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "set":
            continue
        target = func.value
        owner = (target.attr if isinstance(target, ast.Attribute)
                 else target.id if isinstance(target, ast.Name) else None)
        if owner not in names:
            continue
        has_principal = any(kw.arg == "principal" for kw in node.keywords)
        found.append((node.lineno, lines[node.lineno - 1].strip(), has_principal))
    return found


def test_every_arming_site_records_a_principal():
    """Structural, for the reason every structural test in this tree exists:
    a hand-maintained list of arming sites is exactly what the next arming
    site forgets, and a state armed with no owner can be answered by nobody
    -- a silent dead end rather than a loud one.

    `main.py` arms outside a turn (the notification flusher runs on its own
    task, with no `current_principal` installed), so its sites cannot inherit
    an owner from the turn in flight the way a handler in `actions/` does.
    They have to name one, and this fails the moment one does not.
    """
    calls = _arming_calls_in(_MAIN_PY)

    assert calls, (
        "the walk found no arming sites in main.py at all -- either the "
        "spelling changed or _pending_state_attribute_names() went empty, "
        "and a structural test that walks nothing passes forever")

    unowned = [(lineno, text) for lineno, text, ok in calls if not ok]
    assert not unowned, (
        f"pending states armed in main.py with no principal: {unowned}. "
        "Pass `principal=` -- `LOCAL_PRINCIPAL` for anything this machine "
        "arms on its own behalf. A state with no owner is answerable by "
        "nobody, which reads as a timeout.")


def test_a_lazily_arming_handler_carries_an_owner_and_checks_it():
    """The invariant the sixth column depends on, stated as what it protects.

    The dispatch loop asks `state.active` BEFORE calling the handler, so a
    handler that responds to an inactive state by arming it is exempt from the
    loop's owner check by construction: the check looked at the only moment
    the answer was False, and the owner would be recorded a frame later by
    whoever just spoke. That was the KI-13 hole in row ten.

    "Never arm lazily" is the wrong rule, though, and trying it broke three
    things: arming `knowledge_approval` at proposal-creation time made the
    planner read the step as waiting on the user and abandon the rest of the
    plan, made the notification flusher stall for the 60s timeout, and let a
    second proposal replace the first. Some questions genuinely only become
    questions on a later turn.

    So the rule is the pair that makes a lazy arm safe:

    1. it arms with a principal **carried from the work that created the
       question** (`principal=` on the `.set(`), never the ambient one, and
    2. it **checks ownership itself** (`owned_by`) before treating the message
       as an answer.

    LIMIT, stated here rather than only in a report: this is a shape matcher.
    It sees `state.set(...)` inside an `if` that asks whether the state is
    inactive, which is the shape row ten has and the shape a copy would have.
    An *indirect* self-arm -- a helper function, a `pending_registry.get(name)`
    lookup, anything one call away -- slips past it. A fuzzier rule (any `.set(`
    anywhere in a handler) was rejected rather than overlooked: it fires on
    `pending_handlers.py:330`, where `handle_pending_messaging_disambig`
    legitimately arms `pending_messaging_send` *after* the loop has already
    owner-checked the row it was called for. A matcher that cries wolf on the
    legitimate chained arm would be turned off, and then it protects nothing.

    SECOND LIMIT, in the same spirit: half one only checks that a `principal=`
    keyword is *present*. It says nothing about the value, so
    `set(entry, principal=current_principal.get())` -- the defect verbatim,
    with the ambient principal spelled out by hand -- passes it. That is
    acceptable rather than overlooked, because a matcher that judged the
    *expression* would have to model where a value came from, and the
    behavioural test below catches exactly this case without pretending to.

    `test_a_foreign_yes_cannot_write_a_knowledge_entry` and
    `test_a_proposal_that_lost_its_owner_is_answerable_by_nobody` are the
    behavioural half, and neither depends on spelling.
    """
    import inspect
    import textwrap
    from assistant import main as main_mod

    _INACTIVE_TESTS = ("payload is None", "not ", "is None")

    offenders = []
    for handler, label, _mi, _nb, _cap, _state in main_mod._PENDING_HANDLERS:
        try:
            src = textwrap.dedent(inspect.getsource(handler))
        except (OSError, TypeError):  # pragma: no cover - stdlib-less handler
            continue
        tree = ast.parse(src)
        lazy_arms = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            if not (("payload" in test_src or "active" in test_src)
                    and any(t in test_src for t in _INACTIVE_TESTS)):
                continue
            lazy_arms += [n for n in ast.walk(node)
                          if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute)
                          and n.func.attr == "set"]
        if not lazy_arms:
            continue

        # Half one: every lazy arm names the owner explicitly.
        unowned = [ast.unparse(n) for n in lazy_arms
                   if not any(kw.arg == "principal" for kw in n.keywords)]
        if unowned:
            offenders.append(
                (label, handler.__name__, "arms without principal=", unowned))

        # Half two: the handler asks whether it may act on what it just armed.
        if "owned_by(" not in src:
            offenders.append(
                (label, handler.__name__, "never calls owned_by()", []))

    assert not offenders, (
        "handlers that arm their own state lazily without both halves of the "
        f"invariant: {offenders}. The dispatch loop reads `state.active` "
        "before calling the handler, so a lazily-armed row is never checked "
        "by the loop and must check itself: arm with a principal carried from "
        "the work that created the question (never the ambient one -- that is "
        "whoever spoke next), then `owned_by(current_principal.get())` before "
        "treating the message as an answer. See "
        "`handle_pending_knowledge_approval`.")


def test_a_foreign_yes_cannot_write_a_knowledge_entry(monkeypatch):
    """The behavioural half, which does not care how the check is spelled.

    A code-executor run learns a lesson and queues it for approval. A paired
    device then says "yes" -- to something else entirely, or to this. It must
    not become a write to the operator's knowledge base, and the operator must
    still be able to approve it afterwards: a foreign answer may not consume
    the question either."""
    import assistant.actions as _act
    from assistant import knowledge
    from assistant.actions import LOCAL_PRINCIPAL, current_principal, set_principal
    from assistant.actions.pending_handlers import handle_pending_knowledge_approval
    from assistant.code_executor import retry as _retry

    writes: list = []
    monkeypatch.setattr(knowledge, "add_works_entry",
                        lambda *a, **k: writes.append(a) or True)
    monkeypatch.setattr(_retry, "_pending_knowledge_queue", [])

    # The operator's own run produces the lesson.
    token = set_principal(LOCAL_PRINCIPAL)
    try:
        _retry._queue_knowledge_proposal("svc", "slug", "a pattern", "a reason")
    finally:
        current_principal.reset(token)

    # A device answers first.
    token = set_principal(_A_DEVICE)
    try:
        resp = asyncio.run(handle_pending_knowledge_approval("yes"))
    finally:
        current_principal.reset(token)

    assert resp is None, (
        f"the device's answer was treated as an approval: {resp!r}")
    assert writes == [], (
        f"a device wrote to the operator's knowledge base: {writes}")
    assert _act.pending_knowledge_approval.active, (
        "the device's answer consumed the operator's proposal -- she can no "
        "longer approve her own lesson")
    assert _act.pending_knowledge_approval.principal == LOCAL_PRINCIPAL, (
        "the state recorded the device that answered rather than the run that "
        "learned the lesson -- this is KI-13's confused deputy exactly")

    # And the operator, afterwards, still can.
    token = set_principal(LOCAL_PRINCIPAL)
    try:
        resp = asyncio.run(handle_pending_knowledge_approval("yes"))
    finally:
        current_principal.reset(token)

    assert writes == [("svc", "slug", "a pattern", "a reason")], (
        f"the owner could not approve her own lesson: {writes}")
    assert resp is not None


def test_a_proposal_that_lost_its_owner_is_answerable_by_nobody(monkeypatch):
    """The direction the fix for the lazy arm got wrong, now pinned.

    `_principal_for_item` returns None for a studio item whose principal slot
    is missing or is not a string, and `main.py` then installs
    `set_principal(None)` -- "a lost principal is refused, never assumed", as
    its own docstring puts it. A lesson learned during such a turn is queued
    with `principal: None`.

    This row is the one place in the tree that could invert that rule, because
    it arms from a carried value rather than from the turn. If the carried
    None is read as "not specified", the state takes the *answering* caller's
    identity and the `owned_by` check on the very next line approves them --
    a foreign write to the knowledge base by way of the fix that was supposed
    to prevent one. Nothing produces such an item today, which is precisely
    why it needs a test rather than a comment.
    """
    import assistant.actions as _act
    from assistant import knowledge
    from assistant.actions import current_principal, set_principal
    from assistant.actions.pending_handlers import handle_pending_knowledge_approval
    from assistant.code_executor import retry as _retry

    writes: list = []
    monkeypatch.setattr(knowledge, "add_works_entry",
                        lambda *a, **k: writes.append(a) or True)
    # A lesson learned by a turn that had no identity: exactly what
    # `_queue_knowledge_proposal` stamps when `current_principal` is unset.
    monkeypatch.setattr(_retry, "_pending_knowledge_queue", [{
        "service": "svc", "slug": "slug",
        "pattern": "a pattern", "reason": "a reason",
        "principal": None,
    }])

    token = set_principal(_A_DEVICE)
    try:
        resp = asyncio.run(handle_pending_knowledge_approval("yes"))
    finally:
        current_principal.reset(token)

    assert resp is None, (
        f"an ownerless proposal was approved by the first voice to say yes: "
        f"{resp!r}")
    assert writes == [], (
        f"a caller that never learned the lesson wrote it to the knowledge "
        f"base: {writes}")
    assert _act.pending_knowledge_approval.principal is None, (
        "arming from a carried None took the answering caller's identity -- "
        "the fail-open direction KI-13 is about")
    assert _act.pending_knowledge_approval.owned_by(_A_DEVICE) is False
    assert _act.pending_knowledge_approval.owned_by(None) is False

    # Nobody else inherits it either -- not the console, not a second device.
    from assistant.actions import LOCAL_PRINCIPAL
    for who in (LOCAL_PRINCIPAL, _ANOTHER_DEVICE):
        token = set_principal(who)
        try:
            assert asyncio.run(handle_pending_knowledge_approval("yes")) is None
        finally:
            current_principal.reset(token)
    assert writes == [], (
        f"an ownerless proposal found an owner on a later turn: {writes}")


@pytest.mark.asyncio
async def test_a_plan_step_that_learns_a_lesson_runs_to_completion(monkeypatch):
    """Queueing a proposal must not look like "this step is waiting on you".

    `planner/executor.py` snapshots every registered pending state around each
    step and treats inactive->active as an interactive flow: the step is
    marked `"waiting"` and `planner._suspend_plan` abandons the remaining
    steps. `_save_success_knowledge` runs on any code step that self-healed
    through retry, which is common rather than exotic -- and the rendered
    proposal is discarded by the orchestrator, so the plan would have
    suspended waiting for an answer to a question nobody was asked.

    This drives the real `execute_step` with a stubbed tool that calls the
    real `_save_success_knowledge` -- the production function a self-healed
    code step ends in -- rather than reaching past it to the helper it uses.
    That matters: an arm re-added *inside* `_save_success_knowledge`, beside
    its `_queue_knowledge_proposal` call, is the most natural way to
    re-introduce the regression, and a test that called the helper directly
    would have walked straight past it.

    LIMIT: the chain above `_save_success_knowledge` is not driven.
    `orchestrator.execute_code_task` needs generated code, a sandbox run and a
    real retry loop, so reaching it from here would mean a live model. The
    two edges stubbed below are the model call and the on-disk knowledge
    store; everything between `execute_step` and the queue append is real.
    """
    from assistant import knowledge
    from assistant.actions.planner.executor import execute_step
    from assistant.actions.planner.planner import Plan, PlanStep
    from assistant.code_executor import retry as _retry
    from assistant.pending import pending_registry

    monkeypatch.setattr(_retry, "_pending_knowledge_queue", [])
    # The lesson store is on disk; novelty is not what this test is about.
    monkeypatch.setattr(knowledge, "has_knowledge", lambda service: False)

    async def _lesson_llm(prompt, **kw):
        return '{"pattern": "a pattern", "reason": "a reason"}'

    async def _execute(intent, params, llm_response="", bridge=None, **kw):
        # What a self-healed code step does on its way out -- the real one.
        await _retry._save_success_knowledge(
            "svc", "slug",
            broken_code="value = fetch(1)\n",
            fixed_code="value = fetch(1, retries=3)\n",
            history=[{"error": "boom"}],
            llm_func=_lesson_llm,
        )
        return "Played it."

    import assistant.actions as _actions_pkg
    monkeypatch.setattr(_actions_pkg, "execute", _execute)

    step = PlanStep(step_id=1, tool="code_executor", goal="play some music")
    plan = Plan(original_goal="play some music", steps=[step])

    async def _llm(*a, **k):
        return "ok"

    await execute_step(step, plan, llm_func=_llm)

    assert step.status != "waiting", (
        "queueing a knowledge proposal suspended the plan -- the remaining "
        "steps would be abandoned waiting for a question nobody was asked")
    assert step.status == "success", step.status
    assert not pending_registry.get("knowledge_approval").active, (
        "the proposal armed a pending state at creation time")

    # The queue, not a state: order preserved, nothing replaced.
    _retry._queue_knowledge_proposal("svc2", "slug2", "second", "second reason")
    assert [e["slug"] for e in _retry._pending_knowledge_queue] == ["slug", "slug2"], (
        "a second proposal inside the window replaced the first")
    # No turn is installed here, so both entries carry no owner. What that
    # then means at arm time is
    # `test_a_proposal_that_lost_its_owner_is_answerable_by_nobody`.
    assert all(e["principal"] is None for e in _retry._pending_knowledge_queue)


def test_a_queued_proposal_does_not_stall_notification_announcements():
    """The other system that reads "a pending state is active": `main.py`'s
    notification flusher defers every announcement while one is, and
    `knowledge_approval` has a 60-second timeout. Queueing must not trip it."""
    from assistant.code_executor import retry as _retry
    from assistant.pending import pending_registry

    _retry._pending_knowledge_queue.clear()
    _retry._queue_knowledge_proposal("svc", "slug", "a pattern", "a reason")
    try:
        assert not pending_registry.any_active(
            exclude={"incoming_messages", "teaching_session"}), (
            "queueing a proposal made the notification flusher go quiet for "
            "the state's full timeout")
    finally:
        _retry._pending_knowledge_queue.clear()


def test_a_studio_item_with_no_principal_slot_owns_nothing():
    """The identity twin of `_grants_for_item`'s missing-slot rule. Nothing
    produces such an item today, which is exactly why it needs pinning: a lost
    principal must be refused, never promoted to the local one. Promoting it
    would hand a turn of unknown provenance the operator's own confirmations."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    assert main_mod._principal_for_item(("studio", "hello")) is None
    assert main_mod._principal_for_item(("studio", "hello", frozenset())) is None
    assert main_mod._principal_for_item(
        ("studio", "hello", frozenset(), 12345)) is None, (
        "a non-string in the principal slot was accepted as an identity")
    assert main_mod._principal_for_item(
        ("studio", "hi", frozenset(), _A_DEVICE)) == _A_DEVICE

    # Local sources have no fourth slot and never need one.
    assert main_mod._principal_for_item(("stt", "hello", 250)) == LOCAL_PRINCIPAL
    assert main_mod._principal_for_item(("chat", "hello")) == LOCAL_PRINCIPAL

    # A source nobody has added yet owns nothing -- `_LOCAL_SOURCES` is a
    # literal allow-list, not `!= "studio"`.
    assert main_mod._principal_for_item(("sms", "hello")) is None


@pytest.mark.asyncio
async def test_the_scheduler_arms_as_local():
    """A scheduled task has no requester attached to it. `scheduler.py`
    already states `LOCAL_GRANTS` rather than inheriting an unset grant set;
    the principal is the same argument with the same answer -- installing the
    schedule required EXECUTE, so the identity being spent is the operator's,
    stated rather than assumed.

    Without this the scheduler would arm every confirmation its tasks raise
    with no owner, and the operator could not answer her own reminder."""
    from assistant import actions
    from assistant import scheduler

    seen = []

    async def _spy_execute(intent, params, *a, **k):
        seen.append(actions.current_principal.get())
        return "ok"

    original = actions.execute
    actions.execute = _spy_execute
    try:
        await scheduler._async_run_handler(
            {"task_type": "web_search", "task_goal": "rain tomorrow"})
    finally:
        actions.execute = original

    assert seen == [actions.LOCAL_PRINCIPAL], (
        f"the scheduler ran with principal {seen!r} -- anything it arms is "
        "owned by nobody and answerable by nobody")
