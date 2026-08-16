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

    async def _detect(*a, **k):
        return IntentResult(intent="small_talk", response="ok", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)

    class _Run:
        def __init__(self):
            self.spoken = spoken
            self.saved = saved
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
async def test_a_mismatch_is_refused_loudly_not_silently_ignored(
        turn, monkeypatch):
    """A dropped confirmation reads as a timeout to whoever armed it and as
    nothing at all to whoever tried to answer. KI-13 asks for the opposite:
    the mismatch is refused out loud, so the operator learns that something
    else tried to answer.

    The refusal is spoken, so it carries the same constraints every spoken
    string in this tree carries -- under 120 characters, no paths, no error
    codes -- plus one of its own: it must not name the device, because the
    thing that tried to answer is exactly what a refusal must not disclose."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    answered, spy = _spy()
    state = _arm("destructive", LOCAL_PRINCIPAL)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS", _one_row(spy, state))

    out = await turn("yes", _remote_grants(), _A_DEVICE)

    said = [a[2] for a, _ in out.saved if len(a) > 2]
    assert said, f"the mismatch was silently ignored -- nothing saved: {out.saved}"
    refusal = said[-1]
    assert refusal.strip(), "an empty refusal is a silent one"
    assert len(refusal) < 120, f"{len(refusal)} chars, spoken: {refusal!r}"
    assert _A_DEVICE not in refusal, refusal
    assert "phone" not in refusal.lower(), refusal
    assert "\\" not in refusal and "/" not in refusal, refusal


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
    """Every attribute name in `assistant.actions` bound to a PendingState.

    Derived from the live module, not typed out here. A hand-written list of
    state names is the same failure mode as a hand-written list of arming
    sites, one level down.
    """
    from assistant import actions
    from assistant.pending import PendingState
    return {name for name, obj in vars(actions).items()
            if isinstance(obj, PendingState)}


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
