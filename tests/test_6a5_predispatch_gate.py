"""The pre-dispatch branches of `process_text_from_queue` are capability-gated.

Milestone 6a.5 put the EXECUTE boundary at one choke point --
`actions/__init__.py`, immediately before `tool_registry.get(intent)`. That
check is correct and exhaustive *for anything that reaches it*. The adversarial
review found that `main.py`'s turn pipeline has ~330 lines of branches that
produce an effect and `return` before dispatch is ever called, so the gate was
guarding the last door while five earlier ones stood open.

The mechanism this file pins is deliberately not "one bolt-on check per
branch" -- that is whack-a-mole, and the next branch someone adds misses it.
It is three pieces:

1. `actions.capability_refusal(required)` -- the single predicate. It reads the
   same `current_grants` contextvar `actions.execute()` reads, and
   `actions.execute()` now calls it too, so there is exactly one answer to
   "may this turn do X".
2. A `_gate(...)` closure inside `process_text_from_queue` that a
   refuse-and-stop branch calls, and a bare `capability_refusal(...)` call for
   the branches that must *skip* rather than refuse.
3. `test_every_returning_predispatch_branch_is_guarded` below, which walks the
   function's AST and fails if a branch that returns before dispatch contains
   neither. A branch may opt out only with an in-code `CAPABILITY-EXEMPT:`
   marker, and the set of markers is pinned here, so opting out is loud.

Threat model, from the spec's standing constraint: the widest device that
paired over a non-loopback listener holds `effective(device_grants, ceiling)`.
The `tailnet`/`funnel` ceilings omit EXECUTE and SYSTEM_CONTROL, so that set is
{OBSERVE, RECALL, CHAT_SEND, SCREEN, FILES}. Every test below asks what that
device can still make TENKA do.

Run with:  py -3.11 -m pytest tests/test_6a5_predispatch_gate.py -v
"""
import ast
import contextvars
import inspect
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.capabilities import Capability

_MAIN_PY = _ROOT / "assistant" / "main.py"

_UNSET = object()
"""Sentinel for `turn(principal=...)`. `None` is a meaningful principal -- it
is the one that owns nothing -- so it cannot double as "you did not say"."""


def _remote_grants() -> "frozenset[Capability]":
    """Built from policy.py rather than hand-written, so this file tracks the
    ceiling if it moves."""
    from assistant.io.api.policy import POLICIES
    return frozenset(POLICIES["funnel"].ceiling)


def _local_grants() -> "frozenset[Capability]":
    from assistant.actions import LOCAL_GRANTS
    return LOCAL_GRANTS


# ─────────────────────────────────────────────────────────────────────────
# Harness: run one real turn through main.py with a real grant set.
#
# Only the *edges* are stubbed -- TTS, the Unity bridge, the LLM, SQLite,
# telemetry. Every branch of the dispatcher itself is the real code, and so
# is the capability check.
# ─────────────────────────────────────────────────────────────────────────

class _FakeBridge:
    def __init__(self):
        self.commands = []

    async def send_command(self, name, **kw):
        self.commands.append((name, kw))


class _FakeTracker:
    """`_telemetry.TurnTracker` writes to SQLite in the turn's `finally`."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return None

    def save(self):
        pass


@pytest.fixture
def turn(monkeypatch):
    from assistant import main as main_mod

    spoken: list = []
    saved: list = []

    # -- telemetry: no SQLite, no session table --------------------------
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

    # -- speech / avatar --------------------------------------------------
    async def _speak(text, *a, **k):
        spoken.append(text)
    monkeypatch.setattr(main_mod.tts, "speak", _speak)

    async def _finish(*a, **k):
        return None
    monkeypatch.setattr(main_mod, "_finish_turn", _finish)
    monkeypatch.setattr(main_mod, "_publish_turn_status", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_wake_listener", None)

    # -- LLM: never called for real ---------------------------------------
    async def _chat(*a, **k):
        return "ok"
    monkeypatch.setattr(main_mod.llm, "chat", _chat)

    # -- topic tracker touches the DB -------------------------------------
    class _Topic:
        def push_turn(self, *a, **k): pass
        def resolve_query(self, q): return q
        def get_topic_hint(self): return None
    monkeypatch.setattr(main_mod, "_get_topic_tracker", lambda: _Topic())

    # -- procedure / shortcut lookups hit SQLite; default to "no match" ---
    monkeypatch.setattr(main_mod.procedures, "match_trigger", lambda t: None)
    monkeypatch.setattr(main_mod.procedures, "find_by_name_or_trigger",
                        lambda *a, **k: None)
    monkeypatch.setattr(main_mod.shortcuts, "match_shortcut", lambda t: None)
    monkeypatch.setattr(main_mod.regex_router, "match_procedure_command",
                        lambda t: None)

    class _Run:
        def __init__(self):
            self.spoken = spoken
            self.saved = saved
            self.tracker = None
            self.bridge = None

        async def __call__(self, text, grants, source="studio", bridge=None,
                           principal=_UNSET):
            bridge = bridge or _FakeBridge()
            # Derived from the source the way `_principal_for_item` derives
            # it, so a turn through this harness carries the identity a real
            # queued item of the same source would. Overridable, because two
            # devices with the same ceiling are only distinguishable by it.
            if principal is _UNSET:
                from assistant.actions import LOCAL_PRINCIPAL
                principal = (LOCAL_PRINCIPAL
                             if source in main_mod._LOCAL_SOURCES else None)
            await main_mod.process_text_from_queue(source, text, bridge,
                                                   grants=grants,
                                                   principal=principal)
            self.tracker = tracker_box.get("t")
            self.bridge = bridge
            return self

    yield _Run()


# ═════════════════════════════════════════════════════════════════════════
# HOLE 1 -- taught-procedure install and replay (CRITICAL)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_procedure_replay_is_refused_without_execute(turn, monkeypatch):
    """`main.py`'s procedure-replay branch called `run_procedure()` directly,
    and that function drives `automation.native`/`pyautogui` itself -- it never
    passes through `actions.execute()`. A device holding only CHAT_SEND could
    say any stored procedure's trigger phrase and get its keystrokes, clicks
    and app launches performed on the operator's desktop."""
    from assistant import main as main_mod
    from assistant import procedure_executor

    ran = []

    async def _spy_run_procedure(proc, original_text):
        ran.append(proc["name"])
        return "step 1: done"

    monkeypatch.setattr(procedure_executor, "run_procedure", _spy_run_procedure)
    monkeypatch.setattr(main_mod.procedures, "match_trigger", lambda t: {
        "id": 1, "name": "evil", "trigger": "deploy the thing",
        "steps": [{"type": "app", "action": "open", "params": {"name": "cmd"}}],
    })

    out = await turn("deploy the thing", _remote_grants())

    assert ran == [], f"the procedure replayed for a device without EXECUTE: {ran}"
    assert any("execute permission" in str(a) for a, _ in out.saved), out.saved


@pytest.mark.asyncio
async def test_procedure_replay_still_works_locally(turn, monkeypatch):
    """The other direction, and the one that must never regress: voice and
    console carry LOCAL_GRANTS, so the same trigger still replays."""
    from assistant import main as main_mod
    from assistant import procedure_executor

    ran = []

    async def _spy_run_procedure(proc, original_text):
        ran.append(proc["name"])
        return "step 1: done"

    monkeypatch.setattr(procedure_executor, "run_procedure", _spy_run_procedure)
    monkeypatch.setattr(main_mod.procedures, "match_trigger", lambda t: {
        "id": 1, "name": "morning", "trigger": "deploy the thing", "steps": [],
    })

    await turn("deploy the thing", _local_grants(), source="stt")
    assert ran == ["morning"]


@pytest.mark.asyncio
async def test_run_procedure_refuses_with_an_empty_grant_set():
    """The backstop, not the boundary. `run_procedure` is also reached from
    `scheduler.py`, so the caller check in main.py is not the only door. With
    the grant set explicitly empty -- the fail-closed state `execute()` refuses
    on -- no step may run. Only the OS layer is stubbed."""
    from assistant import actions, procedure_executor

    pressed = []
    fake_pyautogui = types.SimpleNamespace(
        hotkey=lambda *k: pressed.append(("hotkey",) + k),
        press=lambda k: pressed.append(("press", k)),
        typewrite=lambda t, interval=0: pressed.append(("type", t)),
    )
    fake_native = types.SimpleNamespace()

    async def _open_app(name):
        pressed.append(("open", name))
        return "opened"
    fake_native.open_app = _open_app

    saved_pg = sys.modules.get("pyautogui")
    saved_nat = sys.modules.get("assistant.automation.native")
    sys.modules["pyautogui"] = fake_pyautogui
    sys.modules["assistant.automation.native"] = fake_native
    saved_usage = procedure_executor.procedures.record_usage
    procedure_executor.procedures.record_usage = lambda _id: None

    token = actions.current_grants.set(frozenset())
    try:
        result = await procedure_executor.run_procedure(
            {"id": 1, "name": "pwn", "trigger": "go",
             "steps": [
                 {"type": "app", "action": "open", "params": {"name": "cmd"}},
                 {"type": "app", "action": "press_key",
                  "params": {"key": "ctrl+shift+enter"}},
             ]},
            "go",
        )
    finally:
        actions.current_grants.reset(token)
        procedure_executor.procedures.record_usage = saved_usage
        for name, saved in (("pyautogui", saved_pg),
                            ("assistant.automation.native", saved_nat)):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    assert pressed == [], f"steps ran with no grants: {pressed}"
    assert "execute permission" in result


def test_run_procedure_reads_the_one_contextvar():
    """Supporting evidence: the backstop is not a second, parallel notion of
    permission -- it calls the same predicate `actions.execute()` calls."""
    src = (_ROOT / "assistant" / "procedure_executor.py").read_text(encoding="utf-8")
    assert "capability_refusal" in src
    assert "Capability.EXECUTE" in src


def test_scheduler_grants_a_stored_procedure_before_running_it():
    """`scheduler.py` runs stored procedures with no requester attached. Before
    this fix it called `run_procedure` with `current_grants` unset, which the
    new backstop would refuse -- breaking every scheduled procedure. The
    scheduler states the grant explicitly, exactly as it already does for its
    `web_search` task type."""
    from assistant import scheduler
    src = inspect.getsource(scheduler._async_run_handler)
    proc_half = src[src.index('task_type == "procedure"'):]
    assert "set_grants(LOCAL_GRANTS)" in proc_half, proc_half
    assert "current_grants.reset" in proc_half


@pytest.mark.asyncio
async def test_batch_teaching_is_refused_without_execute(turn, monkeypatch):
    """A regex match on a multi-line paste installed a procedure with no gate.
    `manage_procedure` requires EXECUTE; this door did not. Combined with the
    replay hole above it is full interactive control of the machine:

        teach you how to sync files
        open cmd
        type whoami > C:/Users/Public/pwn.txt
        press enter
    """
    import assistant.actions as actions_pkg

    installed = []
    monkeypatch.setattr(actions_pkg, "start_batch_teaching",
                        lambda seed, body: installed.append(seed) or "Learned it.")

    payload = ("teach you how to sync files\n"
               "open cmd\n"
               "type whoami\n"
               "press enter")
    out = await turn(payload, _remote_grants())

    assert installed == [], "a procedure was installed without EXECUTE"
    assert any("execute permission" in str(a) for a, _ in out.saved), out.saved


@pytest.mark.asyncio
async def test_teach_trigger_is_refused_without_execute(turn, monkeypatch):
    """The single-line form is the same door. Once the session is open every
    subsequent turn routes to `handle_pending_teaching`, so leaving this ungated
    puts the whole multi-turn teaching state machine on CHAT_SEND."""
    import assistant.actions as actions_pkg

    started = []
    monkeypatch.setattr(actions_pkg, "start_teaching_session",
                        lambda seed: started.append(seed) or "Teach me.")

    out = await turn("teach you how to open my shell", _remote_grants())

    assert started == [], "a teaching session opened without EXECUTE"
    assert any("execute permission" in str(a) for a, _ in out.saved), out.saved


@pytest.mark.asyncio
async def test_teaching_still_works_locally(turn, monkeypatch):
    import assistant.actions as actions_pkg

    started = []
    monkeypatch.setattr(actions_pkg, "start_teaching_session",
                        lambda seed: started.append(seed) or "Teach me.")

    await turn("teach you how to open my shell", _local_grants(), source="stt")
    assert started, "the local teach trigger stopped working"


@pytest.mark.asyncio
async def test_an_open_teaching_session_ignores_a_caller_without_execute(
        turn, monkeypatch):
    """A teaching session armed at the keyboard must not be steerable from a
    remote device. This branch *skips* rather than refuses: the remote text
    still gets an ordinary turn, and the session keeps waiting for the operator
    -- refusing would both hijack the reply and disclose that a session is
    open."""
    from assistant import main as main_mod
    import assistant.actions as actions_pkg

    fed = []

    async def _spy_teaching(text):
        fed.append(text)
        return "next step?"

    monkeypatch.setattr(main_mod, "handle_pending_teaching", _spy_teaching,
                        raising=False)
    monkeypatch.setattr(actions_pkg, "handle_pending_teaching", _spy_teaching)
    actions_pkg.teaching_session.set(
        {"trigger": "sync files", "steps": []},
        principal=actions_pkg.LOCAL_PRINCIPAL)
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    from assistant.intent import IntentResult

    async def _detect(*a, **k):
        return IntentResult(intent="small_talk", response="hi", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)

    try:
        await turn("open cmd", _remote_grants())
        assert fed == [], "a remote caller fed a step into the operator's session"
        assert actions_pkg.teaching_session.active, (
            "the operator's session was cleared by the remote turn")
    finally:
        actions_pkg.teaching_session.clear()


@pytest.mark.asyncio
async def test_an_open_teaching_session_still_takes_local_steps(turn, monkeypatch):
    """The other direction: the operator's own next utterance still lands in
    the session it belongs to."""
    from assistant import main as main_mod
    import assistant.actions as actions_pkg

    fed = []

    async def _spy_teaching(text):
        fed.append(text)
        return "next step?"

    monkeypatch.setattr(actions_pkg, "handle_pending_teaching", _spy_teaching)
    actions_pkg.teaching_session.set(
        {"trigger": "sync files", "steps": []},
        principal=actions_pkg.LOCAL_PRINCIPAL)
    try:
        await turn("open cmd", _local_grants(), source="stt")
        assert fed == ["open cmd"], "the local teaching session stopped working"
    finally:
        actions_pkg.teaching_session.clear()


# ═════════════════════════════════════════════════════════════════════════
# HOLE 2 -- `shutdown` handled inline, above the gate (HIGH)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_shutdown_is_refused_without_execute(turn, monkeypatch):
    """`core/intent_capabilities.py` puts `shutdown` behind EXECUTE, but the
    intent was handled inline and returned before `execute_action()` was ever
    called, so the table row was dead code. Any CHAT_SEND device had a remote
    off switch for a security product."""
    from assistant import main as main_mod
    from assistant.intent import IntentResult

    main_mod._shutdown_event.clear()

    async def _detect(*a, **k):
        return IntentResult(intent="shutdown", response="bye", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    try:
        out = await turn("shut yourself down", _remote_grants())
        assert not main_mod._shutdown_event.is_set(), (
            "a device without EXECUTE shut the assistant down")
        assert any("execute permission" in str(a) for a, _ in out.saved), out.saved
    finally:
        main_mod._shutdown_event.clear()


@pytest.mark.asyncio
async def test_shutdown_still_works_locally(turn, monkeypatch):
    from assistant import main as main_mod
    from assistant.intent import IntentResult

    main_mod._shutdown_event.clear()

    async def _detect(*a, **k):
        return IntentResult(intent="shutdown", response="bye", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    try:
        await turn("shut yourself down", _local_grants(), source="stt")
        assert main_mod._shutdown_event.is_set(), (
            "the local shutdown path stopped working")
    finally:
        main_mod._shutdown_event.clear()


# ═════════════════════════════════════════════════════════════════════════
# HOLE 3 -- speaker verification switched off from chat (HIGH)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_listen_to_everyone_is_refused_without_system_control(
        turn, monkeypatch):
    """A plain-English substring called `speaker_verify.set_listen_to_everyone`
    and persisted it. `PATCH /v1/settings` gates the very same runtime-config
    key behind SYSTEM_CONTROL -- two doors onto one switch, gated differently.

    The escalation, not just the setting change: with verification off, anyone
    audible near the machine drives a *voice* turn, and voice carries
    LOCAL_GRANTS. Remote-to-local privilege escalation."""
    from assistant import main as main_mod

    calls = []
    monkeypatch.setattr(main_mod.speaker_verify, "set_listen_to_everyone",
                        lambda v: calls.append(v))
    monkeypatch.setattr(main_mod.config, "SPEAKER_VERIFY_ENABLED", True)

    out = await turn("listen to everyone", _remote_grants())

    assert calls == [], f"speaker verification was switched off remotely: {calls}"
    assert any("system_control permission" in str(a) for a, _ in out.saved), out.saved


@pytest.mark.asyncio
async def test_listen_to_owner_is_refused_without_system_control(
        turn, monkeypatch):
    """Same switch, other direction. It tightens rather than loosens, so it is
    not an escalation -- but it lets a remote device undo the operator's own
    deliberate setting, and an asymmetric gate on one switch is a footgun."""
    from assistant import main as main_mod

    calls = []
    monkeypatch.setattr(main_mod.speaker_verify, "set_listen_to_everyone",
                        lambda v: calls.append(v))
    monkeypatch.setattr(main_mod.config, "SPEAKER_VERIFY_ENABLED", True)

    await turn("only listen to me", _remote_grants())
    assert calls == [], f"the switch moved without SYSTEM_CONTROL: {calls}"


@pytest.mark.asyncio
async def test_listen_mode_still_works_locally(turn, monkeypatch):
    from assistant import main as main_mod

    calls = []
    monkeypatch.setattr(main_mod.speaker_verify, "set_listen_to_everyone",
                        lambda v: calls.append(v))
    monkeypatch.setattr(main_mod.config, "SPEAKER_VERIFY_ENABLED", True)

    await turn("listen to everyone", _local_grants(), source="stt")
    assert calls == [True], "the local listen-mode toggle stopped working"


def test_the_chat_door_and_the_settings_route_demand_the_same_capability():
    """Supporting evidence: the two doors onto `listen_to_everyone` now agree."""
    from assistant.core import runtime_config
    assert "listen_to_everyone" in runtime_config.REGISTRY

    routes = (_ROOT / "assistant" / "io" / "api" / "routes" / "settings.py").read_text(
        encoding="utf-8")
    assert "require(Capability.SYSTEM_CONTROL)" in routes

    main_src = _MAIN_PY.read_text(encoding="utf-8")
    i_listen = main_src.index("_LISTEN_ALL")
    window = main_src[i_listen - 1200:i_listen + 2500]
    assert "Capability.SYSTEM_CONTROL" in window


# ═════════════════════════════════════════════════════════════════════════
# HOLE 4 -- the pending-handler chain (HIGH)
# ═════════════════════════════════════════════════════════════════════════

def test_every_pending_row_declares_a_capability():
    """The fix is data, not a bolt-on: `_PENDING_HANDLERS` gained a fifth
    column naming what the handler's effect costs. A row added without one is a
    `ValueError` on the loop's unpack, and this test names it earlier.

    6b added a sixth -- the PendingState the handler reads -- for the owner
    check that closes KI-13. Both are asserted here, because a row is only
    complete when it says what the effect costs *and* whose question it is."""
    from assistant import main as main_mod
    from assistant.pending import PendingState

    for entry in main_mod._PENDING_HANDLERS:
        assert len(entry) == 6, (
            f"row is not (handler, label, intent, bridge, cap, state): {entry}")
        assert isinstance(entry[4], Capability), entry
        assert isinstance(entry[5], PendingState), entry


@pytest.mark.asyncio
async def test_pending_handler_is_skipped_when_the_caller_lacks_its_capability(
        turn, monkeypatch):
    """Pending state is process-global, so the device that ANSWERS a
    confirmation need not be the one that ASKED. The operator says "delete my
    downloads" at the keyboard; TENKA arms `pending_destructive` and waits for
    "yes"; a remote device races in with "yes".

    Skipped, not refused: the remote text still gets an ordinary turn, and a
    refusal would disclose that a confirmation is armed."""
    from assistant import main as main_mod
    from assistant.intent import IntentResult

    answered = []

    async def _spy_destructive(text):
        answered.append(text)
        return "Deleted."

    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS",
                        [(_spy_destructive, "DESTRUCTIVE", "file_task", False,
                          Capability.EXECUTE, main_mod._s_destructive)])
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    async def _detect(*a, **k):
        return IntentResult(intent="small_talk", response="ok", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)

    await turn("yes", _remote_grants())
    assert answered == [], "a remote 'yes' drove a locally-armed confirmation"


@pytest.mark.asyncio
async def test_pending_handler_still_runs_for_a_caller_that_holds_it(
        turn, monkeypatch):
    from assistant import main as main_mod
    from assistant.intent import IntentResult

    answered = []

    async def _spy_destructive(text):
        answered.append(text)
        return "Deleted."

    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS",
                        [(_spy_destructive, "DESTRUCTIVE", "file_task", False,
                          Capability.EXECUTE, main_mod._s_destructive)])
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    async def _detect(*a, **k):
        return IntentResult(intent="small_talk", response="ok", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)

    await turn("yes", _local_grants(), source="stt")
    assert answered == ["yes"], "the local confirmation chain stopped working"


@pytest.mark.asyncio
async def test_pending_state_remembers_which_principal_armed_it(turn, monkeypatch):
    """6a.5's known residual, closed in 6b. A capability check cannot close a
    confused deputy that *holds* the capability: a tunnel device holds FILES,
    so it could answer a locally-armed `pending_file_search`. The fix is an
    owner on the state, compared at answer time -- see KI-13 and
    `tests/test_6b_principal.py`, which covers the four ownership directions.

    This was a strict `xfail` and the marker is gone. Its body arms the state
    it is talking about, which the xfail version never did: with nothing
    armed, the assertion held only because the spy answered unconditionally,
    and no principal design can distinguish a remote caller from a local one
    when there is no armed question for either to answer. The claim it makes
    is unchanged and now actually exercised.
    """
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL
    from assistant.intent import IntentResult

    answered = []

    async def _spy_file_search(text):
        answered.append(text)
        return "Opened it."

    state = main_mod._s_file_search
    state.set({"name": "notes", "tier": 1}, principal=LOCAL_PRINCIPAL)
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS",
                        [(_spy_file_search, "FILE", "file_task", False,
                          Capability.FILES, state)])
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    async def _detect(*a, **k):
        return IntentResult(intent="small_talk", response="ok", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)

    try:
        await turn("the first one", _remote_grants())
    finally:
        state.clear()
    assert answered == [], (
        "a remote device answered a confirmation it did not arm")


# ═════════════════════════════════════════════════════════════════════════
# HOLE 5 -- the remote hot mic (CRITICAL, worse than the four above)
# ═════════════════════════════════════════════════════════════════════════

def test_finish_turn_requires_the_caller_to_name_the_source():
    """`_finish_turn` runs `_follow_up_listen()` -> `record_until_silence()`
    and re-queues what it hears as `("stt", ...)`. `_grants_for_item` hands an
    `stt` item `frozenset(Capability)` -- the FULL local grant set. So this is
    the one function in the tree that can mint privilege a turn did not arrive
    with, and a caller that does not say which turn it is finishing mints it
    blind. No default is permitted, for the same reason
    `ChatDispatch.submit()` refuses one for its grants."""
    from assistant import main as main_mod
    sig = inspect.signature(main_mod._finish_turn)
    assert list(sig.parameters) == ["bridge", "source"], sig
    assert sig.parameters["source"].default is inspect.Parameter.empty, (
        "_finish_turn grew a default source -- a caller can now mint the "
        "local grant set by forgetting an argument")


def test_no_call_site_finishes_a_turn_without_naming_the_source():
    """Structural, because the risk is a *missed* call site rather than a
    wrong one."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call)
           and getattr(n.func, "id", None) == "_finish_turn"
           and len(n.args) < 2]
    assert not bad, f"_finish_turn called without a source at lines {bad}"


@pytest.mark.asyncio
async def test_a_remote_turn_never_opens_the_microphone(monkeypatch):
    """The attack, end to end. A remote Studio message that merely matched the
    teach-trigger regex reached `_finish_turn`, which opened the local
    microphone, recorded until silence, and re-queued the result as an `stt`
    item -- carrying `frozenset(Capability)`. The attacker speaks nothing: it
    waits for the operator to say anything at all in the room, and that
    utterance runs with every capability there is."""
    from assistant import main as main_mod

    listened = []

    async def _spy_follow_up():
        listened.append(True)
        return ("delete everything", 100)

    monkeypatch.setattr(main_mod, "_follow_up_listen", _spy_follow_up)
    monkeypatch.setattr(main_mod, "_wake_listener", None)

    queued_before = main_mod._input_queue.qsize()
    await main_mod._finish_turn(_FakeBridge(), "studio")

    assert listened == [], "a remote turn opened the local microphone"
    assert main_mod._input_queue.qsize() == queued_before, (
        "a remote turn queued a full-privilege stt item")


@pytest.mark.asyncio
async def test_a_local_turn_still_opens_the_microphone(monkeypatch):
    """The follow-up window is the whole point of `_finish_turn` for a person
    at the machine. It must keep working exactly as today."""
    from assistant import main as main_mod

    async def _spy_follow_up():
        return ("what's the time", 100)

    monkeypatch.setattr(main_mod, "_follow_up_listen", _spy_follow_up)
    monkeypatch.setattr(main_mod, "_wake_listener", None)

    before = main_mod._input_queue.qsize()
    await main_mod._finish_turn(_FakeBridge(), "stt")
    assert main_mod._input_queue.qsize() == before + 1
    assert main_mod._input_queue.get_nowait()[0] == "stt"


def test_local_sources_is_an_allow_list_not_a_studio_denylist():
    """`!= "studio"` fails open on the next source anyone adds, and the thing
    it would fail open into is the full local grant set."""
    from assistant import main as main_mod
    assert main_mod._LOCAL_SOURCES == frozenset({"stt", "chat"})
    assert "studio" not in main_mod._LOCAL_SOURCES

    src = _MAIN_PY.read_text(encoding="utf-8")
    fn = src[src.index("async def _finish_turn"):]
    assert "_LOCAL_SOURCES" in fn[:1600], (
        "_finish_turn stopped consulting the allow-list")


@pytest.mark.asyncio
async def test_a_remote_pre_dispatch_answer_is_not_spoken_aloud(turn, monkeypatch):
    """Same class as the refusals integration already fixed, on the answers.
    Every pre-dispatch branch spoke its result unconditionally, so a device
    holding only CHAT_SEND could make the local speaker talk on demand."""
    from assistant import main as main_mod

    monkeypatch.setattr(main_mod.speaker_verify, "set_listen_to_everyone",
                        lambda v: None)
    monkeypatch.setattr(main_mod.config, "SPEAKER_VERIFY_ENABLED", True)

    out = await turn("listen to everyone", _local_grants(), source="studio")
    assert out.spoken == [], f"a remote turn spoke in the room: {out.spoken}"


# ═════════════════════════════════════════════════════════════════════════
# HOLE 6 -- pre-dispatch turns saved under a date, so Studio never sees them
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_pre_dispatch_answer_is_saved_under_the_session_id(
        turn, monkeypatch):
    """Four branches saved under `date.today().isoformat()`. Studio settles a
    turn by re-reading `memory.get_recent(conversation_id)` where the id is the
    session id, so those turns looked saved and rendered as vanished. Not a
    capability bug, but it is what made every refusal below invisible, so it
    is fixed with them."""
    from assistant import main as main_mod

    monkeypatch.setattr(main_mod.speaker_verify, "set_listen_to_everyone",
                        lambda v: None)
    monkeypatch.setattr(main_mod.config, "SPEAKER_VERIFY_ENABLED", True)

    out = await turn("listen to everyone", _local_grants(), source="studio")
    assert out.saved, "the turn was not recorded at all"
    ids = [a[3] for a, _ in out.saved if len(a) > 3]
    assert "probe-session" in ids, (
        f"saved under something other than the session id: {ids}")


def test_no_pre_dispatch_branch_saves_under_a_date():
    """Structural: no `save_turn(...)` in the turn pipeline may pass a date as
    its conversation id."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "process_text_from_queue")

    bad = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "save_turn"):
            continue
        conv_id = node.args[3] if len(node.args) > 3 else None
        if conv_id is None:
            continue
        rendered = ast.unparse(conv_id)
        if "isoformat" in rendered or "today" in rendered:
            bad.append((node.lineno, rendered))

    assert not bad, (
        f"turns saved under a date instead of the session id: {bad}. Studio "
        "re-reads memory.get_recent(session_id), so these never render.")


# ═════════════════════════════════════════════════════════════════════════
# THE MECHANISM -- what stops the seventh branch reopening the hole
# ═════════════════════════════════════════════════════════════════════════

# A branch may skip the gate only by saying so in code, on a line inside
# itself, as `# CAPABILITY-EXEMPT: <reason>`. The reasons are pinned here, so
# a *new* exemption fails this test by name and has to be argued for. Adding
# a branch with neither a guard nor a marker fails the test above it.
_EXPECTED_EXEMPTIONS = frozenset({
    "the pending chain gated each handler before awaiting it",
    "declining to act is not an effect",
    "this branch is itself a refusal",
})

# The guard calls a branch may use. Both read `actions.current_grants`.
#   `_gate`               -- refuse, tell the caller, stop the turn.
#   `capability_refusal`  -- the raw predicate, for branches that must skip
#                            silently rather than refuse.
_GUARD_CALLS = frozenset({"_gate", "capability_refusal"})


def _predispatch_region():
    """Return (statements, source_lines, dispatch_line).

    The pre-dispatch region is every top-level statement of
    `process_text_from_queue`'s `try:` body that begins before the
    `execute_action(...)` call -- the point where the real gate takes over.
    """
    src = _MAIN_PY.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "process_text_from_queue")
    try_stmt = next(s for s in fn.body if isinstance(s, ast.Try))

    dispatch_line = min(
        n.lineno for n in ast.walk(try_stmt)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "execute_action"
    )
    region = [s for s in try_stmt.body if s.lineno < dispatch_line]
    return region, lines, dispatch_line


def test_every_returning_predispatch_branch_is_guarded():
    """THE test. Every branch that produces an effect and returns before
    dispatch must consult the turn's grants.

    Structural on purpose. A hand-maintained list of branches is exactly what
    the next person forgets to extend; this walks the AST, so a new
    `if ...: ...; return` inside the region fails here the moment it is
    written, whatever it is called and wherever in the region it sits.
    """
    region, lines, dispatch_line = _predispatch_region()

    unguarded = []
    exemptions = set()

    for stmt in region:
        returns = [n for n in ast.walk(stmt) if isinstance(n, ast.Return)]
        if not returns:
            continue

        guarded = any(
            isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) in _GUARD_CALLS
                 or getattr(n.func, "attr", None) in _GUARD_CALLS)
            for n in ast.walk(stmt)
        )

        body = "\n".join(lines[stmt.lineno - 1:(stmt.end_lineno or stmt.lineno)])
        marker = None
        for line in body.splitlines():
            if "CAPABILITY-EXEMPT:" in line:
                marker = line.split("CAPABILITY-EXEMPT:", 1)[1].strip()
                break

        if guarded:
            continue
        if marker is not None:
            exemptions.add(marker)
            continue
        unguarded.append((stmt.lineno, lines[stmt.lineno - 1].strip()))

    assert not unguarded, (
        "pre-dispatch branches that return without consulting the turn's "
        f"grants: {unguarded}. Call `_gate(Capability.X, ...)` to refuse and "
        "stop, or `_actions_module.capability_refusal(Capability.X)` to skip "
        "silently. If the branch really produces no effect, say so on a line "
        "inside it as `# CAPABILITY-EXEMPT: <reason>` and add the reason to "
        "_EXPECTED_EXEMPTIONS in this file."
    )

    assert exemptions == _EXPECTED_EXEMPTIONS, (
        "the set of ungated pre-dispatch branches changed. Added: "
        f"{sorted(exemptions - _EXPECTED_EXEMPTIONS)}; removed: "
        f"{sorted(_EXPECTED_EXEMPTIONS - exemptions)}. Each exemption is a "
        "claim that a branch reachable from a remote device produces no "
        "effect -- argue it before pinning it."
    )


def test_the_region_the_structural_test_walks_is_not_empty():
    """A structural test that silently walks nothing passes forever. Pin that
    the region is real and covers the branches the review named."""
    region, lines, dispatch_line = _predispatch_region()
    returning = [s for s in region
                 if any(isinstance(n, ast.Return) for n in ast.walk(s))]
    assert len(returning) >= 8, (
        f"only {len(returning)} returning branches found before line "
        f"{dispatch_line} -- the region moved or the parse is wrong")


def test_there_is_one_source_of_truth_about_the_turns_grants():
    """`execute()` and every pre-dispatch guard answer the question the same
    way, because they are the same function. A second, parallel check would be
    a second thing to keep in sync -- which is how the hole opened."""
    from assistant import actions
    src = inspect.getsource(actions.execute)
    assert "capability_refusal(" in src, (
        "actions.execute() stopped using the shared predicate")

    pred = inspect.getsource(actions.capability_refusal)
    assert "current_grants.get()" in pred

    # And the check still precedes handler resolution, which is what the
    # original 6a.5 control asserted before the predicate was extracted.
    src = (_ROOT / "assistant" / "actions" / "__init__.py").read_text(encoding="utf-8")
    assert src.index("_refusal = capability_refusal(") < src.index("tool_registry.get(")


def test_the_predicate_fails_closed_on_an_unset_grant_set():
    from assistant import actions
    token = actions.current_grants.set(None)
    try:
        assert actions.capability_refusal(Capability.CHAT_SEND) is not None
    finally:
        actions.current_grants.reset(token)


def test_the_predicate_allows_exactly_what_is_granted():
    from assistant import actions
    token = actions.current_grants.set(frozenset({Capability.FILES}))
    try:
        assert actions.capability_refusal(Capability.FILES) is None
        assert actions.capability_refusal(Capability.EXECUTE) is not None
    finally:
        actions.current_grants.reset(token)


# ═════════════════════════════════════════════════════════════════════════
# HOW A REFUSAL IS DELIVERED
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_studio_refusal_is_recorded_and_never_spoken(turn, monkeypatch):
    """Studio settles a turn by re-reading the transcript, not from a return
    value, so a refusal that saves nothing looks like a lost turn. And it must
    NOT reach TTS: a remote device that can make the local speaker talk on
    demand has a standing way to interrupt the owner's room. Same split the
    slash-command and policy refusals already use."""
    from assistant import main as main_mod
    from assistant.intent import IntentResult

    main_mod._shutdown_event.clear()

    async def _detect(*a, **k):
        return IntentResult(intent="shutdown", response="bye", params={})
    monkeypatch.setattr(main_mod, "detect_intent", _detect)
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    try:
        out = await turn("shut yourself down", _remote_grants(), source="studio")
    finally:
        main_mod._shutdown_event.clear()

    assert out.spoken == [], f"a remote refusal was spoken aloud: {out.spoken}"
    assert out.saved, "the refusal left no transcript record"


def test_the_refusal_text_is_safe_to_speak():
    """It can reach TTS on a local path: under 120 chars, no paths, no codes."""
    from assistant import actions
    for cap in Capability:
        token = actions.current_grants.set(frozenset())
        try:
            msg = actions.capability_refusal(cap)
        finally:
            actions.current_grants.reset(token)
        assert msg and len(msg) < 120, (len(msg), msg)
        assert "\\" not in msg and "/" not in msg, msg
        assert not any(ch.isdigit() for ch in msg), msg
