"""Milestone 6a.5, stream A: the EXECUTE capability boundary.

Spec: `.superpowers/sdd/2026-08-16-milestone6a5-security/spec.md` §5.1.

Each test keeps the reasoning that motivated it. The boundary this file pins
is new, so every default in it is fail-closed: an unset grant set refuses, an
unclassified intent refuses, and a ceiling is a literal rather than something
derived from the enum.
"""
import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ─── A1: Capability lives in core/ ───────────────────────────────────────
def test_capability_lives_in_core_and_imports_nothing():
    """core/ is the layer that imports nothing. The enum has to live there for
    actions/ to read it without an import-linter exemption."""
    src = (_ROOT / "assistant" / "core" / "capabilities.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    non_stdlib = [n for n in imports
                  if not (isinstance(n, ast.Import) and
                          all(a.name in ("enum",) for a in n.names))]
    assert not non_stdlib, f"core/capabilities.py must import only enum: {non_stdlib}"


def test_vault_still_exports_capability():
    """Every existing `from .vault import Capability` must keep working."""
    from assistant.io.api.vault import Capability as FromVault
    from assistant.core.capabilities import Capability as FromCore
    assert FromVault is FromCore


# ─── A2: EXECUTE, and ceilings as explicit literals ──────────────────────
def test_execute_exists():
    from assistant.core.capabilities import Capability
    assert Capability.EXECUTE.value == "execute"


@pytest.mark.parametrize("name", ["tailnet", "funnel", "quick"])
def test_no_transport_carries_execute(name):
    """funnel is the open internet and CHAT_SEND reaches every intent. The
    ceiling is what stops a pair code becoming code execution on this machine."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES
    assert Capability.EXECUTE not in POLICIES[name].ceiling


@pytest.mark.parametrize("name", ["tailnet", "funnel", "quick"])
def test_no_transport_carries_system_control(name):
    """PATCH /v1/settings turns the camera on and speaker verification off."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES
    assert Capability.SYSTEM_CONTROL not in POLICIES[name].ceiling


def test_local_carries_everything():
    """The operator at the keyboard keeps full power; this milestone is not a
    downgrade of the local path."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES
    assert POLICIES["local"].ceiling == frozenset(Capability)


def test_no_ceiling_is_derived_from_the_enum():
    """A ceiling spelled `frozenset(Capability)` grants every future capability
    automatically, over exactly the listeners that must never get one for free.
    Only `local` may be enum-derived, and it is asserted by value above."""
    src = (_ROOT / "assistant" / "io" / "api" / "policy.py").read_text(encoding="utf-8")
    assert "_ALL_CAPABILITIES" not in src


def test_effective_can_only_narrow():
    """policy.py's central invariant. Pinned so a future raise mechanism (6b)
    cannot quietly turn the intersection into a union."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES, effective
    every = frozenset(Capability)
    for policy in POLICIES.values():
        assert effective(every, policy) <= policy.ceiling
        assert effective(frozenset(), policy) == frozenset()


# ─── A3: the intent -> capability table ──────────────────────────────────
def test_default_is_the_strongest_capability():
    """An intent nobody classified must be refused over a transport, not
    admitted. Fail closed, same reasoning as `quick`'s literal ceiling."""
    from assistant.core.capabilities import Capability
    from assistant.core.intent_capabilities import DEFAULT_REQUIRED
    assert DEFAULT_REQUIRED is Capability.EXECUTE


def test_every_configured_intent_has_a_row():
    """A missing row still fails closed, but silently. This makes adding an
    intent without classifying it a loud test failure instead."""
    from assistant import config
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY
    missing = set(config.INTENTS) - set(REQUIRED_CAPABILITY)
    assert not missing, f"intents with no capability row: {sorted(missing)}"


def test_no_row_names_an_intent_that_does_not_exist():
    from assistant import config
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY
    stale = set(REQUIRED_CAPABILITY) - set(config.INTENTS)
    assert not stale, f"rows for intents that no longer exist: {sorted(stale)}"


def test_the_installing_intents_require_execute():
    """A monitor's _fire_action calls execute("code_executor", ...) directly, so
    gating the installed thing and not the installer would be theatre."""
    from assistant.core.capabilities import Capability
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY
    for intent in ("manage_monitor", "manage_schedule",
                   "manage_procedure", "manage_shortcut"):
        assert REQUIRED_CAPABILITY[intent] is Capability.EXECUTE


def test_code_execution_intents_require_execute():
    from assistant.core.capabilities import Capability
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY
    for intent in ("code_executor", "computer_task", "planner",
                   "find_and_click", "manifest_dispatch", "shutdown"):
        assert REQUIRED_CAPABILITY[intent] is Capability.EXECUTE


def test_conversation_stays_chat_send():
    from assistant.core.capabilities import Capability
    from assistant.core.intent_capabilities import REQUIRED_CAPABILITY
    for intent in ("small_talk", "unknown", "get_time", "web_search"):
        assert REQUIRED_CAPABILITY[intent] is Capability.CHAT_SEND


def test_intent_capabilities_imports_only_the_enum():
    """Pure data, same as core/intent_scopes.py. It may not grow a dependency
    on config or storage -- `actions/` imports it on the dispatch path."""
    src = (_ROOT / "assistant" / "core" / "intent_capabilities.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise AssertionError(f"unexpected plain import: {ast.dump(node)}")
        if isinstance(node, ast.ImportFrom):
            assert node.module in ("capabilities", "__future__"), node.module


# ─── A4: enforcement at the dispatch choke point ─────────────────────────
@pytest.mark.asyncio
async def test_a_chat_send_only_device_cannot_reach_code_executor():
    from assistant.core.capabilities import Capability
    from assistant import actions
    token = actions.set_grants(frozenset({Capability.CHAT_SEND}))
    try:
        result = await actions.execute("code_executor", {"goal": "print(1)"}, "")
    finally:
        actions.current_grants.reset(token)
    assert "permission" in result.lower()
    assert "execute" in result.lower()


@pytest.mark.asyncio
async def test_a_chat_send_only_device_can_still_reach_small_talk():
    """The refusal must not collapse into refusing everyone."""
    from assistant.core.capabilities import Capability
    from assistant import actions
    token = actions.set_grants(frozenset({Capability.CHAT_SEND}))
    try:
        result = await actions.execute("small_talk", {}, "hello there")
    finally:
        actions.current_grants.reset(token)
    assert "permission" not in result.lower()


@pytest.mark.asyncio
async def test_unset_grants_is_a_hard_failure_not_full_access():
    """The single most dangerous failure mode in this milestone: a turn that
    reaches dispatch without anyone deciding what it may do must refuse, never
    inherit local privileges."""
    from assistant import actions
    assert actions.current_grants.get() is None
    result = await actions.execute("code_executor", {"goal": "print(1)"}, "")
    assert "permission" in result.lower()


@pytest.mark.asyncio
async def test_an_unlisted_intent_requires_execute():
    from assistant.core.capabilities import Capability
    from assistant import actions
    token = actions.set_grants(frozenset({Capability.CHAT_SEND}))
    try:
        result = await actions.execute("some_intent_nobody_classified", {}, "")
    finally:
        actions.current_grants.reset(token)
    assert "permission" in result.lower()


@pytest.mark.asyncio
async def test_a_planner_step_is_checked_by_the_same_rule():
    """planner/executor.py re-enters actions.execute(), so the check is
    recursive. A plan must not be a way around the gate."""
    from assistant.actions.planner import executor  # noqa: F401 -- the re-entry site
    from assistant.core.capabilities import Capability
    from assistant import actions
    token = actions.set_grants(frozenset({Capability.CHAT_SEND}))
    try:
        result = await actions.execute("planner", {"goal": "do a thing"}, "")
    finally:
        actions.current_grants.reset(token)
    assert "permission" in result.lower()


@pytest.mark.asyncio
async def test_the_refusal_is_speakable():
    """It may reach tts.speak(): under 120 chars, no paths, no error codes."""
    from assistant.core.capabilities import Capability
    from assistant import actions
    token = actions.set_grants(frozenset({Capability.CHAT_SEND}))
    try:
        result = await actions.execute("file_task", {"goal": "read notes"}, "")
    finally:
        actions.current_grants.reset(token)
    assert len(result) < 120, result
    assert "\\" not in result and "/" not in result, result


@pytest.mark.asyncio
async def test_the_refusal_does_not_say_what_the_intent_would_have_done():
    """A caller that cannot reach an intent should not learn what it is."""
    from assistant.core.capabilities import Capability
    from assistant import actions
    token = actions.set_grants(frozenset({Capability.CHAT_SEND}))
    try:
        result = await actions.execute("shutdown", {}, "")
    finally:
        actions.current_grants.reset(token)
    assert "shutdown" not in result.lower(), result


# ─── A5: carrying grants from the request into the turn ──────────────────
def test_submit_requires_grants_with_no_default():
    """'Forgot to pass grants' must be a TypeError, not full access."""
    import inspect
    from assistant.actions.studio_runtime import ChatDispatch
    sig = inspect.signature(ChatDispatch.submit)
    assert "grants" in sig.parameters
    assert sig.parameters["grants"].default is inspect.Parameter.empty
    # And 'forgot to say who' must not be spelled the same way as "the person
    # at the keyboard" -- same rule, one question over. See KI-13.
    assert "principal" in sig.parameters
    assert sig.parameters["principal"].default is inspect.Parameter.empty


def test_the_chat_route_passes_the_devices_effective_grants():
    """Not the device's issued grants -- the intersection with the listener
    ceiling, which is what `effective()` returns and what the ceiling exists
    to enforce. `require()` already narrows before it hands the Device back,
    so `device.grants` *is* that intersection; recomputing it would be a
    second chance to get it wrong."""
    import inspect
    import assistant.io.api.routes.chat as chat_route
    src = inspect.getsource(chat_route.send_chat)
    assert "grants" in src


@pytest.mark.asyncio
async def test_a_studio_turn_runs_with_only_its_own_grants():
    """End-to-end: a CHAT_SEND-only device driving a turn must not have
    EXECUTE inside the pipeline."""
    from assistant.core.capabilities import Capability
    import queue as _queue
    from assistant import main as main_mod
    # `_input_queue` is module-level and shared: another test file that
    # submitted without draining leaves an item ahead of this one.
    while True:
        try:
            main_mod._input_queue.get_nowait()
        except _queue.Empty:
            break
    dispatch = main_mod._StudioDispatch()
    limited = frozenset({Capability.CHAT_SEND})
    turn_id, conv_id, accepted, reason = await dispatch.submit(
        "hello", limited, "device:probe")
    assert accepted
    source, text, grants, principal = main_mod._input_queue.get_nowait()
    assert source == "studio"
    assert grants == limited
    assert Capability.EXECUTE not in grants
    # The identity rides the same item, in its own slot (KI-13). It is not
    # `LOCAL_PRINCIPAL`, and it never can be: the route builds it as
    # `f"device:{device_id}"`, so the operator's own confirmations stay hers.
    assert principal == "device:probe"


@pytest.mark.asyncio
async def test_a_studio_turn_with_no_grants_on_the_queue_gets_none():
    """The consumer accepts the local 2-tuple too, but a *studio* item that
    somehow arrives without a grant set must not be read as the local one.
    Fail closed: an empty set, refused by every intent."""
    from assistant import main as main_mod
    grants, stt_ms = main_mod._grants_for_item(("studio", "hello"))
    assert grants == frozenset()
    assert stt_ms is None


@pytest.mark.asyncio
async def test_a_local_source_keeps_the_full_set_and_its_stt_timing():
    """The 3rd slot of a local item is stt_ms, not grants. Reading it as
    grants would both lose the timing and hand a number to the gate."""
    from assistant import main as main_mod
    from assistant.core.capabilities import Capability
    grants, stt_ms = main_mod._grants_for_item(("stt", "hello", 250))
    assert grants == frozenset(Capability)
    assert stt_ms == 250


# ─── A6: pinning the pair-grant invariants, and G12's main.py half ───────
def test_the_bootstrap_device_holds_execute():
    """main.py issues the first Studio device `frozenset(Capability)`. If
    someone 'fixes' that the way policy.py's enum-derived ceiling was fixed,
    the operator's own desktop Studio silently loses code execution and it
    reads as a bug rather than as a policy change."""
    from assistant.core.capabilities import Capability
    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    assert "frozenset(Capability)" in src
    assert Capability.EXECUTE in frozenset(Capability)


def test_a_code_carrying_execute_is_narrowed_when_redeemed_over_a_tunnel():
    """routes/pairing.py computes effective(pair_code.grants, policy) at
    redemption, using the ceiling of the listener the code arrives on. So a
    tunnel cannot receive EXECUTE no matter what the code was minted with."""
    from assistant.core.capabilities import Capability
    from assistant.io.api.policy import POLICIES, effective
    minted = frozenset(Capability)
    for name in ("tailnet", "funnel", "quick"):
        got = effective(minted, POLICIES[name])
        assert Capability.EXECUTE not in got
        assert Capability.SYSTEM_CONTROL not in got


def test_chat_text_cannot_forge_a_line_in_the_debug_log():
    """G12, main.py's half. A newline in chat text writes a second, fabricated
    log line into the file an operator greps after an incident. `!r` makes the
    newline visible as `\\n` inside one quoted line instead.

    The plan named main.py:839 and :841; the real sites are the two
    `Transcription (...)` lines, which had drifted. Found by content rather
    than by line number so the next drift does not silently pass."""
    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    sites = [line for line in src.splitlines()
             if "redact_secrets(transcription)" in line]
    assert sites, "the transcription log lines vanished -- retarget this test"
    for line in sites:
        assert "!r" in line, f"main.py interpolates chat text without !r: {line.strip()}"


# ─── The two runners that fire outside a turn ────────────────────────────
# Not in the plan. They are the call sites A4's fail-closed default breaks:
# neither has a turn around it, so `current_grants` would be unset and every
# fired monitor and scheduled task would answer with the refusal string. Each
# now states its grant, and each statement is pinned here.
@pytest.mark.asyncio
async def test_a_fired_monitor_runs_with_the_local_grant_set(monkeypatch):
    from assistant import actions
    from assistant.automation.event_bus import EventBus
    from assistant.core.capabilities import Capability

    seen = {}

    async def _spy(intent, params, llm_response="", bridge=None, _from_planner=False):
        seen["intent"] = intent
        seen["grants"] = actions.current_grants.get()
        return "ok"

    monkeypatch.setattr(actions, "execute", _spy)
    assert await EventBus()._run_code_executor("do the thing") == "ok"
    assert seen["intent"] == "code_executor"
    assert Capability.EXECUTE in seen["grants"]
    # ...and put back afterwards, so one fired monitor does not leave the
    # process permanently privileged.
    assert actions.current_grants.get() is None


@pytest.mark.asyncio
async def test_a_scheduled_task_runs_with_the_local_grant_set(monkeypatch):
    from assistant import actions, scheduler
    from assistant.core.capabilities import Capability

    seen = {}

    async def _spy(intent, params, llm_response="", bridge=None, _from_planner=False):
        seen["intent"] = intent
        seen["grants"] = actions.current_grants.get()
        return "ok"

    monkeypatch.setattr(actions, "execute", _spy)
    result = await scheduler._async_run_handler(
        {"task_type": "web_search", "task_goal": "tide times", "name": "t"})
    assert result == "ok"
    assert seen["intent"] == "web_search"
    assert Capability.CHAT_SEND in seen["grants"]
    assert actions.current_grants.get() is None


# ─── Integration finding: the pair default had the enum-inheritance shape ────

def test_the_pair_default_never_hands_out_execute():
    """`/studio pair` defaulted to `frozenset(Capability) - {SYSTEM_CONTROL}`.
    Adding EXECUTE to the enum silently widened it to include the strongest
    capability in the model -- the same trap `_ALL_CAPABILITIES` was deleted
    from policy.py for, in a file no stream owned. Found at integration."""
    from assistant.slash_commands import _studio_pair_default_grants
    from assistant.core.capabilities import Capability
    grants = _studio_pair_default_grants()
    assert Capability.EXECUTE not in grants
    assert Capability.SYSTEM_CONTROL not in grants


def test_the_pair_default_is_still_a_useful_remote():
    """Excluding the acting grants must not collapse pairing into uselessness:
    a paired phone still watches, recalls, chats, reads files, sees the screen."""
    from assistant.slash_commands import _studio_pair_default_grants
    from assistant.core.capabilities import Capability
    grants = _studio_pair_default_grants()
    for c in (Capability.OBSERVE, Capability.RECALL, Capability.CHAT_SEND,
              Capability.SCREEN, Capability.FILES):
        assert c in grants


def test_the_pair_default_is_not_derived_from_the_enum():
    """A default spelled as a subtraction from `Capability` grants every future
    capability to every paired device the day someone adds one."""
    import inspect
    from assistant import slash_commands
    src = inspect.getsource(slash_commands._studio_pair_default_grants)
    assert "Capability.EXECUTE" in src, "EXECUTE must be excluded by name"
