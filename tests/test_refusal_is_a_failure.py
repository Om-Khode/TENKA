"""A refused step is a failed step, and a refused turn is not summarised.

The defect. `planner/executor.py` records a step's result and asks
`_step_failed()` whether it went wrong. That predicate matches some sixty
failure phrases and eleven prefixes, and **not one of the five refusal
sentences contains any of them** -- so a step the capability choke point
refused was recorded `status="success"`, with the refusal as its output. The
plan carried on, a dependent step took the refusal text as its `$step_N`
input, and `_synthesize_result` composed the spoken answer out of steps marked
successful.

The gate held. Nothing ran that should not have. What broke is the *report*,
which is KI-28's shape reached through a door `main.py`'s two
`_security_skip_this_turn` sites do not cover: they see refusals at the top of
a turn, and a refusal inside a planner step is six frames down.

Reachable on tailnet with a live EXECUTE raise -- `planner` costs EXECUTE and
no remote ceiling includes it -- where the device's issued grants omit
something one of its steps needs, or where the raise expires mid-plan. Not
reachable locally, which holds all seven.

The three properties, and each one's mutation:

  R1  every sentence either predicate can write is recognised as a refusal
      -- delete a template from `_REFUSAL_TEMPLATES`, this reds
  R2  a refused step is a failed step, and is not recovered from
      -- drop the `is_capability_refusal` call in `_step_failed`, this reds
  R3  a refusal anywhere in the turn reaches `security_skip`
      -- drop the `_note_refusal` call, this reds

Run with:  py -3.11 -m pytest tests/test_refusal_is_a_failure.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import assistant.actions as A  # noqa: E402
from assistant.actions import (  # noqa: E402
    _refuse, capability_refusal, durable_capability_refusal,
    is_capability_refusal,
)
from assistant.core.capabilities import Capability  # noqa: E402


class _Ctx:
    """Stand-in for a RaiseContext. Only the three sets are read."""

    def __init__(self, issued=(), raisable=(), ceiling=()):
        self.issued = frozenset(issued)
        self.raisable = frozenset(raisable)
        self.ceiling = frozenset(ceiling)


@pytest.fixture()
def clean_context():
    """Reset both contextvars after each test.

    Not optional hygiene: these are process-wide, and a leaked grant set makes
    the *next* file's tests pass for the wrong reason.
    """
    g = A.current_grants.set(None)
    r = A.current_raise_context.set(None)
    yield
    A.current_grants.reset(g)
    A.current_raise_context.reset(r)


# ─── R1 — every refusal sentence is recognised ───────────────────────────────

def test_every_sentence_refuse_can_write_is_recognised(clean_context):
    """All three branches of `_refuse`, every capability. Nine of these are the
    exact strings measured as invisible to `_step_failed` before the fix."""
    cases = [
        ("never issued", None),
        ("raisable here", _Ctx(issued=Capability, raisable=Capability)),
        ("never carried", _Ctx(issued=Capability, raisable=())),
    ]
    for label, ctx in cases:
        tok = A.current_raise_context.set(ctx)
        try:
            for cap in Capability:
                text = _refuse(cap)
                assert is_capability_refusal(text), (
                    f"unrecognised refusal ({label}, {cap.value}): {text!r}"
                )
        finally:
            A.current_raise_context.reset(tok)


def test_every_sentence_the_durable_gate_can_write_is_recognised(clean_context):
    """**The one the first draft of the predicate missed.**

    `durable_capability_refusal` does not always delegate to `_refuse` -- it
    writes two sentences of its own, and a predicate built only from
    `_refuse`'s three would have left a durable refusal inside a planner step
    still recorded as a success. Enumerating the sentences a boundary can
    produce is the same discipline as enumerating the paths around it, and it
    failed the same way the first time.
    """
    cap = Capability.EXECUTE

    # No raise context at all -- durable authority cannot be established.
    assert is_capability_refusal(durable_capability_refusal(cap))

    # Held only under a live raise: issued, but not within the ceiling.
    tok_r = A.current_raise_context.set(
        _Ctx(issued={cap}, raisable={cap}, ceiling=frozenset()))
    tok_g = A.current_grants.set(frozenset({cap}))     # live raise in force
    try:
        text = durable_capability_refusal(cap)
        assert text is not None and "keyboard" in text
        assert is_capability_refusal(text), f"unrecognised: {text!r}"
    finally:
        A.current_grants.reset(tok_g)
        A.current_raise_context.reset(tok_r)

    # Not in the durable set and not live either -- delegates to `_refuse`.
    tok_r = A.current_raise_context.set(
        _Ctx(issued=frozenset(), raisable=frozenset(), ceiling=frozenset()))
    try:
        assert is_capability_refusal(durable_capability_refusal(cap))
    finally:
        A.current_raise_context.reset(tok_r)


def test_ordinary_failure_text_is_not_a_refusal():
    """The other direction. A predicate that answered True often enough would
    satisfy every assertion above while making `_step_failed` useless -- and
    would stop legitimate recovery on any step whose output mentioned a
    permission."""
    for text in [
        "no results found",
        "Error: page.goto: net::ERR_NAME_NOT_RESOLVED",
        "I couldn't find that file",
        "Permission denied writing to C:\\Windows\\System32",
        "the app needs the microphone permission",
        "", "   ", None,
    ]:
        assert not is_capability_refusal(text), f"false positive on {text!r}"


def test_a_refusal_inside_a_longer_sentence_is_not_a_match(clean_context):
    """Exact match, not substring. A handler that *reports* a permission
    problem in its own words has not been refused by this module, and treating
    it as though it had would mark honest failures unrecoverable."""
    inner = _refuse(Capability.FILES)
    assert not is_capability_refusal(f"The tool said: {inner} Retrying.")


@pytest.mark.asyncio
async def test_execute_returns_the_refusal_unwrapped(clean_context):
    """What makes the exact match above safe.

    `is_capability_refusal` matching by identity only works while `execute()`
    returns `_refuse`'s output as its whole return value. If a future path ever
    wraps it -- an emotion tag, a prefix, a personality pass -- this reds, and
    that is the intended signal: the wrapping is fine, but the predicate has to
    learn about it in the same change.
    """
    tok = A.current_grants.set(frozenset())      # holds nothing
    try:
        out = await A.execute(intent="file_task", params={}, llm_response="")
    finally:
        A.current_grants.reset(tok)
    assert is_capability_refusal(out), f"execute() wrapped the refusal: {out!r}"


# ─── R2 — a refused step is a failed step ────────────────────────────────────

def test_step_failed_sees_a_refusal(clean_context):
    """The measurement that found this. Every sentence x capability pair
    returned False from `_step_failed` before the fix, so every one of them was
    recorded as a successful step."""
    from assistant.actions.planner.planner import _step_failed

    for ctx in (None, _Ctx(issued=Capability, raisable=Capability),
                _Ctx(issued=Capability, raisable=())):
        tok = A.current_raise_context.set(ctx)
        try:
            for cap in Capability:
                text = _refuse(cap)
                assert _step_failed(text), (
                    f"a refused step was recorded as a success: {text!r}"
                )
        finally:
            A.current_raise_context.reset(tok)


def test_step_failed_still_passes_ordinary_success_output():
    """Both directions. A predicate that called everything a failure would
    satisfy the test above and halt every plan that ever worked."""
    from assistant.actions.planner.planner import _step_failed

    for text in ["Opened the file.", "It is 4pm.", "Found 3 results: a, b, c"]:
        assert not _step_failed(text), f"{text!r} was called a failure"


@pytest.mark.asyncio
async def test_a_refusal_is_never_replanned_around(clean_context):
    """A security decision is not a failure to route around.

    The answer to "may this caller do that" is the same for every step in the
    turn, so replanning can only produce a differently-worded no -- while
    paying a plan-generating model to look for a way around a capability
    decision. The LLM must not be called at all.
    """
    from assistant.actions.planner.planner import _attempt_recovery, Plan, PlanStep

    called = []

    async def _llm(*a, **kw):
        called.append(a)
        return "[]"

    step = PlanStep(step_id=1, tool="file_task", goal="delete x")
    step.error = _refuse(Capability.FILES)
    plan = Plan(original_goal="delete x", steps=[step])

    out = await _attempt_recovery(step, plan, _llm)
    assert out == []
    assert not called, "a refusal was handed to the replanner"


# ─── R3 — the refusal reaches security_skip ──────────────────────────────────

def test_a_refusal_notes_the_turn_tracker(clean_context):
    """The ledger that carries a deep refusal back up to the turn loop.

    Without it, a capability decision inside a planner step leaves no trace at
    the level that writes the conversation row, so the turn is summarised into
    `session_snapshots` and replayed verbatim as fact next session.
    """
    import assistant.telemetry as telemetry

    tracker = telemetry.TurnTracker("s1", "voice", "delete the file")
    tok_t = telemetry.set_current_tracker(tracker)
    tok_g = A.current_grants.set(frozenset({Capability.CHAT_SEND}))
    try:
        assert capability_refusal(Capability.FILES) is not None
        assert tracker.refused_capabilities == {"files"}
    finally:
        A.current_grants.reset(tok_g)
        telemetry.reset_current_tracker(tok_t)


def test_a_permitted_call_notes_nothing(clean_context):
    """Both directions again. A ledger that recorded every check would mark
    every turn security_skip, suppressing every session summary -- which is a
    rule that gets reverted, taking the honesty property with it."""
    import assistant.telemetry as telemetry

    tracker = telemetry.TurnTracker("s1", "voice", "what time is it")
    tok_t = telemetry.set_current_tracker(tracker)
    tok_g = A.current_grants.set(frozenset({Capability.CHAT_SEND}))
    try:
        assert capability_refusal(Capability.CHAT_SEND) is None
        assert tracker.refused_capabilities == set()
    finally:
        A.current_grants.reset(tok_g)
        telemetry.reset_current_tracker(tok_t)


def test_noting_a_refusal_never_raises(clean_context):
    """No tracker installed -- the scheduler and the event bus set grants but
    no tracker. A bookkeeping failure must not turn a refusal into an exception
    at a security boundary; the refusal is already the safe answer."""
    import assistant.telemetry as telemetry

    assert telemetry.get_current_tracker() is None
    tok = A.current_grants.set(frozenset())
    try:
        assert capability_refusal(Capability.FILES) is not None
    finally:
        A.current_grants.reset(tok)


def test_the_turn_loop_reads_the_ledger():
    """Source-level. The ledger is only worth writing if something reads it,
    and the read is one line inside a 700-line turn loop that no unit test
    reaches without standing up STT, the classifier and the LLM.

    Pinned by source rather than left to the live test alone: this is the line
    that decides whether a false claim reaches next session's opening line.
    """
    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    assert "refused_capabilities" in src, (
        "main.py never reads the refusal ledger, so a refusal inside a "
        "planner step still cannot reach security_skip"
    )
    fold = src.index("_refused_anywhere")
    save = src.index("security_skip=_skip_this_turn")
    assert fold < save, (
        "the ledger is folded in after the conversation row is written"
    )


def test_the_ledger_fold_does_not_shadow_the_outer_skip_flag():
    """The bug this pins cost every turn that reached `_save_turn`.

    The fold lives in the nested `_save_turn`, and the first version wrote
    `_security_skip_this_turn = _security_skip_this_turn or _refused_anywhere`.
    Python decides local-vs-closure at compile time from the presence of an
    assignment, so that line made the name a local of `_save_turn` and reading
    it on the same line raised `UnboundLocalError` -- killing the turn after the
    work was done and before the conversation row was written.

    Checked structurally because the failure is invisible to reasoning about the
    line in isolation: it looks like an ordinary read-modify-write, and it is
    one, of a name that belongs to another scope. `tests/
    test_reply_cannot_contradict_the_machine.py` catches the behaviour; this
    catches the shape, and names it.
    """
    import ast

    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    inner = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "_save_turn"]
    assert inner, "_save_turn no longer exists -- move this check with it"

    for fn in inner:
        declared = {name for node in ast.walk(fn)
                    if isinstance(node, ast.Nonlocal) for name in node.names}
        assigned = {t.id for node in ast.walk(fn)
                    if isinstance(node, ast.Assign)
                    for t in node.targets if isinstance(t, ast.Name)}
        assert "_security_skip_this_turn" not in (assigned - declared), (
            "_save_turn assigns the enclosing turn's `_security_skip_this_turn` "
            "without `nonlocal`, which makes it a local and raises "
            "UnboundLocalError on the read. Compute a new name instead."
        )
