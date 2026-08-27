"""An executor runs a step. It does not decide what the step meant.

P5. A plan step's meaning is a string today: `actions/planner/executor.py`
builds `params = {param_key: resolved_goal}` from a manifest-chosen key, and
`automation/router.py` then re-reads that sentence with its own regexes --
`\\b(open|launch|start|run)\\s+(\\w+)`, `\\bsearch\\s+(for|on|the)`, and a dozen
more. The user's phrasing is parsed at least twice by components with no shared
vocabulary, and the second parse can disagree with the first.

`brain/executor.py` is the seam where that stops. What this file pins is the
five properties P5 names, each as its own test, because an aggregate would pass
while any one of them regressed:

- an executor never calls an intent classifier
- an executor never reads the original user utterance *to decide*
- user-pinned constraint values reach the adapter byte-identical
- an adapter with no route returns `UNSUPPORTED`, distinct from `FAILED`
- `actions.execute()` is still the only handler-resolution site

The scope this does **not** cover is stated in the module's own docstring:
`goal` survives as a payload field and the adapters still read it. Deleting it
would rewrite roughly forty files at once. What changed is that structure
decides and the string follows.

Run with:  py -3.11 -m pytest tests/test_brain_executor.py -v
"""
import ast
import inspect
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.brain.executor import Executor  # noqa: E402
from assistant.brain.task import TaskStep  # noqa: E402
from assistant.core.verdict import (  # noqa: E402
    ObservationKind, Outcome, Verdict,
)

_EXECUTOR_PY = _ROOT / "assistant" / "brain" / "executor.py"


def _recorder(result="done"):
    """A dispatch that records what it was handed and returns `result`."""
    seen = {}

    async def _dispatch(**kwargs):
        seen.update(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    return _dispatch, seen


def _step(**kw):
    kw.setdefault("step_id", 1)
    kw.setdefault("intent", "create_note")
    return TaskStep(**kw)


# ─── the step's structure is what decides ────────────────────────────────────

@pytest.mark.asyncio
async def test_structured_parameters_reach_the_adapter():
    dispatch, seen = _recorder()
    step = _step(operation="open",
                 parameters={"target": "notepad", "mode": "maximised"})

    await Executor(dispatch).run(step)

    assert seen["intent"] == "create_note"
    assert seen["params"]["target"] == "notepad"
    assert seen["params"]["mode"] == "maximised"


@pytest.mark.asyncio
async def test_the_goal_rides_along_without_replacing_the_structure():
    """The compromise P5 states outright. Adapters still read `goal`, so it is
    passed -- but it does not overwrite a structured field of the same name,
    which is what "the string is data, not the meaning" has to mean in code."""
    dispatch, seen = _recorder()
    step = _step(parameters={"goal": "structured"}, goal="the user's sentence")

    await Executor(dispatch).run(step)

    assert seen["params"]["goal"] == "structured", (
        "the raw utterance overwrote a structured parameter")


@pytest.mark.asyncio
async def test_a_step_with_no_structure_still_carries_its_goal():
    """The migration is incremental: most steps today have only a goal, and
    dropping it would break every one of them."""
    dispatch, seen = _recorder()

    await Executor(dispatch).run(_step(goal="open notepad"))

    assert seen["params"]["goal"] == "open notepad"


# ─── pinned constraints are hard ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_pinned_constraint_reaches_the_adapter_byte_identical():
    """`CLAUDE.md`'s gotcha: "mobile as 99999" is a HARD constraint. It is not
    rounded, not normalised, not replaced by something more plausible."""
    dispatch, seen = _recorder()
    step = _step(parameters={"mobile": "0000000000"},
                 constraints={"mobile": "99999"})

    await Executor(dispatch).run(step)

    assert seen["params"]["mobile"] == "99999", (
        f"a pinned value was substituted: {seen['params']['mobile']!r}")


@pytest.mark.asyncio
async def test_a_constraint_beats_a_parameter_whatever_the_order():
    """Applied last, so it wins regardless of what `parameters` holds. A
    merge that let `parameters` win would satisfy the test above only while
    the two happened not to collide."""
    dispatch, seen = _recorder()
    step = _step(parameters={"seat": "any", "date": "tomorrow"},
                 constraints={"seat": "14A"})

    await Executor(dispatch).run(step)

    assert seen["params"]["seat"] == "14A"
    assert seen["params"]["date"] == "tomorrow", (
        "applying constraints dropped the unconstrained parameters")


@pytest.mark.asyncio
async def test_a_caller_supplied_constraint_is_applied_too():
    """A Task's constraints apply to every step in it, and the step need not
    repeat them."""
    dispatch, seen = _recorder()

    await Executor(dispatch).run(_step(parameters={"mobile": "0"}),
                                 constraints={"mobile": "99999"})

    assert seen["params"]["mobile"] == "99999"


def test_the_adapter_cannot_mutate_the_pin_for_the_next_step():
    """Copied, not referenced. An adapter that normalises its params in place
    would otherwise rewrite the constraint for every later step."""
    pinned = {"mobile": "99999"}
    step = _step(constraints=pinned)

    params = Executor.build_params(step)
    params["mobile"] = "0000000000"

    assert pinned["mobile"] == "99999", "the adapter mutated the pin itself"
    assert Executor.build_params(step)["mobile"] == "99999"


# ─── UNSUPPORTED is not FAILED ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_unroutable_intent_is_unsupported_not_failed():
    """**The distinction that changes behaviour.** A planner reading "no route
    exists" picks another tool; one reading "it failed" retries the same one.
    Collapsing them turns a missing adapter into an apology for a failure that
    never happened."""
    dispatch, seen = _recorder()

    verdict = await Executor(dispatch).run(
        _step(intent="an_intent_nobody_registered"))

    assert verdict.outcome is Outcome.UNSUPPORTED
    assert verdict.outcome is not Outcome.FAILED
    assert not seen, "an unroutable step was dispatched anyway"


@pytest.mark.asyncio
async def test_a_step_that_ran_and_failed_is_failed_not_unsupported():
    """The other direction, and the one a lenient mapping breaks: everything
    reported as UNSUPPORTED would satisfy the test above perfectly."""
    dispatch, _ = _recorder("Error: could not open the file")

    verdict = await Executor(dispatch).run(_step())

    assert verdict.outcome is Outcome.FAILED
    assert verdict.outcome is not Outcome.UNSUPPORTED


@pytest.mark.asyncio
async def test_a_handler_that_returns_nothing_is_unverified_not_succeeded():
    """Absence of an exception is not evidence. `UNVERIFIED` says nothing
    confirmed the effect; `SUCCEEDED` would be a claim nobody made."""
    dispatch, _ = _recorder(None)

    verdict = await Executor(dispatch).run(_step())

    assert verdict.outcome is Outcome.UNVERIFIED
    assert not verdict.outcome.is_evidence_of_success


def test_nothing_observed_is_not_the_same_as_nothing_changed():
    """`NOTHING_CHANGED` is a claim: someone looked and the state was the same.
    `NOT_OBSERVED` is the weaker and far more common fact. Reporting the first
    where the second is true turns "no evidence" into "evidence of no effect",
    which is the inversion the Outcome ladder exists to prevent."""
    verdict = Executor.describe(None)
    assert verdict.observation.kind is ObservationKind.NOT_OBSERVED
    assert verdict.observation.kind is not ObservationKind.NOTHING_CHANGED


@pytest.mark.asyncio
async def test_a_successful_step_is_succeeded():
    """The control. A mapping that never returns SUCCEEDED passes every test
    above and makes the assistant useless."""
    dispatch, _ = _recorder("Created the note.")

    verdict = await Executor(dispatch).run(_step())

    assert verdict.outcome is Outcome.SUCCEEDED
    assert verdict.outcome.is_evidence_of_success
    assert "Created the note." in verdict.observation.detail


@pytest.mark.asyncio
async def test_a_raising_handler_is_failed_and_described_not_propagated():
    dispatch, _ = _recorder(RuntimeError("adapter exploded"))

    verdict = await Executor(dispatch).run(_step())

    assert verdict.outcome is Outcome.FAILED
    assert "adapter exploded" in verdict.observation.detail


@pytest.mark.asyncio
async def test_an_abort_is_not_swallowed():
    """The operator pressed ESC. Describing it as a failed step is what made a
    second ESC necessary once already -- the planner's recovery path treats a
    failure as something to retry."""
    from assistant.core.abort import UserAborted

    dispatch, _ = _recorder(UserAborted("stop"))

    with pytest.raises(UserAborted):
        await Executor(dispatch).run(_step())


# ─── structural: no reinterpretation is reachable from here ──────────────────

_CLASSIFIER_NAMES = frozenset({
    "detect_intent", "ask_for_intent", "classify_intent", "pre_route",
    "needs_planning",
})


def test_the_executor_never_calls_a_classifier():
    """P5's first required mutation. A classifier here would mean the intent
    the authority check approved and the intent that ran could differ -- the
    shape of every confused-deputy bug in this tree."""
    tree = ast.parse(_EXECUTOR_PY.read_text(encoding="utf-8"))

    called = {
        (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert called, "walked nothing -- the parse found no calls at all"

    offenders = called & _CLASSIFIER_NAMES
    assert not offenders, (
        f"the executor reinterprets its step: {sorted(offenders)}. If a step "
        "is ambiguous the answer is a better plan, not a second opinion "
        "invented at execution time.")

    imported = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    }
    assert not any(m and ("intent" in m or "regex_router" in m)
                   for m in imported), (
        f"the executor imports a classifier module: {sorted(m for m in imported if m)}")


def test_the_executor_never_branches_on_the_utterance():
    """The second property, and the subtle one. Passing `goal` through is
    allowed and necessary; *deciding* on it is the re-interpretation being
    removed. So: `goal` may be read, never compared, matched or split."""
    tree = ast.parse(_EXECUTOR_PY.read_text(encoding="utf-8"))

    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and n.value == "goal"]
    assert reads, "walked nothing -- `goal` is not mentioned in the executor"

    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and "goal" in ast.unparse(node):
            bad.append(ast.unparse(node))
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) in (
                    "startswith", "endswith", "split", "lower", "search",
                    "match", "findall")
                and "goal" in ast.unparse(node)):
            bad.append(ast.unparse(node))

    # `if goal:` -- an emptiness check, not a reading of what it says -- is the
    # one permitted form, and it is not a Compare so it never reaches here.
    assert not bad, (
        f"the executor decides from the user's sentence: {bad}")


def test_dispatch_still_goes_through_the_one_resolution_site():
    """`actions.execute()` is the EXECUTE enforcement point. An executor that
    resolved a handler itself would be a second door, which is exactly what
    6a.5 spent a milestone closing."""
    src = inspect.getsource(Executor.run)
    assert "tool_registry.get(" not in src, (
        "the executor resolves handlers itself, bypassing the gate")

    default = inspect.getsource(
        sys.modules["assistant.brain.executor"]._default_dispatch)
    assert "from ..actions import execute" in default, (
        "the default dispatch is no longer actions.execute")


def test_support_is_asked_of_the_registry_without_resolving():
    """`has` rather than `get`: answering "is there a route" must not become a
    way to obtain the handler and call it around the gate."""
    src = inspect.getsource(Executor.supports)
    assert "tool_registry.has(" in src
    assert "tool_registry.get(" not in src


def test_a_real_registered_intent_is_supported():
    """Anti-vacuity for the two tests above. `supports` returning False for
    everything would make the UNSUPPORTED test pass and the executor inert."""
    import assistant.actions  # noqa: F401  (registers the handlers)

    assert Executor.supports("create_note") is True
    assert Executor.supports("an_intent_nobody_registered") is False
    assert Executor.supports("") is False


def test_the_verdict_type_is_the_shared_one():
    """Not a local shape that happens to have the same fields -- the whole
    point of `core/verdict.py` was that six call sites each decided what `ok`
    meant."""
    verdict = Executor.describe("done")
    assert isinstance(verdict, Verdict)
    assert isinstance(verdict.outcome, Outcome)
