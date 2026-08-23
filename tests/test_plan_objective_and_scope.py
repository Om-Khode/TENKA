"""Recovery may not rewrite what was asked, nor widen what it may use.

Two properties, one subject: a plan that fails partway is handed to a model and
asked for another approach, and both the objective and the permitted means have
to survive that round trip.

**The objective.** `Plan.original_goal` is the one field the untrusted-content
fence treats as the user's own words -- `executor.py` copies it into
`_planner_goal` for two tools declared to accept no prior-step output at all.
6a.5's H3 review found the 3D continuation splicing a synthesize step's output
(laundered file content) into it. The fix stands; these tests are the guard on
it, because the same three lines would reintroduce it invisibly.

**The means.** `Capability` is a set, not a lattice, so "recovery must not
widen scope" needs a definition that is actually checkable. This is it: the
plan as written asked for a particular set of permissions, and a recovery step
requiring one the plan never asked for is doing something the original was
never authorised to do -- regardless of whether this caller happens to hold it.

Run with:  py -3.11 -m pytest tests/test_plan_objective_and_scope.py -v
"""
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import assistant.actions as A  # noqa: E402
from assistant.actions.planner.planner import (  # noqa: E402
    Plan, PlanStep, _plan_capability_footprint, _recovery_tool_in_scope,
)
from assistant.core.capabilities import Capability  # noqa: E402

_PLANNER_SRC = (_ROOT / "assistant" / "actions" / "planner" / "planner.py").read_text(
    encoding="utf-8")


@pytest.fixture()
def clean_context():
    g = A.current_grants.set(None)
    r = A.current_raise_context.set(None)
    yield
    A.current_grants.reset(g)
    A.current_raise_context.reset(r)


def _plan(*tools):
    return Plan(
        original_goal="find the invoice and email it",
        steps=[PlanStep(step_id=i, tool=t, goal="g")
               for i, t in enumerate(tools, start=1)],
    )


# ─── The objective survives ──────────────────────────────────────────────────

def test_original_goal_is_never_assigned_after_construction():
    """Source sweep. `Plan` is a mutable dataclass -- `plan.status` is assigned
    in several places -- so nothing structurally stops a future line from
    editing the objective too. Today none does, and that is worth pinning
    while it is still true and cheap.
    """
    assignments = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(_PLANNER_SRC.splitlines(), 1)
        if re.search(r"\.original_goal\s*=(?!=)", line.split("#", 1)[0])
    ]
    assert not assignments, (
        f"the plan's objective is rewritten after construction: {assignments}. "
        f"It is the one field the fence treats as the user's own words."
    )


def test_the_continuation_never_splices_a_step_output_into_the_goal():
    """The H3 regression guard.

    The continuation used to be `f"{goal}\\n\\nProgress so far: {last_step.output}"`,
    which made a synthesize step's output -- content read off a file or a page --
    part of the next plan's `original_goal`, and from there into `_planner_goal`
    on `browser_action`. The progress now travels beside the goal as fenced
    data via `_prior_context`.

    Checked at source because reproducing it needs a three-step plan ending in
    a successful synthesize, a live LLM and a depth-0 entry -- and the thing to
    catch is the *shape* of one expression.
    """
    call = _PLANNER_SRC.index("continuation_result = await execute_plan(")
    first_arg = _PLANNER_SRC[call:_PLANNER_SRC.index(")", call)]

    assert "goal" in first_arg, "the continuation lost the user's goal entirely"
    assert ".output" not in first_arg.split("_prior_context")[0], (
        f"a step's output is spliced into the continuation's objective: "
        f"{first_arg!r}"
    )
    assert "_prior_context" in first_arg, (
        "the progress no longer travels as fenced data beside the goal"
    )


def test_the_continuation_carries_the_users_words_verbatim():
    """The user's goal is a prefix, and what follows is a fixed, code-written
    instruction -- not interpolated text. A constant suffix is a deliberate
    design (the continuation has to say what it is for); the property that
    matters is that nothing *variable* joins it.
    """
    call = _PLANNER_SRC.index("continuation_result = await execute_plan(")
    first_arg = _PLANNER_SRC[call:_PLANNER_SRC.index(")", call)]
    fstring = re.search(r'f"([^"]*)"', first_arg)
    assert fstring, f"the continuation goal is no longer a literal: {first_arg!r}"

    body = fstring.group(1)
    assert body.startswith("{goal}"), (
        f"the user's words are no longer the prefix: {body!r}")
    interpolations = set(re.findall(r"\{([^}]+)\}", body))
    assert interpolations == {"goal"}, (
        f"the continuation objective interpolates {interpolations - {'goal'}} "
        f"besides the user's goal"
    )


# ─── The footprint ───────────────────────────────────────────────────────────

def test_the_footprint_is_the_union_of_the_steps():
    p = _plan("web_search", "file_task")
    assert _plan_capability_footprint(p) == frozenset(
        {Capability.CHAT_SEND, Capability.FILES})


def test_an_unlisted_tool_widens_the_footprint_rather_than_narrowing_it():
    """Fail-closed direction check. A tool with no row in
    `REQUIRED_CAPABILITY` contributes `DEFAULT_REQUIRED` (EXECUTE), so the
    unknown case makes the footprint *larger* -- it can never be the thing that
    quietly shrinks what a recovery is compared against."""
    from assistant.core.intent_capabilities import DEFAULT_REQUIRED
    p = _plan("a_tool_from_the_future")
    assert _plan_capability_footprint(p) == frozenset({DEFAULT_REQUIRED})


def test_an_empty_plan_has_an_empty_footprint_and_admits_nothing(clean_context):
    """Degenerate but reachable, and it fails in the right direction: an empty
    footprint contains nothing, so every recovery tool is out of scope."""
    A.current_grants.set(frozenset(Capability))
    assert _plan_capability_footprint(_plan()) == frozenset()
    assert not _recovery_tool_in_scope("web_search", frozenset())


# ─── Scope: what a recovery step may use ─────────────────────────────────────

def test_a_recovery_may_reuse_a_capability_the_plan_already_asked_for(clean_context):
    """**The answer, not the refusal.** A rule that dropped everything would
    satisfy every rejection test below while silently deleting recovery
    entirely -- which is the shape of failure this project keeps paying for.
    """
    A.current_grants.set(frozenset({Capability.CHAT_SEND}))
    footprint = _plan_capability_footprint(_plan("web_search"))
    assert _recovery_tool_in_scope("browse_url", footprint), (
        "a same-capability alternative was dropped; recovery now does nothing"
    )


def test_a_recovery_may_not_introduce_a_capability_the_plan_never_asked_for(clean_context):
    """The widening this closes: a `web_search` step (CHAT_SEND) fails and the
    replanner proposes `code_executor` (EXECUTE) -- running arbitrary Python to
    recover a web search. Refused in a plan that never had an EXECUTE step,
    even though the caller holds EXECUTE."""
    A.current_grants.set(frozenset(Capability))          # holds everything
    footprint = _plan_capability_footprint(_plan("web_search"))
    assert not _recovery_tool_in_scope("code_executor", footprint)


def test_the_same_recovery_is_allowed_when_the_plan_already_had_that_step(clean_context):
    """The other half of the rule, and what keeps it from being a blanket ban:
    a plan that already runs code may recover a failed step with code. The
    footprint is about what the plan asked for, not about ranking tools."""
    A.current_grants.set(frozenset(Capability))
    footprint = _plan_capability_footprint(_plan("web_search", "code_executor"))
    assert _recovery_tool_in_scope("code_executor", footprint)


def test_a_recovery_this_caller_cannot_run_never_enters_the_plan(clean_context):
    """Second, independent narrowing. Queuing a step nobody may dispatch is
    dishonest about what is planned, and it hands the refused step its own
    recovery round -- a refusal laundered into a failure."""
    A.current_grants.set(frozenset({Capability.CHAT_SEND}))   # no EXECUTE
    footprint = _plan_capability_footprint(_plan("code_executor"))
    assert not _recovery_tool_in_scope("code_executor", footprint)


def test_unset_grants_admit_no_recovery_step(clean_context):
    """`None` refuses, here as everywhere. The absence of a decision is not a
    decision to allow."""
    assert A.current_grants.get() is None
    footprint = _plan_capability_footprint(_plan("web_search"))
    assert not _recovery_tool_in_scope("web_search", footprint)
