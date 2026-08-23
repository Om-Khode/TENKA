"""An incoherent plan is rejected before its first step runs.

The point of "before" is the whole point. Every other check in the planner is
per-step, and per-step is too late: a plan whose step 4 depends on a step that
does not exist still runs steps 1 to 3 first, and those steps send messages,
write files and click things. Discovering the incoherence afterwards is
discovering it after the side effects.

`step_id` and `depends_on` come straight from the model, and a step naming an
unknown tool is *skipped* while the survivors keep their model-assigned ids --
so `depends_on: [2]` outliving step 2 is not a hypothetical, it is the ordinary
consequence of one bad tool name.

The reference cases are the ones that actually bite, because they fail
**quietly**: an unresolvable `$step_N` becomes empty text rather than an error,
so a search step searches for nothing and reports success.

Run with:  py -3.11 -m pytest tests/test_plan_coherence.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.actions.planner.planner import (  # noqa: E402
    PlanStep, _plan_incoherence,
)


def _step(step_id, *, tool="web_search", goal="do a thing",
          depends_on=None, condition=None):
    return PlanStep(step_id=step_id, tool=tool, goal=goal,
                    depends_on=depends_on or [], condition=condition)


# ─── the permitted shapes ────────────────────────────────────────────────────
#
# First, and deliberately. A validator that rejected everything would satisfy
# every rejection test below while making the planner unable to produce any
# plan at all -- a failure that looks like "the LLM is being unhelpful today"
# rather than like a bug.

def test_a_plain_sequential_plan_is_coherent():
    assert _plan_incoherence([_step(1), _step(2), _step(3)]) is None


def test_a_backward_dependency_is_coherent():
    assert _plan_incoherence([
        _step(1),
        _step(2, depends_on=[1], goal="search for $step_1"),
    ]) is None


def test_the_prompts_own_worked_example_is_coherent():
    """`_PLAN_SYSTEM_PROMPT` ships example plans, and a validator that rejected
    them would reject the shape the model was told to produce. Uses the same
    structure as the condition example: step 2 depends on step 1 and reads it in
    both `goal` and `condition`."""
    assert _plan_incoherence([
        _step(1, tool="code_executor", goal="read my messages"),
        _step(2, tool="code_executor", goal="reply to $step_1",
              depends_on=[1], condition="if $step_1 contains 'Mom'"),
    ]) is None


def test_a_step_may_depend_on_several_earlier_steps():
    assert _plan_incoherence([
        _step(1), _step(2),
        _step(3, depends_on=[1, 2], goal="combine $step_1 and $step_2"),
    ]) is None


def test_non_contiguous_ids_are_fine_as_long_as_they_resolve():
    """A skipped bad-tool step leaves a gap in the numbering. The gap itself is
    not the problem -- a dependency pointing *into* it is."""
    assert _plan_incoherence([
        _step(1), _step(4, depends_on=[1], goal="use $step_1"),
    ]) is None


# ─── duplicate ids ───────────────────────────────────────────────────────────

def test_duplicate_step_ids_are_rejected():
    """Both `$step_N` and `depends_on` address steps by id, so with two steps
    sharing one, a reference means whichever the resolver reaches first."""
    why = _plan_incoherence([_step(1), _step(1)])
    assert why is not None and "duplicate" in why


# ─── dependencies ────────────────────────────────────────────────────────────

def test_a_dependency_on_a_missing_step_is_rejected():
    """The shape a skipped unknown-tool step produces. Previously this plan ran
    step 1, then reached step 2 with a dependency that could never resolve."""
    why = _plan_incoherence([_step(1), _step(3, depends_on=[2])])
    assert why is not None and "not in the plan" in why


def test_a_forward_dependency_is_rejected():
    why = _plan_incoherence([_step(1, depends_on=[2]), _step(2)])
    assert why is not None and "runs later" in why


def test_a_self_dependency_is_rejected():
    why = _plan_incoherence([_step(1, depends_on=[1])])
    assert why is not None and "itself" in why


# ─── $step_N references ──────────────────────────────────────────────────────

def test_a_reference_to_a_missing_step_is_rejected():
    """The quiet one. An unresolvable reference becomes empty text, so the step
    runs with a blank where its input should have been and reports success."""
    why = _plan_incoherence([_step(1, goal="search for $step_9")])
    assert why is not None and "$step_9" in why


def test_a_forward_reference_is_rejected():
    why = _plan_incoherence([
        _step(1, goal="search for $step_2"),
        _step(2),
    ])
    assert why is not None and "runs later" in why


def test_a_self_reference_is_rejected():
    why = _plan_incoherence([_step(1, goal="refine $step_1")])
    assert why is not None and "its own output" in why


def test_a_reference_inside_a_condition_counts():
    """`condition` is `"if $step_N contains 'x'"`, evaluated the same way and
    just as unresolvable. Checking `goal` alone would leave half the surface
    open."""
    why = _plan_incoherence([
        _step(1, condition="if $step_5 contains 'yes'"),
    ])
    assert why is not None and "$step_5" in why


# ─── it is actually wired in ─────────────────────────────────────────────────

def test_the_generator_rejects_an_incoherent_plan(monkeypatch):
    """The check exists and is *called*. A validator nothing invokes is the
    most expensive kind of dead code: it reads as coverage.

    Drives the real `_generate_plan` with a canned LLM reply, so what is pinned
    is the wiring rather than the function in isolation.
    """
    import asyncio

    from assistant.actions.planner import planner as pl

    async def _llm(*args, **kwargs):
        # Step 2 depends on a step that is not in the plan.
        return ('[{"step_id": 1, "tool": "web_search", "goal": "a",'
                '  "depends_on": [], "condition": null},'
                ' {"step_id": 2, "tool": "web_search", "goal": "b",'
                '  "depends_on": [7], "condition": null}]')

    # `_generate_plan` does `from ... import memory`, which resolves to the
    # real `assistant.memory` module object -- so the patch goes on that
    # module's attribute, which is where the import will look. Without it the
    # test reads the operator's live conversation history.
    import assistant.memory as memory_mod
    monkeypatch.setattr(memory_mod, "build_recent_context", lambda **kw: "")
    plan = asyncio.run(pl._generate_plan("do two things", _llm))
    assert plan is None, (
        f"an incoherent plan was returned and would have executed step 1: "
        f"{plan}"
    )


def test_the_generator_still_accepts_a_coherent_plan(monkeypatch):
    """The other direction of the wiring. If the hook rejected everything, the
    planner would silently stop working and the test above would still pass."""
    import asyncio

    from assistant.actions.planner import planner as pl

    async def _llm(*args, **kwargs):
        return ('[{"step_id": 1, "tool": "web_search", "goal": "a",'
                '  "depends_on": [], "condition": null},'
                ' {"step_id": 2, "tool": "web_search", "goal": "b $step_1",'
                '  "depends_on": [1], "condition": null}]')

    # `_generate_plan` does `from ... import memory`, which resolves to the
    # real `assistant.memory` module object -- so the patch goes on that
    # module's attribute, which is where the import will look. Without it the
    # test reads the operator's live conversation history.
    import assistant.memory as memory_mod
    monkeypatch.setattr(memory_mod, "build_recent_context", lambda **kw: "")
    plan = asyncio.run(pl._generate_plan("do two things", _llm))
    assert plan is not None and len(plan.steps) == 2, (
        "a valid two-step plan was rejected -- the planner now produces nothing"
    )
