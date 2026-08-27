"""One step loop, not two. TENKA-v2 §17.P8, first half.

`execute_plan` walked the steps with a `while` and an index. `resume_plan`
walked them again with a `for` over a range. Same job, written twice, and the
two copies had drifted apart in seven ways -- every one of them a defect in the
resumed path rather than a deliberate difference:

    abort           a resumed plan could not be cancelled
    status          the overlay went blank for the second half of a plan
    recovered       a repaired step stayed `failed` in the synthesis
    speech          the retry happened silently
    insertion       recovery ran in a nested inline loop instead of being
                    inserted -- and that loop indexed `plan.steps` for a step
                    nothing had inserted, so a recovery step that ended
                    `waiting` raised ValueError instead of suspending
    the `for`       is why it could not insert: mutating the list it indexed
                    would have shifted every later step
    3D             re-planning did not happen (this one is deliberate and
                    stays that way -- see the last test)

None of these are visible from either function alone. They are visible only
when the two are read side by side, which nobody does. So the loop is now one
function that both entry points call, and these tests hold it to that: the
structural pair says there is exactly one, and the behavioural ones say the
resumed path now has each of the things it was missing.

Run with:  py -3.11 -m pytest tests/test_one_step_loop.py -v
"""
import ast
import pathlib
import sys
from unittest.mock import patch

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PLANNER_DIR = _ROOT / "assistant" / "actions" / "planner"


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def planner():
    """The planner module, with suspension and abort state clean either side.

    Both are process-wide singletons: a leaked suspended plan makes the next
    test resume something it never created, and a leaked abort makes it raise.
    """
    from assistant.actions.planner import planner as mod
    from assistant.core.abort import abort

    abort.reset()
    mod.clear_suspended_plan()
    yield mod
    mod.clear_suspended_plan()
    abort.reset()


async def _llm(*a, **k):
    return "synthesized"


def _step(planner, step_id, tool="synthesize", **kw):
    return planner.PlanStep(step_id=step_id, tool=tool,
                            goal=f"step {step_id}", **kw)


def _plan(planner, *steps):
    return planner.Plan(original_goal="do the thing", steps=list(steps))


def _executor_returning(behaviour):
    """Patch `execute_step` where `run_steps` looks it up.

    `run_steps` does `from .executor import execute_step` at call time, so the
    name is read off the real `executor` module every run -- patching the
    attribute there intercepts it. Patching `planner.execute_step` would not:
    there is no such global any more, which is the point of the refactor.
    """
    return patch("assistant.actions.planner.executor.execute_step",
                 new=behaviour)


# ─── structural: there is exactly one loop ───────────────────────────────────

def _calls_named(tree, name):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) == name
                 or getattr(n.func, "attr", None) == name)]


def test_only_one_place_executes_a_step():
    """The property the whole refactor exists for.

    A second call site is how the two loops diverged in the first place: the
    person who added the abort check had no reason to think there was another
    loop that also needed one.
    """
    sites = []
    walked = 0
    for path in sorted(_PLANNER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        walked += 1
        for call in _calls_named(tree, "execute_step"):
            sites.append(f"{path.name}:{call.lineno}")

    assert walked >= 3, "walked nothing -- the planner package moved"
    assert len(sites) == 1, (
        f"a plan step is executed from {len(sites)} places: {sites}. "
        "There must be exactly one, inside run_steps()."
    )


def test_the_one_call_site_is_inside_the_loop():
    """Not merely one call site -- one call site *in the loop*. A single call
    made from `execute_plan` would satisfy the count above and leave
    `resume_plan` with no way to run anything."""
    tree = ast.parse((_PLANNER_DIR / "planner.py").read_text(encoding="utf-8"))
    run_steps = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_steps"),
        None,
    )
    assert run_steps is not None, "run_steps is gone"
    assert _calls_named(run_steps, "execute_step"), (
        "run_steps does not execute a step")


@pytest.mark.parametrize("entry", ["execute_plan", "resume_plan"])
def test_both_entry_points_go_through_the_loop(entry):
    tree = ast.parse((_PLANNER_DIR / "planner.py").read_text(encoding="utf-8"))
    func = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.AsyncFunctionDef) and n.name == entry),
        None,
    )
    assert func is not None, f"{entry} is gone"
    assert _calls_named(func, "run_steps"), (
        f"{entry} does not call run_steps -- it has its own loop again")


# ─── the resumed path gained what it was missing ─────────────────────────────

@pytest.mark.asyncio
async def test_a_resumed_plan_can_be_aborted(planner):
    """The worst of the seven. The stop word worked before the interaction and
    silently did nothing after it, which is the half of a plan most likely to
    be doing something the user just changed their mind about."""
    from assistant.core.abort import abort, UserAborted

    ran = []

    async def _never(step, plan, **kw):
        ran.append(step.step_id)

    plan = _plan(planner, _step(planner, 1), _step(planner, 2))
    plan.steps[0].status = "success"
    planner._suspend_plan(plan, 1, _llm, None, None)

    abort.request_abort("user said stop")
    with _executor_returning(_never):
        with pytest.raises(UserAborted):
            await planner.resume_plan("done")

    assert not ran, f"steps ran after abort: {ran}"


@pytest.mark.asyncio
async def test_a_resumed_plan_reports_progress(planner):
    """Without this the overlay shows nothing from the interaction onward, so
    a plan that is still working is indistinguishable from one that hung."""
    from assistant.io.status_broadcaster import StatusPhase

    seen = []

    async def _ok(step, plan, **kw):
        step.status = "success"

    class _Status:
        def set(self, phase, *, detail="", step=None, **kw):
            seen.append((phase, step))

    plan = _plan(planner, _step(planner, 1), _step(planner, 2))
    plan.steps[0].status = "success"
    planner._suspend_plan(plan, 1, _llm, None, None)

    with _executor_returning(_ok), \
            patch("assistant.io.status_broadcaster.status", new=_Status()):
        await planner.resume_plan("done")

    assert seen, "a resumed plan reported no progress at all"
    assert all(p is StatusPhase.PLANNING for p, _ in seen)
    assert seen[0][1] == (2, 2), (
        f"progress started at {seen[0][1]}, not at the resumed step")


@pytest.mark.asyncio
async def test_a_repaired_step_is_not_reported_as_failed(planner):
    """A step that failed, was recovered, and whose recovery succeeded is
    `recovered`. Leaving it `failed` makes the synthesis report a failure that
    did not survive -- the same lie as claiming a success, pointed the other
    way."""
    async def _ok(step, plan, **kw):
        step.status = "success"

    origin = _step(planner, 1)
    origin.status = "failed"
    origin.error = "boom"
    plan = _plan(planner, origin, _step(planner, 2))
    plan._recovery_origin = 1
    plan._recovery_step_ids = [2]
    planner._suspend_plan(plan, 1, _llm, None, None)

    with _executor_returning(_ok):
        await planner.resume_plan("done")

    assert origin.status == "recovered", (
        f"origin step stayed {origin.status!r} after its recovery succeeded")


@pytest.mark.asyncio
async def test_a_recovery_step_that_waits_suspends_rather_than_crashing(
        planner):
    """The latent crash. The old resumed path ran recovery steps in a nested
    inline loop and then called `plan.steps.index(rs)` on a step nothing had
    inserted into `plan.steps` -- ValueError, mid-plan, on a path only reached
    after a failure. Inserting them into the plan and letting the ordinary loop
    run them is what makes recovery, dependencies and suspension compose."""
    async def _fail_then_wait(step, plan, **kw):
        if step.step_id == 1:
            step.status = "failed"
            step.error = "first attempt failed"
        else:
            step.status = "waiting"
            step.output = "need a hand"

    recovery = [_step(planner, 9), _step(planner, 10)]

    async def _recover(failed_step, plan, llm_func):
        return recovery

    plan = _plan(planner, _step(planner, 1), _step(planner, 2))
    planner._suspend_plan(plan, 0, _llm, None, None)

    with _executor_returning(_fail_then_wait), \
            patch.object(planner, "_attempt_recovery", new=_recover):
        result = await planner.resume_plan("")

    assert result == "need a hand", (
        f"a waiting recovery step returned {result!r} instead of suspending")
    assert planner.has_suspended_plan(), "the plan did not suspend"
    assert recovery[0] in plan.steps, (
        "recovery steps were run without being inserted into the plan -- "
        "which is exactly what made index() raise")


@pytest.mark.asyncio
async def test_the_retry_is_not_silent(planner):
    """`execute_plan` said "let me try a different approach" before recovering
    and `resume_plan` said nothing, so the same failure was narrated or not
    depending on whether the user had been asked a question earlier in the
    plan."""
    spoken = []

    async def _tts(text):
        spoken.append(text)

    async def _fail(step, plan, **kw):
        step.status = "failed"
        step.error = "boom"

    async def _recover(failed_step, plan, llm_func):
        return [_step(planner, 9)]

    plan = _plan(planner, _step(planner, 1), _step(planner, 2))
    planner._suspend_plan(plan, 0, _llm, _tts, None)

    with _executor_returning(_fail), \
            patch.object(planner, "_attempt_recovery", new=_recover):
        await planner.resume_plan("")

    assert any("try" in t.lower() or "again" in t.lower() for t in spoken), (
        f"nothing was said before retrying; spoken: {spoken}")


# ─── the loop's own contract ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_suspended_plan_is_not_synthesized(planner):
    """A suspended plan is healthy and unfinished. Synthesizing it would
    produce a summary of a task still in progress and, worse, would report
    the pending steps as though they had been decided."""
    synthesized = []

    async def _wait(step, plan, **kw):
        step.status = "waiting"
        step.output = "which one?"

    async def _synth(plan, llm_func):
        synthesized.append(plan)
        return "summary"

    plan = _plan(planner, _step(planner, 1), _step(planner, 2))
    planner._suspend_plan(plan, 0, _llm, None, None)

    with _executor_returning(_wait), \
            patch.object(planner, "_synthesize_result", new=_synth):
        result = await planner.resume_plan("")

    assert result == "which one?"
    assert not synthesized, "a suspended plan was synthesized as if finished"


@pytest.mark.asyncio
async def test_a_dependency_failure_skips_what_waits_on_it(planner):
    """Held here rather than only in the `execute_plan` tests because the
    resumed path had its own copy of this and could drift again."""
    async def _fail_first(step, plan, **kw):
        step.status = "failed" if step.step_id == 1 else "success"
        step.error = "boom" if step.step_id == 1 else ""

    async def _no_recovery(failed_step, plan, llm_func):
        return []

    dependent = _step(planner, 2, depends_on=[1])
    plan = _plan(planner, _step(planner, 1), dependent)
    planner._suspend_plan(plan, 0, _llm, None, None)

    with _executor_returning(_fail_first), \
            patch.object(planner, "_attempt_recovery", new=_no_recovery):
        await planner.resume_plan("")

    assert dependent.status == "skipped"
    assert "dependency step 1 failed" in dependent.error


@pytest.mark.asyncio
async def test_the_loop_starts_where_it_is_told(planner):
    """`run_steps` is the seam, and `start_index` is the only thing the two
    entry points disagree about. If it were ignored, a resumed plan would
    re-run every step the user already sat through."""
    ran = []

    async def _ok(step, plan, **kw):
        ran.append(step.step_id)
        step.status = "success"

    plan = _plan(planner, _step(planner, 1), _step(planner, 2),
                 _step(planner, 3))

    with _executor_returning(_ok):
        result = await planner.run_steps(plan, 2, llm_func=_llm)

    assert ran == [3], f"started at the wrong step: ran {ran}"
    assert not result.suspended


# ─── what stays different, on purpose ────────────────────────────────────────

def test_3d_replanning_stays_out_of_the_loop():
    """The one divergence that is not a defect, asserted so it is not
    "fixed" later by someone tidying up.

    3D re-planning writes a continuation from the original goal when a plan
    hits the step limit. A resumed plan is already a continuation, and it has
    no `_depth` to stop the recursion -- so it belongs to `execute_plan`, after
    the loop, and not inside it.
    """
    tree = ast.parse((_PLANNER_DIR / "planner.py").read_text(encoding="utf-8"))
    by_name = {n.name: n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef)}

    assert not _calls_named(by_name["run_steps"], "execute_plan"), (
        "run_steps re-plans; a resumed plan would recurse without a depth "
        "guard")
    assert _calls_named(by_name["execute_plan"], "execute_plan"), (
        "3D re-planning is gone from execute_plan")
