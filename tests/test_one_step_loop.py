"""One step loop, in one place, above `actions`. §17.P8.

Two properties, and the file is in two halves because they fail differently.

**One loop.** `execute_plan` walked the steps with a `while` and an index;
`resume_plan` walked them again with a `for` over a range. Same job, written
twice, drifted apart in seven ways -- every one a defect in the resumed path:

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
    3D              re-planning did not happen (deliberate; see the last test)

None of these are visible from either function alone. They are visible only
when the two are read side by side, which nobody does.

**And it is above `actions` now.** The planner decides what the steps are;
`brain/plan_runner.py` runs them. The structural tests hold both halves: there
is exactly one call site, and it is not in `actions/planner/` any more.
"""
import ast
import pathlib
import sys
import time
from unittest.mock import patch

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PLANNER_DIR = _ROOT / "assistant" / "actions" / "planner"
_RUNNER = _ROOT / "assistant" / "brain" / "plan_runner.py"
_PLANNER_MOD = "assistant.actions.planner.planner"


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def runner():
    """The plan runner, with suspension and abort state clean either side.

    Both are process-wide singletons: a leaked suspended plan makes the next
    test resume something it never created, and a leaked abort makes it raise.
    """
    from assistant.brain import plan_runner
    from assistant.core.abort import abort

    abort.reset()
    plan_runner.clear_suspended_plan()
    yield plan_runner
    plan_runner.clear_suspended_plan()
    abort.reset()


async def _llm(*a, **k):
    return "synthesized"


def _step(step_id, tool="synthesize", **kw):
    from assistant.actions.planner.planner import PlanStep
    return PlanStep(step_id=step_id, tool=tool, goal=f"step {step_id}", **kw)


def _plan(*steps):
    from assistant.actions.planner.planner import Plan
    return Plan(original_goal="do the thing", steps=list(steps))


def _executor_returning(behaviour):
    """Patch `execute_step` where the loop looks it up.

    `run_steps` does `from ..actions.planner.executor import execute_step` at
    call time, so the name is read off the real `executor` module every run --
    patching the attribute there intercepts it.
    """
    return patch("assistant.actions.planner.executor.execute_step",
                 new=behaviour)


# ─── structural: exactly one loop, and it is not in actions/ ─────────────────

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
    for path in sorted(_PLANNER_DIR.glob("*.py")) + [_RUNNER]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        walked += 1
        for call in _calls_named(tree, "execute_step"):
            sites.append(f"{path.name}:{call.lineno}")

    assert walked >= 4, "walked nothing -- the modules moved"
    assert len(sites) == 1, (
        f"a plan step is executed from {len(sites)} places: {sites}. "
        "There must be exactly one, inside run_steps()."
    )
    assert sites[0].startswith("plan_runner.py"), (
        f"steps are executed from {sites[0]} -- §17.P8 puts the loop above "
        "actions/, and the planner does not execute")


def test_the_planner_package_cannot_run_a_plan():
    """P8's deliverable, as an absence.

    `actions` sits below `brain`, so this is not style: a module down there
    that reached for the loop would be reaching across a layer, and the
    breakage would surface as an ImportError somewhere unrelated.
    """
    offenders = []
    for path in sorted(_PLANNER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in ("execute_step", "run_steps"):
            for call in _calls_named(tree, name):
                offenders.append(f"{path.name}:{call.lineno} -> {name}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name in ("execute_plan", "resume_plan", "run_steps"):
                offenders.append(f"{path.name}:{node.lineno} defines "
                                 f"{node.name}")
    assert not offenders, (
        f"actions/planner can still run a plan: {offenders}")


def test_the_loop_never_imports_io():
    """`brain -> io` is a forbidden contract, and the loop wanted two things
    from it: the overlay's step chips and a voice for the retry. Both are
    injected. `lint-imports` enforces this too -- asserted here as well because
    a contract that only lives in a config file is one `ignore_imports` away
    from gone.
    """
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("io") or ".io." in node.module or \
                    node.module.endswith(".io"):
                bad.append(f"line {node.lineno}: {node.module}")
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            if node.module.split(".")[0] == "io":
                bad.append(f"line {node.lineno}: relative {node.module}")
    assert not bad, f"the loop reaches into io/: {bad}"


@pytest.mark.parametrize("entry", ["run", "resume"])
def test_both_entry_points_go_through_the_loop(entry):
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    func = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.AsyncFunctionDef) and n.name == entry),
        None,
    )
    assert func is not None, f"{entry} is gone"
    assert _calls_named(func, "run_steps"), (
        f"{entry} does not call run_steps -- it has its own loop again")


def test_a_plan_is_obtained_through_the_capability_gate():
    """`actions/__init__.py` is the only site in the tree that resolves a
    handler, and it is where the EXECUTE check lives. Importing `_generate_plan`
    directly would let a caller with no authority still make TENKA plan the
    work -- spending model calls, and handing the loop a plan it should never
    have been given. This was written the wrong way first."""
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef) and n.name == "run")
    rendered = ast.unparse(run)
    assert '_actions.execute(' in rendered, (
        "run() does not go through actions.execute -- the capability gate is "
        "bypassed")
    assert "_generate_plan" not in rendered, (
        "run() imports the generator directly, around the gate")


# ─── the resumed path gained what it was missing ─────────────────────────────

@pytest.mark.asyncio
async def test_a_resumed_plan_can_be_aborted(runner):
    """The worst of the seven. Holding ESC worked before the interaction and
    silently did nothing after it -- on the half of a plan a user is most
    likely to have changed their mind about."""
    from assistant.core.abort import abort

    ran = []

    async def _never(step, plan, **kw):
        ran.append(step.step_id)

    plan = _plan(_step(1), _step(2))
    plan.steps[0].status = "success"
    runner._suspend_plan(plan, 1, _llm, None, None)

    abort.request_abort("esc_hold")
    with _executor_returning(_never):
        result = await runner.resume("done")

    assert not ran, f"steps ran after abort: {ran}"
    assert result, "an aborted resume said nothing at all"
    assert plan.status == "failed"


@pytest.mark.asyncio
async def test_an_aborted_resume_does_not_escape_as_an_exception(runner):
    """Where the exception lands, not whether it is raised.

    `run()` raises `UserAborted` and survives because `main.py` wraps the
    branch it is called from. `resume()` is called straight from the pending
    epilogue, which catches nothing. Adding the abort check without asking that
    question would have turned an ignored ESC into an unhandled exception,
    which is worse than the bug it fixed.
    """
    from assistant.core.abort import abort, UserAborted

    async def _never(step, plan, **kw):
        raise AssertionError("should not run")

    plan = _plan(_step(1), _step(2))
    plan.steps[0].status = "success"
    runner._suspend_plan(plan, 1, _llm, None, None)

    abort.request_abort("esc_hold")
    with _executor_returning(_never):
        try:
            result = await runner.resume("done")
        except UserAborted:  # pragma: no cover - the regression
            pytest.fail(
                "UserAborted escaped resume(); main.py has no handler for it "
                "at that call site and the turn would die")

    assert isinstance(result, str) and result.strip()


@pytest.mark.asyncio
async def test_a_resumed_plan_reports_progress(runner):
    """Without this the overlay shows nothing from the interaction onward, so
    a plan that is still working is indistinguishable from one that hung.

    The callback has to survive the suspension, which is its own way to get
    this wrong: a loop that reports progress correctly and is handed `None`
    on resume is exactly as blank as one that never reported.
    """
    seen = []

    async def _ok(step, plan, **kw):
        step.status = "success"

    def _progress(detail, index, total):
        seen.append((detail, index, total))

    plan = _plan(_step(1), _step(2))
    plan.steps[0].status = "success"
    runner._suspend_plan(plan, 1, _llm, None, None, _progress)

    with _executor_returning(_ok):
        await runner.resume("done")

    assert seen, "a resumed plan reported no progress at all"
    assert seen[0][1:] == (2, 2), (
        f"progress started at {seen[0][1:]}, not at the resumed step")


@pytest.mark.asyncio
async def test_a_repaired_step_is_not_reported_as_failed(runner):
    """A step that failed, was recovered, and whose recovery succeeded is
    `recovered`. Leaving it `failed` makes the synthesis report a failure that
    did not survive -- the same lie as claiming a success, pointed the other
    way."""
    async def _ok(step, plan, **kw):
        step.status = "success"

    origin = _step(1)
    origin.status = "failed"
    origin.error = "boom"
    plan = _plan(origin, _step(2))
    plan._recovery_origin = 1
    plan._recovery_step_ids = [2]
    runner._suspend_plan(plan, 1, _llm, None, None)

    with _executor_returning(_ok):
        await runner.resume("done")

    assert origin.status == "recovered", (
        f"origin step stayed {origin.status!r} after its recovery succeeded")


@pytest.mark.asyncio
async def test_a_recovery_step_that_waits_suspends_rather_than_crashing(
        runner):
    """The latent crash. The old resumed path ran recovery steps in a nested
    inline loop and then called `plan.steps.index(rs)` on a step nothing had
    inserted into `plan.steps` -- ValueError, mid-plan, on a path only reached
    after a failure. Inserting them and letting the ordinary loop run them is
    what makes recovery, dependencies and suspension compose."""
    async def _fail_then_wait(step, plan, **kw):
        if step.step_id == 1:
            step.status = "failed"
            step.error = "first attempt failed"
        else:
            step.status = "waiting"
            step.output = "need a hand"

    recovery = [_step(9), _step(10)]

    async def _recover(failed_step, plan, llm_func):
        return recovery

    plan = _plan(_step(1), _step(2))
    runner._suspend_plan(plan, 0, _llm, None, None)

    with _executor_returning(_fail_then_wait), \
            patch(f"{_PLANNER_MOD}._attempt_recovery", new=_recover):
        result = await runner.resume("")

    assert result == "need a hand", (
        f"a waiting recovery step returned {result!r} instead of suspending")
    assert runner.has_suspended_plan(), "the plan did not suspend"
    assert recovery[0] in plan.steps, (
        "recovery steps were run without being inserted into the plan -- "
        "which is exactly what made index() raise")


@pytest.mark.asyncio
async def test_the_retry_is_not_silent(runner):
    """One path narrated the retry and the other did not, so the same failure
    was explained or not depending on whether the user had been asked a
    question earlier in the plan."""
    spoken = []

    async def _speak(text):
        spoken.append(text)

    async def _fail(step, plan, **kw):
        step.status = "failed"
        step.error = "boom"

    async def _recover(failed_step, plan, llm_func):
        return [_step(9)]

    plan = _plan(_step(1), _step(2))
    runner._suspend_plan(plan, 0, _llm, _speak, None)

    with _executor_returning(_fail), \
            patch(f"{_PLANNER_MOD}._attempt_recovery", new=_recover):
        await runner.resume("")

    assert any("try" in t.lower() or "again" in t.lower() for t in spoken), (
        f"nothing was said before retrying; spoken: {spoken}")


@pytest.mark.asyncio
async def test_the_loop_runs_without_a_voice_or_an_overlay(runner):
    """`speak=None` and `progress=None` are a working configuration, not a
    degraded one -- which is what makes a remote turn safe to run with no
    local voice attached, and what let every test above be written without
    standing up a bridge."""
    ran = []

    async def _ok(step, plan, **kw):
        ran.append(step.step_id)
        step.status = "success"

    plan = _plan(_step(1), _step(2))

    with _executor_returning(_ok):
        result = await runner.run_steps(plan, 0, llm_func=_llm)

    assert ran == [1, 2]
    assert not result.suspended


# ─── the loop's own contract ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_suspended_plan_is_not_synthesized(runner):
    """A suspended plan is healthy and unfinished. Synthesizing it would
    summarise a task still in progress and report pending steps as decided."""
    synthesized = []

    async def _wait(step, plan, **kw):
        step.status = "waiting"
        step.output = "which one?"

    async def _synth(plan, llm_func):
        synthesized.append(plan)
        return "summary"

    plan = _plan(_step(1), _step(2))
    runner._suspend_plan(plan, 0, _llm, None, None)

    with _executor_returning(_wait), \
            patch(f"{_PLANNER_MOD}._synthesize_result", new=_synth):
        result = await runner.resume("")

    assert result == "which one?"
    assert not synthesized, "a suspended plan was synthesized as if finished"


@pytest.mark.asyncio
async def test_a_dependency_failure_skips_what_waits_on_it(runner):
    async def _fail_first(step, plan, **kw):
        step.status = "failed" if step.step_id == 1 else "success"
        step.error = "boom" if step.step_id == 1 else ""

    async def _no_recovery(failed_step, plan, llm_func):
        return []

    dependent = _step(2, depends_on=[1])
    plan = _plan(_step(1), dependent)
    runner._suspend_plan(plan, 0, _llm, None, None)

    with _executor_returning(_fail_first), \
            patch(f"{_PLANNER_MOD}._attempt_recovery", new=_no_recovery):
        await runner.resume("")

    assert dependent.status == "skipped"
    assert "dependency step 1 failed" in dependent.error


@pytest.mark.asyncio
async def test_the_loop_starts_where_it_is_told(runner):
    """`start_index` is the only thing the two entry points disagree about. If
    it were ignored, a resumed plan would re-run every step the user already
    sat through."""
    ran = []

    async def _ok(step, plan, **kw):
        ran.append(step.step_id)
        step.status = "success"

    plan = _plan(_step(1), _step(2), _step(3))

    with _executor_returning(_ok):
        result = await runner.run_steps(plan, 2, llm_func=_llm)

    assert ran == [3], f"started at the wrong step: ran {ran}"
    assert not result.suspended


# ─── an answered step is not a finished step ─────────────────────────────────

@pytest.mark.asyncio
async def test_an_answered_step_is_not_offered_as_a_result(runner):
    """Found by live testing, and it made her state a falsehood.

    A step that suspends resolves to `success` so the steps behind it are not
    skipped -- but what it carries is the *exchange*, not a result. Step 1 here
    said "I couldn't find model.vroid, fast search or deep?"; the synthesis was
    handed it under the heading "[file_task] produced:" and reported "I found
    the model.vroid file and opened it."
    """
    seen = {}

    async def _capture(prompt, **kw):
        seen["prompt"] = prompt
        return "summary"

    async def _ok(step, plan, **kw):
        step.status = "success"
        step.output = "opened it"

    plan = _plan(_step(1, tool="file_task"), _step(2))
    plan.steps[0].status = "waiting"
    runner._suspend_plan(plan, 1, _capture, None, None)

    with _executor_returning(_ok):
        await runner.resume(
            "Starting a fast 3-level search. I'll let you know.")

    prompt = seen["prompt"]
    assert "Starting a fast 3-level search" in prompt, "the test set nothing up"
    assert "[file_task] produced" not in prompt, (
        "the interaction was offered to the synthesis as the step's product")
    assert "did NOT finish" in prompt, (
        "nothing told the synthesis this step never completed")


@pytest.mark.asyncio
async def test_a_step_that_really_ran_is_still_offered_as_a_result(runner):
    """The control. A fix that labels every step "unfinished" would pass the
    test above and make her unable to report anything she did do."""
    seen = {}

    async def _capture(prompt, **kw):
        seen["prompt"] = prompt
        return "summary"

    async def _ok(step, plan, **kw):
        step.status = "success"
        step.output = "the deed is done"

    plan = _plan(_step(1), _step(2))
    plan.steps[0].status = "success"
    plan.steps[0].output = "already finished"
    runner._suspend_plan(plan, 1, _capture, None, None)

    with _executor_returning(_ok):
        await runner.resume("")

    assert "produced" in seen["prompt"], (
        "a step that genuinely ran was reported as unfinished")
    assert "did NOT finish" not in seen["prompt"]


# ─── what stays different, on purpose ────────────────────────────────────────

def test_3d_replanning_stays_out_of_the_loop():
    """The one divergence that is not a defect, asserted so it is not "fixed"
    later by someone tidying up.

    3D re-planning writes a continuation from the original goal when a plan
    hits the step limit. A resumed plan is already a continuation, and it has
    no `_depth` to stop the recursion -- so it belongs to `run()`, after the
    loop, and not inside it.
    """
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    by_name = {n.name: n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef)}

    assert not _calls_named(by_name["run_steps"], "run"), (
        "run_steps re-plans; a resumed plan would recurse without a depth "
        "guard")
    assert _calls_named(by_name["run"], "run"), (
        "3D re-planning is gone from run()")
