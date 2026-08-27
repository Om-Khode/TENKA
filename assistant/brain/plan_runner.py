"""Plans are run here. The planner writes them and does not execute them.

TENKA-v2 §17.P8. `actions/planner/` kept both halves: it decided what the steps
were *and* drove them, and `da_handlers.handle_planner` wrapped the whole thing
in an abort task and a status phase. So "the planner" meant three different
jobs, and the only way to change how a plan is *run* was to edit the module
that decides what a plan *is*.

The split is by question:

    what should happen        actions/planner/  -- generation, validation,
                              tool manifest, reference resolution, synthesis
    make it happen            here             -- the step loop, recovery,
                              suspension, abort, the bypass

**Why this could not stay below.** `brain` may reach everything under it and
nothing beside or above it, so `actions` cannot call in here -- that is the
sixth time a phase of this plan has specified a component in `brain/` that a
lower layer must reach, and the fifth different answer would have been the
wrong one. The answer here is not "move it to `core/`" like the others: running
a plan *is* coordination, which is what `brain` is for. What changes is who
calls whom. `main.py` routes the intent here; this module asks `actions` for a
plan and then dispatches each step back through `actions.execute()`.

**The capability gate did not move, and that is deliberate.** Getting a plan is
`actions.execute("planner", ...)`, so the `EXECUTE` check still happens at the
one site in the tree that resolves a handler, and every step re-enters through
the same door. `main.py` gates the branch as well -- belt and braces, and the
pre-dispatch sweep requires it -- but if that gate were deleted tomorrow the
inner one would still refuse.

**`handle_planner` stays registered**, doing generation only. Removing it would
have been tidier and wrong: `brain/affordance.py:seed_from_handlers` mirrors
`tool_registry`, so dropping the entry would delete `planner` from what she
says she can do -- while she was still doing it. An assistant that denies a
capability it has is the same defect as one that claims a capability it lacks,
and §13's K1 forbids both.

**`io` is injected, never imported.** The `brain -> io` contract is forbidden,
and the loop wanted two things from it: the overlay's step chips and a voice
for "let me try a different approach". Both arrive as callables. That is the
pattern `brain/turn.py` and `brain/executor.py` already use, and it leaves the
loop able to run with no UI attached at all -- which is what made the tests
below possible to write without standing up a bridge.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("planner")

# What the loop wants from the outside world, neither of which `brain` may
# import. `None` for either is a working configuration, not a degraded one.
Speak = Callable[[str], Awaitable[None]]
Progress = Callable[..., None]


# ─── Suspension state ────────────────────────────────────────────────────────
#
# Process-wide, because a suspended plan outlives the turn that started it: the
# user is asked something, the turn ends, and the answer arrives on a later
# turn through `main.py`'s pending epilogue. It moved here with the loop -- the
# state and the only code that reads it belong together, and leaving it in
# `actions/` would have meant `brain` reaching into another package's globals.

_suspended_plan = None
_suspended_step_index: int = 0
_suspended_llm_func = None
_suspended_speak = None
_suspended_bridge = None
# Carried across the suspension too. Losing it here would silently
# reinstate one of the seven divergences the single loop just closed: a
# resumed plan that runs correctly and reports no progress at all.
_suspended_progress = None


def has_suspended_plan() -> bool:
    """Is a plan waiting to resume after user interaction?"""
    return _suspended_plan is not None


def clear_suspended_plan() -> None:
    """Drop any suspended plan — e.g. the user changed the subject."""
    global _suspended_plan, _suspended_step_index
    global _suspended_llm_func, _suspended_speak, _suspended_bridge
    global _suspended_progress
    if _suspended_plan:
        logger.info("[PLANNER] Clearing suspended plan")
    _suspended_plan = None
    _suspended_step_index = 0
    _suspended_llm_func = None
    _suspended_speak = None
    _suspended_bridge = None
    _suspended_progress = None


def _suspend_plan(plan, resume_from_index, llm_func, speak, bridge,
                  progress=None) -> None:
    """Save plan state for later resumption."""
    global _suspended_plan, _suspended_step_index
    global _suspended_llm_func, _suspended_speak, _suspended_bridge
    global _suspended_progress
    _suspended_plan = plan
    _suspended_step_index = resume_from_index
    _suspended_llm_func = llm_func
    _suspended_speak = speak
    _suspended_bridge = bridge
    _suspended_progress = progress
    logger.info(
        f"[PLANNER] Plan SUSPENDED at step {resume_from_index + 1}/"
        f"{len(plan.steps)} — waiting for user interaction"
    )


# ─── The step loop ───────────────────────────────────────────────────────────
#
# One loop, called from two places. It used to be two loops -- one walked the
# steps with a `while` and an index, the other walked them again with a `for`
# over a range -- and they had drifted apart in seven ways, every one of them a
# defect in the resumed path. The commit that merged them lists all seven; the
# short version is that a resumed plan could not be cancelled, reported no
# progress, and raised `ValueError` if a recovery step asked a question.
#
# The `while` form is the one that survives, because inserting recovery steps
# into the plan and letting the ordinary loop pick them up is what makes
# recovery, dependencies and suspension compose. A nested executor has to
# re-implement all three, and did not.
#
# 3D re-planning stays in `run()` and is deliberately not here: it needs the
# original goal to write a continuation, and a resumed plan is a continuation
# already.


@dataclass
class StepLoopResult:
    """Why the loop stopped.

    `suspended` is not "did it fail" -- a suspended plan is healthy and waiting
    on a person. The caller must return `output` to the user untouched and must
    not synthesize, because the plan is not finished.
    """

    suspended: bool
    output: Optional[str] = None


def _blocked_by_dependency(step, plan) -> bool:
    """Mark `step` skipped if anything it depends on failed or was skipped."""
    if not step.depends_on:
        return False
    for dep_id in step.depends_on:
        dep_step = next(
            (s for s in plan.steps if s.step_id == dep_id), None
        )
        if dep_step and dep_step.status in ("failed", "skipped"):
            step.status = "skipped"
            step.error = (
                f"dependency step {dep_id} "
                f"{dep_step.status}: {dep_step.error[:80]}"
            )
            logger.info(
                f"[PLANNER] Step {step.step_id} SKIPPED: {step.error}"
            )
            return True
    return False


def _skip_dependents(step, plan) -> None:
    """Cascade a failure to every pending step waiting on it."""
    for later in plan.steps:
        if later.status == "pending" and step.step_id in later.depends_on:
            later.status = "skipped"
            later.error = (
                f"dependency step {step.step_id} failed: {step.error[:80]}"
            )
            logger.info(
                f"[PLANNER] Step {later.step_id} SKIPPED: dependency failed"
            )


def _mark_origin_recovered(step, plan) -> None:
    """A step whose recovery steps all succeeded is `recovered`, not `failed`.

    Without this the synthesis reports a failure that was repaired, which is
    the same lie as claiming a success, pointed the other way.
    """
    if step.status != "success":
        return
    if not hasattr(plan, "_recovery_step_ids"):
        return
    if step.step_id != plan._recovery_step_ids[-1]:
        return
    origin = next(
        (s for s in plan.steps if s.step_id == plan._recovery_origin), None
    )
    if origin and origin.status == "failed":
        origin.status = "recovered"
        logger.info(
            f"[PLANNER] Step {plan._recovery_origin} marked 'recovered' "
            f"— all recovery steps succeeded"
        )


async def _insert_recovery(step, plan, index: int, llm_func,
                           speak: Optional[Speak]) -> bool:
    """Try once per plan to plan around a failed step.

    Returns True when recovery steps were inserted, which means the caller
    should advance past the failure and let the ordinary loop run them -- they
    are ordinary steps now, and get dependency handling, abort checks and
    suspension for free.
    """
    from ..actions.planner.planner import _attempt_recovery

    if not hasattr(plan, "_recovery_attempted"):
        plan._recovery_attempted = False

    if plan._recovery_attempted:
        logger.info(
            f"[PLANNER] Recovery already attempted this plan — skipping for "
            f"step {step.step_id}"
        )
        return False

    plan._recovery_attempted = True
    logger.info(f"[PLANNER] Attempting recovery for step {step.step_id}")

    if speak:
        from ..automation import verification as _ver
        parsed = _ver.parse_verify_failed(step.error or "")
        if parsed:
            await speak(f"{_ver.format_failure_for_user(parsed)} Trying again.")
        else:
            await speak(
                "Hmm, that didn't work. Let me try a different approach."
            )

    recovery_steps = await _attempt_recovery(step, plan, llm_func)
    if not recovery_steps:
        return False

    for i, rs in enumerate(recovery_steps):
        plan.steps.insert(index + 1 + i, rs)
    plan._recovery_origin = step.step_id
    plan._recovery_step_ids = [rs.step_id for rs in recovery_steps]
    logger.info(
        f"[PLANNER] Inserted {len(recovery_steps)} recovery steps after "
        f"step {step.step_id}"
    )
    return True


async def run_steps(
    plan,
    start_index: int,
    *,
    llm_func,
    speak: Optional[Speak] = None,
    progress: Optional[Progress] = None,
    bridge=None,
) -> StepLoopResult:
    """Walk `plan.steps` from `start_index`, executing each.

    The single place a plan step is executed. `run()` starts it at 0 and
    `resume()` starts it wherever the interaction left off; nothing else
    differs, which is the point.
    """
    from ..actions.planner.executor import execute_step
    from ..core.abort import abort, UserAborted

    index = start_index
    while index < len(plan.steps):
        if abort.is_aborted():
            raise UserAborted(abort.reason)

        step = plan.steps[index]

        if progress:
            # Detail uses the step intent (e.g. "browser_action") replacing
            # underscores with spaces. Empty if missing — the step chip
            # carries N/M.
            _intent = getattr(step, "intent", "") or ""
            progress(str(_intent).replace("_", " ")[:32],
                     index + 1, len(plan.steps))

        if _blocked_by_dependency(step, plan):
            index += 1
            continue

        await execute_step(
            step, plan, llm_func=llm_func, bridge=bridge, tts_func=speak,
        )

        if step.status == "waiting":
            resume_index = plan.steps.index(step) + 1
            if resume_index < len(plan.steps):
                _suspend_plan(plan, resume_index, llm_func, speak, bridge,
                              progress)
                return StepLoopResult(suspended=True, output=step.output)
            step.status = "success"
            logger.info(
                f"[PLANNER] Step {step.step_id} was last step, "
                f"no suspension needed"
            )

        _mark_origin_recovered(step, plan)

        if step.status == "failed":
            if await _insert_recovery(step, plan, index, llm_func, speak):
                index += 1
                continue
            _skip_dependents(step, plan)

        index += 1

    return StepLoopResult(suspended=False)


# ─── Entry points ────────────────────────────────────────────────────────────


async def run(
    goal: str,
    *,
    llm_func,
    speak: Optional[Speak] = None,
    progress: Optional[Progress] = None,
    bridge=None,
    params: Optional[dict] = None,
    llm_response: str = "",
    _depth: int = 0,
    _prior_context: str = "",
) -> Optional[str]:
    """Ask `actions` for a plan, then run it.

    Returns the spoken answer, or None when the goal turned out not to need a
    plan at all and the caller should fall back to ordinary routing.
    """
    from .. import actions as _actions
    from ..actions.planner.planner import _synthesize_result

    logger.info(f'[PLANNER] Goal: "{goal}"')

    # Through `actions.execute`, not by importing `_generate_plan` directly.
    # That is not ceremony: `actions/__init__.py` is the only site in the tree
    # that resolves a handler, and it is where the `EXECUTE` capability check
    # lives. Reaching past it for the plan would mean a caller with no
    # authority could still make TENKA think about how to do the thing --
    # spending model calls, and handing this loop a plan it should never have
    # been given. A refused call returns the refusal sentence instead of a
    # plan, which is why the type is checked below rather than assumed.
    plan = await _actions.execute(
        "planner",
        {**(params or {}), "goal": goal, "prior_context": _prior_context},
        llm_response, bridge, _from_planner=True,
    )

    if isinstance(plan, str):
        # A refusal. It is already a sentence; say it and stop.
        return plan

    if not plan or not plan.steps:
        logger.warning("[PLANNER] Plan generation failed — falling back")
        return None

    if len(plan.steps) == 1:
        # Not a plan. One step through the planner costs a generation call, a
        # synthesis call and a loop, to do what the ordinary dispatch does in
        # one -- so it goes straight to the tool.
        tool = plan.steps[0].tool
        step_goal = plan.steps[0].goal
        logger.info(
            f"[PLANNER] Single-step plan → bypassing planner, direct to {tool}"
        )
        return await _bypass(tool, step_goal, goal,
                             params or {}, llm_response, bridge)

    logger.info(f"[PLANNER] Plan: {len(plan.steps)} steps")
    for s in plan.steps:
        cond = f" [if: {s.condition}]" if s.condition else ""
        deps = f" [needs: step {s.depends_on}]" if s.depends_on else ""
        logger.info(
            f"  Step {s.step_id}: [{s.tool}] {s.goal[:80]}{deps}{cond}"
        )

    plan.status = "executing"
    loop = await run_steps(plan, 0, llm_func=llm_func, speak=speak,
                           progress=progress, bridge=bridge)
    if loop.suspended:
        return loop.output

    plan.status = "completed"
    last_step = plan.steps[-1] if plan.steps else None
    if (
        _depth == 0
        and last_step is not None
        and last_step.tool == "synthesize"
        and last_step.status == "success"
        and len(plan.steps) >= 3
    ):
        # 6a.5 review H3. This used to be
        #   f"{goal}\n\nProgress so far: {last_step.output}"
        # which made a synthesize step's output -- laundered file content --
        # part of the continuation plan's `original_goal`, i.e. the one field
        # the fence treats as the user's own words. From there it rode
        # `_planner_goal` into `browser_action`, a tool declared to accept no
        # prior-step output at all. The progress now travels beside the goal as
        # fenced data and never joins it.
        logger.info("[PLANNER] 3D: Step limit hit — re-planning continuation")
        continuation = await run(
            f"{goal}\n\nContinue completing the remaining work.",
            llm_func=llm_func, speak=speak, progress=progress, bridge=bridge,
            params=params, llm_response=llm_response,
            _depth=1, _prior_context=last_step.output,
        )
        if continuation:
            return continuation

    return await _synthesize_result(plan, llm_func)


async def _bypass(tool: str, step_goal: str, goal: str, params: dict,
                  llm_response: str, bridge) -> str:
    """Dispatch a one-step "plan" straight to its tool and phrase the result."""
    from .. import actions as _actions
    from ..automation import verification as _ver

    result = await _actions.execute(
        tool, {**params, "goal": step_goal}, llm_response, bridge,
        _from_planner=True,
    )

    parsed = _ver.parse_verify_failed(result or "")
    if parsed:
        return _ver.format_failure_for_user(parsed)

    from ..llm.contracts import ask_for_synthesis
    try:
        synth = await ask_for_synthesis(
            f'User asked: "{goal}"\n'
            f'Result:\n{(result or "")[:1500]}\n\n'
            f'Concise spoken response (1-2 sentences). '
            f'Summarize what happened and the key information.',
            max_tokens=200,
        )
        if synth and synth != "__LLM_UNAVAILABLE__":
            return synth
    except Exception:
        pass
    return result


async def resume(interaction_result: str = "") -> Optional[str]:
    """Continue a suspended plan once the user has answered.

    Called from `main.py` after a pending handler resolves.
    """
    from ..actions.planner.planner import _step_failed, _synthesize_result
    from ..core.abort import UserAborted

    from ..core.abort import abort

    global _suspended_plan

    if _suspended_plan is None:
        return None

    # Before the bookkeeping and before the voice, not after.
    #
    # `run_steps` checks abort at the top of every iteration, which is the
    # right place for a step -- but everything between here and the first
    # iteration happens regardless: the suspended step is resolved, the plan is
    # cleared, and "Alright, continuing. 1 step left." is spoken. Live test:
    # ESC at 23:19:52, "Alright, continuing" at 23:19:58, "Stopped" at
    # 23:20:01. She announced she was carrying on six seconds after being told
    # to stop, and then stopped -- which reads as ignoring the user twice
    # rather than obeying them once.
    if abort.is_aborted():
        logger.info("[PLANNER] Resume refused — aborted before it began")
        # Marked before clearing, not after: a plan that stopped is `failed`,
        # and leaving it `pending` would describe an abandoned run as one that
        # never started.
        _suspended_plan.status = "failed"
        clear_suspended_plan()
        try:
            from ..actions.responses import personality_say
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."

    plan = _suspended_plan
    resume_from = _suspended_step_index
    llm_func = _suspended_llm_func
    speak = _suspended_speak
    bridge = _suspended_bridge
    progress = _suspended_progress

    clear_suspended_plan()

    if resume_from > 0:
        waiting_step = plan.steps[resume_from - 1]
        if waiting_step.status == "waiting":
            if interaction_result and not _step_failed(interaction_result):
                waiting_step.status = "success"
                waiting_step.answered = True
                waiting_step.output = interaction_result
                plan.context[f"step_{waiting_step.step_id}"] = interaction_result
                logger.info(
                    f"[PLANNER] Suspended step {waiting_step.step_id} "
                    f"resolved: SUCCESS"
                )
            else:
                waiting_step.status = "failed"
                waiting_step.error = (
                    interaction_result[:300] if interaction_result
                    else "cancelled"
                )
                waiting_step.output = interaction_result or ""
                logger.info(
                    f"[PLANNER] Suspended step {waiting_step.step_id} "
                    f"resolved: FAILED"
                )

    logger.info(
        f"[PLANNER] Resuming plan from step {resume_from + 1}/"
        f"{len(plan.steps)}"
    )

    if speak:
        remaining = len(plan.steps) - resume_from
        await speak(
            f"Alright, continuing. "
            f"{remaining} step{'s' if remaining != 1 else ''} left."
        )

    try:
        plan.status = "executing"
        loop = await run_steps(plan, resume_from, llm_func=llm_func,
                               speak=speak, progress=progress, bridge=bridge)
        if loop.suspended:
            return loop.output

        plan.status = "completed"
        return await _synthesize_result(plan, llm_func)
    except UserAborted:
        # `run()` raises this too, and survives because `main.py` gates the
        # branch it is called from. `resume()` is called straight from the
        # pending epilogue, which catches nothing -- so the ESC-hold check the
        # loop performs would end the turn on an unhandled exception rather
        # than stopping it. The boundary was easy; the paths around it were
        # not.
        plan.status = "failed"
        logger.info("[PLANNER] Resumed plan aborted by user")
        try:
            from ..actions.responses import personality_say
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
