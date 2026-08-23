"""
executor.py — Single-step execution for the planner.

Dispatches each PlanStep to either a pseudo-tool (handled locally) or
a real tool handler via actions.execute(). Manages pending-state
snapshotting for suspension detection.
"""

import logging
import re

logger = logging.getLogger("planner")

_STEP_REF_RE = re.compile(r'\$step_(\d+)')

#: `_planner_goal` becomes the Tavily recon query in `automation/router.py`,
#: which is network egress. A user goal is a sentence; anything longer is
#: something else having got in.
_MAX_PLANNER_GOAL_CHARS = 400


# --- Pending state snapshotting ---

def _snapshot_pending_states() -> dict:
    """Return {name: is_active} for every registered PendingState."""
    from assistant.pending import pending_registry
    return pending_registry.snapshot()


def _pending_state_changed(before: dict, after: dict) -> bool:
    """Check if any pending state went from inactive to active."""
    for var_name, was_active in before.items():
        if not was_active and after.get(var_name, False):
            logger.info(
                f"[PLANNER] Pending state activated: {var_name}"
            )
            return True
    return False


def _clear_auth_pending_states(pending_before: dict) -> "list[str]":
    """Tear down pending states this step armed while asking for credentials.

    A step that returns an auth sentinel has parked an OAuth or device-auth
    prompt that the planner is not going to answer, so it is cleared rather
    than left armed for the next unrelated utterance to collide with. Only
    states that were inactive *before* this step are touched -- anything
    already armed belonged to someone else's conversation.

    **KI-27.** This was a bare `state.clear()`, and the variable name is why it
    was a known hole: both pending arm/clear AST sweeps match on the
    *receiver's name*, so a clear reached through a loop-local called `state`
    was invisible to them while every other clear in the tree was covered.
    Reasoned safe at the time -- same-request teardown of a state this same
    call armed -- but the sweeps' whole promise is that a new unguarded site
    cannot ship silently, and this was the one shape where that promise did not
    hold.

    Two changes close it. `try_clear` is the mechanism built for exactly this
    position: a clear in an ordinary flow rather than inside a
    `handle_pending_*` that the dispatch loop already gates by ownership. For
    the owner it behaves identically to the bare clear; for anyone else it
    refuses and parks the attempt via `note_foreign_attempt`, so the owner is
    told rather than finding her open question silently gone. And the sweep now
    follows a name back to `pending_registry.get(...)`, so the next clear
    written this way is visible whatever it is called.

    Extracted from `execute_step` so the refusal branch is reachable by a test
    without standing up a whole plan run. Returns the names actually cleared.
    """
    from assistant.pending import pending_registry, try_clear

    cleared: "list[str]" = []
    for name, was_active in pending_before.items():
        if was_active:
            continue
        state = pending_registry.get(name)
        if not (state and state.active):
            continue
        if try_clear(state):
            cleared.append(name)
            logger.info(f"[PLANNER] Cleared auth pending state: {name}")
        else:
            logger.warning(
                f"[PLANNER] Auth pending state {name} is owned by another "
                f"principal — left armed"
            )
    return cleared


# --- Step execution ---

async def execute_step(
    step,
    plan,
    llm_func,
    bridge=None,
    tts_func=None,
) -> None:
    """
    Execute a single plan step by dispatching to the appropriate tool handler.

    Calls EXISTING handlers via actions.execute().
    Pseudo-tools are handled via pseudo_tools module.
    """
    from .planner import (
        _evaluate_condition, _split_references, _step_failed,
        _brief, _extract_note_params, TOOL_MANIFEST, EGRESS_REFUSED,
    )
    from .pseudo_tools import (
        run_synthesize_step, run_vision_analyze_step,
        run_camera_preview_step, run_prompt_user_step,
    )

    # ── Check condition ────────────────────────────────────────────
    if not _evaluate_condition(step.condition, plan):
        step.status = "skipped"
        step.error = "condition not met"
        logger.info(f"[PLANNER] Step {step.step_id} SKIPPED: condition not met")
        return

    # ── Resolve $step_N references in the goal ─────────────────────
    # The fence (milestone 6a.5, spec §5.3). Prior-step output is untrusted --
    # a planted file, OCR of the screen, a fetched page -- so it is split out
    # of the instruction and carried separately in `step_context`. Everything
    # downstream of here (TTS, logging, llm_response, the cache key) uses the
    # instruction, which is the user's own words.
    resolved_goal, step_context = _split_references(step.goal, plan, step.tool)

    # An egress param (a URL to navigate to, a query shipped to a search
    # provider) would not reduce to one -- 6a.5 review H1. Fail the step here
    # rather than hand a planted document to the network. The reason is the
    # step's error, so recovery and the final synthesis both see it.
    if resolved_goal.startswith(EGRESS_REFUSED):
        reason = resolved_goal[len(EGRESS_REFUSED):].strip()
        step.status = "failed"
        step.error = f"blocked: {reason}"
        step.output = step.error
        logger.warning(
            f"[PLANNER] Step {step.step_id} BLOCKED [{step.tool}]: {reason}"
        )
        return

    step.status = "running"

    logger.info(f"[PLANNER] Step {step.step_id} RUNNING: [{step.tool}] "
                f"{resolved_goal[:120]}")

    # ── Announce step via TTS ──────────────────────────────────────
    if tts_func and len(plan.steps) > 1:
        _silent_tools = ("vision_analyze", "synthesize", "camera_preview",
                         "code_executor", "browser_action", "app_action")
        if step.tool in _silent_tools:
            pass
        elif step.tool == "prompt_user":
            pass
        elif step.step_id == 1:
            await tts_func(
                f"Let me work on this. First — {_brief(resolved_goal)}."
            )
        else:
            await tts_func(f"Next — {_brief(resolved_goal)}.")

    try:
        # ── Handle pseudo-tools internally ─────────────────────────
        if step.tool == "synthesize":
            result = await run_synthesize_step(resolved_goal, llm_func,
                                               context=step_context)
        elif step.tool == "vision_analyze":
            result = await run_vision_analyze_step(resolved_goal, tts_func,
                                                   context=step_context)
        elif step.tool == "camera_preview":
            result = await run_camera_preview_step(resolved_goal, tts_func)
        elif step.tool == "prompt_user":
            result = await run_prompt_user_step(resolved_goal, tts_func)
        else:
            # ── Dispatch to existing tool via actions.execute() ────
            import assistant.actions as _actions_mod

            _entry = TOOL_MANIFEST.get(step.tool, {})
            param_key = _entry.get("param_key", "goal")
            params = {param_key: resolved_goal}

            # Prior-step output travels in its own param, never merged back
            # into the instruction. Omitted entirely when there is none, so a
            # single-step goal does not carry an empty labelled block.
            context_key = _entry.get("context_key")
            if step_context and context_key:
                params[context_key] = step_context

            if step.tool in ("browser_action", "app_action"):
                # 6a.5 review H3. This is a SECOND string reaching two tools
                # whose manifest rows say they accept no prior-step output,
                # and `_split_references` never sees it. Two things make it
                # safe rather than one: the 3D continuation no longer splices
                # a step's output into `original_goal` (planner.py), and the
                # value is scrubbed here so a future path that reintroduces a
                # reference cannot deliver it. `$step_N` is dropped rather
                # than resolved -- this field is the user's words or nothing.
                _pg = _STEP_REF_RE.sub("", plan.original_goal or "")
                params["_planner_goal"] = _pg[:_MAX_PLANNER_GOAL_CHARS]

            if step.tool == "create_note":
                params = _extract_note_params(resolved_goal)

            # Snapshot pending states BEFORE the step runs
            pending_before = _snapshot_pending_states()

            result = await _actions_mod.execute(
                intent=step.tool,
                params=params,
                llm_response=resolved_goal,
                bridge=bridge,
                _from_planner=True,
            )

            # ── Auth sentinel check BEFORE pending state check ────────
            # Only match machine-readable sentinels and the specific setup
            # prompt prefix. Generic phrases like "developer app" cause
            # false positives on web search result content.
            _AUTH_SENTINELS = (
                "__NEEDS_OAUTH__", "NEEDS_OAUTH|",
                "__NEEDS_DEVICE_AUTH__", "NEEDS_DEVICE_AUTH|",
                "I need to set up",
            )
            if result and any(s in result for s in _AUTH_SENTINELS):
                _clear_auth_pending_states(pending_before)
                step.status = "failed"
                step.error = (
                    f"Authentication required — set up the service "
                    f"first, then retry: {result[:150]}"
                )
                step.output = result
                logger.info(
                    f"[PLANNER] Step {step.step_id} FAILED: auth "
                    f"required (not suspending)"
                )
                return

            # Check if step triggered an interactive pending state
            pending_after = _snapshot_pending_states()
            if _pending_state_changed(pending_before, pending_after):
                step.status = "waiting"
                step.output = result or "(awaiting user input)"
                logger.info(
                    f"[PLANNER] Step {step.step_id} WAITING: "
                    f"interactive pending state detected"
                )
                return

        # ── Verify output ─────────────────────────────────────────
        if result is None:
            result = "(no output)"

        if _step_failed(result):
            step.status = "failed"
            step.error = result[:300]
            step.output = result
            logger.warning(
                f"[PLANNER] Step {step.step_id} FAILED (verified): "
                f"{step.error[:100]}"
            )
            # Self-heal: invalidate cached automation steps that led
            # to a semantic failure (e.g. "No results found")
            if step.tool == "browser_action":
                try:
                    from assistant.automation.step_cache import delete_cached_steps
                    delete_cached_steps("browser", "browser", resolved_goal)
                    logger.info("[PLANNER] Invalidated browser cache for failed step")
                except Exception:
                    pass
            return

        # ── Success ───────────────────────────────────────────────
        step.status = "success"
        step.output = result
        clean_result = re.sub(
            r'^\[(?:neutral|happy|excited|sad|angry|sarcastic|worried|surprised)\]\s*',
            '', result
        )
        plan.context[f"step_{step.step_id}"] = clean_result
        logger.info(
            f"[PLANNER] Step {step.step_id} SUCCESS: {result[:120]}"
        )

    except Exception as e:
        # user-initiated abort must propagate up — do NOT mark the step
        # as failed (which would trigger the planner's recovery path and
        # require a second ESC to actually stop).
        from assistant.core.abort import UserAborted
        if isinstance(e, UserAborted):
            raise
        step.status = "failed"
        step.error = str(e)
        step.output = f"Error: {e}"
        logger.error(f"[PLANNER] Step {step.step_id} EXCEPTION: {e}")
