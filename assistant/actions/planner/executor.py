"""
executor.py — Single-step execution for the planner.

Dispatches each PlanStep to either a pseudo-tool (handled locally) or
a real tool handler via actions.execute(). Manages pending-state
snapshotting for suspension detection.
"""

import logging
import re

logger = logging.getLogger("planner")


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
        _evaluate_condition, _resolve_references, _step_failed,
        _brief, _extract_note_params, TOOL_MANIFEST,
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
    resolved_goal = _resolve_references(step.goal, plan)
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
            result = await run_synthesize_step(resolved_goal, llm_func)
        elif step.tool == "vision_analyze":
            result = await run_vision_analyze_step(resolved_goal, tts_func)
        elif step.tool == "camera_preview":
            result = await run_camera_preview_step(resolved_goal, tts_func)
        elif step.tool == "prompt_user":
            result = await run_prompt_user_step(resolved_goal, tts_func)
        else:
            # ── Dispatch to existing tool via actions.execute() ────
            import assistant.actions as _actions_mod

            param_key = TOOL_MANIFEST.get(step.tool, {}).get("param_key", "goal")
            params = {param_key: resolved_goal}

            if step.tool in ("browser_action", "app_action"):
                params["_planner_goal"] = plan.original_goal

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
                from assistant.pending import pending_registry
                for name, was_active in pending_before.items():
                    if not was_active:
                        state = pending_registry.get(name)
                        if state and state.active:
                            state.clear()
                            logger.info(
                                f"[PLANNER] Cleared auth pending state: "
                                f"{name}"
                            )
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
