"""manifest_dispatch.py — handler for the synthetic manifest_dispatch intent.

The regex_router fires this with params {app_id, intent_id, slots}. The
handler resolves through ManifestDispatcher via manifest_runtime.get_dispatcher().
On selector exhaustion (or any other escalation), it falls through to the computer_task layer's
handle_computer_task so the user still gets their action — the manifest path
only adds capability, never blocks it.

The handler returns an empty string on dispatch success; the orchestrator
decides phrasing/TTS. On escalation, the computer_task handler's return value bubbles
up unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .registry import tool_registry

logger = logging.getLogger("manifest")


@tool_registry.decorator("manifest_dispatch")
async def handle_manifest_dispatch(
    params: dict, llm_response: str, bridge: Any = None,
    _from_planner: bool = False,
) -> str:
    from ..core.abort import abort, UserAborted
    from ..io.status_broadcaster import status, StatusPhase

    detail = f"{params.get('app_id', '')}/{params.get('intent_id', '')}"[:40]
    status.set(StatusPhase.THINKING, detail=detail)

    try:
        if abort.is_aborted():
            raise UserAborted(abort.reason)

        # Local imports per feedback_actions_local_imports — keeps the module
        # import-time graph free of action ↔ automation cycles.
        from ..automation.manifest_registry import get_singleton as _get_reg
        from ..automation.router import detect_active_app
        from ..automation import manifest_runtime

        reg = _get_reg()
        if reg is None:
            logger.warning("[manifest] registry not initialized; escalating to computer_task")
            return await _escalate_to_computer_task(params, llm_response, bridge)

        app_id = params.get("app_id", "")
        intent_id = params.get("intent_id", "")
        slots = params.get("slots", {}) or {}

        disp = manifest_runtime.get_dispatcher()
        if disp is None:
            logger.warning("[manifest] dispatcher not initialized; escalating to computer_task")
            return await _escalate_to_computer_task(params, llm_response, bridge)

        active = await asyncio.to_thread(detect_active_app)
        window_title = active.get("window_title", "")

        try:
            result = await asyncio.to_thread(
                disp.dispatch,
                app_id=app_id, intent_id=intent_id,
                slots=slots, active_window=window_title,
            )
        except UserAborted:
            raise
        except Exception as e:
            logger.warning(
                f"[manifest] dispatcher raised {e!r} for {app_id}:{intent_id}; "
                f"escalating to computer_task"
            )
            return await _escalate_to_computer_task(params, llm_response, bridge)

        if result.ok:
            logger.info(
                f"[manifest] dispatch ok {app_id}:{intent_id} "
                f"(selector idx={result.selector_used_index})"
            )
            return ""  # caller decides phrasing or stays silent
        logger.info(
            f"[manifest] dispatch failed {app_id}:{intent_id} "
            f"({result.error}); escalating to computer_task"
        )
        return await _escalate_to_computer_task(params, llm_response, bridge)
    except UserAborted:
        # When dispatched from the planner, re-raise so the planner sees the
        # abort instead of a "Stopped." string treated as a successful step.
        if _from_planner:
            raise
        try:
            from .responses import personality_say
            return personality_say("stopped", default="Stopped.")
        except Exception:
            return "Stopped."
    finally:
        status.set(StatusPhase.IDLE)


async def _escalate_to_computer_task(params: dict, llm_response: str, bridge: Any) -> str:
    """Fallback to computer_task planner via handle_computer_task.

    Local import of da_handlers to avoid circular import risk per
    feedback_actions_local_imports.
    """
    from .da_handlers import handle_computer_task

    app_id = params.get("app_id", "")
    intent_id = params.get("intent_id", "")
    # Strip manifest suffix (e.g. "test_app.desktop" → "test app") so the
    # computer_task planner prompt reads as natural language.
    app_label = app_id.split(".")[0].replace("_", " ").strip() if app_id else ""
    intent_label = intent_id.replace("_", " ") if intent_id else ""
    goal = f"{intent_label} in {app_label}" if intent_label and app_label else intent_label or app_label
    return await handle_computer_task(
        params={"goal": goal}, llm_response=llm_response, bridge=bridge,
    )
