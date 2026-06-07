"""
recovery.py — Adaptive Recovery for agentic steps.

Generic, goal-driven recovery loop invoked by step executors (step loops,
procedure_executor) when verification reports a divergence between plan
and reality. Re-perceives the screen, classifies HOW reality diverged
into one of three closed classes, dispatches a single generic recovery
strategy, and re-verifies. Bounded by max_attempts and a same-observation
loop guard.

Classification set (load-bearing — adding a new class is a design discussion,
never a quiet PR):

  overlay_appeared → goal-driven bbox click on the affordance matching <goal>
  error_shown      → planner replans the input value with observation context
  no_change        → settle 0.8s + retry once; same-detail twice → escalate
  success          → no action needed; step achieved its goal (false alarm)
  unknown          → escalate to user immediately

Failure-open: any infra failure (screenshot None, vision crash, JSON parse)
is classified as `unknown` and escalates. Never silently fakes success.

Orchestrator + diagnose prompt + result types. The error_shown and no_change
strategy functions are stubs returning (False, 0). With stubs, the loop
guard or attempt limit will escalate cleanly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("recovery")


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class RecoveryAttempt:
    diagnose_class: str          # "overlay_appeared" | "error_shown" | "no_change" | "success" | "unknown"
    detail: str                  # human-readable observation from diagnose
    action_taken: str            # "bbox_click" | "replan_input" | "retry" | "escalated"
    succeeded: bool              # True if verify passed after recovery
    cost_calls: int              # vision/text calls used in this attempt


@dataclass
class RecoveryOutcome:
    succeeded: bool
    attempts: list[RecoveryAttempt] = field(default_factory=list)
    final_observation: str = ""  # last verify observation, used for user message
    escalated: bool = False      # True if we gave up and need user input


# ─── Diagnose prompt ─────────────────────────────────────────────────────────

_DIAGNOSE_PROMPT = """You are looking at a screen after a desktop automation step ran.

The agent was trying to: {goal}
The step that ran was: {step_summary}
Verification context: {verify_observation}

Decide whether the screen is in a state that needs RECOVERY. There are
three recoverable classes plus "unknown" (no recovery needed).

If "Verification context" describes a concrete failure observed by the
upstream verifier, classify that failure into one of the three recoverable
classes. If the verification context is empty / "(none)", you are doing a
PROACTIVE checkpoint — be conservative and use "unknown" unless there is
CLEAR evidence one of the recoverable classes applies.

Respond STRICTLY in JSON:
{{
  "class": "overlay_appeared" | "error_shown" | "no_change" | "success" | "unknown",
  "detail": "<one short sentence describing what you actually see — name the specific element or message>",
  "recovery_target": "<for overlay_appeared and error_shown: a SHORT description (3-8 words, NOT a full sentence) of the specific UI element to interact with to advance the goal — examples: 'Maths option in suggestion list', 'Reading checkbox under Hobbies', 'OK button on dialog', 'date 25 in April calendar', 'email input field'. For no_change and unknown: empty string.>"
}}

The recovery_target is consumed downstream by a UI-element locator that
expects short, element-shaped text. Long sentences or multi-clause
descriptions WILL fail to ground.

Class definitions (use EXACTLY one):

- "overlay_appeared": A new interactive widget is BLOCKING the next intended
  action — a popup, picker, dropdown, modal, suggestion list, calendar, or
  file dialog has appeared and the original goal can plausibly be achieved
  by interacting with it. The widget must be in the way of progress, not
  just visible somewhere on screen. CRITICAL: do NOT classify as
  "overlay_appeared" if the overlay is the EXPECTED RESULT of the last
  action — for example, clicking a dropdown control naturally opens its
  menu, clicking a date input opens its calendar, clicking "Browse" opens
  a file dialog. If "the agent was trying to" describes opening this
  overlay, the action SUCCEEDED and you should return "unknown" (the next
  planner loop will continue from the open overlay).

- "error_shown": A visible error or validation message has appeared in or
  near the field that was just operated on, AND the same UI is still in
  place waiting for a corrected value. Generic page text or instructions do
  NOT count — only a real error or validation message tied to the last
  action.

- "no_change": The action clearly DID NOT register — focus / cursor / button
  state / typed value is exactly as it would be if the action had never
  executed. Do NOT use this class merely because the screen looks calm or
  because no overlay appeared. Do NOT use this class when an action
  succeeded and the result is what you would expect (e.g. a checkbox being
  checked after a click on it, focus advancing to the next field after
  Tab). You need POSITIVE evidence that the action had no effect.

- "success": The action completed correctly — the screen shows the expected
  outcome of the step. The page loaded, the field has the typed value, or the
  intended result is visible. There is nothing to recover. Use when you see
  clear evidence the step achieved its goal despite the verification context
  claiming otherwise (false alarm from URL redirect, element re-render, etc.).

- "unknown": None of the above clearly applies. Use this when:
  * the action appears to have completed successfully and the agent can
    continue with the next step (this is the common case in proactive
    checkpoints), OR
  * the screen is in a state the agent cannot reasonably recover from
    (CAPTCHA, login wall, fatal error), OR
  * you cannot tell what happened.
  This is the SAFE / DEFAULT answer when nothing obviously needs fixing.

Do NOT invent classes. Do NOT name specific widget types in "class".
The widget type, if any, belongs in "detail" only.
"""

_VALID_CLASSES = {"overlay_appeared", "error_shown", "no_change", "success", "unknown"}

_DIAGNOSE_SYSTEM_PROMPT = (
    "You are a precise computer-vision diagnostician. Reply only with JSON "
    "matching the requested schema."
)


def _step_summary(step: dict) -> str:
    stype = step.get("type", "?")
    action = step.get("action", "?")
    params = json.dumps(step.get("params", {}), default=str)[:200]
    return f"{stype}/{action} {params}"


async def _diagnose(goal: str, verify_observation: str, step: dict) -> dict:
    """One Flash vision call returning {class, detail, recovery_target}.

    `recovery_target` is a SHORT (3-8 word) description of the specific UI
    element to click for overlay/error recovery. Empty string for no_change
    and unknown classes. The bbox locator downstream needs short element-
    shaped text — long sentences fail to ground.

    Fail-open: any infra error (screenshot None, llm crash, parse fail,
    unknown class) returns {"class": "unknown", "detail": <why>,
    "recovery_target": ""} so the orchestrator escalates cleanly without
    silently faking success.
    """
    try:
        from ..io import screen
    except Exception as e:
        logger.warning(f"[recovery] screen module unavailable: {e}")
        return {"class": "unknown", "detail": f"screen module unavailable: {e}", "recovery_target": ""}

    image_b64 = screen.capture_screenshot_base64()
    if not image_b64:
        logger.warning("[recovery] screenshot capture returned None")
        return {"class": "unknown", "detail": "screenshot capture failed", "recovery_target": ""}

    try:
        from .. import llm
    except Exception as e:
        logger.warning(f"[recovery] llm module unavailable: {e}")
        return {"class": "unknown", "detail": f"llm module unavailable: {e}", "recovery_target": ""}

    prompt = _DIAGNOSE_PROMPT.format(
        goal=goal or "(no goal)",
        step_summary=_step_summary(step),
        verify_observation=verify_observation or "(none)",
    )

    try:
        raw = (await llm.get_vision_response(
            image_base64=image_b64,
            prompt=prompt,
            system_prompt=_DIAGNOSE_SYSTEM_PROMPT,
            json_mode=True,
        )).text
    except Exception as e:
        logger.warning(f"[recovery] diagnose vision call crashed: {e}")
        return {"class": "unknown", "detail": f"vision call crashed: {e}", "recovery_target": ""}

    if not raw or raw == "__LLM_UNAVAILABLE__":
        logger.warning("[recovery] no diagnose response")
        return {"class": "unknown", "detail": "no vision response", "recovery_target": ""}

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"[recovery] diagnose JSON parse failed: {e} — head: {raw[:200]}")
        return {"class": "unknown", "detail": f"diagnose JSON parse failed: {e}", "recovery_target": ""}

    cls = str(data.get("class", "")).strip()
    detail = str(data.get("detail", "")).strip()
    recovery_target = str(data.get("recovery_target", "")).strip()
    if cls not in _VALID_CLASSES:
        logger.warning(f"[recovery] diagnose returned invalid class '{cls}' — coercing to unknown")
        return {"class": "unknown", "detail": detail or f"invalid class '{cls}'", "recovery_target": ""}

    # Observability: log the full parsed result so future debugging can see
    # exactly what the model said vs what we did with it.
    logger.debug(
        f"[recovery] diagnose parsed: class={cls!r} detail={detail!r} "
        f"recovery_target={recovery_target!r}"
    )

    return {"class": cls, "detail": detail, "recovery_target": recovery_target}


# ─── Strategy implementations ─────────────────────────────────────────────────

# Sentinel max length: bbox locator wants short, element-shaped text.
# Anything longer than this is almost certainly a sentence/paragraph that
# won't ground reliably. We refuse to send such text to the bbox call to
# avoid wasting a vision call on a doomed lookup.
_MAX_BBOX_TARGET_CHARS = 80


def _overlay_goal_text(goal: str, step: dict, target_hint: str = "") -> str:
    """Build the text description that the bbox locator grounds against.

    Priority:
      1. target_hint  (model-derived from diagnose's recovery_target — best)
      2. goal         (planner-supplied per-step goal — fallback)
      3. step synthesis (action + most descriptive params field)
      4. raw action name

    All sources are clipped at _MAX_BBOX_TARGET_CHARS — the bbox locator
    needs short element-shaped text. This is a critical guard: passing
    a sentence-long planner-thinking paragraph causes the bbox model to
    return a guess somewhere on screen rather than failing cleanly.

    Signature keeps `(goal, step)` positional for backward compat with
    existing callers; `target_hint` is keyword-only by convention.
    """
    for candidate in (target_hint, goal):
        if candidate and candidate.strip():
            stripped = candidate.strip()
            return stripped[:_MAX_BBOX_TARGET_CHARS]
    params = step.get("params", {}) or {}
    action = step.get("action", "?")
    for key in ("value", "text", "label", "selector", "name", "url"):
        v = params.get(key)
        if v:
            return f"{action}: {str(v)[:_MAX_BBOX_TARGET_CHARS]}"
    return action


def _bbox_target_source(target_hint: str, goal: str) -> str:
    """Identify which source produced the bbox text — for log observability."""
    if target_hint and target_hint.strip():
        return "recovery_target"
    if goal and goal.strip():
        return "goal"
    return "step_synthesis"


async def _recover_overlay(
    goal: str,
    page,
    active_window,
    step: dict | None = None,
    target_hint: str = "",
) -> tuple[bool, int]:
    """Goal-driven bbox click on the affordance matching the recovery target.

    Asks vision: "where on the current screen is <target>?" Click those
    coordinates. No widget knowledge — same path handles datepickers,
    dropdowns, modals, autocomplete suggestions.

    `target_hint` (preferred) comes from the diagnose model's
    `recovery_target` field — short, element-shaped, derived from full
    screen context. Falls back to `goal` then step synthesis if missing.

    Click via pyautogui on screen coords (returned by locate_element_bbox).
    For browser overlays, page.bring_to_front() first so the click lands on
    the browser window rather than whatever else has foreground. For native
    overlays, the active window already has focus.

    Returns (succeeded, vision_calls_used). Fail-open on every infra error.
    """
    try:
        from ..io import screen
        from .. import llm
    except Exception as e:
        logger.warning(f"[recovery] modules unavailable for overlay click: {e}")
        return False, 0

    image_b64 = screen.capture_screenshot_base64()
    if not image_b64:
        logger.warning("[recovery] overlay: screenshot capture returned None")
        return False, 0

    bbox_target = _overlay_goal_text(goal, step or {}, target_hint=target_hint)
    source = _bbox_target_source(target_hint, goal)

    # Observability: print the EXACT text being sent to the bbox locator
    # so any future "wrong text reached the bbox" bug is immediately visible.
    logger.info(f"[recovery] overlay bbox lookup: source={source} target={bbox_target!r}")

    try:
        coords = llm.locate_element_bbox(bbox_target, image_b64)
    except Exception as e:
        logger.warning(f"[recovery] overlay: bbox locate crashed: {e}")
        return False, 1

    if coords is None:
        logger.info(f"[recovery] overlay: bbox could not locate {bbox_target!r}")
        return False, 1

    x, y = coords

    if page is not None:
        try:
            await page.bring_to_front()
        except Exception as e:
            logger.debug(f"[recovery] overlay: bring_to_front failed (continuing): {e}")

    try:
        import pyautogui
    except Exception as e:
        logger.warning(f"[recovery] overlay: pyautogui import failed: {e}")
        return False, 1

    try:
        pyautogui.click(x, y)
        logger.info(f"[recovery] overlay: clicked ({x},{y}) for target {bbox_target!r}")
        return True, 1
    except Exception as e:
        logger.warning(f"[recovery] overlay: click crashed at ({x},{y}): {e}")
        return False, 1


async def _recover_error(step: dict, detail: str, page, active_window) -> tuple[bool, int]:
    """Replan the field input given the validation observation.

    Stub returns (False, 0).
    """
    logger.debug("[recovery] _recover_error stub — not yet implemented")
    return False, 0


async def _recover_no_change(step: dict, page, active_window) -> tuple[bool, int]:
    """Settle 0.8s + retry the original step verbatim.

    Stub returns (False, 0).
    """
    logger.debug("[recovery] _recover_no_change stub — not yet implemented")
    return False, 0


# ─── Orchestrator ─────────────────────────────────────────────────────────────

async def attempt_recovery(
    *,
    step: dict,
    goal: str,
    verify_result,                      # VerifyResult that triggered recovery
    page=None,
    active_window: Optional[str] = None,
    max_attempts: int = 3,
) -> RecoveryOutcome:
    """Generic, goal-driven recovery loop.

    Caller invokes this on verification failure. Returns RecoveryOutcome with
    succeeded=True if recovery worked, or escalated=True if max_attempts
    reached, unknown class encountered, or the same observation was seen
    twice (loop guard).

    Caller responsibilities:
      - succeeded → continue to next step
      - escalated → halt the run and surface final_observation to the user
    """
    attempts: list[RecoveryAttempt] = []
    last_observation = getattr(verify_result, "observation", "") or ""

    for attempt_num in range(max_attempts):
        diag = await _diagnose(goal, last_observation, step)
        cls = diag["class"]
        detail = diag["detail"]

        # Success — verification was a false alarm, step actually worked.
        if cls == "success":
            attempts.append(RecoveryAttempt(cls, detail, "false_alarm", True, 1))
            logger.info(f"[recovery] success class — false alarm, step OK: {detail}")
            return RecoveryOutcome(
                succeeded=True,
                attempts=attempts,
                final_observation=detail or last_observation,
                escalated=False,
            )

        # Bail out on unknown — never improvise a strategy.
        if cls == "unknown":
            attempts.append(RecoveryAttempt(cls, detail, "escalated", False, 1))
            logger.info(f"[recovery] unknown class — escalating: {detail}")
            return RecoveryOutcome(
                succeeded=False,
                attempts=attempts,
                final_observation=detail or last_observation,
                escalated=True,
            )

        # Loop guard: same detail twice in a row → immediate escalate.
        # Compares against the previous attempt's diagnose detail (not its
        # post-recovery verify observation). If the world looks the same
        # after one recovery cycle, more cycles won't help.
        if attempts and detail and detail == attempts[-1].detail:
            attempts.append(RecoveryAttempt(cls, detail, "escalated", False, 1))
            logger.info(f"[recovery] same diagnose detail twice — escalating: {detail}")
            return RecoveryOutcome(
                succeeded=False,
                attempts=attempts,
                final_observation=detail,
                escalated=True,
            )

        # Dispatch — strategy ↔ class is 1:1.
        if cls == "overlay_appeared":
            ok, calls = await _recover_overlay(
                goal, page, active_window, step,
                target_hint=diag.get("recovery_target", ""),
            )
            action = "bbox_click"
        elif cls == "error_shown":
            ok, calls = await _recover_error(step, detail, page, active_window)
            action = "replan_input"
        elif cls == "no_change":
            ok, calls = await _recover_no_change(step, page, active_window)
            action = "retry"
        else:
            # _VALID_CLASSES is closed and unknown is handled above; this
            # branch is unreachable. Treat as escalate to be safe.
            attempts.append(RecoveryAttempt(cls, detail, "escalated", False, 1))
            return RecoveryOutcome(
                succeeded=False,
                attempts=attempts,
                final_observation=detail,
                escalated=True,
            )

        # Re-verify after the recovery action. Diagnose was 1 vision call;
        # the strategy may have used more; verify is 1 more.
        try:
            from . import verification
            vr = await verification.post_verify(step, page=page, active_window=active_window)
        except Exception as e:
            logger.warning(f"[recovery] post-verify after recovery crashed: {e}")
            attempts.append(RecoveryAttempt(cls, detail, action, False, calls + 1))
            return RecoveryOutcome(
                succeeded=False,
                attempts=attempts,
                final_observation=f"post-verify crashed: {e}",
                escalated=True,
            )

        attempts.append(RecoveryAttempt(cls, detail, action, bool(vr.ok), calls + 2))

        if vr.ok:
            logger.info(f"[recovery] recovered via {action} on attempt {attempt_num + 1}")
            return RecoveryOutcome(
                succeeded=True,
                attempts=attempts,
                final_observation=getattr(vr, "observation", "") or "",
                escalated=False,
            )

        last_observation = getattr(vr, "observation", "") or ""

    # Exhausted attempts.
    logger.info(f"[recovery] exhausted {max_attempts} attempts — escalating")
    return RecoveryOutcome(
        succeeded=False,
        attempts=attempts,
        final_observation=last_observation,
        escalated=True,
    )


# ─── Checkpoint for computer_agent (best-effort, single-shot) ─────────────────

@dataclass
class CheckpointOutcome:
    diagnosed_class: str         # "overlay_appeared" | "error_shown" | "no_change" | "unknown"
    detail: str                  # diagnose observation
    action_taken: str            # "bbox_click" | "replan_input" | "retry" | "none"
    recovered: bool              # True if a strategy fired and succeeded
    cost_calls: int              # vision/text calls consumed


# Map computer_agent action names → recovery step's "action" field.
# Data-driven on purpose: new ca actions don't need new branches in checkpoint().
_CA_ACTION_NORMALIZE = {
    "keyboard_type": "type",
    "keyboard_press": "press",
    "keyboard_hotkey": "press",
    "mouse_click": "click",
    "mouse_double_click": "click",
    "mouse_right_click": "click",
    "vision_guided_click": "click",
    "find_and_click_text": "click",
    "find_and_double_click_text": "click",
}


def _synthesize_step_from_ca_action(ca_action: dict, task_goal: str) -> dict:
    """Translate a computer_agent action dict into the step format consumed
    by the recovery strategies (so _diagnose's step_summary and
    _overlay_goal_text can ground on something meaningful).

    The mapping is intentionally minimal — a new ca action just falls through
    to the 'else' branch with its name preserved.
    """
    name = ca_action.get("type") or ca_action.get("action") or ""
    normalized = _CA_ACTION_NORMALIZE.get(name, name or "?")
    params = dict(ca_action.get("params", {}) or {})
    # Some ca actions put params at top level (e.g. keyboard_press has 'key').
    for k in ("text", "key", "x", "y", "label", "keys"):
        if k not in params and k in ca_action:
            params[k] = ca_action[k]
    return {
        "type": "computer_agent",
        "action": normalized,
        "params": params,
        "goal": task_goal or "",
    }


# ─── Dialog-Engagement Gate ───────────────────────────────────────────────────
#
# Suppresses overlay dismissal when the agent has recently engaged with a
# modal element (typed into a field, clicked through a control, marked a
# select-TODO via Rule S, etc.). Prevents misclassifying a form-modal the
# agent was filling as an unwanted overlay and clicking its close X.
#
# Generic by construction — depends only on TODO timestamps and the generic
# _action_failed predicate. Works for form-dialogs, settings panels, file
# pickers, file dialogs, any modal UI surface. Never form-specific.
#
# The gate has a 2-batch sliding window: an engagement signal one or two
# batches old keeps the gate hot. Three-or-more stale = gate opens (legit
# new overlay can be dismissed). Window size empirically chosen to survive
# a single intervening screenshot_and_continue pseudo-batch without letting
# stale engagement suppress legitimately new overlays.

_ENGAGEMENT_WINDOW = 2  # batches


def _is_dialog_engagement_active(
    *,
    todo_snapshot: list[dict] | None,
    recent_action_results: list[str] | None,
    current_batch_idx: int,
    window: int = _ENGAGEMENT_WINDOW,
) -> tuple[bool, str]:
    """
    Decide whether the agent has recently engaged with the visible modal
    surface, in which case overlay dismissal must be suppressed.

    Primary signal: TODO progress timestamps. A TODO marked done or
    deferred (pending_visual_confirm) within the last `window` batches is
    high-precision evidence of intentional interaction with the surface.

    Fallback signal: when no TODOs are tracked (vision-only ad-hoc tasks),
    any non-failed action in the recent batch counts as engagement.

    Returns (engaged, reason). reason is a short human-readable string for
    logging; empty when not engaged.
    """
    threshold = max(0, current_batch_idx - window)

    if todo_snapshot:
        for todo in todo_snapshot:
            if not isinstance(todo, dict):
                continue
            # Recently marked done by Rule T or Rule C.
            stamp_done = todo.get("batch_marked_done", -1)
            if isinstance(stamp_done, int) and stamp_done >= threshold and stamp_done > 0:
                return True, (f"TODO #{todo.get('id', '?')} marked done "
                              f"in batch {stamp_done} (current={current_batch_idx})")
            # Recently deferred by Rule S — agent clicked into a dropdown
            # awaiting visual confirm.
            stamp_def = todo.get("batch_deferred", -1)
            if isinstance(stamp_def, int) and stamp_def >= threshold and stamp_def > 0:
                return True, (f"TODO #{todo.get('id', '?')} deferred "
                              f"in batch {stamp_def} (current={current_batch_idx})")
        # TODOs are tracked but none recently progressed → not engaged.
        return False, ""

    # Fallback for non-TODO tasks: any non-failed action in the recent results.
    if recent_action_results:
        # Inline mini-version of _action_failed to avoid a circular import
        # back to computer_agent. Keep semantics in sync if either changes.
        for r in recent_action_results:
            if not isinstance(r, str):
                continue
            rs = r.strip().lower()
            if not rs:
                continue
            if rs.startswith(("failed:", "error:", "aborted")):
                continue
            if "aborted_wrong_focus" in rs:
                continue
            return True, "non-failed action in recent batch (no TODO tracking)"

    return False, ""


async def checkpoint(
    *,
    goal: str,
    last_action: dict,
    page=None,
    active_window: Optional[str] = None,
    todo_snapshot: list[dict] | None = None,
    recent_action_results: list[str] | None = None,
    current_batch_idx: int = 0,
) -> CheckpointOutcome:
    """Single-shot, best-effort recovery used by computer_agent between
    action batches. Diagnoses the screen once; if the result is one of the
    three recoverable classes, fires the matching strategy. Never escalates,
    never loops — caller continues regardless.

    Dialog-engagement gate: when `todo_snapshot` / `recent_action_results`
    show recent successful interaction with the visible surface, the
    `overlay_appeared` dispatch is suppressed. This prevents the failure
    mode where checkpoints dismissed form-dialogs the agent was filling.
    The other classes (error_shown, no_change, unknown) are NOT gated —
    they are non-destructive.

    Reuses _diagnose and the three strategy functions wholesale.
    """
    diag = await _diagnose(goal, "", last_action)
    cls = diag["class"]
    detail = diag["detail"]

    # Success / unknown: no recovery action needed.
    if cls in ("success", "unknown"):
        return CheckpointOutcome(
            diagnosed_class=cls,
            detail=detail,
            action_taken="none",
            recovered=cls == "success",
            cost_calls=1,
        )

    # Dialog-engagement gate — only intercepts overlay_appeared (the destructive class).
    if cls == "overlay_appeared":
        gate_enabled = True
        try:
            from .. import config as _cfg
            gate_enabled = getattr(_cfg, "DIALOG_ENGAGEMENT_GATE_ENABLED", True)
        except Exception:
            pass
        if gate_enabled:
            engaged, reason = _is_dialog_engagement_active(
                todo_snapshot=todo_snapshot,
                recent_action_results=recent_action_results,
                current_batch_idx=current_batch_idx,
            )
            if engaged:
                logger.info(
                    f"[recovery] engagement gate: SUPPRESSED overlay dismiss — "
                    f"engagement active ({reason}); diag.detail={detail!r} "
                    f"diag.recovery_target={diag.get('recovery_target', '')!r}"
                )
                return CheckpointOutcome(
                    diagnosed_class=cls,             # honest about what we saw
                    detail=f"[gated: {reason}] {detail}",
                    action_taken="none",             # but no dismissive action ran
                    recovered=False,
                    cost_calls=1,
                )

    # Dispatch — strategy ↔ class is 1:1, same as attempt_recovery.
    if cls == "overlay_appeared":
        ok, calls = await _recover_overlay(
            goal, page, active_window, last_action,
            target_hint=diag.get("recovery_target", ""),
        )
        action = "bbox_click"
    elif cls == "error_shown":
        ok, calls = await _recover_error(last_action, detail, page, active_window)
        action = "replan_input"
    elif cls == "no_change":
        ok, calls = await _recover_no_change(last_action, page, active_window)
        action = "retry"
    else:
        # _VALID_CLASSES is closed; defensive fallback.
        return CheckpointOutcome(
            diagnosed_class=cls,
            detail=detail,
            action_taken="none",
            recovered=False,
            cost_calls=1,
        )

    return CheckpointOutcome(
        diagnosed_class=cls,
        detail=detail,
        action_taken=action,
        recovered=bool(ok),
        cost_calls=1 + calls,
    )
