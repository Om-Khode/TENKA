"""
intent_scopes.py — Runtime intent scope detection.

Domain-layer module. Reads system state from automation/ and recording to
determine the active scope. Sticky scope persists for 2 turns after state ends.
"""

import logging

from .core.intent_scopes import SCOPES, ALWAYS_AVAILABLE

logger = logging.getLogger("intent_scopes")

_STICKY_TURNS = 2

_last_scope: tuple[str, int] = ("general", 0)


# ─── State Accessors (mockable seams) ─────────────────────────────────────

def _get_browser_driver_available() -> bool:
    """Can the browser tier drive a browser the user is actually using?

    This does not merely pick a code path -- it decides whether the whole turn
    is scoped to `browser_mode`, which is what TENKA believes she can currently
    do. Left pointing at a mechanism that no longer exists, she would go on
    claiming a browser affordance she had lost, and the first sign of it would
    be a task that quietly does nothing.

    False when nothing is connected, deliberately: the bundled browser is a
    fallback for a task already under way, not a reason to believe the user's
    browser is drivable.
    """
    try:
        from .io.api.extension_ws import latch_state_snapshot
        return latch_state_snapshot().connected
    except Exception:
        return False


def _get_recording_active() -> bool:
    try:
        from . import recording
        return recording.is_active()
    except Exception:
        return False


def _get_camera_pending() -> bool:
    try:
        from .pending import pending_registry
        state = pending_registry.get("pending_camera_settings")
        return state is not None and state.active
    except Exception:
        return False


# ─── Scope Detection ──────────────────────────────────────────────────────

def _get_all_intents() -> set[str]:
    result = set(ALWAYS_AVAILABLE)
    for scope_intents in SCOPES.values():
        result |= scope_intents
    return result


def detect_scope(turn_number: int) -> tuple[str, set[str]]:
    global _last_scope

    detected = "general"
    if _get_browser_driver_available():
        detected = "browser_mode"
    elif _get_recording_active():
        detected = "recording_mode"
    elif _get_camera_pending():
        detected = "camera_mode"

    if detected != "general":
        _last_scope = (detected, turn_number)
    elif _last_scope[0] != "general" and turn_number - _last_scope[1] <= _STICKY_TURNS:
        detected = _last_scope[0]
        logger.debug(
            f"[SCOPE] Sticky: {detected} (set at turn {_last_scope[1]}, "
            f"now turn {turn_number})"
        )
    else:
        _last_scope = ("general", turn_number)

    if detected == "general":
        return ("general", _get_all_intents())

    active = set(ALWAYS_AVAILABLE) | SCOPES[detected]
    logger.info(f"[SCOPE] {detected} — {len(active)} intents active")
    return (detected, active)
