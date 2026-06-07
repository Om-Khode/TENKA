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

def _get_cdp_available() -> bool:
    try:
        from .automation.browser.cdp import cdp_state_snapshot
        snap = cdp_state_snapshot()
        return snap is not None and snap.available
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
    if _get_cdp_available():
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
