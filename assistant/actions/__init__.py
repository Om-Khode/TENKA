"""
actions.py â€" Intent handlers (tools) for the Voice Assistant.

Mirrors the C# ToolRegistry.cs. Maps intent names to handler functions
and executes them with the extracted parameters.

Available tools:
  - create_note    : Save a text note to the sandbox directory
  - open_browser   : Open a URL in the default web browser
  - get_time       : Return the current date and time
  - get_weather    : Placeholder (offline, no API)
  - small_talk     : Return the LLM's conversational response
  - unknown        : Fallback handler
  - computer_task  : Run the agentic computer control loop
  - read_screen    : OCR the screen and summarize via LLM
  - find_and_click : Find text on screen and click it
  - code_executor  : Run LLM-generated Python code for system info / computations
  - memory_query   : Search past conversations and facts
"""

import logging

from .. import config

logger = logging.getLogger("actions")

from .registry import tool_registry
from .responses import personality_say
from .simple import (
    _sanitize_filename, handle_create_note, handle_open_browser,
    handle_get_time, handle_small_talk, handle_unknown,
    handle_set_reminder, handle_cancel_reminder,
    handle_hide_avatar, handle_show_avatar,
)
from .recording import (
    handle_start_recording, handle_stop_recording,
    handle_get_recording, handle_summarize_recording,
)
from .voice import handle_enroll_voice, handle_forget_voice
from .browser_cdp_setup import handle_browser_cdp_setup
from .camera import (
    handle_pending_camera_settings, handle_camera_look,
    handle_pending_forget_face, handle_meet_face,
    handle_recognize_face, handle_forget_face,
)
from .teaching import (
    start_teaching_session, start_batch_teaching,
    handle_pending_teaching, _parse_teaching_step,
    _step_description, _extract_slots_from_steps,
)
from .web import handle_web_search, handle_browse_url
from .memory_search import handle_memory_query, handle_store_memory
from .shortcuts import handle_manage_shortcut
from .procedures import handle_manage_procedure
from .schedule import handle_manage_schedule
from .monitors import handle_manage_monitor, handle_pending_monitor_disambig
from .file_ops import (
    handle_file_task,
    handle_pending_file_search, handle_pending_destructive,
)
from .pending_handlers import (
    handle_pending_device_auth, handle_pending_oauth_setup,
    handle_pending_messaging_disambig, handle_pending_messaging_send,
    handle_pending_incoming_message, handle_pending_knowledge_approval,
)
from .da_handlers import (
    handle_computer_task, handle_read_screen, handle_find_and_click,
    handle_planner, handle_code_executor,
    handle_browser_action, handle_app_action,
)
from .manifest_dispatch import handle_manifest_dispatch  # noqa: F401  (registers via decorator)

# --- Pending states ---
# Each replaces a (_pending_X, _pending_X_ts, _X_TIMEOUT) triplet.
# The planner snapshots via pending_registry.snapshot().
# Adding a new pending state = register one more PendingState here.

from ..pending import PendingState, pending_registry

pending_file_search = pending_registry.register(PendingState("file_search", timeout=60.0))
pending_destructive = pending_registry.register(PendingState("destructive", timeout=60.0))
pending_camera_settings = pending_registry.register(PendingState("camera_settings", timeout=60.0))
pending_forget_face = pending_registry.register(PendingState("forget_face", timeout=30.0))
pending_oauth_setup = pending_registry.register(PendingState("oauth_setup", timeout=120.0))
pending_device_auth = pending_registry.register(PendingState("device_auth", timeout=120.0))
pending_knowledge_approval = pending_registry.register(PendingState("knowledge_approval", timeout=60.0))
pending_messaging_send = pending_registry.register(PendingState("messaging_send", timeout=60.0))
pending_messaging_disambig = pending_registry.register(PendingState("messaging_disambig", timeout=60.0))
pending_incoming_messages = pending_registry.register(PendingState("incoming_messages", timeout=30.0))
pending_monitor_disambig = pending_registry.register(PendingState("monitor_disambig", timeout=30.0))
teaching_session = pending_registry.register(PendingState("teaching_session", timeout=300.0))

_destructive_disclosed: bool = False

# Background search result queue
import queue as _queue
_search_result_queue: _queue.Queue = _queue.Queue()


# â"€â"€â"€ Preference-Aware Defaults â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€


def _apply_preference_defaults(intent: str, params: dict) -> dict:
    """
    Fill missing action parameters from user preferences.

    Checks active preferences and injects defaults when the user hasn't
    specified an app, platform, or other routing detail. Also enriches
    code_executor goals with preference hints.

    Adds '_pref_applied' key to params when a preference is used,
    so downstream code can track it for confidence feedback.

    Args:
        intent: The detected intent name.
        params: The current parameter dict from intent detection.

    Returns:
        The (potentially enriched) params dict.
    """
    try:
        from .. import preferences

        # â"€â"€ Goal enrichment for code_executor / planner â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
        # These intents pass raw user speech as 'goal'. We append
        # preference hints so the code generator knows which apps to use.
        if intent in ("code_executor", "planner") and "goal" in params:
            hints = _build_goal_hints()
            if hints:
                params["_pref_hints"] = hints  # available for code_executor prompt

        # â"€â"€ Messaging platform defaults â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
        # When user says "message Arjun" without specifying platform
        goal = params.get("goal", "").lower()
        from ..core.known_apps import get_apps_by_category as _get_msg_apps
        _messaging_keywords = {"message", "text", "send"} | frozenset(_get_msg_apps("messaging_default"))
        if intent == "code_executor" and any(
            kw in goal for kw in _messaging_keywords
        ):
            # Check contact-specific preference first
            # Try to extract a contact name from the goal
            contact = _extract_contact_from_goal(goal)
            if contact:
                contact_pref = preferences.get_preference(
                    f"contact_{contact}_app"
                )
                if contact_pref and contact_pref["confidence"] >= preferences.CONFIDENCE_SILENT:
                    params.setdefault("_pref_platform", contact_pref["value"])
                    params["_pref_applied"] = f"contact_{contact}_app"
                    return params

            # Fall back to general messaging preference
            general_pref = preferences.get_preference("messaging_default")
            if general_pref and general_pref["confidence"] >= preferences.CONFIDENCE_SILENT:
                params.setdefault("_pref_platform", general_pref["value"])
                params["_pref_applied"] = "messaging_default"

        # â"€â"€ Music app defaults â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
        if intent == "code_executor" and any(
            kw in goal for kw in ("play", "music", "song", "playlist", "lo-fi", "lofi")
        ):
            music_pref = preferences.get_preference("music_app")
            if music_pref and music_pref["confidence"] >= preferences.CONFIDENCE_SILENT:
                params.setdefault("_pref_app", music_pref["value"])
                params["_pref_applied"] = "music_app"

        # â"€â"€ Environment defaults (project path, downloads, etc.) â"€â"€â"€â"€â"€
        if intent in ("file_task", "code_executor"):
            if "project" in goal:
                proj_pref = preferences.get_preference("project_path")
                if proj_pref and proj_pref["confidence"] >= preferences.CONFIDENCE_SILENT:
                    params.setdefault("_pref_path", proj_pref["value"])
                    params["_pref_applied"] = "project_path"
            elif "download" in goal:
                dl_pref = preferences.get_preference("downloads_folder")
                if dl_pref and dl_pref["confidence"] >= preferences.CONFIDENCE_SILENT:
                    params.setdefault("_pref_path", dl_pref["value"])
                    params["_pref_applied"] = "downloads_folder"

    except Exception as e:
        logger.debug(f"Preference defaults failed (non-fatal): {e}")

    return params


def _build_goal_hints() -> str:
    """
    Build a short hint string from active routing/environment preferences
    for injection into code_executor or planner prompts.

    Returns:
        A string like "User preferences: music_app=<app>, messaging_default=<app>"
        or empty string if no preferences qualify.
    """
    try:
        from .. import preferences

        prefs = preferences.get_active_preferences(
            min_confidence=preferences.CONFIDENCE_SILENT
        )
        routing = [
            p for p in prefs
            if p["category"] in ("app_routing", "contact_routing", "environment")
        ]
        if not routing:
            return ""

        pairs = [f"{p['key']}={p['value']}" for p in routing]
        return "User preferences: " + ", ".join(pairs)

    except Exception:
        return ""


def _extract_contact_from_goal(goal: str) -> str:
    """
    Try to extract a contact name from a messaging goal string.
    Very basic â€" looks for common patterns like 'message arjun', 'text mom',
    'send to john'. Returns lowercase name or empty string.
    """
    import re
    from ..core.known_apps import get_apps_by_category as _get_cat
    _msg_app_alt = '|'.join(re.escape(a) for a in _get_cat("messaging_default"))
    patterns = [
        r"(?:message|text|send\s+(?:a\s+)?(?:message\s+)?to)\s+(\w+)",
        rf"(?:{_msg_app_alt})\s+(?:to\s+)?(\w+)",
        r"(?:tell|ask)\s+(\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, goal, re.IGNORECASE)
        if match:
            name = match.group(1).lower()
            # Filter out common non-name words
            if name not in ("me", "my", "the", "a", "an", "it", "that", "this"):
                return name
    return ""


async def execute(intent: str, params: dict, llm_response: str = "",
                  bridge=None, _from_planner: bool = False) -> str:
    """
    Execute the tool matching the given intent.

    Args:
        intent:        The intent name (e.g. "create_note").
        params:        Dictionary of parameters from the IntentResult.
        llm_response:  The LLM's conversational response (used for small_talk/unknown).
        bridge:        Optional UnityBridge instance (needed for computer_task).
        _from_planner: If True, skip multi-step re-routing in code_executor
                       to prevent plannerâ†’code_executorâ†’planner loops.

    Returns:
        A human-readable response string describing what happened.
    """
    if intent == "read_file":
        intent = "file_task"

    # Apply preference defaults before routing
    params = _apply_preference_defaults(intent, params)

    # Look up the handler; fall back to handle_unknown
    handler = tool_registry.get(intent) or handle_unknown

    try:
        # Check if the handler is async (new handlers)
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            # Pass _from_planner to code_executor if it accepts it
            import inspect
            sig = inspect.signature(handler)
            if '_from_planner' in sig.parameters:
                result = await handler(params, llm_response, bridge,
                                       _from_planner=_from_planner)
            else:
                result = await handler(params, llm_response, bridge)
        else:
            result = handler(params, llm_response)
        logger.info(f"Executed '{intent}': {result}")

        # Track successful preference application
        # If a preference was applied and the handler completed without
        # the user correcting it, record the successful use.
        _pref_key = params.get("_pref_applied")
        if _pref_key:
            try:
                from .. import preferences
                preferences.record_preference_used(_pref_key)
                logger.debug(f"Preference '{_pref_key}' applied successfully")
            except Exception:
                pass

        return result
    except Exception as e:
        logger.error(f"Error executing '{intent}': {e}")
        return f"Sorry, I encountered an error: {e}"


