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
from .backup import handle_manage_backup
from .backup_pending import (
    handle_pending_backup_confirm_phrase, handle_pending_backup_oauth,
    handle_pending_backup_unlock_phrase, handle_pending_backup_restore_phrase,
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
pending_backup_confirm_phrase = pending_registry.register(PendingState("backup_confirm_phrase", timeout=180.0))
# 300s: the "I don't have an OAuth app yet" branch sends the user off to
# create one, enable the Drive API, and set a scope. Each step also
# .touch()es, so this is a per-step silence budget, not a total budget.
pending_backup_oauth = pending_registry.register(PendingState("backup_oauth", timeout=300.0))
pending_backup_unlock_phrase = pending_registry.register(PendingState("backup_unlock_phrase", timeout=60.0))
pending_backup_restore_phrase = pending_registry.register(PendingState("backup_restore_phrase", timeout=60.0))

_destructive_disclosed: bool = False

# Background search result queue
import queue as _queue
_search_result_queue: _queue.Queue = _queue.Queue()


# ─── Capability grants for the turn in flight ────────────────────────────

import contextvars as _contextvars

from ..core.capabilities import Capability
from ..core.intent_capabilities import DEFAULT_REQUIRED, REQUIRED_CAPABILITY

current_grants: _contextvars.ContextVar["frozenset[Capability] | None"] = \
    _contextvars.ContextVar("tenka_current_grants", default=None)
"""What the caller driving this turn is allowed to ask for.

A contextvar rather than a parameter for the same reason
`_telemetry.set_current_tracker` is one: `execute()` re-enters itself through
`planner/executor.py`, and threading a grant set through every handler
signature would mean every future handler is one forgotten argument away from
running unchecked. A contextvar is inherited by nested tasks automatically.

**The default is `None`, and `None` refuses.** Not "no restriction" -- the
absence of a decision is not a decision to allow. A turn that reaches dispatch
without anyone setting this is a bug, and the safe behaviour for that bug is
to do nothing.
"""

LOCAL_GRANTS: "frozenset[Capability]" = frozenset(Capability)
"""What a caller physically at this machine holds: everything.

Voice, the console, and the background automation runners (the scheduler and
the event bus) all set this explicitly. Setting it is a deliberate grant --
the person is at the keyboard, or the thing firing was installed by someone
who held EXECUTE at the time. It is never what an unset contextvar falls back
to; see `current_grants`.
"""


def set_grants(grants: "frozenset[Capability]") -> "_contextvars.Token":
    """Declare what the turn about to run may do. Reset the token when it ends.

    Deliberately a plain setter with no default: every call site has to name a
    grant set, so "which caller is this?" is answered once, visibly, at the
    place that knows.
    """
    return current_grants.set(grants)


def _refuse(required: Capability) -> str:
    """The refusal a caller sees when its grant set does not cover the intent.

    Names the capability -- the operator needs to know which one to grant --
    and nothing else. It must not leak what the intent would have done: a
    device that cannot reach `shutdown` should not learn that shutdown is a
    thing she can do. Under 120 chars with no paths and no error codes,
    because it may be spoken.
    """
    return (f"That needs the {required.value} permission, "
            f"which this device doesn't have.")


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

    # ── The capability gate ──────────────────────────────────────────────
    # This is the only site in the tree that resolves a handler, and
    # planner/executor.py re-enters through it, so a planned step is checked
    # by the same rule as a direct turn, recursively -- without the planner
    # knowing the rule exists.
    #
    # Placed above _apply_preference_defaults deliberately: a refused intent
    # must not first read the preference store and mutate its params. The
    # refusal costs one dict lookup and touches nothing.
    #
    # An unclassified intent requires EXECUTE, and an unset grant set refuses
    # everything. Both are the fail-closed direction; see
    # core/intent_capabilities.py and `current_grants`.
    _required = REQUIRED_CAPABILITY.get(intent, DEFAULT_REQUIRED)
    _granted = current_grants.get()
    if _granted is None or _required not in _granted:
        logger.info(
            f"Refused '{intent}': needs {_required.value}, caller holds "
            f"{'nothing (grants unset)' if _granted is None else sorted(c.value for c in _granted)}"
        )
        return _refuse(_required)

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


