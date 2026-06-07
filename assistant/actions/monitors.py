"""Handler for the manage_monitor intent ."""
from __future__ import annotations

import logging
import re

from .registry import tool_registry

logger = logging.getLogger(__name__)

_LIST_RE = re.compile(
    r"\b(list|show|display|what)\b.*\b(monitor|watcher|event)",
    re.IGNORECASE,
)
_PAUSE_RE = re.compile(
    r"\b(pause|stop|disable|deactivate)\b.*\b(monitor|watcher|the\b)",
    re.IGNORECASE,
)
_RESUME_RE = re.compile(
    r"\b(resume|start|enable|activate|unpause)\b.*\b(monitor|watcher|the\b)",
    re.IGNORECASE,
)
_DELETE_RE = re.compile(
    r"\b(delete|remove|cancel|clear)\b.*\b(monitor|watcher|all|the\b)",
    re.IGNORECASE,
)


def _detect_action(goal: str) -> str:
    if _LIST_RE.search(goal):
        return "list"
    if _DELETE_RE.search(goal):
        return "delete"
    if _PAUSE_RE.search(goal):
        return "pause"
    if _RESUME_RE.search(goal):
        return "resume"
    return "create"


@tool_registry.decorator("manage_monitor")
async def handle_manage_monitor(
    params: dict, llm_response: str, bridge=None
) -> str:
    from assistant import event_monitoring

    goal = params.get("goal", "")
    action = _detect_action(goal)

    if action == "list":
        return event_monitoring.list_monitors()
    elif action == "pause":
        return event_monitoring.pause_monitor(goal)
    elif action == "resume":
        return event_monitoring.resume_monitor(goal)
    elif action == "delete":
        return event_monitoring.delete_monitor(goal)
    else:
        return await event_monitoring.create_monitor(goal)


async def handle_pending_monitor_disambig(text: str, bridge=None) -> str | None:
    from assistant import event_monitoring
    return event_monitoring.resolve_disambig(text)
