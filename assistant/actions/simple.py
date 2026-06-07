"""Simple tool handlers: create_note, open_browser, get_time, small_talk,
unknown, reminders, avatar."""

import logging
import webbrowser
from datetime import datetime

from .. import config
from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")


def _sanitize_filename(name: str) -> str:
    """Remove invalid characters from a filename."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = name
    for c in invalid_chars:
        sanitized = sanitized.replace(c, "_")
    sanitized = sanitized.replace("..", "_")
    return sanitized.strip() or "untitled"


@tool_registry.decorator("create_note")
def handle_create_note(params: dict, llm_response: str) -> str:
    """Create a text note in the sandbox Notes directory."""
    title = params.get("title", "untitled")
    content = params.get("content", "")

    safe_name = _sanitize_filename(title)
    file_path = config.NOTES_DIR / f"{safe_name}.txt"

    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)

    file_path.write_text(content, encoding="utf-8")
    logger.info(f"Note created: {file_path}")
    return personality_say("note_created", title=title)


@tool_registry.decorator("open_browser")
def handle_open_browser(params: dict, llm_response: str) -> str:
    """Open a URL in the default web browser."""
    url = params.get("url", "")

    if not url:
        return personality_say("need_url")

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    webbrowser.open(url)
    return personality_say("url_opened", url=url)


@tool_registry.decorator("get_time")
def handle_get_time(params: dict, llm_response: str) -> str:
    """Return the current date and time."""
    now = datetime.now()
    return f"The current time is {now.strftime('%I:%M %p on %A, %B %d, %Y')}."


@tool_registry.decorator("small_talk")
def handle_small_talk(params: dict, llm_response: str) -> str:
    """Return the LLM's natural language response directly."""
    if llm_response:
        return llm_response
    return "I'm here and ready to help!"


@tool_registry.decorator("unknown")
def handle_unknown(params: dict, llm_response: str) -> str:
    """Fallback for unrecognized intents."""
    if llm_response:
        return llm_response
    return "I'm not sure what you're asking. Could you try rephrasing that?"


# --- Reminders ---

@tool_registry.decorator("set_reminder")
async def handle_set_reminder(params: dict, llm_response: str, bridge=None) -> str:
    from .. import reminders
    goal = params.get("goal", params.get("query", "")).strip()
    if not goal:
        return "What would you like me to remind you about, and when?"
    return await reminders.parse_and_save(goal)


@tool_registry.decorator("cancel_reminder")
async def handle_cancel_reminder(params: dict, llm_response: str, bridge=None) -> str:
    """Cancel pending reminders — all, or by keyword with synonym expansion."""
    from .. import reminders
    goal = params.get("goal", "").strip()
    if not goal:
        return "What reminders would you like me to cancel?"
    return await reminders.cancel_reminders(goal)


# --- Avatar ---

@tool_registry.decorator("hide_avatar")
async def handle_hide_avatar(params: dict, llm_response: str, bridge=None) -> str:
    if bridge:
        await bridge.send_command("hide_avatar")
    return "Okay, I'll hide for now!"


@tool_registry.decorator("show_avatar")
async def handle_show_avatar(params: dict, llm_response: str, bridge=None) -> str:
    if bridge:
        await bridge.send_command("show_avatar")
    return "I'm back!"
