"""Simple tool handlers: create_note, open_browser, get_time, small_talk,
unknown, reminders, avatar."""

import logging
import re
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


def _unused_note_path(base: "pathlib.Path") -> "pathlib.Path":
    """`groceries.txt`, or `groceries (2).txt` if that is taken.

    Bounded, and the bound is not decoration: a directory somebody has filled
    with a thousand `groceries (n).txt` is a directory where the right answer
    is to say so, not to spin.
    """
    if not base.exists():
        return base
    for n in range(2, 1000):
        candidate = base.with_name(f"{base.stem} ({n}){base.suffix}")
        if not candidate.exists():
            return candidate
    return base.with_name(f"{base.stem} (overflow){base.suffix}")


@tool_registry.decorator("create_note")
def handle_create_note(params: dict, llm_response: str) -> str:
    """Create a text note in the sandbox Notes directory.

    **Never truncates an existing note.** It did, and it cost one:

        21:58:48  'make a note called groceries saying milk and vegetables'
                  -> Notes\groceries.txt          (the contents)
        22:00:28  'make a note called groceries'
                  -> params {'title': 'groceries'}  (no content at all)
                  -> Notes\groceries.txt           (zero bytes)
                  -> "Done -- 'groceries' is saved."

    A bare `write_text(content)` with `content=""` emptied a note written a
    hundred seconds earlier, and the reply called it saving. Two separate
    failures in one line: the write was destructive, and the sentence
    describing it was false.

    Both directions are handled here rather than by asking the classifier to
    behave. It is doing nothing wrong -- "make a note called groceries" really
    does name a note and give it no body, and deciding what to do about that
    is this handler's job:

    - **the name is taken and there is content** -- write beside it, under a
      numbered name, and say which file it went to. Overwriting is the one
      thing that cannot be undone from here; a second file can be deleted in a
      breath.
    - **the name is taken and there is no content** -- change nothing at all,
      and say the note already exists. An empty request is not permission to
      empty the note.
    - **the name is free** -- create it, and if it is empty, say that it is
      empty rather than implying something was written into it.
    """
    import pathlib

    title = params.get("title", "untitled")
    content = params.get("content", "") or ""

    safe_name = _sanitize_filename(title)
    file_path = config.NOTES_DIR / f"{safe_name}.txt"

    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        if not content.strip():
            logger.info(f"Note left untouched (nothing to add): {file_path}")
            return (f"'{title}' already exists and you didn't give me anything "
                    f"to put in it, so I left it alone.")

        target = _unused_note_path(file_path)
        target.write_text(content, encoding="utf-8")
        logger.info(f"Note created beside an existing one: {target}")
        return (f"'{title}' already existed, so I saved this one as "
                f"'{target.stem}' instead of overwriting it.")

    file_path.write_text(content, encoding="utf-8")
    logger.info(f"Note created: {file_path}")
    if not content.strip():
        return f"Created '{title}' — it's empty, so tell me what goes in it."
    return personality_say("note_created", title=title)


# 6a.5 review H1, sink-side hardening. The planner's egress guard decides
# whether an untrusted step output may become a URL at all; this is the last
# line before the address bar, and it applies on every path including the
# direct intent one. Deliberately provenance-free and narrow: it rejects what
# is never a legitimate navigation target rather than trying to judge hosts,
# so "open localhost:3000" keeps working for the developer at the keyboard.
_NON_WEB_SCHEME_RE = re.compile(
    r"(?i)^\s*(?:javascript|data|file|vbscript|about|blob|chrome|res)\s*:")

#: Credentials in the authority (`https://bank.example@evil.host`) are a
#: phishing primitive and never appear in an honest link.
_URL_USERINFO_RE = re.compile(r"(?i)^(?:https?://)?[^/?#]*@")

#: A URL is one line. Anything with a newline in it is a document that
#: reached this param, which is the shape the review's H1 attack produced.
_MAX_URL_CHARS = 2048


@tool_registry.decorator("open_browser")
def handle_open_browser(params: dict, llm_response: str) -> str:
    """Open a URL in the default web browser."""
    url = params.get("url", "")

    if not url:
        return personality_say("need_url")

    url = url.strip()

    if (_NON_WEB_SCHEME_RE.match(url)
            or len(url) > _MAX_URL_CHARS
            or any(c in url for c in "\r\n\t")
            or _URL_USERINFO_RE.match(url)):
        logger.warning(
            f"[BROWSER] Refused to open a value that is not a web URL: "
            f"{url[:80]!r}"
        )
        return ("That doesn't look like a web address I should open, so I "
                "didn't.")

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
