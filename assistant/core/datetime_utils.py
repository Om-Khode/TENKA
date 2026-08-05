"""Datetime context for LLM prompts.

Ensures LLMs can interpret relative date words ("today", "tomorrow")
in user input by providing the current date/time as absolute context.
"""

from datetime import datetime, timezone


def date_context_line() -> str:
    """One-line current date/time string for injection into LLM prompts."""
    now = datetime.now()
    return f"Current date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


def humanize_relative(iso_timestamp: str) -> str:
    """"2 hours ago"-style rendering of a stored ISO timestamp, for speech.

    Same bucket semantics as session.py's _format_gap. Handles both
    tz-aware (backup's datetime.now(timezone.utc).isoformat()) and naive
    timestamps — a naive one is assumed to already be UTC.
    """
    try:
        then = datetime.fromisoformat(iso_timestamp)
    except (ValueError, TypeError):
        return "at an unknown time"

    now = datetime.now(timezone.utc)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)

    total_seconds = int((now - then).total_seconds())
    if total_seconds < 300:
        return "just now"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes} minutes ago"
    hours = total_seconds // 3600
    if hours < 24:
        return f"{hours} hours ago"
    days = total_seconds // 86400
    return f"{days} days ago"
