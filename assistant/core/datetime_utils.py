"""Datetime context for LLM prompts.

Ensures LLMs can interpret relative date words ("today", "tomorrow")
in user input by providing the current date/time as absolute context.
"""

from datetime import datetime


def date_context_line() -> str:
    """One-line current date/time string for injection into LLM prompts."""
    now = datetime.now()
    return f"Current date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
