"""Handler for the manage_schedule intent ."""
from __future__ import annotations

import logging
import re
from datetime import datetime

from croniter import croniter

from .registry import tool_registry

logger = logging.getLogger(__name__)


def _get_repo():
    from assistant.storage.db import get_db
    from assistant.storage.repos.schedule import ScheduleRepo
    db = get_db()
    if db is None:
        raise RuntimeError("DB not initialized")
    return ScheduleRepo(db)


@tool_registry.decorator("manage_schedule")
async def handle_manage_schedule(
    params: dict, llm_response: str, bridge=None
) -> str:
    action = params.get("action", "create")
    if action == "create":
        return await _create_schedule(params.get("goal", ""))
    elif action == "list":
        return _list_schedules()
    elif action == "cancel":
        return _cancel_schedule(params.get("goal", ""))
    elif action == "toggle":
        return _toggle_schedule(params.get("goal", ""))
    return "I'm not sure what to do with that schedule command."


_EXPLICIT_MINUTE_CADENCE = re.compile(
    r"\bevery\s+(?:(\d+)\s*)?(?:minute|min)s?\b", re.IGNORECASE,
)
""""every minute" / "every 5 minutes" / "every 15 mins", stated plainly.

Deterministic because the model is *inconsistent* here, not incapable. Two
schedules created minutes apart in the same session, both saying "every
minute": one got `* * * * *` and fired on the minute as asked, the other got
`0 * * * *` -- hourly -- and the operator was told "Got it, I scheduled ..."
either way. Nothing in the parse prompt pinned the mapping, so the same words
landed on two different cadences.

That is the shape `.claude/rules/llm-and-intents.md` names outright: skip the
LLM whenever a deterministic path exists. An explicit interval in the
utterance is not a judgement call.

Deliberately narrow: it wants a number or nothing between "every" and
"minute(s)". "every few minutes" and "every couple of minutes" are left to the
model, because a word is a judgement and the point here is to pin what is not.

"every minute or so" **is** claimed -- the phrase "every minute" is in it, and
roughly-every-minute is what `* * * * *` does. Noted because the first draft of
this docstring said the opposite about that case and the regex was right.
"""


def _explicit_minute_cron(text: str) -> "str | None":
    """The cron a plainly-stated minute cadence demands, or None.

    Returns `* * * * *` for "every minute" and `*/N * * * *` for "every N
    minutes". `N >= 60` is rejected rather than clamped: `*/60` is not a valid
    minute field, and a request that large means something the caller should
    phrase as hours.
    """
    match = _EXPLICIT_MINUTE_CADENCE.search(text or "")
    if match is None:
        return None
    raw = match.group(1)
    if raw is None:
        return "* * * * *"
    try:
        n = int(raw)
    except ValueError:            # pragma: no cover - the group is \d+
        return None
    if n <= 0 or n >= 60:
        return None
    if n == 1:
        return "* * * * *"
    return f"*/{n} * * * *"


async def _create_schedule(goal: str) -> str:
    from assistant.llm.contracts import ask_for_schedule_parse

    parsed = await ask_for_schedule_parse(goal)
    if parsed is None:
        return "Sorry, I couldn't understand that schedule request."

    cron_expr = parsed.get("cron_expr")

    # A plainly-stated minute cadence overrides the model. Not a fallback for
    # when it fails -- an override for when it answers something else, which is
    # what actually happened. Silently turning a one-minute check into an hourly
    # one is the kind of wrong that reads as working software.
    _explicit = _explicit_minute_cron(goal)
    if _explicit and _explicit != cron_expr:
        logger.info(
            f"[SCHEDULE] Cadence stated explicitly in the request: using "
            f"{_explicit!r} instead of the parsed {cron_expr!r}"
        )
        cron_expr = _explicit

    if not cron_expr or not croniter.is_valid(cron_expr):
        return "Sorry, I couldn't figure out the timing for that schedule."

    next_fire = croniter(cron_expr, datetime.now()).get_next(datetime).isoformat()

    repo = _get_repo()
    repo.create(
        name=parsed["name"],
        cron_expr=cron_expr,
        task_type=parsed.get("task_type", "web_search"),
        task_goal=parsed.get("goal", goal),
        notify_mode=parsed.get("notify_mode", "on_match_only"),
        condition_text=parsed.get("condition_text"),
        next_fire_at=next_fire,
    )

    return f"Got it, I scheduled '{parsed['name']}'."


def _list_schedules() -> str:
    repo = _get_repo()
    schedules = repo.list_all()
    if not schedules:
        return "You don't have any schedules."

    names = []
    for s in schedules:
        status = "" if s["enabled"] else " (paused)"
        names.append(f"{s['name']}{status}")

    if len(names) == 1:
        return f"You have 1 schedule: {names[0]}."
    return f"You have {len(names)} schedules: {', '.join(names)}."


_NOISE_WORDS = frozenset({
    "cancel", "delete", "remove", "pause", "disable", "resume", "enable",
    "the", "my", "a", "an", "schedule", "monitor", "task",
})


def _fuzzy_match(schedules: list[dict], goal: str) -> list[dict]:
    goal_lower = goal.lower()
    goal_words = {w for w in goal_lower.split() if w not in _NOISE_WORDS}

    scored = []
    for s in schedules:
        name_lower = s["name"].lower()
        if name_lower in goal_lower or goal_lower in name_lower:
            scored.append((s, 1.0))
            continue
        name_words = set(name_lower.split())
        overlap = goal_words & name_words
        if overlap:
            scored.append((s, len(overlap) / len(name_words)))

    if not scored:
        return []
    best_score = max(score for _, score in scored)
    return [s for s, score in scored if score == best_score]


def _cancel_schedule(goal: str) -> str:
    repo = _get_repo()
    schedules = repo.list_all()
    if not schedules:
        return "You don't have any schedules to cancel."

    matches = _fuzzy_match(schedules, goal)

    if len(matches) == 0:
        return "I couldn't find a schedule matching that name."
    if len(matches) > 1:
        names = ", ".join(m["name"] for m in matches)
        return f"I found multiple matches: {names}. Which one?"

    repo.delete(matches[0]["id"])
    return f"Cancelled the '{matches[0]['name']}' schedule."


def _toggle_schedule(goal: str) -> str:
    repo = _get_repo()
    schedules = repo.list_all()
    if not schedules:
        return "You don't have any schedules."

    matches = _fuzzy_match(schedules, goal)

    if len(matches) == 0:
        return "I couldn't find a schedule matching that name."
    if len(matches) > 1:
        names = ", ".join(m["name"] for m in matches)
        return f"I found multiple matches: {names}. Which one?"

    target = matches[0]
    new_state = not bool(target["enabled"])
    repo.toggle(target["id"], enabled=new_state)

    verb = "Resumed" if new_state else "Paused"
    return f"{verb} the '{target['name']}' schedule."
