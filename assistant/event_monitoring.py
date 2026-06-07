"""Event monitoring domain facade.

Orchestrates monitor creation, validation, listing, and lifecycle.
Handlers in actions/monitors.py call this; this module calls storage
and automation layers.
"""
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger("event_monitoring")

def _get_event_types() -> set[str]:
    from assistant.automation.event_sources import source_registry
    types: set[str] = set()
    for source in source_registry.list_all().values():
        types |= source.event_types
    return types


_EVENT_TYPES = _get_event_types()

_DUMMY_EVENTS = {
    "media_changed": {
        "event_type": "media_changed",
        "source_app": "TestApp",
        "title": "Test Title",
        "artist": "Test Artist",
        "album": "Test Album",
        "playback_status": "playing",
        "timestamp": "2026-01-01T00:00:00",
    },
    "window_focus": {
        "event_type": "window_focus",
        "source_app": "TestApp",
        "window_title": "Test Window",
        "prev_app": "",
        "prev_title": "",
        "timestamp": "2026-01-01T00:00:00",
    },
    "window_title": {
        "event_type": "window_title",
        "source_app": "TestApp",
        "window_title": "Test Window",
        "timestamp": "2026-01-01T00:00:00",
    },
}


def _get_repo():
    from assistant.storage.db import get_db
    from assistant.storage.repos.monitor import MonitorRepo
    db = get_db()
    if db is None:
        raise RuntimeError("DB not initialized")
    return MonitorRepo(db)


def _validate_condition(expr: str, event_type: str) -> bool:
    from assistant.automation.event_bus import compile_condition, eval_condition_code

    compiled = compile_condition(expr)
    if compiled is None:
        return False
    dummy = _DUMMY_EVENTS.get(event_type, _DUMMY_EVENTS["media_changed"])
    try:
        eval_condition_code(compiled, dummy)
        return True
    except Exception:
        return False


async def create_monitor(goal: str) -> str:
    from assistant.llm.contracts import ask_for_monitor_definition
    from assistant.automation.event_bus import event_bus

    defn = await ask_for_monitor_definition(goal)
    if defn is None:
        return "Sorry, I couldn't understand that monitor request."

    event_type = defn.get("event_type")
    if not event_type or event_type not in _EVENT_TYPES:
        types = ", ".join(sorted(_EVENT_TYPES))
        return f"I can't monitor that yet. I can watch for: {types}."

    repo = _get_repo()
    active = repo.get_active()
    from assistant import config
    max_active = getattr(config, "EVENT_MONITOR_MAX_ACTIVE", 20)
    if len(active) >= max_active:
        return f"You already have {len(active)} monitors. Delete some first."

    condition_mode = "code"
    condition_expr = defn.get("condition_code")
    condition_prompt = defn.get("condition_prompt")
    validation_failed = False

    if condition_expr and not _validate_condition(condition_expr, event_type):
        defn_retry = await ask_for_monitor_definition(
            f"{goal} (previous condition_code was invalid, try again)"
        )
        if defn_retry and defn_retry.get("condition_code"):
            condition_expr = defn_retry["condition_code"]
            if not _validate_condition(condition_expr, event_type):
                condition_expr = None
                validation_failed = True
        else:
            condition_expr = None
            validation_failed = True

    if condition_expr is None:
        if condition_prompt:
            condition_mode = "llm"
        elif validation_failed:
            return "Sorry, I couldn't figure out the right condition for that."

    monitor_id = repo.create(
        name=defn["name"],
        event_type=event_type,
        source_filter=defn.get("source_filter"),
        condition_mode=condition_mode,
        condition_expr=condition_expr,
        condition_prompt=condition_prompt,
        action_type=defn["action_type"],
        action_payload=defn["action_payload"],
        cooldown_secs=defn.get("cooldown_secs", 5),
        user_goal=goal,
    )

    event_bus.reload_monitors()
    return f"Done. I'll watch for that — monitor '{defn['name']}' is active."


def list_monitors() -> str:
    repo = _get_repo()
    monitors = repo.get_all()
    if not monitors:
        return "You don't have any event monitors."

    lines = []
    for m in monitors:
        status = "" if m["enabled"] else " (paused)"
        fired = f" — fired {m['fire_count']} times" if m["fire_count"] else ""
        lines.append(f"{m['name']}{status}{fired}")

    if len(lines) == 1:
        return f"You have 1 monitor: {lines[0]}."
    return f"You have {len(lines)} monitors: {', '.join(lines)}."


_NOISE_WORDS = frozenset({
    "pause", "resume", "stop", "start", "enable", "disable",
    "delete", "remove", "cancel", "the", "my", "a", "an",
    "monitor", "event", "watcher",
})


def _fuzzy_match(monitors: list[dict], goal: str) -> list[dict]:
    goal_lower = goal.lower()
    goal_words = {w for w in goal_lower.split() if w not in _NOISE_WORDS}

    scored = []
    for m in monitors:
        name_lower = m["name"].lower()
        if name_lower in goal_lower or goal_lower in name_lower:
            scored.append((m, 1.0))
            continue
        name_words = set(name_lower.split())
        overlap = goal_words & name_words
        if overlap:
            scored.append((m, len(overlap) / len(name_words)))

    if not scored:
        return []
    best_score = max(score for _, score in scored)
    return [m for m, score in scored if score == best_score]


def _set_disambig(action: str, matches: list[dict]) -> str:
    import assistant.actions as _act
    _act.pending_monitor_disambig.set({
        "action": action,
        "matches": matches,
    })
    names = [m["name"] for m in matches]
    if len(names) != len(set(names)):
        labels = [f"{m['name']} (#{m['id']})" for m in matches]
    else:
        labels = names
    return f"I found multiple matches: {', '.join(labels)}. Which one?"


def pause_monitor(goal: str) -> str:
    from assistant.automation.event_bus import event_bus

    repo = _get_repo()
    monitors = repo.get_all()
    if not monitors:
        return "You don't have any monitors to pause."

    matches = _fuzzy_match(monitors, goal)
    if not matches:
        return "I couldn't find a monitor matching that name."
    if len(matches) > 1:
        return _set_disambig("pause", matches)

    repo.toggle(matches[0]["id"], enabled=False)
    event_bus.reload_monitors()
    return f"Paused the '{matches[0]['name']}' monitor."


def resume_monitor(goal: str) -> str:
    from assistant.automation.event_bus import event_bus

    repo = _get_repo()
    monitors = repo.get_all()
    if not monitors:
        return "You don't have any monitors to resume."

    matches = _fuzzy_match(monitors, goal)
    if not matches:
        return "I couldn't find a monitor matching that name."
    if len(matches) > 1:
        return _set_disambig("resume", matches)

    repo.toggle(matches[0]["id"], enabled=True)
    event_bus.reload_monitors()
    return f"Resumed the '{matches[0]['name']}' monitor."


def delete_monitor(goal: str) -> str:
    from assistant.automation.event_bus import event_bus

    repo = _get_repo()
    monitors = repo.get_all()
    if not monitors:
        return "You don't have any monitors to delete."

    goal_lower = goal.lower()
    if "all" in goal_lower:
        count = len(monitors)
        for m in monitors:
            repo.delete(m["id"])
        event_bus.reload_monitors(flush_pending=True)
        return f"Deleted all {count} monitors."

    matches = _fuzzy_match(monitors, goal)
    if not matches:
        return "I couldn't find a monitor matching that name."
    if len(matches) > 1:
        return _set_disambig("delete", matches)

    repo.delete(matches[0]["id"])
    event_bus.reload_monitors(flush_pending=True)
    return f"Deleted the '{matches[0]['name']}' monitor."


def resolve_disambig(user_text: str) -> str | None:
    """Resolve a pending monitor disambiguation. Returns response or None."""
    import assistant.actions as _act
    from assistant.automation.event_bus import event_bus

    if not _act.pending_monitor_disambig.active:
        return None

    payload = _act.pending_monitor_disambig.payload
    if payload is None:
        return None

    text_low = user_text.strip().lower()
    if any(w in text_low for w in ("cancel", "never mind", "forget it", "stop", "abort")):
        _act.pending_monitor_disambig.clear()
        return "Okay, cancelled."

    action = payload["action"]
    matches = payload["matches"]
    _act.pending_monitor_disambig.clear()

    import re
    id_match = re.search(r"#?(\d+)", user_text.strip())
    if id_match:
        target_id = int(id_match.group(1))
        by_id = [m for m in matches if m["id"] == target_id]
        if by_id:
            target = by_id[0]
        else:
            return "That ID doesn't match any of the options."
    else:
        picked = _fuzzy_match(matches, user_text)
        if not picked:
            return "I couldn't match that to any of the options."
        target = picked[0]

    repo = _get_repo()
    if action == "delete":
        repo.delete(target["id"])
        event_bus.reload_monitors(flush_pending=True)
        return f"Deleted the '{target['name']}' monitor."
    elif action == "pause":
        repo.toggle(target["id"], enabled=False)
        event_bus.reload_monitors()
        return f"Paused the '{target['name']}' monitor."
    elif action == "resume":
        repo.toggle(target["id"], enabled=True)
        event_bus.reload_monitors()
        return f"Resumed the '{target['name']}' monitor."
    return None
