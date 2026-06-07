"""Procedure management handler: list, delete, rename, edit taught procedures."""

import logging

from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")


@tool_registry.decorator("manage_procedure")
async def handle_manage_procedure(params: dict, llm_response: str, bridge=None) -> str:
    """Handle procedure listing, deletion, renaming, and editing. Zero LLM cost."""
    import assistant.actions as _act
    from .. import procedures

    action = params.get("action", "").lower()
    name = params.get("name", "").strip()
    goal = params.get("goal", "").strip()

    if not action and goal:
        from .. import regex_router
        parsed = regex_router.match_procedure_command(goal)
        if parsed:
            action = parsed.params.get("action", "")
            name = parsed.params.get("name", "")
            params = parsed.params

    if action == "list":
        procs = procedures.list_procedures(enabled_only=True)
        if not procs:
            return personality_say("proc_list_empty")
        lines = []
        for p in procs:
            used = f", used {p['use_count']}x" if p['use_count'] > 0 else ""
            lines.append(f"  '{p['trigger']}' — {len(p['steps'])} steps{used}")
        return f"You have {len(procs)} procedure(s):\n" + "\n".join(lines)

    elif action == "delete":
        if not name:
            return "Which procedure should I delete? Give me the name."
        proc = procedures.find_by_name_or_trigger(name)
        if not proc:
            return personality_say("proc_not_found", name=name)
        procedures.delete_procedure(proc["id"])
        return personality_say("proc_deleted", name=proc["trigger"])

    elif action == "rename":
        if not name:
            return "Which procedure should I rename?"
        new_trigger = params.get("new_trigger", "").strip()
        if not new_trigger:
            return "What should the new trigger be?"
        proc = procedures.find_by_name_or_trigger(name)
        if not proc:
            return personality_say("proc_not_found", name=name)
        conflict = procedures.check_trigger_conflict(new_trigger)
        if conflict and new_trigger.lower() != proc["trigger"].lower():
            return f"{conflict} Pick a different name."
        try:
            procedures.update_procedure(proc["id"], trigger=new_trigger)
        except ValueError as e:
            return f"Couldn't rename: {e}"
        return personality_say("proc_renamed", old=proc["trigger"], new=new_trigger)

    elif action == "edit":
        if not name:
            return "Which procedure should I edit?"
        proc = procedures.find_by_name_or_trigger(name)
        if not proc:
            return personality_say("proc_not_found", name=name)
        _act.teaching_session.set({
            "state":     "collecting",
            "name_seed": proc["name"],
            "steps":     [],
            "slots":     [],
            "backend":   proc.get("backend", "auto"),
            "_editing_proc_id": proc["id"],
            "_editing_trigger": proc["trigger"],
        })
        logger.info(f"[PROC-MGMT] Edit mode for id={proc['id']} trigger='{proc['trigger']}'")
        return personality_say("proc_edit_start", name=proc["trigger"])

    return (
        "I'm not sure what you want to do with procedures. "
        "Try 'list my procedures', 'delete procedure X', or 'edit procedure X'."
    )
