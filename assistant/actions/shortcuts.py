"""Shortcut management handler: create, delete, list voice shortcuts."""

import logging

from .. import config
from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")


@tool_registry.decorator("manage_shortcut")
async def handle_manage_shortcut(params: dict, llm_response: str, bridge=None) -> str:
    """Handle shortcut creation, deletion, and listing via LLM parsing."""
    from .. import shortcuts
    from ..llm.contracts import ask_for_intent

    goal = params.get("goal", "").strip()
    import re
    goal = re.sub(rf"^{re.escape(config.ASSISTANT_NAME_LOWER)}\s*,?\s*", "",
                  goal, flags=re.IGNORECASE).strip()
    if not goal:
        return personality_say("error", error="I need to know what shortcut you want")

    parse_prompt = (
        "Parse this voice shortcut command. Extract the action, trigger word, "
        "and what it should do.\n\n"
        "IMPORTANT: Words like 'forgot', 'forget', 'remove', 'delete', 'drop', 'get rid of' "
        "mean action=delete. Do NOT interpret them as create.\n\n"
        f"User said: \"{goal}\"\n\n"
        "Respond ONLY with a JSON object:\n"
        '{\n'
        '  "action": "create" | "delete" | "list",\n'
        '  "trigger": "the trigger word/phrase (lowercase)",\n'
        '  "target_intent": "the intent to execute (e.g. planner, camera_look, code_executor, small_talk)",\n'
        '  "target_goal": "what the shortcut should do (natural language)",\n'
        '  "description": "short description of what this shortcut does"\n'
        '}\n\n'
        "Intent mapping guide:\n"
        "- Opening multiple apps/doing multiple things → planner\n"
        "- Opening camera/looking → camera_look\n"
        "- Playing music/controlling apps/system tasks → code_executor\n"
        "- Searching the web → web_search\n"
        "- Taking a note → create_note\n"
        "- Shutting down/closing → shutdown (special)\n"
        "- Simple response/greeting → small_talk\n\n"
        "For 'list' action, trigger/target fields can be empty.\n"
        "For 'delete' action, only trigger is needed."
    )

    try:
        raw = await ask_for_intent(
            parse_prompt,
            system_prompt="You are a JSON parser. Respond ONLY with valid JSON.",
            max_tokens=150,
            temperature=0,
        )

        import json
        cleaned = raw.strip().strip('`').strip("json").strip()
        data = json.loads(cleaned)

        action = data.get("action", "").lower()

        if action == "list":
            shortcut_list = shortcuts.list_shortcuts()
            if not shortcut_list:
                return "You don't have any shortcuts set up yet. Say something like 'when I say setup, open VS Code and Chrome' to create one!"
            lines = []
            for s in shortcut_list:
                used = f", used {s['times_used']}x" if s['times_used'] > 0 else ""
                lines.append(f"  '{s['trigger']}' → {s['description'] or s['intent']}{used}")
            return "Here are your shortcuts:\n" + "\n".join(lines)

        elif action == "delete":
            trigger = data.get("trigger", "").strip().lower()
            if not trigger:
                return "Which shortcut should I delete? Give me the trigger word."
            ok = shortcuts.delete_shortcut(trigger)
            if ok:
                return f"Done, I removed the '{trigger}' shortcut."
            else:
                return f"I don't have a shortcut called '{trigger}'."

        elif action == "create":
            trigger = data.get("trigger", "").strip().lower()
            target_intent = data.get("target_intent", "planner").strip()
            target_goal = data.get("target_goal", "").strip()
            description = data.get("description", "").strip()

            if not trigger:
                return "I need a trigger word. Like 'when I say SETUP, do this'."

            shortcut_params = {}
            if target_intent in ("planner", "code_executor", "file_task",
                                 "web_search", "computer_task", "set_reminder"):
                shortcut_params["goal"] = target_goal
            elif target_intent == "create_note":
                shortcut_params["title"] = target_goal
                shortcut_params["content"] = ""

            ok = shortcuts.create_shortcut(
                trigger=trigger,
                intent=target_intent,
                params=shortcut_params,
                description=description or target_goal,
            )

            if ok:
                desc = (description or target_goal).rstrip(".").lower()
                return f"Got it! When you say '{trigger}', I'll {desc}. Try it out!"
            else:
                return f"Hmm, I couldn't create that shortcut. The trigger might be too short or the action invalid."

        else:
            return "I'm not sure what you want to do with shortcuts. Try 'create a shortcut called X to do Y' or 'list my shortcuts'."

    except Exception as e:
        logger.error(f"[SHORTCUTS] Parse failed: {e}")
        return "I couldn't understand that shortcut command. Try something like 'when I say setup, open VS Code and Chrome'."
