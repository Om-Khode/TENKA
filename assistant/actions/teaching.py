"""Teachable Procedures — step parser, session state machine, batch teaching."""

import logging
import re

from .responses import personality_say

logger = logging.getLogger("actions")


# --- Step parser ---

def _normalize_key(key_text: str) -> str:
    """Normalize a spoken key description to a pyautogui-compatible string."""
    k = key_text.strip().lower()
    k = k.replace("space bar", "space").replace("spacebar", "space")
    k = k.replace("page up", "pageup").replace("page down", "pagedown")
    k = k.replace("num lock", "numlock").replace("caps lock", "capslock")
    k = k.replace("return", "enter").replace("escape", "esc")
    k = k.replace("control", "ctrl")
    k = re.sub(r'\b(ctrl|shift|alt|win)\s+(\S+)', r'\1+\2', k)
    return k.strip()


def _is_url_like(text: str) -> bool:
    t = text.strip().lower()
    return (
        t.startswith("http://") or t.startswith("https://") or
        t.startswith("www.") or
        bool(re.search(r'\.\w{2,4}(/|$|\s)', t))
    )


_POSITIONAL_MAP = {
    "first": 1, "1st": 1, "top": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
}


def _positional_click_steps(ordinal: str) -> list[dict]:
    """Convert 'click first result' into keyboard nav: N x down + enter."""
    n = _POSITIONAL_MAP.get(ordinal, 1)
    steps: list[dict] = []
    for _ in range(n):
        steps.append({"type": "app", "action": "press_key", "params": {"key": "down"}})
    steps.append({"type": "app", "action": "press_key", "params": {"key": "enter"}})
    return steps


_NL_VAR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r'^(?:whatever|what(?:ever)?)\s+(?:I|i|the user)\s+'
        r'(?:say|said|ask|asked|asked for|want|wanted|mention|mentioned|type|typed|search|searched).*$',
        re.IGNORECASE), "{user_input}"),
    (re.compile(
        r'^(?:the\s+)?rest\s+of\s+(?:what\s+(?:I|i)\s+(?:say|said)|my\s+(?:input|command|message))$',
        re.IGNORECASE), "{user_input}"),
    (re.compile(
        r'^(?:my|the|user)\s*(?:input|query|search(?:\s+term)?|request|message)$',
        re.IGNORECASE), "{user_input}"),
    (re.compile(
        r"^(?:today'?s?\s+)?date$|^(?:the\s+)?current\s+date$|^(?:the\s+)?date\s+today$",
        re.IGNORECASE), "{date}"),
    (re.compile(
        r'^(?:the\s+)?(?:current\s+)?time$|^what\s+time\s+it\s+is$',
        re.IGNORECASE), "{time}"),
    (re.compile(
        r'^(?:(?:what(?:\'?s)?\s+(?:in\s+)?)?(?:my\s+|the\s+)?clipboard(?:\s+content(?:s)?)?)$|'
        r'^what\s+(?:I|i)\s+copied$|^(?:the\s+)?copied\s+text$',
        re.IGNORECASE), "{clipboard}"),
]


def _detect_nl_variable(text: str) -> str:
    """Map natural-language phrases to {variable} tokens during teaching."""
    t = text.strip().rstrip(".!?,;")
    for pat, var in _NL_VAR_PATTERNS:
        if pat.match(t):
            return var
    return text


_VAR_DESCRIPTIONS: dict[str, str] = {
    "{user_input}": "whatever you say after the trigger",
    "{date}":       "today's date",
    "{time}":       "the current time",
    "{clipboard}":  "clipboard contents",
}


_STEP_PATTERNS = [
    (re.compile(
        r'^(?:go to|navigate to|visit)\s+(https?://\S+|www\.\S+|\S+\.\w{2,4}(?:/\S*)?)$',
        re.IGNORECASE),
     lambda m: {"type": "browser", "action": "navigate", "params": {
         "url": m.group(1) if m.group(1).startswith("http") else f"https://{m.group(1)}"
     }}),

    (re.compile(r'^open\s+(.+)$', re.IGNORECASE),
     lambda m: (
         {"type": "browser", "action": "navigate", "params": {
             "url": m.group(1).strip() if m.group(1).strip().startswith("http")
                    else f"https://{m.group(1).strip()}"
         }} if _is_url_like(m.group(1))
         else {"type": "app", "action": "open", "params": {"name": m.group(1).strip()}}
     )),

    (re.compile(r'^close\s+(.+)$', re.IGNORECASE),
     lambda m: {"type": "app", "action": "close", "params": {"name": m.group(1).strip()}}),

    (re.compile(r'^focus\s+(?:on\s+)?(.+)$', re.IGNORECASE),
     lambda m: {"type": "app", "action": "focus", "params": {"name": m.group(1).strip()}}),

    (re.compile(
        r'^(?:press|hit)\s+(.+?)\s+'
        r'(?:(\d+)\s*times?|x(\d+)|(\d+)x)$',
        re.IGNORECASE),
     lambda m: [
         {"type": "app", "action": "press_key",
          "params": {"key": _normalize_key(m.group(1))}}
         for _ in range(int(m.group(2) or m.group(3) or m.group(4)))
     ]),

    (re.compile(r'^(?:press|hit)\s+(.+)$', re.IGNORECASE),
     lambda m: {"type": "app", "action": "press_key",
                "params": {"key": _normalize_key(m.group(1))}}),

    (re.compile(r'^type\s+["\']?(.+?)["\']?(?:\s+in(?:to)?\s+(.+))?$', re.IGNORECASE),
     lambda m: {"type": "app", "action": "type", "params": {
         "text": _detect_nl_variable(m.group(1).strip().strip("'\"")),
         **( {"window": m.group(2).strip()} if m.group(2) else {} )
     }}),

    (re.compile(r'^paste(?:\s+(?:from\s+)?(?:the\s+)?clipboard)?$', re.IGNORECASE),
     lambda _: {"type": "app", "action": "press_key", "params": {"key": "ctrl+v"}}),

    (re.compile(
        r'^click\s+(?:on\s+)?(?:the\s+)?'
        r'(first|second|third|top|1st|2nd|3rd)\s+'
        r'(?:result|item|option|entry|match|contact|chat)$',
        re.IGNORECASE),
     lambda m: _positional_click_steps(m.group(1).lower())),

    (re.compile(r'^click\s+(?:on\s+)?(.+)$', re.IGNORECASE),
     lambda m: {"type": "app", "action": "click",
                "params": {"selector": f"name:{m.group(1).strip()}"}}),

    (re.compile(
        r'^wait\s+(?:for\s+)?'
        r'(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)'
        r'\s*(?:seconds?|secs?)?$',
        re.IGNORECASE),
     lambda m: {"type": "app", "action": "wait", "params": {
         "seconds": float({
             "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         }.get(m.group(1).lower(), m.group(1)))
     }}),

    (re.compile(r'^wait(?:\s+(?:for\s+)?a\s+(?:moment|second|bit|few\s+seconds?))?$',
                re.IGNORECASE),
     lambda _: {"type": "app", "action": "wait", "params": {"seconds": 2}}),
]


_STEP_NUMBER_RE = re.compile(r'^\s*(?:\d+[\.\)\-:]\s*|[-*]\s+)')
_INLINE_SPLIT_RE = re.compile(r'\s{2,}(?=\d+[\.\)\-:])|(?<=\S)\s+(?=\d+[\.\)\-:]\s)')


def _split_pasted_steps(text: str) -> list[str]:
    if "\n" in text:
        return [l.strip() for l in text.splitlines() if l.strip()]
    parts = _INLINE_SPLIT_RE.split(text)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    return [text]


def _parse_teaching_step(text: str) -> dict | list[dict] | None:
    text = _STEP_NUMBER_RE.sub("", text).strip()
    if not text:
        return None
    for pattern, builder in _STEP_PATTERNS:
        m = pattern.match(text)
        if m:
            return builder(m)
    return None


def _step_description(step: dict) -> str:
    """One-line human-readable description of a step for TTS readback."""
    t = step.get("type", "")
    a = step.get("action", "")
    p = step.get("params", {})

    if t == "app":
        if a == "open":
            return f"open {p.get('name', '?')}"
        if a == "close":
            return f"close {p.get('name', '?')}"
        if a == "press_key":
            return f"press {p.get('key', '?')}"
        if a == "type":
            txt    = p.get("text", "?")
            win    = p.get("window")
            suffix = f" in {win}" if win else ""
            desc   = _VAR_DESCRIPTIONS.get(txt, f"'{txt}'")
            return f"type {desc}{suffix}"
        if a == "click":
            sel = p.get("selector", "?").replace("name:", "")
            return f"click {sel}"
        if a == "wait":
            return f"wait {p.get('seconds', 1)} seconds"
        if a == "focus":
            return f"focus on {p.get('name', '?')}"
    elif t == "browser":
        if a == "navigate":
            return f"go to {p.get('url', '?')}"
    elif t == "tool":
        return f"{step.get('intent', '?')}: {step.get('goal', '')}"
    return str(step)


# --- Teaching session state machine ---

_SLOT_RE = re.compile(r'\{(\w+)\}')
_BUILTIN_VARS = frozenset({"user_input", "date", "time", "clipboard"})

_LITERAL_SKIP = frozenset([
    "a", "an", "the", "this", "that", "it", "its", "my", "your",
    "here", "there", "now", "then", "all", "none", "yes", "no",
])


def _find_suspect_literals(steps: list[dict]) -> list[tuple[int, str]]:
    """Find type steps with short literal words that might be dynamic slots."""
    suspects: list[tuple[int, str]] = []
    for i, step in enumerate(steps):
        if step.get("action") != "type":
            continue
        val = step.get("params", {}).get("text", "")
        if (
            val
            and not val.startswith("{")
            and len(val) <= 15
            and " " not in val
            and val.lower() not in _LITERAL_SKIP
        ):
            suspects.append((i, val))
    return suspects


def _extract_slots_from_steps(steps: list[dict]) -> list[str]:
    """Return named (non-builtin) {slot} names found across all step params."""
    import json as _json
    raw  = _json.dumps(steps)
    seen: set[str] = set()
    result: list[str] = []
    for name in _SLOT_RE.findall(raw):
        if name not in _BUILTIN_VARS and name not in seen:
            seen.add(name)
            result.append(name)
    return result


_TEACH_DONE = (
    "done", "that's it", "thats it", "that's all", "thats all",
    "finish", "finished", "save it", "save", "that'll do", "that will do",
    "all done", "i'm done", "im done", "you're done", "youre done",
    "we're done", "were done", "end", "stop",
)

_TEACH_YES = (
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
    "correct", "right", "perfect", "good",
)

_TEACH_NO = (
    "no", "nope", "nah", "wrong", "incorrect",
)


def start_teaching_session(name_seed: str) -> str:
    import assistant.actions as _act
    _act.teaching_session.set({
        "state":     "collecting",
        "name_seed": name_seed.strip(),
        "steps":     [],
        "slots":     [],
        "backend":   "auto",
    })
    logger.info(f"[TEACH] Session started: name_seed='{name_seed}'")
    return (
        f"Alright, teach me how to {name_seed}. "
        "Prefer keyboard shortcuts over mouse clicks — like 'press ctrl+f' instead of 'click on search'. "
        "For values that change each time, use curly brackets — like 'type {contact}'. "
        "What's the first step?"
    )


def _enter_confirming(session: dict) -> str:
    """Build the step-readback confirmation message and transition state."""
    steps = session["steps"]
    session["state"] = "confirming"
    step_lines = "\n".join(
        f"Step {i + 1}: {_step_description(s)}"
        for i, s in enumerate(steps)
    )
    slots = _extract_slots_from_steps(steps)
    session["slots"] = slots
    slot_note = ""
    if slots:
        slot_note = (
            f"\nDynamic values: {', '.join(slots)}. "
            "Fill these in when you run the procedure."
        )
    return personality_say("teach_confirm", steps=step_lines) + slot_note


async def handle_pending_teaching(text: str) -> str | None:
    import assistant.actions as _act
    if not _act.teaching_session.active:
        return None

    from .. import procedures as _ps

    state   = _act.teaching_session.payload["state"]
    lowered = text.strip().lower().rstrip(".,!?")

    # --- collecting: accumulate steps ---
    if state == "collecting":
        if any(
            lowered == p or lowered.startswith(p + " ") or lowered.endswith(" " + p)
            for p in _TEACH_DONE
        ):
            steps = _act.teaching_session.payload["steps"]
            if not steps:
                return personality_say("teach_need_step")

            suspects = _find_suspect_literals(steps)
            if suspects:
                _act.teaching_session.payload["state"] = "slot_confirm"
                _act.teaching_session.payload["_suspects"] = suspects
                _act.teaching_session.payload["_suspect_idx"] = 0
                idx, val = suspects[0]
                return (
                    f"Quick check — in step {idx + 1}, you said type '{val}'. "
                    f"Does '{val}' change each time, like a name or a search term? Yes or no."
                )

            return _enter_confirming(_act.teaching_session.payload)

        if len(_act.teaching_session.payload["steps"]) >= _ps._MAX_STEPS:
            return personality_say("teach_max_steps")

        lines = _split_pasted_steps(text.strip())
        if len(lines) > 1:
            added = []
            bad = []
            for line in lines:
                if len(_act.teaching_session.payload["steps"]) >= _ps._MAX_STEPS:
                    break
                s = _parse_teaching_step(line)
                if s is not None:
                    expanded = s if isinstance(s, list) else [s]
                    _act.teaching_session.payload["steps"].extend(expanded)
                    added.extend(_step_description(e) for e in expanded)
                else:
                    bad.append(line)
            if not added:
                return personality_say("teach_cant_parse", text=text.strip())
            resp = f"Got {len(added)} steps."
            if bad:
                resp += f" Skipped {len(bad)} I couldn't parse: {', '.join(bad[:3])}."
            resp += " Say more steps or 'done' to finish."
            return resp

        step = _parse_teaching_step(text.strip())
        if step is None:
            return personality_say("teach_cant_parse", text=text.strip())

        expanded = step if isinstance(step, list) else [step]
        _act.teaching_session.payload["steps"].extend(expanded)
        desc = ", ".join(_step_description(s) for s in expanded)

        expansion_note = ""
        if isinstance(step, list) and len(step) > 1:
            expansion_note = (
                f" I'll use keyboard navigation for that — "
                f"{len(step)} key presses instead of a mouse click, much more reliable."
            )

        warn = _ps.step_count_warning(_act.teaching_session.payload["steps"])
        if warn:
            return personality_say("teach_got_it_warn", desc=desc, warn=warn) + expansion_note
        return personality_say("teach_got_it", desc=desc) + expansion_note

    # --- slot_confirm: ask about each suspect literal ---
    elif state == "slot_confirm":
        is_yes = any(w == lowered or lowered.startswith(w + " ") for w in _TEACH_YES)
        is_no = any(
            w == lowered or lowered.startswith(w + " ") or lowered.endswith(" " + w)
            for w in _TEACH_NO
        )

        suspects = _act.teaching_session.payload["_suspects"]
        si = _act.teaching_session.payload["_suspect_idx"]
        step_idx, val = suspects[si]

        if is_yes:
            self_steps = _act.teaching_session.payload["steps"]
            self_steps[step_idx]["params"]["text"] = f"{{{val}}}"
        elif not is_no:
            return f"Is '{val}' a value that changes? Just yes or no."

        si += 1
        _act.teaching_session.payload["_suspect_idx"] = si
        if si < len(suspects):
            next_idx, next_val = suspects[si]
            return (
                f"What about step {next_idx + 1} — does '{next_val}' change each time? "
                "Yes or no."
            )

        _act.teaching_session.payload.pop("_suspects", None)
        _act.teaching_session.payload.pop("_suspect_idx", None)
        return _enter_confirming(_act.teaching_session.payload)

    # --- confirming: read-back + yes/no ---
    elif state == "confirming":
        is_yes = (
            any(w == lowered or lowered.startswith(w + " ") for w in _TEACH_YES)
            or any(p in lowered for p in (
                "that's right", "looks good", "that's correct",
                "sounds right", "sounds good",
            ))
        )
        is_no = (
            any(w == lowered or lowered.startswith(w + " ") or lowered.endswith(" " + w)
                for w in _TEACH_NO)
            or any(p in lowered for p in (
                "not right", "start over", "restart", "redo", "that's wrong",
            ))
        )

        if is_yes:
            editing_id = _act.teaching_session.payload.get("_editing_proc_id")
            if editing_id is not None:
                steps = _act.teaching_session.payload["steps"]
                trigger = _act.teaching_session.payload["_editing_trigger"]
                slots = _extract_slots_from_steps(steps)
                _ps.update_procedure(editing_id, steps=steps)
                n = len(steps)
                _act.teaching_session.clear()
                logger.info(f"[TEACH] Updated id={editing_id} steps={n}")
                reply = personality_say("teach_saved", trigger=trigger, n=n)
                if slots:
                    reply += f" When you run it, include: {', '.join(slots)}."
                return reply
            _act.teaching_session.payload["state"] = "naming"
            return personality_say("teach_ask_trigger", name_seed=_act.teaching_session.payload["name_seed"])

        if is_no:
            _act.teaching_session.payload["steps"] = []
            _act.teaching_session.payload["state"] = "collecting"
            return personality_say("teach_restart")

        return personality_say("teach_yes_or_no")

    # --- naming: user says the trigger phrase ---
    elif state == "naming":
        name_seed = _act.teaching_session.payload["name_seed"]
        is_yes    = any(w == lowered or lowered.startswith(w + " ") for w in _TEACH_YES)
        trigger   = name_seed if is_yes else text.strip().rstrip(".!?").strip()

        if len(trigger) < 3:
            return "That trigger is too short. Say a phrase like 'start my coding session'."

        conflict = _ps.check_trigger_conflict(trigger)
        if conflict:
            return f"{conflict} Try a different phrase."

        try:
            steps   = _act.teaching_session.payload["steps"]
            name    = name_seed.title()
            proc_id = _ps.create_procedure(
                trigger=trigger,
                name=name,
                steps=steps,
                backend=_act.teaching_session.payload.get("backend", "auto"),
            )
            n     = len(steps)
            slots = _act.teaching_session.payload.get("slots", [])
            _act.teaching_session.clear()
            logger.info(f"[TEACH] Saved id={proc_id} trigger='{trigger}' steps={n} slots={slots}")
            reply = personality_say("teach_saved", trigger=trigger, n=n)
            if slots:
                reply += f" When you run it, include: {', '.join(slots)}."
            return reply
        except ValueError as e:
            return f"Couldn't save that: {e}. Try a different trigger phrase."

    _act.teaching_session.clear()
    return None


# --- Batch teaching (paste a full plan) ---

_BATCH_LINE_RE = re.compile(
    r'^\s*(?:\d+[\.\)\-:]\s*|[-*]\s+)?(.+)$'
)


def start_batch_teaching(name_seed: str, body: str) -> str:
    import assistant.actions as _act

    lines = [
        m.group(1).strip()
        for raw in body.strip().splitlines()
        if (m := _BATCH_LINE_RE.match(raw))
    ]

    if not lines:
        return "I couldn't find any steps in that. Each line should be a step like 'open notepad' or 'press ctrl+s'."

    steps: list[dict] = []
    bad: list[str] = []
    for line in lines:
        step = _parse_teaching_step(line)
        if step is not None:
            if isinstance(step, list):
                steps.extend(step)
            else:
                steps.append(step)
        else:
            bad.append(line)

    if not steps:
        return "None of those lines looked like steps I can run. Try 'open X', 'press Y', 'type Z', etc."

    _act.teaching_session.set({
        "state":     "collecting",
        "name_seed": name_seed.strip(),
        "steps":     steps,
        "slots":     [],
        "backend":   "auto",
    })
    logger.info(f"[TEACH] Batch parsed {len(steps)} steps from {len(lines)} lines")

    warn = ""
    if bad:
        warn = f" I skipped {len(bad)} line(s) I couldn't parse: {', '.join(bad[:3])}."

    suspects = _find_suspect_literals(steps)
    if suspects:
        _act.teaching_session.payload["state"] = "slot_confirm"
        _act.teaching_session.payload["_suspects"] = suspects
        _act.teaching_session.payload["_suspect_idx"] = 0
        idx, val = suspects[0]
        return (
            f"Got {len(steps)} steps.{warn} "
            f"Quick check — in step {idx + 1}, you said type '{val}'. "
            f"Does '{val}' change each time? Yes or no."
        )

    return _enter_confirming(_act.teaching_session.payload) + warn
