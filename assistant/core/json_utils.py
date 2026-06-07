"""Robust JSON extraction from LLM output.

Handles code fences, think tags, unicode quote/dash mangling, trailing
commas, and truncated output. Pure utility — no project-internal imports.
"""

import json
import re

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_UNICODE_MAP = str.maketrans({
    "‐": "-", "‑": "-", "–": "-", "—": "-",
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
})


def sanitize_json(text: str) -> str:
    """Strip code fences, think tags, fix unicode quotes/dashes, remove trailing commas."""
    if not text:
        return text

    text = _THINK_TAG_RE.sub("", text).strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
        text = text.strip()

    text = text.translate(_UNICODE_MAP)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def recover_truncated_json(text: str) -> str:
    """
    Repair JSON truncated mid-stream: close unterminated strings,
    strip trailing commas, and balance unclosed braces/brackets.
    """
    if not isinstance(text, str) or not text:
        return text

    in_string = False
    escape = False
    stack: list[str] = []

    for ch in text:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if not in_string and not stack:
        return text

    repaired = text
    if in_string:
        if escape:
            repaired = repaired.rstrip("\\")
        repaired += '"'

    stripped = repaired.rstrip()
    if stripped.endswith(","):
        repaired = stripped[:-1]

    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    return repaired


def extract_json_object(
    text: str, *, sanitize: bool = False, repair: bool = False,
) -> str | None:
    """
    Extract the first JSON object ``{...}`` from *text*.

    Args:
        sanitize: pre-process with :func:`sanitize_json` (fences, unicode, etc.)
        repair:   attempt :func:`recover_truncated_json` when normal extraction fails

    Returns the JSON substring, or ``None`` if nothing extractable.
    """
    if not text:
        return None

    text = text.strip()
    if sanitize:
        text = sanitize_json(text)

    if text.startswith("{"):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            json.loads(fence.group(1))
            return fence.group(1)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    break

    if repair:
        recovered = recover_truncated_json(text[start:])
        try:
            json.loads(recovered)
            return recovered
        except json.JSONDecodeError:
            pass

    return None


def extract_json_array(text: str, *, sanitize: bool = False) -> list:
    """
    Extract the first JSON array ``[...]`` from *text*.

    Prefers ``[{`` over bare ``[`` to avoid matching personality tags
    like ``[sarcastic]``.

    Args:
        sanitize: pre-process with :func:`sanitize_json` (fences, think tags, etc.)

    Returns the parsed list, or ``[]`` if nothing extractable.
    """
    if not text:
        return []

    text = text.strip()
    if sanitize:
        text = sanitize_json(text)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        try:
            result = json.loads(fence.group(1))
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    obj_start = text.find("[{")
    if obj_start != -1:
        end = text.rfind("]")
        if end > obj_start:
            candidate = text[obj_start : end + 1]
            for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
                try:
                    result = json.loads(attempt)
                    if isinstance(result, list):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                result = json.loads(attempt)
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

    return []
