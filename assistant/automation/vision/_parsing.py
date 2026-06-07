"""Robust JSON parsing for LLM vision-plan responses.

Handles truncated output, code fences, and mid-string breaks.
"""

import json
import logging
import re

from ...core.json_utils import recover_truncated_json as _recover_truncated_json

logger = logging.getLogger("computer_agent")


def _parse_plan(raw: str) -> dict | None:
    """
    Parse the LLM's JSON action plan.

    Robust to:
      - Pure JSON (`{...}`)
      - Closed code fences
      - Truncated code fences with no closing fence
      - Mid-string / mid-array truncation (delegates to recoverer)

    Returns the parsed dict on success, None on irrecoverable failure.
    """
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None

    closed_fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", raw, re.DOTALL)
    if closed_fence:
        body = closed_fence.group(1).strip()
    else:
        open_fence = re.match(r"^```(?:json)?\s*", raw)
        body = raw[open_fence.end():].strip() if open_fence else raw

    try:
        if body.startswith("{"):
            return json.loads(body)
    except json.JSONDecodeError:
        pass

    start = body.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(body)):
            ch = body[i]
            if escape:
                escape = False
                continue
            if in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(body[start:i + 1])
                    except json.JSONDecodeError:
                        break

    try:
        recovered = _recover_truncated_json(body)
        if recovered != body:
            try:
                obj = json.loads(recovered)
                logger.warning(
                    f"[AGENT] Plan parse: recovered from truncation "
                    f"(orig len={len(body)}, recovered len={len(recovered)})"
                )
                return obj
            except json.JSONDecodeError:
                pass
    except Exception as rec_e:
        logger.debug(f"[AGENT] Truncation recovery raised: {rec_e}")

    logger.error(f"[AGENT] Plan parse error — body preview: {body[:200]!r}")
    return None
