"""
dom_mapper.py — LLM goal-to-field mapper (Option B).

Single LLM call: given a user's goal and a list of form fields, produce a
flat mapping of {ref → value} plus the submit button ref. No interaction
strategy — the filler handles HOW to fill each widget type.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Union

from . import dom as browser_dom
from ...llm.contracts import ask_for_plan

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────


@dataclass(frozen=True)
class FillInstruction:
    """One field fill: put `value` into the element at `ref`."""
    ref: str
    field_name: str
    value: Union[str, list[str]]  # list for multi-value comboboxes


@dataclass
class FormMapping:
    """LLM output: which values go into which fields, plus submit target."""
    fills: list[FillInstruction]
    submit_ref: str        # ref of the submit button, "" if none
    thinking: str          # ≤2 sentences, for debug logs
    skip_submit: bool = False  # True when goal doesn't require submission


# ─── Mapper System Prompt ───────────────────────────────────────────────

MAPPER_SYSTEM_PROMPT = """\
You are a form field mapper. You receive a user's goal and a list of \
interactive form elements. Your job: decide which VALUE goes into which \
FIELD. You do NOT decide HOW to interact — that's the executor's job.

OUTPUT — STRICT JSON, NO MARKDOWN:
{
  "thinking": "≤2 sentences — which fields you'll fill and why",
  "fills": [
    {"ref": "<ref>", "value": "<string or list of strings>"},
    ...
  ],
  "submit_ref": "<ref of submit button, or empty string if no submit>",
  "skip_submit": false
}

RULES:
1. Every `ref` you emit MUST exist in the field list. Never invent refs.
2. For single-value fields (text, email, select, radio): value is a string.
3. For multi-value fields (tagged combobox accepting multiple selections): \
value is a JSON array of strings, e.g. ["Maths", "English"].
4. For radio groups: emit one fill whose `ref` is the radio button matching \
the desired choice. Value should be the radio's name/label.
5. For checkboxes: value is "check" or "uncheck".
6. For native selects (fields marked `select` with listed options): value \
is the EXACT text of the desired option from the options list.
7. Skip fields that already have the correct value (check `current_value`).
8. Set `skip_submit: true` only when the goal explicitly says not to submit.
9. If the goal mentions test data without specifics, use safe defaults: \
First name "Test", Last name "User", Email "test@example.com", \
Phone "1234567890".
10. For dropdowns: never pick placeholder rows ("-- Select --", "Choose...").
"""


# ─── Prompt Builder ─────────────────────────────────────────────────────


def _field_summary(e: browser_dom.ElementInfo) -> dict:
    """Compact field description for the mapper prompt."""
    d: dict[str, Any] = {
        "ref": e.ref,
        "role": e.role,
        "name": e.name,
    }
    if e.value:
        d["current_value"] = e.value
    if e.options:
        d["options"] = list(e.options)
    if e.type and e.type not in ("text", "submit"):
        d["type"] = e.type
    if e.placeholder and e.placeholder != e.name:
        d["placeholder"] = e.placeholder
    if e.role == "combobox" and e.autocomplete:
        d["widget"] = "autocomplete_combobox"
    elif e.role == "combobox" and not e.options:
        d["widget"] = "custom_combobox"
    if not e.visible:
        d["visible"] = False
    if not e.enabled:
        d["enabled"] = False
    return d


def build_mapper_prompt(
    goal: str,
    elements: list[browser_dom.ElementInfo],
) -> str:
    """Build the user-message prompt for the mapper LLM call."""
    fields = [_field_summary(e) for e in elements]
    payload = {
        "goal": goal,
        "fields": fields,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


# ─── Response Parser ────────────────────────────────────────────────────


def parse_mapper_response(
    raw: str,
    valid_refs: set[str],
    *,
    ref_to_name: dict[str, str] | None = None,
) -> FormMapping:
    """Parse the LLM's JSON response into a validated FormMapping."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[DOM_MAPPER] LLM returned invalid JSON")
        return FormMapping(fills=[], submit_ref="", thinking="parse_error")

    if not isinstance(data, dict):
        logger.warning("[DOM_MAPPER] LLM response is not a JSON object")
        return FormMapping(fills=[], submit_ref="", thinking="parse_error")

    thinking = str(data.get("thinking", ""))[:200]
    skip_submit = bool(data.get("skip_submit", False))

    # Validate submit ref
    submit_ref = str(data.get("submit_ref", ""))
    if submit_ref and submit_ref not in valid_refs:
        logger.warning(f"[DOM_MAPPER] submit_ref {submit_ref!r} not in valid refs — dropping")
        submit_ref = ""

    # Validate fills
    fills: list[FillInstruction] = []
    name_map = ref_to_name or {}
    for entry in data.get("fills", []):
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("ref", ""))
        if ref not in valid_refs:
            logger.info(f"[DOM_MAPPER] dropping fill with unknown ref {ref!r}")
            continue
        value = entry.get("value", "")
        if isinstance(value, list):
            value = [str(v) for v in value]
        else:
            value = str(value)
        fills.append(FillInstruction(
            ref=ref,
            field_name=name_map.get(ref, ""),
            value=value,
        ))

    return FormMapping(
        fills=fills,
        submit_ref=submit_ref,
        thinking=thinking,
        skip_submit=skip_submit,
    )


# ─── LLM Call ───────────────────────────────────────────────────────────


async def map_goal_to_fields(
    goal: str,
    tree: browser_dom.PageDomTree,
    *,
    feedback: str = "",
) -> FormMapping:
    """Single LLM call: map goal text to form field fills."""
    elements = [e for e in tree.elements if e.visible and e.enabled]
    prompt = build_mapper_prompt(goal, elements)
    if feedback:
        prompt += f"\n\n{feedback}"

    logger.info(f"[DOM_MAPPER] mapping goal to {len(elements)} fields")

    raw = await ask_for_plan(
        prompt,
        system_prompt=MAPPER_SYSTEM_PROMPT,
        json_mode=True,
        max_tokens=1024,
    )

    valid_refs = {e.ref for e in tree.elements}
    ref_to_name = {e.ref: e.name for e in tree.elements}
    result = parse_mapper_response(raw, valid_refs, ref_to_name=ref_to_name)

    logger.info(
        f"[DOM_MAPPER] mapped {len(result.fills)} fills, "
        f"submit_ref={result.submit_ref!r}, thinking={result.thinking!r}"
    )
    return result
