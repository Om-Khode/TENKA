"""
browser_dom_planner.py — DOM-aware planner.

Reads a `PageDomTree` (from browser_dom.read_page_dom), asks an LLM to
produce a batched action plan in DOM-ref form, validates the plan against
the tree, and returns it as a `DomPlan`.

Architectural contract (from project_phase1_full_design.md §4):
  - Planner emits ALL form actions in ONE batch. Single LLM call per
    iteration. No per-field replanning.
  - Output uses ref IDs from the tree, never fabricated coordinates.
  - For native <select>, planner emits select_option_ref with EXACT option
    text — no Down-arrow guessing.
  - For custom React comboboxes (role=combobox with empty options), planner
    emits click_ref + needs_reperceive. The orchestrator opens the combo,
    re-reads the tree, and a follow-up batch picks the option.
  - The planner DOES NOT share TODO machinery with the vision-loop —
    completeness is intrinsic to the action list.

Validation guarantees the executor (1C-b) cannot receive:
  - Refs that don't exist in the tree
  - Actions targeting invisible / disabled elements
  - Actions of unknown types
  - Malformed structures (missing required keys)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ... import config

logger = logging.getLogger("browser_dom_planner")


# ─── Result types ─────────────────────────────────────────────────────────────


@dataclass
class DomPlan:
    """
    Validated planner output. Always safe to pass directly to the executor.

    `actions` only contains action dicts whose `ref` (when present) was in
    the tree's ref_to_locator map AND whose target was visible+enabled.
    Rejected actions are summarised in `rejection_notes` for the orchestrator
    to decide whether to feed back into the next planner iteration.
    """
    thinking: str
    plan: str
    actions: list[dict]
    done: bool
    needs_reperceive: bool
    rejection_notes: list[str] = field(default_factory=list)
    raw_response: str = ""


# ─── System prompt ────────────────────────────────────────────────────────────

DOM_PLANNER_SYSTEM_PROMPT = """\
You are a web automation planner. You receive a goal and a list of interactive \
elements on the current page (the DOM tree), each with a stable `ref` ID and \
structural metadata. Your job is to produce ONE batch of actions that completes \
the goal — fills, clicks, submits — in DOM order, then declares done.

ELEMENTS YOU RECEIVE
Each element carries:
  ref         — stable string ID (use this in your actions; never invent refs)
  role        — ARIA role: textbox, button, link, checkbox, radio, combobox, ...
  name        — accessible name (label / aria-label / placeholder fallback)
  value       — current value (empty string for unfilled inputs)
  options     — for native <select>: the available option strings
  placeholder — extra hint, present only when distinct from name
  type        — for inputs: "email", "tel", "password", "date", "url", ...
  autocomplete — for custom comboboxes: "list", "both", or absent. Present = type-to-filter.
  visible     — false = off-screen / display:none / opacity 0 — DO NOT act on these
  enabled     — false = disabled / aria-disabled — DO NOT act on these

OUTPUT — STRICT JSON, NO MARKDOWN
{
  "thinking": "≤2 short sentences. Name every field you'll fill and the submit ref.",
  "plan":     "one-line summary",
  "actions":  [<action>, <action>, ...],
  "done":     true | false,
  "needs_reperceive": false | true
}

ACTION TYPES — use EXACTLY these shapes:

  {"type":"form_input",        "ref":"<ref>", "value":"<string>"}
  {"type":"click_ref",         "ref":"<ref>"}
  {"type":"select_option_ref", "ref":"<ref>", "option":"<exact option text>"}
  {"type":"press_ref",         "ref":"<ref>", "key":"Enter"}
  {"type":"wait_ms",           "ms":250}
  {"type":"reperceive"}

RULES (load-bearing)

1. Use `form_input` for text fields. Do NOT click_ref a textbox first — fill()
   handles focus and clears existing value.

2. For native `<select>` (role=combobox WITH a non-empty `options` array):
   use `select_option_ref` with EXACT text from the element's `options`. Do
   NOT guess. Do NOT use arrow keys.

3. Custom comboboxes (role=combobox with EMPTY `options`):

   3a. Autocomplete combobox (element has `autocomplete` field like "list"
       or "both"): use `form_input` to type a short distinctive prefix of
       the desired value into the combobox (e.g. "Math" for "Maths", "NCR"
       for "NCR"), then emit `{"type":"reperceive"}` and STOP the batch.
       The typing filters options — the next iteration shows matching
       role="option" elements.

   3b. Non-autocomplete custom combobox (no `autocomplete` field): emit
       `click_ref` to open the dropdown, then `{"type":"reperceive"}` and
       STOP the batch. The next iteration sees the dropdown children.

   3c. After reperceive — selecting from opened dropdown/autocomplete:
       when role="option" elements appear in the tree, use `click_ref` on
       the one whose `name` matches your desired value. Do NOT use
       `select_option_ref` on role="option" — that action only works on
       native `<select>`. After clicking the option, continue with
       remaining fields or submit.

4. For role=button (submit, link, action): use `click_ref`. Identify the
   correct form-submit button by its NAME — for forms the user wants
   submitted, look for name like "Schedule a Demo", "Submit", "Send",
   "Continue", "Sign in", "Register", "Subscribe".

5. NEVER fabricate refs. Every `ref` you emit MUST appear in the elements
   list. The dispatcher will reject unknown refs.

6. NEVER act on elements where `visible: false` or `enabled: false`. Skip them.

7. Set `done: true` when the action sequence completes the goal (typically
   after the submit click). The orchestrator will verify the resulting state.

8. Set `needs_reperceive: true` ONLY when you opened a combobox / expect a
   new form section to appear / clicked something likely to navigate. The
   next iteration will get a fresh tree.

9. Don't emit screenshot actions — DOM mode does not consume vision per step.

10. Don't ramble in `thinking`. ≤2 sentences. State which fields you'll fill,
    which submit you'll click. That's it. Verbose thinking is wasted tokens.

MULTI-FORM PAGES
Many sites have multiple forms (header CTA + footer + modal). Pick ONE form
based on goal context:
  - "fill the demo form" / "book a demo" → form whose submit button name
    contains "demo"
  - "contact us" / "get in touch" → form whose submit button name contains
    "contact" / "send"
  - "the modal" → prefer elements with non-empty value (focus signal) or
    that share the same form ancestor pattern
When unsure, pick the form with the FIRST visible+enabled submit button.

VALUE FORMATTING
- Email: produce a valid syntax like "test@example.com".
- Phone: digits only unless placeholder shows formatting.
- For "testing values" / "test data" goals, use safe defaults:
    First name "Test", Last name "User", Company "Test Co",
    Email "test@example.com", Phone "1234567890".
- For dropdowns: pick a plausible non-default option (skip "-- select --"
  / "Choose..." / "Industry" / "Staff Size" placeholder rows).

If `options` includes a placeholder ("Industry", "Staff Size", "-- pick --"),
NEVER select it — it's the dropdown's default-empty state.

11. Multi-select comboboxes: if a combobox accepts multiple values (tags,
    subjects, etc.), handle one value per loop. Type one value via
    form_input → reperceive → click_ref on the matching option. The next
    loop handles the next value. Do NOT try to select multiple values in
    one batch.

VALIDATION FEEDBACK
After a submit, FEEDBACK FROM PREVIOUS ITERATION may begin with
`VALIDATION ERRORS FROM PRIOR SUBMIT:` followed by per-field error lines
like `- "Email" (ref=ref0001): Please enter a valid email address.` This
means the form was rejected — produce a corrective fill batch:
  - Issue a `form_input` (or `select_option_ref` / `click_ref` for non-text
    widgets) for each named field, using a value that satisfies the error
    message ("valid email" → switch to a real-looking address).
  - Re-click the same submit button at the end. Set `done: true`.
  - Do NOT refill fields the feedback didn't flag — they were accepted.
If the feedback's only entry is `page-level: ...` and you can't infer the
target field, set `done: true` with no actions and explain in `thinking`.
"""


# ─── Plan parsing + validation ────────────────────────────────────────────────

# Action types accepted by the executor. Kept here as the single source of
# truth so future executor extensions add to one place. Action shapes
# (required keys per type) live in _ACTION_REQUIRED_KEYS below.
_VALID_ACTION_TYPES = frozenset({
    "form_input",
    "click_ref",
    "select_option_ref",
    "press_ref",
    "wait_ms",
    "reperceive",
})

_ACTION_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "form_input": frozenset({"ref", "value"}),
    "click_ref": frozenset({"ref"}),
    "select_option_ref": frozenset({"ref", "option"}),
    "press_ref": frozenset({"ref", "key"}),
    "wait_ms": frozenset({"ms"}),
    "reperceive": frozenset(),
}

# Actions that operate on a ref must verify the ref exists in the tree.
_ACTIONS_USING_REF = frozenset({
    "form_input", "click_ref", "select_option_ref", "press_ref",
})


def _parse_planner_response(raw: str) -> Optional[dict]:
    """
    Parse the LLM's JSON output. Tolerant to:
      - bare JSON (preferred)
      - code fences (closed and unclosed)
      - prose around the JSON
      - mid-string truncation (delegates to computer_agent's recoverer for
        consistency with the vision planner — same robustness, same place)
    Returns None on irrecoverable failure.
    """
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None

    # Strip code-fence opener (closed and unclosed forms)
    fence_closed = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", raw, re.DOTALL)
    if fence_closed:
        body = fence_closed.group(1).strip()
    else:
        fence_open = re.match(r"^```(?:json)?\s*", raw)
        body = raw[fence_open.end():].strip() if fence_open else raw

    # Direct parse first
    try:
        if body.startswith("{"):
            return json.loads(body)
    except json.JSONDecodeError:
        pass

    # Brace-balanced first object
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

    # Last resort: shared truncation recovery from the vision planner.
    try:
        from .. import vision as _ca
        recovered = _ca._recover_truncated_json(body)
        if recovered != body:
            try:
                return json.loads(recovered)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    return None


def _validate_action(
    action: dict, valid_refs: set[str], element_lookup: dict[str, dict]
) -> tuple[bool, str]:
    """
    Per-action validation. Returns (ok, reason). When `ok` is False, the
    reason explains why so the orchestrator can format planner-feedback for
    the next iteration ("you used ref X which does not exist; available
    refs: ...").

    `element_lookup` maps ref → planner-dict (the same shape `_to_planner_dict`
    produces). Used for visible/enabled checks.
    """
    if not isinstance(action, dict):
        return False, "action is not a dict"
    atype = action.get("type")
    if not isinstance(atype, str):
        return False, "missing 'type' string"
    if atype not in _VALID_ACTION_TYPES:
        return False, f"unknown type {atype!r}"

    required = _ACTION_REQUIRED_KEYS[atype]
    missing = [k for k in required if k not in action]
    if missing:
        return False, f"missing keys {missing!r} for type {atype!r}"

    # ref existence + targetability
    if atype in _ACTIONS_USING_REF:
        ref = action.get("ref")
        if not isinstance(ref, str) or not ref:
            return False, f"ref must be a non-empty string"
        if ref not in valid_refs:
            return False, f"ref {ref!r} not in tree"
        elem = element_lookup.get(ref) or {}
        # visible: only present in serialized form when False
        if elem.get("visible") is False:
            return False, f"ref {ref!r} is not visible"
        if elem.get("enabled") is False:
            return False, f"ref {ref!r} is not enabled"

    # Type-specific value sanity
    if atype == "form_input":
        v = action.get("value")
        if not isinstance(v, (str, int, float)) and v is not None:
            return False, f"form_input.value must be string-coercible, got {type(v).__name__}"
    elif atype == "select_option_ref":
        opt = action.get("option")
        if not isinstance(opt, str) or not opt:
            return False, "select_option_ref.option must be a non-empty string"
        # If element has an options array, verify membership for clearer errors
        elem = element_lookup.get(action["ref"]) or {}
        opts = elem.get("options") or []
        if opts and opt not in opts:
            return False, (
                f"select_option_ref option {opt!r} not in element's options "
                f"{opts!r}"
            )
    elif atype == "press_ref":
        k = action.get("key")
        if not isinstance(k, str) or not k:
            return False, "press_ref.key must be a non-empty string"
    elif atype == "wait_ms":
        ms = action.get("ms")
        if not isinstance(ms, (int, float)) or ms < 0 or ms > 30000:
            return False, f"wait_ms.ms must be 0..30000, got {ms!r}"

    return True, ""


def _validate_and_filter_actions(
    actions_raw: list, valid_refs: set[str], element_lookup: dict[str, dict]
) -> tuple[list[dict], list[str]]:
    """
    Walk the planner's actions. Keep valid ones; record reasons for rejected
    ones. Returns (kept, rejection_notes).
    """
    kept: list[dict] = []
    notes: list[str] = []
    for i, action in enumerate(actions_raw):
        ok, reason = _validate_action(action, valid_refs, element_lookup)
        if ok:
            kept.append(action)
        else:
            label = ""
            if isinstance(action, dict):
                label = (
                    f"type={action.get('type')!r} "
                    f"ref={action.get('ref', '')!r} "
                ).strip()
            notes.append(f"action[{i}] ({label}): {reason}")
    return kept, notes


# ─── Plan entry point ─────────────────────────────────────────────────────────


async def plan_dom_actions(
    goal: str,
    tree: Any,  # PageDomTree from browser_dom
    *,
    feedback: str = "",
) -> DomPlan:
    """
    Ask the LLM to produce a batched DOM action plan against `tree`.

    `feedback` is optional context the orchestrator passes back in subsequent
    iterations: "the previous batch's read-back showed Email field empty",
    "ref X was rejected because not in tree", etc. Plumbs into the planner
    prompt so the model has the same view the orchestrator does.

    Returns a `DomPlan`. On any failure (LLM unavailable, parse fail, all
    actions invalid), returns a plan with empty `actions`, `done=False`,
    and rejection_notes describing what went wrong so the orchestrator
    can decide whether to retry / fall back.
    """
    from . import dom as bdom
    from ...llm.contracts import ask_for_plan

    # Build the prompt body — tree serialization is the single source of
    # truth for what the planner can refer to.
    serialized_tree = bdom.serialize_for_planner(tree)
    feedback_block = ""
    if feedback:
        feedback_block = f"\n\nFEEDBACK FROM PREVIOUS ITERATION:\n{feedback}\n"

    user_prompt = (
        f"GOAL: {goal}\n\n"
        f"PAGE DOM (interactive elements):\n{serialized_tree}"
        f"{feedback_block}\n\n"
        f"Produce the next batch of actions. Output JSON only."
    )

    logger.debug(f"[DOM_PLAN] planner input:\n{user_prompt}")

    try:
        raw = await ask_for_plan(
            user_prompt,
            system_prompt=DOM_PLANNER_SYSTEM_PROMPT,
            json_mode=True,
            max_tokens=2048,
        )
    except Exception as e:
        logger.warning(f"[DOM_PLAN] LLM call crashed: {type(e).__name__}: {e}")
        return DomPlan(
            thinking="", plan="", actions=[], done=False, needs_reperceive=False,
            rejection_notes=[f"llm_crash:{type(e).__name__}"], raw_response="",
        )

    if not raw or raw == "__LLM_UNAVAILABLE__":
        logger.warning("[DOM_PLAN] LLM unavailable")
        return DomPlan(
            thinking="", plan="", actions=[], done=False, needs_reperceive=False,
            rejection_notes=["llm_unavailable"], raw_response=str(raw or ""),
        )

    logger.debug(f"[DOM_PLAN] planner raw output:\n{raw}")

    parsed = _parse_planner_response(raw)
    if parsed is None:
        logger.error(f"[DOM_PLAN] parse failed; raw preview: {raw[:200]!r}")
        return DomPlan(
            thinking="", plan="", actions=[], done=False, needs_reperceive=False,
            rejection_notes=["parse_failed"], raw_response=raw,
        )

    thinking = str(parsed.get("thinking", "") or "")
    plan_summary = str(parsed.get("plan", "") or "")
    actions_raw = parsed.get("actions") or []
    if not isinstance(actions_raw, list):
        actions_raw = []
    done = bool(parsed.get("done"))
    needs_reperceive = bool(parsed.get("needs_reperceive"))

    # Build the validation lookup table from the tree's serialized form so
    # visible/enabled checks see the same shape the planner saw.
    valid_refs = set(tree.ref_to_locator.keys())
    element_lookup: dict[str, dict] = {}
    try:
        # Re-parse the serialized tree to recover the per-element flags
        # in the exact form the planner consumed (visible/enabled appear
        # only when False, so default is True).
        serialized_obj = json.loads(serialized_tree)
        for row in serialized_obj.get("elements") or []:
            if isinstance(row, dict) and isinstance(row.get("ref"), str):
                element_lookup[row["ref"]] = row
    except json.JSONDecodeError:
        pass

    kept_actions, rejection_notes = _validate_and_filter_actions(
        actions_raw, valid_refs, element_lookup,
    )

    if rejection_notes:
        logger.info(
            f"[DOM_PLAN] rejected {len(rejection_notes)} of "
            f"{len(actions_raw)} actions: {rejection_notes}"
        )

    if kept_actions:
        logger.info(
            f"[DOM_PLAN] {len(kept_actions)} action(s), done={done}, "
            f"reperceive={needs_reperceive}, thinking={thinking[:80]!r}"
        )
    else:
        logger.warning(
            f"[DOM_PLAN] empty action batch (rejections={len(rejection_notes)}); "
            f"thinking={thinking[:80]!r}"
        )

    return DomPlan(
        thinking=thinking,
        plan=plan_summary,
        actions=kept_actions,
        done=done,
        needs_reperceive=needs_reperceive,
        rejection_notes=rejection_notes,
        raw_response=raw,
    )
