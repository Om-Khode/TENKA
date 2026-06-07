"""
browser_dom_orchestrator.py — Top-level perceive→plan→execute loop.

Drives the DOM-aware automation flow:
  1. Perceive the page (browser_dom.read_page_dom)
  2. Disambiguate which form to fill (multi-form page handling)
  3. Plan a batched action set (browser_dom_planner.plan_dom_actions)
  4. Dispatch the actions (browser_dom_executor.execute_dom_batch)
  5. On failure, feed observations back to the planner
  6. On tree-dirty / needs-reperceive, invalidate cache and re-read
  7. Bail at MAX_DOM_LOOPS

Multi-form disambiguation:
  - Group elements by `form_id` (computed in browser_dom from
    document.forms index)
  - Prefer forms whose elements are in_dialog=True (modal = active focus)
  - Score by token overlap between goal text and submit-button names
  - Tiebreak: lowest form_id alphabetically

Replan-on-validation-error:
  When the executor returns failed actions, the orchestrator formats a
  concise feedback string and passes it to the next plan_dom_actions call.
  The planner sees:
    "Previous batch had failures:
     - form_input ref=ref0001: read-back mismatch: expected 'John' got ''"
  and adjusts. The feedback-plumbing alone fixes most cases.

Never raises. Failures decay into a `DomTaskResult(success=False, reason=...)`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from . import dom as browser_dom, dom_executor as browser_dom_executor, dom_planner as browser_dom_planner
from . import dom_mapper as browser_dom_mapper, dom_filler as browser_dom_filler

logger = logging.getLogger("browser_dom_orchestrator")


# ─── Result type ──────────────────────────────────────────────────────────────


@dataclass
class DomTaskResult:
    """
    What the orchestrator returns to its caller.

    `success`         — True iff a planner iteration declared `done=True`
                        AND the corresponding batch all-succeeded
    `reason`          — short tag for telemetry / logging
                        ("completed", "max_loops", "perceive_failed",
                        "planner_failed", "no_actions", "loop_failure")
    `loops_used`      — how many perceive→plan→execute iterations ran
    `final_summary`   — TTS-friendly one-liner ≤200 chars
    `history`         — per-loop dict for debug / postmortem (NOT for
                        production telemetry — keep it short)
    """
    success: bool
    reason: str
    loops_used: int = 0
    final_summary: str = ""
    history: list[dict] = field(default_factory=list)


# ─── Multi-form disambiguation ────────────────────────────────────────────────

# Tokens that suggest "this is the form's primary submit button". Used by
# the form-selection scorer. Generic across CTAs we've seen — extend if
# new patterns emerge.
_SUBMIT_NAME_TOKENS = frozenset({
    "submit", "send", "schedule", "demo", "book",
    "register", "signup", "sign", "login", "log",
    "subscribe", "join", "continue", "next",
    "save", "create", "request", "contact", "search",
})

# Default loop cap. Per design — most form-fill flows finish in 1-2 loops;
# 5 covers cascading reveals + reperceives without letting a stuck loop
# burn money. Configurable via `max_loops` kwarg of run_dom_task.
MAX_DOM_LOOPS = 5

# When post-submit perception returns the *same* set of validation errors as
# the previous post-submit pass, the planner is stuck (typically because the
# value the user supplied is intrinsically invalid — e.g. "9999999" for a
# phone field that requires 10 digits). One repeat is enough evidence; bail
# out with an honest summary rather than burn the remaining loop budget on
# identical re-fills.
NO_PROGRESS_REPEAT_THRESHOLD = 1


def _looks_like_submit(name: str) -> bool:
    """True if the element name suggests a form-submit button."""
    if not isinstance(name, str) or not name:
        return False
    lower = name.lower()
    return any(tok in lower for tok in _SUBMIT_NAME_TOKENS)


def _select_target_form(
    elements: list[browser_dom.ElementInfo], goal: str
) -> Optional[tuple[str, list[browser_dom.ElementInfo]]]:
    """
    Pick the form to operate on from a multi-form page.

    Returns (form_id, elements_in_that_form) or None when no element has
    a form ancestor (e.g. a search bar floating in page chrome — caller
    should then operate on the full tree).

    Selection rules in priority order:
      1. Single form on page → use it.
      2. Modal preference: any form with in_dialog=True elements wins
         over background forms.
      3. Goal-vs-submit-name scoring: token overlap between the goal text
         and each form's submit-button names. Highest score wins.
      4. Tiebreak by form_id alphabetical order (deterministic).
    """
    by_form: dict[str, list[browser_dom.ElementInfo]] = {}
    for e in elements:
        if e.form_id:
            by_form.setdefault(e.form_id, []).append(e)

    if not by_form:
        return None

    if len(by_form) == 1:
        fid = next(iter(by_form))
        return fid, by_form[fid]

    # Modal preference. If at least one form has any in_dialog=True
    # element, restrict the candidate set to in-dialog forms.
    modal_candidates = {
        fid: els for fid, els in by_form.items()
        if any(e.in_dialog for e in els)
    }
    if len(modal_candidates) == 1:
        fid = next(iter(modal_candidates))
        return fid, modal_candidates[fid]
    candidates = modal_candidates if modal_candidates else by_form

    # Goal-vs-submit scoring. Token overlap is permissive enough to match
    # "fill the demo form" → form whose submit says "Schedule a Demo".
    goal_tokens = {t for t in goal.lower().split() if len(t) >= 3}
    best_score = -1
    best_fid: Optional[str] = None
    for fid in sorted(candidates.keys()):
        score = 0
        for e in candidates[fid]:
            if (e.role == "button" and e.visible and e.enabled
                    and _looks_like_submit(e.name)):
                btn_tokens = {t for t in e.name.lower().split() if len(t) >= 3}
                score += len(goal_tokens & btn_tokens)
        if score > best_score:
            best_score = score
            best_fid = fid

    if best_fid is None:
        # No form had a recognizable submit button — fall back to first.
        best_fid = sorted(candidates.keys())[0]

    return best_fid, candidates[best_fid]


def _scope_tree_to_elements(
    tree: browser_dom.PageDomTree,
    target_elements: list[browser_dom.ElementInfo],
) -> browser_dom.PageDomTree:
    """
    Build a scoped PageDomTree containing only the specified elements
    (and their corresponding ref→Locator mappings). Truncated count and
    viewport carried forward so the planner sees the same context.

    Validation errors anchored to a ref outside the scope are
    dropped; page-level alerts (field_ref="") are kept so the planner sees
    "the form rejected with: <message>" even when the failing field isn't
    in the scoped form. (In practice both forms typically share the same
    document-level alert region — this is a conservative pass-through.)
    """
    target_refs = {e.ref for e in target_elements}
    scoped_ref_map = {
        r: loc for r, loc in tree.ref_to_locator.items() if r in target_refs
    }
    scoped_errors = tuple(
        ve for ve in tree.validation_errors
        if not ve.field_ref or ve.field_ref in target_refs
    )
    return browser_dom.PageDomTree(
        elements=target_elements,
        ref_to_locator=scoped_ref_map,
        truncated=tree.truncated,
        read_at=tree.read_at,
        viewport=tree.viewport,
        validation_errors=scoped_errors,
    )


# ─── Submit detection + validation-error feedback ─────────────────────────────


def _batch_contained_submit(
    actions: list[dict],
    scoped_tree: browser_dom.PageDomTree,
) -> bool:
    """
    True when the executed batch ended with (or included) a click on a
    button whose name looks like a form submit. Used to decide whether to
    force a post-submit perceive→plan loop even when the planner declared
    `done=True`.

    We look up the click target's name via the scoped tree's ref→element
    map rather than a separate name field on the action — actions only
    carry refs, never names, by design.
    """
    if not actions:
        return False
    name_by_ref = {e.ref: (e.name or "") for e in scoped_tree.elements}
    role_by_ref = {e.ref: (e.role or "") for e in scoped_tree.elements}
    type_by_ref = {e.ref: (e.type or "") for e in scoped_tree.elements}
    for a in actions:
        if not isinstance(a, dict):
            continue
        if a.get("type") != "click_ref":
            continue
        ref = a.get("ref")
        if not isinstance(ref, str):
            continue
        name = name_by_ref.get(ref, "")
        role = role_by_ref.get(ref, "")
        # Buttons named like submits OR <input type="submit"> count.
        if role == "button" and _looks_like_submit(name):
            return True
        if type_by_ref.get(ref) == "submit":
            return True
    return False


def _post_submit_navigated(
    tree: browser_dom.PageDomTree,
    baseline_url: str,
    baseline_refs: set[str],
) -> tuple[bool, str]:
    """
    Post-submit navigation detection.

    Decide whether `tree` represents a page that has successfully navigated
    or transitioned away from the form we just submitted. Three independent
    signals — any one fires returns ``(True, reason)``:

      1. `tree.evaluate_failed` — read_page_dom hit "Execution context was
         destroyed" mid-perception. Playwright raises this exclusively when
         the page is navigating during the JS pass. Strongest signal.

      2. URL changed since baseline — a hard navigation completed before
         the next perception (`https://site/form` → `https://site/thanks`).
         Both URLs must be non-empty to avoid false positives when capture
         failed (test stubs without a `.url` property, etc.).

      3. Webflow-style soft transition — every ref the form had at submit
         time is now absent from the DOM OR present-but-invisible, AND the
         tree carries no validation errors. Webflow keeps the form node
         and toggles a sibling success message; the form refs survive but
         go ``visible=False``. The "no errors" guard prevents this firing
         on a re-rendered form that's still showing field-level rejections.

    Returns ``(False, "")`` when no signal fires — caller falls through to
    the existing post-submit verification logic.
    """
    if tree.evaluate_failed:
        return True, "page navigated during perception (evaluate context destroyed)"

    if baseline_url and tree.url and baseline_url != tree.url:
        return True, f"page navigated: {baseline_url} → {tree.url}"

    if baseline_refs and not tree.validation_errors:
        present_refs = {e.ref for e in tree.elements}
        # Build a set of baseline refs that are still present AND visible.
        # If empty, the form is gone-or-hidden — strong soft-transition
        # signal as long as no errors are anchored to the (perhaps still
        # present-but-invisible) refs.
        still_visible = {
            e.ref for e in tree.elements
            if e.ref in baseline_refs and e.visible
        }
        if not still_visible:
            absent = baseline_refs - present_refs
            invisible = baseline_refs - absent - still_visible
            return True, (
                f"form transitioned away after submit "
                f"(absent={len(absent)}, invisible={len(invisible)})"
            )

    return False, ""


def _value_appears_in_goal(value: str, goal: str) -> bool:
    """
    Whole-word match — true iff `value` appears in
    `goal` flanked by non-alphanumeric characters (or string edges). The
    word-boundary check prevents `"Test"` from matching within
    `"testing"` while still matching standalone tokens, multi-word
    phrases, numbers, and punctuated values like emails.

    Strict: match is case-insensitive and whitespace-collapsed but no
    fuzzy / synonym expansion. The pin-detection contract is "user typed
    these literal characters in the goal" — anything looser would
    mistakenly pin planner-invented values that incidentally overlap.
    """
    if not value or not goal:
        return False
    v = " ".join(value.split()).lower()
    g = " ".join(goal.split()).lower()
    if not v or v not in g:
        return False
    idx = g.find(v)
    while idx >= 0:
        before_ok = (idx == 0) or (not g[idx - 1].isalnum())
        after = idx + len(v)
        after_ok = (after == len(g)) or (not g[after].isalnum())
        if before_ok and after_ok:
            return True
        idx = g.find(v, idx + 1)
    return False


def _normalize_field_name(name: str) -> str:
    """
    Canonical key for pin-by-name matching.
    Whitespace-collapsed, lowercased. Used to match across perception
    iterations where refs can mutate (refs include bounds_quantized;
    a re-render that shifts layout mutates refs even though the field
    is semantically the same).
    """
    if not isinstance(name, str):
        return ""
    return " ".join(name.split()).strip().lower()


def _extract_pinned_field_names(
    actions: list[dict],
    goal: str,
    scoped_tree: browser_dom.PageDomTree,
) -> dict[str, str]:
    """
    Identify which fill actions' values came from the user's explicit goal
    text vs. from the planner's imagination. User-pinned fields MUST NOT
    be silently overwritten on validation feedback — instead the
    orchestrator bails with a summary so the user knows their explicit
    value was rejected.

    Keyed by NORMALIZED FIELD NAME, not ref. Refs are content-addressed
    (`sha1(role|name|placeholder|bounds_quantized)`) and mutate when the
    form re-renders after a validation rejection (the error message DOM
    insertion shifts layout → bounds change → ref changes). Field labels
    are stable across re-renders, so `_normalize_field_name(name)` is the
    durable key.

    Only `form_input` actions count — clicks/selects don't carry text
    values the user can have specified literally. Empty `value` strings
    and refs with no captured name are skipped.
    """
    pinned: dict[str, str] = {}
    if not goal:
        return pinned
    name_by_ref = {e.ref: (e.name or "") for e in scoped_tree.elements}
    for a in actions:
        if not isinstance(a, dict):
            continue
        if a.get("type") != "form_input":
            continue
        ref = a.get("ref")
        value = a.get("value")
        if not isinstance(ref, str) or not isinstance(value, str):
            continue
        if not value.strip():
            continue
        name = name_by_ref.get(ref, "")
        norm = _normalize_field_name(name)
        if not norm:
            continue
        if _value_appears_in_goal(value, goal):
            pinned[norm] = value
    return pinned


# Backwards-compat shim: keep the old ref-based name in case any external
# caller (or test) imports it. Returns the same data converted to the old
# (set, dict) shape using the scoped_tree to back-fill ref→name lookups.
# Internal orchestrator code uses _extract_pinned_field_names directly.
def _extract_pinned_refs(
    actions: list[dict],
    goal: str,
    scoped_tree: browser_dom.PageDomTree,
) -> tuple[set[str], dict[str, str]]:
    """
    Deprecated: ref-keyed shim around `_extract_pinned_field_names`.
    Refs are NOT stable across re-renders; prefer the name-based API
    inside the orchestrator. Kept callable for tests that pre-date the
    name-keyed implementation.
    """
    pinned_names = _extract_pinned_field_names(actions, goal, scoped_tree)
    pinned_refs: set[str] = set()
    pinned_value_by_ref: dict[str, str] = {}
    for e in scoped_tree.elements:
        norm = _normalize_field_name(e.name or "")
        if norm and norm in pinned_names:
            pinned_refs.add(e.ref)
            pinned_value_by_ref[e.ref] = pinned_names[norm]
    return pinned_refs, pinned_value_by_ref


def _format_pinned_rejection_summary(
    pinned_value_by_name: dict[str, str],
    rejected_field_names: list[str],
    *,
    max_fields: int = 2,
) -> str:
    """
    Build a TTS-friendly summary for the user-value-rejected bail.
    Caller passes `rejected_refs` in priority order; this formatter
    quotes back the user's literal value and the rejecting field name.

    Examples:
      "Form rejected '99999' for 'Contact Number' — give me a different value."
      "Form rejected '99999' for 'Contact Number' and 'bad' for 'Email'
       — give me different values."

    TTS budget enforcement: <120 chars per `feedback_tts_hygiene`. If the
    2-field shape exceeds budget, fall back to 1 field. If the 1-field
    shape ALSO exceeds budget (pathologically long field name), the
    field name is truncated with an ellipsis to fit.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for field_name in rejected_field_names:
        norm = _normalize_field_name(field_name)
        if not norm or norm in seen:
            continue
        if norm not in pinned_value_by_name:
            continue
        seen.add(norm)
        # Use the original-cased field_name for display (passed in by
        # caller), but resolve value from the normalized-key dict.
        display_name = field_name.strip() or "this field"
        pairs.append((display_name, pinned_value_by_name[norm]))
        if len(pairs) >= max_fields:
            break
    if not pairs:
        return (
            "Form rejected one of your values — give me a different one."
        )
    TTS_BUDGET = 119

    def _build(items: list[tuple[str, str]]) -> str:
        if len(items) == 1:
            field, value = items[0]
            return (
                f"Form rejected {value!r} for {field!r} — give me a "
                f"different value."
            )
        formatted = " and ".join(
            f"{value!r} for {field!r}" for field, value in items
        )
        return f"Form rejected {formatted} — give me different values."

    summary = _build(pairs)
    # Drop pairs from the tail until the summary fits.
    while len(summary) > TTS_BUDGET and len(pairs) > 1:
        pairs.pop()
        summary = _build(pairs)
    # Last-ditch: a single pair with a pathologically long field name still
    # over budget — truncate the field name with an ellipsis.
    if len(summary) > TTS_BUDGET and pairs:
        field, value = pairs[0]
        overrun = len(summary) - TTS_BUDGET
        # Leave at least 8 chars + "…" of the field name visible.
        new_len = max(8, len(field) - overrun - 1)
        if new_len < len(field):
            field = field[:new_len] + "…"
            summary = (
                f"Form rejected {value!r} for {field!r} — give me a "
                f"different value."
            )
    return summary


def _log_validation_dump(
    errors: tuple[browser_dom.ValidationError, ...],
    elements: list[browser_dom.ElementInfo],
    *,
    tag: str = "",
) -> None:
    """Emit one DEBUG line per ValidationError for diagnostic correlation."""
    name_by_ref = {e.ref: (e.name or "") for e in elements}
    aria_invalid_by_ref = {e.ref: e.aria_invalid for e in elements}
    prefix = f"[DOM_ORCH][diag:{tag}]" if tag else "[DOM_ORCH][diag]"
    aria_invalid_fields = [
        f"{(e.name or '?')!r}(ref={e.ref})"
        for e in elements if e.aria_invalid
    ]
    logger.debug(
        f"{prefix} aria_invalid fields ({len(aria_invalid_fields)}): "
        f"{', '.join(aria_invalid_fields) if aria_invalid_fields else '(none)'}"
    )
    for i, ve in enumerate(errors):
        anchor = "page-level" if not ve.field_ref else (
            f"{name_by_ref.get(ve.field_ref, '?')!r}(ref={ve.field_ref}, "
            f"aria_invalid={aria_invalid_by_ref.get(ve.field_ref, '?')})"
        )
        msg = (ve.message or "").replace("\n", " ").strip()[:160]
        logger.debug(
            f"{prefix} err[{i}] source={ve.source} anchor={anchor} "
            f"msg={msg!r}"
        )


def _validation_signature(
    errors: tuple[browser_dom.ValidationError, ...],
) -> frozenset:
    """
    Build a stable signature of the current validation
    error set so two consecutive passes can be compared for "did anything
    change". Uses (field_ref, message) pairs because the same field can
    flip between different error messages as the user-supplied value
    changes, and we only want to bail when both stay constant.
    """
    return frozenset(
        (ve.field_ref, (ve.message or "").strip().lower())
        for ve in errors
    )


# Synonym aliases used by goal-aware error prioritization. The user
# typically parameterizes one specific field via the goal text ("mobile
# number as 99999"); the form labels that field with a different word
# ("Contact Number"). When the goal token doesn't literally match a
# field-name token, the alias map bridges the two so the user-supplied
# bad-value field still surfaces first in the summary.
#
# Generic only — no app-specific entries. Each canonical key maps to a
# tuple of equivalent tokens; matching is bidirectional (any token in
# either set matches the other set).
_FIELD_ALIASES: tuple[frozenset[str], ...] = (
    frozenset({"mobile", "phone", "contact", "tel", "telephone", "cell", "cellphone"}),
    frozenset({"email", "mail", "e-mail"}),
    frozenset({"name", "first", "last", "full", "firstname", "lastname"}),
    frozenset({"company", "organization", "business", "employer", "org"}),
    frozenset({"address", "street", "location"}),
    frozenset({"zip", "postal", "postcode"}),
    frozenset({"country", "nation", "region"}),
    frozenset({"password", "passcode", "pwd"}),
)


def _expand_goal_tokens(goal: str) -> frozenset[str]:
    """
    Tokenize `goal` into lowercase ≥3-letter tokens, then expand by
    pulling in every alias-set sibling for each token. So "mobile" ⇒
    {"mobile", "phone", "contact", "tel", ...} — the field-name match
    later treats any of those as a hit on a "Contact Number" label.

    Returns an empty frozenset for empty/short goals (no prioritization).
    """
    raw = {t.lower().strip(",.!?:;'\"()[]") for t in goal.split() if len(t) >= 3}
    if not raw:
        return frozenset()
    expanded = set(raw)
    for alias_set in _FIELD_ALIASES:
        if raw & alias_set:
            expanded |= alias_set
    return frozenset(expanded)


def _field_matches_goal(field_name: str, goal_tokens: frozenset[str]) -> bool:
    """True when any ≥3-letter token in `field_name` is in `goal_tokens`."""
    if not field_name or not goal_tokens:
        return False
    field_tokens = {
        t.lower() for t in field_name.split() if len(t) >= 3
    }
    return bool(field_tokens & goal_tokens)


def _format_unresolved_for_summary(
    errors: tuple[browser_dom.ValidationError, ...],
    elements: list[browser_dom.ElementInfo],
    *,
    goal: str = "",
    max_fields: Optional[int] = None,
) -> str:
    """
    Build a TTS-friendly summary naming the fields the form rejected. Used
    as `final_summary` when the run ends with unresolved validation errors
    (last-loop verification or early no-progress exit).

    Goal-aware prioritization: errors whose anchored field name shares a
    token with `goal` (after alias expansion via `_FIELD_ALIASES`) sort
    BEFORE other errors. The user's deliberately-supplied bad field appears
    first in the summary so the personality LLM has the right anchor to
    comment on.

    Adaptive cap: when total errors <= 5, list up to 5 named fields (still
    <120 chars per `feedback_tts_hygiene`); otherwise cap at 3 to keep
    audio short. Override via `max_fields`.

    Examples:
      "The form rejected 'Contact Number' — please give me a valid value."
      "The form rejected 'Email' and 'Phone' — please give me valid values."
      "The form rejected 'Contact Number', 'First name', 'Last Name',
         'Company Name' and 'Work Email' — please give me valid values."
      "The form still has 8 errors — I couldn't fix them automatically."
    """
    if not errors:
        return "Done."

    # Adaptive cap: anchor-error count fits in TTS budget when small.
    if max_fields is None:
        max_fields = 5 if len(errors) <= 5 else 3

    name_by_ref = {e.ref: (e.name or "") for e in elements}
    goal_tokens = _expand_goal_tokens(goal)

    # Sort errors so goal-matching ones come first; preserve original
    # relative order within each group (stable sort).
    def _priority(ve: browser_dom.ValidationError) -> int:
        nm = name_by_ref.get(ve.field_ref, "")
        return 0 if _field_matches_goal(nm, goal_tokens) else 1

    ordered = sorted(errors, key=_priority)

    named: list[str] = []
    seen: set[str] = set()
    for ve in ordered:
        if not ve.field_ref:
            continue
        nm = name_by_ref.get(ve.field_ref, "").strip()
        # Dedupe case-insensitively — forms often anchor multiple errors
        # to the same field with cosmetic case variation in the label
        # ("First name" placeholder vs "First Name" aria-label).
        nm_key = nm.lower()
        if nm and nm_key not in seen:
            seen.add(nm_key)
            named.append(nm)
        if len(named) >= max_fields:
            break
    if not named:
        return f"The form still has {len(errors)} errors — I couldn't fix them automatically."

    # TTS hygiene (per `feedback_tts_hygiene`): keep summary under 120 chars
    # so audio playback stays under ~6 seconds. With 5 long field names
    # (e.g. Truein's "Company Name", "Contact Number") the canonical form
    # exceeds 120; trim the named-list one entry at a time until it fits.
    # The earliest-listed names are the highest-priority anchors (goal
    # match first, DOM order second), so dropping from the tail is correct.
    TTS_BUDGET = 119

    def _build(items: list[str]) -> str:
        if len(items) == 1:
            return f"The form rejected {items[0]!r} — please give me a valid value."
        quoted = ", ".join(repr(n) for n in items[:-1]) + f" and {items[-1]!r}"
        return f"The form rejected {quoted} — please give me valid values."

    summary = _build(named)
    while len(summary) > TTS_BUDGET and len(named) > 1:
        named.pop()
        summary = _build(named)
    return summary


def _filter_wasteful_refills(
    actions: list[dict],
    flagged_refs: set[str],
) -> tuple[list[dict], int]:
    """
    When we're in a corrective pass (validation feedback is active), drop
    `form_input` actions targeting refs the form did NOT flag — the planner
    sometimes re-fills already-correct fields out of habit, which is a
    no-op at best and overwrites a valid value with the same one at cost.
    Keeps clicks (re-submit), selects on flagged fields, and any non-fill
    action verbatim.

    Returns (filtered_actions, dropped_count). When no flagged_refs are
    known (page-level errors only), returns the input untouched — there's
    no field-level signal to filter on.
    """
    if not flagged_refs or not actions:
        return list(actions), 0
    kept: list[dict] = []
    dropped = 0
    for a in actions:
        if not isinstance(a, dict):
            kept.append(a)
            continue
        atype = a.get("type")
        ref = a.get("ref")
        if atype == "form_input" and ref and ref not in flagged_refs:
            dropped += 1
            continue
        kept.append(a)
    return kept, dropped


def _format_validation_for_planner(
    errors: tuple[browser_dom.ValidationError, ...],
    elements: list[browser_dom.ElementInfo],
    *,
    max_lines: int = 8,
) -> str:
    """
    Build the planner-feedback block describing which fields the form
    rejected. Anchors each error to its field name (planner reasons over
    names, not refs).

      VALIDATION ERRORS FROM PRIOR SUBMIT:
      - "Email" (ref=ref0001): Please enter a valid email address.
      - page-level: Some fields are missing required information.

    Capped at `max_lines` to keep prompt tokens bounded.
    """
    if not errors:
        return ""
    name_by_ref = {e.ref: (e.name or "") for e in elements}
    lines = ["VALIDATION ERRORS FROM PRIOR SUBMIT (the form rejected — fix and resubmit):"]
    for ve in errors[:max_lines]:
        if ve.field_ref:
            field_name = name_by_ref.get(ve.field_ref, "") or "<unnamed field>"
            lines.append(f"- {field_name!r} (ref={ve.field_ref}): {ve.message}")
        else:
            lines.append(f"- page-level: {ve.message}")
    if len(errors) > max_lines:
        lines.append(f"  ... ({len(errors) - max_lines} more)")
    return "\n".join(lines)


# ─── Feedback formatter for failed batches ────────────────────────────────────


def _format_failures_for_planner(
    batch_result: browser_dom_executor.DomBatchResult, *, max_lines: int = 6,
) -> str:
    """
    Replan-on-validation-error feedback. Builds a concise multi-line
    summary the planner can consume verbatim:

      Previous batch had failures:
      - form_input ref=ref0001: read-back mismatch: expected 'John' got ''
      - click_ref ref=ref0007: click failed: TimeoutError: ...

    Caps at `max_lines` to keep planner-prompt token cost bounded.
    """
    failed = batch_result.failed
    if not failed:
        return ""
    lines = ["Previous batch had failures:"]
    for r in failed[:max_lines]:
        atype = r.action.get("type", "?")
        ref = r.action.get("ref", "")
        ref_part = f" ref={ref}" if ref else ""
        if r.observed_value:
            lines.append(
                f"- {atype}{ref_part}: {r.error}; observed={r.observed_value!r}"
            )
        else:
            lines.append(f"- {atype}{ref_part}: {r.error}")
    if len(failed) > max_lines:
        lines.append(f"  ... ({len(failed) - max_lines} more)")
    return "\n".join(lines)


# ─── Main orchestrator ───────────────────────────────────────────────────────


async def run_dom_task(
    goal: str,
    page: Any,
    *,
    max_loops: int = MAX_DOM_LOOPS,
) -> DomTaskResult:
    """
    Top-level entry point. Drives the perceive→plan→execute loop until
    the planner declares done OR `max_loops` is exhausted.

    `page` is a Playwright `Page` (or test stub exposing the same surface
    used by browser_dom / browser_dom_executor).

    Never raises. All failure modes return a DomTaskResult with
    success=False and a tagged `reason`.
    """
    history: list[dict] = []
    feedback = ""
    # When a successful batch ended in a submit click, we suppress the
    # planner's `done=True` and force one more perceive→plan loop so
    # post-submit validation errors get a chance to surface. The flag is
    # set after the executing loop and consumed by the next loop's done
    # check. Reset after one verification pass so we don't override forever.
    pending_post_submit_check = False
    # Track the previous post-submit error signature so we can detect "no
    # progress" — when the same set of (field_ref, message) pairs surfaces
    # twice in a row, the planner is stuck on a value it can't fix and we
    # bail out early.
    last_error_signature: Optional[frozenset] = None
    # The set of field refs the form is currently complaining about. Set
    # when a corrective pass starts; consumed when the next plan is
    # filtered to drop refills on un-flagged fields.
    flagged_refs: set[str] = set()
    # Post-submit navigation baseline. Captured at the perception
    # immediately before a submit batch is dispatched. Persists across
    # loops — every subsequent perception runs nav-detection so a transition
    # that happens DURING a corrective fill batch (not just immediately
    # after the first submit) still gets caught. Reset only when a fresh
    # submit fires (overwrite) — never cleared mid-run.
    submit_baseline_url: str = ""
    submit_baseline_refs: set[str] = set()
    # User-pinned value tracking. Fields whose fill values came directly
    # from the goal text (literal whole-word match, case-insensitive) are
    # NEVER silently substituted on validation feedback — the user's
    # explicit intent wins over auto-correction. Computed once from the
    # loop-1 plan and persisted unchanged through the rest of the run.
    # Keyed by NORMALIZED FIELD NAME (not ref) — refs are content-addressed
    # including bounds, which mutate when the form re-renders after a
    # rejection. Field labels are stable.
    pinned_value_by_name: dict[str, str] = {}

    for loop_i in range(max_loops):
        # Give client-side validation a tick to render its alerts before
        # we re-perceive. 300 ms is enough for React's
        # synchronous error-state render; we don't need network-idle here
        # because validation errors are by definition client-side.
        if pending_post_submit_check:
            try:
                await asyncio.sleep(0.3)
            except Exception:
                pass
            browser_dom.invalidate_tree_cache(page)

        # ── 1. Perceive ──
        try:
            full_tree = await browser_dom.read_page_dom(page, filter="interactive")
        except Exception as e:
            # read_page_dom is fail-open and should never reach here, but
            # belt-and-braces guards the orchestrator's contract.
            logger.warning(f"[DOM_ORCH] perceive raised: {type(e).__name__}: {e}")
            return DomTaskResult(
                success=False, reason="perceive_failed",
                loops_used=loop_i,
                final_summary="Could not read the page.",
                history=history,
            )

        # ── Post-submit navigation short-circuit ──
        # Once any submit has been dispatched in this run, every subsequent
        # perception is a candidate for the nav-success path. Catches three
        # cases the prior verification path missed:
        #   - Form's submit click triggered a hard navigation (URL change)
        #     completed before perception ran.
        #   - Page was mid-navigation during perception (evaluate_failed).
        #   - Webflow-style soft transition: form node hidden in favor of a
        #     success message; URL unchanged but baseline refs all invisible.
        # Runs BEFORE the existing pending_post_submit_check block so a
        # navigation signal wins over an empty validation_errors snapshot
        # that just happened to coincide with the transition moment.
        if submit_baseline_url or submit_baseline_refs:
            navigated, nav_reason = _post_submit_navigated(
                full_tree, submit_baseline_url, submit_baseline_refs,
            )
            if navigated:
                history.append({
                    "loop": loop_i, "scope": "post_submit_nav",
                    "result": "navigation_success",
                    "detail": nav_reason,
                })
                logger.info(
                    f"[DOM_ORCH] post-submit navigation detected — "
                    f"{nav_reason} — accepting submit"
                )
                return DomTaskResult(
                    success=True, reason="completed",
                    loops_used=loop_i + 1,
                    final_summary="Form submitted.",
                    history=history,
                )

        # ── Post-submit verification short-circuit ──
        # If the previous loop ended with a submit, this loop's job is just
        # to look for validation errors. If perception came back clean, the
        # form was accepted and we can declare success without burning an
        # LLM call on a no-op plan.
        if pending_post_submit_check:
            errs = full_tree.validation_errors
            if not errs:
                history.append({
                    "loop": loop_i, "scope": "post_submit_clean",
                    "elements": len(full_tree.elements),
                    "result": "post_submit_no_errors",
                })
                logger.info(
                    "[DOM_ORCH] post-submit perception clean — accepting submit"
                )
                return DomTaskResult(
                    success=True, reason="completed",
                    loops_used=loop_i + 1,
                    final_summary="Form submitted.",
                    history=history,
                )
            # If any error anchors to a user-pinned field name,
            # bail IMMEDIATELY. The user gave us literal values in the
            # goal text — the form rejected one. Silently substituting
            # would disrespect the user's explicit intent. The planner
            # must never see a corrective-fill prompt for these fields.
            # Match by NORMALIZED NAME, not ref — refs include bounds
            # and mutate when the form re-renders post-rejection.
            if pinned_value_by_name:
                name_by_ref = {
                    e.ref: (e.name or "") for e in full_tree.elements
                }
                rejected_pinned_names: list[str] = []
                seen_norms: set[str] = set()
                for ve in errs:
                    if not ve.field_ref:
                        continue
                    raw_name = name_by_ref.get(ve.field_ref, "")
                    norm = _normalize_field_name(raw_name)
                    if norm and norm in pinned_value_by_name and norm not in seen_norms:
                        seen_norms.add(norm)
                        rejected_pinned_names.append(raw_name)
                if rejected_pinned_names:
                    summary = _format_pinned_rejection_summary(
                        pinned_value_by_name,
                        rejected_pinned_names,
                    )
                    history.append({
                        "loop": loop_i, "scope": "post_submit_user_pin_rejected",
                        "errors": len(errs),
                        "rejected_pinned": rejected_pinned_names,
                        "result": "user_value_rejected",
                    })
                    logger.info(
                        f"[DOM_ORCH] form rejected {len(rejected_pinned_names)} "
                        f"user-pinned field(s) {rejected_pinned_names!r} — "
                        f"bailing without corrective fill to respect "
                        f"explicit user values"
                    )
                    _log_validation_dump(
                        errs, full_tree.elements, tag="user_pin_bail",
                    )
                    return DomTaskResult(
                        success=False, reason="user_value_rejected",
                        loops_used=loop_i + 1,
                        final_summary=summary,
                        history=history,
                    )
            # Compare against the previous post-submit signature. If
            # the same errors keep coming back, the planner is stuck on a
            # value it can't recover (e.g. the user gave a 7-digit phone
            # for a 10-digit field). Bail with an honest summary.
            sig = _validation_signature(errs)
            if (
                last_error_signature is not None
                and sig == last_error_signature
            ):
                summary = _format_unresolved_for_summary(
                    errs, full_tree.elements, goal=goal,
                )
                history.append({
                    "loop": loop_i, "scope": "post_submit_no_progress",
                    "errors": len(errs),
                    "result": "validation_no_progress",
                })
                logger.info(
                    f"[DOM_ORCH] post-submit perception found {len(errs)} "
                    f"validation error(s) — same as prior pass, bailing out"
                )
                _log_validation_dump(errs, full_tree.elements, tag="bail")
                return DomTaskResult(
                    success=False, reason="validation_no_progress",
                    loops_used=loop_i + 1,
                    final_summary=summary,
                    history=history,
                )
            last_error_signature = sig
            # Remember which field refs are flagged so the next plan
            # can have its un-flagged fill_input actions filtered out.
            flagged_refs = {ve.field_ref for ve in errs if ve.field_ref}
            # Errors present — fall through into normal scoping/planning so
            # the planner can emit a corrective fill batch.
            logger.info(
                f"[DOM_ORCH] post-submit perception found "
                f"{len(errs)} validation error(s) — replanning"
            )
            _log_validation_dump(errs, full_tree.elements, tag="replan")
            feedback = _format_validation_for_planner(errs, full_tree.elements)
            pending_post_submit_check = False

        # ── 2. Disambiguate forms ──
        scope = _select_target_form(full_tree.elements, goal)
        if scope is None:
            # No form ancestors detected — operate on the full tree. This
            # covers search-bars-only pages, single-button workflows, etc.
            scoped_tree = full_tree
            scope_label = "no-form"
        else:
            target_form_id, target_elements = scope
            portal_els = [
                e for e in full_tree.elements
                if not e.form_id and e.role in _PORTAL_ROLES
            ]
            if portal_els:
                logger.info(
                    f"[DOM_ORCH] portal elements found: "
                    f"{[(e.ref, e.role, e.name) for e in portal_els]}"
                )
                target_elements = list(target_elements) + portal_els
            else:
                comboboxes = [e for e in target_elements if e.role == "combobox"]
                if comboboxes:
                    logger.debug(
                        f"[DOM_ORCH] NO portal elements but {len(comboboxes)} "
                        f"combobox(es) in form: "
                        f"{[(e.ref, e.name, e.autocomplete) for e in comboboxes]}"
                    )
            scoped_tree = _scope_tree_to_elements(full_tree, target_elements)
            scope_label = target_form_id

        if not scoped_tree.elements:
            # Perception returned nothing usable. Could be a canvas page
            # or the form hasn't loaded yet. Don't loop forever — bail.
            history.append({
                "loop": loop_i, "scope": scope_label, "elements": 0,
                "result": "empty_tree",
            })
            if loop_i == max_loops - 1:
                return DomTaskResult(
                    success=False, reason="empty_tree",
                    loops_used=loop_i + 1,
                    final_summary="No interactive elements found on page.",
                    history=history,
                )
            # Try once more with a fresh perception.
            browser_dom.invalidate_tree_cache(page)
            continue

        logger.info(
            f"[DOM_ORCH] loop {loop_i + 1}/{max_loops}: scope={scope_label}, "
            f"elements={len(scoped_tree.elements)}, feedback={'yes' if feedback else 'no'}"
        )

        # ── 3. Plan ──
        plan = await browser_dom_planner.plan_dom_actions(
            goal, scoped_tree, feedback=feedback,
        )

        # ── 3a. Extract user-pinned fields from the FIRST plan ──
        # Run only on loop 1's plan because subsequent corrective batches
        # are internal — only the user's original fills can carry user
        # intent. Persists for the rest of the run; later validation
        # errors on these named fields trigger the bail path instead of
        # corrective fill, so the user's explicit value never gets
        # silently overwritten. Keyed by normalized field name — see
        # `_extract_pinned_field_names` docstring for ref-vs-name choice.
        if loop_i == 0 and plan.actions and not pinned_value_by_name:
            pinned_value_by_name = _extract_pinned_field_names(
                plan.actions, goal, scoped_tree,
            )
            if pinned_value_by_name:
                pinned_log = ", ".join(
                    f"{name!r}={value!r}"
                    for name, value in pinned_value_by_name.items()
                )
                logger.debug(
                    f"[DOM_ORCH] user-pinned fields from goal: {pinned_log}"
                )

        # ── 3b. Filter wasteful refills on corrective passes ──
        # When the previous post-submit pass surfaced field-level errors,
        # the planner is supposed to refill only those fields (per the 2E
        # system prompt). It doesn't always obey — this is a defensive
        # second pass that drops form_input actions on un-flagged fields.
        # No-op when flagged_refs is empty (initial pass or page-level
        # errors only).
        #
        # Side-effect guard: if the filter would empty the batch
        # entirely, the planner's refs disagree with the form's anchor refs
        # (form re-renders, mis-attributed errors, etc.). Dropping every
        # action and letting the orchestrator return `completed_no_actions`
        # is a false-positive success — over-filling is recoverable, falsely
        # reporting success isn't. Restore the original batch in that case.
        if flagged_refs and plan.actions:
            kept, dropped = _filter_wasteful_refills(plan.actions, flagged_refs)
            if dropped and not kept:
                logger.info(
                    f"[DOM_ORCH] filter would empty batch ({dropped} actions "
                    f"all on un-flagged refs) — keeping original to avoid "
                    f"false-positive completion"
                )
            elif dropped:
                logger.info(
                    f"[DOM_ORCH] dropped {dropped} wasteful refill(s) on "
                    f"un-flagged fields"
                )
                plan.actions = kept

        if not plan.actions:
            # No actionable items.
            if plan.done:
                # Guard: never declare success on an empty plan when
                # the scoped tree still carries validation errors. The
                # planner believing "nothing to do" while the form is
                # actively complaining is a contradiction we MUST resolve
                # in favour of the form's ground truth — the user thinks
                # the submission worked otherwise.
                if scoped_tree.validation_errors:
                    history.append({
                        "loop": loop_i, "scope": scope_label,
                        "actions": 0,
                        "errors": len(scoped_tree.validation_errors),
                        "result": "empty_plan_with_errors",
                    })
                    logger.warning(
                        f"[DOM_ORCH] planner emitted empty plan with done=True "
                        f"but {len(scoped_tree.validation_errors)} validation "
                        f"error(s) remain — reporting unresolved"
                    )
                    return DomTaskResult(
                        success=False, reason="validation_unresolved",
                        loops_used=loop_i + 1,
                        final_summary=_format_unresolved_for_summary(
                            scoped_tree.validation_errors,
                            scoped_tree.elements,
                            goal=goal,
                        ),
                        history=history,
                    )
                # Planner says nothing left to do — accept.
                history.append({
                    "loop": loop_i, "scope": scope_label,
                    "actions": 0, "result": "done_no_actions",
                })
                return DomTaskResult(
                    success=True, reason="completed_no_actions",
                    loops_used=loop_i + 1,
                    final_summary=plan.plan or "Nothing to do.",
                    history=history,
                )
            # Plan failed (LLM error, all-rejected, parse fail). Try once more
            # with the rejection notes as feedback; bail at the cap.
            history.append({
                "loop": loop_i, "scope": scope_label, "actions": 0,
                "rejection_notes": plan.rejection_notes,
                "result": "planner_empty",
            })
            if loop_i == max_loops - 1:
                return DomTaskResult(
                    success=False, reason="planner_failed",
                    loops_used=loop_i + 1,
                    final_summary="Planner produced no usable actions.",
                    history=history,
                )
            feedback = (
                "Previous plan was empty / rejected. Reasons: "
                + "; ".join(plan.rejection_notes or ["unknown"])
            )
            continue

        # ── 4. Execute ──
        batch = await browser_dom_executor.execute_dom_batch(
            page, plan.actions, scoped_tree.ref_to_locator,
        )
        history.append({
            "loop": loop_i, "scope": scope_label,
            "thinking": plan.thinking[:120],
            "actions_planned": len(plan.actions),
            "actions_succeeded": len(batch.succeeded),
            "actions_failed": len(batch.failed),
            "tree_dirty": batch.tree_dirty,
            "reperceive": batch.requires_reperceive,
        })

        # ── 5. Cache invalidation ──
        if batch.tree_dirty or batch.requires_reperceive or plan.needs_reperceive:
            browser_dom.invalidate_tree_cache(page)

        # ── 6. Feedback for next iteration on failures ──
        if batch.failed:
            feedback = _format_failures_for_planner(batch)
            if loop_i == max_loops - 1:
                # Last loop and we still have failures — report.
                return DomTaskResult(
                    success=False, reason="loop_failure_at_max",
                    loops_used=loop_i + 1,
                    final_summary=(
                        f"Could not complete: {len(batch.failed)} of "
                        f"{len(plan.actions)} actions failed."
                    ),
                    history=history,
                )
            continue

        # ── 7. Done condition ──
        if plan.done and batch.all_succeeded:
            submit_detected = _batch_contained_submit(plan.actions, scoped_tree)
            # If the batch ended with a submit click, don't trust
            # the planner's done=True yet — give the page a chance to render
            # validation errors, then re-perceive once. The next loop's
            # post-submit short-circuit either accepts (no errors) or
            # forwards errors to the planner for a corrective fill.
            if submit_detected and loop_i < max_loops - 1:
                history[-1]["post_submit_check"] = True
                pending_post_submit_check = True
                # Capture nav baseline from the perception that fed
                # this submit. URL + visible scoped-form refs are what the
                # next perception(s) will be compared against.
                submit_baseline_url = full_tree.url
                submit_baseline_refs = {
                    e.ref for e in scoped_tree.elements if e.visible
                }
                feedback = ""
                flagged_refs = set()
                continue
            # Last loop with submit — no replan budget remains, but
            # we MUST verify the submit cleared validation before claiming
            # success. Inline final perceive (no plan, no execute), then
            # either accept or downgrade to loop_failure_at_max with an
            # honest summary listing the unresolved fields.
            if submit_detected and loop_i == max_loops - 1:
                # Capture baseline so the verify-tree gets the same
                # nav-detection treatment as a normal post-submit perception.
                submit_baseline_url = full_tree.url
                submit_baseline_refs = {
                    e.ref for e in scoped_tree.elements if e.visible
                }
                try:
                    await asyncio.sleep(0.3)
                except Exception:
                    pass
                browser_dom.invalidate_tree_cache(page)
                try:
                    verify_tree = await browser_dom.read_page_dom(
                        page, filter="interactive",
                    )
                except Exception as e:
                    logger.warning(
                        f"[DOM_ORCH] final verify perceive raised: "
                        f"{type(e).__name__}: {e}"
                    )
                    verify_tree = None
                # Run nav-detection first — if the submit triggered a
                # navigation/transition, no validation errors will exist on
                # the new page, but neither will the form. Treat as success.
                if verify_tree is not None:
                    navigated, nav_reason = _post_submit_navigated(
                        verify_tree, submit_baseline_url, submit_baseline_refs,
                    )
                    if navigated:
                        history.append({
                            "loop": loop_i, "scope": "final_verify_nav",
                            "result": "navigation_success",
                            "detail": nav_reason,
                        })
                        logger.info(
                            f"[DOM_ORCH] final verify detected navigation — "
                            f"{nav_reason} — accepting submit"
                        )
                        return DomTaskResult(
                            success=True, reason="completed",
                            loops_used=loop_i + 1,
                            final_summary="Form submitted.",
                            history=history,
                        )
                if verify_tree is not None and verify_tree.validation_errors:
                    # Even on the last-loop verify path, a pinned
                    # field rejection takes precedence over the generic
                    # loop_failure_at_max summary — the user needs to know
                    # their literal value was rejected, not just that the
                    # form has unresolved errors. Name-based matching same
                    # rationale as the post-submit bail above.
                    if pinned_value_by_name:
                        name_by_ref = {
                            e.ref: (e.name or "")
                            for e in verify_tree.elements
                        }
                        rejected_pinned_names: list[str] = []
                        seen_norms: set[str] = set()
                        for ve in verify_tree.validation_errors:
                            if not ve.field_ref:
                                continue
                            raw_name = name_by_ref.get(ve.field_ref, "")
                            norm = _normalize_field_name(raw_name)
                            if norm and norm in pinned_value_by_name and norm not in seen_norms:
                                seen_norms.add(norm)
                                rejected_pinned_names.append(raw_name)
                        if rejected_pinned_names:
                            summary = _format_pinned_rejection_summary(
                                pinned_value_by_name,
                                rejected_pinned_names,
                            )
                            history.append({
                                "loop": loop_i, "scope": "final_verify",
                                "errors": len(verify_tree.validation_errors),
                                "rejected_pinned": rejected_pinned_names,
                                "result": "user_value_rejected_at_max",
                            })
                            logger.info(
                                f"[DOM_ORCH] final verify: user-pinned "
                                f"field(s) {rejected_pinned_names!r} rejected"
                                f" — surfacing user_value_rejected"
                            )
                            return DomTaskResult(
                                success=False, reason="user_value_rejected",
                                loops_used=loop_i + 1,
                                final_summary=summary,
                                history=history,
                            )
                    history.append({
                        "loop": loop_i, "scope": "final_verify",
                        "errors": len(verify_tree.validation_errors),
                        "result": "validation_unresolved_at_max",
                    })
                    logger.warning(
                        f"[DOM_ORCH] final verify found "
                        f"{len(verify_tree.validation_errors)} validation "
                        f"error(s) at max loops — reporting failure"
                    )
                    return DomTaskResult(
                        success=False, reason="loop_failure_at_max",
                        loops_used=loop_i + 1,
                        final_summary=_format_unresolved_for_summary(
                            verify_tree.validation_errors,
                            verify_tree.elements,
                            goal=goal,
                        ),
                        history=history,
                    )
                history.append({
                    "loop": loop_i, "scope": "final_verify",
                    "errors": 0, "result": "post_submit_no_errors",
                })
                logger.info(
                    "[DOM_ORCH] final verify perception clean — accepting submit"
                )
                return DomTaskResult(
                    success=True, reason="completed",
                    loops_used=loop_i + 1,
                    final_summary="Form submitted.",
                    history=history,
                )
            return DomTaskResult(
                success=True, reason="completed",
                loops_used=loop_i + 1,
                final_summary=plan.plan or "Done.",
                history=history,
            )

        # ── 8. Loop continues — clear feedback, plan said more to do ──
        feedback = ""
        flagged_refs = set()

    # Max loops exhausted without explicit success or failure.
    return DomTaskResult(
        success=False, reason="max_loops",
        loops_used=max_loops,
        final_summary=f"Could not complete within {max_loops} steps.",
        history=history,
    )


# ─── Option B: Deterministic Form Fill ──────────────────────────────────

_MAX_FORM_FILL_ATTEMPTS = 2
_PORTAL_ROLES = {"option", "listbox"}


async def run_dom_form_fill(
    goal: str,
    page: Any,
) -> DomTaskResult:
    """
    Option B entry point: perceive → map (1 LLM call) → fill
    deterministically → detect dependent fields → fill dependents
    (optional 2nd LLM call) → submit → verify.

    LLM budget: 1 call (simple form), 2 calls (cascading fields or
    validation recovery), 3 calls worst-case (cascading + validation).
    """
    history: list[dict] = []

    # ── 1. Perceive ──
    try:
        full_tree = await browser_dom.read_page_dom(page, filter="interactive")
    except Exception as e:
        logger.warning(f"[DOM_FORM_FILL] perceive raised: {type(e).__name__}: {e}")
        return DomTaskResult(
            success=False, reason="perceive_failed",
            loops_used=0,
            final_summary="Could not read the page.",
            history=history,
        )

    # ── 2. Scope to target form ──
    scope = _select_target_form(full_tree.elements, goal)
    if scope is None:
        scoped_tree = full_tree
    else:
        target_form_id, target_elements = scope
        portal_els = [
            e for e in full_tree.elements
            if not e.form_id and e.role in _PORTAL_ROLES
        ]
        if portal_els:
            target_elements = list(target_elements) + portal_els
        scoped_tree = _scope_tree_to_elements(full_tree, target_elements)

    if not scoped_tree.elements:
        return DomTaskResult(
            success=False, reason="empty_tree",
            loops_used=0,
            final_summary="No interactive elements found on page.",
            history=history,
        )

    logger.info(
        f"[DOM_FORM_FILL] perceived {len(scoped_tree.elements)} elements"
    )

    # ── 3. Map → Fill → Verify loop (max 2 attempts) ──
    feedback = ""
    pinned_value_by_name: dict[str, str] = {}
    submit_baseline_url = ""
    submit_baseline_refs: set[str] = set()

    for attempt in range(_MAX_FORM_FILL_ATTEMPTS):
        # ── 3a. Map goal to fields (1 LLM call) ──
        mapping = await browser_dom_mapper.map_goal_to_fields(
            goal, scoped_tree, feedback=feedback,
        )

        if not mapping.fills:
            history.append({
                "attempt": attempt, "result": "mapper_empty",
                "thinking": mapping.thinking,
            })
            return DomTaskResult(
                success=False, reason="mapper_failed",
                loops_used=attempt + 1,
                final_summary="Could not determine which fields to fill.",
                history=history,
            )

        # Extract user-pinned values on first attempt
        if attempt == 0:
            pinned_value_by_name = _extract_pinned_field_names(
                [{"type": "form_input", "ref": f.ref, "value": str(f.value)}
                 for f in mapping.fills if isinstance(f.value, str)],
                goal, scoped_tree,
            )

        # ── 3b. Fill fields deterministically, defer submit ──
        original_submit_ref = mapping.submit_ref
        original_skip_submit = mapping.skip_submit
        mapping.skip_submit = True

        fill_result = await browser_dom_filler.fill_form(
            mapping, scoped_tree, page,
        )
        all_fill_results: list[browser_dom_filler.FillResult] = list(
            fill_result.fills,
        )
        mapping.skip_submit = original_skip_submit

        history.append({
            "attempt": attempt,
            "fills_total": len(fill_result.fills),
            "fills_succeeded": sum(1 for f in fill_result.fills if f.succeeded),
            "fills_failed": sum(1 for f in fill_result.fills if not f.succeeded),
            "submit_clicked": False,
        })

        # ── 3c. Fail early if fills failed ──
        if not fill_result.all_succeeded:
            failed = [f for f in fill_result.fills if not f.succeeded]
            names = [f.field_name or f.ref for f in failed]
            logger.warning(f"[DOM_FORM_FILL] fills failed: {names}")
            return DomTaskResult(
                success=False, reason="fills_failed",
                loops_used=attempt + 1,
                final_summary=f"Could not fill: {', '.join(names)}.",
                history=history,
            )

        # ── 3d. Re-perceive for newly-enabled dependent fields ──
        # Track elements that were visible but DISABLED before fill.
        # After fill, if any of these become enabled, they're cascading
        # dependents (e.g. City enabled after State is selected).
        # Match by (name, role) — refs are unstable across DOM mutations.
        pre_fill_disabled = {
            (e.name, e.role) for e in scoped_tree.elements
            if e.visible and not e.enabled
        }

        current_tree = scoped_tree
        if fill_result.fills and pre_fill_disabled:
            await asyncio.sleep(0.3)
            browser_dom.invalidate_tree_cache(page)
            try:
                fresh_full = await browser_dom.read_page_dom(
                    page, filter="interactive",
                )
                fresh_scope = _select_target_form(
                    fresh_full.elements, goal,
                )
                if fresh_scope is not None:
                    fid, fels = fresh_scope
                    portal = [
                        e for e in fresh_full.elements
                        if not e.form_id and e.role in _PORTAL_ROLES
                    ]
                    if portal:
                        fels = list(fels) + portal
                    fresh_scoped = _scope_tree_to_elements(
                        fresh_full, fels,
                    )
                else:
                    fresh_scoped = fresh_full
                current_tree = fresh_scoped

                newly_enabled = [
                    e for e in fresh_scoped.elements
                    if e.visible and e.enabled
                    and (e.name, e.role) in pre_fill_disabled
                ]

                if newly_enabled:
                    logger.info(
                        f"[DOM_FORM_FILL] {len(newly_enabled)} newly-enabled "
                        f"field(s): {[e.name for e in newly_enabled]}"
                    )
                    dep_mapping = await browser_dom_mapper.map_goal_to_fields(
                        goal, fresh_scoped, feedback=feedback,
                    )
                    already_filled_names = {
                        f.field_name for f in all_fill_results
                    }
                    dep_new = [
                        f for f in dep_mapping.fills
                        if f.field_name not in already_filled_names
                    ]
                    if dep_new:
                        dep_obj = browser_dom_mapper.FormMapping(
                            fills=dep_new,
                            submit_ref="",
                            thinking=dep_mapping.thinking,
                            skip_submit=True,
                        )
                        dep_result = await browser_dom_filler.fill_form(
                            dep_obj, fresh_scoped, page,
                        )
                        all_fill_results.extend(dep_result.fills)

                        dep_failed = [
                            f for f in dep_result.fills
                            if not f.succeeded
                        ]
                        if dep_failed:
                            names = [
                                f.field_name or f.ref for f in dep_failed
                            ]
                            logger.warning(
                                f"[DOM_FORM_FILL] dependent fills "
                                f"failed: {names}"
                            )
                            history[-1]["fills_total"] = len(all_fill_results)
                            history[-1]["fills_failed"] = len(dep_failed)
                            return DomTaskResult(
                                success=False, reason="fills_failed",
                                loops_used=attempt + 1,
                                final_summary=(
                                    f"Could not fill: {', '.join(names)}."
                                ),
                                history=history,
                            )
            except Exception as exc:
                logger.warning(
                    f"[DOM_FORM_FILL] dependent field check failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        # ── 3e. Submit ──
        submit_clicked = False
        if original_submit_ref and not original_skip_submit:
            submit_loc = current_tree.ref_to_locator.get(original_submit_ref)
            if not submit_loc:
                for e in current_tree.elements:
                    if (e.role == "button"
                            and "submit" in e.name.lower()
                            and current_tree.ref_to_locator.get(e.ref)):
                        submit_loc = current_tree.ref_to_locator[e.ref]
                        break
            if submit_loc:
                try:
                    await submit_loc.click(timeout=10_000)
                    submit_clicked = True
                except Exception as exc:
                    logger.warning(
                        f"[DOM_FORM_FILL] submit click failed: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    )
                    return DomTaskResult(
                        success=False, reason="submit_failed",
                        loops_used=attempt + 1,
                        final_summary="Could not click submit.",
                        history=history,
                    )
            else:
                return DomTaskResult(
                    success=False, reason="submit_failed",
                    loops_used=attempt + 1,
                    final_summary="Could not locate submit button.",
                    history=history,
                )

        history[-1]["submit_clicked"] = submit_clicked
        history[-1]["fills_total"] = len(all_fill_results)
        history[-1]["fills_succeeded"] = sum(
            1 for f in all_fill_results if f.succeeded
        )

        if not submit_clicked:
            return DomTaskResult(
                success=True, reason="completed_no_submit",
                loops_used=attempt + 1,
                final_summary="Fields filled.",
                history=history,
            )

        # ── 3f. Post-submit verification ──
        submit_baseline_url = full_tree.url
        submit_baseline_refs = {
            e.ref for e in current_tree.elements if e.visible
        }

        await asyncio.sleep(0.3)
        browser_dom.invalidate_tree_cache(page)

        try:
            verify_tree = await browser_dom.read_page_dom(
                page, filter="interactive",
            )
        except Exception:
            return DomTaskResult(
                success=True, reason="completed",
                loops_used=attempt + 1,
                final_summary="Form submitted.",
                history=history,
            )

        navigated, nav_reason = _post_submit_navigated(
            verify_tree, submit_baseline_url, submit_baseline_refs,
        )
        if navigated:
            history.append({
                "attempt": attempt, "result": "navigation_success",
                "detail": nav_reason,
            })
            return DomTaskResult(
                success=True, reason="completed",
                loops_used=attempt + 1,
                final_summary="Form submitted.",
                history=history,
            )

        if not verify_tree.validation_errors:
            return DomTaskResult(
                success=True, reason="completed",
                loops_used=attempt + 1,
                final_summary="Form submitted.",
                history=history,
            )

        # Check user-pinned value rejection
        if pinned_value_by_name:
            name_by_ref = {
                e.ref: (e.name or "") for e in verify_tree.elements
            }
            rejected_pinned_names: list[str] = []
            seen_norms: set[str] = set()
            for ve in verify_tree.validation_errors:
                if not ve.field_ref:
                    continue
                raw_name = name_by_ref.get(ve.field_ref, "")
                norm = _normalize_field_name(raw_name)
                if norm and norm in pinned_value_by_name and norm not in seen_norms:
                    seen_norms.add(norm)
                    rejected_pinned_names.append(raw_name)
            if rejected_pinned_names:
                summary = _format_pinned_rejection_summary(
                    pinned_value_by_name, rejected_pinned_names,
                )
                return DomTaskResult(
                    success=False, reason="user_value_rejected",
                    loops_used=attempt + 1,
                    final_summary=summary,
                    history=history,
                )

        if attempt == _MAX_FORM_FILL_ATTEMPTS - 1:
            return DomTaskResult(
                success=False, reason="validation_unresolved",
                loops_used=attempt + 1,
                final_summary=_format_unresolved_for_summary(
                    verify_tree.validation_errors,
                    verify_tree.elements,
                    goal=goal,
                ),
                history=history,
            )

        feedback = _format_validation_for_planner(
            verify_tree.validation_errors, verify_tree.elements,
        )
        browser_dom.invalidate_tree_cache(page)
        try:
            scoped_tree = await browser_dom.read_page_dom(
                page, filter="interactive",
            )
        except Exception:
            pass
        logger.info(
            f"[DOM_FORM_FILL] attempt {attempt + 1}: "
            f"{len(verify_tree.validation_errors)} validation error(s), "
            f"remapping"
        )

    return DomTaskResult(
        success=False, reason="max_attempts",
        loops_used=_MAX_FORM_FILL_ATTEMPTS,
        final_summary="Could not complete form fill.",
        history=history,
    )
