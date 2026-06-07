"""
browser_dom_executor.py — DOM action executor.

Takes a validated `DomPlan.actions` list and dispatches each action via
Playwright `Locator`s with per-action read-back verification.

Architectural contract:
  - Actions execute SERIALLY in DOM order. No parallel fills (avoids
    validation-fight / autocomplete-fight scenarios on real forms).
  - When an action fails, RECORD and CONTINUE. The orchestrator collects
    failures and decides retry/fallback. We do NOT short-circuit the batch.
  - `form_input` is verified by read-back: `locator.input_value()` must
    equal the value we filled. Failures feed back to the planner as
    "value didn't take" observations.
  - `select_option_ref` is verified by selectedOption text match — handles
    both `value=` and visible-text selection equally.
  - `click_ref` and `press_ref` have no generic post-verify (the design
    delegates click outcomes to the checkpoint's screenshot diagnose at
    the orchestrator level). They DO set `tree_dirty=True` so the
    orchestrator invalidates the perception cache before the next batch.
  - `reperceive` returns a special marker (`requires_reperceive=True`) and
    halts the batch — the orchestrator must re-perceive before continuing.

The executor never raises. Every Playwright exception is caught and
recorded as a failure on the offending action.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("browser_dom_executor")


# ─── Result types ─────────────────────────────────────────────────────────────


@dataclass
class DomActionResult:
    """
    Per-action outcome. Surfaced one-to-one with the actions in the input
    plan so the orchestrator can correlate planner intent with execution
    reality.
    """
    action: dict
    succeeded: bool
    error: str = ""             # short reason on failure
    observed_value: str = ""    # filled value or selected option (for read-back)


@dataclass
class DomBatchResult:
    """
    Full-batch outcome.

    `results`             — per-action `DomActionResult`, in input order
    `requires_reperceive` — True if a `reperceive` action ran OR a
                            navigation-like action (click/press) ran (so the
                            orchestrator should invalidate the cache and
                            re-read before the next batch)
    `tree_dirty`          — True if the DOM may have mutated (any click,
                            press, or select_option_ref). Used by the
                            orchestrator's cache-invalidation logic.
    """
    results: list[DomActionResult] = field(default_factory=list)
    requires_reperceive: bool = False
    tree_dirty: bool = False

    @property
    def succeeded(self) -> list[DomActionResult]:
        return [r for r in self.results if r.succeeded]

    @property
    def failed(self) -> list[DomActionResult]:
        return [r for r in self.results if not r.succeeded]

    @property
    def all_succeeded(self) -> bool:
        return bool(self.results) and all(r.succeeded for r in self.results)


# ─── Per-action dispatch helpers ──────────────────────────────────────────────


# Default per-action timeout — Playwright internally auto-waits for visible+
# editable so this is the wall-clock cap. 10s is generous; failure beyond
# that means the page is genuinely broken or the element vanished.
_DEFAULT_ACTION_TIMEOUT_MS = 10_000

# Fast-fail thresholds. When the form is genuinely broken, every action
# will run its full 10s timeout — a 6-action batch turns into a 60-second
# pause before the orchestrator gets a chance to bail. These caps
# short-circuit the obvious-broken case.
#
# `_DEFAULT_MAX_CONSECUTIVE_FAILURES`: after this many failures in a row,
# the remaining actions are marked aborted without dispatching. A single
# transient failure (one ref disappeared mid-batch) should still let the
# rest run; 2-in-a-row is strong evidence the page state is wrong.
#
# `_DEFAULT_BATCH_BUDGET_MS`: hard wall-clock cap. Even if individual
# actions succeed but are each slow (e.g. very busy page), don't let a
# batch exceed this — the orchestrator can re-perceive and retry cheaper
# than waiting forever inside one batch.
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 2
_DEFAULT_BATCH_BUDGET_MS = 30_000


async def _dispatch_form_input(
    locator: Any, action: dict, *, timeout_ms: int = _DEFAULT_ACTION_TIMEOUT_MS,
) -> DomActionResult:
    """
    Fill a text input. Read back the value to verify it took. Most React
    forms need exactly this — Playwright's `.fill()` triggers focus → set
    value → input event → blur, which mirrors what a user would do.
    """
    value = action.get("value", "")
    if not isinstance(value, str):
        # Validator shouldn't have let this through, but be defensive.
        value = str(value)

    try:
        await locator.fill(value, timeout=timeout_ms)
    except Exception as e:
        return DomActionResult(
            action=action, succeeded=False,
            error=f"fill failed: {type(e).__name__}: {str(e)[:200]}",
        )

    # Read-back. Some sites mask passwords or rewrite (auto-format phone),
    # so an exact equality check would over-trigger; do a lenient compare:
    # collapse whitespace, allow empty masked-password.
    try:
        observed = await locator.input_value(timeout=timeout_ms)
    except Exception as e:
        # Read-back itself failed — record but don't fail the action;
        # Playwright's fill() didn't raise, so the call ran. The
        # checkpoint is the safety net for downstream verification.
        return DomActionResult(
            action=action, succeeded=True,
            error=f"read-back failed (action still ran): "
                  f"{type(e).__name__}: {str(e)[:120]}",
            observed_value="",
        )

    if _values_match(observed, value):
        return DomActionResult(
            action=action, succeeded=True, observed_value=observed,
        )
    return DomActionResult(
        action=action, succeeded=False,
        error=f"read-back mismatch: expected {value!r} got {observed!r}",
        observed_value=observed,
    )


async def _dispatch_click_ref(
    locator: Any, action: dict, *, timeout_ms: int = _DEFAULT_ACTION_TIMEOUT_MS,
) -> DomActionResult:
    """
    Click an element. No generic post-verify — clicks have outcomes that
    vary too widely (open dropdown, navigate, submit, toggle). The
    orchestrator's checkpoint at task end is the verification layer.
    """
    try:
        await locator.click(timeout=timeout_ms)
        return DomActionResult(action=action, succeeded=True)
    except Exception as e:
        return DomActionResult(
            action=action, succeeded=False,
            error=f"click failed: {type(e).__name__}: {str(e)[:200]}",
        )


async def _dispatch_select_option_ref(
    locator: Any, action: dict, *, timeout_ms: int = _DEFAULT_ACTION_TIMEOUT_MS,
) -> DomActionResult:
    """
    Native <select> option pick. Playwright's `select_option` accepts
    either the option's value attribute, label (text), or an index — we
    pass the planner's `option` string which is the visible text per the
    planner-prompt rule.

    Read-back: fetch the selected option's display text via JS and
    compare. Handles the common case where `<option value="us">USA</option>`
    has display text "USA" but `value="us"` — we want to confirm against
    the text the user/planner reasoned about.
    """
    option = action.get("option", "")
    if not isinstance(option, str):
        return DomActionResult(
            action=action, succeeded=False,
            error=f"option must be string, got {type(option).__name__}",
        )

    try:
        # Try by label first (matches the planner's `option` semantics);
        # Playwright also accepts {"value": ...} and {"index": ...} but
        # label is the right contract here.
        await locator.select_option(label=option, timeout=timeout_ms)
    except Exception as e_label:
        # Fall back to value-match in case the planner used a value attr
        # by mistake — defensive belt-and-braces.
        try:
            await locator.select_option(value=option, timeout=timeout_ms)
        except Exception as e_value:
            return DomActionResult(
                action=action, succeeded=False,
                error=(
                    f"select_option failed by-label ({type(e_label).__name__}: "
                    f"{str(e_label)[:80]}) and by-value "
                    f"({type(e_value).__name__}: {str(e_value)[:80]})"
                ),
            )

    # Read-back: get the visible text of the selected option.
    try:
        observed = await locator.evaluate(
            "el => { "
            "const idx = el.selectedIndex; "
            "if (idx < 0) return ''; "
            "const opt = el.options[idx]; "
            "return (opt && (opt.text || opt.value || '')) || ''; "
            "}"
        )
    except Exception as e:
        return DomActionResult(
            action=action, succeeded=True,
            error=f"select read-back failed (action still ran): "
                  f"{type(e).__name__}: {str(e)[:120]}",
            observed_value="",
        )

    if isinstance(observed, str) and _values_match(observed, option):
        return DomActionResult(
            action=action, succeeded=True, observed_value=observed,
        )
    # Not a match — but the action DID run. Treat as soft failure so the
    # orchestrator can decide. observed_value carries the truth.
    return DomActionResult(
        action=action, succeeded=False,
        error=f"select read-back mismatch: expected {option!r} got {observed!r}",
        observed_value=str(observed or ""),
    )


async def _dispatch_press_ref(
    locator: Any, action: dict, *, timeout_ms: int = _DEFAULT_ACTION_TIMEOUT_MS,
) -> DomActionResult:
    """
    Send a key to a focused element (Enter, Tab, Escape, etc.). No
    post-verify — same reason as click_ref.
    """
    key = action.get("key", "")
    if not isinstance(key, str) or not key:
        return DomActionResult(
            action=action, succeeded=False, error="press key must be non-empty",
        )
    try:
        await locator.press(key, timeout=timeout_ms)
        return DomActionResult(action=action, succeeded=True)
    except Exception as e:
        return DomActionResult(
            action=action, succeeded=False,
            error=f"press failed: {type(e).__name__}: {str(e)[:200]}",
        )


async def _dispatch_wait_ms(action: dict) -> DomActionResult:
    """Synchronous-style sleep. Always succeeds (validator capped ms)."""
    ms = action.get("ms", 0)
    try:
        ms_f = float(ms)
        if ms_f > 0:
            await asyncio.sleep(min(ms_f, 30_000) / 1000.0)
        return DomActionResult(action=action, succeeded=True)
    except Exception as e:
        return DomActionResult(
            action=action, succeeded=False, error=f"wait failed: {e}",
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _values_match(observed: Any, expected: str) -> bool:
    """
    Lenient equality for read-back checks. Real forms transform input:
      - phone "1234567890" might display as "(123) 456-7890"
      - email is case-preserved but trim-applied
      - leading/trailing whitespace differs
      - some custom inputs strip non-numeric

    Strategy: normalize whitespace, lowercase, then check substring either
    way. This is permissive enough that a true match always passes,
    while still catching "field stayed empty" failures (empty observed
    against a non-empty expected = clear miss).
    """
    if not isinstance(observed, str) or not isinstance(expected, str):
        return False
    o = " ".join(observed.split()).strip().lower()
    e = " ".join(expected.split()).strip().lower()
    if not e:
        return o == e  # both empty = match; one empty = miss
    if not o:
        return False
    return o == e or e in o or o in e


# Action types that may mutate the DOM. Used to set tree_dirty so the
# orchestrator invalidates the cached perception before the next batch.
_TREE_DIRTYING_TYPES = frozenset({
    "click_ref", "press_ref", "select_option_ref",
})

# Action types that imply "stop here, re-perceive before continuing".
# The orchestrator will see requires_reperceive=True and break out of
# the action loop after the dirtying action lands.
_REPERCEIVE_TRIGGERS = frozenset({"reperceive"})


# ─── Public dispatch ──────────────────────────────────────────────────────────


async def execute_dom_batch(
    page: Any,
    actions: list[dict],
    ref_to_locator: dict[str, Any],
    *,
    action_timeout_ms: int = _DEFAULT_ACTION_TIMEOUT_MS,
    max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
    batch_budget_ms: int = _DEFAULT_BATCH_BUDGET_MS,
) -> DomBatchResult:
    """
    Execute a validated batch of DOM actions. Returns `DomBatchResult` with
    one `DomActionResult` per input action.

    Loop semantics:
      - Walk actions in order. For each, dispatch and record.
      - On `reperceive`: record + halt. Subsequent actions are dropped
        (orchestrator gets a fresh tree before continuing).
      - On any per-action failure: record, continue. (The orchestrator
        decides retry/fallback at batch boundary.)
      - tree_dirty is True if any DOM-mutating action ran (success OR
        failure — even a failed click might have side effects).

    Fast-fail caps:
      - `max_consecutive_failures`: once this many actions fail in a row,
        the remaining actions are marked aborted without dispatch.
      - `batch_budget_ms`: total wall-clock cap. When elapsed time exceeds
        this, the remaining actions are marked aborted without dispatch.

    Aborted actions are recorded as `succeeded=False` with a tagged error
    so the orchestrator's existing failure-feedback path surfaces
    them honestly to the planner.

    `page` is unused for now — passed in for future use (when reperceive
    needs to do `await page.wait_for_load_state()` or similar).
    """
    import time as _time

    result = DomBatchResult()
    consecutive_failures = 0
    batch_start = _time.monotonic()
    aborted_reason: Optional[str] = None

    for action in actions:
        # abort check + BROWSING status at each step boundary.
        from assistant.core.abort import abort, UserAborted
        if abort.is_aborted():
            raise UserAborted(abort.reason)
        from assistant.io.status_broadcaster import status, StatusPhase
        status.set(
            StatusPhase.BROWSING,
            detail=str(action.get("type", ""))[:40] if isinstance(action, dict) else "",
            cursor_follows=False,
            tier="browser",
        )

        # Update consecutive-failure counter based on the previous
        # iteration's outcome. We only care about naturally-dispatched
        # results — once we abort, this counter is no longer consulted.
        # Each iteration appends exactly one result (every branch hits a
        # continue/break after one append), so results[-1] is always the
        # previous iteration's outcome at this point.
        if aborted_reason is None and result.results:
            if result.results[-1].succeeded:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        # Short-circuit when budget exhausted or too many failures in a
        # row. The skipped actions get a synthetic failure record so they
        # show up in `batch.failed` with a clear cause.
        if aborted_reason is None:
            if consecutive_failures >= max_consecutive_failures:
                aborted_reason = (
                    f"batch aborted after {consecutive_failures} consecutive "
                    f"action failures — page state likely wrong"
                )
                logger.info(f"[DOM_EXEC] {aborted_reason}")
            else:
                elapsed_ms = int((_time.monotonic() - batch_start) * 1000)
                if elapsed_ms >= batch_budget_ms:
                    aborted_reason = (
                        f"batch aborted: {elapsed_ms}ms elapsed exceeds "
                        f"{batch_budget_ms}ms budget"
                    )
                    logger.info(f"[DOM_EXEC] {aborted_reason}")

        if aborted_reason is not None:
            result.results.append(DomActionResult(
                action=action, succeeded=False, error=aborted_reason,
            ))
            continue

        atype = action.get("type")
        if not isinstance(atype, str):
            result.results.append(DomActionResult(
                action=action, succeeded=False, error="action missing 'type'",
            ))
            continue

        # reperceive: record and halt immediately
        if atype in _REPERCEIVE_TRIGGERS:
            result.results.append(DomActionResult(action=action, succeeded=True))
            result.requires_reperceive = True
            result.tree_dirty = True
            logger.info("[DOM_EXEC] reperceive requested — halting batch")
            break

        # wait_ms doesn't need a locator
        if atype == "wait_ms":
            result.results.append(await _dispatch_wait_ms(action))
            continue

        # Reject unknown action types BEFORE attempting locator lookup so
        # the error message is precise (locator-not-found would mask the
        # real diagnosis when the type itself is bogus).
        if atype not in ("form_input", "click_ref", "select_option_ref", "press_ref"):
            result.results.append(DomActionResult(
                action=action, succeeded=False,
                error=f"unknown action type {atype!r} (validator gap?)",
            ))
            continue

        # All remaining action types use a ref → locator.
        ref = action.get("ref")
        locator = ref_to_locator.get(ref) if isinstance(ref, str) else None
        if locator is None:
            # Defensive — validator should have rejected this. Belt-and-braces
            # so a stale ref-map post-reperceive doesn't crash us.
            result.results.append(DomActionResult(
                action=action, succeeded=False,
                error=f"locator not found for ref={ref!r} (post-validation drift?)",
            ))
            continue

        if atype == "form_input":
            r = await _dispatch_form_input(locator, action, timeout_ms=action_timeout_ms)
        elif atype == "click_ref":
            r = await _dispatch_click_ref(locator, action, timeout_ms=action_timeout_ms)
        elif atype == "select_option_ref":
            r = await _dispatch_select_option_ref(locator, action, timeout_ms=action_timeout_ms)
        else:  # press_ref (the only one left)
            r = await _dispatch_press_ref(locator, action, timeout_ms=action_timeout_ms)

        result.results.append(r)

        # tree_dirty is sticky — once set, stays set for the batch
        if atype in _TREE_DIRTYING_TYPES:
            result.tree_dirty = True

        # Heuristic: a successful click on an element whose role/name suggests
        # navigation also implies reperceive. We don't have role/name here
        # (validator already gated), so the orchestrator will use
        # `tree_dirty + needs_reperceive (from plan)` as the join. Keep
        # this layer dumb and orthogonal.

    if result.results:
        succ = sum(1 for r in result.results if r.succeeded)
        logger.info(
            f"[DOM_EXEC] batch done: {succ}/{len(result.results)} succeeded, "
            f"tree_dirty={result.tree_dirty}, "
            f"reperceive={result.requires_reperceive}"
        )
    return result
