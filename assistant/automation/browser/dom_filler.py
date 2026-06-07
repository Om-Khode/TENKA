"""
dom_filler.py — Deterministic per-widget form filler (Option B).

Given a FormMapping (from dom_mapper), fills each field using a fixed
algorithm based on widget type. No LLM calls — just Playwright operations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from . import dom as browser_dom
from .dom_mapper import FillInstruction, FormMapping

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────


@dataclass
class FillResult:
    """Outcome of filling one field."""
    ref: str
    field_name: str
    intended_value: str
    observed_value: str
    succeeded: bool
    error: str = ""


@dataclass
class FormFillResult:
    """Outcome of filling all fields + optional submit."""
    fills: list[FillResult]
    submit_clicked: bool = False
    submit_error: str = ""

    @property
    def all_succeeded(self) -> bool:
        return all(f.succeeded for f in self.fills)


# ─── Widget Classification ──────────────────────────────────────────────


def classify_widget(e: browser_dom.ElementInfo) -> str:
    """Determine the widget type from element metadata."""
    if e.role == "radio":
        return "radio"
    if e.role == "checkbox":
        return "checkbox"
    if e.role == "button" or e.tag == "button":
        return "button"
    if e.role == "combobox":
        if e.tag == "select" or e.options:
            return "native_select"
        if e.autocomplete:
            return "autocomplete_combobox"
        return "click_combobox"
    return "textbox"


_DEFAULT_TIMEOUT_MS = 10_000


# ─── Value Matching ─────────────────────────────────────────────────────


def _values_match(observed: str, expected: str) -> bool:
    """Lenient value comparison: whitespace-collapsed, case-insensitive."""
    o = " ".join(observed.split()).lower()
    e = " ".join(expected.split()).lower()
    return o == e or o in e or e in o


# ─── Simple Widget Fillers ──────────────────────────────────────────────


async def fill_textbox(
    locator: Any,
    elem: browser_dom.ElementInfo,
    value: str,
    *,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> FillResult:
    """Fill a text input and read back to verify."""
    try:
        await locator.fill(value, timeout=timeout_ms)
    except Exception as e:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="",
            succeeded=False, error=f"fill failed: {type(e).__name__}: {str(e)[:200]}",
        )
    try:
        observed = await locator.input_value(timeout=timeout_ms)
    except Exception:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="",
            succeeded=True, error="read-back failed but fill ran",
        )
    if _values_match(observed, value):
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value=observed,
            succeeded=True,
        )
    return FillResult(
        ref=elem.ref, field_name=elem.name,
        intended_value=value, observed_value=observed,
        succeeded=False, error=f"read-back mismatch: expected {value!r} got {observed!r}",
    )


async def fill_radio(
    locator: Any,
    elem: browser_dom.ElementInfo,
    value: str,
    *,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> FillResult:
    """Click a radio button."""
    try:
        await locator.click(timeout=timeout_ms)
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value=value,
            succeeded=True,
        )
    except Exception as e:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="",
            succeeded=False, error=f"click failed: {type(e).__name__}: {str(e)[:200]}",
        )


async def fill_checkbox(
    locator: Any,
    elem: browser_dom.ElementInfo,
    value: str,
    *,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> FillResult:
    """Check or uncheck a checkbox based on desired value."""
    want_checked = value.lower().strip() in ("check", "true", "yes", "1")
    try:
        is_checked = await locator.is_checked(timeout=timeout_ms)
    except Exception as e:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="",
            succeeded=False, error=f"is_checked failed: {type(e).__name__}",
        )
    if is_checked == want_checked:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="checked" if is_checked else "unchecked",
            succeeded=True,
        )
    try:
        if want_checked:
            await locator.check(timeout=timeout_ms)
        else:
            await locator.uncheck(timeout=timeout_ms)
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="checked" if want_checked else "unchecked",
            succeeded=True,
        )
    except Exception as e:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="",
            succeeded=False, error=f"toggle failed: {type(e).__name__}: {str(e)[:200]}",
        )


async def fill_native_select(
    locator: Any,
    elem: browser_dom.ElementInfo,
    value: str,
    *,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> FillResult:
    """Pick an option from a native <select>."""
    try:
        await locator.select_option(label=value, timeout=timeout_ms)
    except Exception as e_label:
        try:
            await locator.select_option(value=value, timeout=timeout_ms)
        except Exception as e_value:
            return FillResult(
                ref=elem.ref, field_name=elem.name,
                intended_value=value, observed_value="",
                succeeded=False,
                error=f"select_option failed: {type(e_label).__name__}, {type(e_value).__name__}",
            )
    try:
        observed = await locator.evaluate(
            "el => { "
            "const idx = el.selectedIndex; "
            "if (idx < 0) return ''; "
            "const opt = el.options[idx]; "
            "return (opt && (opt.text || opt.value || '')) || ''; "
            "}"
        )
    except Exception:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="",
            succeeded=True, error="select read-back failed but action ran",
        )
    if isinstance(observed, str) and _values_match(observed, value):
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value=observed,
            succeeded=True,
        )
    return FillResult(
        ref=elem.ref, field_name=elem.name,
        intended_value=value, observed_value=str(observed or ""),
        succeeded=False,
        error=f"select read-back mismatch: expected {value!r} got {observed!r}",
    )


# ─── Combobox Fillers ───────────────────────────────────────────────────


_COMBOBOX_OPTION_WAIT_MS = 500
_COMBOBOX_MAX_RETRIES = 2


def _find_matching_option(
    tree: browser_dom.PageDomTree,
    target_value: str,
) -> tuple[str, Any] | None:
    """Find a portal-rendered option element whose name matches the target."""
    target_lower = target_value.lower().strip()
    for elem in tree.elements:
        if elem.role != "option":
            continue
        if elem.name.lower().strip() == target_lower:
            loc = tree.ref_to_locator.get(elem.ref)
            if loc:
                return elem.ref, loc
    for elem in tree.elements:
        if elem.role != "option":
            continue
        if target_lower in elem.name.lower().strip() or elem.name.lower().strip() in target_lower:
            loc = tree.ref_to_locator.get(elem.ref)
            if loc:
                return elem.ref, loc
    return None


async def fill_combobox(
    locator: Any,
    elem: browser_dom.ElementInfo,
    value: str,
    *,
    page: Any,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> FillResult:
    """
    Fill a custom combobox: type prefix (or click to open) → wait for
    portal options → click matching option. Deterministic — no LLM.
    """
    is_autocomplete = bool(elem.autocomplete)

    # Step 1: Open the dropdown
    try:
        if is_autocomplete:
            prefix = value[:4] if len(value) > 4 else value
            await locator.fill(prefix, timeout=timeout_ms)
        else:
            await locator.click(timeout=timeout_ms)
    except Exception as e:
        return FillResult(
            ref=elem.ref, field_name=elem.name,
            intended_value=value, observed_value="",
            succeeded=False,
            error=f"open combobox failed: {type(e).__name__}: {str(e)[:200]}",
        )

    # Step 2: Wait for options to render, then find matching option
    for attempt in range(_COMBOBOX_MAX_RETRIES + 1):
        await asyncio.sleep(_COMBOBOX_OPTION_WAIT_MS / 1000)
        browser_dom.invalidate_tree_cache(page)
        try:
            option_tree = await browser_dom.read_page_dom(page, filter="interactive")
        except Exception as e:
            return FillResult(
                ref=elem.ref, field_name=elem.name,
                intended_value=value, observed_value="",
                succeeded=False,
                error=f"re-perceive failed: {type(e).__name__}",
            )

        match = _find_matching_option(option_tree, value)
        if match:
            opt_ref, opt_loc = match
            try:
                await opt_loc.click(timeout=timeout_ms)
                return FillResult(
                    ref=elem.ref, field_name=elem.name,
                    intended_value=value, observed_value=value,
                    succeeded=True,
                )
            except Exception as e:
                return FillResult(
                    ref=elem.ref, field_name=elem.name,
                    intended_value=value, observed_value="",
                    succeeded=False,
                    error=f"option click failed: {type(e).__name__}: {str(e)[:200]}",
                )

    return FillResult(
        ref=elem.ref, field_name=elem.name,
        intended_value=value, observed_value="",
        succeeded=False,
        error=f"no matching option found for {value!r} after {_COMBOBOX_MAX_RETRIES + 1} attempts",
    )


async def fill_combobox_multi(
    locator: Any,
    elem: browser_dom.ElementInfo,
    values: list[str],
    *,
    page: Any,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> list[FillResult]:
    """Fill a multi-value combobox: one value at a time, re-opening each time."""
    results: list[FillResult] = []
    for val in values:
        result = await fill_combobox(locator, elem, val, page=page, timeout_ms=timeout_ms)
        results.append(result)
        if not result.succeeded:
            logger.warning(
                f"[DOM_FILLER] combobox multi-value fill failed on {val!r}: {result.error}"
            )
    return results


# ─── Top-Level Form Dispatcher ──────────────────────────────────────────


async def fill_form(
    mapping: FormMapping,
    tree: browser_dom.PageDomTree,
    page: Any,
    *,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> FormFillResult:
    """
    Fill all fields from a FormMapping, then optionally click submit.
    Routes each fill to the correct widget handler based on ElementInfo.
    """
    ref_to_elem = {e.ref: e for e in tree.elements}
    fills: list[FillResult] = []

    for instr in mapping.fills:
        elem = ref_to_elem.get(instr.ref)
        locator = tree.ref_to_locator.get(instr.ref)

        if not locator:
            fills.append(FillResult(
                ref=instr.ref, field_name=instr.field_name,
                intended_value=str(instr.value), observed_value="",
                succeeded=False, error="no locator found for ref",
            ))
            continue

        if not elem:
            fills.append(FillResult(
                ref=instr.ref, field_name=instr.field_name,
                intended_value=str(instr.value), observed_value="",
                succeeded=False, error="no element metadata for ref",
            ))
            continue

        widget = classify_widget(elem)
        logger.info(f"[DOM_FILLER] filling {elem.name!r} ({widget}) with {instr.value!r}")

        if widget == "textbox":
            fills.append(await fill_textbox(locator, elem, str(instr.value), timeout_ms=timeout_ms))
        elif widget == "radio":
            fills.append(await fill_radio(locator, elem, str(instr.value), timeout_ms=timeout_ms))
        elif widget == "checkbox":
            fills.append(await fill_checkbox(locator, elem, str(instr.value), timeout_ms=timeout_ms))
        elif widget == "native_select":
            fills.append(await fill_native_select(locator, elem, str(instr.value), timeout_ms=timeout_ms))
        elif widget in ("autocomplete_combobox", "click_combobox"):
            if isinstance(instr.value, list):
                multi_results = await fill_combobox_multi(
                    locator, elem, instr.value, page=page, timeout_ms=timeout_ms,
                )
                fills.extend(multi_results)
            else:
                fills.append(await fill_combobox(
                    locator, elem, str(instr.value), page=page, timeout_ms=timeout_ms,
                ))
        else:
            fills.append(FillResult(
                ref=instr.ref, field_name=instr.field_name,
                intended_value=str(instr.value), observed_value="",
                succeeded=False, error=f"unknown widget type: {widget}",
            ))

    # Submit
    submit_clicked = False
    submit_error = ""
    if mapping.submit_ref and not mapping.skip_submit:
        submit_loc = tree.ref_to_locator.get(mapping.submit_ref)
        if submit_loc:
            try:
                await submit_loc.click(timeout=timeout_ms)
                submit_clicked = True
            except Exception as e:
                submit_error = f"submit click failed: {type(e).__name__}: {str(e)[:200]}"
        else:
            submit_error = "no locator for submit ref"

    return FormFillResult(
        fills=fills,
        submit_clicked=submit_clicked,
        submit_error=submit_error,
    )
