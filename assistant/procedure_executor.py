"""
procedure_executor.py — Robust procedure execution.

Improvements over the initial implementation:
  - Named {slot} placeholders: extracted from invocation via 8b LLM (1 call, only when needed)
  - Wait-for-element pre-flight before every click/focus step
  - Retry with exponential backoff (2 retries per step, zero LLM cost)
  - Self-heal on total failure: 1 vision LLM call to verify/skip/abort
  - Per-step result reporting

Public API:
    run_procedure(proc: dict, original_text: str) -> str
"""

import asyncio
import json
import logging
import re
from datetime import date, datetime
from typing import Optional

from . import config, procedures

logger = logging.getLogger("proc_exec")

_MAX_RETRIES  = 2
_RETRY_DELAYS = [0.8, 1.6]   # seconds between retry attempts

_SLOT_RE      = re.compile(r'\{(\w+)\}')
_BUILTIN_VARS = frozenset({"user_input", "date", "time", "clipboard"})
_BROWSER_NAMES = config.BROWSER_NAMES


def _is_browser_name(name: str) -> bool:
    """Check if name is a browser (canonical name or alias via KNOWN_APPS)."""
    if name in _BROWSER_NAMES:
        return True
    from .core.known_apps import get_category
    return get_category(name) == "browser"


def _default_browser() -> str:
    """Return the user's preferred browser name, falling back to 'chrome'."""
    try:
        from . import preferences
        pref = preferences.get_preference("default_browser")
        if pref and pref.get("value"):
            return pref["value"].lower().strip()
    except Exception:
        pass
    return "chrome"


# ─── Window Context ──────────────────────────────────────────────────────────


async def _ensure_foreground(window_title: str) -> bool:
    """Force a window to foreground using the Win32 Alt-key trick.

    Plain SetForegroundWindow fails silently when another process holds
    the foreground lock.  Sending a synthetic Alt press before the call
    convinces Windows to hand over focus.

    No-ops if the window is already active — the Alt trick can disrupt
    focus on inner elements (address bar, search box) already focused
    inside the target window.
    """
    try:
        import pygetwindow as gw
        import ctypes

        active = gw.getActiveWindow()
        if active is not None and window_title.lower() in (active.title or "").lower():
            return True

        matches = [w for w in gw.getAllWindows()
                   if window_title.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return False

        hwnd = matches[0]._hWnd
        user32 = ctypes.windll.user32
        user32.keybd_event(0x12, 0, 0, 0)   # VK_MENU down
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)       # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        user32.keybd_event(0x12, 0, 2, 0)   # VK_MENU up

        await asyncio.sleep(0.3)
        active = gw.getActiveWindow()
        focused = active is not None and window_title.lower() in (active.title or "").lower()
        if not focused:
            logger.warning(f"[PROC] Focus verify failed for '{window_title}', "
                           f"active: {active.title if active else 'None'}")
        return focused
    except Exception as e:
        logger.debug(f"[PROC] ensure_foreground error: {e}")
        return False


async def _wait_for_target_in_window(selector: str, window: str, timeout: float = 8.0) -> bool:
    """Poll the window's accessibility tree until the target element appears.

    Unlike the global `wait_for_element`, this scopes to the target window's
    PID, so it won't resolve prematurely when a similarly-named element
    exists in some other window (e.g. IDE 'Code Search' matching 'search').
    Also handles the YouTube-style race where the page is loaded visually
    but Chrome hasn't finished populating the UIA tree yet.
    """
    import sys
    import time
    try:
        # Prefer sys.modules lookup so test patches on sys.modules are honored
        # consistently. `from . import` semantics can vary based on whether the
        # parent package has the submodule cached as an attribute.
        app_automation = sys.modules.get("assistant.automation.native")
        if app_automation is None:
            from .automation import native as _aa
            app_automation = _aa
        target_name, target_role = app_automation._parse_selector_parts(selector)
        desktop = app_automation.ensure_desktop()
    except Exception as e:
        logger.debug(f"[PROC] Element wait init failed: {e}")
        return False

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            bounds = app_automation._find_element_bounds_in_tree(
                desktop, window, target_name, target_role
            )
            if bounds:
                return True
        except Exception as e:
            logger.debug(f"[PROC] Element bounds check failed: {e}")
        await asyncio.sleep(0.4)
    return False


def _get_window_context(step: dict) -> str | None:
    """Infer which window a step establishes context for."""
    action = step.get("action", "")
    params = step.get("params", {})
    if action == "open":
        name = params.get("name", "").lower().strip()
        return _default_browser() if _is_browser_name(name) else (name or None)
    if action == "focus":
        return params.get("name") or None
    if step.get("type") == "browser" and action == "navigate":
        return _default_browser()
    return None


# ─── Main Entry Point ─────────────────────────────────────────────────────────


async def run_procedure(proc: dict, original_text: str) -> str:
    """
    Execute a stored procedure step by step.

    - Resolves {user_input}, {date}, {time}, {clipboard}, and named {slots} in params
    - Retries failed steps up to _MAX_RETRIES times with backoff
    - Self-heals via vision LLM when all retries exhausted
    - Stops on unrecoverable error to avoid cascading broken state
    - Records usage in procedure_store after completion
    """
    steps = proc.get("steps", [])
    if not steps:
        return "Procedure has no steps."

    variables    = await _build_variables(proc, original_text)
    results: list[str] = []
    active_window: str | None = None

    for i, raw_step in enumerate(steps):
        step     = _resolve_variables(raw_step, variables)
        step_num = i + 1
        stype    = step.get("type", "")

        if _should_skip_open_before_navigate(step, steps, i):
            results.append(f"Step {step_num}: Skipped (browser navigate opens the browser)")
            logger.info(f"[PROC] Step {step_num}: skipped 'open browser' — next step is browser navigate")
            continue

        logger.info(
            f"[PROC] '{proc['name']}' step {step_num}/{len(steps)}: "
            f"{stype} {step.get('action', step.get('intent', '?'))}"
        )

        if active_window and step.get("action") in ("click", "type", "press_key"):
            await _ensure_foreground(active_window)

        result = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                if stype == "app":
                    result = await _execute_app_step(step, active_window, attempt=attempt)
                elif stype == "browser":
                    result = await _execute_browser_step_via_app(step, active_window)
                else:
                    result = f"Unknown step type '{stype}'"
            except Exception as exc:
                result = f"Exception: {exc}"

            if not _is_error(result):
                break

            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt]
                logger.info(
                    f"[PROC] Step {step_num} failed (attempt {attempt + 1}/{_MAX_RETRIES + 1}), "
                    f"retrying in {delay}s — {result}"
                )
                await asyncio.sleep(delay)
            else:
                healed = await _self_heal(step, result)
                if healed is not None:
                    result = healed

        results.append(f"Step {step_num}: {result}")

        new_ctx = _get_window_context(step)
        if new_ctx:
            active_window = new_ctx

        if _is_error(result):
            logger.warning(f"[PROC] Stopped at step {step_num}: {result}")
            break

    try:
        procedures.record_usage(proc["id"])
    except Exception as e:
        logger.debug(f"[PROC] record_usage failed (non-critical): {e}")

    return "\n".join(results)


# ─── Variable Resolution ──────────────────────────────────────────────────────


async def _build_variables(proc: dict, original_text: str) -> dict:
    """
    Build the substitution table for {placeholder} tokens.

    Built-ins (always present, zero LLM cost):
        {user_input}  — text after the trigger phrase
        {date}        — today's ISO date
        {time}        — current HH:MM
        {clipboard}   — current clipboard content

    Named slots (1 cheap 8b LLM call, only when procedure has them):
        {name}, {message}, {contact}, etc. — extracted from invocation text
    """
    trigger   = proc.get("trigger", "")
    remainder = _extract_user_input(original_text, trigger)

    variables = {
        "user_input": remainder,
        "clipboard":  _get_clipboard(),
        "date":       date.today().isoformat(),
        "time":       datetime.now().strftime("%H:%M"),
    }

    named_slots = _get_named_slots(proc)
    if named_slots:
        extracted = await _extract_named_slots(remainder, named_slots)
        variables.update(extracted)
        logger.info(f"[PROC] Slot values: {extracted}")

    return variables


def _get_named_slots(proc: dict) -> list[str]:
    """Scan procedure steps for {slot_name} tokens that aren't built-in variables."""
    raw   = json.dumps(proc.get("steps", []))
    found = _SLOT_RE.findall(raw)
    seen: set[str] = set()
    result: list[str] = []
    for name in found:
        if name not in _BUILTIN_VARS and name not in seen:
            seen.add(name)
            result.append(name)
    return result


_JSON_BLOCK_RE = re.compile(r'\{[^{}]*\}')


async def _extract_named_slots(remainder: str, slot_names: list[str]) -> dict[str, str]:
    """
    Extract values for named slots from the invocation remainder via 8b LLM.
    Falls back to putting the full remainder in every slot on parse failure.
    """
    if not remainder.strip():
        return {s: "" for s in slot_names}

    from . import llm

    slot_list    = ", ".join(slot_names)
    example_json = ", ".join(f'"{s}": "..."' for s in slot_names)
    prompt = (
        f'IMPORTANT: Return ONLY valid JSON, nothing else.\n'
        f'Extract values for these slots from the user text.\n'
        f'Text: "{remainder}"\n'
        f'Slots: {slot_list}\n'
        f'Output: {{{example_json}}}'
    )

    try:
        raw = await llm.chat(
            prompt,
            system_prompt="You are a JSON extraction tool. Return ONLY valid JSON, no explanation.",
            task_type="intent",
        )
        data = _parse_json_from_llm(raw)
        if data is None:
            raise ValueError(f"No valid JSON found in: {raw!r:.80}")
        return {s: str(data.get(s, "")).strip() for s in slot_names}
    except Exception as e:
        logger.warning(f"[PROC] Slot extraction failed ({e}), using full remainder for all slots")
        return {s: remainder for s in slot_names}


def _parse_json_from_llm(raw: str) -> dict | None:
    """Try to extract a JSON object from LLM output, tolerating surrounding text."""
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    for m in _JSON_BLOCK_RE.finditer(raw):
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _extract_user_input(text: str, trigger: str) -> str:
    """Return the portion of text that isn't part of the trigger phrase.

    Handles three cases:
      1. Trigger is a prefix: "search on youtube cats" → "cats"
      2. Trigger is a subsequence: "search cats on youtube" → "cats"
      3. No match: returns full text
    """
    low_text    = text.strip().lower()
    low_trigger = trigger.strip().lower()
    if low_text.startswith(low_trigger):
        return text[len(trigger):].strip().lstrip(",").strip()
    remainder = procedures.subsequence_remainder(trigger, text)
    if remainder != text.strip():
        return remainder
    return text.strip()


def _get_clipboard() -> str:
    try:
        import pyperclip
        return pyperclip.paste() or ""
    except Exception as e:
        logger.debug(f"[PROC] Clipboard paste failed: {e}")
        return ""


def _resolve_variables(step: dict, variables: dict) -> dict:
    """Deep-substitute {placeholder} tokens in all string values of a step dict."""
    raw = json.dumps(step)
    for key, val in variables.items():
        raw = raw.replace("{" + key + "}", str(val).replace('"', '\\"'))
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return step


def _should_skip_open_before_navigate(step: dict, steps: list, idx: int) -> bool:
    """Skip 'open chrome' when the next step is a browser navigate (Playwright opens its own browser)."""
    if step.get("type") != "app" or step.get("action") != "open":
        return False
    name = step.get("params", {}).get("name", "").lower().strip()
    if not _is_browser_name(name):
        return False
    if idx + 1 < len(steps):
        nxt = steps[idx + 1]
        if nxt.get("type") == "browser" and nxt.get("action") == "navigate":
            return True
    return False


def _is_error(result: str) -> bool:
    if not result:
        return False
    low = result.lower()
    return any(p in low for p in (
        "error", "failed", "not found", "couldn't", "cannot",
        "exception", "timed out", "timeout", "not available",
    ))


# ─── Self-Heal ────────────────────────────────────────────────────────────────


async def _self_heal(step: dict, error: str) -> str | None:
    """
    Last-resort recovery after all retries fail.
    Takes a screenshot and asks the vision LLM:
      - Did the step actually succeed? → return success string
      - Should it be skipped? → return skip string
      - Did it truly fail? → return None (caller stops execution)
    """
    try:
        from .io import screen as _screen
        from . import llm as _llm

        img = _screen.capture_screenshot_base64()
        if not img:
            return None

        action = step.get("action", "?")
        params = json.dumps(step.get("params", {}))
        prompt = (
            f"A UI automation step failed after all retries.\n"
            f"Step: {action} with params {params}\n"
            f"Error: {error}\n\n"
            f"Look at the current screen and decide:\n"
            f"  SUCCEEDED — the action completed despite the error\n"
            f"  SKIP: <brief reason> — this step can be safely skipped\n"
            f"  FAILED: <brief reason> — execution should stop\n\n"
            f"Reply with exactly one of those lines."
        )

        resp = (await _llm.get_vision_response(
            img, prompt,
            system_prompt="You are a UI automation assistant. Be concise and precise.",
        )).text.strip()

        upper = resp.upper()
        if upper.startswith("SUCCEEDED"):
            logger.info(f"[PROC] Self-heal: step verified succeeded via screen")
            return f"{action} completed (screen-verified)"

        if upper.startswith("SKIP"):
            reason = resp[4:].lstrip(":").strip()
            logger.info(f"[PROC] Self-heal: skipping step — {reason}")
            return f"Skipped ({reason})"

        logger.warning(f"[PROC] Self-heal: step confirmed failed — {resp}")
        return None

    except Exception as e:
        logger.debug(f"[PROC] Self-heal error: {e}")
        return None


# ─── Pre-flight Wait ──────────────────────────────────────────────────────────


async def _wait_for_target(selector: str, timeout: float = 4.0) -> None:
    """
    Best-effort wait for a UI element before clicking/focusing it.
    Swallows errors — if element isn't found in time the step will
    run anyway and produce a clearer error message.
    """
    from .automation import native as app_automation
    try:
        await app_automation.wait_for_element(selector, timeout=timeout)
    except Exception as e:
        logger.debug(f"[PROC] Pre-step element wait timed out: {e}")


# ─── App Step Execution ───────────────────────────────────────────────────────


async def _execute_app_step(step: dict, active_window: str | None = None, attempt: int = 0) -> str:
    from .automation import native as app_automation

    action = step.get("action", "")
    params = step.get("params", {})
    wait_timeout = 2.0 if attempt > 0 else 8.0

    if action == "open":
        return await app_automation.open_app(params.get("name", ""))

    elif action == "focus":
        selector = f"name:{params.get('name', '')}"
        await _wait_for_target(selector, timeout=min(wait_timeout, 4.0))
        return await app_automation.focus_window(params.get("name", ""))

    elif action == "close":
        return await app_automation.close_app(params.get("name", ""))

    elif action == "click":
        selector = params.get("selector", "")
        window = params.get("window") or active_window
        if window:
            await _wait_for_target_in_window(selector, window, timeout=wait_timeout)
        else:
            await _wait_for_target(selector, timeout=min(wait_timeout, 4.0))
        return await app_automation.click_element(selector, window)

    elif action == "type":
        window = params.get("window") or active_window
        if window and not params.get("selector"):
            await app_automation.focus_window(window)
            await asyncio.sleep(0.2)
        return await app_automation.type_text(
            params.get("text", ""),
            params.get("selector"),
            window,
        )

    elif action == "press_key":
        key = params.get("key", "")
        try:
            import pyautogui
            parts = [k.strip().lower() for k in key.split("+")]
            if len(parts) > 1:
                pyautogui.hotkey(*parts)
            else:
                pyautogui.press(parts[0])
            return f"Pressed {key}"
        except Exception as e:
            return f"Error pressing {key}: {e}"

    elif action == "wait":
        seconds = float(params.get("seconds", 1))
        await asyncio.sleep(seconds)
        return f"Waited {seconds}s"

    return f"Unknown app action '{action}'"


# ─── Browser Step Execution (via app automation) ────────────────────────────
#
# Procedures replay user-taught steps like a human would — using the system
# browser, not Playwright. Navigate = open browser → Ctrl+L → type URL → Enter.
# This eliminates Playwright startup failures and uses the user's real browser.


async def _execute_browser_step_via_app(step: dict, active_window: str | None = None) -> str:
    """
    Execute a browser-type step using app automation (keyboard/mouse),
    NOT Playwright. Works with any system browser.
    """
    from .automation import native as app_automation
    import pyautogui

    action = step.get("action", "")
    params = step.get("params", {})
    browser = _default_browser()
    browser_window = active_window or browser

    try:
        if action == "navigate":
            url = params.get("url", "")
            await app_automation.open_app(browser)
            await asyncio.sleep(1.0)
            if not await _ensure_foreground(browser):
                logger.warning("[PROC] Browser may not have focus — attempting navigate anyway")
            pyautogui.hotkey("ctrl", "l")
            await asyncio.sleep(0.3)
            pyautogui.typewrite(url, interval=0.02)
            pyautogui.press("enter")
            await asyncio.sleep(2)
            return f"Navigated to {url}"

        elif action == "click":
            selector = params.get("selector", "")
            await _wait_for_target_in_window(selector, browser_window, timeout=8.0)
            return await app_automation.click_element(selector, browser_window)

        elif action == "fill":
            selector = params.get("selector", "")
            value    = params.get("value", params.get("text", ""))
            await _wait_for_target_in_window(selector, browser_window, timeout=8.0)
            return await app_automation.type_text(str(value), selector, browser_window)

        elif action == "press":
            key = params.get("key", "")
            parts = [k.strip().lower() for k in key.split("+")]
            if len(parts) > 1:
                pyautogui.hotkey(*parts)
            else:
                pyautogui.press(parts[0])
            return f"Pressed {key}"

        elif action == "wait":
            seconds = float(params.get("seconds", params.get("time_ms", 1000)) / 1000
                            if "time_ms" in params
                            else params.get("seconds", 1))
            await asyncio.sleep(seconds)
            return f"Waited {seconds}s"

        else:
            return f"Unknown browser action '{action}'"

    except Exception as e:
        return f"Error in browser {action}: {e}"
