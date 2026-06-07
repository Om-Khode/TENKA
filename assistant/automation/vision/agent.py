"""Vision agent — agentic computer control loop.

Plans and executes multi-step computer tasks using screen reading,
pyautogui + pynput for mouse/keyboard, LLM for planning, and
LLM-based goal verification after each action batch.
"""

import json
import logging
import time
import subprocess
import sys
from typing import Callable, Optional

from .todo_classifier import _make_todo_dict
from .verifier import (
    _is_yes_answer,
    _quick_verify_from_window_title,
    _verify_goal,
    VERIFICATION_SYSTEM_PROMPT,
)
from ._parsing import _parse_plan, _recover_truncated_json
from ... import config

logger = logging.getLogger("computer_agent")

# ─── Safety Configuration ────────────────────────────────────────────────────

MAX_STEPS = 15       # Max actions per planning cycle
MAX_LOOPS = 8        # Max re-plan loops before giving up. Bumped 5→8 on
                     # 2026-04-26 after demoqa form-fill repeatedly hit the
                     # cap — 5 loops × ~1.5 fields = 7-8 fields, but realistic
                     # forms have 10+ fields and the planner sometimes burns
                     # a loop on placeholder confusion or mid-batch replans.
                     # Cost increment: ~$0.001 per extra loop (vision plan +
                     # verify), so worst-case +$0.003/task.
SAFE_MODE = True     # Log every action before executing
_agent_typing = False  # Flag indicating pyautogui is currently driving input
_search_activated_this_batch = False


class _TaskState:
    """
    Per-task state tracker.  Reset at the start of every computer_task.

    Provides:
      - action_cache:  stores results of data-fetching actions so the LLM
                       can reference them without re-calling the action.
      - done_effects:  tracks one-shot side effects (e.g. 'auto_paste_tabs')
                       so they are never repeated within the same task.

    Any new data-fetching action can opt in by adding its name to
    CACHEABLE_ACTIONS.  Any new auto-side-effect can opt in by calling
    mark_done() / is_done() with a descriptive key.
    """

    # Actions whose results get cached per task
    CACHEABLE_ACTIONS = {"get_browser_tabs", "clipboard_read"}

    # Hard cap on TODO list size. If a goal genuinely needs more, escalate to
    # user rather than silently truncate. Covers form-fill (typical 6-12
    # fields) plus a few cascading-reveal items.
    TODO_MAX = 15

    def __init__(self):
        self.action_cache: dict[str, str] = {}
        self.done_effects: set[str] = set()
        # Independent completeness signal. Each item is a dict with base
        # fields (id, task, done) plus classifier fields (kind, target,
        # field, value, pending_visual_confirm, confirm_strikes). Empty list
        # means TODO tracking is inactive for this task — completion falls
        # back to the vision verifier alone.
        self.todo_list: list[dict] = []
        self._next_todo_id: int = 1
        # zero_progress_streak counts consecutive batches with zero TODO marks
        # AND zero adds — the stuck detector aborts at 3. loop_budget is the
        # dynamic step budget — seeded in set_initial_todos; defaults to
        # MAX_LOOPS for legacy paths. confirm_fallback_count = total
        # fallback-confirm calls (YES + NO). confirm_abandoned_count = subset
        # that ended in NO and abandoned the TODO via Fix A (action signature
        # trusted because vision couldn't confirm). Telemetry only — used to
        # spot Rule-S over-deferral or vision-confirm prompt drift in future
        # tuning. The stuck detector should NOT treat abandoned-marks as full
        # progress (an abandoned-confirm is exactly the failure shape
        # stuck-detection exists to catch); count abandoned as fractional or
        # excluded.
        self.zero_progress_streak: int = 0
        self.loop_budget: int = MAX_LOOPS
        self.confirm_fallback_count: int = 0
        self.confirm_abandoned_count: int = 0
        # Dialog-engagement gate counter. Incremented at the top of each
        # planner step in _run_computer_task_inner. Per-TODO
        # `batch_marked_done` / `batch_deferred` fields are stamped against
        # this counter when rules mark or defer a TODO; the gate compares
        # against `current_batch_idx - window` to decide if engagement is
        # still hot.
        self.batch_idx: int = 0

    def reset(self):
        """Clear all state — called at the start of each task."""
        self.action_cache.clear()
        self.done_effects.clear()
        self.todo_list.clear()
        self._next_todo_id = 1
        self.zero_progress_streak = 0
        self.loop_budget = MAX_LOOPS
        self.confirm_fallback_count = 0
        self.confirm_abandoned_count = 0
        self.batch_idx = 0

    # ── cache helpers ──
    def cache_result(self, action_name: str, result: str):
        if action_name in self.CACHEABLE_ACTIONS:
            self.action_cache[action_name] = result

    def get_cached(self, action_name: str) -> str | None:
        return self.action_cache.get(action_name)

    # ── side-effect helpers ──
    def mark_done(self, effect: str):
        self.done_effects.add(effect)

    def is_done(self, effect: str) -> bool:
        return effect in self.done_effects

    # ── LLM context ──
    def format_cached_data(self) -> str:
        """Format cached results for inclusion in the vision prompt."""
        if not self.action_cache:
            return ""
        parts = []
        for action, result in self.action_cache.items():
            label = action.replace("_", " ").title()
            parts.append(f"[{label}]:\n{result}")
        return "\n\n".join(parts)

    # ── TODO list helpers ──
    def set_initial_todos(self, tasks: list[str]) -> int:
        """
        Replace todo_list with a fresh set generated at task start.
        Returns the number of items actually stored (may be less than input
        if TODO_MAX cap is hit). Empty/whitespace strings are skipped.

        Each TODO is classified into kind/target/field/value via
        `_classify_todo` so the action-signature matcher can mark it
        deterministically without an LLM call.
        """
        self.todo_list.clear()
        self._next_todo_id = 1
        added = 0
        for raw in tasks:
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                continue
            if added >= self.TODO_MAX:
                break
            self.todo_list.append(_make_todo_dict(self._next_todo_id, text))
            self._next_todo_id += 1
            added += 1
        return added

    def add_todo(self, text: str) -> int | None:
        """
        Append a single TODO discovered mid-task (e.g. cascading dropdown
        revealed a new field). Skips empties, dedupes against existing items,
        and respects TODO_MAX. Returns the new id, or None if nothing added.

        Newly-added TODOs are classified the same way as the initial set
        so they participate in deterministic matching.
        """
        if not isinstance(text, str):
            return None
        clean = text.strip()
        if not clean:
            return None
        if len(self.todo_list) >= self.TODO_MAX:
            return None
        # Case-insensitive dedupe — the updater LLM tends to re-emit existing
        # items occasionally; we don't want the list to grow with duplicates.
        clean_lower = clean.lower()
        for item in self.todo_list:
            if item["task"].strip().lower() == clean_lower:
                return None
        new_id = self._next_todo_id
        self.todo_list.append(_make_todo_dict(new_id, clean))
        self._next_todo_id += 1
        return new_id

    def mark_todo_done(self, todo_id: int, batch_idx: int | None = None) -> bool:
        """
        Mark the TODO with the given id as done. Returns True if found and
        toggled (or already done), False if no such id exists.

        When `batch_idx` is provided (or omitted — defaults to the current
        `self.batch_idx`), the TODO's `batch_marked_done` is stamped with
        that value. The dialog-engagement gate uses this to detect recent
        successful interaction with a modal surface.
        """
        stamp = batch_idx if batch_idx is not None else self.batch_idx
        for item in self.todo_list:
            if item["id"] == todo_id:
                item["done"] = True
                item["batch_marked_done"] = stamp
                return True
        return False

    def all_todos_done(self) -> bool:
        """True only if todo_list is non-empty AND every item is done."""
        if not self.todo_list:
            return False
        return all(item["done"] for item in self.todo_list)

    def abandoned_field_summary(self, max_fields: int = 2) -> str:
        """
        Fix A: build a short comma-joined list of `confirm_abandoned=True`
        TODO field names for the final TTS reply suffix. Returns '' when
        no abandoned TODOs exist. Caps at `max_fields` and adds "and N more"
        when truncated (TTS-friendly — the reply must stay ≤200 chars).

        Example: "Staff Size, Industry"  /  "Staff Size and 2 more"
        """
        abandoned = [t for t in self.todo_list if t.get("confirm_abandoned")]
        if not abandoned:
            return ""

        def _label(t: dict) -> str:
            # Prefer the structured field name; fall back to a clipped task
            # text when the TODO wasn't a select-style item (defensive — Fix A
            # is only set by the select fallback today, but the schema is
            # generic).
            f = t.get("field") or t.get("task") or ""
            return f.strip()[:30]

        labels = [lbl for lbl in (_label(t) for t in abandoned) if lbl]
        if not labels:
            return ""
        if len(labels) <= max_fields:
            return ", ".join(labels)
        head = ", ".join(labels[:max_fields])
        rest = len(labels) - max_fields
        return f"{head} and {rest} more"

    def todo_progress_str(self) -> str:
        """
        Human-readable progress block for inclusion in the planner prompt.
        Returns '' when todo_list is empty (caller decides whether to inject).

        Three states rendered distinctly:
          ✓ done           — completed (planner skips, treats as success)
          · pending confirm — Rule S deferred; awaiting visual-confirm result.
                              Annotated "(awaiting confirm — do not retry)"
                              so the planner doesn't re-attempt during the
                              1-3 strike window. Fix B (2026-04-26).
          ✗ open           — not yet attempted; "← NEXT" marker on the first

        Fix A: abandoned ✓ items get an `(unconfirmed)` suffix for debug.log
        readability — the planner just sees ✓ and treats as done.

        Header includes a `(N awaiting confirm)` clause when any pending
        TODOs exist, so the planner knows to expect the · symbol.
        """
        if not self.todo_list:
            return ""
        done = sum(1 for item in self.todo_list if item["done"])
        pending = sum(1 for item in self.todo_list
                      if item.get("pending_visual_confirm") and not item["done"])
        total = len(self.todo_list)

        header = f"TODO PROGRESS ({done} of {total} done"
        if pending:
            header += f", {pending} awaiting confirm"
        header += "):"
        lines = [header]

        next_marked = False  # only one "← NEXT" marker, on the first ✗
        for item in self.todo_list:
            if item["done"]:
                annotation = " (unconfirmed)" if item.get("confirm_abandoned") else ""
                lines.append(f"  ✓ {item['task']}{annotation}")
            elif item.get("pending_visual_confirm"):
                # Pending: planner must NOT retry while we're confirming.
                lines.append(
                    f"  · {item['task']}      (awaiting confirm — do not retry)"
                )
            else:
                # Open: this is the next actionable item.
                suffix = "      ← NEXT" if not next_marked else ""
                lines.append(f"  ✗ {item['task']}{suffix}")
                next_marked = True
        return "\n".join(lines)


_task_state = _TaskState()

# ─── pyautogui Setup ─────────────────────────────────────────────────────────

try:
    import pyautogui
    # Disable pyautogui's corner-of-screen failsafe — it fires whenever the
    # user's cursor *happens* to be at a screen corner when the agent calls
    # pyautogui (not just when it's moved there mid-call), aborting every
    # action with PyAutoGUIFailSafeException. We have our own deliberate
    # abort path: hold ESC ~1s (see start_esc_monitor / _check_abort), which
    # is the user-facing "stop now" mechanism. The corner check is a
    # duplicate that actively breaks runs when the user happens to leave
    # their mouse near a screen edge.
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.3       # Brief pause between actions
except ImportError:
    pyautogui = None
    logger.warning("[AGENT] pyautogui not installed — pip install pyautogui")


# ─── Abort ────────────────────────────────────────────────────────────
# Lifted to assistant/core/abort.py + assistant/io/esc_monitor.py in overlay rollout.
# These shims preserve the old call surface used elsewhere in this file.

from assistant.core.abort import abort as _abort_controller
from assistant.core.abort import UserAborted as _UserAborted
from assistant.io.esc_monitor import esc_monitor as _esc_monitor_singleton
from assistant.io.status_broadcaster import status as _status_broadcaster
from assistant.io.status_broadcaster import StatusPhase as _StatusPhase

# Back-compat shims for call sites that used the old local abort/monitor API.
def _check_abort() -> bool:
    return _abort_controller.is_aborted()

def reset_abort() -> None:
    _abort_controller.reset()
    _task_state.reset()

def start_esc_monitor() -> None:
    _esc_monitor_singleton.start()

def stop_esc_monitor() -> None:
    _esc_monitor_singleton.stop()


def _pct_to_pixels(x_pct: float, y_pct: float) -> tuple[int, int]:
    """Convert percentage coordinates to screen pixels."""
    width, height = pyautogui.size()
    return int(x_pct * width), int(y_pct * height)


# ─── Action Functions ────────────────────────────────────────────────────────


def mouse_move(x: int, y: int) -> str:
    """Move mouse to absolute coordinates."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: mouse_move({x}, {y})")
    try:
        pyautogui.moveTo(x, y, duration=0.3)
        return f"Moved mouse to ({x}, {y})"
    except Exception as e:
        logger.error(f"[AGENT] mouse_move failed: {e}")
        return f"Failed: {e}"


def mouse_click(x: int, y: int) -> str:
    """Click at coordinates."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: mouse_click({x}, {y})")
    try:
        pyautogui.click(x, y)
        return f"Clicked at ({x}, {y})"
    except Exception as e:
        logger.error(f"[AGENT] mouse_click failed: {e}")
        return f"Failed: {e}"


def mouse_double_click(x: int, y: int) -> str:
    """Double-click at coordinates."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: mouse_double_click({x}, {y})")
    try:
        pyautogui.doubleClick(x, y)
        return f"Double-clicked at ({x}, {y})"
    except Exception as e:
        logger.error(f"[AGENT] mouse_double_click failed: {e}")
        return f"Failed: {e}"


def mouse_right_click(x: int, y: int) -> str:
    """Right-click at coordinates."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: mouse_right_click({x}, {y})")
    try:
        global _agent_typing
        _agent_typing = True
        pyautogui.rightClick(x, y)
        _agent_typing = False
        return f"Right-clicked at ({x}, {y})"
    except Exception as e:
        _agent_typing = False
        logger.error(f"[AGENT] mouse_right_click failed: {e}")
        return f"Failed: {e}"


def _check_focus_or_dialog(expected_window: str) -> bool:
    """
    Returns True if expected_window is currently active OR a related dialog is visible on screen.
    Uses three checks in order:
      1. Window title substring match (fast)
      2. Process-level match — the active window belongs to the same process as a
         window whose title matches expected_window (handles apps that change their
         title dynamically, e.g. media players showing "Artist - Song")
      3. Vision LLM check for dialogs/popups (slow, last resort)
    """
    from ...io import screen
    active = screen.get_active_window()
    if expected_window.lower() in active.lower():
        return True

    # Check 2: Process executable name match
    # Apps like Spotify rename their window to "Artist - Song" when playing,
    # so no window with the original title exists. Instead, check if the
    # foreground process's executable name matches any word in expected_window.
    # e.g., expected_window="Spotify Premium" → ["spotify","premium"] → "spotify" in "spotify.exe" ✓
    try:
        import ctypes
        import psutil

        user32 = ctypes.windll.user32
        foreground_hwnd = user32.GetForegroundWindow()
        fg_pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(foreground_hwnd, ctypes.byref(fg_pid))

        process = psutil.Process(fg_pid.value)
        exe_name = process.name().lower().replace(".exe", "")
        expected_words = [w for w in expected_window.lower().split() if len(w) > 2]

        if any(word in exe_name for word in expected_words):
            logger.info(
                f"[AGENT] Focus check: title mismatch but process matches "
                f"('{exe_name}' matches '{expected_window}') — '{active}' is correct app"
            )
            return True
    except Exception as e:
        logger.debug(f"[AGENT] Process-based focus check failed: {e}")

    # Check 3: Active window doesn't match — take screenshot and ask vision LLM if a dialog is visible
    screenshot = screen.capture_screenshot_base64()
    if screenshot is None:
        return False
    from ... import llm
    answer = llm._vision_yes_no_sync(
        image_base64=screenshot,
        prompt=f"Is there a dialog box, popup, or modal window currently visible on screen that is related to '{expected_window}'? Examples: Save dialog, Open dialog, Print dialog, file picker, rename dialog. Reply with only YES or NO."
    )
    return answer == "YES"


def keyboard_type(text: str, interval: float = 0.05, expected_window: str = None) -> str:
    """Type text using the keyboard."""
    global _agent_typing
    if expected_window:
        if not _check_focus_or_dialog(expected_window):
            from ...io import screen as _screen
            active = _screen.get_active_window()
            return (
                f"ABORTED_WRONG_FOCUS: Expected window or dialog containing '{expected_window}' "
                f"not found. Active window is '{active}'. "
                f"Use focus_application('{expected_window}') then retry."
            )
    # Generic input focus guard: check if any input field is focused.
    # If GetFocus() returns 0, no input field has keyboard focus —
    # keypresses will hit app shortcuts instead of entering text.
    # In that case, try Ctrl+K (universal search shortcut for most apps:
    # Spotify, VS Code, Slack, Discord, Notion, etc.) to activate search.
    # We do NOT use Ctrl+L here because that's browser-specific and causes
    # issues in file dialogs (focuses location bar) and Notepad (Go To Line).
    #
    # Browsers are excluded entirely: Ctrl+K opens the omnibox/search-bar in
    # Chrome/Firefox/Edge, hijacking typed text away from the page form the
    # vision agent was trying to fill.
    try:
        import ctypes
        user32 = ctypes.windll.user32
        focused_hwnd = user32.GetFocus()
        if focused_hwnd == 0 and not _search_activated_this_batch:
            from ...io import screen as _screen_focus
            active_lower = (_screen_focus.get_active_window() or "").lower()
            is_browser = any(b in active_lower for b in config.BROWSER_NAMES)
            if is_browser:
                logger.info(
                    "[AGENT] keyboard_type: no input focused but active window is a browser — "
                    "skipping Ctrl+K (would open omnibox); typing anyway"
                )
            else:
                logger.info("[AGENT] keyboard_type: no input focused (GetFocus=0), sending Ctrl+K to activate search")
                _agent_typing = True
                pyautogui.hotkey("ctrl", "k")
                time.sleep(0.4)
                _agent_typing = False
                # Re-check after Ctrl+K
                focused_hwnd = user32.GetFocus()
                if focused_hwnd == 0:
                    logger.warning("[AGENT] keyboard_type: still no input focused after Ctrl+K — typing anyway")
    except Exception as _focus_e:
        logger.debug(f"[AGENT] keyboard_type: focus check failed: {_focus_e}")
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: keyboard_type(\"{text[:50]}...\")")
    try:
        _agent_typing = True
        pyautogui.typewrite(text, interval=interval)
        _agent_typing = False
        from ...io import screen as _screen_kt
        active_after = _screen_kt.get_active_window()
        # Heuristic: if active window looks like "Artist - Song" (contains " - " 
        # and no common app keywords), typing likely hit media shortcuts not a text field
        from ...core.known_apps import KNOWN_APPS as _ka
        app_keywords = frozenset(
            word for name in _ka for word in name.split()
        ) | {"explorer", "premium", "word", "excel"}
        looks_like_media = (
            " - " in active_after 
            and not any(kw in active_after.lower() for kw in app_keywords)
        )
        if looks_like_media:
            logger.warning(f"[AGENT] keyboard_type: after typing, window changed to '{active_after}' — "
                           f"likely hit media shortcuts instead of input field. "
                           f"Search bar was probably not focused.")
            return (f"WARNING: Typed '{text[:50]}' but active window changed to '{active_after}' — "
                    f"keypresses likely triggered media shortcuts instead of entering text. "
                    f"Use keyboard_hotkey ctrl+k first to focus search, then retype.")
        return f"Typed: \"{text[:50]}\""
    except Exception as e:
        try:
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            _agent_typing = False
            return f"Typed (via clipboard): \"{text[:50]}\""
        except Exception as e2:
            _agent_typing = False
            logger.error(f"[AGENT] keyboard_type failed: {e2}")
            return f"Failed: {e2}"


def keyboard_hotkey(*keys: str, expected_window: str = None) -> str:
    """Press a keyboard shortcut (e.g., 'ctrl', 'c')."""
    if expected_window:
        if not _check_focus_or_dialog(expected_window):
            from ...io import screen as _screen
            active = _screen.get_active_window()
            return (
                f"ABORTED_WRONG_FOCUS: Expected window or dialog containing '{expected_window}' "
                f"not found. Active window is '{active}'. "
                f"Use focus_application('{expected_window}') then retry."
            )
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: keyboard_hotkey({', '.join(keys)})")
    try:
        global _agent_typing
        _agent_typing = True
        pyautogui.hotkey(*keys)
        # Brief pause after hotkeys that open new UI (new tab, new window, save dialog, etc.)
        # This ensures the new UI element is focused before the next action fires
        UI_OPENING_HOTKEYS = {frozenset(["ctrl", "t"]), frozenset(["ctrl", "n"]), frozenset(["ctrl", "s"]), frozenset(["ctrl", "shift", "s"]), frozenset(["alt", "f4"])}
        if frozenset(k.lower() for k in keys) in UI_OPENING_HOTKEYS:
            time.sleep(0.8)
        _agent_typing = False
        return f"Pressed hotkey: {'+'.join(keys)}"
    except Exception as e:
        _agent_typing = False
        logger.error(f"[AGENT] keyboard_hotkey failed: {e}")
        return f"Failed: {e}"


def keyboard_press(key: str, expected_window: str = None) -> str:
    """Press a single key (e.g., 'enter', 'tab', 'escape')."""
    if expected_window:
        if not _check_focus_or_dialog(expected_window):
            from ...io import screen as _screen
            active = _screen.get_active_window()
            return (
                f"ABORTED_WRONG_FOCUS: Expected window or dialog containing '{expected_window}' "
                f"not found. Active window is '{active}'. "
                f"Use focus_application('{expected_window}') then retry."
            )
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: keyboard_press({key})")
    try:
        global _agent_typing
        _agent_typing = True
        parts = [k.strip().lower() for k in key.split('+')]
        if len(parts) > 1:
            pyautogui.hotkey(*parts)
        else:
            pyautogui.press(parts[0])
        _agent_typing = False
        return f"Pressed key: {key}"
    except Exception as e:
        _agent_typing = False
        logger.error(f"[AGENT] keyboard_press failed: {e}")
        return f"Failed: {e}"


def scroll(clicks: int, x: int | None = None, y: int | None = None) -> str:
    """Scroll the mouse wheel. Positive = up, negative = down."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: scroll({clicks}, x={x}, y={y})")
    try:
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x, y)
        else:
            pyautogui.scroll(clicks)
        direction = "up" if clicks > 0 else "down"
        return f"Scrolled {direction} {abs(clicks)} clicks"
    except Exception as e:
        logger.error(f"[AGENT] scroll failed: {e}")
        return f"Failed: {e}"


def clipboard_read() -> str:
    """Read the current clipboard content."""
    try:
        import pyperclip
        content = pyperclip.paste()
        logger.info(f"[AGENT] Clipboard read: \"{content[:80]}...\"")
        return content
    except Exception as e:
        logger.error(f"[AGENT] clipboard_read failed: {e}")
        return ""


def clipboard_write(text: str) -> str:
    """Write text to the clipboard."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: clipboard_write(\"{text[:50]}...\")")
    try:
        import pyperclip
        pyperclip.copy(text)
        return f"Copied to clipboard: \"{text[:50]}\""
    except Exception as e:
        logger.error(f"[AGENT] clipboard_write failed: {e}")
        return f"Failed: {e}"


def wait(seconds: float) -> str:
    """Wait for a number of seconds."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: wait({seconds})")
    capped = min(seconds, 10.0)  # Cap at 10 seconds for safety
    time.sleep(capped)
    return f"Waited {capped} seconds"


def _should_open_new_file(app_name: str, window_title: str) -> bool:
    """
    Determine whether to open a new file (Ctrl+N) when a text editor is focused.
    
    Rules:
      1. Only applies to known text editors.
      2. If title starts with * — unsaved changes exist → always open new file.
      3. If title is bare "Untitled - AppName" (no * , no real filename) → empty new file → no Ctrl+N.
      4. If title has a real filename (not "Untitled") → file has saved content → open new file.
         Edge case: if filename is something like "Untitled.txt", check file size on disk.
         If size > 0 → open new file. If file not found → open new file to be safe.
    
    Returns True if Ctrl+N should be sent, False if the editor is already clean.
    """
    from ...core.known_apps import get_apps_by_category
    text_editors = get_apps_by_category("text_editor")
    if not any(ed in app_name.lower() for ed in text_editors):
        return False  # Not a text editor we manage

    title = window_title.strip()

    # Rule 2: unsaved changes
    if title.startswith("*"):
        return True

    # Parse out just the filename part — strip " - AppName" suffix
    # e.g. "notes.txt - Notepad" → "notes.txt"
    # e.g. "Untitled - Notepad"  → "Untitled"
    filename = title
    for ed in TEXT_EDITORS:
        suffix = f" - {ed}"
        if filename.lower().endswith(suffix):
            filename = filename[:-len(suffix)].strip()
            break
    # Also handle capitalised app names like "Notepad"
    if " - " in filename:
        filename = filename.rsplit(" - ", 1)[0].strip()

    # Rule 3: bare "Untitled" with no extension — genuinely empty new file
    if filename.lower() == "untitled":
        return False

    # Rule 4: real filename — check if it has an extension or is not "Untitled"
    # Sub-case: filename is "Untitled.txt" or similar — saved-as-untitled edge case
    # Try to find the file on disk and check its size
    from pathlib import Path

    # Search common locations: Desktop, Documents, home dir, current dir
    search_dirs = [
        Path.home() / "Desktop",
        Path.home() / "Documents",
        Path.home(),
        Path.cwd(),
    ]

    for directory in search_dirs:
        candidate = directory / filename
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                logger.info(f"[AGENT] _should_open_new_file: found '{candidate}' with size {candidate.stat().st_size}B → has content")
                return True
            elif candidate.exists() and candidate.stat().st_size == 0:
                logger.info(f"[AGENT] _should_open_new_file: found '{candidate}' but it is empty → no Ctrl+N needed")
                return False
        except Exception:
            continue

    # File not found on disk — if it has a real filename (not untitled), be safe and open new
    logger.info(f"[AGENT] _should_open_new_file: '{filename}' not found on disk — opening new file to be safe")
    return True


def open_application(name: str) -> str:
    """Open an application using Windows search (Win+S)."""
    global _agent_typing
    from ...io import screen as _screen
    open_wins = _screen.get_open_windows()
    already_open = any(name.lower() in w.lower() for w in open_wins)
    if already_open:
        logger.info(f"[AGENT] '{name}' is already open — focusing instead of re-opening")
        result = focus_application(name)
        from ...io import screen as _screen
        active = _screen.get_active_window()
        if _should_open_new_file(name, active):
            logger.info(f"[AGENT] '{active}' has existing content — opening new file to preserve it")
            global _agent_typing
            _agent_typing = True
            pyautogui.hotkey("ctrl", "n")
            time.sleep(0.5)
            _agent_typing = False
        return result

    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: open_application(\"{name}\")")
    try:
        # Use Win+S to open search, type the name, press Enter
        _agent_typing = True
        pyautogui.hotkey("win", "s")
        time.sleep(0.6)
        pyautogui.typewrite(name, interval=0.07)
        time.sleep(0.8)
        pyautogui.press("enter")
        time.sleep(1.5)
        _agent_typing = False

        from ...io import screen
        # Post-open: if window has unsaved/existing content, open a fresh file
        active = screen.get_active_window()
        if _should_open_new_file(name, active):
            logger.info(f"[AGENT] '{active}' has existing content — opening new file to preserve it")
            _agent_typing = True
            pyautogui.hotkey("ctrl", "n")
            time.sleep(0.5)
            _agent_typing = False

        return f"Opening application: {name}"
    except Exception as e:
        _agent_typing = False
        logger.error(f"[AGENT] open_application failed: {e}")
        return f"Failed: {e}"


def focus_application(name: str) -> str:
    """Focus an already-open application by clicking its taskbar button."""
    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: focus_application(\"{name}\")")
    try:
        import ctypes
        import pygetwindow as gw

        # Find matching window — first by title, then by process name
        all_windows = gw.getAllWindows()
        matches = [w for w in all_windows if name.lower() in w.title.lower() and w.title.strip()]

        user32 = ctypes.windll.user32

        # If no title match, try matching by process executable name
        # (handles apps that change their title, e.g. media players showing "Artist - Song")
        if not matches:
            try:
                import psutil
                for w in all_windows:
                    if not w.title.strip() or not w._hWnd:
                        continue
                    w_pid = ctypes.c_ulong()
                    user32.GetWindowThreadProcessId(w._hWnd, ctypes.byref(w_pid))
                    try:
                        proc = psutil.Process(w_pid.value)
                        exe = proc.name().lower().replace(".exe", "")
                        name_words = [word for word in name.lower().split() if len(word) > 2]
                        if any(word in exe for word in name_words):
                            matches.append(w)
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                logger.debug(f"[AGENT] Process-name fallback failed: {e}")

        if not matches:
            return f"Could not find window for '{name}'"

        hwnd = matches[0]._hWnd

        # Step 1: If minimized, restore it first
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            time.sleep(0.5)

        # Step 2: Try AllowSetForegroundWindow trick — tell Windows our process is allowed
        target_pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(target_pid))
        user32.AllowSetForegroundWindow(target_pid.value)

        # Step 3: Force foreground via keybd_event trick
        # Simulate an Alt key tap then immediately dismiss with Escape.
        # The Alt tap makes Windows think the user is interacting,
        # which temporarily lifts the focus lock. The Escape dismisses
        # any menu bar that Alt may activate (e.g. in Firefox).
        VK_MENU = 0x12  # Alt key
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.05)
        # Dismiss any menu bar that Alt activated
        VK_ESCAPE = 0x1B
        user32.keybd_event(VK_ESCAPE, 0, 0, 0)
        user32.keybd_event(VK_ESCAPE, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)

        # Step 4: Now SetForegroundWindow should work
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.5)

        # Step 5: Post-focus — handle unsaved content in text editors
        from ...io import screen as _screen
        active = _screen.get_active_window()
        from ...core.known_apps import get_apps_by_category
        text_editors = get_apps_by_category("text_editor")
        if any(ed in name.lower() for ed in text_editors) and active.startswith("*"):
            logger.info(f"[AGENT] '{active}' has unsaved changes — opening new file")
            global _agent_typing
            _agent_typing = True
            pyautogui.hotkey("ctrl", "n")
            time.sleep(0.5)
            _agent_typing = False
            active = _screen.get_active_window()  # refresh after Ctrl+N

        # Step 6: Verify — check by title first, then by process name
        if name.lower() in active.lower():
            return f"Focused window: '{active}'"

        # Title mismatch — verify by process name (handles apps that rename their window)
        try:
            import psutil
            foreground_hwnd = user32.GetForegroundWindow()
            fg_pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(foreground_hwnd, ctypes.byref(fg_pid))
            proc = psutil.Process(fg_pid.value)
            exe = proc.name().lower().replace(".exe", "")
            name_words = [w for w in name.lower().split() if len(w) > 2]
            if any(word in exe for word in name_words):
                return f"Focused window: '{matches[0].title}'"
        except Exception:
            pass

        return f"Could not bring '{name}' to foreground — window exists but focus failed"
    except Exception as e:
        logger.warning(f"[AGENT] focus_application failed: {e}")
        return f"focus_application failed: {e}"


def _pick_best_match(matches: list, screen_height: int = 1080) -> tuple:
    """
    Pick the best match from OCR results.
    Filters out toolbar area (y < 150) and taskbar area (y > screen_height - 120).
    Among remaining matches, prefers the lowest on screen (highest y value) — because interactive content rows (songs, files, search results) appear lower than decorative headers, banners, and album art near the top of the content area.
    Falls back to first match if nothing passes the filter.
    """
    filtered = [(x, y) for (x, y) in matches if 150 < y < (screen_height - 120)]
    if not filtered:
        return matches[0]
    # Among filtered matches, pick lowest (largest y) — avoids album art / top banners
    return max(filtered, key=lambda m: m[1])


def find_and_click_text(text: str) -> str:
    """
    Find text on screen via OCR and click on it.

    Preferred over mouse_click with raw coordinates — uses
    screen.find_text_on_screen() to locate the element precisely.
    Automatically picks the best match (skipping toolbar/taskbar).

    Returns a result string (success or failure for the verifier to see).
    """
    from ...io import screen

    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: find_and_click_text(\"{text}\")")
    # Guard: single-character OCR matching is extremely unreliable
    if len(text) < 2:
        msg = (f"find_and_click_text FAILED: text \"{text}\" is too short (single char). "
               f"Single-character OCR matching is unreliable. Use keyboard_type(\"{text}\") "
               f"or keyboard_press(\"{text}\") instead.")
        logger.warning(f"[AGENT] {msg}")
        return msg
    try:
        matches = screen.find_text_on_screen(text)
        if not matches:
            msg = f"find_and_click_text FAILED: \"{text}\" not found on screen"
            logger.warning(f"[AGENT] {msg}")
            return msg

        x, y = _pick_best_match(matches)
        pyautogui.click(x, y)
        return f"Found and clicked \"{text}\" at ({x}, {y})"
    except Exception as e:
        logger.error(f"[AGENT] find_and_click_text failed: {e}")
        return f"Failed: {e}"


def find_and_double_click_text(text: str) -> str:
    """
    Find text on screen via OCR and double-click on it.

    Use this instead of find_and_click_text when a double-click is needed
    (e.g., playing a song in a media app, opening a file).
    Automatically picks the best match (skipping toolbar/taskbar).
    """
    from ...io import screen

    if SAFE_MODE:
        logger.info(f"[AGENT] ACTION: find_and_double_click_text(\"{text}\")")
    if len(text) < 2:
        msg = (f"find_and_double_click_text FAILED: text \"{text}\" is too short (single char). "
               f"Use keyboard_type(\"{text}\") or keyboard_press(\"{text}\") instead.")
        logger.warning(f"[AGENT] {msg}")
        return msg
    try:
        matches = screen.find_text_on_screen(text)
        if not matches:
            msg = f"find_and_double_click_text FAILED: \"{text}\" not found on screen. SUGGESTION: Use keyboard_press(tab) to navigate to first search result, then keyboard_press(enter) to play it."
            logger.warning(f"[AGENT] {msg}")
            return msg

        x, y = _pick_best_match(matches)
        pyautogui.doubleClick(x, y)
        return f"Found and double-clicked \"{text}\" at ({x}, {y})"
    except Exception as e:
        logger.error(f"[AGENT] find_and_double_click_text failed: {e}")
        return f"Failed: {e}"


def _snap_to_ocr(x: int, y: int, search_text: str, radius: int = 250) -> tuple[int, int]:
    """
    Snap vision-LLM coordinates (x, y) to the OCR-detected text block whose
    text best matches `search_text`.

    Vision LLMs commonly mis-locate by 100-300px — bigger than the old 120px
    snap radius — so we rank candidates by word-overlap with the search phrase
    first, distance second. A candidate that shares 2+ significant words with
    the search phrase is allowed beyond the base radius (up to a hard cap of
    1/3 of screen width) because multi-word agreement is strong evidence even
    when the LLM coord is far off.

    Returns snapped (x, y) on a confident match, else original (x, y).
    """
    from ...io import screen

    STOPWORDS = {"the", "and", "for", "with", "from", "this", "that", "your"}

    def _significant(text: str) -> set[str]:
        return {
            w.lower().strip(".,!?:;\"'()[]")
            for w in text.split()
            if len(w) > 3 and w.lower() not in STOPWORDS
        } - {""}

    target_words = _significant(search_text)
    if not target_words:
        target_words = {search_text.lower().strip()}

    blocks = screen.list_ocr_blocks(min_confidence=0.6) or []
    if not blocks:
        logger.info(f"[AGENT] OCR snap: no OCR blocks for '{search_text}', using LLM coord ({x},{y})")
        return (x, y)

    # Hard cap: never snap halfway across the screen, even on multi-word agreement.
    try:
        sw, _ = pyautogui.size()
        hard_cap = max(radius, sw // 3)
    except Exception:
        hard_cap = max(radius, 600)

    # Score each block: (matched_word_count, -distance, longer_match_first)
    scored = []
    for blk in blocks:
        block_words = _significant(blk["text"])
        # Also accept substring fallback for short search phrases.
        match_count = len(target_words & block_words)
        if match_count == 0:
            sl = search_text.lower().strip()
            tl = blk["text"].lower().strip()
            if len(sl) >= 4 and (sl in tl or tl in sl):
                match_count = 1
        if match_count == 0:
            continue
        dist = ((blk["x"] - x) ** 2 + (blk["y"] - y) ** 2) ** 0.5
        # Distance gate: 1-word match → base radius; 2+ word match → hard cap.
        if match_count >= 2:
            if dist > hard_cap:
                continue
        else:
            if dist > radius:
                continue
        scored.append((match_count, len(blk["text"]), -dist, blk["x"], blk["y"]))

    if not scored:
        logger.info(
            f"[AGENT] OCR snap: blocks={len(blocks)} but none matched within "
            f"{radius}px (or {hard_cap}px multi-word) of ({x},{y}) for '{search_text}', "
            f"using LLM coord"
        )
        return (x, y)

    # Sort priority: most matched words → longer detected text (placeholder
    # inside an input is longer than the label beside it; prefer the input)
    # → closer to LLM coord.
    scored.sort(reverse=True)
    match_count, _, neg_dist, best_x, best_y = scored[0]
    if (best_x, best_y) != (x, y):
        logger.info(
            f"[AGENT] OCR snap: '{search_text}' snapped ({x},{y}) → ({best_x},{best_y}) "
            f"[words_matched={match_count}, dist={-neg_dist:.0f}px]"
        )
    return (best_x, best_y)


def vision_guided_click(x: int, y: int, text: str, double: bool = False) -> str:
    """
    Click an element using vision LLM coordinates refined by OCR snap.
    
    Vision LLM provides approximate (x, y). OCR searches for `text` near
    that coordinate and snaps to the exact bounding box center if found.
    Falls back to direct coordinate click if OCR finds nothing nearby.
    
    Args:
        x, y:   Approximate coordinates from vision LLM
        text:   The visible text label of the element to click
        double: If True, double-click (for opening/playing items)
    """
    text = " ".join(text.split())  # sanitize whitespace

    # Detect if LLM passed percentage coordinates (0.0-1.0) instead of pixels
    # Convert to pixels if so — handles both float fractions and small integers
    if 0.0 < x <= 1.0 and 0.0 < y <= 1.0:
        screen_w, screen_h = pyautogui.size()
        x = int(x * screen_w)
        y = int(y * screen_h)
        logger.info(f"[AGENT] vision_guided_click: converted percentage coords to pixels: ({x},{y})")

    snapped_x, snapped_y = _snap_to_ocr(x, y, text)
    snap_failed = (snapped_x == x and snapped_y == y)
    locate_source = "ocr" if not snap_failed else "llm_coord"

    # If OCR snap failed (returned original coords) AND the coordinate is in
    # the OS chrome zone (title bar / top toolbar), refuse to click there.
    # Top 150px is almost never a valid click target for content interactions.
    TYPEABLE_CHARS = set("0123456789+-*/=.%()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if snap_failed and len(text.strip()) == 1 and text.strip() in TYPEABLE_CHARS:
        logger.info(f"[AGENT] vision_guided_click: snap failed + single typeable char '{text}' → redirecting to keyboard_type")
        try:
            pyautogui.typewrite(text.strip(), interval=0.05)
            return f"Typed '{text}' via keyboard (redirected from failed vision_guided_click)"
        except Exception as e:
            return f"Failed: {e}"

    # Gemini bbox fallback — when OCR-snap couldn't find the element and the
    # text label is a real description (not a single character handled above),
    # ask Gemini's spatial-understanding mode for the bounding box. This
    # catches the cases OCR misses: empty inputs without placeholder text,
    # light-coloured buttons that drop below EasyOCR's 0.6 confidence
    # threshold, and LLM coords too far from the target for the snap radius.
    # One extra Gemini Flash call per failed-snap click — purpose-built for
    # element localisation, much more accurate than free-form pixel coords.
    if snap_failed and len(text.strip()) > 1:
        try:
            from ...io import screen as _screen
            from ... import llm as _llm
            screenshot_b64 = _screen.capture_screenshot_base64()
            if screenshot_b64:
                bbox_xy = _llm.locate_element_bbox(text, screenshot_b64)
                if bbox_xy is not None:
                    snapped_x, snapped_y = bbox_xy
                    snap_failed = False
                    locate_source = "gemini_bbox"
        except Exception as e:
            logger.debug(f"[AGENT] gemini bbox locate failed (non-fatal): {e}")

    action = "double-click" if double else "click"
    if SAFE_MODE:
        logger.info(
            f"[AGENT] ACTION: vision_guided_click({x},{y}, text='{text}', double={double}) "
            f"→ ({snapped_x},{snapped_y}) via {locate_source}"
        )
    try:
        if double:
            pyautogui.doubleClick(snapped_x, snapped_y)
        else:
            pyautogui.click(snapped_x, snapped_y)
        if locate_source == "ocr":
            snap_note = f" (OCR-snapped from {x},{y})"
        elif locate_source == "gemini_bbox":
            snap_note = f" (Gemini bbox from LLM coord {x},{y})"
        else:
            snap_note = " (no snap, used LLM coord)"
        return f"{action.capitalize()}ed '{text}' at ({snapped_x},{snapped_y}){snap_note}"
    except Exception as e:
        logger.error(f"[AGENT] vision_guided_click failed: {e}")
        return f"Failed: {e}"


def get_browser_tabs() -> str:
    """
    Returns all open browser tab titles using Windows UI Automation.
    Reads from the browser's tab strip (top-most TabControl) only —
    excludes in-page tab elements (e.g. YouTube categories/channels).
    Supports Firefox, Chrome, Brave, Edge.
    """
    try:
        import comtypes.client
        import comtypes.gen

        UIAutomationCore = comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen.UIAutomationClient import (
            CUIAutomation, IUIAutomation, IUIAutomationElement,
            TreeScope_Descendants, UIA_NamePropertyId,
            UIA_ControlTypePropertyId, UIA_TabItemControlTypeId
        )
        # UIA_TabControlTypeId may not be exported — define manually
        UIA_TabControlTypeId = 50018

        automation = comtypes.client.CreateObject(
            "{ff48dba4-60ef-4201-aa87-54103eef594e}",
            interface=IUIAutomation
        )

        # Find browser windows
        import pygetwindow as gw
        all_windows = gw.getAllWindows()
        browser_hwnds = []
        for w in all_windows:
            if w.title.strip() and any(k in w.title.lower() for k in config.BROWSER_NAMES):
                browser_hwnds.append(w._hWnd)

        tabs = []
        for hwnd in browser_hwnds:
            try:
                element = automation.ElementFromHandle(hwnd)

                # --- Strategy: find the browser's tab bar (top-most TabControl) ---
                # Then read TabItems ONLY from that container.
                # This avoids in-page tab elements (YouTube categories, etc.)
                strip_condition = automation.CreatePropertyCondition(
                    UIA_ControlTypePropertyId, UIA_TabControlTypeId
                )
                tab_strips = element.FindAll(TreeScope_Descendants, strip_condition)

                # Pick the TabControl closest to the top of the window
                best_strip = None
                best_top = float('inf')
                for i in range(tab_strips.Length):
                    strip = tab_strips.GetElement(i)
                    try:
                        rect = strip.CurrentBoundingRectangle
                        if rect.top < best_top:
                            best_top = rect.top
                            best_strip = strip
                    except Exception:
                        # If bounding rect unavailable, use first found as fallback
                        if best_strip is None:
                            best_strip = strip

                if best_strip is not None:
                    # Read TabItems from the tab bar only
                    tab_condition = automation.CreatePropertyCondition(
                        UIA_ControlTypePropertyId, UIA_TabItemControlTypeId
                    )
                    tab_elements = best_strip.FindAll(TreeScope_Descendants, tab_condition)
                    for j in range(tab_elements.Length):
                        tab = tab_elements.GetElement(j)
                        name = tab.CurrentName
                        if name and name.strip():
                            tabs.append(name.strip())
                else:
                    # No TabControl found — fall back to whole-window search
                    # with position filtering (only tabs near the top)
                    tab_condition = automation.CreatePropertyCondition(
                        UIA_ControlTypePropertyId, UIA_TabItemControlTypeId
                    )
                    all_tab_elements = element.FindAll(TreeScope_Descendants, tab_condition)
                    try:
                        win_rect = element.CurrentBoundingRectangle
                        win_top = win_rect.top
                    except Exception:
                        win_top = 0
                    for j in range(all_tab_elements.Length):
                        tab = all_tab_elements.GetElement(j)
                        try:
                            rect = tab.CurrentBoundingRectangle
                            # Only include tabs within ~80px of window top edge
                            if rect.top <= win_top + 80:
                                name = tab.CurrentName
                                if name and name.strip():
                                    tabs.append(name.strip())
                        except Exception:
                            continue
            except Exception:
                continue

        if not tabs:
            return "No browser tabs found"

        # Deduplicate while preserving order
        seen = set()
        unique_tabs = []
        for t in tabs:
            if t not in seen:
                seen.add(t)
                unique_tabs.append(t)

        result = "\n".join(f"- {t}" for t in unique_tabs)
        logger.info(f"[AGENT] Browser tabs found ({len(unique_tabs)}): {result[:200]}")
        return result

    except Exception as e:
        logger.warning(f"[AGENT] UI Automation tab read failed: {e}")
        # Fallback: return active tab titles from window titles
        from ...io import screen as _screen
        browser_suffixes = [
            " — Mozilla Firefox", " - Mozilla Firefox",
            " — Google Chrome", " - Google Chrome",
            " — Brave", " - Brave",
            " — Microsoft Edge", " - Microsoft Edge",
        ]
        all_windows = _screen.get_open_windows()
        tabs = []
        for title in all_windows:
            for suffix in browser_suffixes:
                if title.endswith(suffix):
                    tab_title = title[:-len(suffix)].strip()
                    if tab_title:
                        tabs.append(tab_title)
                    break
        if not tabs:
            return "No browser tabs found"
        return "\n".join(f"- {t}" for t in tabs)


# ─── Action Dispatcher ───────────────────────────────────────────────────────

ACTION_MAP = {
    "mouse_move": lambda p: mouse_move(*_pct_to_pixels(p["x_pct"], p["y_pct"])) if "x_pct" in p else mouse_move(p["x"], p["y"]),
    "mouse_click": lambda p: mouse_click(*_pct_to_pixels(p["x_pct"], p["y_pct"])) if "x_pct" in p else mouse_click(p["x"], p["y"]),
    "mouse_double_click": lambda p: mouse_double_click(*_pct_to_pixels(p["x_pct"], p["y_pct"])) if "x_pct" in p else mouse_double_click(p["x"], p["y"]),
    "mouse_right_click": lambda p: mouse_right_click(*_pct_to_pixels(p["x_pct"], p["y_pct"])) if "x_pct" in p else mouse_right_click(p["x"], p["y"]),
    "keyboard_type": lambda p: keyboard_type(p["text"], p.get("interval", 0.05), expected_window=p.get("expected_window")),
    "keyboard_hotkey": lambda p: keyboard_hotkey(*p["keys"], expected_window=p.get("expected_window")),
    "keyboard_press": lambda p: keyboard_press(p["key"], expected_window=p.get("expected_window")),
    "scroll": lambda p: scroll(p["clicks"], p.get("x"), p.get("y")),
    "clipboard_read": lambda p: clipboard_read(),
    "clipboard_write": lambda p: clipboard_write(p["text"]),
    "wait": lambda p: wait(p["seconds"]),
    "open_application": lambda p: open_application(p["name"]),
    "focus_application": lambda p: focus_application(p["name"]),
    "vision_guided_click": lambda p: vision_guided_click(
        p["x"], p["y"], p["text"], p.get("double", False)
    ),
    "find_and_click_text": lambda p: find_and_click_text(p["text"]),
    "find_and_double_click_text": lambda p: find_and_double_click_text(p["text"]),
    "get_browser_tabs": lambda p: get_browser_tabs(),
}


# ─── Dropdown commit safety ───────────────────────────────────────────────────
#
# Failure mode this fixes:
# Planner emits a batch ending with keyboard_press("down") × N to navigate a
# dropdown but never sends keyboard_press("enter"). The highlight moves but
# nothing commits. The next vision pass reads the highlighted row as "selected"
# and the planner moves on with the wrong value chosen.
#
# Two-part fix:
#   1. VISION_PLANNER_SYSTEM_PROMPT now instructs the planner to always commit
#      after arrow navigation (a soft fix — the LLM may still drift).
#   2. _inject_dropdown_commit_if_needed (this code, the hard fix) detects the
#      pattern and injects the missing Enter keystroke before the batch runs.
#
# The guard is gated on TWO conditions to avoid false-injection on chat boxes
# / search-result lists / file-pickers that happen to use arrow navigation:
#   (a) the trailing real action is an arrow keystroke, AND
#   (b) the batch shows dropdown context — either an earlier click in the same
#       batch hit something whose target text overlaps an active select-TODO's
#       field/value, or a `pending_visual_confirm` select-TODO is currently
#       outstanding.
# When both are true, we inject keyboard_press("enter") AFTER the trailing
# arrow press, BEFORE any tail screenshot_and_continue. When either is false
# the actions list is returned unchanged.

_DROPDOWN_COMMIT_ARROW_KEYS = {"down", "up", "pagedown", "pageup"}


def _canonical_token(s: str) -> str:
    """Lowercase, strip surrounding quotes/punctuation, collapse whitespace.
    Used for consistent target/value comparison across matchers."""
    if not isinstance(s, str):
        return ""
    out = s.strip().lower()
    # Drop surrounding quote characters (single, double, smart-quotes)
    out = out.strip("'\"`'‘’“”")
    # Collapse internal whitespace
    out = " ".join(out.split())
    return out


def _action_target_text(action: dict) -> str:
    """Extract the human-readable target text from a click-shaped action."""
    for key in ("text", "target_description", "name"):
        v = action.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _batch_has_recent_dropdown_click(actions: list[dict]) -> bool:
    """
    Return True if any earlier click action in this batch targets text that
    overlaps an active select-TODO's field or value (case-insensitive
    substring either way). Uses the planner's stated targets, not OCR — the
    planner already had to identify the dropdown to click it. Free signal.

    Returns False when:
      - no select-TODOs are active
      - no click actions earlier in the batch
      - no overlap between any click target and any select-TODO field/value
    """
    select_todos = [t for t in _task_state.todo_list
                    if t.get("kind") == "select" and not t.get("done")]
    if not select_todos:
        return False

    click_types = {
        "vision_guided_click",
        "find_and_click_text",
        "find_and_double_click_text",
        "mouse_click",
        "mouse_double_click",
    }
    for a in actions:
        if (a.get("type") or a.get("action") or "") not in click_types:
            continue
        target = _canonical_token(_action_target_text(a))
        if not target:
            continue
        for t in select_todos:
            for needle in (t.get("field") or "", t.get("value") or ""):
                needle_canon = _canonical_token(needle)
                if not needle_canon:
                    continue
                if needle_canon in target or target in needle_canon:
                    return True
    return False


def _inject_dropdown_commit_if_needed(actions: list[dict]) -> list[dict]:
    """
    Scan an action batch for the failure pattern (trailing arrow keystrokes
    with no commit + dropdown context) and inject keyboard_press("enter")
    to commit. Returns the (possibly modified) actions list. Never raises —
    fail-open returns the input unchanged.

    Gated on `config.DROPDOWN_COMMIT_GUARD_ENABLED`; off = no-op.
    """
    if not actions:
        return actions

    try:
        from ... import config as _cfg
        if not getattr(_cfg, "DROPDOWN_COMMIT_GUARD_ENABLED", True):
            return actions
    except Exception:
        # If config import fails, default to enabled (safer to commit than to
        # silently leak the failure pattern).
        pass

    # Find the trailing real action (skip wait + screenshot_and_continue).
    last_real_idx = None
    for i in range(len(actions) - 1, -1, -1):
        atype = actions[i].get("type") or actions[i].get("action") or ""
        if atype in ("screenshot_and_continue", "wait"):
            continue
        last_real_idx = i
        break
    if last_real_idx is None:
        return actions  # only screenshot/wait actions — nothing to commit

    last_real = actions[last_real_idx]
    if (last_real.get("type") or last_real.get("action") or "") != "keyboard_press":
        return actions

    key = str(last_real.get("key", "")).lower()
    if key not in _DROPDOWN_COMMIT_ARROW_KEYS:
        return actions

    # Dropdown context check. Requires todo_list items with `kind`/`field`/
    # `value` fields (populated by the classifier). Tests for this helper
    # synthesize the required TODO state directly.
    has_dropdown_context = (
        _batch_has_recent_dropdown_click(actions)
        or any(t.get("pending_visual_confirm") and t.get("kind") == "select"
               for t in _task_state.todo_list)
    )
    if not has_dropdown_context:
        return actions

    # Insert Enter immediately AFTER the trailing arrow press, BEFORE any
    # screenshot_and_continue / wait that follow it.
    insert_at = last_real_idx + 1
    new_actions = list(actions)
    new_actions.insert(insert_at, {"type": "keyboard_press", "key": "enter"})
    logger.info(
        "[AGENT] injected keyboard_press(enter) after trailing arrow "
        f"navigation (batch had {len(actions)} action(s), now {len(new_actions)})"
    )
    return new_actions


def execute_all_actions(actions: list[dict]) -> list[str]:
    """
    Execute a list of action dictionaries sequentially.

    Each action dict should have:
        {"action": "mouse_click", "params": {"x": 100, "y": 200}}

    Returns:
        List of result strings for each action.
    """
    global _search_activated_this_batch
    _search_activated_this_batch = False

    results = []

    for i, action in enumerate(actions):
        # Detail prefers the action verb (e.g. "click", "type"); falls back
        # to a rotating descriptive label from theme.READING_FALLBACK_LABELS.
        _act_type = getattr(action, "type", None) or (
            action.get("type") if isinstance(action, dict) else None
        )
        if _act_type:
            _act_label = str(_act_type).replace("_", " ")[:32]
        else:
            from assistant.io.overlay.theme import READING_FALLBACK_LABELS, rotating_label
            _act_label = rotating_label(READING_FALLBACK_LABELS, i)
        _status_broadcaster.set(_StatusPhase.READING,
                                detail=_act_label,
                                step=(i + 1, len(actions)),
                                tier="vision")
        if _check_abort():
            raise _UserAborted("esc_hold")

        if i >= MAX_STEPS:
            results.append(f"[WARN] Reached max {MAX_STEPS} actions, stopping")
            break

        logger.debug(f"[AGENT] Executing action: {json.dumps(action)}")
        action_name = action.get("type") or action.get("action") or ""
        params = action.get("params", action)

        handler = ACTION_MAP.get(action_name)
        
        if action_name == "screenshot_and_continue":
            # This is a control flow signal — not a real action
            # It tells the loop to stop executing and replan with fresh screenshot
            # Return a special sentinel that the loop checks for
            results.append("SCREENSHOT_AND_CONTINUE")
            logger.info("[AGENT] Screenshot checkpoint — stopping batch, replanning")
            break  # Stop executing remaining actions, trigger replan

        if action_name == "keyboard_press":
            key_val = action.get("key", params.get("key", ""))
            MODIFIER_KEYS = {"ctrl", "alt", "shift", "win", "command"}
            if key_val.lower() in MODIFIER_KEYS:
                msg = f"SKIPPED: '{key_val}' is a modifier key and cannot be pressed alone. Use keyboard_hotkey for combinations like ctrl+s."
                logger.warning(f"[AGENT] Skipping lone modifier key '{key_val}' \u2014 use keyboard_hotkey instead")
                results.append(msg)
                continue

        if action_name == "keyboard_hotkey":
            keys = params.get("keys", [])
            keys_set = set(k.lower() for k in keys)
            if keys_set == {"ctrl", "k"} or keys_set in ({"ctrl", "t"}, {"ctrl", "l"}, {"ctrl", "f"}):
                _search_activated_this_batch = True

        if handler is None:
            msg = f"Unknown action: {action_name}"
            logger.warning(f"[AGENT] Unknown action type '{action_name}' \u2014 full action: {json.dumps(action)}")
            results.append(msg)
            continue

        try:
            result = handler(params)
            results.append(result)

            # ── Cache data-fetching results ──
            _task_state.cache_result(action_name, result)

            # ── Auto-paste: get_browser_tabs → clipboard paste into focused window ──
            if action_name == "get_browser_tabs" and result.startswith("- "):
                if _task_state.is_done("auto_paste_tabs"):
                    logger.info("[AGENT] Browser tabs already pasted — skipping duplicate")
                    results.append("Browser tabs already written — do NOT call get_browser_tabs again. Set done: true.")
                    continue
                from ...io import screen as _scr
                active_win = _scr.get_active_window()
                active_win_lower = active_win.lower()
                is_browser = any(b in active_win_lower for b in config.BROWSER_NAMES)
                if is_browser:
                    logger.info("[AGENT] Skipping auto-paste — active window is a browser")
                    results.append("SCREENSHOT_AND_CONTINUE")
                    break
                # If Notepad has unsaved content, open a new file first
                if active_win.startswith("*") and "notepad" in active_win_lower:
                    logger.info("[AGENT] Notepad has unsaved content — opening new file before paste")
                    _agent_typing = True
                    pyautogui.hotkey("ctrl", "n")
                    time.sleep(0.5)
                    _agent_typing = False
                try:
                    import pyperclip
                    pyperclip.copy(result)
                    _agent_typing = True
                    pyautogui.hotkey("ctrl", "v")
                    time.sleep(0.5)
                    _agent_typing = False
                    _task_state.mark_done("auto_paste_tabs")
                    results.append("AUTO-TYPED browser tabs into focused window via clipboard paste. Task is DONE — set done: true.")
                    logger.info("[AGENT] Auto-typed browser tabs via clipboard paste")
                except Exception as paste_err:
                    _agent_typing = False
                    logger.warning(f"[AGENT] Auto-paste failed: {paste_err}")
                    results.append("SCREENSHOT_AND_CONTINUE")
                    logger.info("[AGENT] get_browser_tabs complete — stopping batch for LLM to process results")
                    break
            # Auto-wait after opening an app so it has time to load
            if action_name == "open_application":
                result_str = results[-1] if results else ""
                if "already open" not in result_str.lower() and "focused" not in result_str.lower():
                    logger.info("[AGENT] Waiting 5s for application to load...")
                    time.sleep(5.0)
            # Auto-wait after focus_application so the window settles
            # before any subsequent keyboard actions
            if action_name == "focus_application":
                time.sleep(0.3)
            # Auto-wait after pressing Enter — gives pages/dialogs time to load
            if action_name == "keyboard_press":
                key_val = (action.get("key") or params.get("key", "")).lower()
                if key_val == "enter":
                    logger.info("[AGENT] Waiting 1.5s for page/dialog to load after Enter...")
                    time.sleep(1.5)
        except Exception as e:
            msg = f"Action '{action_name}' error: {e}"
            logger.error(f"[AGENT] {msg}")
            results.append(msg)

    return results


# ─── Computer Agent System Prompt ────────────────────────────────────────────

COMPUTER_AGENT_SYSTEM_PROMPT = """\
You are a computer control agent. You can see the user's screen and control their mouse and keyboard to accomplish tasks.

Based on the screen context provided, plan a sequence of actions to accomplish the user's goal.

Available actions:
- mouse_move: {"action": "mouse_move", "params": {"x": 100, "y": 200}}
- mouse_click: {"action": "mouse_click", "params": {"x": 100, "y": 200}}
- mouse_double_click: {"action": "mouse_double_click", "params": {"x": 100, "y": 200}}
- mouse_right_click: {"action": "mouse_right_click", "params": {"x": 100, "y": 200}}
- find_and_click_text: {"action": "find_and_click_text", "params": {"text": "Settings"}}
- find_and_double_click_text: {"action": "find_and_double_click_text", "params": {"text": "song title"}}
  (Use for playing media items — single click selects, double click plays)
- keyboard_type: {"action": "keyboard_type", "params": {"text": "hello"}}
- keyboard_hotkey: {"action": "keyboard_hotkey", "params": {"keys": ["ctrl", "c"]}}
- keyboard_press: {"action": "keyboard_press", "params": {"key": "enter"}}
- scroll: {"action": "scroll", "params": {"clicks": -3}}
- clipboard_read: {"action": "clipboard_read", "params": {}}
- clipboard_write: {"action": "clipboard_write", "params": {"text": "hello"}}
- wait: {"action": "wait", "params": {"seconds": 1.0}}
- open_application: {"action": "open_application", "params": {"name": "chrome"}}
- screenshot_and_continue: {"action": "screenshot_and_continue", "params": {}}
  (Use this LAST to take a new screenshot and re-plan if you need to verify or continue)

⚡ IMPORTANT: ALWAYS prefer find_and_click_text over mouse_click when clicking UI elements \
that have visible text labels. Only use mouse_click with raw coordinates as a last resort \
when the element has no visible text label (e.g., an icon-only button, a specific pixel region, \
or a location only identifiable by coordinates).

IMPORTANT WINDOW TITLE CONVENTIONS:
- Music apps show the currently playing song as the window title
  e.g. "Ed Sheeran - Perfect" means music is playing and "Perfect" by Ed Sheeran is the track
- Browsers show page title as window title
- If active window is "{Artist} - {Song}" format, music IS playing

When planning music playback goals, if the active window title contains \
the song name or artist name from the goal, the music app IS already open AND playing. Do NOT re-open it.

NEVER RE-OPEN APPS: If an app appears in the OPEN WINDOWS list OR if you already opened it \
in a previous loop, do NOT call open_application again. Instead, focus on interacting with \
the already-open app. If the app is not the active window, use keyboard_hotkey('alt', 'tab') \
to bring it to the foreground.

DOUBLE-CLICK FOR MEDIA PLAYBACK: In most media apps, to play a song from search \
results you must use find_and_double_click_text on the song title — single click only selects, \
double click plays. Use find_and_click_text for navigation/buttons, find_and_double_click_text \
for playing media items.

SMART MATCH SELECTION: When multiple matches of the same text exist on screen, the agent \
automatically skips toolbar and taskbar matches and clicks the content area match. You do not \
need to handle this — just call find_and_click_text or find_and_double_click_text normally.

KEYBOARD FALLBACK: If find_and_double_click_text fails on a search result, use \
keyboard_press('tab') once to move focus to the first search result, then keyboard_press('enter') \
to select/play it. This is the keyboard fallback for when OCR cannot locate the item visually.

APP LOAD TIMES: After open_application, the system waits automatically. But some apps \
load slowly — always wait at least 4 seconds before any keyboard input after opening \
heavy apps like music players or IDEs.

APP-SPECIFIC SHORTCUTS (always use these instead of find_and_click_text):
- To search or navigate in a browser: ALWAYS open a new tab first with keyboard_hotkey('ctrl', 't'), then keyboard_type the search query or URL directly (the new tab address bar is auto-focused), then keyboard_press('enter'). NEVER type into an existing tab — always use a new tab to preserve the user's open pages.
- To focus browser address bar in current tab (only if explicitly needed): keyboard_hotkey('ctrl', 'l')
- To open a new browser tab: keyboard_hotkey('ctrl', 't')

KEYBOARD OVER CLICKS: If an application has a focused input area, text field, or accepts keyboard \
shortcuts that accomplish the same thing as clicking small on-screen buttons, ALWAYS use keyboard_type \
or keyboard_press instead of mouse_click or find_and_click_text. Typing is instant and exact; clicking \
small buttons with estimated coordinates frequently misses. This applies to: calculators (type digits \
and operators like 1234*56= directly), terminals, search bars, text editors, form fields, etc. \
For Calculator specifically: type the entire expression using keyboard_type (digits, +, -, *, /) \
then keyboard_press("enter") or keyboard_type("="). NEVER use find_and_click_text on calculator buttons.

NEVER USE find_and_click_text FOR SINGLE CHARACTERS: Single-character OCR matching is extremely \
unreliable — a search for "C" may match "Code", "Calculator", "Chrome", etc. anywhere on screen. \
For any single character (digits, operators, letters), always use keyboard_type or keyboard_press.

ACTIVATE BEFORE TYPING: NEVER use keyboard_type to enter a search query or input without FIRST \
activating the correct input field. Before typing into any search bar, address bar, or input field, \
you must either click on it or use the app's search keyboard shortcut (common ones: Ctrl+K, Ctrl+L, \
Ctrl+F, Ctrl+E, or / for slash-search). If you just type without activating the input field first, \
the keystrokes may trigger unintended actions (like app shortcuts or hotkeys).

Respond with ONLY a JSON object:
{
  "thinking": "Brief explanation of your plan",
  "actions": [
    {"action": "action_name", "params": {...}},
    ...
  ],
  "done": true/false,
  "summary": "What was accomplished (if done=true)"
}

Rules:
- Keep action sequences short (max 15 actions). Keep batches to maximum 4-5 actions when navigating unfamiliar UI. End with screenshot_and_continue to verify state before proceeding. Only use longer batches for repetitive mechanical actions like filling a form.
- Use open_application to launch programs via Windows search
- Use screenshot_and_continue as the LAST action if you need to see results before continuing
- Set done=true when the goal is achieved
- Be precise with coordinates based on the OCR text positions
- SEE-BEFORE-ACT ON SEARCHES: After typing in any search bar and pressing enter, you MUST end your action list with screenshot_and_continue. Never click or interact with search results in the same action batch as the search — always re-plan after seeing what appeared.
- PREFER RESULTS OVER PERFECT MATCHES: When search results appear, pick the closest match to the goal — do not re-search if results are visible. If the user asked for 'Lo-Fi music' and you see 'Lo-Fi Beats playlist', click it.
- CHECK PLAYER STATE BEFORE SEARCHING: When the goal is to play a specific song, video, or any media item, and the app is already open — look at the player bar at the bottom of the screen FIRST. If the exact item requested is already loaded in the player (title matches) but is paused, do NOT search or click anything — just press the play button or use keyboard_press("space") to resume. Only start a search if the wrong item is loaded or the player is empty. This applies to all media apps. NEVER click on a song title shown in the mini-player tray or bottom bar as a navigation action — that opens the album/playlist, it does not play the song.
- PLAYING SONGS IN MEDIA APPS: When search results appear in a music app, NEVER double-click the "Top result" panel (the large card on the left) — it opens an album or playlist, not the song. Instead: (1) first use find_and_click_text to click the "Songs" filter/tab if visible, then (2) use find_and_double_click_text on the song title from the songs list. If there is no "Songs" tab, use find_and_double_click_text on the song title that appears in a list/row format (not the large card). This applies to any media app with search results.
"""


VISION_PLANNER_SYSTEM_PROMPT = """\
You are a computer control agent. You will receive a screenshot of the current screen and a goal to accomplish.

Analyze the screenshot carefully — identify open windows, UI elements, buttons, text fields, and the current state.

Return a JSON action plan with this exact structure:
{
  "thinking": "1-3 short sentences explaining what you'll do this step. Include the FULL action sequence (e.g. 'tab to focus, then type X' — never just 'type X').",
  "plan": "one sentence summary",
  "actions": [...],
  "done": false
}

KEEP `thinking` CONCISE (≤3 sentences) but COMPLETE — when you describe the
action sequence, name every step (focus + type, not just type; click + arrow +
enter, not just arrow). Skipping a step in `thinking` predicts skipping it in
`actions` too. Do NOT spend tokens narrating what's already-filled or
describing pixel layouts in detail — those are visible in the screenshot.

ACTION TYPES — use exactly these formats:

Visual click — ONLY for UI elements without readable text (icons, images, graphical buttons):
{"type": "mouse_click", "target_description": "what you're clicking", "x_pct": 0.5, "y_pct": 0.5}
⚠️ Do NOT use mouse_click to play or open items that have visible text titles — use find_and_double_click_text instead.

Visual double-click:
{"type": "mouse_double_click", "target_description": "what you're clicking", "x_pct": 0.5, "y_pct": 0.5}

Vision-guided click — USE THIS as the default click action for any element 
with visible text. Combines your coordinate estimate with OCR for pixel-perfect accuracy:
{"type": "vision_guided_click", "x": 295, "y": 717, "text": "exact visible text of element", "double": false}
Set double=true when opening/playing items (songs, files).
Prefer this over mouse_click (no OCR snap) and find_and_double_click_text 
(no vision guidance, picks wrong instance when text appears multiple times).

Text-based click (when element has clear text label — more reliable):
{"type": "find_and_click_text", "text": "exact text to find"}

Text-based double-click — USE THIS to play/open items in media apps and file lists:
{"type": "find_and_double_click_text", "text": "exact text to find"}
⚠️ MANDATORY: When your goal involves PLAYING a song, video, or file and you can see its title text on screen, you MUST use find_and_double_click_text. Never use mouse_click with coordinates for this — coordinate clicks frequently miss by a few pixels. find_and_double_click_text uses OCR to find the exact text position and is always more accurate.

- SEARCH FLOW — always follow this exact sequence, no exceptions:
  1. keyboard_hotkey ctrl+k  (activate search — MANDATORY first step)
  2. keyboard_type with the search query
  3. keyboard_press enter
  4. screenshot_and_continue  (MANDATORY — never interact with results you haven't seen)
  Never skip step 1. Never combine steps. Never type before activating search.
{"type": "keyboard_press", "key": "enter", "expected_window": "Notepad"}
{"type": "keyboard_hotkey", "keys": ["ctrl", "s"], "expected_window": "Notepad"}

Focus an already-open application (faster than open_application):
{"type": "focus_application", "name": "Notepad"}

App launch:
{"type": "open_application", "name": "app name"}

Wait:
{"type": "wait", "seconds": 1.0}

Get open browser tabs — returns tab titles as text for use in keyboard_type:
{"type": "get_browser_tabs"}

Checkpoint (take new screenshot before continuing):
{"type": "screenshot_and_continue"}

CRITICAL RULES:
- Always include expected_window in keyboard actions when working inside a specific app
- NEVER add expected_window to system-wide hotkeys that work regardless of which window is focused. These hotkeys must be sent without expected_window: Win+D (show desktop), Win+E (file explorer), Win+L (lock screen), Win+Tab (task view), Alt+F4 (close window), PrintScreen. Adding expected_window to these will cause ABORTED_WRONG_FOCUS because no app "owns" them.
- READING BROWSER TABS: Always call get_browser_tabs as the ONLY action in a batch — it will auto-paste the tab list into the focused window automatically via clipboard. Do NOT call get_browser_tabs more than once per task. After calling it, use screenshot_and_continue to verify the result was pasted, then set done: true if it looks correct.
- NO DUPLICATE WRITING: Before typing content into any text editor, look at the screenshot carefully. If the content you are about to type is already visible in the editor, set done: true instead of typing again. Never write the same list or content twice.
- If execution results contain 'ABORTED_WRONG_FOCUS', immediately add focus_application \
  for the correct app then retry the failed keyboard action
- expected_window is a substring match — use short names: "Notepad", "Spotify", "Firefox"
- For Save dialogs, expected_window should be the dialog title e.g. "Save As"
- x_pct and y_pct are fractions of screen width/height (0.0 to 1.0). Look carefully at element positions in the screenshot.
- Prefer find_and_click_text for anything with readable text — it is more precise than coordinate estimation
- Use mouse_click with x_pct/y_pct for icons, images, graphical buttons without text labels
- After open_application, ALWAYS add {"type": "wait", "seconds": 5} then {"type": "screenshot_and_continue"}
- BROWSER NAVIGATION: Whenever you need to search or open a URL in a browser, always: (1) focus the browser with focus_application, (2) open a new tab with keyboard_hotkey('ctrl', 't'), (3) type the search query or URL directly — the new tab address bar is always auto-focused and ready. Never navigate in existing tabs.
- KEYBOARD OVER CLICKS: If an app has a focused input area or accepts keyboard input for the same \
  actions as its on-screen buttons, ALWAYS use keyboard_type or keyboard_press instead of mouse_click \
  or find_and_click_text. For Calculator: type the entire expression (e.g. keyboard_type("1234*56=")) \
  — NEVER click calculator buttons with find_and_click_text.
- NEVER find_and_click_text FOR SINGLE CHARACTERS: Searching for "C", "=", "*" etc. via OCR is \
  unreliable — it matches unrelated text elsewhere on screen. Always use keyboard_type or keyboard_press.
- ACTIVATE BEFORE TYPING: NEVER use keyboard_type to enter a search query or input without FIRST \
  activating the correct input field. Click on it or use the app's search shortcut (Ctrl+K, Ctrl+L, \
  Ctrl+F, Ctrl+E, or /). Typing without activating the input field first may trigger unintended \
  app shortcuts instead of entering text.
- FORM FILLING — TAB BETWEEN FIELDS, DON'T CLICK EACH ONE: When filling a multi-field form \
  (signup, login, contact form, registration, checkout, SSO email/password, etc.) follow this \
  exact pattern: \
  (1) vision_guided_click on the FIRST input field — use its placeholder text (e.g. "Enter your \
  full name") if visible; \
  (2) keyboard_type the value; \
  (3) keyboard_press("tab") to move focus to the next input; \
  (4) keyboard_type the next value; \
  (5) repeat tab+type for every remaining field in DOM order; \
  (6) AFTER the last field, vision_guided_click the submit/next/continue button. \
  Do NOT vision_guided_click on each field — empty inputs (no placeholder) and label text above \
  fields cannot be OCR-snapped, so per-field clicks frequently land on the LABEL or in dead \
  space. Tab is deterministic and skips this entire failure mode. Same pattern applies to SSO \
  flows: click email field → type email → tab → type password → click "Sign in" / "Next". \
  If a Tab unexpectedly lands on a non-input (e.g. a checkbox or link), fall back to \
  vision_guided_click for that single step, then resume tabbing.
- DROPDOWN COMMIT — every keyboard-arrow navigation must end with a commit. When a \
  batch contains keyboard_press("down"), keyboard_press("up"), keyboard_press("pagedown") \
  or keyboard_press("pageup") to navigate a list/dropdown/combobox, the LAST \
  non-screenshot_and_continue action of that batch MUST be either keyboard_press("enter") \
  or a vision_guided_click on the highlighted row. Bare arrow-key sequences NEVER commit \
  a selection — the highlight is not the selection. Violating this rule causes the \
  dropdown to close with no value chosen and the agent will incorrectly believe a value \
  was selected on the next vision pass.
- TODO PROGRESS SYMBOLS — the TODO PROGRESS block uses three symbols that change \
  what you should do:
    ✓ DONE — already completed; never re-attempt.
    · AWAITING CONFIRM — the agent already attempted this step; an automatic background \
      check is verifying the outcome. NEVER re-attempt a `·` TODO. The planner's job is \
      to move on to the NEXT `✗` TODO and let the confirmation finish in parallel.
    ✗ OPEN — not yet attempted; the first `✗` (marked `← NEXT`) is what to do this batch.
  Re-attempting `·` TODOs causes infinite retry loops and burns the loop budget. Trust the \
  symbol: if you see `·`, that step is in flight — work on something else.
- After any search, ALWAYS end batch with screenshot_and_continue — never interact with results you haven't seen yet
- Maximum 5 actions per batch before screenshot_and_continue
- NEVER use keyboard_hotkey alt+tab — use open_application to switch focus instead
- If you see the goal is already accomplished in the screenshot, set done: true immediately
- ALL OPEN APPLICATIONS are listed in the context — check this before opening anything

COORDINATE GUIDANCE:
- Top of screen: y_pct ≈ 0.0-0.08 (title bars, menu bars, browser tabs)
- Taskbar: y_pct ≈ 0.92-1.0
- Content area: y_pct ≈ 0.08-0.92
- Never click y_pct < 0.08 or > 0.92 for content interactions
"""


# ─── Plan-and-Execute TODO Tracking ───────────────────────────────────────────
#
# An independent completeness signal that does NOT rely on vision verification.
# Generated once at task start, updated after each action batch via cheap text
# LLM calls. Fixes the verifier-hallucinates-completion bug observed on
# form-fill tasks where the vision verifier confused placeholder text for
# filled values, or treated "submitted" dialogs as success when half the form
# was empty.

TODO_GENERATOR_SYSTEM_PROMPT = """\
You decompose a user's goal into a complete checklist of discrete UI steps.

You will receive:
  - GOAL: what the user wants accomplished
  - A screenshot of the current screen

Return a JSON list of short imperative strings — each one a single concrete UI action
that must happen for the goal to be complete. The list is the agent's TODO checklist.

RULES:
- Each item must be ONE action: typing into one field, clicking one button, selecting
  one dropdown option. Not "fill the form" — break it down.
- Use the EXACT visible field labels and values when possible. Example:
  "Type 'test@example.com' in Work Email" — not "fill email field".
- Include the FINAL action that finishes the goal (Submit, Send, Save, Schedule a Demo,
  Sign in, etc.). Do not stop at "fill all fields" — the user wants the form SUBMITTED
  unless they explicitly said otherwise.
- If the user explicitly limited scope ("just fill the fields, don't submit"; "draft
  but don't send"), respect that — omit the final action.
- Hard limit: 15 items maximum. If a goal genuinely needs more, list the first 15
  highest-priority items.
- Skip steps that are already obviously complete in the screenshot (e.g. an already-open
  app, a pre-filled default, a checkbox already in the right state).
- For tasks with no clear discrete steps (browse, watch, look at, read), return an
  empty list [] — the agent will fall back to vision-based verification.
- For trivial single-action goals (open Notepad, click X), return a list with that
  one item.

Return ONLY the JSON list, no prose. Examples:

["Type 'Test' in First Name",
 "Type 'User' in Last Name",
 "Type 'Test Company' in Company Name",
 "Select '1-50' from Staff Size dropdown",
 "Click 'Schedule a Demo' button"]

[]
"""


TODO_UPDATER_SYSTEM_PROMPT = """\
You track which checklist items just got completed by the agent's last batch of actions.

You will receive:
  - The TODO list (each item: id + task text + done-status)
  - The actions that were just executed and their results

Decide:
  1. Which TODO items are NOW COMPLETE based on what just happened?
     A typed value in the right field → that "Type X in Y" item is done.
     A successful click on the right button → that "Click X" item is done.
     A dropdown selection → that "Select X from Y" item is done.
     Be conservative: only mark done if the action result clearly indicates success.
     ABORTED / FAILED action results → do NOT mark the item done.
  2. Are there any NEW TODO items now needed that were not in the original list?
     Common cause: a dropdown selection revealed a new dependent field
     (e.g. selecting State revealed a City dropdown). Add only items the
     existing list does not cover. Use the same short-imperative format as
     the existing items. Skip if nothing genuinely new is required.

Return ONLY a JSON object:
{
  "completed": [list of integer ids that are now done],
  "new":       [list of new TODO strings to append]
}

If you are unsure, return empty arrays — the agent will rely on later batches.
"""


async def _generate_initial_todos(goal: str, screenshot_b64: str | None) -> list[str]:
    """
    One-shot vision call at task start. Asks the LLM to decompose the
    goal into a concrete UI-step checklist. Returns the list of task strings,
    or [] on any failure (fail-open — the agent falls back to verifier-only).
    """
    from ... import llm
    from ...llm.contracts import ask_for_plan
    prompt = (
        f"GOAL: {goal}\n\n"
        f"Decompose this goal into the complete checklist of discrete UI actions "
        f"needed. Return ONLY a JSON list of short strings, max 15 items."
    )
    try:
        if screenshot_b64:
            raw = (await llm.get_vision_response(
                image_base64=screenshot_b64,
                prompt=prompt,
                system_prompt=TODO_GENERATOR_SYSTEM_PROMPT,
                json_mode=True,
            )).text
        else:
            raw = await ask_for_plan(
                prompt,
                system_prompt=TODO_GENERATOR_SYSTEM_PROMPT,
                json_mode=True,
            )
    except Exception as e:
        logger.warning(f"[AGENT] initial TODO LLM call failed: {e}")
        return []

    if not raw or raw == "__LLM_UNAVAILABLE__":
        return []
    return _parse_todo_list(raw)


def _parse_todo_list(raw: str) -> list[str]:
    """
    Extract a JSON list of strings from the LLM response. Tolerant of:
      - bare list (preferred): ["a", "b"]
      - object wrapper: {"todos": [...]} or {"items": [...]}
      - code-fenced output: ```json\n[...]\n```
      - extra prose before/after
    Returns [] on any parse failure (fail-open).
    """
    if not isinstance(raw, str):
        return []
    text = raw.strip()
    if not text:
        return []

    import re
    # Strip code fences if present
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    candidates = []
    # Try direct parse
    candidates.append(text)
    # Extract first [...] block
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        candidates.append(text[bracket_start:bracket_end + 1])
    # Extract first {...} block (object wrapper case)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidates.append(text[brace_start:brace_end + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, list):
            extracted = [str(x) for x in parsed
                         if isinstance(x, (str, int, float)) and str(x).strip()]
            if extracted:
                return extracted
            # Empty list at top level — fall through to the next candidate;
            # don't return [] prematurely (audit fix #9).
            continue
        if isinstance(parsed, dict):
            # Audit fix #2: removed "completed" / "new" from this key list —
            # those are the TODO *updater*'s output keys, not the generator's.
            # Including them caused a wrong-output bug when a generator
            # response happened to use the same shape.
            # Audit fix #9: skip empty-list values so `{"list": [], "tasks":
            # ["a","b"]}` no longer silently shadows the populated key.
            for key in ("todos", "items", "tasks", "checklist", "list"):
                v = parsed.get(key)
                if isinstance(v, list):
                    extracted = [str(x) for x in v
                                 if isinstance(x, (str, int, float)) and str(x).strip()]
                    if extracted:
                        return extracted
    return []


async def _update_todos_after_batch(
    actions: list[dict],
    results: list[str],
    plan_thinking: str,
) -> tuple[int, int]:
    """
    3-pass TODO state update after each action batch.
      Pass 1 — Visual confirm: vision-LLM yes/no on any pending_visual_confirm
               select-TODOs (deferred from a previous batch's Rule S match).
               Strikes counter; permissive fallback after 3 strikes.
      Pass 2 — Action-signature matching: deterministic Rules T/S/C against
               the open TODOs. No LLM calls. Marks Type/Click TODOs done;
               defers Select TODOs to pending_visual_confirm.
      Pass 3 — LLM last resort: only fired for TODOs the rules didn't reach
               (kind="other") or when new-TODO discovery is needed (e.g. a
               cascading dropdown reveal). Same Flash-Lite call, but narrower
               prompt with explicit "only consider these listed TODOs."

    Mutates `_task_state.todo_list` in place. Returns (marked_count, added_count)
    where marked_count includes all three passes' done-marks and added_count
    is from Pass 3 only (rules don't add new TODOs).

    Fail-open everywhere: returns (0, 0) on early-exit conditions; individual
    pass failures don't block the next pass.

    Kill-switch: when `config.DETERMINISTIC_MATCHING_ENABLED` is False,
    skips Passes 1 + 2 entirely and falls back to the text-only path
    (Pass 3 with all open TODOs in scope).
    """
    if not _task_state.todo_list:
        return 0, 0
    if not actions or not results:
        return 0, 0

    # Kill-switch — revert to text-only behaviour when disabled.
    deterministic_matching_on = True
    try:
        from ... import config as _cfg
        deterministic_matching_on = getattr(_cfg, "DETERMINISTIC_MATCHING_ENABLED", True)
    except Exception:
        pass

    marked_total = 0
    added_total = 0
    addressed_ids: set[int] = set()

    if deterministic_matching_on:
        # ── Pass 1: visual-confirm pending select-TODOs ──
        try:
            confirmed, _fb = await _confirm_pending_select_todos(recent_action_results=results)
            marked_total += confirmed
        except Exception as e:
            logger.warning(f"[AGENT] pass 1 (visual confirm) crashed: {e}")

        # ── Pass 2: deterministic rule matching ──
        try:
            addressed_ids, marked_p2 = _match_actions_to_todos(actions, results)
            marked_total += marked_p2
        except Exception as e:
            logger.warning(f"[AGENT] pass 2 (rule matching) crashed: {e}")
            addressed_ids = set()

    # ── Pass 3: LLM last-resort for unresolved TODOs + new-TODO discovery ──
    # When deterministic matching is disabled, all open TODOs are unresolved
    # (legacy path). When enabled, only TODOs the rules didn't reach AND that aren't
    # currently pending_visual_confirm need the LLM.
    if deterministic_matching_on:
        unresolved = [
            t for t in _task_state.todo_list
            if not t["done"]
            and not t.get("pending_visual_confirm")
            and t["id"] not in addressed_ids
            and t.get("kind") in ("other", "")  # "" defends against pre-classifier dicts
        ]
    else:
        unresolved = [t for t in _task_state.todo_list if not t["done"]]

    # Skip the LLM call entirely when there are no unresolved TODOs left for
    # the LLM to judge (deterministic matching enabled). New-TODO discovery (e.g. cascading
    # dropdowns) is sacrificed as a tradeoff — the planner re-encounters
    # newly-revealed fields on the next iteration via its own vision pass
    # and adds them to its plan organically. This keeps cost predictable:
    # LLM only fires for kind="other" TODOs the rules can't structurally
    # judge (Submit form, navigate, etc.).
    if deterministic_matching_on and not unresolved:
        if marked_total or added_total:
            logger.info(
                f"[AGENT] rules-only update: marked {marked_total} done "
                f"(skipped LLM — no unresolved TODOs)"
            )
        return marked_total, added_total

    # Build the LLM prompt — narrowed to unresolved TODOs only when deterministic matching is on.
    visible_todos = unresolved if deterministic_matching_on else _task_state.todo_list
    todo_lines = []
    for item in visible_todos:
        status = "DONE" if item["done"] else "open"
        todo_lines.append(f"  [{item['id']}] ({status}) {item['task']}")
    todo_block = "\n".join(todo_lines) if todo_lines else "  (none)"

    action_lines = []
    for a, r in zip(actions, results):
        a_type = a.get("type") or a.get("action") or "?"
        target = a.get("text") or a.get("target_description") or a.get("name") or ""
        target_part = f" '{target}'" if target else ""
        action_lines.append(f"  - {a_type}{target_part} → {r}")
    action_block = "\n".join(action_lines)

    extra_rule = ""
    if deterministic_matching_on:
        extra_rule = (
            "\n\nIMPORTANT: only consider TODOs explicitly listed below. "
            "Other TODOs were already judged by deterministic rules — do NOT "
            "guess about them or include them in 'completed'."
        )

    prompt = (
        f"PLAN INTENT (what this batch was meant to do):\n{plan_thinking or '(none)'}\n\n"
        f"ACTIONS THAT JUST RAN:\n{action_block}\n\n"
        f"OPEN TODOS YOU MAY MARK:\n{todo_block}{extra_rule}\n\n"
        f"Which listed TODO ids are NOW complete? Are any NEW TODOs needed "
        f"that the list doesn't already cover?\n"
        f"Return JSON: {{\"completed\": [ids], \"new\": [strings]}}."
    )

    from ...llm.contracts import ask_for_synthesis
    try:
        raw = await ask_for_synthesis(
            prompt,
            system_prompt=TODO_UPDATER_SYSTEM_PROMPT,
            json_mode=True,
            max_tokens=300,
        )
    except Exception as e:
        logger.warning(f"[AGENT] TODO update LLM call failed: {e}")
        return marked_total, added_total

    if not raw or raw == "__LLM_UNAVAILABLE__":
        return marked_total, added_total

    completed_ids, new_tasks = _parse_todo_update(raw)

    # Guard: only allow the LLM to mark TODOs that were in the visible
    # subset. Without this, a hallucinating model could "complete" a TODO
    # the rules already deferred (e.g. a select-TODO awaiting visual confirm).
    visible_ids = {t["id"] for t in visible_todos}
    marked_p3 = 0
    for tid in completed_ids:
        if tid in visible_ids and _task_state.mark_todo_done(tid):
            marked_p3 += 1
    marked_total += marked_p3

    added_p3 = 0
    for new_text in new_tasks:
        if _task_state.add_todo(new_text) is not None:
            added_p3 += 1
    added_total += added_p3

    if marked_total or added_total:
        logger.info(
            f"[AGENT] TODO update: marked {marked_total} done, "
            f"added {added_total} new (LLM contribution: {marked_p3} marks, {added_p3} adds)"
        )
    return marked_total, added_total


def _parse_todo_update(raw: str) -> tuple[list[int], list[str]]:
    """
    Parse the per-batch TODO updater response.
    Expected shape: {"completed": [int, ...], "new": [str, ...]}.
    Returns (completed_ids, new_tasks). Empty tuple on parse failure (fail-open).
    """
    if not isinstance(raw, str) or not raw.strip():
        return [], []

    import re
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    candidates = [text]
    bs, be = text.find("{"), text.rfind("}")
    if bs >= 0 and be > bs:
        candidates.append(text[bs:be + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        completed_raw = parsed.get("completed", [])
        new_raw = parsed.get("new", [])
        completed: list[int] = []
        if isinstance(completed_raw, list):
            for x in completed_raw:
                try:
                    completed.append(int(x))
                except (ValueError, TypeError):
                    continue
        new_tasks: list[str] = []
        if isinstance(new_raw, list):
            new_tasks = [str(x).strip() for x in new_raw
                         if isinstance(x, (str, int, float)) and str(x).strip()]
        return completed, new_tasks
    return [], []


# ─── Action-signature matching + visual confirm ──────────────────────────────
#
# The deterministic core. Replaces "LLM judges from action result string"
# with "did the action signature match what an open TODO needs?"
# This kills the L191 hallucination class: a click result that says
# "Clicked '1-50'" no longer marks "Select '1-50' from Staff Size" done —
# select-TODOs are deferred to a vision-LLM confirmation pass that asks
# "is the value visibly selected in the field?"
#
# Tunables (intentionally module-level for testability):
#   _CLICK_FUZZ_THRESHOLD     — overlap ratio for click target match
#   _VALUE_CONFIRM_THRESHOLD  — *unused for vision path* (kept for OCR
#                                    fallback if we ever change Q1 default)
#   _MAX_CONFIRM_STRIKES      — strikes before vision-fallback fires

import difflib as _difflib

_CLICK_FUZZ_THRESHOLD = 0.7
_VALUE_CONFIRM_THRESHOLD = 0.85
_MAX_CONFIRM_STRIKES = 3

# Action types that are clicks (used by Rule C and Rule S).
_CLICK_ACTION_TYPES = {
    "vision_guided_click",
    "find_and_click_text",
    "find_and_double_click_text",
    "mouse_click",
    "mouse_double_click",
}


def _action_failed(result: str) -> bool:
    """
    True if the action result string indicates failure or abort.
    Used by the matcher to skip marking TODOs from doomed actions.
    Kept lenient (substring checks) — execute_all_actions returns prose,
    not structured codes.
    """
    if not isinstance(result, str):
        return True
    r = result.strip().lower()
    if not r:
        return True
    # Common failure prefixes from action handlers
    if r.startswith(("failed:", "error:", "aborted")):
        return True
    # Common failure tokens anywhere in the message
    return any(token in r for token in (
        "aborted_wrong_focus",
        "not found",
        "no input focused",  # keyboard_type's no-focus warning is informational, not failure — handle below
    )) and "aborted" in r  # belt-and-braces: only treat as failed when actually aborted


def _texts_overlap(a: str, b: str, threshold: float = _CLICK_FUZZ_THRESHOLD) -> bool:
    """
    True when canonicalized `a` and `b` overlap as substrings either way OR
    their SequenceMatcher ratio is ≥ threshold. Used for fuzzy matching of
    action targets to TODO field/value labels (handles OCR noise on buttons,
    minor rewording by the planner).
    """
    a_canon = _canonical_token(a)
    b_canon = _canonical_token(b)
    if not a_canon or not b_canon:
        return False
    if a_canon in b_canon or b_canon in a_canon:
        return True
    return _difflib.SequenceMatcher(None, a_canon, b_canon).ratio() >= threshold


def _find_focus_anchor_in_batch(
    actions: list[dict], type_action_index: int, field: str
) -> dict | None:
    """
    Walk BACKWARD from a `keyboard_type` at index `type_action_index` through
    the same batch, looking for a focus-establishing action. Returns the
    anchor action dict or None.

    Acceptable anchors (in priority order):
      1. Click action (vision_guided_click / find_and_click_text /
         find_and_double_click_text / mouse_click / mouse_double_click) whose
         target text overlaps `field` (substring or fuzz ≥ 0.6).
      2. keyboard_press(tab) — deterministic field hop (the FORM FILLING rule
         in the planner prompt actively recommends this pattern).
      3. keyboard_hotkey containing "tab" (ctrl+tab etc.).

    Any other action between the type and a candidate anchor breaks the
    chain (we conservatively return None). Skips screenshot_and_continue
    and wait actions transparently.

    Same-batch only (Q4 user decision). Across-batch lookback would be more
    permissive but raises false-match risk on intervening focus changes.
    """
    field_threshold = 0.6  # slightly looser than click match — field labels
                           # vary more between TODO and OCR snap text
    for i in range(type_action_index - 1, -1, -1):
        a = actions[i]
        atype = a.get("type") or a.get("action") or ""
        if atype in ("screenshot_and_continue", "wait"):
            continue
        if atype == "keyboard_press":
            key = str(a.get("key", "")).lower()
            if key == "tab":
                return a
            return None  # any other keyboard_press before a type breaks the chain
        if atype == "keyboard_hotkey":
            keys = a.get("keys", []) or []
            if any(str(k).lower() == "tab" for k in keys):
                return a
            return None
        if atype in _CLICK_ACTION_TYPES:
            target = _action_target_text(a)
            if _texts_overlap(target, field, threshold=field_threshold):
                return a
            return None  # click on something unrelated — not a valid anchor
        # Any other action type is unexpected before a type — break.
        return None
    return None


def _match_action_to_todo(
    action: dict,
    result: str,
    action_index: int,
    all_actions: list[dict],
) -> tuple[int | None, bool]:
    """
    Pass 2 single-action matcher. Try each rule (T → S → C) against
    the open TODOs and return the matched TODO id (or None) and whether it
    was marked done.

    Return semantics:
      (None, False)        — no TODO matched this action
      (todo_id, True)      — matched and marked done (Rule T or Rule C)
      (todo_id, False)     — matched but DEFERRED to visual confirm (Rule S)

    Failed/aborted actions never match — return (None, False) immediately.
    Mutates `_task_state` directly (marks done, sets pending_visual_confirm).
    """
    if _action_failed(result):
        return (None, False)

    atype = action.get("type") or action.get("action") or ""

    # ── Rule T: type ──
    if atype == "keyboard_type":
        typed = action.get("text", "")
        if not isinstance(typed, str) or not typed.strip():
            return (None, False)
        for todo in _task_state.todo_list:
            if todo["done"] or todo.get("kind") != "type":
                continue
            value = todo.get("value", "")
            if not value:
                continue
            if not _texts_overlap(typed, value, threshold=_CLICK_FUZZ_THRESHOLD):
                continue
            anchor = _find_focus_anchor_in_batch(
                all_actions, action_index, todo.get("field", "")
            )
            if anchor is None:
                continue
            _task_state.mark_todo_done(todo["id"])
            logger.info(
                f"[AGENT] rule T: TODO #{todo['id']} marked done — "
                f"keyboard_type matched value '{value}' with anchor "
                f"{anchor.get('type', '?')!r}"
            )
            return (todo["id"], True)
        return (None, False)

    # ── Rules S + C: click family ──
    if atype in _CLICK_ACTION_TYPES:
        target = _action_target_text(action)
        if not target:
            return (None, False)

        # Rule S FIRST — a click whose target matches a select-TODO's field
        # or value should NEVER auto-mark done. Defer to visual confirm.
        # This must run before Rule C because a click on a dropdown option
        # ('1-50') would otherwise look like a click-TODO match.
        for todo in _task_state.todo_list:
            if todo["done"] or todo.get("kind") != "select":
                continue
            if todo.get("pending_visual_confirm"):
                # Already pending from an earlier batch — don't re-defer
                # (would reset confirm_strikes erroneously).
                continue
            value = todo.get("value", "")
            field = todo.get("field", "")
            if (_texts_overlap(target, value, threshold=_CLICK_FUZZ_THRESHOLD)
                    or _texts_overlap(target, field, threshold=_CLICK_FUZZ_THRESHOLD)):
                todo["pending_visual_confirm"] = True
                todo["confirm_strikes"] = 0
                # Stamp the deferral so the dialog-engagement gate treats
                # this click into a select-TODO surface as engagement.
                todo["batch_deferred"] = _task_state.batch_idx
                logger.info(
                    f"[AGENT] rule S: TODO #{todo['id']} deferred — "
                    f"click on '{target}' may have selected '{value}' "
                    f"from '{field}'; awaiting visual confirm"
                )
                return (todo["id"], False)

        # Rule C — straight click-TODO match.
        for todo in _task_state.todo_list:
            if todo["done"] or todo.get("kind") != "click":
                continue
            click_target = todo.get("target", "")
            if not click_target:
                continue
            if _texts_overlap(target, click_target, threshold=_CLICK_FUZZ_THRESHOLD):
                _task_state.mark_todo_done(todo["id"])
                logger.info(
                    f"[AGENT] rule C: TODO #{todo['id']} marked done — "
                    f"click on '{target}' matched target '{click_target}'"
                )
                return (todo["id"], True)
        return (None, False)

    return (None, False)


def _match_actions_to_todos(
    actions: list[dict], results: list[str]
) -> tuple[set[int], int]:
    """
    Pass 2 driver: walk the batch and apply _match_action_to_todo to each
    (action, result) pair. Returns (addressed_ids, marked_count) where
    addressed_ids includes both done-marked AND deferred TODOs.
    """
    addressed: set[int] = set()
    marked = 0
    for i, (a, r) in enumerate(zip(actions, results)):
        todo_id, did_mark = _match_action_to_todo(a, r, i, actions)
        if todo_id is not None:
            addressed.add(todo_id)
            if did_mark:
                marked += 1
    return addressed, marked


_VISUAL_CONFIRM_PROMPT_TEMPLATE = (
    # Softened 2026-04-26 after live test where Gemini Flash strict yes/no
    # produced false-NOs on dropdown selections styled differently from
    # plain text inputs (chevron, smaller font, italic). Previous prompt
    # lumped "selected dropdown option" with "placeholder" implicitly.
    # New prompt explicitly accepts dropdown-styled values inside the
    # control's value area while still rejecting empty / placeholder /
    # open-menu-list-not-selected-back states.
    "Look at this screenshot. Find the UI control labeled '{field}' (it may "
    "be a text input, dropdown, combobox, radio group, or similar). Does "
    "that control's value area visibly contain '{value}' (or an obvious "
    "equivalent of it) — INCLUDING when the value is rendered in the "
    "styling typical of a selected dropdown option (smaller font, chevron, "
    "italic, or different colour from the label)?\n\n"
    "Answer YES if the value is shown inside the control's value area, "
    "regardless of styling.\n"
    "Answer NO if the control's value area is empty, shows greyed-out "
    "placeholder/hint text, shows a clearly different value, or '{value}' "
    "is only visible inside an OPEN dropdown menu list (not yet selected "
    "back into the closed control).\n"
    "Reply ONLY 'YES' or 'NO'."
)

_FALLBACK_CONFIRM_PROMPT_TEMPLATE = (
    "Look at this screenshot carefully. The user wanted to select the value "
    "'{value}' from a UI element labeled '{field}' (e.g. dropdown, list, "
    "combobox). After several attempts, we need a final judgment.\n\n"
    "Recent attempts: {recent_summary}\n\n"
    "Is the selection EFFECTIVELY COMPLETE? Accept reasonable equivalents "
    "(e.g. 'IT' shown as 'Information Technology', '1-50' shown as '1 to 50', "
    "'Yes' shown as a checked radio). Be permissive on equivalence but strict "
    "on emptiness — a placeholder or unselected field is NOT complete. "
    "Reply ONLY 'YES' or 'NO'."
)


async def _confirm_pending_select_todos(
    recent_action_results: list[str] | None = None,
) -> tuple[int, int]:
    """
    Pass 1: vision-LLM visual confirmation for any select-TODO with
    pending_visual_confirm=True. Captures one screenshot, asks Gemini Flash
    yes/no per pending TODO whether the expected value is the field's
    current visible value.

    Returns (confirmed_count, fallback_count). confirmed_count is how many
    TODOs were marked done by this pass. fallback_count is how many TODOs
    hit the 3-strike limit and triggered a permissive vision-LLM tie-break.

    Behaviour matrix per pending TODO:
      LLM says YES         → mark_todo_done, clear pending, strikes=0
      LLM says NO           → strikes++.
        if strikes ≥ 3      → fire fallback prompt (more permissive). If
                              fallback says YES, mark done. Else clear
                              pending (give up — TODO stays not-done) and
                              bump _task_state.confirm_fallback_count.
      LLM call fails / unavailable → leave state untouched (fail-open).

    Fail-open everywhere: never raises, never blocks task progress.
    """
    pending = [t for t in _task_state.todo_list
               if t.get("pending_visual_confirm") and not t["done"]]
    if not pending:
        return (0, 0)

    from ...io import screen
    from ... import llm

    try:
        screenshot_b64 = screen.capture_screenshot_base64()
    except Exception as e:
        logger.warning(f"[AGENT] visual-confirm: screenshot failed: {e}")
        return (0, 0)

    if not screenshot_b64:
        logger.warning("[AGENT] visual-confirm: empty screenshot — skipping pass")
        return (0, 0)

    confirmed = 0
    fallbacks = 0

    # Build a short summary of recent action results for the fallback prompt.
    if recent_action_results:
        recent_summary = "; ".join(recent_action_results[-4:])
    else:
        recent_summary = "no action context available"

    for todo in pending:
        field = todo.get("field", "")
        value = todo.get("value", "")
        if not value or not field:
            # Defensive: malformed pending TODO — skip without strike.
            continue

        prompt = _VISUAL_CONFIRM_PROMPT_TEMPLATE.format(field=field, value=value)
        try:
            answer = (await llm.get_vision_response(
                image_base64=screenshot_b64,
                prompt=prompt,
                json_mode=False,
            )).text
        except Exception as e:
            logger.warning(f"[AGENT] visual-confirm LLM call crashed: {e}")
            continue

        if not isinstance(answer, str) or not answer or answer == "__LLM_UNAVAILABLE__":
            # Fail-open: leave pending state intact.
            continue

        if _is_yes_answer(answer):
            _task_state.mark_todo_done(todo["id"])
            todo["pending_visual_confirm"] = False
            todo["confirm_strikes"] = 0
            confirmed += 1
            logger.info(
                f"[AGENT] visual-confirm: TODO #{todo['id']} done — "
                f"value '{value}' confirmed in '{field}'"
            )
            continue

        # NO answer
        todo["confirm_strikes"] = int(todo.get("confirm_strikes", 0)) + 1
        logger.info(
            f"[AGENT] visual-confirm: TODO #{todo['id']} not visible "
            f"(strike {todo['confirm_strikes']}/{_MAX_CONFIRM_STRIKES})"
        )

        if todo["confirm_strikes"] < _MAX_CONFIRM_STRIKES:
            continue

        # Strikes ≥ 3 — fire the fallback (more permissive vision call).
        fb_prompt = _FALLBACK_CONFIRM_PROMPT_TEMPLATE.format(
            field=field, value=value, recent_summary=recent_summary
        )
        try:
            fb_answer = (await llm.get_vision_response(
                image_base64=screenshot_b64,
                prompt=fb_prompt,
                json_mode=False,
            )).text
        except Exception as e:
            logger.warning(f"[AGENT] fallback-confirm LLM call crashed: {e}")
            fb_answer = ""

        if isinstance(fb_answer, str) and _is_yes_answer(fb_answer):
            _task_state.mark_todo_done(todo["id"])
            todo["pending_visual_confirm"] = False
            todo["confirm_strikes"] = 0
            _task_state.confirm_fallback_count += 1
            fallbacks += 1
            logger.warning(
                f"[AGENT] fallback-confirm: TODO #{todo['id']} marked "
                f"done by permissive tie-break (value '{value}' in '{field}')"
            )
        else:
            # Fix A: vision can't confirm → trust the action signature.
            #
            # Rule S only deferred this TODO because a click action's target
            # text overlapped its field/value. After 3 strict NOs and one
            # permissive NO, vision still says no — but the chevron-styled
            # selected value of a closed dropdown is exactly what Gemini
            # Flash false-NOs on (observed 2026-04-26 third live test).
            # Without this branch, the TODO leaks into todo_progress_str()
            # as ✗ and the planner retries the same select indefinitely,
            # burning the loop budget on the wrong field.
            #
            # We mark done with confirm_abandoned=True so:
            #  - the planner sees ✓ and moves on (no more retry loop)
            #  - all_todos_done() returns True if everything else is done
            #    (the verifier sanity check at end-of-loop is the proper
            #    backstop — it'll catch a genuine miss with achieved=False)
            #  - the final TTS reply discloses "(couldn't visually confirm:
            #    {fields})" so the user knows we trusted the action
            #  - mark_todo_done stamps batch_marked_done, keeping the
            #    engagement gate hot (the agent IS still working with this
            #    surface — don't let the checkpoint suddenly dismiss it as overlay)
            _task_state.mark_todo_done(todo["id"])
            todo["pending_visual_confirm"] = False
            todo["confirm_strikes"] = 0
            todo["confirm_abandoned"] = True
            _task_state.confirm_fallback_count += 1
            _task_state.confirm_abandoned_count += 1
            fallbacks += 1
            logger.warning(
                f"[AGENT] fallback-confirm: TODO #{todo['id']} marked "
                f"done as ABANDONED — action signature trusted after "
                f"{_MAX_CONFIRM_STRIKES} strict NOs + permissive NO "
                f"(field={field!r}, value={value!r}). Final verifier remains "
                f"the authority on whether the goal is genuinely complete."
            )

    return (confirmed, fallbacks)


# ─── Agentic Planning Loop ──────────────────────────────────────────────────


def _append_abandoned_suffix(success_text: str) -> str:
    """
    Fix A: append "(couldn't visually confirm: <fields>)" to a success
    message when one or more TODOs were marked done as ABANDONED. Honest
    disclosure for the TTS reply — the user hears that the goal succeeded
    AND learns which specific fields the agent couldn't visually verify.

    No-op when there are no abandoned TODOs (the common case). Capped at
    ~200 chars total so it fits a single TTS utterance without truncation.
    """
    suffix_body = _task_state.abandoned_field_summary(max_fields=2)
    if not suffix_body:
        return success_text
    base = (success_text or "").rstrip()
    # Strip trailing punctuation we'd duplicate
    while base.endswith((".", "!", "?")):
        base = base[:-1]
    return f"{base} (couldn't visually confirm: {suffix_body})."


def _format_action_history(history: list[str]) -> str:
    """Format action history for inclusion in vision prompt."""
    if not history:
        return "None yet"
    # Last 4 actions only — keeps context window tight, reduces LLM confusion
    recent = history[-4:]
    return "\n".join(f"  - {action}" for action in recent)


async def run_computer_task(
    goal: str,
    llm_func: Callable,
    tts_func: Optional[Callable] = None,
    bridge_func: Optional[Callable] = None,
) -> str:
    # ESC monitor lifecycle is now owned by main.py (session-level singleton).
    # run_computer_task must NOT start/stop the shared daemon — doing so kills
    # ESC abort for all subsequent tasks in the session.  reset_abort() is
    # also NOT called here: the outer handler (handle_computer_task) owns the
    # abort reset when operating standalone; when called from a planner step
    # the planner owns the session abort state.

    # Minimize CMD/terminal window so it doesn't appear in screenshots
    # and confuse the vision LLM with past log output
    cmd_windows = []
    try:
        import pygetwindow as gw
        cmd_windows = [w for w in gw.getAllWindows()
                      if 'cmd' in w.title.lower()
                      or 'command prompt' in w.title.lower()
                      or 'windows powershell' in w.title.lower()]
        for w in cmd_windows:
            w.minimize()
            logger.info(f"[AGENT] Minimized terminal: '{w.title}'")
        if cmd_windows:
            time.sleep(0.3)
    except Exception as e:
        logger.warning(f"[AGENT] Could not minimize terminal: {e}")

    try:
        return await _run_computer_task_inner(goal, llm_func, tts_func, bridge_func)
    finally:
        # Restore terminal window after task completes
        try:
            for w in cmd_windows:
                w.restore()
        except Exception:
            pass


async def _run_computer_task_inner(
    goal: str,
    llm_func: Callable,
    tts_func: Optional[Callable] = None,
    bridge_func: Optional[Callable] = None,
) -> str:
    """
    Run the agentic computer control loop.

    1. Try direct system query (psutil bypass) — skip the loop if answered
    2. Capture screen context
    3. Send to LLM with goal → get JSON action plan
    4. Execute actions
    5. Verify goal via separate LLM call on fresh screenshot
    6. If not achieved → re-plan (up to MAX_LOOPS)

    Args:
        goal:        The user's goal (e.g., "open Chrome and go to google.com")
        llm_func:    Async function(prompt, system_prompt, json_mode) → str
        tts_func:    Optional async function(text) for speaking status updates
        bridge_func: Optional async function(cmd, **kwargs) for Unity animations

    Returns:
        A summary string of what was accomplished.
    """
    from ...io import screen

    # Audit fix #1/#6: reset_abort() is now done at the run_computer_task
    # entry point (before start_esc_monitor) to close the race window where
    # a still-held ESC from a prior task could re-set TASK_ABORTED. Removed
    # the duplicate reset here. _run_computer_task_inner is internal and
    # never called from outside run_computer_task, so this is safe.

    logger.info(f"[AGENT] Starting computer task: \"{goal}\"")

    if tts_func:
        await tts_func(f"Working on it: {goal}")
    if bridge_func:
        await bridge_func("play_animation", name="thinking")

    remaining_context = ""  # Extra context from failed verification
    action_history = []
    last_execution_results = []  # Results from last action batch

    # ── Initial TODO generation (one vision call) ──
    # Independent completeness signal. Always-on by design — empty/short
    # goals just produce small lists. If the LLM returns [] (e.g. for open-
    # ended browse goals) we fall back to verifier-only completion (legacy).
    # Fail-open on any error so this never breaks an existing working task.
    try:
        from ...io import screen as _screen_init
        initial_screen_b64 = _screen_init.capture_screenshot_base64()
        todos = await _generate_initial_todos(goal, initial_screen_b64)
        if todos:
            requested = len(todos)
            added = _task_state.set_initial_todos(todos)
            if added < requested:
                logger.warning(
                    f"[AGENT] TODO list capped at {added} of {requested} items "
                    f"(TODO_MAX={_TaskState.TODO_MAX}) — goal may be too complex for one task"
                )
            logger.info(f"[AGENT] initial TODO list ({added} items):")
            for it in _task_state.todo_list:
                logger.info(f"[AGENT]   {it['id']}. {it['task']}")
        else:
            logger.info(
                "[AGENT] no initial TODOs (open-ended goal or LLM unavailable) "
                "— completion will use vision verifier alone"
            )
    except Exception as e:
        logger.warning(f"[AGENT] initial TODO generation crashed (fail-open): {e}")

    for step_i in range(MAX_LOOPS):
        # Rotate through descriptive labels so the pill has character
        # instead of reading "loop 3" / "step".
        from assistant.io.overlay.theme import VISION_LOOP_LABELS, rotating_label as _rot
        _status_broadcaster.set(_StatusPhase.VISION,
                                detail=_rot(VISION_LOOP_LABELS, step_i),
                                step=(step_i + 1, MAX_LOOPS),
                                tier="vision")
        if _check_abort():
            raise _UserAborted("esc_hold")

        # Bump per-task batch counter. Rule matches and Rule S deferrals
        # stamp this onto the TODO they touch; the dialog-engagement gate
        # compares against (current_batch_idx - window) to decide if recent
        # successful interaction makes the visible modal a work surface (and
        # not an overlay to dismiss).
        _task_state.batch_idx = step_i + 1

        # Step 1: Capture screen context
        logger.info(f"[AGENT] Step {step_i + 1}/{MAX_LOOPS} — capturing screen...")
        screenshot_b64 = screen.capture_screenshot_base64()
        active_window = screen.get_active_window()
        open_windows = screen.get_open_windows()

        # Step 2: Build vision prompt
        last_results_str = chr(10).join(last_execution_results) if last_execution_results else "None"
        cached_data_str = _task_state.format_cached_data()
        vision_prompt = (
            f"GOAL: {goal}\n\n"
            f"CURRENTLY ACTIVE WINDOW: \"{active_window}\"\n"
            f"ALL OPEN APPLICATIONS: {open_windows}\n\n"
            f"PREVIOUS ACTIONS TAKEN:\n{_format_action_history(action_history)}\n\n"
            f"LAST EXECUTION RESULTS:\n{last_results_str}\n\n"
        )
        # Surface the TODO checklist + progress so the planner has
        # explicit visibility into what remains. Empty when no TODOs (open-
        # ended goals — falls back to legacy planning behaviour).
        todo_progress = _task_state.todo_progress_str()
        if todo_progress:
            vision_prompt += f"{todo_progress}\n\n"
        if cached_data_str:
            vision_prompt += (
                f"DATA ALREADY RETRIEVED (do NOT re-fetch):\n{cached_data_str}\n\n"
            )
        vision_prompt += (
            f"What actions should I take next to accomplish the goal? "
            f"Analyze the screenshot and return the action plan JSON."
        )

        if step_i > 0:
            vision_prompt += (
                f"\n\nThis is re-plan step {step_i + 1}. "
                "The previous actions have been executed. "
                "Check if the goal is achieved, or continue working toward it."
            )

        if remaining_context:
            vision_prompt += (
                f"\n\nVERIFICATION FEEDBACK: The goal was NOT yet achieved. "
                f"What still needs to be done: {remaining_context}"
            )

        # Step 3: Ask the LLM for a plan
        logger.info("[AGENT] Requesting action plan from LLM...")
        from ... import llm
        from ...llm.contracts import ask_for_plan
        if screenshot_b64:
            raw_response = (await llm.get_vision_response(
                image_base64=screenshot_b64,
                prompt=vision_prompt,
                system_prompt=VISION_PLANNER_SYSTEM_PROMPT,
                json_mode=True,
            )).text
        else:
            # Fallback to OCR text pipeline if screenshot failed
            logger.warning("[AGENT] Screenshot failed — falling back to OCR pipeline")
            screen_desc = screen.describe_screen_for_llm()
            raw_response = await ask_for_plan(
                vision_prompt + f"\n\nSCREEN TEXT:\n{screen_desc}",
                system_prompt=VISION_PLANNER_SYSTEM_PROMPT,
                json_mode=True,
            )

        if raw_response == "__LLM_UNAVAILABLE__":
            return "Sorry, I couldn't plan the task — no LLM is available right now."

        # Step 4: Parse the plan
        plan = _parse_plan(raw_response)
        if plan is None:
            logger.error(f"[AGENT] Failed to parse LLM plan: {raw_response[:200]}")
            return "Sorry, I couldn't understand the action plan from the LLM."

        thinking = plan.get("thinking", "")
        actions = plan.get("actions", [])
        summary = plan.get("summary", "")

        logger.info(f"[AGENT] Plan: {thinking}")
        logger.info(f"[AGENT] Actions: {len(actions)}")

        # Step 5 & 6: Execute actions
        if actions:
            # Structural guard against uncommitted dropdown navigation
            # (Down×N with no Enter). No-op when the batch isn't an arrow-key
            # navigation OR when no dropdown context is detected.
            actions = _inject_dropdown_commit_if_needed(actions)
            results = execute_all_actions(actions)
            logger.info(f"[AGENT] Execution results: {results}")
            action_history.extend(results)
            last_execution_results = results

            if any("ABORTED" in r for r in results):
                return "Task aborted by user."

            # Update TODO state after the batch. Fail-open inside the helper
            # — never blocks task progress. Runs BEFORE checkpoint so the
            # checkpoint and the next planning loop both see fresh progress.
            try:
                await _update_todos_after_batch(actions, results, thinking)
            except Exception as e:
                logger.warning(f"[AGENT] TODO update crashed (continuing): {e}")

            # Detect explicit replan request — but DON'T `continue` yet.
            # The recovery checkpoint must run on every batch (including
            # those that ended in screenshot_and_continue), because that
            # pseudo-action is the planner saying "I want fresh eyes" —
            # exactly when the checkpoint is most likely to have something
            # useful to say.
            request_replan = any("SCREENSHOT_AND_CONTINUE" in r for r in results)

            # Post-batch recovery checkpoint. One vision diagnose; if the
            # screen is in a recoverable state (overlay opened, validation
            # error, no_change), fire the matching strategy once. Never
            # escalates — falls through to the existing _verify_goal either
            # way. The outcome is appended to action_history so the next
            # loop's planner can route around any noticed issue.
            from ... import config as _cfg
            from .. import recovery as _rec
            if getattr(_cfg, "RECOVERY_CHECKPOINT_ENABLED", True):
                last_real_action = next(
                    (a for a in reversed(actions)
                     if (a.get("type") or a.get("action") or "") != "screenshot_and_continue"),
                    None,
                )
                if last_real_action is not None:
                    try:
                        # Per recovery design, `goal` is per-step intent
                        # — NOT the high-level user task. The planner's
                        # `thinking` describes what THIS batch was meant to do.
                        # Diagnose uses it for context (recognize expected
                        # outcomes — clicking a dropdown opens it = success).
                        # NOTE: `thinking` is often a multi-sentence paragraph;
                        # the diagnose model can handle that for reasoning, but
                        # the bbox locator downstream needs SHORT element-shaped
                        # text — that comes from diagnose's `recovery_target`
                        # field, not from this `goal`. See design doc §17K.
                        batch_intent = (thinking or "").strip() or goal
                        synth = _rec._synthesize_step_from_ca_action(last_real_action, batch_intent)
                        last_ca_name = (last_real_action.get("type")
                                        or last_real_action.get("action") or "?")
                        logger.info(
                            f"[AGENT] checkpoint: last_action={last_ca_name} "
                            f"goal_len={len(batch_intent)} "
                            f"goal_preview={batch_intent[:80]!r}"
                        )
                        co = await _rec.checkpoint(
                            goal=batch_intent,
                            last_action=synth,
                            active_window=active_window,
                            # Gate context: pass current TODO state + this
                            # batch's results so the gate can detect
                            # engagement with a modal surface and suppress
                            # destructive overlay dismissal.
                            todo_snapshot=list(_task_state.todo_list),
                            recent_action_results=results,
                            current_batch_idx=_task_state.batch_idx,
                        )
                        if co.recovered:
                            note = (
                                f"checkpoint recovered "
                                f"({co.diagnosed_class} → {co.action_taken}): {co.detail}"
                            )
                            logger.info(f"[AGENT] {note}")
                            action_history.append(note)
                        elif co.diagnosed_class != "unknown":
                            note = (
                                f"checkpoint noticed: "
                                f"{co.diagnosed_class} — {co.detail} (recovery did not resolve)"
                            )
                            logger.warning(f"[AGENT] {note}")
                            action_history.append(note)
                    except Exception as e:
                        logger.warning(f"[AGENT] checkpoint crashed (continuing): {e}")

            # Now act on the deferred replan request. The checkpoint has had
            # its turn (recovered, noticed, or returned unknown — all logged
            # above), and any side-effect it had on screen state is visible
            # to the next loop's planner.
            if request_replan:
                logger.info("[AGENT] Re-planning after screenshot_and_continue...")
                remaining_context = ""
                continue

        # Step 7: Verify goal — TODO-first + verifier as sanity check
        logger.info("[AGENT] Verifying goal achievement...")

        # Fast path: window title quick verify (ground-truth signal for media
        # playback — leave as-is, not subject to TODO logic).
        quick_ver = _quick_verify_from_window_title(goal, active_window)
        if quick_ver is True:
            result_text = _append_abandoned_suffix(summary or f"Completed: {goal}")
            logger.info(f"[AGENT] [OK] Goal verified achieved (quick window title match): {result_text}")
            if tts_func:
                await tts_func(result_text)
            if bridge_func:
                await bridge_func("play_animation", name="wave")
            return result_text

        # ── Primary path: TODO list is the completeness signal ──
        # When TODO tracking is active for this task, we judge completion by
        # the TODO state. The vision verifier is demoted to a sanity check
        # that runs only when all TODOs report done.
        if _task_state.todo_list:
            done_count = sum(1 for t in _task_state.todo_list if t["done"])
            total_count = len(_task_state.todo_list)
            if not _task_state.all_todos_done():
                # Still work to do — skip the verifier call entirely. The
                # next planner iteration will see updated TODO progress
                # and continue from there. This also fixes the
                # verifier-hallucinates-completion bug: the verifier never
                # gets a chance to falsely say "achieved" while real TODOs
                # remain.
                logger.info(
                    f"[AGENT] {done_count}/{total_count} TODOs done — "
                    f"continuing without verifier call"
                )
                remaining_context = ""
                continue

            # All TODOs done — sanity-check via vision verifier.
            logger.info(
                f"[AGENT] all {total_count} TODOs marked done — "
                f"running verifier sanity check..."
            )
            verification = await _verify_goal(goal, llm_func)
            if verification.get("achieved", False):
                result_text = _append_abandoned_suffix(
                    verification.get("result", "") or summary or f"Completed: {goal}"
                )
                logger.info(f"[AGENT] [OK] TODO + verifier agree: {result_text}")
                if tts_func:
                    await tts_func(result_text)
                if bridge_func:
                    await bridge_func("play_animation", name="wave")
                return result_text

            # Disagreement: TODOs say done but verifier disagrees. TRUST THE
            # VERIFIER (cautious) — the TODO updater may have over-marked,
            # or a finalisation step is missing. Feed the verifier's
            # `remaining` back into the next plan; the per-batch updater
            # will likely add a new TODO once the planner addresses it.
            new_remaining = verification.get("remaining", "")
            logger.warning(
                f"[AGENT] TODO/verifier disagreement — TODOs say done "
                f"but verifier says not. Trusting verifier. Remaining: {new_remaining}"
            )
            remaining_context = new_remaining
            continue

        # ── Legacy path: TODO list empty (open-ended goal or LLM failure) ──
        # Verifier-only completion with the every-other-step gate to save
        # rate limits.
        should_verify = (step_i % 2 == 1) or (step_i == MAX_LOOPS - 1)
        if not should_verify:
            logger.info(f"[AGENT] Skipping full LLM verification on step {step_i + 1} to save rate limits")
            remaining_context = ""
            continue

        verification = await _verify_goal(goal, llm_func)

        if verification.get("achieved", False):
            result_text = _append_abandoned_suffix(
                verification.get("result", "") or summary or f"Completed: {goal}"
            )
            logger.info(f"[AGENT] [OK] Goal verified achieved: {result_text}")

            if tts_func:
                await tts_func(result_text)
            if bridge_func:
                await bridge_func("play_animation", name="wave")

            return result_text

        # Not achieved — check if this was a first failure (retry same plan once before replanning)
        new_remaining = verification.get("remaining", "Unknown — re-check the screen")
        logger.info(f"[AGENT] [FAIL] Goal not yet achieved. Remaining: {new_remaining}")

        if new_remaining == remaining_context and remaining_context:
            # Same failure reason as last loop — no point retrying same plan, force replan
            logger.info("[AGENT] Same failure reason repeated — forcing replan with new screenshot")
            remaining_context = new_remaining
        elif remaining_context and not last_execution_results:
            # No actions were taken last loop — skip retry
            remaining_context = new_remaining
        else:
            # First time seeing this failure — retry same loop once more with fresh screenshot
            logger.info("[AGENT] First failure — will retry with fresh screenshot before replanning")
            remaining_context = new_remaining

    # Exhausted all loops — do one final verification before giving up
    logger.info("[AGENT] Max loops reached, running final verification...")
    final_check = await _verify_goal(goal, llm_func)

    if final_check.get("achieved", False):
        result_text = _append_abandoned_suffix(
            final_check.get("result", "") or f"Completed: {goal}"
        )
        logger.info(f"[AGENT] [OK] Final verification passed: {result_text}")
        if tts_func:
            await tts_func(result_text)
        if bridge_func:
            await bridge_func("play_animation", name="wave")
        return result_text

    remaining = final_check.get("remaining", "")
    msg = f"I wasn't able to complete that fully. Here's where I got to: {remaining}" if remaining else f"I wasn't able to fully complete: {goal}"
    logger.warning(f"[AGENT] {msg}")
    try:
        from ... import telemetry as _telemetry
        _telemetry.mark_action_failure(
            "VisionAgentMaxLoops",
            (remaining or goal)[:160],
        )
    except Exception:
        pass
    if tts_func:
        await tts_func(msg)
    return msg


