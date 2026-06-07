"""
desktop_automation.py — Unified desktop automation router for TENKA.

Routes automation tasks to the best backend:
  1. Playwright (browser_automation.py) — for web/browser tasks
  2. Terminator (app_automation.py) — for native Windows app tasks
  3. Vision loop (computer_agent.py) — fallback for complex/undetectable tasks

Routing is fully dynamic — zero hardcoded app lists:
  - URL regex detection → browser
  - Running process/window detection → native app (Terminator)
  - User preferences → override any automatic routing
  - Accessibility tree probe → verify Terminator can handle it
  - Vision loop → last resort fallback
"""

import asyncio
import logging
import re
import json
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse

from .. import config
from ..core.known_apps import get_apps_by_category

logger = logging.getLogger("desktop_automation")


async def _maybe_await(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)

_URL_PATTERN = re.compile(
    r'https?://[^\s]+|'
    r'www\.[^\s]+|'
    r'(?<!@)\b\w+\.(com|org|net|io|dev|edu|gov|co|app|ai)\b',
    re.IGNORECASE
)


def _extract_domain(url_or_text: str) -> str | None:
    """Extract the registrable domain from a URL or text containing a URL.

    Returns the domain without www prefix (e.g. "github.com"), or None
    if no URL can be found.
    """
    m = _URL_PATTERN.search(url_or_text)
    if not m:
        return None
    raw = m.group(0)
    if not raw.startswith("http"):
        raw = "https://" + raw
    host = urlparse(raw).hostname or ""
    return host.removeprefix("www.").lower() or None


# Strips the trailing browser-name suffix from an OS window title to recover
# the underlying page <title>. Handles -, en-dash, em-dash separators and
# Chrome/Chromium/Brave/Edge/Opera/Vivaldi.
_BROWSER_WINDOW_SUFFIX_RE = re.compile(
    r'\s*[-–—]\s*(Google\s+Chrome|Chromium|Brave(?:\s+Browser)?|'
    r'Microsoft\s*Edge|Edge|Opera|Vivaldi)\s*$',
    re.IGNORECASE,
)

_BROWSER_INTENT_PATTERNS = re.compile(
    r'\b(visit|browse|web\s*page|website|login\s+to|sign\s+in\s+to|'
    r'fill\s+(out\s+)?(the\s+)?form|submit\s+form|'
    r'download\s+from|scrape|fetch\s+page|'
    r'search\s+(for|on|the\s+web|google|internet))\b',
    re.IGNORECASE
)

# ─── DOM-mode routing patterns ──────────────────────────────────────────────
#
# Goals shaped like form-fill / submit / login → prefer DOM-mode when CDP
# is up. Keep the keyword list aligned with browser_dom_orchestrator's
# _SUBMIT_NAME_TOKENS so the routing layer and the planner agree on what
# "form-shape" means.
_FORM_INTENT_RE = re.compile(
    r'\b(fill|set|choose|pick|submit|enter|select|type|register|sign\s*up|signup|'
    r'log\s*in|login|sign\s*in|signin|checkout|book|schedule|'
    r'subscribe|register|create\s+account|complete\s+(?:the\s+)?form)\b',
    re.IGNORECASE,
)

# Goals shaped like extraction / read / summarize → vision-loop is fine
# (no need for DOM perception). Saves the cost of perceiving a tree we
# won't act against.
_EXTRACTION_RE = re.compile(
    r'\b(read|summari[sz]e|tell\s+me|what\s+(does|is)|extract|'
    r'show\s+me|describe|find\s+text|copy\s+text)\b',
    re.IGNORECASE,
)

# Canvas / WebGL / Flutter Web heavy apps where DOM perception is opaque.
# These ALWAYS route to vision regardless of CDP availability.
_CANVAS_APP_RE = re.compile(
    r'\b(figma|miro|whiteboard|canvas|excalidraw|flutter|tldraw|'
    r'google\s+slides|google\s+docs|google\s+drawings|sketch|'
    r'draw|paint)\b',
    re.IGNORECASE,
)

_BROWSER_PLAN_PROMPT = """\
Convert this browser task into a sequence of steps.
Return ONLY a JSON array of steps, no other text.
Each step: {{"action": "...", "params": {{...}}}}

Actions:
- navigate: {{"url": "https://..."}} 
- click: {{"selector": "CSS selector or text"}}
- fill: {{"selector": "CSS selector", "value": "text to type"}}
- extract_text: {{}} (get all visible text from current page)
- extract_selector: {{"selector": "CSS selector"}}
- screenshot: {{}}
- wait: {{"selector": "CSS selector"}} or {{"time_ms": 2000}}
- press: {{"key": "Enter"}}
- select: {{"selector": "CSS selector", "value": "option text"}}

Example — "check the latest news on bbc":
[
  {{"action": "navigate", "params": {{"url": "https://www.bbc.com/news"}}}},
  {{"action": "wait", "params": {{"time_ms": 2000}}}},
  {{"action": "extract_text", "params": {{}}}}
]

Example — "log in to my account on example site":
[
  {{"action": "navigate", "params": {{"url": "https://example.com/login"}}}},
  {{"action": "fill", "params": {{"selector": "input[type=email]", "value": "user@example.com"}}}},
  {{"action": "fill", "params": {{"selector": "input[type=password]", "value": "mypassword"}}}},
  {{"action": "click", "params": {{"selector": "button[type=submit]"}}}},
  {{"action": "wait", "params": {{"time_ms": 3000}}}},
  {{"action": "extract_text", "params": {{}}}}
]

RULES:
- First step should almost always be navigate
- If a [Target URL: ...] is provided, use that EXACT URL for the navigate step — do not guess or search within the site
- If [Interactive elements] are listed in the task, you MUST pick selectors from that list for fill/click actions
- Otherwise use generic CSS selectors (input[type=email], button[type=submit])
- End with extract_text if the user needs information back
- ONLY interact with elements directly relevant to the goal — do NOT click or fill elements unrelated to the task. If the page has 10 elements but only 2 are needed, use only those 2
- Each click/fill must have a clear reason tied to the goal. "Click everything and see what happens" is WRONG
- For booking/purchasing: click the item link or "Book" button first, then fill forms if needed. Do NOT click city selectors, location buttons, or navigation menus unless the goal requires changing them

{region_hint}
Task: {goal}
"""

_APP_PLAN_PROMPT = """\
Convert this desktop application task into a sequence of steps.
Return ONLY a JSON array of steps, no other text.
Each step: {{"action": "...", "params": {{...}}}}

Actions:
- open: {{"name": "app name"}}
- focus: {{"name": "window title substring"}}
- click: {{"selector": "name:ElementName", "window": "optional window title"}}
- type: {{"text": "text to type", "selector": "optional element", "window": "optional"}}
- get_text: {{"selector": "name:ElementName", "window": "optional"}}
- wait: {{"seconds": 2}}
- press_key: {{"key": "enter"}}
- close: {{"name": "window title"}}

Selector formats (Windows Accessibility Tree):
- name:ElementName — find by accessible name (visible label or control name). This is the PRIMARY selector.
- role:ControlType — find by control type (Button, Edit, Text, etc.)
- window:Title — scope to a specific window
NOTE: ONLY use name: and role: selectors. Do NOT use automationid: — it is not supported.

RULES:
- Use name: selectors based on the visible text labels you'd see in the app
- Always include "window" param when interacting with a specific app
- Add wait after open to let the app load
- If available elements are provided below, use ONLY those exact selectors
- Prefer name: selectors with the exact visible text shown in the available elements list
- ALWAYS end with a get_text step if the task produces a visible result (calculation, status, content). The user needs to hear the result.
- IMPORTANT: For entering values, numbers, expressions, or any data — use "type" action, NOT individual "click" steps. The system sends keyboard input directly. For example, to compute 25+4 on a calculator, use type with text "25+4=" — do NOT click digit buttons one by one. Use "click" only for UI navigation (menu items, tabs, toggles), not for data entry.
- For typing into apps without a specific selector, omit the selector — the system types into the focused field
- For text editors (Notepad, etc.), do NOT click on existing text content — just use type action without a selector to type at the current cursor position. NEVER overwrite the user's existing work. If told to open a new document, use press_key "ctrl+n" first.
- NEVER use existing content text as a selector (e.g., don't click on "remove extra noise..." to position cursor). Content changes and is not a reliable selector.
- For browsers, if told to open a new tab, use press_key "ctrl+t" first. NEVER overwrite the user's current tab.

Example — "open notepad and type a reminder":
[
  {{"action": "open", "params": {{"name": "notepad"}}}},
  {{"action": "wait", "params": {{"seconds": 2}}}},
  {{"action": "type", "params": {{"text": "Remember to buy groceries", "window": "Notepad"}}}}
]

Example — "focus settings and click on display":
[
  {{"action": "focus", "params": {{"name": "Settings"}}}},
  {{"action": "wait", "params": {{"seconds": 1}}}},
  {{"action": "click", "params": {{"selector": "name:Display", "window": "Settings"}}}}
]

{available_elements}

Task: {goal}
"""


def _check_routing_preference(goal: str) -> Optional[str]:
    """
    Check preferences for automation routing overrides.
    Returns "browser", "native", "vision", or None if no preference.
    """
    try:
        from .. import preferences
    except ImportError:
        return None
    
    _SKIP_WORDS = frozenset({
        "open", "close", "click", "type", "search", "find", "go", "navigate",
        "the", "a", "an", "in", "on", "to", "for", "and", "or", "my", "me",
        "please", "can", "you", config.ASSISTANT_NAME_LOWER, "hey", "i", "want", "need", "use",
        "with", "from", "this", "that", "it", "is", "at", "of", "do",
    })
    words = [w.lower().strip(".,!?") for w in goal.split() if len(w) > 2]
    candidates = [w for w in words if w not in _SKIP_WORDS]
    
    for word in candidates:
        try:
            pref = preferences.get_preference(f"automation_{word}")
            if pref and pref.get("confidence", 0) >= 0.4:
                return pref["value"]
        except Exception:
            pass
    return None

# Window title patterns that indicate an empty/new document or tab
_EMPTY_TAB_PATTERNS = re.compile(
    r'(?:^untitled|^new\s|^new tab|^unnamed|^\*?untitled)',
    re.IGNORECASE,
)
# App names that are text editors (type into them)
_text_editor_escaped = sorted(
    (re.escape(name) for name in get_apps_by_category("text_editor")),
    key=len, reverse=True,
)
_TEXT_EDITOR_NAMES = re.compile(
    r'\b(?:' + '|'.join(_text_editor_escaped) + r')\b',
    re.IGNORECASE,
)
# App names that are browsers (navigate in them)
_browser_escaped = sorted(
    (re.escape(b) for b in config.BROWSER_NAMES),
    key=len, reverse=True,
)
_BROWSER_ALT = '|'.join(_browser_escaped)
_BROWSER_NAMES = re.compile(r'\b(?:' + _BROWSER_ALT + r')\b', re.IGNORECASE)


def _extract_doc_part(window_title: str) -> tuple[str, str]:
    """Strip unsaved-indicator and split 'Doc - App' into (doc_part, clean_title)."""
    clean = window_title.lstrip('*').strip()
    parts = clean.rsplit(' - ', 1)
    doc = parts[0].strip() if len(parts) >= 2 else clean
    return doc, clean


def _detect_new_tab_key(window_title: str, goal: str) -> str | None:
    """
    Return the keyboard shortcut to open a new tab/document if existing content detected.
    Returns 'ctrl+n' for text editors, 'ctrl+t' for browsers, or None.
    """
    doc_part, clean_title = _extract_doc_part(window_title)

    is_empty = bool(_EMPTY_TAB_PATTERNS.search(doc_part)) or not doc_part
    if is_empty:
        return None

    if _TEXT_EDITOR_NAMES.search(window_title):
        return "ctrl+n"

    if _BROWSER_NAMES.search(window_title):
        goal_lower = goal.lower()
        if any(w in goal_lower for w in ("search", "go to", "open", "navigate", "browse", "look up")):
            return "ctrl+t"

    return None


def _build_new_tab_hint(window_title: str, goal: str) -> str:
    """
    Detect if the running app's current tab/document has existing content.
    Returns an instruction string for the LLM planner to open a new tab first,
    or empty string if the current tab is already empty/new.
    """
    doc_part, clean_title = _extract_doc_part(window_title)

    is_empty = bool(_EMPTY_TAB_PATTERNS.search(doc_part)) or not doc_part

    if is_empty:
        return ""  # Already empty, no new tab needed

    # Text editors: existing content → Ctrl+N for new document
    if _TEXT_EDITOR_NAMES.search(window_title):
        return (
            f"The current document has existing content ('{doc_part}'). "
            f"Do NOT type into it — press Ctrl+N FIRST to open a new blank document, "
            f"then wait 1 second, then proceed with your task.\n"
        )

    # Browsers: existing page → Ctrl+T for new tab (for navigation/search goals)
    if _BROWSER_NAMES.search(window_title):
        goal_lower = goal.lower()
        is_nav = any(w in goal_lower for w in ("search", "go to", "open", "navigate", "browse", "look up"))
        if is_nav:
            return (
                f"The current tab has an existing page ('{doc_part}'). "
                f"Press Ctrl+T FIRST to open a new tab, then wait 1 second, "
                f"then proceed with navigation/search.\n"
            )

    return ""


# ─── Active-app signal helpers ──────────────────────────────────────────────
#
# Three small, side-effect-free probes onto the OS so callers can ask "what is
# the user actually looking at right now?" without each one re-implementing
# psutil / pygetwindow / CDP plumbing. `detect_active_app()` is the public
# composite — the manifest layer's manifest_registry reads it to choose which app manifest
# to apply, and PV-5 routing can be re-expressed in terms of it later.
#
# Keep these synchronous. The browser-URL probe deliberately returns "" when
# the only available source would require an async Playwright/CDP attachment
# — async callers that need a live URL go through `_pick_active_page`.


def _get_running_processes() -> list[str]:
    """Return process executable names currently running on the system.

    Uses psutil (same library `automation/native.py` uses for its tray-aware
    process scan). Names are returned in their native case (e.g. "Spotify.exe")
    with empty / unreadable entries filtered out. Returns `[]` on any failure
    so callers can treat "no signal" and "scan failed" identically.
    """
    try:
        import psutil
    except ImportError as e:
        logger.debug(f"[ACTIVE_APP] psutil unavailable: {e}")
        return []
    try:
        names: list[str] = []
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info.get('name') or ''
            except Exception:
                continue
            if name:
                names.append(name)
        return names
    except Exception as e:
        logger.debug(f"[ACTIVE_APP] process scan failed: {e}")
        return []


def _get_foreground_window_title() -> str:
    """Return the title of the OS-level focused window, or "" on failure.

    Thin wrapper over `io.screen.get_active_window()` so the active-app
    detection surface stays self-contained at the router layer.
    """
    try:
        from ..io import screen
        return screen.get_active_window() or ""
    except Exception as e:
        logger.debug(f"[ACTIVE_APP] foreground window probe failed: {e}")
        return ""


def _get_active_browser_url() -> str:
    """Return the URL of the user's focused browser tab, or "" if unknown.

    Reading the live tab URL requires an async Playwright/CDP attachment
    (see `_pick_active_page`), which can't run from a synchronous probe.
    The synchronous path returns "" so manifest dispatch can fall back to
    the window-title / process-name signals. Async callers that need the
    URL must go through the CDP attachment directly.
    """
    # TODO(manifest): wire async CDP read once a sync-from-async bridge exists (Session 4+).
    return ""


def detect_active_app() -> dict[str, Any]:
    """Return a snapshot of what the user is currently focused on.

    Returns ``{"process_names": [...], "window_title": "...", "active_url": "..."}``.

    Used by the manifest layer's ``manifest_registry.get_for_active_app()`` and reusable by
    any other caller that needs the same active-window signal that backs PV-5
    routing — avoids duplicating the psutil / pygetwindow / CDP plumbing.
    """
    return {
        "process_names": _get_running_processes(),
        "window_title": _get_foreground_window_title(),
        "active_url": _get_active_browser_url(),
    }


_DETECT_STOP_WORDS = frozenset({
    "open", "close", "launch", "start", "run", "focus", "switch", "go",
    "the", "a", "an", "in", "on", "to", "for", "and", "or", "my", "me",
    "please", "can", "you", config.ASSISTANT_NAME_LOWER, "hey", "with", "from", "this", "that",
    "it", "is", "at", "of", "do", "use", "check", "show", "find", "get",
    "set", "turn", "make", "put", "click", "type", "search", "play",
    "pause", "stop", "next", "back", "forward", "new", "tab", "window",
})


def _detect_running_app(goal: str) -> Optional[str]:
    """
    Check if the goal references a currently running application.
    Returns the window title if found, None otherwise.

    Extracts meaningful words from the goal and checks if they appear in
    window titles (goal→title direction prevents over-matching).
    """
    try:
        from ..io import screen
        open_windows = screen.get_open_windows()
    except Exception:
        return None

    goal_words = [w.lower().strip(".,!?") for w in goal.split()]
    candidates = [w for w in goal_words if w not in _DETECT_STOP_WORDS and len(w) > 2]

    if not candidates:
        return None

    for window_title in open_windows:
        if not window_title.strip():
            continue
        # Match against the app-name portion (after last " - "), not the
        # full title.  Page/document text in the title (e.g. "Hello World
        # Page - Mozilla Firefox") must not cause a false app match.
        parts = window_title.rsplit(' - ', 1)
        app_part = parts[-1].strip().lower()
        for word in candidates:
            if word in app_part:
                return window_title

    return None

# ─── Choose browser mode (DOM vs bundled-Playwright vs vision) ──────────────


def _choose_browser_mode(
    goal: str,
    cdp_state: Any,
    *,
    user_preference: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Pick how to handle a browser-content task. Returns
    `(mode, reason_meta)` where mode is one of:
      - "dom"                — DOM orchestrator (CDP attach + DOM planner)
      - "vision"             — legacy vision-loop fallback (computer_agent.py)
      - "playwright_bundled" — bundled Chromium (TENKA launches its own
                               browser; used when goal needs a browser we
                               control end-to-end, not the user's session)

    `cdp_state` is the cached `CdpProbeResult` from `browser_cdp.cdp_state_snapshot()`.
    `user_preference` overrides the heuristics — if set, returns that mode
    immediately. Loaded from preferences at the call site.

    Decision priority:
      1. DOM-mode kill-switch off → bundled
      2. Canvas/WebGL keyword → vision (DOM is opaque)
      3. CDP unavailable → bundled (DOM-mode requires user's Chrome)
      4. User preference → that mode
      5. Form-intent keyword → DOM
      6. Extraction-intent keyword → bundled (DOM perception waste)
      7. Default → DOM (CDP is up; lean on the cheap path)

    The function is PURE — no I/O, no LLM calls, no side effects. It's safe
    to call repeatedly per task without performance impact.
    """
    from .. import config

    # 1. Master kill-switch
    if not getattr(config, "BROWSER_DOM_MODE_ENABLED", True):
        return "playwright_bundled", {"reason": "dom_mode_flag_off"}

    # 2. Canvas / WebGL apps — DOM is opaque, vision is the only path.
    #    This wins even when CDP is up because the DOM tree won't have
    #    actionable content (the canvas element is a single <canvas>).
    if _CANVAS_APP_RE.search(goal):
        return "vision", {"reason": "canvas_intent"}

    # 3. CDP availability is necessary for DOM-mode (it's how we get to
    #    the user's existing tab). When unreachable, fall back to bundled.
    cdp_available = bool(getattr(cdp_state, "available", False)) if cdp_state else False
    if not cdp_available:
        return "playwright_bundled", {"reason": "cdp_unavailable"}

    # 4. User preference — explicit user override of the heuristic
    if user_preference in ("dom", "vision", "playwright_bundled"):
        return user_preference, {"reason": "user_preference"}

    # 5. Form-shape goal — the headline DOM-mode case
    if _FORM_INTENT_RE.search(goal):
        return "dom", {"reason": "form_intent"}

    # 6. Extraction goal — vision is fine here, no perception needed
    if _EXTRACTION_RE.search(goal):
        return "playwright_bundled", {"reason": "extraction_intent"}

    # 7. Default when CDP is up — lean toward DOM-mode (cheaper path)
    return "dom", {"reason": "cdp_default"}


def _route_browser_content(goal: str, running_window: str) -> Tuple[str, Dict[str, Any]]:
    """
    Bridge `_choose_browser_mode` into `detect_backend`'s return vocabulary.

    `detect_backend` returns one of: "browser" | "native" | "vision" | "dom"
                                   | "unknown" | "playwright_bundled".
    `_choose_browser_mode` returns: "dom" | "vision" | "playwright_bundled".

    For the browser-content branch (user has Firefox/Chrome open with the
    target page already loaded), we delegate the choice to the routing
    function and tag the meta with the running window so downstream code
    can log which app the decision was made against.
    """
    cdp_state = None
    user_pref = None
    try:
        from .browser import cdp as browser_cdp
        cdp_state = browser_cdp.cdp_state_snapshot()
    except Exception:
        cdp_state = None
    # User can persist a preference for "always vision" or "always dom"
    # via the preference store. Pure read-side — no mutations from the
    # routing path. Wrap in try/except so a corrupted preference DB
    # doesn't break routing.
    try:
        from .. import preferences
        pref_row = preferences.get_preference("automation_browser_mode")
        if isinstance(pref_row, dict):
            user_pref = pref_row.get("value")
    except Exception:
        user_pref = None

    mode, meta = _choose_browser_mode(goal, cdp_state, user_preference=user_pref)
    # In the browser-content scenario (user already has their own browser
    # open at the target page), "playwright_bundled" doesn't make sense —
    # we can't operate on the user's existing session without CDP. Fall
    # to vision-loop, the established legacy behaviour. The Chrome setup
    # script is what makes CDP available in the first place; until the
    # user opts in, vision is the right fallback.
    if mode == "playwright_bundled":
        meta = {**meta, "translated_from": "playwright_bundled"}
        mode = "vision"
    meta = {**meta, "app": running_window}
    return mode, meta


def detect_backend(goal: str) -> Tuple[str, Dict[str, Any]]:
    """
    Return the detected backend and metadata dict explaining the routing.
    Backend: "browser", "native", "vision", "unknown"
    """
    # Priority 1: User Preference
    pref_backend = _check_routing_preference(goal)
    if pref_backend in ("browser", "native", "vision"):
        return pref_backend, {"reason": "user_preference", "preference": pref_backend}
    
    # Priority 2: URL pattern
    if _URL_PATTERN.search(goal):
        url_match = _URL_PATTERN.search(goal).group(0)
        return "browser", {"reason": "url_detected", "url": url_match}
    
    run_app_match = re.search(r'\b(open|launch|start|run)\s+(\w+)', goal, re.IGNORECASE)
    
    if _BROWSER_INTENT_PATTERNS.search(goal):
        # Search tasks always go to Playwright (zero vision, own browser instance)
        is_search = re.search(r'\bsearch\s+(for|on|the)', goal, re.IGNORECASE)
        running = _detect_running_app(goal)
        if running and _BROWSER_NAMES.search(running):
            # Browser-content goal targeting an already-open browser session.
            # Try to upgrade to DOM-mode when CDP is attached to the user's
            # Chrome. Falls back to vision-loop when CDP isn't available,
            # the goal is canvas-shaped, or the kill-switch is off.
            return _route_browser_content(goal, running)
        if is_search or (not running and not run_app_match):
            return "browser", {"reason": "browser_intent"}

    # Priority 3: Running OS process
    running_window = _detect_running_app(goal)
    if running_window:
        # Same browser-content guard for goals that don't trip
        # _BROWSER_INTENT_PATTERNS but still target an open browser.
        # `open|launch|start|run <browser>` normally wants native focus —
        # but if the goal ALSO has form-intent ("open chrome and fill the
        # form"), the form-fill is the actual task and we should route to
        # browser-content despite the "open browser" prefix.
        if _BROWSER_NAMES.search(running_window) and (
            not run_app_match or _FORM_INTENT_RE.search(goal)
        ):
            return _route_browser_content(goal, running_window)
        return "native", {"reason": "running_app_detected", "app": running_window}
    
    # Priority 4: App launch regex
    if run_app_match:
        app_name = run_app_match.group(2)
        return "native", {"reason": "launch_keyword", "app": app_name}

    # Priority 5: "... on/in/with/using [app_name]" pattern
    # e.g., "multiply 3 and 4 on calculator", "play music on spotify"
    # Guard: when the preposition is "with"/"using" AND a form-intent verb
    # is present, Y is likely data ("fill form with john"), not an app.
    # "on"/"in" almost always indicate the target app even with form verbs
    # ("type hello world in notepad").
    app_context_match = re.search(
        r'\b(?P<prep>on|in|with|using)\s+(?:the\s+)?(\w+)\s*$', goal, re.IGNORECASE
    )
    if app_context_match:
        prep = app_context_match.group("prep").lower()
        skip = prep in ("with", "using") and _FORM_INTENT_RE.search(goal)
        if not skip:
            app_name = app_context_match.group(2).lower()
            if len(app_name) > 2 and app_name not in ("it", "that", "this", "now", "here", "them", "all", "mode"):
                return "native", {"reason": "app_context_pattern", "app": app_name}

    # Priority 6 (fallback): browser-content via form-shape goal.
    # Catches phrasings like "fill this form" that the strict
    # _BROWSER_INTENT_PATTERNS regex misses because of its rigid
    # (the\s+)? clause. When ANY open window is a browser AND the goal
    # is form-shape (per _FORM_INTENT_RE), delegate to browser-content
    # routing — which picks DOM-mode (CDP attach) or falls to vision.
    if _FORM_INTENT_RE.search(goal):
        try:
            from ..io import screen as _screen
            for window_title in (_screen.get_open_windows() or []):
                if window_title and _BROWSER_NAMES.search(window_title):
                    return _route_browser_content(goal, window_title)
        except Exception:
            pass

    return "unknown", {"reason": "no_match"}

async def can_handle(goal: str) -> Tuple[bool, str]:
    backend, meta = detect_backend(goal)
    if backend in ("browser", "native", "dom"):
        if backend == "browser":
            try:
                from .browser import automation as browser_automation
                if not browser_automation.PLAYWRIGHT_AVAILABLE:
                    return (False, "vision")
            except ImportError:
                return (False, "vision")
        elif backend == "native":
            try:
                from . import native as app_automation
                if not app_automation.is_available():
                    return (False, "vision")
            except ImportError:
                return (False, "vision")
        elif backend == "dom":
            try:
                from .browser import automation as browser_automation
                if not browser_automation.PLAYWRIGHT_AVAILABLE:
                    return (False, "vision")
            except ImportError:
                return (False, "vision")
        return (True, backend)
    return (False, backend)

from ..core.json_utils import extract_json_array as _extract_json_array


# Map of browser name → list of (App Paths registry key, fallback exe paths).
# Resolves explicit browser requests on Windows where Python's webbrowser
# module doesn't auto-register named browsers.
_BROWSER_EXE_HINTS = {
    "chrome": (
        "chrome.exe",
        [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        r"Google\Chrome\Application\chrome.exe",
    ),
    "firefox": (
        "firefox.exe",
        [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ],
        None,
    ),
    "edge": (
        "msedge.exe",
        [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        r"Microsoft\Edge\Application\msedge.exe",
    ),
    "brave": (
        "brave.exe",
        [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        ],
        r"BraveSoftware\Brave-Browser\Application\brave.exe",
    ),
    "opera": (
        "opera.exe",
        [
            r"C:\Program Files\Opera\opera.exe",
            r"C:\Program Files (x86)\Opera\opera.exe",
        ],
        None,
    ),
    "safari": ("safari.exe", [], None),
}


def _find_browser_executable(name: str) -> Optional[str]:
    """Locate a named browser's .exe on Windows.

    Lookup order: shutil.which (PATH) → registry App Paths
    (HKLM/HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\X.exe)
    → well-known Program Files paths + LOCALAPPDATA fallback.
    Returns absolute path or None. No-op on non-Windows."""
    import sys, os, shutil
    if not sys.platform.startswith("win"):
        return None
    hint = _BROWSER_EXE_HINTS.get(name)
    if not hint:
        return None
    exe_name, fallback_paths, localappdata_subpath = hint

    # 1. PATH
    found = shutil.which(exe_name)
    if found:
        return found

    # 2. Registry App Paths (canonical Windows app-launch lookup)
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}"
                with winreg.OpenKey(hive, key_path) as k:
                    val, _ = winreg.QueryValueEx(k, None)  # default value
                    if val and os.path.isfile(val):
                        return val
            except OSError:
                continue
    except Exception:
        pass

    # 3. Well-known install paths
    for p in fallback_paths:
        if os.path.isfile(p):
            return p

    # 4. LOCALAPPDATA (per-user install)
    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app and localappdata_subpath:
        candidate = os.path.join(local_app, localappdata_subpath)
        if os.path.isfile(candidate):
            return candidate

    return None


def _format_elements(elements: list[dict]) -> str:
    """Format interactive element descriptors into prompt-ready lines."""
    return "\n".join(
        f"  - {e['s']}"
        + (f' ({e["type"]})' if e.get("type") else "")
        + (f' placeholder="{e["ph"]}"' if e.get("ph") else "")
        + (f' "{e["text"]}"' if e.get("text") and e["tag"] in ("button", "a") else "")
        for e in elements
    )


# ─── URL Recon ───────────────────────────────────────────────────────────────

async def _tavily_recon_search(query: str) -> list[dict]:
    import asyncio
    import itertools

    try:
        from .. import config as _cfg
        keys = getattr(_cfg, "TAVILY_API_KEYS", [])
    except Exception:
        return []

    if not keys:
        return []

    key_iter = itertools.cycle(keys)
    max_attempts = min(2, len(keys))

    for _ in range(max_attempts):
        api_key = next(key_iter)
        try:
            import requests as _req

            def _do_search(k=api_key):
                resp = _req.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": k,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                    timeout=6,
                )
                resp.raise_for_status()
                return resp.json()

            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _do_search)
            return data.get("results", [])

        except Exception as e:
            error_str = str(e).lower()
            if any(code in error_str for code in ("429", "401", "403", "rate")):
                continue
            return []

    return []


async def _url_recon(goal: str, *, planner_goal: str = "") -> str | None:
    if _URL_PATTERN.search(goal):
        return None

    # Build a richer search query: parent planner goal + GEO city
    search_query = goal
    if planner_goal and planner_goal != goal:
        search_query = planner_goal
    from ..core.geolocation import get_cached_region
    _geo = get_cached_region() or {}
    _city = _geo.get("city", "")
    if _city and _city.lower() not in search_query.lower():
        search_query = f"{search_query} {_city}"

    try:
        results = await _tavily_recon_search(search_query)
    except Exception:
        return None

    results = [r for r in results if r.get("url")]
    if not results:
        return None

    # Domain hints come from the STEP goal (has site name like "BookMyShow")
    hints = [
        w.lower().strip(".,!?")
        for w in goal.split()
        if len(w) > 2 and w.lower().strip(".,!?") not in _DETECT_STOP_WORDS
    ]

    for hint in hints:
        for r in results:
            host = urlparse(r["url"]).hostname or ""
            if hint in host.lower():
                logger.info(f"[DA] URL recon: matched '{hint}' in {host} → {r['url']}")
                return r["url"]

    url = results[0]["url"]
    logger.info(f"[DA] URL recon: no domain hint match, using first result → {url}")
    return url


async def _execute_browser_task(goal: str, llm_func, *, _from_planner: bool = False,
                                _planner_goal: str = "") -> str:
    from .browser import automation as browser_automation

    # Shortcut: "open <browser> [and] [go to] <url>" — honor the explicit
    # browser name. Playwright Chromium is wrong here (sandboxed, no user
    # data, closed by run_browser_steps' finally block).
    open_browser_match = re.match(
        rf'^(?:open|launch|start|run)\s+({_BROWSER_ALT})'
        r'(?:\s+(?:and|then)\s+)?'
        r'(?:\s*(?:go\s+to|navigate\s+to|visit|at|with|to|on|open)\s+)?'
        r'(?:(https?://[^\s]+|www\.[^\s]+|[\w-]+(?:\.[\w-]+)+(?:/[^\s]*)?))?\s*$',
        goal, re.IGNORECASE,
    )
    if open_browser_match and not _from_planner:
        import webbrowser, subprocess
        browser_name = open_browser_match.group(1).lower()
        url = open_browser_match.group(2)
        if url and not url.startswith("http"):
            url = "https://" + url
        target = url or "about:blank"

        # "browser" is generic → use system default.
        if browser_name == "browser":
            logger.info(f"[DA] Open-browser shortcut → default, target={target}")
            webbrowser.open(target)
            return f"Opened {target} in your browser" if url else "Opened your browser"

        exe = _find_browser_executable(browser_name)
        if exe:
            try:
                args = [exe, target] if url else [exe]
                subprocess.Popen(args, close_fds=True)
                logger.info(f"[DA] Open-browser shortcut → {exe} {target}")
                return f"Opened {target} in {browser_name.capitalize()}" if url else f"Opened {browser_name.capitalize()}"
            except Exception as e:
                logger.warning(f"[DA] Failed to launch {browser_name} via {exe}: {e}")

        logger.warning(f"[DA] {browser_name} not found on system; falling back to default")
        webbrowser.open(target)
        return f"Opened {target} in your default browser ({browser_name} not found)" if url else "Opened your browser"

    # Simple shortcut: direct URL navigation
    # Skip when from planner — route through run_browser_steps for page persistence.
    simple_match = re.match(r'^(go\s+to|visit|open)\s+(https?://[^\s]+|www\.[^\s]+|\w+\.\w+)$', goal, re.IGNORECASE)
    if not _from_planner and simple_match and "and" not in goal.lower() and "then" not in goal.lower():
        url = simple_match.group(2)
        if not url.startswith('http'):
            url = 'https://' + url
        return await browser_automation.extract_text(url)

    # Simple shortcut: web search → open Google URL in user's real browser
    # Uses webbrowser.open(), not Playwright — user sees results in their actual browser,
    # no headless rendering, no CAPTCHA, no Playwright subprocess deadlock on Windows.
    # Skip when called from planner — the search should target the page loaded
    # by a prior step (e.g. "search for X" on Wikipedia), not Google.
    if not _from_planner:
        search_match = re.match(
            rf'^search\s+(?:for\s+|on\s+(?:the\s+)?(?:web|google|internet|{_BROWSER_ALT})\s+(?:for\s+)?)?(.+?)(?:\s+on\s+(?:{_BROWSER_ALT}|the\s+web|google|internet))?$',
            goal, re.IGNORECASE,
        )
        if search_match:
            from urllib.parse import quote_plus
            import webbrowser
            query = search_match.group(1).strip()
            if query:
                url = f"https://www.google.com/search?q={quote_plus(query)}"
                logger.info(f"[DA] Browser search shortcut: '{query}' → {url}")
                webbrowser.open(url)
                return f"Opened Google search for '{query}'"

    # ── Cache check: try cached steps before calling LLM planner ──────
    from . import step_cache as _step_cache
    _cached_steps = _step_cache.load_cached_steps("browser", "browser", goal)
    if _cached_steps is not None:
        logger.info(f"[DA] Cache HIT for browser task: {len(_cached_steps)} steps")
        try:
            res = await browser_automation.run_browser_steps(_cached_steps, _from_planner=_from_planner)
            if res.startswith("Error") or "VERIFY_FAILED" in (res or "")[:600]:
                logger.warning(f"[DA] Cached browser steps failed, deleting cache: {res[:120]}")
                _step_cache.delete_cached_steps("browser", "browser", goal)
            else:
                logger.info(f"[DA] Cached browser task completed: {res}")
                return res
        except Exception as e:
            logger.warning(f"[DA] Cached browser steps raised: {e}, deleting cache")
            _step_cache.delete_cached_steps("browser", "browser", goal)

    # ── Planner context injection: tell LLM what page is already loaded ──
    _cache_goal = goal
    _had_elements = False
    _planner_page_loaded = False
    if _from_planner:
        _page_info = await browser_automation.get_planner_page_info()
        if _page_info:
            _goal_domain = _extract_domain(goal)
            _page_domain = _extract_domain(_page_info["url"])
            _domain_match = (
                not _goal_domain
                or not _page_domain
                or _goal_domain == _page_domain
                or _page_domain.endswith("." + _goal_domain)
                or _goal_domain.endswith("." + _page_domain)
            )
            if not _domain_match:
                logger.info(
                    f"[DA] Planner page domain mismatch: "
                    f"page={_page_domain}, goal={_goal_domain} — skipping stale context"
                )
            else:
                _planner_page_loaded = True
                goal = f"[Current browser: {_page_info['title']} — {_page_info['url']}]\n{goal}"
                logger.info(f"[DA] Injected planner page context: {_page_info['title']} — {_page_info['url']}")
                _elements = await browser_automation.get_interactive_elements()
                if _elements:
                    _el_lines = _format_elements(_elements)
                    goal = f"{goal}\n[Interactive elements on page (use these selectors for fill/click):\n{_el_lines}\n]"
                    logger.info(f"[DA] Injected {len(_elements)} interactive element(s) into prompt")
                    _had_elements = True

    # ── URL Recon: web search for target URL before LLM step generation ──
    if not _planner_page_loaded and not _URL_PATTERN.search(_cache_goal):
        _recon_url = await _url_recon(_cache_goal, planner_goal=_planner_goal)
        if _recon_url:
            goal = f"[Target URL: {_recon_url}]\n{goal}"

    # Ask LLM for steps
    from ..core.geolocation import get_cached_region, format_region_hint
    _region_hint = format_region_hint(get_cached_region())
    prompt = _BROWSER_PLAN_PROMPT.format(goal=goal, region_hint=_region_hint)
    _plan_system = "You are a browser automation planner. Return ONLY valid JSON arrays, no other text."
    try:
        response = await _maybe_await(llm_func, prompt, task_type="intent", system_prompt=_plan_system, max_tokens=500)
    except Exception as e:
        logger.error(f"[DA] Browser LLM plan failed: {e}")
        return "__FALLBACK__"

    steps = _extract_json_array(response)
    if not steps:
        url_match = _URL_PATTERN.search(goal)
        if url_match:
            url = url_match.group(0)
            if not url.startswith('http'): url = 'https://' + url
            if _from_planner:
                logger.info(f"[DA] JSON parse failed, fallback navigate to {url} (planner path)")
                return await browser_automation.run_browser_steps(
                    [{"action": "navigate", "params": {"url": url}}],
                    _from_planner=True,
                )
            return await browser_automation.extract_text(url)
        return "__FALLBACK__"

    # ── Two-pass: navigate first, scan DOM, re-plan remaining steps ────
    # When the LLM planned navigate + interact without real selectors,
    # execute the navigate, scan the live DOM, and re-plan so the LLM
    # picks from real elements instead of guessing.
    _INTERACT_ACTIONS = frozenset({"fill", "click", "select"})
    _needs_two_pass = (
        not _had_elements
        and len(steps) > 1
        and steps[0].get("action") == "navigate"
        and any(s.get("action") in _INTERACT_ACTIONS for s in steps[1:])
    )

    if _needs_two_pass:
        logger.info("[DA] Two-pass: executing navigate first to scan DOM")
        try:
            nav_res = await browser_automation.run_browser_steps(
                [steps[0]], _from_planner=_from_planner
            )
            if nav_res.startswith("Error"):
                return "__FALLBACK__"
            if "VERIFY_FAILED" in nav_res[:600]:
                return nav_res

            _elements = await browser_automation.get_interactive_elements()
            if _elements:
                _el_lines = _format_elements(_elements)
                _pi = await browser_automation.get_planner_page_info()
                _ctx = f"[Current browser: {_pi['title']} — {_pi['url']}]\n" if _pi else ""
                _remaining_goal = (
                    f"{_ctx}{_cache_goal}\n"
                    f"[Interactive elements on page (use these selectors for fill/click):\n{_el_lines}\n]\n"
                    f"The page is already loaded. Do NOT include a navigate step."
                )
                logger.info(f"[DA] Two-pass: injected {len(_elements)} element(s), re-planning")
                try:
                    _replan_resp = await _maybe_await(
                        llm_func,
                        _BROWSER_PLAN_PROMPT.format(goal=_remaining_goal, region_hint=_region_hint),
                        task_type="intent",
                        system_prompt=_plan_system,
                        max_tokens=500,
                    )
                    _replan_steps = _extract_json_array(_replan_resp)
                    if _replan_steps:
                        _replan_steps = [
                            s for s in _replan_steps if s.get("action") != "navigate"
                        ]
                        if _replan_steps:
                            steps = [steps[0]] + _replan_steps
                            logger.info(f"[DA] Two-pass: re-planned to {len(steps)} total steps")
                except Exception as e:
                    logger.warning(f"[DA] Two-pass re-plan failed, using original steps: {e}")

            remaining = steps[1:]
            if remaining:
                rem_res = await browser_automation.run_browser_steps(
                    remaining, _from_planner=_from_planner
                )
                res = f"{nav_res}\n{rem_res}"
            else:
                res = nav_res

            if res.startswith("Error"):
                return "__FALLBACK__"
            if "VERIFY_FAILED" not in res[:600] and "[ABORT]" not in res[:600]:
                try:
                    _step_cache.save_cached_steps("browser", "browser", _cache_goal, steps)
                except Exception as e:
                    logger.debug(f"[DA] Cache save failed (non-fatal): {e}")
            return res
        except Exception as e:
            logger.error(f"[DA] Two-pass run failed: {e}")
            return "__FALLBACK__"

    try:
        res = await browser_automation.run_browser_steps(steps, _from_planner=_from_planner)
        if res.startswith("Error"):
            return "__FALLBACK__"
        if "VERIFY_FAILED" not in (res or "")[:600] and "[ABORT]" not in (res or "")[:600]:
            try:
                _step_cache.save_cached_steps("browser", "browser", _cache_goal, steps)
            except Exception as e:
                logger.debug(f"[DA] Cache save failed (non-fatal): {e}")
        return res
    except Exception as e:
        logger.error(f"[DA] run_browser_steps failed: {e}")
        return "__FALLBACK__"

_APP_TARGET_SUFFIX_RE = re.compile(
    r'\b(?:on|in|with|using)\s+(?:the\s+)?(\w+)\s*$',
    re.IGNORECASE,
)
_APP_TARGET_STOP_WORDS = frozenset({
    "it", "that", "this", "now", "here", "them", "all", "mode",
    "field", "form", "input", "textbox", "box",
})


def _extract_target_app(goal: str) -> tuple[Optional[str], str]:
    """
    Parse `...in/on/with/using <app>` suffix.

    Returns (target_app_lowercase or None, goal_with_suffix_stripped).
    Used by _execute_native_task to honour an explicit target instead of
    silently falling back to the active window — which previously caused
    Ctrl+N to fire inside the user's IDE when they said `type X in notepad`.
    """
    m = _APP_TARGET_SUFFIX_RE.search(goal)
    if not m:
        return None, goal
    candidate = m.group(1).lower()
    if len(candidate) <= 2 or candidate in _APP_TARGET_STOP_WORDS:
        return None, goal
    return candidate, goal[:m.start()].rstrip()


async def _resolve_target_window(target_app: str) -> Optional[str]:
    """Find or open `target_app`; return its window title or None.

    Order: existing open window → app_automation.open_app → re-check
    open windows. Never falls back to the active window — callers must
    decide what to do when the target genuinely cannot be reached.
    """
    from ..io import screen
    from . import native as app_automation

    needle = target_app.lower()
    try:
        for w in screen.get_open_windows():
            if needle in w.lower():
                logger.info(f"[DA] Target app '{target_app}' already running: '{w}'")
                return w
    except Exception:
        pass

    logger.info(f"[DA] Target app '{target_app}' not running — opening it")
    try:
        open_res = await app_automation.open_app(target_app)
        logger.info(f"[DA] open_app result: {open_res}")
    except Exception as e:
        logger.warning(f"[DA] open_app crashed for '{target_app}': {e}")
        return None

    await asyncio.sleep(1.0)
    try:
        for w in screen.get_open_windows():
            if needle in w.lower():
                logger.info(f"[DA] Target app '{target_app}' opened: '{w}'")
                return w
    except Exception:
        pass

    logger.warning(f"[DA] Target app '{target_app}' window did not appear after open")
    return None


async def _execute_native_task(goal: str, llm_func) -> str:
    from . import native as app_automation
    from . import verification

    async def _verified(action: str, name: str, res: str) -> str:
        """Post-verify a single-step action. Returns the structured
        VERIFY_FAILED|... prefix on confident verification failure so the
        planner / actions layer parses it the same way as run_app_steps."""
        try:
            vr = await verification.post_verify(
                {"type": "app", "action": action, "params": {"name": name}},
            )
        except Exception as e:
            logger.warning(f"[DA] verification crashed: {e}")
            return res
        if vr.tier == "ambiguous" and getattr(config, "VERIFY_VISION_FALLBACK", True):
            try:
                vr = await verification.vision_verify(
                    {"type": "app", "action": action, "params": {"name": name}}, vr,
                )
            except Exception as e:
                logger.warning(f"[DA] vision verification crashed: {e}")
        if (not vr.ok and not vr.skipped
                and vr.confidence >= getattr(config, "VERIFY_MIN_CONFIDENCE", 0.5)):
            logger.warning(f"[DA] verify_failed (single-step {action}): {vr.observation}")
            return f"VERIFY_FAILED|step=1|tier={vr.tier}|obs={vr.observation}\n{res}"
        return res

    # Simple shortcut — but focus existing window instead of launching a new instance
    simple_match = re.match(r'^(open|launch|start|run)\s+(.+)$', goal, re.IGNORECASE)
    if simple_match and "and" not in goal.lower() and "then" not in goal.lower():
        app_name = simple_match.group(2)
        running_window = _detect_running_app(goal)
        if running_window:
            logger.info(f"[DA] App already running: '{running_window}', focusing instead of opening")
            res = await app_automation.focus_window(running_window)
            if "Error" not in res and "not found" not in res.lower():
                return await _verified("focus", running_window, res)
            logger.info(f"[DA] Focus failed ({res}), falling through to open_app")
        res = await app_automation.open_app(app_name)
        return await _verified("open", app_name, res)

    # Honour explicit target from "...in/on/with/using <app>" suffix BEFORE
    # the active-window fallback. Without this, "type X in notepad" with
    # Notepad closed used to fall back to whatever was foreground (the user's
    # IDE), focus it, fire Ctrl+N (new tab in IDE), and type into the IDE.
    target_app, goal_for_planner = _extract_target_app(goal)
    running_window: Optional[str] = None
    if target_app:
        logger.info(f"[DA] Target app from suffix: '{target_app}', stripped goal: '{goal_for_planner}'")
        running_window = await _resolve_target_window(target_app)
        if not running_window:
            logger.warning(
                f"[DA] Target '{target_app}' could not be opened/located — "
                f"refusing active-window fallback to avoid hijacking the foreground app"
            )
            return "__FALLBACK__"

    # Gather available elements if the app is already open
    if not running_window:
        running_window = _detect_running_app(goal_for_planner)

    # Fallback: if no keyword match, wait briefly for dialogs to render
    # (e.g., planner step just clicked Save As menu → dialog is still appearing)
    # then use the active (foreground) window. Generic — works for any dialog.
    # Skipped when an explicit target was specified — that path already
    # decided above and bailed if it couldn't resolve the target.
    if not running_window:
        await asyncio.sleep(0.5)
        running_window = _detect_running_app(goal_for_planner)
    if not running_window:
        try:
            from ..io import screen
            active = screen.get_active_window()
            if active and active.strip():
                running_window = active
                logger.info(f"[DA] Fallback: using active window '{active}' (no keyword match)")
        except Exception:
            pass

    available_elements = ""
    pre_steps = []  # Steps to execute in code BEFORE the LLM plan
    if running_window:
        logger.info(f"[DA] Detected running window: '{running_window}'")

        # Code-level: always focus the window first (don't trust LLM to do it)
        pre_steps.append({"action": "focus", "params": {"name": running_window}})

        # Code-level: open new tab/document if existing content detected.
        # ONLY for type/write goals — not for click/save/navigate goals.
        # Without this guard, ctrl+n fires on "click File menu" and kills
        # the tab the user was working in.
        _is_typing_goal = re.search(r'\b(type|write|enter|input|paste)\b', goal_for_planner, re.IGNORECASE)
        if _is_typing_goal:
            new_tab_key = _detect_new_tab_key(running_window, goal_for_planner)
            if new_tab_key:
                pre_steps.append({"action": "press_key", "params": {"key": new_tab_key}})
                pre_steps.append({"action": "wait", "params": {"seconds": 1}})
                logger.info(f"[DA] Pre-step: new tab/document via {new_tab_key}")

        already_open_hint = (
            f"IMPORTANT: The application is ALREADY OPEN, FOCUSED, and ready for input. "
            f"The currently focused window title is EXACTLY: \"{running_window}\". "
            f"For any step that takes a 'window' param, you MUST use this EXACT string. "
            f"Do NOT invent, shorten, expand, or substitute any other window title — "
            f"if you are tempted to use a different name, you are wrong. "
            f"Do NOT include 'open', 'focus', or 'press_key ctrl+n/ctrl+t' steps — that is already handled. "
            f"Start directly with interaction steps (click, type, get_text).\n"
        )
        elements_dump = await app_automation.list_elements(running_window)
        if elements_dump and "Error" not in elements_dump and "Could not find" not in elements_dump:
            logger.debug(f"[DA] Available elements for '{running_window}':\n{elements_dump}")
            available_elements = f"\n{already_open_hint}\nAvailable elements in '{running_window}':\n{elements_dump}\n"
        else:
            logger.info(f"[DA] No usable elements for '{running_window}' (got: {elements_dump[:120] if elements_dump else 'None'})")
            available_elements = f"\n{already_open_hint}\n"
    else:
        logger.info(f"[DA] No running window detected for goal: '{goal_for_planner}'")

    # Type-shortcut: "type/write/enter/input/paste X" where X is unambiguous
    # literal text. Skips the LLM step planner — for long prose the planner
    # would have to JSON-encode the entire paragraph in its output, which
    # truncates and fails after ~90s.
    #
    # Three triggers (only one needs to match):
    #   1. Quoted: `type 'foo'` or `type "foo"` → strip quotes, type the rest.
    #   2. Long blob: > _LONG_TEXT_THRESHOLD chars or contains a newline →
    #      almost certainly a paste-style command, not an instruction.
    #   3. Otherwise FALL THROUGH to LLM. Skips ambiguous goals like
    #      "type my email in the form" (instruction, not literal text) where
    #      the LLM with element context can plan a click+fill correctly.
    _pure_type_match = re.match(
        r'^(?:type|write|enter|input|paste)\s+(.+)$',
        goal_for_planner, re.IGNORECASE | re.DOTALL,
    )
    _LONG_TEXT_THRESHOLD = 80
    if running_window and _pure_type_match:
        text_to_type = _pure_type_match.group(1).strip()
        is_quoted = (
            len(text_to_type) >= 2
            and text_to_type[0] == text_to_type[-1]
            and text_to_type[0] in "\"'"
        )
        is_long_blob = len(text_to_type) > _LONG_TEXT_THRESHOLD or "\n" in text_to_type

        if is_quoted or is_long_blob:
            if is_quoted:
                text_to_type = text_to_type[1:-1]
            logger.info(
                f"[DA] Type-shortcut: bypassing LLM, typing {len(text_to_type)} chars "
                f"into '{running_window}' (reason={'quoted' if is_quoted else 'long_blob'})"
            )
            steps = pre_steps + [{"action": "type", "params": {"text": text_to_type, "window": running_window}}]
            try:
                res = await app_automation.run_app_steps(steps)
                if isinstance(res, str) and res.startswith("Error"):
                    logger.warning(f"[DA] run_app_steps returned error: {res}")
                    return "__FALLBACK__"
                logger.info(f"[DA] Type-shortcut completed: {res[:120]}")
                return res
            except Exception as e:
                # never swallow a user abort into a fallback path —
                # that re-triggers computer_task / TTS / vision while the
                # user is trying to stop everything.
                from assistant.core.abort import UserAborted
                if isinstance(e, UserAborted):
                    raise
                logger.error(f"[DA] Type-shortcut failed: {e}")
                return "__FALLBACK__"
        else:
            logger.info(
                f"[DA] Type-shortcut skipped (text looks like an instruction, not literal: "
                f"{text_to_type[:60]!r}); falling through to LLM planner"
            )

    # ── Cache check: try cached steps before calling LLM planner ──────
    from . import step_cache as _step_cache
    _cache_app = target_app or (running_window or "").split(" - ")[0].strip().lower() or "unknown"
    _cached_steps = _step_cache.load_cached_steps("native", _cache_app, goal_for_planner)
    if _cached_steps is not None:
        logger.info(f"[DA] Cache HIT for native/{_cache_app}: {len(_cached_steps)} steps")
        _replay_steps = (pre_steps + _cached_steps) if pre_steps else _cached_steps
        try:
            res = await app_automation.run_app_steps(_replay_steps)
            _res_str = str(res) if res else ""
            if _res_str.startswith("Error") or "VERIFY_FAILED" in _res_str[:600]:
                logger.warning(f"[DA] Cached steps failed, deleting cache: {_res_str[:120]}")
                _step_cache.delete_cached_steps("native", _cache_app, goal_for_planner)
                # Fall through to LLM planner below
            else:
                logger.info(f"[DA] Cached native task completed: {res}")
                return res
        except Exception as e:
            logger.warning(f"[DA] Cached steps raised: {e}, deleting cache entry")
            _step_cache.delete_cached_steps("native", _cache_app, goal_for_planner)
            # Fall through to LLM planner below

    # Ask LLM for steps — use neutral system prompt to avoid personality leaking into JSON
    prompt = _APP_PLAN_PROMPT.format(goal=goal_for_planner, available_elements=available_elements)
    _plan_system = "You are a desktop automation planner. Return ONLY valid JSON arrays, no other text."
    try:
        response = await _maybe_await(llm_func, prompt, task_type="intent", system_prompt=_plan_system, max_tokens=500)
    except Exception as e:
        logger.error(f"[DA] App LLM plan failed: {e}")
        return "__FALLBACK__"

    steps = _extract_json_array(response)
    if not steps:
        logger.warning(f"[DA] LLM returned no valid steps. Raw response: {response[:300] if response else 'None'}")
        return "__FALLBACK__"
    logger.info(f"[DA] LLM planned {len(steps)} steps: {steps}")

    # --- Deterministic step-plan fixes (like code_executor's deterministic fixes) ---
    # Applied BEFORE execution to compensate for common LLM planning errors.

    # Fix A: Strip get_text steps from pure type/write tasks — LLM often adds get_text
    # with hallucinated selectors, causing 60s+ tree walks for no reason
    _TYPE_WORDS = {"type", "write", "enter", "input", "paste"}
    _RESULT_WORDS = {"calculate", "compute", "result", "read", "get", "show", "what", "check", "display"}
    goal_words_set = set(goal_for_planner.lower().split())
    if goal_words_set & _TYPE_WORDS and not goal_words_set & _RESULT_WORDS:
        before = len(steps)
        steps = [s for s in steps if s.get("action") != "get_text"]
        if len(steps) < before:
            logger.info(f"[DA] Stripped {before - len(steps)} get_text steps from type task")

    # Fix B: Sanitize LLM steps — strip redundant open/focus, conflicting press_key
    steps = _sanitize_steps(steps, has_pre_steps=bool(pre_steps))

    # Prepend code-level steps (focus, new tab) before LLM steps
    if pre_steps:
        steps = pre_steps + steps
        logger.info(f"[DA] Total steps after prepend: {len(steps)}")

    try:
        res = await app_automation.run_app_steps(steps)
        if isinstance(res, str) and res.startswith("Error"):
            logger.warning(f"[DA] run_app_steps returned error: {res}")
            return "__FALLBACK__"
        logger.info(f"[DA] Native task completed: {res}")
        _res_str = res if isinstance(res, str) else ""
        _llm_steps = steps[len(pre_steps):] if pre_steps else steps
        if _llm_steps and "VERIFY_FAILED" not in _res_str[:600] and "[ABORT]" not in _res_str[:600]:
            _cacheable = [
                {"action": s["action"], "params": {k: v for k, v in s.get("params", {}).items() if k != "window"}}
                for s in _llm_steps
            ]
            try:
                _step_cache.save_cached_steps("native", _cache_app, goal_for_planner, _cacheable)
            except Exception as e:
                logger.debug(f"[DA] Cache save failed (non-fatal): {e}")
        return res
    except Exception as e:
        logger.error(f"[DA] run_app_steps failed: {e}")
        return "__FALLBACK__"

def _sanitize_steps(steps: list, has_pre_steps: bool = False) -> list:
    """
    Deterministic cleanup of LLM-generated step plans.
    Same philosophy as code_executor's deterministic fixes:
    catch common LLM planning errors BEFORE execution, zero API calls.

    Fixes applied:
      1. Strip redundant open/focus when pre-steps already handled window management
      2. Strip press_key that duplicates the terminator in a preceding type step
         (e.g., type "76*23=" followed by press_key enter → enter re-evaluates)
      3. Convert type steps that are keyboard shortcuts to press_key
         (LLM types "ctrl+s" as text instead of pressing the key combo)
      4. After press_key with modifier (ctrl/alt/shift), strip window from next type
         steps and inject wait if missing. Keyboard shortcuts open dialogs — typing
         should go to the dialog, not re-focus the original window.
    """
    if not steps:
        return steps

    _SHORTCUT_RE = re.compile(
        r'^(ctrl|alt|shift|win|cmd)[\+\s]',
        re.IGNORECASE,
    )
    _SINGLE_KEYS = frozenset({
        "enter", "return", "escape", "esc", "tab",
        "delete", "backspace", "space", "home", "end",
        "pageup", "pagedown", "f1", "f2", "f3", "f4",
        "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    })
    _MODIFIER_RE = re.compile(r'(ctrl|alt|shift|win|cmd)', re.IGNORECASE)

    # Fix 4 state: track when a modifier shortcut was pressed (dialog likely opened)
    after_modifier_press = False

    cleaned = []
    for step in steps:
        action = step.get("action", "")
        params = step.get("params", {})

        # Fix 1: Strip open/focus — pre-steps already focused the window
        if has_pre_steps and action in ("open", "focus"):
            logger.info(f"[DA] Sanitize: stripped redundant '{action}' (pre-steps handled)")
            continue

        # Fix 2: Strip press_key that duplicates end of preceding type text
        if action == "press_key" and cleaned:
            prev = cleaned[-1]
            if prev.get("action") == "type":
                prev_text = prev.get("params", {}).get("text", "")
                key = params.get("key", "").lower()
                if key in ("enter", "return") and prev_text.rstrip().endswith(("=", "\n", "\r")):
                    logger.info(f"[DA] Sanitize: stripped redundant press_key '{key}' "
                                f"(type already ends with '{prev_text.rstrip()[-1]}')")
                    continue

        # Fix 3: Convert type "ctrl+s" / type "Enter" → press_key
        # 8b LLM confuses text input with keyboard shortcuts ~30% of the time
        if action == "type":
            text = params.get("text", "")
            if _SHORTCUT_RE.match(text):
                key = text.lower().replace(" ", "+")
                step = {"action": "press_key", "params": {"key": key}}
                action = "press_key"  # update for Fix 4 tracking
                logger.info(f"[DA] Sanitize: converted type '{text}' to press_key '{key}'")
            elif text.lower().strip() in _SINGLE_KEYS:
                key = text.lower().strip()
                step = {"action": "press_key", "params": {"key": key}}
                action = "press_key"
                logger.info(f"[DA] Sanitize: converted type '{text}' to press_key '{key}'")

        # Fix 4: Track modifier shortcuts → strip window from next type steps
        # Keyboard shortcuts with modifiers (ctrl+s, ctrl+shift+s, alt+f4, etc.)
        # often open dialogs. The next type step should go to the dialog, not
        # re-focus the original window (which switches away from the dialog).
        if action == "press_key":
            key = step.get("params", {}).get("key", "")
            after_modifier_press = bool(_MODIFIER_RE.search(key))
        elif action in ("focus", "open", "click"):
            # Explicit targeting resets dialog mode
            after_modifier_press = False
        elif action == "type" and after_modifier_press:
            # Inject wait if not present — give the dialog time to open
            if cleaned and cleaned[-1].get("action") not in ("wait",):
                cleaned.append({"action": "wait", "params": {"seconds": 0.5}})
                logger.info("[DA] Sanitize: injected wait after modifier shortcut for dialog")
            # Strip window param so run_app_steps won't re-focus original window
            if params.get("window"):
                new_params = {k: v for k, v in params.items() if k != "window"}
                step = {"action": "type", "params": new_params}
                logger.info(f"[DA] Sanitize: stripped window from type (dialog mode after modifier shortcut)")

        cleaned.append(step)

    return cleaned


async def _execute_dom_task(
    goal: str,
    foreground_window_title: Optional[str] = None,
) -> str:
    """
    Dispatch entry for the DOM-aware browser path.

    Steps:
      1. Attach to the user's Chrome via browser_cdp (CDP probe → connect).
      2. Pick the user's active tab from the attached contexts.
      3. Run the perceive→plan→execute orchestrator (browser_dom_orchestrator).
      4. Format the DomTaskResult into a TTS-friendly reply.

    `foreground_window_title` is the OS-level active Chrome window title
    (e.g. "Truein: AI Based... - Google Chrome"). When supplied, the page
    picker prefers the CDP page whose <title> matches the stripped window
    title — fixes the multi-tab case where MRU order disagrees with the
    user's foreground tab.

    Returns the spoken-style result string, or "__FALLBACK__" when CDP
    attachment fails / no tabs / orchestrator can't proceed — the caller
    routes that to vision-loop.
    """
    try:
        from .browser import cdp as browser_cdp, dom_orchestrator as browser_dom_orchestrator
    except Exception as e:
        logger.warning(f"[DA] DOM-mode imports failed: {e}")
        return "__FALLBACK__"

    # 1. Attach
    try:
        handle = await browser_cdp.get_or_attach_browser(prefer_cdp=True)
    except Exception as e:
        logger.warning(f"[DA] DOM-mode attach raised: {type(e).__name__}: {e}")
        return "__FALLBACK__"

    if handle.kind != "cdp":
        # The router said "dom" but get_or_attach_browser couldn't get CDP.
        # Likely race: probe was cached available but Chrome closed. Fall back.
        logger.info(
            f"[DA] DOM-mode requested but attach returned kind={handle.kind!r} "
            f"— falling back to vision-loop"
        )
        return "__FALLBACK__"

    # 2. Pick the user's active tab. When the OS-level foreground window
    # title is known, prefer the CDP page whose <title> matches — this is
    # the only signal that survives across multi-tab Chrome windows, since
    # Playwright's `pages` list ordering is not aligned with foreground.
    # Falls back to first non-internal page (MRU-ish) when no match.
    try:
        target_page = await _pick_active_page(
            handle.attachment,
            prefer_window_title=foreground_window_title,
        )
    except Exception as e:
        logger.warning(f"[DA] page selection failed: {type(e).__name__}: {e}")
        target_page = None

    if target_page is None:
        logger.info("[DA] DOM-mode: no active page found in CDP-attached browser")
        return "__FALLBACK__"

    # 3. Run the orchestrator
    try:
        try:
            page_url = target_page.url
        except Exception:
            page_url = "(unreadable)"
        logger.info(f"[DA] DOM-mode running on page: {page_url!r}")
        if _FORM_INTENT_RE.search(goal):
            result = await browser_dom_orchestrator.run_dom_form_fill(goal, target_page)
        else:
            result = await browser_dom_orchestrator.run_dom_task(goal, target_page)
    except Exception as e:
        logger.error(f"[DA] DOM-mode orchestrator crashed: {type(e).__name__}: {e}")
        return "__FALLBACK__"

    # 4. Format the result for the TTS reply
    if result.success:
        msg = result.final_summary or "Done."
        logger.info(
            f"[DA] DOM-mode SUCCESS: {result.reason} loops={result.loops_used} "
            f"summary={msg!r}"
        )
        return msg
    # Failure modes carry their own final_summary (max_loops, planner_failed,
    # loop_failure_at_max, perceive_failed, empty_tree).
    logger.warning(
        f"[DA] DOM-mode FAILED: {result.reason} loops={result.loops_used} "
        f"summary={result.final_summary!r}"
    )
    # Return the summary as a non-fallback result — the user gets honest
    # feedback ("could not complete within 5 steps") rather than silent
    # vision-loop retry. If the caller wants vision fallback for specific
    # failure modes, it can match on the reason via the result.
    if result.reason in ("perceive_failed", "empty_tree"):
        # These two failure modes mean DOM-mode never had a usable foundation
        # — vision-loop is the right fallback. Other failures already burned
        # planner/executor cycles; another vision pass would be slower and
        # wouldn't recover the time already spent.
        return "__FALLBACK__"
    return result.final_summary or "I wasn't able to complete that fully."


def _strip_browser_window_suffix(window_title: str) -> str:
    """
    Strip the trailing browser-name suffix from an OS window title to
    recover the underlying page <title>. Returns "" when the input is
    empty / generic ("Google Chrome" with no page).
    """
    if not window_title:
        return ""
    stripped = _BROWSER_WINDOW_SUFFIX_RE.sub("", window_title).strip()
    # If stripping consumed everything, the window had no real page title
    # (e.g. just "Google Chrome" on a blank New Tab). Treat as no-hint.
    if not stripped or stripped.lower() == window_title.strip().lower():
        # Second clause: suffix didn't match at all. Could be a non-Chrome
        # browser we don't know about, or unusual locale. Keep the title
        # as-is — substring matching will still work in most cases.
        return stripped if stripped else ""
    return stripped


async def _pick_active_page(attachment, prefer_window_title: Optional[str] = None) -> Any:
    """
    Among the user's open tabs, pick the most likely "current" one.

    When `prefer_window_title` is supplied (e.g. the OS-level foreground
    Chrome window title), this strips the browser-name suffix and matches
    the remainder against each candidate page's `<title>`. Substring-match
    is bidirectional (Chrome may truncate long page titles in the window
    chrome). On match, returns that page.

    Otherwise — or when no candidate page's title matches — falls through
    to the original MRU-walk heuristic: first non-internal page wins.

    Returns None when there are no usable pages.
    """
    if attachment is None or not getattr(attachment, "contexts", None):
        return None

    # Collect non-internal candidates first so we can title-match across
    # all of them (not just within a single context).
    candidates: list = []
    for ctx in attachment.contexts:
        try:
            pages = list(getattr(ctx, "pages", []) or [])
        except Exception:
            continue
        for p in pages:
            try:
                url = (p.url or "").lower()
            except Exception:
                continue
            if url.startswith(("chrome://", "chrome-extension://", "devtools://",
                               "edge://", "brave://", "about:")):
                continue
            candidates.append(p)

    # Title-match path — only when we have a hint AND multiple candidates.
    # Single-candidate or no-hint short-circuits to the MRU fallback below
    # (cheap; avoids awaiting page.title() in the common case).
    if prefer_window_title and len(candidates) > 1:
        page_title_hint = _strip_browser_window_suffix(prefer_window_title).lower()
        if page_title_hint:
            for p in candidates:
                try:
                    pt = (await p.title() or "").strip().lower()
                except Exception:
                    continue
                if not pt:
                    continue
                if pt == page_title_hint or pt in page_title_hint or page_title_hint in pt:
                    logger.info(
                        f"[DA] _pick_active_page: matched tab by title "
                        f"hint={page_title_hint!r} → page={pt!r}"
                    )
                    return p
            logger.info(
                f"[DA] _pick_active_page: no tab matched window-title hint "
                f"{page_title_hint!r}; falling through to MRU"
            )

    if candidates:
        return candidates[0]

    # Fallback: first page across any context, even chrome://newtab —
    # better than None for goals like "navigate to X".
    for ctx in attachment.contexts:
        try:
            for p in ctx.pages:
                return p
        except Exception:
            continue
    return None


async def execute_automation(goal: str, llm_func, tts_func=None, bridge_func=None) -> str:
    backend, meta = detect_backend(goal)
    extras = {k: v for k, v in meta.items() if k != "reason"}
    extras_str = (" " + " ".join(f"{k}={v!r}" for k, v in extras.items())) if extras else ""
    logger.info(f"[DA] Routing goal '{goal}': backend={backend}, reason={meta.get('reason')}{extras_str}")

    if backend == "browser":
        res = await _execute_browser_task(goal, llm_func)
        if res == "__FALLBACK__":
            return "__FALLBACK__"
        return res
    elif backend == "native":
        res = await _execute_native_task(goal, llm_func)
        if res == "__FALLBACK__":
            return "__FALLBACK__"
        return res
    elif backend == "dom":
        # DOM-aware path via CDP attach + DOM planner orchestrator.
        # On any infrastructure failure (no CDP, no usable page, perceive
        # crash) we return "__FALLBACK__" and the caller routes to vision-loop.
        # Passes the OS-level foreground window title so the page picker can
        # match the right CDP tab in multi-tab Chrome windows.
        res = await _execute_dom_task(
            goal,
            foreground_window_title=meta.get("app"),
        )
        if res == "__FALLBACK__":
            return "__FALLBACK__"
        return res

    return "__FALLBACK__"

async def teach_routing(app_name: str, backend: str) -> str:
    """
    Store a user's routing preference for an app.
    backend: "browser" or "native" or "vision"
    """
    try:
        from .. import preferences
        preferences.set_preference(
            category="automation_routing",
            key=f"automation_{app_name.lower().strip()}",
            value=backend,
            confidence=0.85
        )
        return f"Got it, I'll use {backend} automation for {app_name} from now on."
    except Exception as e:
        logger.error(f"[DA] Failed to save routing preference: {e}")
        return f"Noted, but I couldn't save that preference: {e}"