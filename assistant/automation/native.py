"""
app_automation.py — Native desktop/application automation for TENKA.

Manages a singleton desktop automation instance using Terminator (Rust) 
with a fallback to pywinauto (Python). Provide sync API with async wrappers.

Desktop Automation Layer.
"""

import asyncio
import logging
from typing import Optional, Dict, List

from .. import config

logger = logging.getLogger("app_automation")

_backend = None
_desktop = None

# Module-level flag: set True while pyautogui is driving keyboard/mouse so the
# global push-to-talk hook (main.keyboard_listener) ignores synthesized events.
# Without this, typing a paragraph containing the hotkey letter (e.g. 'V'
# inside "Vestibulum") starts/stops recording mid-type and triggers a flood of
# "I didn't catch that" responses. Mirrors computer_agent._agent_typing.
_app_typing = False

try:
    import terminator
    _backend = "terminator"
    TERMINATOR_AVAILABLE = True
except ImportError:
    TERMINATOR_AVAILABLE = False
    logger.info("[APP] terminator not available, trying pywinauto...")
    try:
        import pywinauto
        from pywinauto import Desktop, Application
        import pywinauto.keyboard
        import pywinauto.findwindows
        _backend = "pywinauto"
        PYWINAUTO_AVAILABLE = True
    except ImportError:
        PYWINAUTO_AVAILABLE = False
        logger.warning("[APP] No native automation backend available — pip install terminator or pywinauto")

def is_available() -> bool:
    """Check if any native automation backend is available."""
    return _backend is not None


class _suppress_hotkey_hook:
    """Context manager that flips _app_typing for the duration of pyautogui-
    driven actions. Use around any block that synthesizes keyboard input."""

    def __enter__(self):
        global _app_typing
        _app_typing = True
        return self

    def __exit__(self, exc_type, exc, tb):
        global _app_typing
        _app_typing = False
        return False  # don't swallow exceptions

def _run_sync(func, *args, **kwargs):
    """Run a synchronous function within an asyncio context (needed by Terminator)"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return func(*args, **kwargs)

async def _to_thread(func, *args, **kwargs):
    """Wrapper to run a function in a thread with event loop setup."""
    def wrapper():
        return _run_sync(func, *args, **kwargs)
    return await asyncio.to_thread(wrapper)

def ensure_desktop():
    """
    Initialize and return the Terminator or pywinauto Desktop instance.
    """
    global _desktop
    if _desktop is not None:
        return _desktop
        
    if _backend == "terminator":
        # Terminator requires loop to init
        _desktop = _run_sync(terminator.Desktop)
        return _desktop
    elif _backend == "pywinauto":
        _desktop = Desktop(backend="uia")
        return _desktop
    return None

def _sync_open_app(name: str) -> str:
    desktop = ensure_desktop()
    if _backend == "terminator":
        desktop.open_application(name)
        return f"Opened application: {name}"
    elif _backend == "pywinauto":
        try:
            Application(backend="uia").start(name)
            return f"Opened application: {name}"
        except Exception as e:
            return f"Failed to open {name}: {e}"
    raise RuntimeError("No backend available")

def is_app_running(name: str) -> bool:
    """Return True if a process or visible window matches `name`.

    Two-stage detection (in order):
      1. psutil.process_iter — catches tray-minimized / no-window processes.
         pygetwindow alone misses Spotify when it's collapsed to the system
         tray (the original bug that bit livetest 2026-05-30).
      2. pygetwindow.getAllWindows — fallback. Catches apps whose process
         name differs from the user-friendly name (e.g. a browser PWA).

    Used by the code_executor launcher to skip the 'Opening ...' TTS +
    open_app + 20s poll when the app is already running (just retry the
    script directly).

    Returns False on any error (including both backends unavailable, or an
    empty/whitespace `name`). Caller should treat False as 'unknown —
    assume not running' and fall back to the full launcher flow.
    """
    if not name or not name.strip():
        return False
    name_lower = name.lower()
    # Stage 1: process scan via psutil (catches tray-minimized apps)
    try:
        import psutil
        for proc in psutil.process_iter(['name']):
            proc_name = (proc.info.get('name') or '').lower()
            if proc_name and name_lower in proc_name:
                return True
    except Exception as e:
        logger.debug(f"[APP] psutil scan for '{name}' failed: {e}")
    # Stage 2: window-title scan via pygetwindow (catches odd process names)
    try:
        import pygetwindow as gw
        return any(
            name_lower in w.title.lower() and w.title.strip()
            for w in gw.getAllWindows()
        )
    except Exception as e:
        logger.debug(f"[APP] pygetwindow scan for '{name}' failed: {e}")
        return False


async def open_app(name: str) -> str:
    """
    Open an application by name.
    Tries: already-running check → Terminator open_application → verify window appeared → Win key search fallback.
    """
    if not is_available(): return "Native automation not available."
    logger.info(f"[APP] Opening application: {name}")
    try:
        # P1: If the window is already open, focus it instead of launching again
        import pygetwindow as gw
        import ctypes
        existing = [w for w in gw.getAllWindows()
                    if name.lower() in w.title.lower() and w.title.strip()]
        if existing:
            hwnd = existing[0]._hWnd
            user32 = ctypes.windll.user32
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            await asyncio.sleep(0.3)
            logger.info(f"[APP] App already open, focused: {existing[0].title}")
            return f"Focused window: {existing[0].title}"

        if _backend == "terminator":
            desktop = ensure_desktop()
            try:
                desktop.open_application(name)
            except Exception as oa_err:
                logger.debug(f"[APP] open_application('{name}') raised: {oa_err}")
        else:
            await _to_thread(_sync_open_app, name)
        await asyncio.sleep(2.0)  # Wait for app to appear

        # Check if app actually opened + bring to foreground
        import pyautogui
        import ctypes
        import pygetwindow as gw

        all_windows = gw.getAllWindows()
        matches = [w for w in all_windows
                   if name.lower() in w.title.lower() and w.title.strip()]

        if matches:
            hwnd = matches[0]._hWnd
            user32 = ctypes.windll.user32
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            await asyncio.sleep(0.5)
            logger.info(f"[APP] Focused window after open: {matches[0].title}")
            return f"Opened application: {matches[0].title}"

        # Fallback: Win key search — works for any app in Start Menu
        # Generic, no hardcoded app-specific rules
        logger.info(f"[APP] open_application didn't produce window, trying Win key search for '{name}'")
        with _suppress_hotkey_hook():
            pyautogui.hotkey("win")
            await asyncio.sleep(0.7)
            pyautogui.write(name, interval=0.04)
            await asyncio.sleep(0.8)
            pyautogui.press("enter")
            await asyncio.sleep(2.5)

        # Re-check for window
        all_windows = gw.getAllWindows()
        matches = [w for w in all_windows
                   if name.lower() in w.title.lower() and w.title.strip()]
        if matches:
            logger.info(f"[APP] Win key search opened: {matches[0].title}")
            return f"Opened application: {matches[0].title}"

        return f"Opened application: {name}"
    except Exception as e:
        logger.error(f"[APP] Open app error: {e}")
        return f"Error opening {name}: {e}"

def _parse_selector_pywinauto(selector: str) -> dict:
    # "name:Seven" -> {"title": "Seven"}
    # "role:Button" -> {"control_type": "Button"}
    # "automationid:Calc" -> {"auto_id": "Calc"}
    kwargs = {}
    if not selector: return kwargs
    parts = selector.split(":", 1)
    if len(parts) == 2:
        k, v = parts
        k = k.lower().strip()
        if k == "name": kwargs["title"] = v
        elif k == "role": kwargs["control_type"] = v
        elif k == "automationid": kwargs["auto_id"] = v
        elif k == "window": kwargs["title"] = v
        else: kwargs["title"] = selector
    else:
        kwargs["title"] = selector
    return kwargs

def _get_pywinauto_element(selector: str, window: str = None):
    desktop = ensure_desktop()
    if window:
        win_kwargs = _parse_selector_pywinauto(window if ":" in window else f"window:{window}")
        win = desktop.window(**win_kwargs)
        if selector:
            sel_kwargs = _parse_selector_pywinauto(selector)
            return win.child_window(**sel_kwargs)
        return win
    else:
        sel_kwargs = _parse_selector_pywinauto(selector)
        return desktop.window(**sel_kwargs)

def _sync_click_element(selector: str, window: str = None) -> str:
    if _backend == "pywinauto":
        try:
            elem = _get_pywinauto_element(selector, window)
            elem.wait("visible", timeout=10)
            elem.click_input()
            return f"Clicked {selector}"
        except Exception as e:
            return f"Failed to click {selector}: {e}"
    raise RuntimeError("No backend available")

async def click_element(selector: str, window: str = None) -> str:
    """
    Click a UI element by accessibility selector.
    When window is specified, uses tree search (exact name match, PID-scoped)
    + coordinate-based clicking (works on Chromium/Electron apps).
    """
    if not is_available(): return "Native automation not available."
    logger.info(f"[APP] Clicking element: {selector} (window: {window})")
    try:
        if _backend == "terminator":
            desktop = ensure_desktop()

            # Extract target name and optional role from selector
            target_name, target_role = _parse_selector_parts(selector)

            # Approach 1: Tree search (PID-scoped, exact name match)
            # Terminator locators do substring matching which causes false positives
            if window:
                bounds = _find_element_bounds_in_tree(desktop, window, target_name, target_role)
                if bounds:
                    cx, cy = int(bounds['x'] + bounds['width'] / 2), int(bounds['y'] + bounds['height'] / 2)
                    import pyautogui
                    pyautogui.click(cx, cy)
                    logger.info(f"[APP] Clicked at ({cx}, {cy}) via tree search + coordinates")
                    return f"Clicked {selector}"
                # Approach 1b: Tree search failed — try locator but verify the
                # element is inside the target window's bounds to avoid clicking
                # the wrong element in another window.
                result = await _locator_within_window(desktop, selector, window)
                if result:
                    return result
                logger.warning(f"[APP] Element {selector} not found in window '{window}'")
                return f"Not found: {selector} in window '{window}'"

            # Approach 2: Locator fallback (only when no window specified)
            elem = await desktop.locator(selector).timeout(8000).first()
            try:
                logger.info(f"[APP] Locator found: name={elem.name()!r}, role={elem.role()!r}")
            except Exception:
                pass
            # Get bounds and click at coordinates
            bounds_fn = elem.bounds
            b = bounds_fn() if callable(bounds_fn) else bounds_fn
            if b:
                bx = getattr(b, 'x', None) or (b.get('x') if isinstance(b, dict) else None)
                by = getattr(b, 'y', None) or (b.get('y') if isinstance(b, dict) else None)
                bw = getattr(b, 'width', None) or (b.get('width') if isinstance(b, dict) else None)
                bh = getattr(b, 'height', None) or (b.get('height') if isinstance(b, dict) else None)
                if all(v is not None for v in (bx, by, bw, bh)):
                    cx, cy = int(bx + bw / 2), int(by + bh / 2)
                    import pyautogui
                    pyautogui.click(cx, cy)
                    logger.info(f"[APP] Clicked at ({cx}, {cy}) via locator + coordinates")
                    return f"Clicked {selector}"
            # Last resort: accessibility invoke
            elem.click()
            logger.info(f"[APP] Clicked via accessibility invoke")
            return f"Clicked {selector}"
        return await _to_thread(_sync_click_element, selector, window)
    except Exception as e:
        logger.error(f"[APP] Click error: {e}")
        return f"Error clicking {selector}: {e}"


async def _locator_within_window(desktop, selector: str, window: str) -> str | None:
    """Fallback: use Terminator's locator but verify the element is inside the target window."""
    try:
        import pygetwindow as gw
        matches = [w for w in gw.getAllWindows()
                   if window.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return None
        win = matches[0]
        wl, wt, wr, wb = win.left, win.top, win.right, win.bottom

        elem = await desktop.locator(selector).timeout(3000).first()
        bounds_fn = elem.bounds
        b = bounds_fn() if callable(bounds_fn) else bounds_fn
        if not b:
            return None
        bx = b.get('x') if isinstance(b, dict) else getattr(b, 'x', None)
        by = b.get('y') if isinstance(b, dict) else getattr(b, 'y', None)
        if bx is None or by is None:
            return None
        if not (wl <= bx <= wr and wt <= by <= wb):
            logger.debug(f"[APP] Locator found element at ({bx},{by}) but outside window bounds ({wl},{wt},{wr},{wb})")
            return None
        bw = b.get('width') if isinstance(b, dict) else getattr(b, 'width', None)
        bh = b.get('height') if isinstance(b, dict) else getattr(b, 'height', None)
        if bw and bh:
            cx, cy = int(bx + bw / 2), int(by + bh / 2)
        else:
            cx, cy = int(bx), int(by)
        import pyautogui
        pyautogui.click(cx, cy)
        logger.info(f"[APP] Clicked at ({cx}, {cy}) via locator (window-verified)")
        return f"Clicked {selector}"
    except Exception as e:
        logger.debug(f"[APP] Locator-within-window failed: {e}")
        return None


def _parse_selector_parts(selector: str) -> tuple[str, str | None]:
    """Extract target name and optional role from a selector like 'name:Play' or 'role:Button'."""
    parts = selector.split(":", 1)
    if len(parts) == 2:
        key, value = parts[0].lower().strip(), parts[1].strip()
        if key == "name":
            return value, None
        if key == "role":
            return "", value
    return selector, None


def _find_element_bounds_in_tree(desktop, window: str, target_name: str, target_role: str | None) -> dict | None:
    """Find an element's bounds by searching the window's UI tree (PID-scoped, exact name match)."""
    try:
        # Find PID
        pid = None
        for app in desktop.applications():
            try:
                if window.lower() in app.name().lower():
                    pid = app.process_id()
                    break
            except Exception:
                continue
        if not pid:
            # Fallback: pygetwindow
            try:
                import pygetwindow as gw
                import ctypes
                for w in gw.getAllWindows():
                    if window.lower() in w.title.lower() and w.title.strip():
                        hwnd = w._hWnd
                        p = ctypes.c_ulong()
                        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
                        pid = p.value
                        break
            except Exception:
                pass
        if not pid:
            logger.debug(f"[APP] Tree search: no PID for window '{window}'")
            return None

        tree = desktop.get_window_tree(pid)

        def _extract_bounds(b):
            bx = b.get('x') if isinstance(b, dict) else getattr(b, 'x', None)
            if bx is None:
                return None
            return dict(b) if isinstance(b, dict) else {
                'x': b.x, 'y': b.y, 'width': b.width, 'height': b.height
            }

        # Skip substring matches inside browser/app chrome (toolbar, menubar,
        # tab strip). Otherwise "search" matches Chrome's omnibox
        # ("Address and search bar") before reaching the page content.
        _CHROME_CONTAINER_ROLES = {
            "toolbar", "menubar", "tabcontrol", "titlebar", "statusbar",
        }

        def search(node, depth=0, max_depth=20, exact=True, in_chrome=False):
            try:
                attrs = node.attributes
                name = (attrs.name or "").strip()
                role = (attrs.role or "").strip()
                role_l = role.lower()

                is_chrome_container = role_l in _CHROME_CONTAINER_ROLES
                current_chrome = in_chrome or is_chrome_container

                if name and target_name:
                    name_l = name.lower()
                    target_l = target_name.lower()
                    if exact:
                        matched = (name_l == target_l)
                    else:
                        matched = (
                            target_l in name_l
                            and role_l not in ("window", "")
                            and not current_chrome
                        )
                    if matched and (target_role is None
                                    or role_l == (target_role or "").lower()):
                        b = attrs.bounds
                        if b:
                            result = _extract_bounds(b)
                            if result:
                                logger.info(f"[APP] Tree found: name={name!r}, role={role!r}, bounds={b}")
                                return result
                if depth < max_depth and node.children:
                    for child in node.children:
                        result = search(child, depth + 1, max_depth, exact, current_chrome)
                        if result:
                            return result
            except Exception:
                pass
            return None

        return search(tree, exact=True) or search(tree, exact=False)
    except Exception as e:
        logger.debug(f"[APP] Tree search failed: {e}")
        return None

def _sync_type_text(text: str, selector: str = None, window: str = None) -> str:
    if _backend == "pywinauto":
        try:
            if selector or window:
                elem = _get_pywinauto_element(selector, window)
                elem.wait("visible", timeout=10)
                elem.set_focus()
                elem.type_keys(text, with_spaces=True)
            else:
                pywinauto.keyboard.send_keys(text, with_spaces=True)
            return f"Typed text into {selector or 'focus'}"
        except Exception as e:
            return f"Failed to type text: {e}"
    raise RuntimeError("No backend available")

async def type_text(text: str, selector: str = None, window: str = None) -> str:
    """
    Type text into a UI element or the focused field.
    """
    if not is_available(): return "Native automation not available."
    logger.info(f"[APP] Typing text: {text} into {selector or 'focus'}")
    try:
        if _backend == "terminator":
            desktop = ensure_desktop()
            if selector:
                loc = desktop.locator(f"name:{window}" if window and ":" not in window else window) if window else desktop
                elem = await loc.locator(selector).first()
                elem.type_text(text)
            else:
                # No selector — type into whatever is focused via keyboard input.
                # Direct keyboard typing (pyautogui.write) works for ANY app:
                # Calculator, search bars, dialogs, text editors, etc.
                # Clipboard paste (Ctrl+V) is fallback for non-ASCII text.
                try:
                    import pyautogui
                    # P3: Calculator doesn't support clipboard paste — use key presses per character.
                    # Detect by active window title so this is generic (not Calculator-specific code).
                    _is_calc = False
                    try:
                        import pygetwindow as gw
                        _aw = gw.getActiveWindow()
                        _is_calc = _aw is not None and "calculator" in (_aw.title or "").lower()
                    except Exception:
                        pass
                    with _suppress_hotkey_hook():
                        if _is_calc:
                            _CALC_OPS = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}
                            for ch in text:
                                pyautogui.press(_CALC_OPS.get(ch, ch))
                        elif text.isascii():
                            pyautogui.write(text, interval=0.03)
                        else:
                            import pyperclip
                            pyperclip.copy(text)
                            pyautogui.hotkey("ctrl", "v")
                        await asyncio.sleep(0.2)
                except ImportError:
                    logger.warning("[APP] pyautogui not available for keyboard input")
                    return "Error: cannot type without a target selector (pyautogui not available)"
            return f"Typed text into {selector or 'focus'}"
        return await _to_thread(_sync_type_text, text, selector, window)
    except Exception as e:
        logger.error(f"[APP] Type text error: {e}")
        return f"Error typing text: {e}"

def _sync_get_text(selector: str, window: str = None) -> str:
    if _backend == "pywinauto":
        try:
            elem = _get_pywinauto_element(selector, window)
            elem.wait("visible", timeout=10)
            return elem.window_text() or ""
        except Exception as e:
            return f"Error getting text: {e}"
    raise RuntimeError("No backend available")

async def get_text(selector: str, window: str = None) -> str:
    """
    Get text content of a UI element.
    """
    if not is_available(): return ""
    try:
        if _backend == "terminator":
            desktop = ensure_desktop()

            # Terminator only supports name: and role: selectors via locator.
            # automationid: selectors don't work — skip to tree search for those.
            is_automationid = selector.lower().startswith("automationid:")

            if not is_automationid:
                # Approach 1: Direct locator → wait for visibility → .first() → .text()
                try:
                    loc1 = desktop.locator(selector)
                    try:
                        await loc1.wait(timeout=2000)
                    except Exception:
                        pass  # proceed even if wait times out — .first() may still succeed
                    element = await loc1.first()
                    text = element.text(max_depth=3)
                    if text:
                        return text
                except Exception as e1:
                    logger.debug(f"[APP] get_text approach 1 (direct) failed: {e1}")

                # Approach 2: Window-scoped using .within()
                if window:
                    try:
                        win_selector = f"name:{window}" if ":" not in window else window
                        win_loc = desktop.locator(win_selector)
                        try:
                            await win_loc.wait(timeout=2000)
                        except Exception:
                            pass
                        win_elem = await win_loc.first()
                        element = await desktop.locator(selector).within(win_elem).first()
                        text = element.text(max_depth=3)
                        if text:
                            return text
                    except Exception as e2:
                        logger.debug(f"[APP] get_text approach 2 (window-scoped) failed: {e2}")

            # Approach 2b: For automationid selectors or failed name selectors,
            # search the UI tree by walking children of the window element
            if window:
                try:
                    apps = desktop.applications()
                    target_pid = None
                    for app in apps:
                        try:
                            if window.lower() in app.name().lower():
                                target_pid = app.process_id()
                                break
                        except Exception:
                            continue
                    if target_pid:
                        tree = desktop.get_window_tree(target_pid)
                        if hasattr(tree, '__await__'):
                            tree = await tree
                        # Search tree for matching element
                        target_name = selector.split(":", 1)[1] if ":" in selector else selector
                        found_text = _search_tree_for_text(tree, target_name)
                        if found_text:
                            return found_text
                except Exception as e2b:
                    logger.debug(f"[APP] get_text approach 2b (tree search) failed: {e2b}")

            # Approach 3: Read from window title as last resort
            # Many apps show key info in the title (Calculator shows "Display is 15")
            if window:
                try:
                    import pygetwindow as gw
                    all_windows = gw.getAllWindows()
                    for w in all_windows:
                        if window.lower() in w.title.lower():
                            return w.title
                except Exception as e3:
                    logger.debug(f"[APP] get_text approach 3 (window title) failed: {e3}")

            return ""
        return await _to_thread(_sync_get_text, selector, window)
    except Exception as e:
        logger.error(f"[APP] Get text error: {e}")
        return f"Error: {e}"

def _sync_wait_for_element(selector: str, timeout: float) -> bool:
    if _backend == "pywinauto":
        try:
            elem = _get_pywinauto_element(selector)
            elem.wait("visible", timeout=timeout)
            return True
        except Exception:
            return False
    return False

async def wait_for_element(selector: str, timeout: float = 10.0) -> bool:
    """
    Wait for a UI element to become visible.
    """
    if not is_available(): return False
    try:
        if _backend == "terminator":
            desktop = ensure_desktop()
            try:
                await desktop.locator(selector).wait(timeout=timeout*1000)
                return True
            except Exception:
                return False
        return await _to_thread(_sync_wait_for_element, selector, timeout)
    except Exception:
        return False

def _sync_list_elements(window: str = None) -> str:
    if _backend == "pywinauto":
        try:
            if window:
                elem = _get_pywinauto_element(selector=None, window=window)
                import io, sys
                old_stdout = sys.stdout
                sys.stdout = my_stdout = io.StringIO()
                elem.print_control_identifiers(depth=2)
                sys.stdout = old_stdout
                return my_stdout.getvalue()
            else:
                return "Specify a window to see elements."
        except Exception as e:
            return f"Error listing elements: {e}"
    raise RuntimeError("No backend available")

def _search_tree_for_text(node, target: str, max_depth: int = 6, depth: int = 0) -> str:
    """Search the UINode tree for an element whose name contains the target string,
    or for 'Result'/'Display' patterns common in output-display elements."""
    try:
        attrs = node.attributes
        name = attrs.name or ""
        role = attrs.role or ""
        if name and name != "None":
            name_lower = name.lower()
            target_lower = target.lower()
            # Skip broad matches on window/app title nodes — too greedy
            if role in ("Window", "Pane"):
                pass  # Don't match on containers, recurse deeper
            elif target_lower == name_lower:
                # Exact match
                return name
            elif target_lower in name_lower:
                # Target is a substring of name (e.g., "Display" in "Display is 35")
                return name
            # Heuristic: if target looks like an automation ID (e.g., "CalculatorResults"),
            # match on common result/display patterns in the name
            if "result" in target_lower or "display" in target_lower:
                if ("display is" in name_lower or "result" in name_lower) and role not in ("Window", "Pane"):
                    return name
        if depth < max_depth:
            children = node.children
            if children:
                for child in children:
                    result = _search_tree_for_text(child, target, max_depth, depth + 1)
                    if result:
                        return result
    except Exception:
        pass
    return ""


def _format_ui_tree(node, depth: int = 0, max_depth: int = 3) -> str:
    """Recursively format a UINode tree into LLM-readable text with selectors."""
    lines = []
    try:
        attrs = node.attributes
        name = attrs.name or ""
        role = attrs.role or ""
        indent = "  " * depth

        # Skip container nodes — recurse into children at same depth
        # to keep output compact while reaching deeply nested elements
        is_skip_container = role in ("Pane", "Group", "Custom", "Window", "")
        if is_skip_container:
            if depth < max_depth:
                children = node.children
                if children:
                    for child in children:
                        child_text = _format_ui_tree(child, depth, max_depth)
                        if child_text.strip():
                            lines.append(child_text)
            return "\n".join(lines)

        # Build selector hints for the LLM — use name: which Terminator supports
        parts = []
        if role:
            parts.append(role)
        if name and name != "None":
            parts.append(f'name:{name}')

        label = f"{indent}[{', '.join(parts)}]" if parts else None
        if label:
            lines.append(label)

        if depth < max_depth:
            children = node.children
            if children:
                for child in children:
                    lines.append(_format_ui_tree(child, depth + 1, max_depth))
    except Exception as e:
        lines.append(f"{'  ' * depth}[error: {e}]")
    return "\n".join(lines)


async def list_elements(window: str = None) -> str:
    """
    List accessible UI elements in a window.
    """
    if not is_available(): return "Native automation not available."
    try:
        if _backend == "terminator":
            desktop = ensure_desktop()
            if not window:
                return "Specify a window to see elements."

            # Find PID by matching window name against running applications
            apps = desktop.applications()
            target_pid = None
            for app in apps:
                try:
                    app_name = app.name()
                    if app_name and window.lower() in app_name.lower():
                        target_pid = app.process_id()
                        break
                except Exception:
                    continue

            if not target_pid:
                # Fallback: try pygetwindow to find PID
                try:
                    import pygetwindow as gw
                    import ctypes
                    all_windows = gw.getAllWindows()
                    for w in all_windows:
                        if window.lower() in w.title.lower() and w.title.strip():
                            hwnd = w._hWnd
                            pid = ctypes.c_ulong()
                            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                            target_pid = pid.value
                            break
                except Exception:
                    pass

            if not target_pid:
                return f"Could not find window '{window}' among running applications."

            tree = desktop.get_window_tree(target_pid)
            if hasattr(tree, '__await__'):
                tree = await tree
            return _format_ui_tree(tree, max_depth=2)
        return await _to_thread(_sync_list_elements, window)
    except Exception as e:
        logger.error(f"[APP] list_elements error: {e}")
        return f"Error: {e}"

def _sync_focus_window(name: str) -> str:
    if _backend == "pywinauto":
        try:
            elem = _get_pywinauto_element(selector=None, window=name)
            elem.set_focus()
            return f"Focused window: {name}"
        except Exception as e:
            return f"Error focusing window: {e}"
    raise RuntimeError("No backend available")

async def focus_window(name: str) -> str:
    """
    Bring a window to the foreground.
    Primary: Win32 API (most reliable). Fallback: Terminator locator / pywinauto.
    """
    if not is_available(): return "Native automation not available."
    # Primary: Win32 API — most reliable for bringing windows to foreground
    try:
        import pygetwindow as gw
        import ctypes

        all_windows = gw.getAllWindows()
        matches = [w for w in all_windows
                   if name.lower() in w.title.lower() and w.title.strip()]
        if matches:
            hwnd = matches[0]._hWnd
            user32 = ctypes.windll.user32
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            await asyncio.sleep(0.3)
            logger.info(f"[APP] Focused window via Win32: {matches[0].title}")
            return f"Focused window: {matches[0].title}"
    except Exception as e1:
        logger.debug(f"[APP] Win32 focus failed: {e1}")

    # Fallback: Terminator / pywinauto
    try:
        if _backend == "terminator":
            desktop = ensure_desktop()
            elem = await desktop.locator(f"name:{name}" if ":" not in name else name).timeout(8000).first()
            elem.click()
            return f"Focused window: {name}"
        return await _to_thread(_sync_focus_window, name)
    except Exception as e:
        return f"Error: {e}"

def _sync_close_app(name: str) -> str:
    if _backend == "pywinauto":
        try:
            elem = _get_pywinauto_element(selector=None, window=name)
            elem.close()
            return f"Closed {name}"
        except Exception as e:
            return f"Error closing {name}: {e}"
    raise RuntimeError("No backend available")

async def close_app(name: str) -> str:
    """Close an application window."""
    if not is_available(): return "Native automation not available."
    try:
        if _backend == "terminator":
            # Use keyboard shortcut to close — more reliable than element.close()
            desktop = ensure_desktop()
            elem = await desktop.locator(f"name:{name}" if ":" not in name else name).timeout(8000).first()
            elem.click()  # focus first
            import pyautogui
            with _suppress_hotkey_hook():
                pyautogui.hotkey("alt", "F4")
            return f"Closed {name}"
        return await _to_thread(_sync_close_app, name)
    except Exception as e:
        return f"Error: {e}"

async def run_app_steps(steps: List[Dict]) -> str:
    """
    Execute a sequence of native app actions.
    Checks for ESC abort before each step (hold ESC ~1s to cancel).

    Each state-changing step is wrapped in pre_check → execute →
    post_verify. Confident failures short-circuit with a structured
    VERIFY_FAILED prefix the planner can parse.
    """
    if not is_available(): return "Native automation not available."

    from . import verification

    # ESC monitor lifecycle is owned by main.py (session-level singleton).
    # run_app_steps must NOT start/stop/reset the shared daemon.
    # _check_abort() is still called between steps to respect an in-flight abort.
    try:
        from . import vision as _ca
    except Exception:
        _ca = None

    # Track the most recent "context window" so verification can scope its
    # readbacks correctly when a step doesn't carry an explicit window param.
    active_window: Optional[str] = None

    results = []
    try:
        for i, step in enumerate(steps):
            # ESC-abort check — fires between steps (within step: depends on timeout).
            # raise UserAborted so the planner doesn't see this as a regular
            # step failure (which would trigger recovery and require a second ESC).
            if _ca and _ca._check_abort():
                from assistant.core.abort import UserAborted
                raise UserAborted("esc_hold")
            action = step.get("action")
            params = step.get("params", {})
            verify_step = {"type": "app", "action": action, "params": params}
            logger.info(f"[APP] Step {i+1}: {action} - {params}")

            # ── Tier 0: pre-check (target reachable, focus context sane) ──
            pre = await verification.pre_check(verify_step, active_window=active_window)
            if not pre.ok and pre.confidence >= config.VERIFY_MIN_CONFIDENCE:
                msg = f"verify_failed (pre): step {i+1} {action} — {pre.observation}"
                logger.warning(f"[APP] {msg}")
                results.append(msg)
                return f"VERIFY_FAILED|step={i+1}|tier=pre_check|obs={pre.observation}\n" + "\n".join(results)
            
            if action == "open":
                res = await open_app(params.get("name"))
                results.append(res)
            elif action == "focus":
                res = await focus_window(params.get("name"))
                results.append(res)
            elif action == "click":
                res = await click_element(params.get("selector"), params.get("window"))
                results.append(res)
            elif action == "type":
                # Re-focus target window before keyboard typing (no selector).
                # Prevents keystrokes going to wrong window — focus can be lost
                # between steps (TTS popup, notification, Unity window).
                window = params.get("window")
                if not params.get("selector") and window:
                    await focus_window(window)
                    await asyncio.sleep(0.2)
                res = await type_text(params.get("text"), params.get("selector"), window)
                results.append(res)
            elif action == "get_text":
                try:
                    res = await asyncio.wait_for(
                        get_text(params.get("selector"), params.get("window")),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[APP] get_text timed out after 15s for: {params.get('selector')}")
                    res = ""
                results.append(f"Extracted: {res}")
            elif action == "wait":
                selector_val = params.get("selector")
                seconds_val = params.get("seconds") or params.get("timeout")
                if selector_val:
                    found = await wait_for_element(selector_val, float(params.get("timeout", 10.0)))
                    results.append(f"Wait found={found} for {selector_val}")
                elif seconds_val:
                    await asyncio.sleep(float(seconds_val))
                    results.append(f"Waited {seconds_val}s")
                else:
                    await asyncio.sleep(1.0)
                    results.append("Waited 1s (default)")
            elif action == "list":
                res = await list_elements(params.get("window"))
                results.append(f"Elements: {res}")
            elif action == "close":
                res = await close_app(params.get("name"))
                results.append(res)
            elif action == "press_key":
                # Press a keyboard key or combo (Enter, ctrl+n, alt+f4, etc.)
                key = params.get("key", "")
                try:
                    import pyautogui
                    parts = [k.strip().lower() for k in key.split('+')]
                    with _suppress_hotkey_hook():
                        if len(parts) > 1:
                            # Key combination: ctrl+n → pyautogui.hotkey('ctrl', 'n')
                            pyautogui.hotkey(*parts)
                        else:
                            # Single key: enter, tab, escape, etc.
                            pyautogui.press(parts[0])
                    results.append(f"Pressed key: {key}")
                except Exception as ke:
                    results.append(f"Error pressing key {key}: {ke}")
            else:
                results.append(f"Unknown action: {action}")

            # Track active_window context for downstream pre-checks.
            if action in ("open", "focus") and params.get("name"):
                active_window = params["name"]
            elif params.get("window"):
                active_window = params["window"]

            # ── Tier 1: post-verify ──
            post = await verification.post_verify(verify_step, active_window=active_window)
            if post.tier == "ambiguous" and config.VERIFY_VISION_FALLBACK:
                # Vision escalation. Currently a no-op pass-through.
                post = await verification.vision_verify(verify_step, post, active_window=active_window)
            if not post.ok and post.confidence >= config.VERIFY_MIN_CONFIDENCE and not post.skipped:
                msg = f"verify_failed (post): step {i+1} {action} — {post.observation}"
                logger.warning(f"[APP] {msg}")
                results.append(msg)
                return f"VERIFY_FAILED|step={i+1}|tier={post.tier}|obs={post.observation}\n" + "\n".join(results)

        return "\n".join(results)
    except Exception as e:
        # never swallow a user-initiated abort into a string error.
        # Re-raise so the planner sees UserAborted and stops cleanly.
        from assistant.core.abort import UserAborted
        if isinstance(e, UserAborted):
            raise
        return f"Error running steps: {e}\nCompleted: " + "\n".join(results)