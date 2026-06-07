"""manifest_runtime.py — singleton holder for the manifest dispatcher.

main.py initializes this once at startup after manifest_registry is up.
Action handlers (and any other callers) reach it via get_dispatcher().

Also provides a thin _TerminatorAdapter that wraps the live
assistant.automation.native desktop. Exposes the surfaces both
manifest_primitives.execute_primitive (send_key / find_element / click)
and healer (enumerate_descendants / screenshot / element_at_point)
need. Dispatcher surface (send_key / find_element / click) is live-wired to
native.py + pyautogui. Healer surface stays as
NotImplementedError stubs in v1; healer's try/except gracefully
degrades to HealResult(ok=False, tier=...) when those raise, and the
dispatcher then falls through to computer_task.

Tests inject FakeTerminator and never touch the adapter, so unit-level
coverage is unaffected.
"""
from __future__ import annotations

from typing import Any

from .manifest_dispatcher import ManifestDispatcher

_dispatcher: ManifestDispatcher | None = None


# ─── Singleton helpers ──────────────────────────────────────────────────
def init_dispatcher(dispatcher: ManifestDispatcher) -> None:
    """Cache the singleton dispatcher. Called once from main.py at startup."""
    global _dispatcher
    _dispatcher = dispatcher


def get_dispatcher() -> ManifestDispatcher | None:
    """Return the cached dispatcher, or None if not yet initialized."""
    return _dispatcher


def reset_for_test() -> None:
    """Clear the cached dispatcher. Test-only helper."""
    global _dispatcher
    _dispatcher = None


# ─── UIA tree walkers (Terminator-only) ────────────────────────────────


def _resolve_window_pid(desktop: Any, window: str) -> int | None:
    """Find the PID for the named window. Mirrors native._find_element_bounds_in_tree.

    Returns None if no PID found — caller treats this as no-match.
    """
    if not window:
        return None
    try:
        for app in desktop.applications():
            try:
                if window.lower() in app.name().lower():
                    return app.process_id()
            except Exception:
                continue
    except Exception:
        pass
    # Fallback: pygetwindow + win32 GetWindowThreadProcessId
    try:
        import ctypes
        import pygetwindow as gw
        for w in gw.getAllWindows():
            if window.lower() in (w.title or "").lower() and (w.title or "").strip():
                hwnd = w._hWnd
                p = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
                return p.value
    except Exception:
        pass
    return None


def _extract_bounds(b: Any) -> dict | None:
    """Coerce a Terminator bounds object (or dict) to a plain dict, or None."""
    if b is None:
        return None
    if isinstance(b, dict):
        return dict(b) if b.get("x") is not None else None
    x = getattr(b, "x", None)
    if x is None:
        return None
    return {"x": b.x, "y": b.y, "width": b.width, "height": b.height}


def _find_node_by_selector(
    desktop: Any,
    window: str,
    automation_id: str,
    name: str,
    control_type: str,
) -> dict | None:
    """Walk the window's UI tree looking for a node matching the selector.

    Matching rules (per UIA convention — apps may expose one or both):
      - If ``automation_id`` is non-empty: match nodes whose automation_id
        equals it. This is the most stable identifier; prefer it when present.
      - If ``automation_id`` is empty AND ``name`` is non-empty: match nodes
        whose accessible name equals ``name`` (case-sensitive — the value in
        the manifest was captured verbatim from the live tree).
      - ``control_type`` further constrains either match when non-empty.
      - If both ``automation_id`` and ``name`` are empty, return None — the
        caller (_exec_uia) is responsible for rejecting empty selectors.

    Returns ``{"automation_id", "control_type", "name", "bounds"}`` on hit,
    or None on miss. Defensive try/except inside the walker so a malformed AT
    subtree doesn't kill the whole search.
    """
    if not automation_id and not name:
        return None
    pid = _resolve_window_pid(desktop, window)
    if pid is None:
        return None
    try:
        tree = desktop.get_window_tree(pid)
    except Exception:
        return None

    target_role_l = (control_type or "").lower()
    target_aid = automation_id or ""
    target_name = name or ""
    use_aid = bool(target_aid)

    def walk(node: Any, depth: int = 0, max_depth: int = 20) -> dict | None:
        try:
            attrs = node.attributes
            aid = getattr(attrs, "automation_id", None) or ""
            node_name = getattr(attrs, "name", None) or ""
            matched = (aid == target_aid) if use_aid else (node_name == target_name)
            if matched:
                role_l = (getattr(attrs, "role", "") or "").lower()
                if not target_role_l or role_l == target_role_l:
                    bounds = _extract_bounds(getattr(attrs, "bounds", None))
                    if bounds is not None:
                        return {
                            "automation_id": aid,
                            "control_type": getattr(attrs, "role", "") or "",
                            "name": node_name,
                            "bounds": bounds,
                        }
            if depth < max_depth:
                children = getattr(node, "children", None) or []
                for child in children:
                    hit = walk(child, depth + 1, max_depth)
                    if hit is not None:
                        return hit
        except Exception:
            pass
        return None

    return walk(tree)


# ─── Live adapter ───────────────────────────────────────────
class _TerminatorAdapter:
    """Adapts assistant.automation.native's desktop to the surfaces both
    manifest_primitives.execute_primitive (send_key / find_element / click)
    and healer (enumerate_descendants / screenshot / element_at_point)
    need.

    Dispatcher surface (send_key / find_element / click) is live-wired to
    native.py + pyautogui. Healer surface stays as
    NotImplementedError stubs in v1; healer's try/except gracefully
    degrades to HealResult(ok=False, tier=...) when those raise, and the
    dispatcher then falls through to computer_task.

    Terminator backend only — pywinauto backend raises NotImplementedError
    for find_element (no get_window_tree API). Tests inject FakeTerminator
    and never touch this adapter.
    """

    def __init__(self, native_desktop: Any) -> None:
        self._desktop = native_desktop

    def send_key(self, key: str) -> None:
        """Press a key or hotkey combination (e.g. 'enter', 'ctrl+s', 'alt+f4').

        Mirrors native.run_app_steps's press_key action. Wrapped in
        _suppress_hotkey_hook so the project's push-to-talk listener
        ignores synthesized events.

        Note: '+' is the separator, so the literal plus key must be spelled
        as 'plus' (or 'add' per pyautogui). A key like 'shift++' parses as
        ['shift'] only — TODO(healer-wire): key-name escaping table.
        """
        parts = [k.strip().lower() for k in (key or "").split("+") if k.strip()]
        if not parts:
            raise ValueError(f"send_key got empty key string: {key!r}")
        from . import native as _native
        import pyautogui

        with _native._suppress_hotkey_hook():
            if len(parts) > 1:
                pyautogui.hotkey(*parts)
            else:
                pyautogui.press(parts[0])

    def find_element(
        self,
        *,
        automation_id: str = "",
        name: str = "",
        control_type: str = "",
        window: str = "",
    ) -> Any:
        """Locate a UIA element under the named window.

        Identification: prefer ``automation_id`` when non-empty; otherwise
        match by ``name`` (the accessible name from the UIA tree). At least
        one must be non-empty — caller (_exec_uia) enforces this.

        Returns the element dict ({"automation_id", "control_type", "name",
        "bounds"}) on hit, or raises LookupError if not found. The dispatcher
        treats LookupError as a selector miss and escalates to the healer.
        """
        from . import native as _native
        if _native._backend != "terminator":
            raise NotImplementedError(
                "_TerminatorAdapter.find_element — pywinauto backend not supported in v1"
            )
        elem = _find_node_by_selector(
            self._desktop,
            window=window,
            automation_id=automation_id,
            name=name,
            control_type=control_type,
        )
        if elem is None:
            raise LookupError(
                f"NoSuchElement: automation_id={automation_id!r} "
                f"name={name!r} control_type={control_type!r} "
                f"in window={window!r}"
            )
        return elem

    def click(self, element: Any) -> None:
        """Click the center of the element's bounds via pyautogui.

        Wrapped in _suppress_hotkey_hook so the project's push-to-talk
        listener ignores synthesized mouse events.
        """
        from . import native as _native
        import pyautogui

        if not isinstance(element, dict):
            raise TypeError(
                f"_TerminatorAdapter.click expects a dict; got {type(element).__name__}"
            )
        bounds = element.get("bounds")
        if not isinstance(bounds, dict):
            raise ValueError(f"element missing 'bounds' dict: {element!r}")
        try:
            cx = bounds["x"] + bounds["width"] // 2
            cy = bounds["y"] + bounds["height"] // 2
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed bounds {bounds!r}: {e}") from e
        with _native._suppress_hotkey_hook():
            pyautogui.click(cx, cy)

    def enumerate_descendants(
        self,
        *,
        parent_window: str,
        max_depth: int = 4,
    ) -> list[dict]:
        """Return every node under ``parent_window`` up to ``max_depth``.

        Each entry: ``{"automation_id", "control_type", "name",
        "parent_chain", "sibling_count"}``. The healer tier-1 scorer
        (assistant.automation.at_fingerprint) consumes this directly.

        ``parent_chain`` is the list of the ancestor nodes' names from the
        window root down to (but not including) the node itself, mirroring
        what the promoter stores in Selector.parent_chain. Used by the
        fingerprint scorer for the chain-similarity sub-score.

        Returns [] on any infrastructure failure (no PID, tree fetch raise,
        empty tree). The healer's try/except already degrades cleanly on
        this path — keep the contract honest by NEVER raising.
        """
        from . import native as _native
        if _native._backend != "terminator":
            return []
        pid = _resolve_window_pid(self._desktop, parent_window)
        if pid is None:
            return []
        try:
            tree = self._desktop.get_window_tree(pid)
        except Exception:
            return []
        if tree is None:
            return []

        out: list[dict] = []

        def walk(node: Any, chain: tuple[str, ...], depth: int) -> None:
            if depth > max_depth:
                return
            try:
                children = list(getattr(node, "children", None) or [])
            except Exception:
                children = []
            sibling_count = len(children)
            for child in children:
                try:
                    attrs = child.attributes
                    aid = getattr(attrs, "automation_id", None) or ""
                    cname = getattr(attrs, "name", None) or ""
                    role = getattr(attrs, "role", None) or ""
                    out.append({
                        "automation_id": aid,
                        "control_type": role,
                        "name": cname,
                        "parent_chain": list(chain),
                        "sibling_count": sibling_count,
                    })
                    next_chain = chain + ((cname or aid or role),)
                    walk(child, next_chain, depth + 1)
                except Exception:
                    # Defensive: a malformed subtree mustn't kill the walk.
                    continue

        try:
            root_attrs = tree.attributes
            root_name = (getattr(root_attrs, "name", None) or "")
        except Exception:
            root_name = ""
        walk(tree, (root_name,) if root_name else (), 0)
        return out

    def screenshot(self) -> bytes:
        """Capture the full primary monitor and return PNG bytes.

        Wraps assistant.io.screen.capture_screenshot (PIL Image) and dumps
        to PNG via BytesIO. Raises on failure so the healer's try/except
        path treats it as a tier-2 infra failure (HealResult(ok=False,
        tier=2, error=...)) and falls through to computer_task cleanly.
        """
        from ..io import screen
        import io as _io

        img = screen.capture_screenshot()
        if img is None:
            raise RuntimeError("screen.capture_screenshot returned None")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def element_at_point(self, x: int, y: int) -> dict | None:
        """Walk the foreground window's AT tree and return the DEEPEST node
        whose bounds contain (x, y), or None if no match.

        Used by healer tier-2 AFTER a vision-grounded pyautogui.click to
        re-resolve a stable automation_id + parent_chain for the freshly
        clicked control. None is acceptable; the caller's try/except logs
        ``tier-2 AT re-resolve skipped`` and proceeds with vision-only
        success (the click already landed — the AT re-resolve is opportunistic
        polish, not load-bearing).

        Native point-hit-test (IUIAutomation::ElementFromPoint via comtypes)
        is the canonical implementation but pulls in a heavy Win32 COM
        dependency for a v1 polish step. The tree-walk-by-bounds approach
        below is O(N) in window-tree size — for the ~100-200 nodes a
        typical app exposes this is sub-millisecond and avoids any new
        dependency.
        """
        from . import native as _native
        from ..io import screen
        if _native._backend != "terminator":
            return None
        try:
            active = screen.get_active_window()
        except Exception:
            return None
        if not active:
            return None
        pid = _resolve_window_pid(self._desktop, active)
        if pid is None:
            return None
        try:
            tree = self._desktop.get_window_tree(pid)
        except Exception:
            return None
        if tree is None:
            return None

        deepest: dict | None = None
        deepest_depth = -1

        def walk(node: Any, parent_chain: tuple[str, ...], depth: int) -> None:
            """parent_chain = path from root to THIS node's PARENT, exclusive
            of THIS node's own name. The match's parent_chain is therefore
            the ancestors of the matched node, not the matched node itself.
            """
            nonlocal deepest, deepest_depth
            try:
                attrs = node.attributes
                bounds = _extract_bounds(getattr(attrs, "bounds", None))
                if bounds is not None and _bounds_contain(bounds, x, y):
                    if depth > deepest_depth:
                        deepest = {
                            "automation_id": getattr(attrs, "automation_id", None) or "",
                            "control_type": getattr(attrs, "role", None) or "",
                            "name": getattr(attrs, "name", None) or "",
                            "parent_chain": list(parent_chain),
                            "bounds": bounds,
                        }
                        deepest_depth = depth
                # Children's parent_chain = our parent_chain + OUR name.
                this_name = (getattr(attrs, "name", None) or
                             getattr(attrs, "automation_id", None) or
                             getattr(attrs, "role", None) or "")
                child_chain = parent_chain + (this_name,) if this_name else parent_chain
                children = list(getattr(node, "children", None) or [])
                for child in children:
                    walk(child, child_chain, depth + 1)
            except Exception:
                return

        walk(tree, (), 0)
        return deepest


def _bounds_contain(bounds: dict, x: int, y: int) -> bool:
    """True iff (x, y) lies within bounds = {x, y, width, height}."""
    try:
        bx = bounds["x"]
        by = bounds["y"]
        return (bx <= x < bx + bounds["width"]
                and by <= y < by + bounds["height"])
    except (KeyError, TypeError):
        return False
