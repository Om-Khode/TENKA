"""Tests for manifest_runtime.py — dispatcher singleton + _TerminatorAdapter.

Covers:
  1. init_dispatcher / get_dispatcher round-trip.
  2. reset_for_test clears the cached dispatcher.
  3. Dispatcher-surface methods (send_key / find_element / click) are
     live-wired — exercise them with monkeypatched pyautogui + tree walker.
  4. Healer-surface stubs (enumerate_descendants / screenshot /
     element_at_point) still raise NotImplementedError in v1.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import assistant.automation.manifest_runtime as manifest_runtime


def _reset():
    manifest_runtime.reset_for_test()


# ─── Singleton helpers ──────────────────────────────────────────────────

def test_init_and_get_dispatcher():
    _reset()
    fake = MagicMock()
    manifest_runtime.init_dispatcher(fake)
    assert manifest_runtime.get_dispatcher() is fake
    _reset()


def test_get_dispatcher_returns_none_before_init():
    _reset()
    assert manifest_runtime.get_dispatcher() is None


def test_reset_for_test_clears_dispatcher():
    fake = MagicMock()
    manifest_runtime.init_dispatcher(fake)
    manifest_runtime.reset_for_test()
    assert manifest_runtime.get_dispatcher() is None


# ─── Dispatcher surface: send_key ──────────────────────────────────────

def test_send_key_single_key_routes_through_pyautogui_press(monkeypatch):
    """send_key('enter') calls pyautogui.press('enter') under the suppress hook."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    calls = []
    monkeypatch.setattr("pyautogui.press", lambda k: calls.append(("press", k)), raising=False)
    monkeypatch.setattr("pyautogui.hotkey", lambda *p: calls.append(("hotkey", p)), raising=False)

    adapter = _TerminatorAdapter(native_desktop=object())
    adapter.send_key("enter")
    assert calls == [("press", "enter")]


def test_send_key_combo_routes_through_pyautogui_hotkey(monkeypatch):
    """send_key('ctrl+s') calls pyautogui.hotkey('ctrl', 's')."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    calls = []
    monkeypatch.setattr("pyautogui.press", lambda k: calls.append(("press", k)), raising=False)
    monkeypatch.setattr("pyautogui.hotkey", lambda *p: calls.append(("hotkey", p)), raising=False)

    adapter = _TerminatorAdapter(native_desktop=object())
    adapter.send_key("ctrl+s")
    assert calls == [("hotkey", ("ctrl", "s"))]


def test_send_key_empty_raises():
    """send_key('') raises ValueError — empty key string is a caller bug."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    adapter = _TerminatorAdapter(native_desktop=object())
    with pytest.raises(ValueError):
        adapter.send_key("")


# ─── Dispatcher surface: click ─────────────────────────────────────────

def test_click_calls_pyautogui_at_bounds_center(monkeypatch):
    """click() computes center of bounds and calls pyautogui.click(cx, cy)."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    calls = []
    monkeypatch.setattr("pyautogui.click", lambda x, y: calls.append((x, y)), raising=False)

    adapter = _TerminatorAdapter(native_desktop=object())
    adapter.click({"bounds": {"x": 100, "y": 50, "width": 40, "height": 20}})
    assert calls == [(120, 60)]


def test_click_rejects_non_dict():
    """click() with a non-dict element raises TypeError."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    adapter = _TerminatorAdapter(native_desktop=object())
    with pytest.raises(TypeError):
        adapter.click("not a dict")


def test_click_rejects_missing_bounds():
    """click() with an element missing 'bounds' raises ValueError."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    adapter = _TerminatorAdapter(native_desktop=object())
    with pytest.raises(ValueError):
        adapter.click({"automation_id": "x"})


# ─── Dispatcher surface: find_element ──────────────────────────────────

def test_find_element_pywinauto_backend_raises_notimplemented(monkeypatch):
    """pywinauto backend has no get_window_tree — find_element raises NotImplementedError."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native

    monkeypatch.setattr(_native, "_backend", "pywinauto")

    adapter = _TerminatorAdapter(native_desktop=object())
    with pytest.raises(NotImplementedError):
        adapter.find_element(automation_id="x", control_type="Button", window="any")


def test_find_element_returns_dict_on_hit(monkeypatch):
    """find_element returns the node dict when the walker finds a matching automation_id."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native
    import assistant.automation.manifest_runtime as _me1

    monkeypatch.setattr(_native, "_backend", "terminator")
    expected = {
        "automation_id": "play-pause-button",
        "control_type": "Button",
        "name": "Play",
        "bounds": {"x": 100, "y": 100, "width": 40, "height": 40},
    }
    monkeypatch.setattr(_me1, "_find_node_by_selector", lambda *a, **k: expected)

    adapter = _TerminatorAdapter(native_desktop=object())
    result = adapter.find_element(
        automation_id="play-pause-button",
        control_type="Button",
        window="TestApp",
    )
    assert result == expected


def test_find_element_raises_lookup_on_miss(monkeypatch):
    """No matching automation_id → LookupError, not None."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native
    import assistant.automation.manifest_runtime as _me1

    monkeypatch.setattr(_native, "_backend", "terminator")
    monkeypatch.setattr(_me1, "_find_node_by_selector", lambda *a, **k: None)

    adapter = _TerminatorAdapter(native_desktop=object())
    with pytest.raises(LookupError):
        adapter.find_element(
            automation_id="ghost",
            control_type="Button",
            window="TestApp",
        )


# ─── F8: find_element by accessible name (no automation_id) ────────────

def test_find_element_by_name_when_no_automation_id(monkeypatch):
    """F8 regression: dispatch by name when the manifest has no automation_id.

    Spotify-style apps don't expose automation_id; the promoter writes
    name-only UIA selectors. The dispatcher MUST be able to walk the
    tree by accessible name for those manifests to dispatch at all.
    """
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native
    import assistant.automation.manifest_runtime as _me1

    monkeypatch.setattr(_native, "_backend", "terminator")
    expected = {
        "automation_id": "",
        "control_type": "Button",
        "name": "Play",
        "bounds": {"x": 50, "y": 60, "width": 40, "height": 40},
    }
    captured: dict = {}

    def _stub_walker(desktop, *, window, automation_id, name, control_type):
        captured["automation_id"] = automation_id
        captured["name"] = name
        return expected if name == "Play" else None

    monkeypatch.setattr(_me1, "_find_node_by_selector", _stub_walker)

    adapter = _TerminatorAdapter(native_desktop=object())
    result = adapter.find_element(name="Play", control_type="Button", window="TestApp")
    assert result == expected
    # The walker received the empty automation_id and the live name.
    assert captured == {"automation_id": "", "name": "Play"}


def test_find_node_by_selector_matches_by_name_when_aid_empty():
    """Walker unit test: empty automation_id + non-empty name → name lookup."""
    from assistant.automation.manifest_runtime import _find_node_by_selector

    # Synthetic AT tree: window root with one child Button whose name='Play'.
    class _Attrs:
        def __init__(self, aid, name, role, bounds):
            self.automation_id = aid
            self.name = name
            self.role = role
            self.bounds = bounds

    class _Node:
        def __init__(self, attrs, children=()):
            self.attributes = attrs
            self.children = list(children)

    play_btn = _Node(
        _Attrs(aid="", name="Play", role="Button",
               bounds={"x": 10, "y": 20, "width": 30, "height": 30}),
    )
    root = _Node(_Attrs(aid="", name="", role="Window", bounds=None),
                 children=[play_btn])

    class _App:
        def name(self): return "TestApp"
        def process_id(self): return 1234

    class _Desktop:
        def applications(self): return [_App()]
        def get_window_tree(self, pid):
            assert pid == 1234
            return root

    hit = _find_node_by_selector(
        _Desktop(), window="TestApp",
        automation_id="", name="Play", control_type="Button",
    )
    assert hit is not None
    assert hit["name"] == "Play"
    assert hit["bounds"] == {"x": 10, "y": 20, "width": 30, "height": 30}


def test_find_node_by_selector_empty_both_returns_none():
    """Walker guard: empty automation_id AND empty name → None (no match attempted)."""
    from assistant.automation.manifest_runtime import _find_node_by_selector

    class _Desktop:
        def applications(self):
            raise AssertionError("walker should not call into desktop when both keys empty")
        def get_window_tree(self, pid):
            raise AssertionError("walker should not call into desktop when both keys empty")

    hit = _find_node_by_selector(
        _Desktop(), window="TestApp",
        automation_id="", name="", control_type="Button",
    )
    assert hit is None


# ─── Healer-surface stubs — still NotImplementedError ──────────────────

# ─── Healer-surface adapter wiring (Session-5 final): enumerate_descendants,
#     screenshot, element_at_point. Was NotImplementedError stubs in v1; the
#     wired implementations below replace those stubs.

class _FakeAttrs:
    def __init__(self, aid="", name="", role="", bounds=None):
        self.automation_id = aid
        self.name = name
        self.role = role
        self.bounds = bounds


class _FakeNode:
    def __init__(self, attrs, children=()):
        self.attributes = attrs
        self.children = list(children)


class _FakeApp:
    def __init__(self, name, pid):
        self._name = name
        self._pid = pid
    def name(self): return self._name
    def process_id(self): return self._pid


class _FakeDesktop:
    def __init__(self, apps_to_trees: dict[int, "_FakeNode"], app_names: dict[int, str]):
        self._trees = apps_to_trees
        self._names = app_names
    def applications(self):
        return [_FakeApp(self._names[pid], pid) for pid in self._names]
    def get_window_tree(self, pid):
        return self._trees.get(pid)


def _make_tree() -> "_FakeNode":
    """Window root with a Toolbar containing two Buttons: Play, Pause."""
    play = _FakeNode(_FakeAttrs(
        aid="play-id", name="Play", role="Button",
        bounds={"x": 100, "y": 200, "width": 40, "height": 40},
    ))
    pause = _FakeNode(_FakeAttrs(
        aid="pause-id", name="Pause", role="Button",
        bounds={"x": 150, "y": 200, "width": 40, "height": 40},
    ))
    toolbar = _FakeNode(
        _FakeAttrs(aid="", name="MainToolbar", role="ToolBar",
                   bounds={"x": 50, "y": 180, "width": 200, "height": 80}),
        children=[play, pause],
    )
    root = _FakeNode(
        _FakeAttrs(aid="", name="TestAppWindow", role="Window",
                   bounds={"x": 0, "y": 0, "width": 800, "height": 600}),
        children=[toolbar],
    )
    return root


def test_enumerate_descendants_walks_tree_with_parent_chain(monkeypatch):
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native

    monkeypatch.setattr(_native, "_backend", "terminator")
    desktop = _FakeDesktop({1234: _make_tree()}, {1234: "TestApp"})
    adapter = _TerminatorAdapter(native_desktop=desktop)

    out = adapter.enumerate_descendants(parent_window="TestApp")
    # Toolbar (depth 1), Play (depth 2), Pause (depth 2) — root excluded.
    names = [e["name"] for e in out]
    assert "MainToolbar" in names
    assert "Play" in names
    assert "Pause" in names

    play_entry = next(e for e in out if e["name"] == "Play")
    # parent_chain reflects the path from root to Play's PARENT.
    assert play_entry["parent_chain"] == ["TestAppWindow", "MainToolbar"]
    assert play_entry["automation_id"] == "play-id"
    assert play_entry["control_type"] == "Button"
    # Play and Pause are siblings → sibling_count of their parent's children list.
    assert play_entry["sibling_count"] == 2


def test_enumerate_descendants_returns_empty_when_pid_not_found(monkeypatch):
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native

    monkeypatch.setattr(_native, "_backend", "terminator")
    desktop = _FakeDesktop({}, {})  # no apps
    adapter = _TerminatorAdapter(native_desktop=desktop)
    assert adapter.enumerate_descendants(parent_window="Ghost") == []


def test_enumerate_descendants_returns_empty_on_pywinauto_backend(monkeypatch):
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native

    monkeypatch.setattr(_native, "_backend", "pywinauto")
    adapter = _TerminatorAdapter(native_desktop=object())
    # Must return [] (not raise) so the healer falls through cleanly.
    assert adapter.enumerate_descendants(parent_window="any") == []


def test_screenshot_returns_png_bytes(monkeypatch):
    """screenshot wraps screen.capture_screenshot → PIL Image → PNG bytes."""
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    # Build a tiny PIL Image to stand in for the screen capture.
    from PIL import Image
    fake_img = Image.new("RGB", (4, 4), color=(0, 128, 255))
    monkeypatch.setattr("assistant.io.screen.capture_screenshot",
                        lambda region=None: fake_img)

    adapter = _TerminatorAdapter(native_desktop=object())
    data = adapter.screenshot()
    assert isinstance(data, bytes)
    # PNG magic bytes — proves the encoder ran end-to-end.
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_screenshot_raises_on_capture_failure(monkeypatch):
    """When screen.capture_screenshot returns None, screenshot must raise.

    Healer wraps in try/except → HealResult(ok=False, tier=2, error=...).
    """
    from assistant.automation.manifest_runtime import _TerminatorAdapter

    monkeypatch.setattr("assistant.io.screen.capture_screenshot",
                        lambda region=None: None)
    adapter = _TerminatorAdapter(native_desktop=object())
    with pytest.raises(RuntimeError):
        adapter.screenshot()


def test_element_at_point_returns_deepest_containing_node(monkeypatch):
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native

    monkeypatch.setattr(_native, "_backend", "terminator")
    desktop = _FakeDesktop({1234: _make_tree()}, {1234: "TestApp"})
    monkeypatch.setattr("assistant.io.screen.get_active_window",
                        lambda: "TestApp")
    adapter = _TerminatorAdapter(native_desktop=desktop)

    # (110, 210) lands inside Play (100-140 × 200-240) AND inside Toolbar AND
    # inside the Window root — element_at_point must return the DEEPEST hit.
    hit = adapter.element_at_point(110, 210)
    assert hit is not None
    assert hit["name"] == "Play"
    assert hit["automation_id"] == "play-id"
    # parent_chain should reflect the path from root to Play's PARENT.
    assert "TestAppWindow" in hit["parent_chain"]
    assert "MainToolbar" in hit["parent_chain"]


def test_element_at_point_returns_none_when_no_node_contains(monkeypatch):
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native

    monkeypatch.setattr(_native, "_backend", "terminator")
    desktop = _FakeDesktop({1234: _make_tree()}, {1234: "TestApp"})
    monkeypatch.setattr("assistant.io.screen.get_active_window",
                        lambda: "TestApp")
    adapter = _TerminatorAdapter(native_desktop=desktop)
    # (5000, 5000) is well outside any node's bounds.
    assert adapter.element_at_point(5000, 5000) is None


def test_element_at_point_returns_none_when_no_active_window(monkeypatch):
    from assistant.automation.manifest_runtime import _TerminatorAdapter
    from assistant.automation import native as _native

    monkeypatch.setattr(_native, "_backend", "terminator")
    monkeypatch.setattr("assistant.io.screen.get_active_window", lambda: None)
    adapter = _TerminatorAdapter(native_desktop=object())
    assert adapter.element_at_point(100, 100) is None
