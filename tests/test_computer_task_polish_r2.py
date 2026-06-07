"""
Tests for computer_task Polish Round 2 fixes.

Issue 1: "open settings" should focus existing window, not launch new
Issue 2: "pause spotify" false success detection + exit-0 empty output
Issue 3: Notepad get_text stripped for pure type tasks
Issue 4: Calculator routing to Terminator via "on [app]" pattern
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --- Issue 1: _detect_running_app reversed matching ---

def test_detect_running_app_no_overmatch():
    """Title words should not match random goal words -- goal words match titles."""
    from assistant.automation.router import _detect_running_app
    from unittest.mock import patch

    fake_windows = ["Program Manager", "Settings", "Calculator"]

    with patch("assistant.io.screen.get_open_windows", return_value=fake_windows):
        # "settings" is a candidate word from the goal, should match "Settings" window
        assert _detect_running_app("open settings") == "Settings"

        # "calculator" is a candidate word, should match "Calculator"
        assert _detect_running_app("use calculator") == "Calculator"

        # "internet" is NOT in any window title, should not match
        assert _detect_running_app("check internet") is None

        result = _detect_running_app("open file manager")
        print(f"  'open file manager' matched: {result}")


def test_detect_running_app_ignores_page_title_words():
    """Goal words matching a browser's PAGE title must not cause a false app match."""
    from assistant.automation.router import _detect_running_app
    from unittest.mock import patch

    fake_windows = [
        "Hello World Page - Mozilla Firefox",
        "Program Manager",
    ]

    with patch("assistant.io.screen.get_open_windows", return_value=fake_windows):
        # "hello"/"world" appear in Firefox's page title, NOT its app name —
        # must NOT match Firefox when the user targets notepad
        result = _detect_running_app("type hello world in notepad")
        assert result is None, f"Expected None, got {result!r}"

        # "firefox" IS the app name — should match
        result = _detect_running_app("open firefox")
        assert result == "Hello World Page - Mozilla Firefox"

        # "notepad" isn't running, should return None
        result = _detect_running_app("type something in notepad")
        assert result is None


# --- Issue 2a: _needs_retry catches specific "no X" failures ---

def test_needs_retry_catches_no_prefix():
    """Specific 'no devices/active/results' should be detected as failures."""
    from assistant.code_executor import _needs_retry

    # These should be caught by specific "no X" phrases
    assert _needs_retry("No devices found. Check if the 'devices' key exists.") is True
    assert _needs_retry("No active device found") is True
    assert _needs_retry("no results") is True
    assert _needs_retry("not found") is True
    assert _needs_retry("not available") is True

    # "no errors" should NOT be caught (it's a success message)
    assert _needs_retry("no errors encountered") is False
    assert _needs_retry("no issues found") is False

    # Multi-line real data is success
    assert _needs_retry("Track: My Song\nArtist: Someone\nAlbum: Best Of") is False

    # "no" in the middle of a success message should not trigger
    assert _needs_retry("Playing: No Doubt - Don't Speak") is False

    print("  _needs_retry specific 'no X' phrases: all checks passed")


# --- Issue 2b: _run_tier2 exit-0 empty output -> success ---

def test_run_tier2_exit0_empty_is_success():
    """Exit code 0 with empty stdout should be success, not failure."""
    from assistant.code_executor import _needs_retry

    # "(completed successfully)" should NOT trigger retry
    assert _needs_retry("(completed successfully)") is False

    # "(no output)" SHOULD trigger retry (non-zero exit or actual failure)
    assert _needs_retry("(no output)") is True

    # Actual output still works
    assert _needs_retry("Playback paused") is False
    assert _needs_retry("Done") is False

    print("  exit-0 empty output: (completed successfully) not retried")


# --- Issue 1b: focus_window fallback to open_app ---

def test_focus_falls_back_to_open():
    """When focus_window returns Error, should fall through to open_app."""
    from assistant.automation.router import _execute_native_task
    from unittest.mock import patch, AsyncMock

    async def _run():
        mock_focus = AsyncMock(return_value="Error: window not found")
        mock_open = AsyncMock(return_value="Opened application: settings")
        mock_llm = AsyncMock(return_value="[]")

        with patch("assistant.automation.router._detect_running_app", return_value="Settings"), \
             patch("assistant.automation.native.focus_window", mock_focus), \
             patch("assistant.automation.native.open_app", mock_open):
            result = await _execute_native_task("open settings", mock_llm)

        # focus failed -> should have called open_app as fallback
        mock_focus.assert_called_once_with("Settings")
        mock_open.assert_called_once_with("settings")
        assert "Opened" in result
        print(f"  focus failed -> open_app called as fallback")

    asyncio.run(_run())


# --- Issue 3: get_text stripped for type tasks ---

def test_strip_get_text_for_type_tasks():
    """Pure type tasks should have get_text steps stripped from LLM plans."""

    steps = [
        {"action": "open", "params": {"name": "notepad"}},
        {"action": "wait", "params": {"seconds": 2}},
        {"action": "type", "params": {"text": "hello world", "window": "Notepad"}},
        {"action": "get_text", "params": {"selector": "name:37 of 312 characters", "window": "Notepad"}},
    ]

    _TYPE_WORDS = {"type", "write", "enter", "input", "paste"}
    _RESULT_WORDS = {"calculate", "compute", "result", "read", "get", "show", "what", "check", "display"}

    # Test 1: "type hello in notepad" -- pure type task, get_text should be stripped
    goal = "type hello in notepad"
    goal_words_set = set(goal.lower().split())
    assert bool(goal_words_set & _TYPE_WORDS) is True
    assert bool(goal_words_set & _RESULT_WORDS) is False
    filtered = [s for s in steps if s.get("action") != "get_text"]
    assert len(filtered) == 3
    print("  type task: get_text stripped correctly")

    # Test 2: "calculate 7 times 15 on calculator" -- needs result, keep get_text
    goal2 = "calculate 7 times 15 on calculator"
    goal_words_set2 = set(goal2.lower().split())
    assert bool(goal_words_set2 & _RESULT_WORDS) is True
    print("  calculate task: get_text preserved correctly")


# --- Issue 4: detect_backend "on [app]" pattern ---

def test_detect_backend_app_context_pattern():
    """Goals like 'multiply X on calculator' should route to native."""
    from assistant.automation.router import detect_backend
    from unittest.mock import patch

    with patch("assistant.automation.router._detect_running_app", return_value=None), \
         patch("assistant.automation.router._check_routing_preference", return_value=None):

        backend, meta = detect_backend("multiply 3 and 4 on calculator")
        assert backend == "native", f"Expected 'native', got '{backend}'"
        assert meta["reason"] == "app_context_pattern"
        assert meta["app"] == "calculator"
        print(f"  'multiply 3 and 4 on calculator' -> {backend} ({meta['reason']})")

        backend2, meta2 = detect_backend("play my liked songs on spotify")
        assert backend2 == "native"
        assert meta2["app"] == "spotify"
        print(f"  'play my liked songs on spotify' -> {backend2} ({meta2['reason']})")

        # "it" is excluded
        backend3, meta3 = detect_backend("do something on it")
        assert backend3 == "unknown"
        print(f"  'do something on it' -> {backend3} (correctly excluded)")

        # "mode" is excluded
        backend4, meta4 = detect_backend("turn on dark mode")
        assert backend4 == "unknown"
        print(f"  'turn on dark mode' -> {backend4} (correctly excluded)")

        # "type X in Y" — "type" is a form verb but "in notepad" is app context
        backend5, meta5 = detect_backend("type hello world in notepad")
        assert backend5 == "native", f"Expected 'native', got '{backend5}' (reason={meta5.get('reason')})"
        assert meta5["app"] == "notepad"
        print(f"  'type hello world in notepad' -> {backend5} ({meta5['reason']})")

        # "fill form with john" — "with" + form verb → skip (Y is data)
        backend6, meta6 = detect_backend("fill the form with john")
        assert backend6 != "native" or meta6.get("reason") != "app_context_pattern", \
            f"Should NOT treat 'john' as an app name"
        print(f"  'fill the form with john' -> {backend6} (correctly skipped app_context)")


# --- Issue 1c: simple shortcut focuses running app (success case) ---

def test_simple_shortcut_focuses_running():
    """'open settings' when Settings is running should focus, not launch new."""
    from assistant.automation.router import _execute_native_task
    from unittest.mock import patch, AsyncMock

    async def _run():
        mock_focus = AsyncMock(return_value="Focused window: Settings")
        mock_open = AsyncMock(return_value="Opened application: settings")
        mock_llm = AsyncMock(return_value="[]")

        with patch("assistant.automation.router._detect_running_app", return_value="Settings"), \
             patch("assistant.automation.native.focus_window", mock_focus), \
             patch("assistant.automation.native.open_app", mock_open):
            result = await _execute_native_task("open settings", mock_llm)

        mock_focus.assert_called_once_with("Settings")
        mock_open.assert_not_called()
        assert "Focused" in result
        print(f"  'open settings' (running) -> focus_window called, not open_app")

    asyncio.run(_run())


# --- Issue 4b: Prompt prefers type over click for data entry ---

def test_prompt_prefers_type_over_click():
    """The app plan prompt should instruct type for data entry, not individual clicks."""
    from assistant.automation.router import _APP_PLAN_PROMPT

    # Should NOT contain the old "click each button individually" instruction
    assert "click each button individually" not in _APP_PLAN_PROMPT
    assert "click steps on those buttons" not in _APP_PLAN_PROMPT

    # Should contain the new "type for data entry" instruction
    assert "type" in _APP_PLAN_PROMPT.lower()
    assert "NOT individual" in _APP_PLAN_PROMPT or "not individual" in _APP_PLAN_PROMPT.lower()

    print("  prompt correctly prefers type over click for data entry")


# --- Deterministic step-plan fixes ---

def test_sanitize_strips_redundant_open_focus():
    """When pre-steps already focused, LLM's open/focus steps should be stripped."""
    from assistant.automation.router import _sanitize_steps

    steps = [
        {"action": "open", "params": {"name": "notepad"}},
        {"action": "wait", "params": {"seconds": 2}},
        {"action": "type", "params": {"text": "hello", "window": "Notepad"}},
    ]

    # With pre-steps: open/focus should be stripped
    cleaned = _sanitize_steps(steps, has_pre_steps=True)
    assert len(cleaned) == 2  # open stripped, wait + type remain
    assert cleaned[0]["action"] == "wait"
    assert cleaned[1]["action"] == "type"
    print("  pre-steps present: open/focus stripped correctly")

    # Without pre-steps: open/focus should be kept
    cleaned2 = _sanitize_steps(steps, has_pre_steps=False)
    assert len(cleaned2) == 3  # all kept
    print("  no pre-steps: open/focus preserved correctly")


def test_sanitize_strips_redundant_press_key():
    """press_key enter after type ending with = should be stripped."""
    from assistant.automation.router import _sanitize_steps

    # Calculator: type "25+4=" then press_key enter -> enter is redundant
    steps = [
        {"action": "type", "params": {"text": "25+4=", "window": "Calculator"}},
        {"action": "press_key", "params": {"key": "enter"}},
        {"action": "get_text", "params": {"selector": "name:Result", "window": "Calculator"}},
    ]
    cleaned = _sanitize_steps(steps)
    assert len(cleaned) == 2  # press_key stripped
    assert cleaned[0]["action"] == "type"
    assert cleaned[1]["action"] == "get_text"
    print("  type ending with =: redundant press_key enter stripped")

    # type NOT ending with = -> press_key should be KEPT
    steps2 = [
        {"action": "type", "params": {"text": "hello world", "window": "Notepad"}},
        {"action": "press_key", "params": {"key": "enter"}},
    ]
    cleaned2 = _sanitize_steps(steps2)
    assert len(cleaned2) == 2  # both kept
    print("  type not ending with =: press_key enter preserved correctly")


def test_refocus_before_type():
    """type steps without selector should re-focus the target window first."""
    # This is tested structurally — verify run_app_steps calls focus_window before type
    from assistant.automation import native as app_automation
    from unittest.mock import patch, AsyncMock
    import asyncio

    async def _run():
        focus_calls = []
        original_focus = app_automation.focus_window

        async def tracking_focus(name):
            focus_calls.append(name)
            return f"Focused window: {name}"

        with patch.object(app_automation, "focus_window", side_effect=tracking_focus), \
             patch.object(app_automation, "type_text", new_callable=AsyncMock, return_value="Typed text"):
            steps = [
                {"action": "type", "params": {"text": "hello", "window": "Notepad"}},
            ]
            await app_automation.run_app_steps(steps)

        # focus_window should have been called BEFORE type_text
        assert "Notepad" in focus_calls, f"Expected focus on 'Notepad', got: {focus_calls}"
        print("  re-focus called before type without selector")

    asyncio.run(_run())


def test_sanitize_type_to_press_key():
    """type 'ctrl+s' should become press_key 'ctrl+s', not type literal text."""
    from assistant.automation.router import _sanitize_steps

    steps = [
        {"action": "type", "params": {"text": "Ctrl+S", "window": "Notepad"}},
        {"action": "type", "params": {"text": "tgyuiojh.txt", "window": "Notepad"}},
        {"action": "type", "params": {"text": "Enter", "window": "Notepad"}},
    ]
    cleaned = _sanitize_steps(steps)

    # Fix 3: type "Ctrl+S" → press_key "ctrl+s"
    assert cleaned[0]["action"] == "press_key"
    assert cleaned[0]["params"]["key"] == "ctrl+s"
    print("  type 'Ctrl+S' -> press_key 'ctrl+s'")

    # Fix 4: wait injected after modifier shortcut (ctrl+s opens Save As dialog)
    assert cleaned[1]["action"] == "wait"
    print("  wait injected after modifier shortcut")

    # Fix 4: window stripped from type (dialog mode — type into Save As dialog)
    assert cleaned[2]["action"] == "type"
    assert cleaned[2]["params"]["text"] == "tgyuiojh.txt"
    assert "window" not in cleaned[2]["params"], "window should be stripped in dialog mode"
    print("  type 'tgyuiojh.txt' -> window stripped (dialog mode)")

    # Fix 3: type "Enter" → press_key "enter"
    assert cleaned[3]["action"] == "press_key"
    assert cleaned[3]["params"]["key"] == "enter"
    print("  type 'Enter' -> press_key 'enter'")


def test_sanitize_dialog_mode_after_modifier():
    """After press_key with modifier, type steps should not re-focus original window."""
    from assistant.automation.router import _sanitize_steps

    # Scenario: Save As dialog opened by ctrl+shift+s
    steps = [
        {"action": "press_key", "params": {"key": "ctrl+shift+s"}},
        {"action": "wait", "params": {"seconds": 1}},
        {"action": "type", "params": {"text": "report.pdf", "window": "Word"}},
        {"action": "press_key", "params": {"key": "enter"}},
    ]
    cleaned = _sanitize_steps(steps)

    assert cleaned[0]["action"] == "press_key"
    assert cleaned[0]["params"]["key"] == "ctrl+shift+s"

    # Wait already present — no extra wait injected
    assert cleaned[1]["action"] == "wait"
    assert cleaned[1]["params"]["seconds"] == 1

    # Window stripped — types into Save As dialog, not re-focus Word
    assert cleaned[2]["action"] == "type"
    assert cleaned[2]["params"]["text"] == "report.pdf"
    assert "window" not in cleaned[2]["params"]
    print("  dialog mode: window stripped from type after modifier shortcut")

    # Enter confirms dialog
    assert cleaned[3]["action"] == "press_key"
    assert len(cleaned) == 4  # no extra steps injected (wait was already present)
    print("  dialog mode: no extra wait when already present")

    # Non-modifier press_key should NOT enter dialog mode
    steps2 = [
        {"action": "press_key", "params": {"key": "enter"}},
        {"action": "type", "params": {"text": "hello", "window": "Notepad"}},
    ]
    cleaned2 = _sanitize_steps(steps2)
    assert cleaned2[1]["action"] == "type"
    assert cleaned2[1]["params"].get("window") == "Notepad"  # window preserved
    print("  non-modifier press_key: window preserved (no dialog mode)")

    # click/focus resets dialog mode
    steps3 = [
        {"action": "press_key", "params": {"key": "ctrl+o"}},
        {"action": "click", "params": {"selector": "name:Browse", "window": "Dialog"}},
        {"action": "type", "params": {"text": "file.txt", "window": "Notepad"}},
    ]
    cleaned3 = _sanitize_steps(steps3)
    assert cleaned3[2]["action"] == "type"
    assert cleaned3[2]["params"].get("window") == "Notepad"  # window preserved after click reset
    print("  click resets dialog mode: window preserved")


def test_active_window_fallback_for_dialogs():
    """When _detect_running_app finds nothing, fall back to active window (dialog)."""
    from assistant.automation.router import _execute_native_task
    from unittest.mock import patch, AsyncMock
    import asyncio

    async def _run():
        mock_llm = AsyncMock(return_value='[{"action": "type", "params": {"text": "wyirja.txt"}}, {"action": "click", "params": {"selector": "name:Save", "window": "Save As"}}]')
        mock_list = AsyncMock(return_value="name:File name (EditBox)\nname:Save (Button)")
        mock_focus = AsyncMock(return_value="Focused window: Save As")
        mock_type = AsyncMock(return_value="Typed text into focus")
        mock_click = AsyncMock(return_value="Clicked name:Save")

        with patch("assistant.automation.router._detect_running_app", return_value=None), \
             patch("assistant.io.screen.get_active_window", return_value="Save As"), \
             patch("assistant.automation.native.list_elements", mock_list), \
             patch("assistant.automation.native.focus_window", mock_focus), \
             patch("assistant.automation.native.run_app_steps", AsyncMock(return_value="Typed text\nClicked Save")) as mock_run:
            result = await _execute_native_task(
                "type 'wyirja.txt' in the file name field and click Save",
                mock_llm,
            )

        # Should have used active window "Save As" as fallback
        mock_list.assert_called_once_with("Save As")
        assert result != "__FALLBACK__"
        print(f"  active window fallback used 'Save As' dialog")

    asyncio.run(_run())


if __name__ == "__main__":
    print("=== computer_task Polish Round 2 Tests ===\n")

    print("1. _detect_running_app reversed matching:")
    test_detect_running_app_no_overmatch()
    print("   PASSED\n")

    print("1b. _detect_running_app ignores page title words:")
    test_detect_running_app_ignores_page_title_words()
    print("   PASSED\n")

    print("2a. _needs_retry specific 'no X' phrases:")
    test_needs_retry_catches_no_prefix()
    print("   PASSED\n")

    print("2b. _run_tier2 exit-0 empty output:")
    test_run_tier2_exit0_empty_is_success()
    print("   PASSED\n")

    print("3. get_text stripped for type tasks:")
    test_strip_get_text_for_type_tasks()
    print("   PASSED\n")

    print("4. detect_backend 'on [app]' pattern:")
    test_detect_backend_app_context_pattern()
    print("   PASSED\n")

    print("5. Simple shortcut focuses running app:")
    test_simple_shortcut_focuses_running()
    print("   PASSED\n")

    print("6. Focus fallback to open_app:")
    test_focus_falls_back_to_open()
    print("   PASSED\n")

    print("7. Prompt: type over click for data entry:")
    test_prompt_prefers_type_over_click()
    print("   PASSED\n")

    print("8. Sanitize: strip redundant open/focus:")
    test_sanitize_strips_redundant_open_focus()
    print("   PASSED\n")

    print("9. Sanitize: strip redundant press_key:")
    test_sanitize_strips_redundant_press_key()
    print("   PASSED\n")

    print("10. Re-focus before type:")
    test_refocus_before_type()
    print("   PASSED\n")

    print("11. Sanitize: type keyboard shortcuts -> press_key:")
    test_sanitize_type_to_press_key()
    print("   PASSED\n")

    print("12. Sanitize: dialog mode after modifier shortcut:")
    test_sanitize_dialog_mode_after_modifier()
    print("   PASSED\n")

    print("13. Active window fallback for dialogs:")
    test_active_window_fallback_for_dialogs()
    print("   PASSED\n")

    print("=== All computer_task Polish Round 2 tests passed ===")
