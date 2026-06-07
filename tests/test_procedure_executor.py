"""
test_procedure_executor.py — TP-1c: procedure_executor unit tests

Tests variable resolution, error detection, and step routing
with mocked computer_task backends.

Run: python test_procedure_executor.py
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import assistant.procedures as ps
import assistant.procedure_executor as pe
from assistant import config as _config_stub


def _fresh_db():
    from assistant.storage.db import _reset_for_testing, init_db
    _reset_for_testing()
    tmp = Path(tempfile.mkdtemp()) / "memory" / "personality.db"
    _config_stub.SANDBOX_DIR = tmp.parent.parent
    tmp.parent.mkdir(parents=True, exist_ok=True)
    init_db(tmp)
    ps._repo = None
    ps.init_procedure_db()


def run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractUserInput(unittest.TestCase):

    def test_strips_trigger(self):
        result = pe._extract_user_input("search cooking recipes on youtube", "search")
        self.assertEqual(result, "cooking recipes on youtube")

    def test_no_match_returns_full(self):
        result = pe._extract_user_input("open notepad", "launch editor")
        self.assertEqual(result, "open notepad")

    def test_exact_match_empty_remainder(self):
        result = pe._extract_user_input("my workflow", "my workflow")
        self.assertEqual(result, "")

    def test_subsequence_extraction(self):
        result = pe._extract_user_input("search mechanical keyboard on youtube", "search on youtube")
        self.assertEqual(result, "mechanical keyboard")

    def test_subsequence_multi_gap(self):
        result = pe._extract_user_input("message john happy birthday on whatsapp", "message on whatsapp")
        self.assertEqual(result, "john happy birthday")


class TestResolveVariables(unittest.TestCase):

    def test_substitutes_date(self):
        step = {"type": "app", "action": "type", "params": {"text": "Today is {date}"}}
        resolved = pe._resolve_variables(step, {"date": "2026-04-18", "time": "10:00",
                                                  "user_input": "", "clipboard": ""})
        self.assertEqual(resolved["params"]["text"], "Today is 2026-04-18")

    def test_substitutes_user_input(self):
        step = {"type": "app", "action": "type", "params": {"text": "{user_input}"}}
        resolved = pe._resolve_variables(step, {"user_input": "hello world",
                                                  "date": "", "time": "", "clipboard": ""})
        self.assertEqual(resolved["params"]["text"], "hello world")

    def test_no_placeholders_unchanged(self):
        step = {"type": "app", "action": "open", "params": {"name": "notepad"}}
        resolved = pe._resolve_variables(step, {"user_input": "x", "date": "y",
                                                  "time": "z", "clipboard": "w"})
        self.assertEqual(resolved, step)

    def test_nested_substitution(self):
        step = {"type": "browser", "action": "navigate",
                "params": {"url": "https://example.com/search?q={user_input}"}}
        resolved = pe._resolve_variables(step, {"user_input": "cats",
                                                  "date": "", "time": "", "clipboard": ""})
        self.assertIn("cats", resolved["params"]["url"])


class TestDefaultBrowser(unittest.TestCase):
    """T5 regression: _default_browser() must use preferences, not hardcode 'chrome'."""

    def test_returns_preference_when_set(self):
        with patch("assistant.preferences.get_preference", return_value={"value": "firefox"}) as mock_get:
            self.assertEqual(pe._default_browser(), "firefox")
            mock_get.assert_called_once_with("default_browser")

    def test_falls_back_to_chrome(self):
        with patch("assistant.preferences.get_preference", return_value=None):
            self.assertEqual(pe._default_browser(), "chrome")

    def test_falls_back_on_error(self):
        with patch("assistant.preferences.get_preference", side_effect=RuntimeError("DB not init")):
            self.assertEqual(pe._default_browser(), "chrome")

    def test_browser_names_is_config(self):
        from assistant import config as cfg
        self.assertIs(pe._BROWSER_NAMES, cfg.BROWSER_NAMES)

    def test_is_browser_name_canonical(self):
        self.assertTrue(pe._is_browser_name("chrome"))
        self.assertTrue(pe._is_browser_name("firefox"))
        self.assertTrue(pe._is_browser_name("brave"))

    def test_is_browser_name_alias(self):
        self.assertTrue(pe._is_browser_name("google chrome"))
        self.assertTrue(pe._is_browser_name("mozilla firefox"))
        self.assertTrue(pe._is_browser_name("microsoft edge"))

    def test_is_browser_name_non_browser(self):
        self.assertFalse(pe._is_browser_name("notepad"))
        self.assertFalse(pe._is_browser_name("spotify"))


class TestSkipOpenBeforeNavigate(unittest.TestCase):

    def test_skips_open_chrome_before_navigate(self):
        steps = [
            {"type": "app", "action": "open", "params": {"name": "Chrome"}},
            {"type": "browser", "action": "navigate", "params": {"url": "https://youtube.com"}},
        ]
        self.assertTrue(pe._should_skip_open_before_navigate(steps[0], steps, 0))

    def test_skips_open_google_chrome(self):
        steps = [
            {"type": "app", "action": "open", "params": {"name": "Google Chrome"}},
            {"type": "browser", "action": "navigate", "params": {"url": "https://x.com"}},
        ]
        self.assertTrue(pe._should_skip_open_before_navigate(steps[0], steps, 0))

    def test_does_not_skip_open_notepad(self):
        steps = [
            {"type": "app", "action": "open", "params": {"name": "notepad"}},
            {"type": "browser", "action": "navigate", "params": {"url": "https://x.com"}},
        ]
        self.assertFalse(pe._should_skip_open_before_navigate(steps[0], steps, 0))

    def test_does_not_skip_when_next_is_app_step(self):
        steps = [
            {"type": "app", "action": "open", "params": {"name": "Chrome"}},
            {"type": "app", "action": "click", "params": {"selector": "name:search"}},
        ]
        self.assertFalse(pe._should_skip_open_before_navigate(steps[0], steps, 0))

    def test_does_not_skip_when_last_step(self):
        steps = [
            {"type": "app", "action": "open", "params": {"name": "Chrome"}},
        ]
        self.assertFalse(pe._should_skip_open_before_navigate(steps[0], steps, 0))

    def test_does_not_skip_non_open_action(self):
        steps = [
            {"type": "app", "action": "focus", "params": {"name": "Chrome"}},
            {"type": "browser", "action": "navigate", "params": {"url": "https://x.com"}},
        ]
        self.assertFalse(pe._should_skip_open_before_navigate(steps[0], steps, 0))


class TestIsError(unittest.TestCase):

    def test_error_string(self):
        self.assertTrue(pe._is_error("Error: app not found"))
        self.assertTrue(pe._is_error("failed to open"))
        self.assertTrue(pe._is_error("Element not found"))
        self.assertTrue(pe._is_error("Operation timed out"))

    def test_success_string(self):
        self.assertFalse(pe._is_error("Opened notepad"))
        self.assertFalse(pe._is_error("Navigated to https://google.com"))
        self.assertFalse(pe._is_error("Pressed ctrl+s"))
        self.assertFalse(pe._is_error("Waited 2s"))


class TestBuildVariables(unittest.TestCase):

    def test_keys_present(self):
        proc = {"trigger": "open my workflow", "name": "Test"}
        with patch.object(pe, "_get_clipboard", return_value="clipboard_content"):
            variables = run(pe._build_variables(proc, "open my workflow please"))
        self.assertIn("user_input", variables)
        self.assertIn("date", variables)
        self.assertIn("time", variables)
        self.assertIn("clipboard", variables)
        self.assertEqual(variables["clipboard"], "clipboard_content")

    def test_user_input_extracted(self):
        proc = {"trigger": "search", "name": "Search"}
        variables = run(pe._build_variables(proc, "search cooking recipes"))
        self.assertEqual(variables["user_input"], "cooking recipes")


# ─────────────────────────────────────────────────────────────────────────────
# App step routing
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteAppStep(unittest.TestCase):

    def _run(self, step):
        return run(pe._execute_app_step(step))

    def test_open(self):
        mock_aa = MagicMock()
        mock_aa.open_app = AsyncMock(return_value="Opened notepad")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}):
            result = self._run({"action": "open", "params": {"name": "notepad"}})
        self.assertEqual(result, "Opened notepad")
        mock_aa.open_app.assert_called_once_with("notepad")

    def test_close(self):
        mock_aa = MagicMock()
        mock_aa.close_app = AsyncMock(return_value="Closed notepad")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}):
            result = self._run({"action": "close", "params": {"name": "notepad"}})
        self.assertEqual(result, "Closed notepad")

    def test_focus(self):
        mock_aa = MagicMock()
        mock_aa.focus_window = AsyncMock(return_value="Focused VS Code")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}):
            result = self._run({"action": "focus", "params": {"name": "VS Code"}})
        self.assertEqual(result, "Focused VS Code")

    def test_click(self):
        mock_aa = MagicMock()
        mock_aa.click_element = AsyncMock(return_value="Clicked save")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}):
            result = self._run({"action": "click",
                                 "params": {"selector": "name:Save", "window": "Notepad"}})
        self.assertEqual(result, "Clicked save")
        mock_aa.click_element.assert_called_once_with("name:Save", "Notepad")

    def test_type_with_window_refocuses(self):
        mock_aa = MagicMock()
        mock_aa.focus_window = AsyncMock(return_value="Focused Notepad")
        mock_aa.type_text = AsyncMock(return_value="Typed hello")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}):
            result = self._run({"action": "type",
                                 "params": {"text": "hello", "window": "Notepad"}})
        mock_aa.focus_window.assert_called_once_with("Notepad")
        mock_aa.type_text.assert_called_once_with("hello", None, "Notepad")
        self.assertEqual(result, "Typed hello")

    def test_wait(self):
        async def _inner():
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await pe._execute_app_step({"action": "wait", "params": {"seconds": 3}})
                mock_sleep.assert_called_once_with(3.0)
                self.assertEqual(result, "Waited 3.0s")
        run(_inner())

    def test_press_key_single(self):
        mock_pag = MagicMock()
        mock_pag.press = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pag}):
            with patch.dict(sys.modules, {"assistant.automation.native":
                                           MagicMock(open_app=AsyncMock())}):
                result = self._run({"action": "press_key", "params": {"key": "enter"}})
        self.assertIn("Pressed", result)

    def test_press_key_combo(self):
        mock_pag = MagicMock()
        mock_pag.hotkey = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pag}):
            with patch.dict(sys.modules, {"assistant.automation.native":
                                           MagicMock(open_app=AsyncMock())}):
                result = self._run({"action": "press_key", "params": {"key": "ctrl+s"}})
        mock_pag.hotkey.assert_called_once_with("ctrl", "s")
        self.assertIn("Pressed", result)

    def test_unknown_action(self):
        with patch.dict(sys.modules, {"assistant.automation.native": MagicMock()}):
            result = self._run({"action": "teleport", "params": {}})
        self.assertIn("Unknown", result)


# ─────────────────────────────────────────────────────────────────────────────
# Browser step routing (via app automation — no Playwright)
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteBrowserStepViaApp(unittest.TestCase):

    def _run(self, step, active_window=None):
        return run(pe._execute_browser_step_via_app(step, active_window))

    def test_navigate_opens_browser_and_types_url(self):
        mock_pag = MagicMock()
        with (
            patch("assistant.automation.native.open_app", new=AsyncMock(return_value="Opened browser")) as mock_open,
            patch("assistant.procedure_executor._default_browser", return_value="chrome"),
            patch("assistant.procedure_executor._ensure_foreground", new=AsyncMock(return_value=True)),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch.dict(sys.modules, {"pyautogui": mock_pag}),
        ):
            result = self._run(
                {"action": "navigate", "params": {"url": "https://youtube.com"}}
            )
        mock_open.assert_called_once_with("chrome")
        mock_pag.hotkey.assert_called_once_with("ctrl", "l")
        mock_pag.typewrite.assert_called_once()
        mock_pag.press.assert_called_once_with("enter")
        self.assertIn("Navigated", result)
        self.assertIn("youtube.com", result)

    def test_click_delegates_to_app_automation(self):
        with patch("assistant.automation.native.click_element",
                   new=AsyncMock(return_value="Clicked search")) as mock_click:
            result = self._run({"action": "click", "params": {"selector": "name:Search"}},
                               active_window="chrome")
        mock_click.assert_called_once_with("name:Search", "chrome")
        self.assertEqual(result, "Clicked search")

    def test_press_key(self):
        mock_pag = MagicMock()
        with patch.dict(sys.modules, {"pyautogui": mock_pag, "assistant.automation.native": MagicMock()}):
            result = self._run({"action": "press", "params": {"key": "enter"}})
        mock_pag.press.assert_called_once_with("enter")
        self.assertIn("Pressed", result)

    def test_unknown_action(self):
        with patch.dict(sys.modules, {"assistant.automation.native": MagicMock()}):
            result = self._run({"action": "teleport", "params": {}})
        self.assertIn("Unknown", result)


# ─────────────────────────────────────────────────────────────────────────────
# run_procedure integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRunProcedure(unittest.TestCase):
    """
    Tests for run_procedure orchestration.

    We patch _execute_app_step / _execute_browser_step_via_app directly
    here to avoid fighting Python's module-level import caching, which is
    tricky to control reliably with patch.dict(sys.modules).
    The routing logic for individual steps is already covered by
    TestExecuteAppStep and TestExecuteBrowserStepViaApp above.
    """

    def setUp(self):
        _fresh_db()
        self._fg_patcher = patch("assistant.procedure_executor._ensure_foreground",
                                 new=AsyncMock(return_value=True))
        self._fg_patcher.start()

    def tearDown(self):
        self._fg_patcher.stop()

    def _proc(self, steps, trigger="my workflow"):
        ps.create_procedure(trigger, "My Workflow", steps)
        return ps.get_procedure(trigger)

    def _run(self, proc, text="my workflow"):
        return run(pe.run_procedure(proc, text))

    def test_empty_steps(self):
        proc = {"id": 1, "name": "Empty", "trigger": "empty", "steps": []}
        result = self._run(proc)
        self.assertIn("no steps", result)

    def test_single_app_step(self):
        steps = [{"type": "app", "action": "open", "params": {"name": "notepad"}}]
        proc = self._proc(steps)
        with patch("assistant.procedure_executor._execute_app_step",
                   new=AsyncMock(return_value="Opened notepad")):
            result = self._run(proc)
        self.assertIn("Step 1", result)
        self.assertIn("Opened notepad", result)

    def test_multiple_steps_all_run(self):
        steps = [
            {"type": "app", "action": "open",      "params": {"name": "notepad"}},
            {"type": "app", "action": "press_key", "params": {"key": "ctrl+n"}},
        ]
        proc = self._proc(steps)
        with patch("assistant.procedure_executor._execute_app_step",
                   new=AsyncMock(side_effect=["Opened notepad", "Pressed ctrl+n"])):
            result = self._run(proc)
        self.assertIn("Step 1", result)
        self.assertIn("Step 2", result)

    def test_stops_on_error(self):
        steps = [
            {"type": "app", "action": "open",      "params": {"name": "notepad"}},
            {"type": "app", "action": "press_key", "params": {"key": "ctrl+n"}},
            {"type": "app", "action": "type",      "params": {"text": "hello"}},
        ]
        proc = self._proc(steps)
        with patch("assistant.procedure_executor._execute_app_step",
                   new=AsyncMock(return_value="Error: app not found")):
            result = self._run(proc)
        self.assertIn("Step 1", result)
        self.assertNotIn("Step 2", result)
        self.assertNotIn("Step 3", result)

    def test_records_usage(self):
        steps = [{"type": "app", "action": "open", "params": {"name": "notepad"}}]
        proc = self._proc(steps)
        self.assertEqual(proc["use_count"], 0)
        with patch("assistant.procedure_executor._execute_app_step",
                   new=AsyncMock(return_value="Opened notepad")):
            self._run(proc)
        updated = ps.get_procedure_by_id(proc["id"])
        self.assertEqual(updated["use_count"], 1)

    def test_variable_substitution_in_run(self):
        steps = [{"type": "app", "action": "type",
                  "params": {"text": "Searching for {user_input}"}}]
        proc = self._proc(steps, trigger="search on web")

        captured = []

        async def _capture(step, active_window=None, **kwargs):
            captured.append(step)
            return "Typed text"

        with patch("assistant.procedure_executor._execute_app_step", new=_capture):
            self._run(proc, "search on web cooking recipes")

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["params"]["text"], "Searching for cooking recipes")


class TestWindowContextTracking(unittest.TestCase):
    """Verify that run_procedure propagates window context to subsequent steps."""

    def setUp(self):
        _fresh_db()
        self._fg_patcher = patch("assistant.procedure_executor._ensure_foreground",
                                 new=AsyncMock(return_value=True))
        self._fg_mock = self._fg_patcher.start()

    def tearDown(self):
        self._fg_patcher.stop()

    def test_navigate_sets_browser_context(self):
        with patch("assistant.procedure_executor._default_browser", return_value="brave"):
            self.assertEqual(pe._get_window_context(
                {"type": "browser", "action": "navigate", "params": {"url": "https://x.com"}}
            ), "brave")

    def test_open_sets_context(self):
        self.assertEqual(pe._get_window_context(
            {"type": "app", "action": "open", "params": {"name": "notepad"}}
        ), "notepad")

    def test_open_browser_normalizes_to_default(self):
        with patch("assistant.procedure_executor._default_browser", return_value="firefox"):
            self.assertEqual(pe._get_window_context(
                {"type": "app", "action": "open", "params": {"name": "Google Chrome"}}
            ), "firefox")

    def test_click_no_context(self):
        self.assertIsNone(pe._get_window_context(
            {"type": "app", "action": "click", "params": {"selector": "name:Save"}}
        ))

    def test_context_propagated_to_click(self):
        """After navigate, app click step receives the default browser window context."""
        steps = [
            {"type": "browser", "action": "navigate", "params": {"url": "https://youtube.com"}},
            {"type": "app", "action": "click", "params": {"selector": "name:search"}},
        ]
        ps.create_procedure("yt search", "YT Search", steps)
        proc = ps.get_procedure("yt search")

        captured_windows = []

        async def _mock_browser(step, active_window=None, **kwargs):
            return "Navigated to https://youtube.com"

        async def _mock_app(step, active_window=None, **kwargs):
            captured_windows.append(active_window)
            return "Clicked search"

        with (
            patch("assistant.procedure_executor._default_browser", return_value="brave"),
            patch("assistant.procedure_executor._execute_browser_step_via_app", new=_mock_browser),
            patch("assistant.procedure_executor._execute_app_step", new=_mock_app),
        ):
            run(pe.run_procedure(proc, "yt search"))

        self.assertEqual(captured_windows, ["brave"])

    def test_refocus_called_before_click(self):
        """_ensure_foreground is called before click steps when active_window is set."""
        steps = [
            {"type": "app", "action": "open", "params": {"name": "notepad"}},
            {"type": "app", "action": "click", "params": {"selector": "name:File"}},
        ]
        ps.create_procedure("notepad flow", "Notepad Flow", steps)
        proc = ps.get_procedure("notepad flow")

        with patch("assistant.procedure_executor._execute_app_step",
                   new=AsyncMock(side_effect=["Opened notepad", "Clicked File"])):
            run(pe.run_procedure(proc, "notepad flow"))

        self._fg_mock.assert_called_with("notepad")


class TestWaitForTargetInWindow(unittest.TestCase):
    """Verify that click steps with a window wait PID-scoped, not globally."""

    def test_returns_true_when_element_appears(self):
        mock_aa = MagicMock()
        mock_aa._parse_selector_parts = MagicMock(return_value=("search", None))
        mock_aa.ensure_desktop = MagicMock(return_value="desktop")
        mock_aa._find_element_bounds_in_tree = MagicMock(
            return_value={"x": 10, "y": 20, "width": 5, "height": 5}
        )

        async def _go():
            with (
                patch.dict(sys.modules, {"assistant.automation.native": mock_aa}),
                patch("asyncio.sleep", new_callable=AsyncMock),
            ):
                return await pe._wait_for_target_in_window(
                    "name:search", "chrome", timeout=2.0
                )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_go())
        finally:
            loop.close()
        self.assertTrue(result)
        mock_aa._find_element_bounds_in_tree.assert_called()

    def test_returns_false_on_timeout(self):
        mock_aa = MagicMock()
        mock_aa._parse_selector_parts = MagicMock(return_value=("search", None))
        mock_aa.ensure_desktop = MagicMock(return_value="desktop")
        mock_aa._find_element_bounds_in_tree = MagicMock(return_value=None)
        with (
            patch.dict(sys.modules, {"assistant.automation.native": mock_aa}),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = run(pe._wait_for_target_in_window("name:search", "chrome", timeout=0.5))
        self.assertFalse(result)

    def test_returns_false_on_setup_error(self):
        mock_aa = MagicMock()
        mock_aa._parse_selector_parts = MagicMock(side_effect=RuntimeError("no backend"))
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}):
            result = run(pe._wait_for_target_in_window("name:search", "chrome"))
        self.assertFalse(result)


class TestEnsureForegroundNoOp(unittest.TestCase):
    """Verify _ensure_foreground skips Alt trick when window already active."""

    def test_skips_when_already_foreground(self):
        mock_gw = MagicMock()
        mock_active = MagicMock()
        mock_active.title = "YouTube - Google Chrome"
        mock_gw.getActiveWindow = MagicMock(return_value=mock_active)

        mock_ctypes = MagicMock()
        with patch.dict(sys.modules, {"pygetwindow": mock_gw, "ctypes": mock_ctypes}):
            result = run(pe._ensure_foreground("chrome"))

        self.assertTrue(result)
        # Alt key trick must NOT have been called (would disrupt focused element)
        mock_ctypes.windll.user32.keybd_event.assert_not_called()
        mock_ctypes.windll.user32.SetForegroundWindow.assert_not_called()


class TestClickWithWindowScope(unittest.TestCase):
    """Verify that app click steps pass window context to click_element."""

    def test_click_uses_active_window(self):
        mock_aa = MagicMock()
        mock_aa.click_element = AsyncMock(return_value="Clicked search")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}), \
             patch("assistant.automation.native", mock_aa):
            result = run(pe._execute_app_step(
                {"action": "click", "params": {"selector": "name:search"}},
                active_window="chrome"
            ))
        mock_aa.click_element.assert_called_once_with("name:search", "chrome")
        self.assertEqual(result, "Clicked search")

    def test_click_step_window_overrides_context(self):
        mock_aa = MagicMock()
        mock_aa.click_element = AsyncMock(return_value="Clicked save")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}), \
             patch("assistant.automation.native", mock_aa):
            result = run(pe._execute_app_step(
                {"action": "click", "params": {"selector": "name:Save", "window": "Notepad"}},
                active_window="chrome"
            ))
        mock_aa.click_element.assert_called_once_with("name:Save", "Notepad")

    def test_type_uses_active_window(self):
        mock_aa = MagicMock()
        mock_aa.focus_window = AsyncMock(return_value="Focused chrome")
        mock_aa.type_text = AsyncMock(return_value="Typed text")
        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}), \
             patch("assistant.automation.native", mock_aa):
            result = run(pe._execute_app_step(
                {"action": "type", "params": {"text": "hello"}},
                active_window="chrome"
            ))
        mock_aa.focus_window.assert_called_once_with("chrome")
        mock_aa.type_text.assert_called_once_with("hello", None, "chrome")


if __name__ == "__main__":
    unittest.main(verbosity=2)
