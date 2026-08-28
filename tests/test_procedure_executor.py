"""
test_procedure_executor.py — TP-1c: procedure_executor unit tests

Tests variable resolution, error detection, and step routing
with mocked computer_task backends.

Fifteen tests here stub the native-automation backend with
`patch.dict(sys.modules, {"assistant.automation.native": mock})`, and that
form only works by luck. `procedure_executor.py` reaches the backend as
`from .automation import native as app_automation` (four sites), which reads an
ATTRIBUTE on the `assistant.automation` package. `sys.modules` is consulted
only while that attribute does not exist yet:

    >>> hasattr(assistant.automation, "native")     # nothing imported it yet
    False
    >>> with patch.dict(sys.modules, {"assistant.automation.native": mock}):
    ...     from assistant.automation import native
    >>> native is mock
    True

    >>> import assistant.automation.native           # ANY earlier import
    >>> hasattr(assistant.automation, "native")
    True
    >>> with patch.dict(sys.modules, {"assistant.automation.native": mock}):
    ...     from assistant.automation import native
    >>> native is mock
    False                                            # the REAL module

Every import of it inside `procedure_executor.py` is function-local, so running
this file alone leaves the attribute unbound and all fifteen stubs hold. Run it
after anything that imports the real module and they all stop holding at once
-- and the calls behind them are `open_app` (launches an application),
`click_element`, `type_text` and `focus_window`, driven through Terminator on
the live desktop. Safe alone, hijacks the machine in company: the exact shape
`.claude/rules/testing.md` records for `test_repo_preference`, except this one
fails by taking the mouse rather than by going red.

`_native_unbound` below removes the luck. See its docstring.

Three tests in `TestClickWithWindowScope` already did the robust thing -- they
patch the package ATTRIBUTE alongside `sys.modules` -- and they are the reason
those patches carry `create=True`: the fixture unbinds the attribute, so there
is no original for `patch` to save. Their pattern is the one to copy when
adding a test here.

Run: py -3.11 -m pytest tests/test_procedure_executor.py -q
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import assistant.procedures as ps
from assistant.core.abort import UserAborted
from assistant.core.verdict import Outcome, speaks_as_done
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


@pytest.fixture(autouse=True)
def _no_screen_no_network():
    """Make the self-heal path inert for every test in this file.

    A second hazard, and nothing in this file stubs it. `run_procedure` retries
    a failing step `_MAX_RETRIES` times and then calls `_self_heal`, which does:

        from .io import screen as _screen        # package attribute
        from . import llm as _llm                # package attribute
        img = _screen.capture_screenshot_base64()

    So any test that makes a step return an error -- `test_stops_on_error` does
    it deliberately, three times over -- takes a real screenshot of whatever is
    on the operator's display and then spends a real Gemini call describing it.
    Neither is stubbed anywhere, and `patch.dict(sys.modules, ...)` would not
    have reached them if it were: these are the same package-attribute reads as
    `native`, and there is no `patch.dict` here to fix.

    Blanket rather than per-test, because the correct rule for this file is
    absolute: a unit test of procedure routing has no business reading the
    screen or calling a model, so make both impossible instead of enumerating
    which tests happen to trip them. `_self_heal` fails closed on a falsy
    screenshot -- it returns `None`, meaning "not healed" -- so returning
    `None` here exercises the same branch a screenshot failure would, without a
    camera roll of the operator's desktop.
    """
    import assistant as _pkg
    import assistant.io as _io_pkg
    import assistant.io.screen as _real_screen
    import assistant.llm as _real_llm

    inert_screen = MagicMock(name="inert_screen")
    inert_screen.capture_screenshot_base64 = MagicMock(return_value=None)
    inert_llm = MagicMock(name="inert_llm")
    inert_llm.get_vision_response = AsyncMock(
        side_effect=AssertionError(
            "a unit test in test_procedure_executor.py tried to call a model"))

    _io_pkg.screen = inert_screen
    _pkg.llm = inert_llm
    try:
        yield
    finally:
        _io_pkg.screen = _real_screen
        _pkg.llm = _real_llm


@pytest.fixture(autouse=True)
def _no_real_retry_delays():
    """Collapse the retry backoff.

    `_RETRY_DELAYS` is `[0.8, 1.6]` and the error-path tests exhaust it, so each
    one sleeps 2.4 real seconds for no benefit. Patched here rather than in each
    test because no test in this file is about wall-clock behaviour.
    """
    with patch.object(pe, "_RETRY_DELAYS", [0, 0]):
        yield


@pytest.fixture(autouse=True)
def _native_unbound():
    """Unbind `assistant.automation.native` for the duration of every test.

    With the attribute absent, `from .automation import native` falls through to
    `sys.modules` -- so the fifteen `patch.dict` stubs in this file intercept
    regardless of what any earlier test file imported. Without it they are
    correct only when this file runs first, and wrong in a way that reaches the
    real desktop rather than a red assertion.

    Unbinding rather than binding the mock, deliberately: there is one fixture
    instead of fifteen rewritten call sites, and a test that forgets to stub the
    backend still gets the real module (a loud failure) rather than a silently
    shared fake left over from its neighbour.

    Autouse, and it covers `unittest.TestCase` subclasses -- pytest runs autouse
    fixtures around those too, which is what makes one fixture enough here.
    Restores whatever was bound, including nothing.
    """
    import assistant.automation as _auto
    had = hasattr(_auto, "native")
    original = getattr(_auto, "native", None)
    if had:
        delattr(_auto, "native")
    try:
        yield
    finally:
        if had:
            _auto.native = original
        elif hasattr(_auto, "native"):
            # A test caused a real import; leaving that bound would hand the
            # next file a different starting state than this one had.
            delattr(_auto, "native")


def test_the_native_stub_intercepts_only_while_the_attribute_is_unbound():
    """The guard on the fixture above, and the reason it is not optional.

    Both directions, because the first assertion alone would be vacuous -- it
    would pass just as happily if `patch.dict` were reliable, which is the very
    thing in question. The second half binds the attribute the way any earlier
    real import does and shows the same stub being bypassed.

    Not `live_automation`-marked: an ordinary pass is exactly the condition
    being checked for, so it has to run in one.
    """
    import assistant.automation as _auto
    sentinel = MagicMock(name="STUB")

    # ── With the fixture's unbinding in force, the stub wins. ──
    assert not hasattr(_auto, "native"), (
        "the fixture did not unbind the attribute, so the fifteen backend "
        "stubs below are bypassed and reach the real desktop"
    )
    with patch.dict(sys.modules, {"assistant.automation.native": sentinel}):
        from assistant.automation import native as stubbed
    assert stubbed is sentinel, (
        "the native stub was bypassed: `_execute_app_step` would call "
        "open_app / click_element / type_text against the live desktop"
    )

    # ── Bind it, and the identical stub loses. ──
    # A repeat `import` does NOT restore a deleted parent attribute -- the
    # setattr happens once, on first load -- so the binding is done explicitly
    # here rather than by re-importing, which would prove nothing.
    import assistant.automation.native as real_module
    _auto.native = real_module
    try:
        with patch.dict(sys.modules, {"assistant.automation.native": sentinel}):
            from assistant.automation import native as bypassed
        assert bypassed is real_module, (
            "the premise no longer holds: `from .automation import native` now "
            "consults sys.modules even with the attribute bound, which would "
            "make the fixture unnecessary. Verify before deleting it."
        )
    finally:
        delattr(_auto, "native")


def run(coro):
    return asyncio.run(coro)


def run_granted(coro):
    """Run a coroutine with the grant set a real caller of `run_procedure`
    carries.

    `run_procedure` checks EXECUTE itself as of 6a.5 -- it drives
    `automation.native` and `pyautogui` directly and never re-enters
    `actions.execute()`, so nothing downstream of it would otherwise check
    anything, and it is reachable from `scheduler.py` as well as from the turn
    pipeline. Both real callers state their grants (main.py from the turn,
    scheduler.py explicitly); these tests are about step orchestration, so they
    state the same thing rather than testing an unreachable call shape.
    """
    from assistant.actions import LOCAL_GRANTS, current_grants, set_grants

    async def _wrapped():
        token = set_grants(LOCAL_GRANTS)
        try:
            return await coro
        finally:
            current_grants.reset(token)

    return asyncio.run(_wrapped())


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
        return run_granted(pe.run_procedure(proc, text))

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
            run_granted(pe.run_procedure(proc, "yt search"))

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
            run_granted(pe.run_procedure(proc, "notepad flow"))

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
             patch("assistant.automation.native", mock_aa,
                   create=True):   # see _native_unbound: the attribute is unbound
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
             patch("assistant.automation.native", mock_aa,
                   create=True):   # see _native_unbound: the attribute is unbound
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
             patch("assistant.automation.native", mock_aa,
                   create=True):   # see _native_unbound: the attribute is unbound
            result = run(pe._execute_app_step(
                {"action": "type", "params": {"text": "hello"}},
                active_window="chrome"
            ))
        mock_aa.focus_window.assert_called_once_with("chrome")
        mock_aa.type_text.assert_called_once_with("hello", None, "chrome")


# ─── P13: the state machine ──────────────────────────────────────────────────
#
# TENKA-v2 §17.P13, loop 2. This loop kept its step generation, its
# retry-with-backoff, its self-heal escalation and its stop-on-error rule, and
# gave up `_is_error` as its *status vocabulary* while keeping it as the
# definition of the *halt decision*.


# Every shape `run_procedure`'s step loop can put in a result, with the answer
# the pre-P13 substring sniff gave. Written out rather than generated, because
# the point is that a human can read the list and see rows 4, 5 and 7 are
# wrong -- which is not visible in a comprehension.
_SHAPES = [
    # (text, _is_error today)
    ("Opened chrome", False),
    ("Pressed ctrl+s", False),
    ("Waited 1.0s", False),
    ("Navigated to https://example.invalid", False),
    ("Typed: hello", False),
    ("Skipped (browser navigate opens the browser)", False),
    ("Skipped (already logged in)", False),
    # The three the sniff gets wrong. Preserved, not fixed -- see the module
    # note in procedure_executor.py.
    ("Skipped (element not found)", True),
    ("Skipped (page still loading, timeout)", True),
    ("Typed: the file was not found on the server", True),
    # Genuine failures.
    ("Error pressing ctrl+s: boom", True),
    ("Exception: something broke", True),
    ("Unknown step type 'wat'", False),
    ("get_text timed out", True),
    # Self-heal's two verdicts.
    ("click completed (screen-verified)", False),
    ("", False),
]


class TestHaltParity(unittest.TestCase):
    """`_classify` must not change any procedure's course.

    This is P13's "identical externally observable behaviour" as a single
    assertion. The migration's claim is that the halt decision is unchanged and
    only the *recorded verdict* is new, so the two predicates are compared
    directly over every shape rather than sampled.
    """

    def test_halt_decision_matches_is_error_for_every_shape(self):
        for text, was_error in _SHAPES:
            with self.subTest(text=text):
                halts = pe._classify(text) in pe._HALTS
                self.assertEqual(
                    halts, pe._is_error(text),
                    f"_classify disagrees with _is_error on {text!r}: "
                    f"halts={halts}, _is_error={pe._is_error(text)}",
                )

    def test_the_shape_table_is_not_stale(self):
        """The table's second column must match the real `_is_error`.

        Without this the parity test above is comparing `_classify` against a
        hand-written list that could have drifted -- a walk over the wrong
        thing rather than over nothing, which is the same class of vacuous.
        """
        for text, was_error in _SHAPES:
            with self.subTest(text=text):
                self.assertEqual(pe._is_error(text), was_error)

    def test_the_table_covers_every_shape_the_loop_can_produce(self):
        """Anti-vacuity: a table that shrinks proves less every time.

        Not a strong check -- it cannot know about a shape nobody added -- but
        it fails loudly if someone deletes rows to make a failure go away.
        """
        self.assertGreaterEqual(len(_SHAPES), 16)
        self.assertTrue(any(t.startswith("Skipped (") for t, _ in _SHAPES))
        self.assertTrue(any(t.endswith(pe._SCREEN_VERIFIED) for t, _ in _SHAPES))


class TestClassifyIsHonest(unittest.TestCase):
    """The half that is new: what the run *records*, as opposed to what it does."""

    def test_a_clean_result_is_evidence_of_success(self):
        o = pe._classify("Opened chrome")
        self.assertIs(o, Outcome.SUCCEEDED)
        self.assertTrue(o.is_evidence_of_success)

    def test_a_vision_confirmed_step_is_uncertain_not_succeeded(self):
        # The retries all failed and a model looking at a screenshot said it
        # had worked. That is not positive evidence, and the run continuing is
        # a separate question from what it may later claim.
        o = pe._classify(f"click completed {pe._SCREEN_VERIFIED}")
        self.assertIs(o, Outcome.UNCERTAIN)
        self.assertFalse(o.is_evidence_of_success)
        self.assertFalse(speaks_as_done(o))
        self.assertNotIn(o, pe._HALTS)      # but the run still continues

    def test_a_deliberate_skip_is_unverified_and_still_speaks_as_done(self):
        o = pe._classify("Skipped (browser navigate opens the browser)")
        self.assertIs(o, Outcome.UNVERIFIED)
        self.assertFalse(o.is_evidence_of_success)
        self.assertTrue(speaks_as_done(o))   # V6
        self.assertNotIn(o, pe._HALTS)

    def test_an_empty_result_is_uncertain_not_succeeded(self):
        # `_is_error("")` is False so it never halted, but nothing was
        # observed either. This is the "absence of an exception is not
        # evidence" rule at the smallest scale.
        o = pe._classify("")
        self.assertIs(o, Outcome.UNCERTAIN)
        self.assertNotIn(o, pe._HALTS)

    def test_a_skip_whose_reason_trips_the_sniff_is_recorded_as_failed(self):
        """The preserved defect, asserted so it cannot be fixed by accident.

        `_self_heal` spent a vision call to decide SKIP, and the substring
        sniff reads its reason text and halts anyway. The honest answer is
        UNVERIFIED. This test encodes the *old* behaviour on purpose: whoever
        corrects it should have to change this test and read why.
        """
        o = pe._classify("Skipped (element not found)")
        self.assertIs(o, Outcome.FAILED)
        self.assertIn(o, pe._HALTS)


class TestProcedureResult(unittest.TestCase):
    def test_text_is_the_same_joined_string_as_before(self):
        r = pe.ProcedureResult(steps=(
            pe.StepResult(1, "Opened chrome", Outcome.SUCCEEDED),
            pe.StepResult(2, "Typed: hello", Outcome.SUCCEEDED),
        ))
        self.assertEqual(r.text, "Step 1: Opened chrome\nStep 2: Typed: hello")

    def test_outcome_rolls_up_and_uncertainty_wins_over_success(self):
        r = pe.ProcedureResult(steps=(
            pe.StepResult(1, "Opened chrome", Outcome.SUCCEEDED),
            pe.StepResult(2, f"click completed {pe._SCREEN_VERIFIED}",
                          Outcome.UNCERTAIN),
        ))
        self.assertIs(r.outcome, Outcome.UNCERTAIN)
        self.assertFalse(speaks_as_done(r.outcome))

    def test_a_skip_alone_does_not_lower_the_procedure(self):
        r = pe.ProcedureResult(steps=(
            pe.StepResult(1, "Skipped (already logged in)", Outcome.UNVERIFIED),
            pe.StepResult(2, "Opened chrome", Outcome.SUCCEEDED),
        ))
        self.assertIs(r.outcome, Outcome.SUCCEEDED)

    def test_a_message_only_result_is_unsupported(self):
        # A refusal, or a procedure with no steps: nothing was attempted.
        r = pe.ProcedureResult(message="Procedure has no steps.")
        self.assertIs(r.outcome, Outcome.UNSUPPORTED)
        self.assertEqual(r.text, "Procedure has no steps.")
        self.assertFalse(speaks_as_done(r.outcome))

    def test_halted_reports_whether_the_run_stopped_early(self):
        ok = pe.ProcedureResult(steps=(
            pe.StepResult(1, "Opened chrome", Outcome.SUCCEEDED),))
        bad = pe.ProcedureResult(steps=(
            pe.StepResult(1, "Error: boom", Outcome.FAILED),))
        self.assertFalse(ok.halted)
        self.assertTrue(bad.halted)


class TestUserAbortedIsNotAStepFailure(unittest.TestCase):
    """An abort must leave `run_procedure` as an exception.

    `Exception: esc_hold` trips `_is_error`, so before P13 an abort raised
    beneath a step would retry twice with backoff and then spend a vision call
    on `_self_heal` asking the screen whether the thing the user had just
    cancelled had worked. Nothing beneath this module raises `UserAborted`
    today (KI-36 -- it has no abort boundary at all), so this guard is ahead of
    a raiser rather than behind one.
    """

    def test_run_procedure_reraises_and_does_not_self_heal(self):
        proc = {"id": 1, "name": "p", "trigger": "t", "steps": [
            {"type": "app", "action": "open", "params": {"name": "chrome"}},
        ]}
        mock_aa = MagicMock()
        mock_aa.open_app = AsyncMock(side_effect=UserAborted("esc_hold"))
        heal = AsyncMock(return_value="open completed (screen-verified)")

        with patch.dict(sys.modules, {"assistant.automation.native": mock_aa}), \
             patch("assistant.automation.native", mock_aa, create=True), \
             patch.object(pe, "_self_heal", new=heal), \
             patch.object(pe.procedures, "record_usage", MagicMock()):
            with pytest.raises(UserAborted):
                run_granted(pe.run_procedure_detailed(proc, "t"))

        heal.assert_not_awaited()
        # One attempt, not three: the retry loop never saw a halting outcome
        # because it never got to classify one.
        self.assertEqual(mock_aa.open_app.await_count, 1)


class TestTrackerRecordsTheTruth(unittest.TestCase):
    """`main.py`'s replay branch must not hand telemetry a literal.

    The mapping is asserted here rather than in a main.py test because the
    branch it lives in needs a whole turn to reach; what matters is that the
    table exists, is exhaustive, and never answers "success" for a halted run.
    """

    def test_the_map_is_exhaustive_over_outcome(self):
        from assistant import main as main_mod
        self.assertEqual(
            set(main_mod._PROC_OUTCOME_TO_TELEMETRY), set(Outcome),
            "an Outcome member with no row means a KeyError on a live turn",
        )

    def test_a_halted_run_is_never_recorded_as_success(self):
        from assistant import main as main_mod
        for outcome in (Outcome.FAILED, Outcome.UNSUPPORTED):
            with self.subTest(outcome=outcome):
                self.assertNotEqual(
                    main_mod._PROC_OUTCOME_TO_TELEMETRY[outcome], "success")

    def test_uncertain_is_not_collapsed_into_failure(self):
        from assistant import main as main_mod
        self.assertNotIn(
            main_mod._PROC_OUTCOME_TO_TELEMETRY[Outcome.UNCERTAIN],
            ("success", "failure"),
        )

    def test_the_replay_branch_asks_for_the_detailed_result(self):
        """Structural: the replay branch must not go back to the string form.

        `run_procedure` returns prose, and prose is what made the literal
        `"success"` the only available answer. If a later edit swaps the call
        back, the tracker silently loses its source and nothing else notices.

        **Name-free on purpose.** The first draft of this test walked
        `process_text_from_queue`, which P4c reduced to a fifty-line wrapper --
        so it walked a body with no such call and failed on its own
        `walked nothing` assertion. That is the same trap that had left
        `test_6a5_predispatch_gate.test_no_pre_dispatch_branch_saves_under_a_date`
        silently vacuous since P4c; finding it here is what led to fixing it
        there. Anchor on the call, never on the function that happens to hold
        it today.
        """
        import ast
        from pathlib import Path

        src = (Path(__file__).parent.parent / "assistant" / "main.py"
               ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) in (
                "run_procedure", "run_procedure_detailed")
        ]
        assert called, (
            "walked nothing: main.py no longer calls run_procedure in any "
            "form. If replay moved, re-point this sweep at the module it "
            "moved to. An empty walk is not a pass."
        )
        names = {n.func.attr for n in called}
        self.assertEqual(
            names, {"run_procedure_detailed"},
            f"main.py calls {sorted(names)}. The replay branch must use the "
            f"detailed form -- the string form gives the tracker nothing to "
            f"branch on, which is how it came to record a literal 'success'.",
        )

    def test_the_tracker_assignment_reads_the_map_not_a_literal(self):
        """The assignment itself, because asking for the detailed result and
        then ignoring it is a passing test away from the original defect.

        This test exists because of a green mutant. Reverting the assignment to
        `_tracker.action_outcome = "success"` -- the exact line P13 removed --
        left every other test in this class green: they assert the *map's*
        properties, and a map nobody reads has whatever properties you like.
        `.claude/rules/testing.md`: a green mutant is investigated, not
        accepted.

        Name-free like its neighbour: any `<x>.action_outcome = ...` in main.py
        whose value subscripts the mapping. There are a dozen other
        `action_outcome` assignments in the tree for other intents, and this
        asserts one of them is the mapped kind rather than auditing all of them.
        """
        import ast
        from pathlib import Path

        src = (Path(__file__).parent.parent / "assistant" / "main.py"
               ).read_text(encoding="utf-8")
        tree = ast.parse(src)

        assigns = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "action_outcome"
                    for t in n.targets)
        ]
        assert assigns, (
            "walked nothing: no `.action_outcome = ...` assignment in main.py"
        )

        mapped = [
            n for n in assigns
            if isinstance(n.value, ast.Subscript)
            and getattr(n.value.value, "id", None) == "_PROC_OUTCOME_TO_TELEMETRY"
        ]
        self.assertTrue(
            mapped,
            "no `action_outcome` assignment in main.py reads "
            "_PROC_OUTCOME_TO_TELEMETRY. The procedure-replay branch is back to "
            "recording a literal, so a run that halted at step 1 is stored as "
            "a success and telemetry._maybe_mark_correction is told the last "
            "turn worked.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
