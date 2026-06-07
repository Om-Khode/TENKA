"""
test_url_recon.py — Unit tests for _url_recon and _tavily_recon_search.

Covers:
  - Gating: goals already containing a URL return None immediately
  - Extraction: best URL is returned from search results
  - Domain preference: goal keyword match beats list order
  - First-result fallback: used when no keyword matches
  - Search delegation: _tavily_recon_search is called with the goal text
  - Planner goal threading: executor injects plan.original_goal
  - Cache self-healing: planner invalidates cache on semantic failure
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _run(coro):
    return asyncio.run(coro)


# ─── Stubs for heavy modules that router.py triggers on import ───────────────

def _stub_heavy_modules():
    """Patch modules that trigger real I/O or GPU usage on import.

    For pure-utility modules (json_utils, known_apps) we prefer the real
    module — they're lightweight and stubbing them leaks the stub into
    sibling test files via sys.modules and breaks downstream imports like
    `from ..core.json_utils import extract_json_object`. Only fall back to
    a stub when the real import fails (e.g. missing optional dependency).
    """
    stubs = {
        "assistant.core.known_apps": {
            "KNOWN_APPS": {},
            "get_apps_by_category": lambda cat: [],
        },
        "assistant.core.json_utils": {
            "extract_json_array": lambda text: [],
        },
        "assistant.core.geolocation": {
            "get_cached_region": lambda: None,
            "format_region_hint": lambda r: "",
        },
    }

    import importlib
    for name, attrs in stubs.items():
        if name in sys.modules:
            # Real or stubbed module already loaded — only fill in missing attrs.
            for attr, val in attrs.items():
                if not hasattr(sys.modules[name], attr):
                    setattr(sys.modules[name], attr, val)
            continue
        try:
            importlib.import_module(name)  # prefer real module
        except Exception:
            mod = types.ModuleType(name)
            for attr, val in attrs.items():
                setattr(mod, attr, val)
            sys.modules[name] = mod


_stub_heavy_modules()


# _url_recon must be re-resolved per-call. Other test files (notably
# test_runtime_config) call _load_real_package() which deletes ALL
# `assistant.*` entries from sys.modules — that invalidates any reference
# we captured at module-import time, while @patch("assistant.automation.
# router._tavily_recon_search") still patches the freshly-imported module.
# Result: the stale _url_recon calls the original (unpatched) function and
# makes real network requests. Re-resolving via sys.modules every call keeps
# the binding in sync with whatever @patch is patching.
def _url_recon(*args, **kwargs):
    from assistant.automation.router import _url_recon as _fn
    return _fn(*args, **kwargs)


# ─── TestUrlReconGating ───────────────────────────────────────────────────────


class TestUrlReconGating(unittest.TestCase):

    def test_skip_when_goal_already_has_url(self):
        goal = "book tickets on https://example.com for tomorrow"
        result = _run(_url_recon(goal))
        self.assertIsNone(result)

    def test_skip_when_goal_has_www_url(self):
        goal = "open www.bookmyshow.com and find movies"
        result = _run(_url_recon(goal))
        self.assertIsNone(result)

    def test_skip_when_goal_has_domain(self):
        goal = "go to bookmyshow.com and search for avengers"
        result = _run(_url_recon(goal))
        self.assertIsNone(result)


# ─── TestUrlReconExtraction ───────────────────────────────────────────────────


class TestUrlReconExtraction(unittest.TestCase):

    @patch("assistant.automation.router._tavily_recon_search")
    def test_returns_best_url_from_results(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "BookMyShow", "url": "https://in.bookmyshow.com/movies", "content": "Book tickets"},
            {"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Cinema", "content": "Cinema info"},
        ])

        result = _run(_url_recon("book movie tickets tonight"))
        self.assertIsNotNone(result)
        self.assertIn("bookmyshow", result)

    @patch("assistant.automation.router._tavily_recon_search")
    def test_returns_none_when_no_results(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[])

        result = _run(_url_recon("find upcoming concerts near me"))
        self.assertIsNone(result)

    @patch("assistant.automation.router._tavily_recon_search")
    def test_returns_none_when_search_fails(self, mock_search):
        async def _raise(*args, **kwargs):
            raise RuntimeError("network error")

        mock_search.side_effect = _raise

        result = _run(_url_recon("search for movie showtimes"))
        self.assertIsNone(result)


# ─── TestUrlReconDomainPreference ─────────────────────────────────────────────


class TestUrlReconDomainPreference(unittest.TestCase):

    @patch("assistant.automation.router._tavily_recon_search")
    def test_prefers_domain_matching_goal_keyword(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "District", "url": "https://district.in/events", "content": "Events"},
            {"title": "BookMyShow", "url": "https://in.bookmyshow.com/movies", "content": "Movies"},
        ])

        result = _run(_url_recon("book tickets on bookmyshow for avengers"))
        self.assertIsNotNone(result)
        self.assertIn("bookmyshow", result)

    @patch("assistant.automation.router._tavily_recon_search")
    def test_falls_back_to_first_when_no_domain_match(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "SiteA", "url": "https://sitea.example.com/page", "content": "Info"},
            {"title": "SiteB", "url": "https://siteb.example.com/page", "content": "Info"},
        ])

        result = _run(_url_recon("buy tickets for the show tonight"))
        self.assertIsNotNone(result)
        self.assertEqual(result, "https://sitea.example.com/page")


# ─── TestUrlReconSearchQuery ──────────────────────────────────────────────────


class TestUrlReconSearchQuery(unittest.TestCase):

    @patch("assistant.automation.router._tavily_recon_search")
    def test_search_uses_step_goal_when_no_planner_goal(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[])
        goal = "find cheap flights to goa"

        _run(_url_recon(goal))

        call_query = mock_search.call_args[0][0]
        self.assertIn("goa", call_query.lower())

    @patch("assistant.core.geolocation.get_cached_region", return_value={"city": "Berlin", "country": "India"})
    @patch("assistant.automation.router._tavily_recon_search")
    def test_search_uses_planner_goal_with_city(self, mock_search, _mock_geo):
        mock_search.side_effect = AsyncMock(return_value=[])
        step_goal = "navigate to BookMyShow website"
        planner_goal = "book spiderman upcoming movie ticket for 2 people in nearby theater"

        _run(_url_recon(step_goal, planner_goal=planner_goal))

        call_query = mock_search.call_args[0][0]
        self.assertIn("spiderman", call_query.lower())
        self.assertIn("berlin", call_query.lower())
        self.assertNotIn("navigate", call_query.lower())

    @patch("assistant.core.geolocation.get_cached_region", return_value={"city": "Berlin"})
    @patch("assistant.automation.router._tavily_recon_search")
    def test_city_not_duplicated_if_already_in_goal(self, mock_search, _mock_geo):
        mock_search.side_effect = AsyncMock(return_value=[])
        goal = "book movie tickets in Berlin"

        _run(_url_recon(goal))

        call_query = mock_search.call_args[0][0]
        self.assertEqual(call_query.lower().count("berlin"), 1)


# ─── Integration Tests: _url_recon inside _execute_browser_task ──────────────


def _make_browser_stub(*, planner_page_info=None, interactive_elements=None,
                       run_result="Done"):
    """Create a stub module for assistant.automation.browser.automation."""
    mod = types.ModuleType("assistant.automation.browser.automation")
    mod.PLAYWRIGHT_AVAILABLE = True
    mod.run_browser_steps = AsyncMock(return_value=run_result)
    mod.get_planner_page_info = AsyncMock(return_value=planner_page_info)
    mod.get_interactive_elements = AsyncMock(return_value=interactive_elements or [])
    return mod


def _make_step_cache_stub(*, cached_steps=None):
    """Create a stub module for assistant.automation.step_cache."""
    mod = types.ModuleType("assistant.automation.step_cache")
    mod.load_cached_steps = MagicMock(return_value=cached_steps)
    mod.save_cached_steps = MagicMock()
    mod.delete_cached_steps = MagicMock()
    return mod


class TestReconIntegrationUrlInjected(unittest.TestCase):
    """When _url_recon returns a URL, it should appear in the LLM prompt."""

    def test_recon_url_injected_into_goal(self):
        ba_stub = _make_browser_stub()
        sc_stub = _make_step_cache_stub(cached_steps=None)

        saved = {}
        for key in ("assistant.automation.browser.automation",
                     "assistant.automation.browser",
                     "assistant.automation.step_cache"):
            saved[key] = sys.modules.get(key)

        # Ensure the browser package exists for `from .browser import automation`
        browser_pkg = types.ModuleType("assistant.automation.browser")
        browser_pkg.automation = ba_stub

        try:
            sys.modules["assistant.automation.browser.automation"] = ba_stub
            sys.modules["assistant.automation.browser"] = browser_pkg
            sys.modules["assistant.automation.step_cache"] = sc_stub

            from assistant.automation.router import (
                _execute_browser_task, _maybe_await, _extract_json_array,
            )

            recon_url = "https://in.bookmyshow.com/movies"
            captured_prompt = {}

            async def fake_llm(prompt, **kwargs):
                captured_prompt["value"] = prompt
                return "[]"

            steps_from_llm = [
                {"action": "navigate", "params": {"url": recon_url}},
                {"action": "extract_text", "params": {}},
            ]

            with patch("assistant.automation.router._url_recon",
                        new=AsyncMock(return_value=recon_url)), \
                 patch("assistant.automation.router._extract_json_array",
                        return_value=steps_from_llm), \
                 patch("assistant.automation.router._maybe_await",
                        new=AsyncMock(return_value="[]")):

                # Capture the prompt passed to _maybe_await
                async def capture_maybe_await(func, prompt, **kw):
                    captured_prompt["value"] = prompt
                    return "[]"

                with patch("assistant.automation.router._maybe_await",
                            new=capture_maybe_await):
                    _run(_execute_browser_task(
                        "book movie tickets tonight", fake_llm,
                    ))

            self.assertIn(recon_url, captured_prompt.get("value", ""))
        finally:
            for key, orig in saved.items():
                if orig is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = orig


class TestReconIntegrationSkippedOnCacheHit(unittest.TestCase):
    """When cached steps exist, _url_recon must not be called."""

    def test_recon_skipped_on_cache_hit(self):
        cached = [
            {"action": "navigate", "params": {"url": "https://example.com"}},
        ]
        ba_stub = _make_browser_stub(run_result="Done — cached")
        sc_stub = _make_step_cache_stub(cached_steps=cached)

        saved = {}
        for key in ("assistant.automation.browser.automation",
                     "assistant.automation.browser",
                     "assistant.automation.step_cache"):
            saved[key] = sys.modules.get(key)

        browser_pkg = types.ModuleType("assistant.automation.browser")
        browser_pkg.automation = ba_stub

        try:
            sys.modules["assistant.automation.browser.automation"] = ba_stub
            sys.modules["assistant.automation.browser"] = browser_pkg
            sys.modules["assistant.automation.step_cache"] = sc_stub

            from assistant.automation.router import _execute_browser_task

            mock_recon = AsyncMock(return_value="https://recon.example.com")

            async def fake_llm(prompt, **kwargs):
                return "[]"

            with patch("assistant.automation.router._url_recon", new=mock_recon):
                result = _run(_execute_browser_task(
                    "book movie tickets tonight", fake_llm,
                ))

            mock_recon.assert_not_called()
            self.assertIn("Done", result)
        finally:
            for key, orig in saved.items():
                if orig is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = orig


class TestReconIntegrationSkippedWhenGoalHasUrl(unittest.TestCase):
    """When the goal already contains a URL, _url_recon must not fire."""

    def test_recon_skipped_when_goal_has_url(self):
        ba_stub = _make_browser_stub()
        sc_stub = _make_step_cache_stub(cached_steps=None)

        saved = {}
        for key in ("assistant.automation.browser.automation",
                     "assistant.automation.browser",
                     "assistant.automation.step_cache"):
            saved[key] = sys.modules.get(key)

        browser_pkg = types.ModuleType("assistant.automation.browser")
        browser_pkg.automation = ba_stub

        try:
            sys.modules["assistant.automation.browser.automation"] = ba_stub
            sys.modules["assistant.automation.browser"] = browser_pkg
            sys.modules["assistant.automation.step_cache"] = sc_stub

            from assistant.automation.router import _execute_browser_task

            mock_recon = AsyncMock(return_value="https://recon.example.com")

            steps_from_llm = [
                {"action": "navigate", "params": {"url": "https://in.bookmyshow.com"}},
                {"action": "extract_text", "params": {}},
            ]

            async def fake_llm(prompt, **kwargs):
                return "[]"

            with patch("assistant.automation.router._url_recon", new=mock_recon), \
                 patch("assistant.automation.router._extract_json_array",
                        return_value=steps_from_llm), \
                 patch("assistant.automation.router._maybe_await",
                        new=AsyncMock(return_value="[]")):
                _run(_execute_browser_task(
                    "go to https://in.bookmyshow.com and find movies", fake_llm,
                ))

            mock_recon.assert_not_called()
        finally:
            for key, orig in saved.items():
                if orig is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = orig


class TestReconIntegrationSkippedWhenPlannerPageLoaded(unittest.TestCase):
    """When planner page is loaded (matching domain), _url_recon must not fire."""

    def test_recon_skipped_when_planner_page_loaded(self):
        page_info = {
            "url": "https://in.bookmyshow.com/movies",
            "title": "BookMyShow — Movies",
        }
        ba_stub = _make_browser_stub(
            planner_page_info=page_info,
            interactive_elements=[],
        )
        sc_stub = _make_step_cache_stub(cached_steps=None)

        saved = {}
        for key in ("assistant.automation.browser.automation",
                     "assistant.automation.browser",
                     "assistant.automation.step_cache"):
            saved[key] = sys.modules.get(key)

        browser_pkg = types.ModuleType("assistant.automation.browser")
        browser_pkg.automation = ba_stub

        try:
            sys.modules["assistant.automation.browser.automation"] = ba_stub
            sys.modules["assistant.automation.browser"] = browser_pkg
            sys.modules["assistant.automation.step_cache"] = sc_stub

            from assistant.automation.router import _execute_browser_task

            mock_recon = AsyncMock(return_value="https://in.bookmyshow.com")

            steps_from_llm = [
                {"action": "click", "params": {"selector": "text=Movies"}},
            ]

            async def fake_llm(prompt, **kwargs):
                return "[]"

            with patch("assistant.automation.router._url_recon", new=mock_recon), \
                 patch("assistant.automation.router._extract_json_array",
                        return_value=steps_from_llm), \
                 patch("assistant.automation.router._maybe_await",
                        new=AsyncMock(return_value="[]")):
                _run(_execute_browser_task(
                    "find movies on bookmyshow", fake_llm,
                    _from_planner=True,
                ))

            mock_recon.assert_not_called()
        finally:
            for key, orig in saved.items():
                if orig is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = orig


# ─── Task 3: Browser plan prompt mentions Target URL ─────────────────────────


class TestBrowserPlanPromptRecon(unittest.TestCase):

    def test_prompt_mentions_target_url(self):
        from assistant.automation.router import _BROWSER_PLAN_PROMPT
        self.assertIn("Target URL", _BROWSER_PLAN_PROMPT)


# ─── Task 4: TTS hygiene — VERIFY_FAILED detection ──────────────────────────


class TestPlannerTtsHygiene(unittest.TestCase):

    def test_step_failed_strips_selectors(self):
        _stub_planner_modules()
        from assistant.actions.planner.planner import _step_failed
        raw = (
            'VERIFY_FAILED|step=7|tier=pre_check|obs=target '
            '"input#search-input" not visible before fill'
        )
        self.assertTrue(_step_failed(raw))

    def test_verify_failed_prefix_detected(self):
        _stub_planner_modules()
        from assistant.actions.planner.planner import _step_failed
        self.assertTrue(_step_failed("VERIFY_FAILED|step=3|tier=pre_check|obs=something"))


def _stub_planner_modules():
    planner_stubs = [
        "assistant.llm",
        "assistant.llm.contracts",
        "assistant.llm.router",
        "assistant.storage",
        "assistant.storage.db",
        "assistant.core",
        "assistant.core.config",
        "assistant.automation",
        "assistant.automation.verification",
        "assistant.actions.planner.executor",
        "assistant.actions.planner.pseudo_tools",
    ]
    # NOTE: "assistant.actions" is intentionally NOT stubbed — tests below
    # do `from assistant.actions.planner.planner import _step_failed`,
    # which requires `assistant.actions` to be a real package (with __path__).
    # A bare ModuleType stub breaks submodule resolution.
    for name in planner_stubs:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    contracts = sys.modules["assistant.llm.contracts"]
    if not hasattr(contracts, "ask_for_intent"):
        contracts.ask_for_intent = AsyncMock(return_value="unknown")
    av = sys.modules["assistant.automation.verification"]
    if not hasattr(av, "parse_verify_failed"):
        av.parse_verify_failed = MagicMock(return_value=None)
    if not hasattr(av, "format_failure_for_user"):
        av.format_failure_for_user = MagicMock(return_value="")
    # executor.py does `from .pseudo_tools import run_synthesize_step, …` at
    # call-time. Populate the stub with no-op AsyncMocks so the import succeeds.
    pseudo = sys.modules["assistant.actions.planner.pseudo_tools"]
    for fn in ("run_synthesize_step", "run_vision_analyze_step",
               "run_camera_preview_step", "run_prompt_user_step"):
        if not hasattr(pseudo, fn):
            setattr(pseudo, fn, AsyncMock(return_value=""))


# ─── Task 5: Edge cases ─────────────────────────────────────────────────────


class TestUrlReconEdgeCases(unittest.TestCase):

    def test_empty_goal_returns_none(self):
        result = _run(_url_recon(""))
        self.assertIsNone(result)

    @patch("assistant.automation.router._tavily_recon_search")
    def test_results_with_empty_urls_skipped(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "No URL", "url": "", "content": ""},
            {"title": "Has URL", "url": "https://example.com", "content": ""},
        ])
        result = _run(_url_recon("find something interesting"))
        self.assertEqual(result, "https://example.com")

    @patch("assistant.automation.router._tavily_recon_search")
    def test_all_empty_urls_returns_none(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "No URL 1", "url": "", "content": ""},
            {"title": "No URL 2", "content": ""},
        ])
        result = _run(_url_recon("find something"))
        self.assertIsNone(result)

    @patch("assistant.automation.router._tavily_recon_search")
    def test_domain_hint_case_insensitive(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "Random", "url": "https://random.com/page", "content": ""},
            {"title": "Amazon", "url": "https://www.AMAZON.in/product/123", "content": ""},
        ])
        result = _run(_url_recon("buy headphones on amazon"))
        self.assertEqual(result, "https://www.AMAZON.in/product/123")

    @patch("assistant.automation.router._tavily_recon_search")
    def test_short_words_not_used_as_domain_hints(self, mock_search):
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "Booking", "url": "https://www.booking.com/hotel", "content": ""},
            {"title": "Hotels", "url": "https://www.hotels.com/search", "content": ""},
        ])
        result = _run(_url_recon("book a hotel room for 2 people"))
        self.assertEqual(result, "https://www.booking.com/hotel")

    @patch("assistant.automation.router._tavily_recon_search")
    def test_matches_hostname_not_path(self, mock_search):
        """Words in the goal match against hostname only, not URL path."""
        mock_search.side_effect = AsyncMock(return_value=[
            {"title": "Generic Travel", "url": "https://generic.com/flights/goa", "content": ""},
            {"title": "Goa Airlines", "url": "https://goaair.example.com/booking", "content": ""},
        ])
        # "goa" appears in the PATH of result 1 but in the HOSTNAME of result 2
        result = _run(_url_recon("cheap flights goa tomorrow"))
        self.assertEqual(result, "https://goaair.example.com/booking")


# ─── TestPlannerGoalThreading ────────────────────────────────────────────────


class TestPlannerGoalThreading(unittest.TestCase):
    """Verify executor threads plan.original_goal into params as _planner_goal."""

    def _stub_executor_deps(self):
        """Ensure modules needed by executor are available."""
        _stub_planner_modules()
        sys.modules.pop("assistant.actions.planner.executor", None)
        # executor calls _snapshot_pending_states → assistant.pending. Also
        # assistant.actions.__init__ does `from ..pending import PendingState,
        # pending_registry` at import time, so the stub must expose both names
        # for the actions package to load.
        if "assistant.pending" not in sys.modules:
            pending_mod = types.ModuleType("assistant.pending")
            pending_mod.pending_registry = MagicMock()
            pending_mod.pending_registry.snapshot = MagicMock(return_value={})
            pending_mod.PendingState = MagicMock()
            sys.modules["assistant.pending"] = pending_mod
        else:
            # Existing stub: make sure both symbols are present.
            existing = sys.modules["assistant.pending"]
            if not hasattr(existing, "PendingState"):
                existing.PendingState = MagicMock()
            if not hasattr(existing, "pending_registry"):
                existing.pending_registry = MagicMock()
                existing.pending_registry.snapshot = MagicMock(return_value={})

    def test_executor_injects_planner_goal_for_browser_action(self):
        self._stub_executor_deps()

        from assistant.actions.planner.planner import PlanStep, Plan

        step = PlanStep(step_id=1, tool="browser_action",
                        goal="navigate to BookMyShow website")
        plan = Plan(
            original_goal="book spiderman movie ticket for 2 people",
            steps=[step],
        )

        captured_params = {}

        async def fake_execute(intent, params, llm_response, bridge, _from_planner):
            captured_params.update(params)
            return "done"

        import assistant.actions as _actions_mod
        _actions_mod.execute = fake_execute

        from assistant.actions.planner.executor import execute_step
        _run(execute_step(step, plan, llm_func=AsyncMock()))

        self.assertEqual(
            captured_params.get("_planner_goal"),
            "book spiderman movie ticket for 2 people",
        )

    def test_executor_does_not_inject_for_non_browser_tools(self):
        self._stub_executor_deps()

        from assistant.actions.planner.planner import PlanStep, Plan

        step = PlanStep(step_id=1, tool="web_search", goal="search for movies")
        plan = Plan(
            original_goal="book spiderman movie ticket",
            steps=[step],
        )

        captured_params = {}

        async def fake_execute(intent, params, llm_response, bridge, _from_planner):
            captured_params.update(params)
            return "done"

        import assistant.actions as _actions_mod
        _actions_mod.execute = fake_execute

        from assistant.actions.planner.executor import execute_step
        _run(execute_step(step, plan, llm_func=AsyncMock()))

        self.assertNotIn("_planner_goal", captured_params)


# ─── Cache self-healing: planner invalidates on semantic failure ─────────────

class TestCacheSelfHealingOnSemanticFailure(unittest.TestCase):
    """When planner detects _step_failed, browser cache is invalidated."""

    def _stub_executor_deps(self):
        _stub_planner_modules()
        sys.modules.pop("assistant.actions.planner.executor", None)
        # assistant.actions.__init__ does `from ..pending import PendingState,
        # pending_registry`; expose both names on the stub.
        if "assistant.pending" not in sys.modules:
            pending_mod = types.ModuleType("assistant.pending")
            pending_mod.pending_registry = MagicMock()
            pending_mod.pending_registry.snapshot = MagicMock(return_value={})
            pending_mod.PendingState = MagicMock()
            sys.modules["assistant.pending"] = pending_mod
        else:
            existing = sys.modules["assistant.pending"]
            if not hasattr(existing, "PendingState"):
                existing.PendingState = MagicMock()
            if not hasattr(existing, "pending_registry"):
                existing.pending_registry = MagicMock()
                existing.pending_registry.snapshot = MagicMock(return_value={})

    def test_browser_cache_deleted_on_step_failed(self):
        """When planner marks browser_action as failed, cache entry is deleted."""
        self._stub_executor_deps()

        from assistant.actions.planner.planner import PlanStep, Plan

        step = PlanStep(step_id=1, tool="browser_action",
                        goal="search for Spiderman on BookMyShow")
        plan = Plan(
            original_goal="book spiderman movie ticket",
            steps=[step],
        )

        # Simulate a result that _step_failed detects (contains "no results")
        async def fake_execute(intent, params, llm_response, bridge, _from_planner):
            return "Navigated to site\nExtracted Text: No results found."

        import assistant.actions as _actions_mod
        _actions_mod.execute = fake_execute

        mock_delete = MagicMock()
        with patch("assistant.automation.step_cache.delete_cached_steps", mock_delete):
            from assistant.actions.planner.executor import execute_step
            _run(execute_step(step, plan, llm_func=AsyncMock()))

        self.assertEqual(step.status, "failed")
        mock_delete.assert_called_once_with(
            "browser", "browser", "search for Spiderman on BookMyShow"
        )

    def test_cache_not_deleted_on_success(self):
        """Cache is NOT deleted when browser_action succeeds."""
        self._stub_executor_deps()

        from assistant.actions.planner.planner import PlanStep, Plan

        step = PlanStep(step_id=1, tool="browser_action",
                        goal="navigate to example.com")
        plan = Plan(
            original_goal="open example site",
            steps=[step],
        )

        async def fake_execute(intent, params, llm_response, bridge, _from_planner):
            return "Navigated to https://example.com. Page loaded successfully."

        import assistant.actions as _actions_mod
        _actions_mod.execute = fake_execute

        mock_delete = MagicMock()
        with patch("assistant.automation.step_cache.delete_cached_steps", mock_delete):
            from assistant.actions.planner.executor import execute_step
            _run(execute_step(step, plan, llm_func=AsyncMock()))

        self.assertEqual(step.status, "success")
        mock_delete.assert_not_called()

    def test_cache_not_deleted_for_non_browser_tools(self):
        """Non-browser tools don't trigger cache invalidation."""
        self._stub_executor_deps()

        from assistant.actions.planner.planner import PlanStep, Plan

        step = PlanStep(step_id=1, tool="web_search",
                        goal="find nearby theaters")
        plan = Plan(
            original_goal="find theaters",
            steps=[step],
        )

        async def fake_execute(intent, params, llm_response, bridge, _from_planner):
            return "No results found."

        import assistant.actions as _actions_mod
        _actions_mod.execute = fake_execute

        mock_delete = MagicMock()
        with patch("assistant.automation.step_cache.delete_cached_steps", mock_delete):
            from assistant.actions.planner.executor import execute_step
            _run(execute_step(step, plan, llm_func=AsyncMock()))

        self.assertEqual(step.status, "failed")
        mock_delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
