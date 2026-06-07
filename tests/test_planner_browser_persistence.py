import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset automation.py globals before each test.

    The conftest ``_restore_sys_modules`` fixture pops
    ``assistant.automation.browser.automation`` from sys.modules after each
    test. That creates a split: ``from assistant.automation.browser import
    automation`` returns the cached package-attribute (old module), while
    ``patch('assistant.automation.browser.automation.X')`` triggers a
    re-import (new module).  Globals set on one are invisible to the other.

    Re-syncing sys.modules here ensures every import path resolves to the
    same module object for the duration of the test.
    """
    from assistant.automation.browser import automation as ba
    sys.modules['assistant.automation.browser.automation'] = ba
    ba._planner_page = None
    ba._planner_context = None
    ba._pages = []
    yield
    ba._planner_page = None
    ba._planner_context = None
    ba._pages = []


@pytest.mark.asyncio
async def test_close_planner_page_noop_when_none():
    from assistant.automation.browser import automation as ba
    assert ba._planner_page is None
    await ba.close_planner_page()
    assert ba._planner_page is None


@pytest.mark.asyncio
async def test_close_planner_page_closes_and_clears():
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_ctx = AsyncMock()

    ba._planner_page = mock_page
    ba._planner_context = mock_ctx
    ba._pages = [mock_page]

    await ba.close_planner_page()

    assert ba._planner_page is None
    assert ba._planner_context is None
    assert mock_page not in ba._pages
    mock_ctx.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_planner_page_info_none_when_no_page():
    from assistant.automation.browser import automation as ba
    assert await ba.get_planner_page_info() is None


@pytest.mark.asyncio
async def test_get_planner_page_info_returns_url_and_title():
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org"
    mock_page.title = AsyncMock(return_value="Wikipedia")

    ba._planner_page = mock_page
    info = await ba.get_planner_page_info()

    assert info is not None
    assert info["url"] == "https://en.wikipedia.org"
    assert info["title"] == "Wikipedia"


@pytest.mark.asyncio
async def test_get_planner_page_info_none_when_closed():
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = True

    ba._planner_page = mock_page
    assert await ba.get_planner_page_info() is None


@pytest.mark.asyncio
async def test_max_pages_eviction_skips_planner_page():
    """When _pages is full, eviction should skip the planner page and close the next oldest."""
    from assistant.automation.browser import automation as ba

    planner_page = MagicMock()
    planner_page.is_closed.return_value = False
    planner_ctx = AsyncMock()

    ba._planner_page = planner_page
    ba._planner_context = planner_ctx

    regular_pages = []
    for i in range(4):
        p = MagicMock()
        p.context = AsyncMock()
        regular_pages.append(p)

    ba._pages = [planner_page] + regular_pages

    new_page = MagicMock()
    ba._pages.append(new_page)

    await ba._evict_oldest_page()

    assert planner_page in ba._pages
    assert regular_pages[0] not in ba._pages
    regular_pages[0].context.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_browser_steps_planner_stores_page():
    """When _from_planner=True, the page should be stored in _planner_page and NOT closed."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="page text")
    mock_page.locator = MagicMock()

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    steps = [{"action": "navigate", "params": {"url": "https://example.com"}}]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=True)

    assert ba._planner_page is mock_page
    assert ba._planner_context is mock_ctx
    assert mock_page in ba._pages
    mock_ctx.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_browser_steps_planner_reuses_page():
    """Second call with _from_planner=True should reuse the existing planner page."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "https://wikipedia.org"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="")
    mock_page.locator = MagicMock()
    fill_locator = AsyncMock()
    fill_locator.first = AsyncMock()
    fill_locator.first.fill = AsyncMock()
    mock_page.locator.return_value = fill_locator

    mock_ctx = AsyncMock()
    ba._planner_page = mock_page
    ba._planner_context = mock_ctx
    ba._pages = [mock_page]

    mock_browser = AsyncMock()

    steps = [{"action": "fill", "params": {"selector": "#searchInput", "value": "Python"}}]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=True)

    assert ba._planner_page is mock_page
    mock_browser.new_context.assert_not_awaited()
    mock_ctx.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_browser_steps_non_planner_still_closes():
    """Regular (non-planner) calls should still close the page in finally."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.context = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.close = AsyncMock()
    mock_page.context = mock_ctx

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    steps = [{"action": "navigate", "params": {"url": "https://example.com"}}]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=False)

    assert ba._planner_page is None
    mock_ctx.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_browser_task_url_shortcut_skipped_from_planner():
    """Simple URL goals like 'go to example.com' should NOT take the extract_text shortcut when from planner."""
    from assistant.automation import router

    llm_called = False
    async def mock_llm(prompt, **kwargs):
        nonlocal llm_called
        llm_called = True
        return '[{"action": "navigate", "params": {"url": "https://example.com"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Navigated to example.com") as mock_run, \
         patch('assistant.automation.browser.automation.extract_text', new_callable=AsyncMock) as mock_extract, \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task("go to example.com", mock_llm, _from_planner=True)

    mock_extract.assert_not_awaited()
    assert llm_called
    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get('_from_planner') is True


@pytest.mark.asyncio
async def test_browser_task_injects_page_context():
    """When a planner page exists, its URL+title should be prepended to the LLM prompt."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    captured_prompt = None
    async def mock_llm(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '[{"action": "fill", "params": {"selector": "#search", "value": "Python"}}]'

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org"
    mock_page.title = AsyncMock(return_value="Wikipedia")
    ba._planner_page = mock_page

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Filled #search") as mock_run, \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock, return_value=None), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task("search for Python", mock_llm, _from_planner=True)

    assert captured_prompt is not None
    assert "[Current browser:" in captured_prompt
    assert "Wikipedia" in captured_prompt
    assert "https://en.wikipedia.org" in captured_prompt

    ba._planner_page = None


@pytest.mark.asyncio
async def test_browser_task_no_context_when_no_planner_page():
    """When no planner page exists, the prompt should be unmodified."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    ba._planner_page = None

    captured_prompt = None
    async def mock_llm(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '[{"action": "navigate", "params": {"url": "https://example.com"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done") as mock_run, \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task("go to example.com", mock_llm, _from_planner=True)

    assert captured_prompt is not None
    assert "[Current browser:" not in captured_prompt


@pytest.mark.asyncio
async def test_open_browser_shortcut_skipped_from_planner():
    """'open chrome and go to wikipedia' should NOT launch real browser when from planner."""
    from assistant.automation import router

    llm_called = False
    async def mock_llm(prompt, **kwargs):
        nonlocal llm_called
        llm_called = True
        return '[{"action": "navigate", "params": {"url": "https://wikipedia.org"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Navigated") as mock_run, \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task("open chrome and go to wikipedia.org", mock_llm, _from_planner=True)

    assert llm_called
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_key_excludes_context_injection():
    """Cache save should use the original goal, not the context-injected one."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org"
    mock_page.title = AsyncMock(return_value="Wikipedia")
    ba._planner_page = mock_page

    async def mock_llm(prompt, **kwargs):
        return '[{"action": "fill", "params": {"selector": "#search", "value": "test"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Filled") as mock_run, \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps') as mock_save:
        await router._execute_browser_task("search for test", mock_llm, _from_planner=True)

    mock_save.assert_called_once()
    saved_goal = mock_save.call_args[0][2]
    assert "[Current browser:" not in saved_goal
    assert saved_goal == "search for test"

    ba._planner_page = None


@pytest.mark.asyncio
async def test_execute_plan_keeps_planner_page_open():
    """execute_plan should NOT close the planner page — user needs to see the result."""
    from assistant.actions.planner import planner

    with patch.object(planner, '_generate_plan', new_callable=AsyncMock, return_value=None), \
         patch('assistant.automation.browser.automation.close_planner_page', new_callable=AsyncMock) as mock_close:
        result = await planner.execute_plan("test goal", AsyncMock())

    mock_close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_browser_steps_stale_planner_page_creates_fresh():
    """If _planner_page is closed (crashed), a new page should be created and stored."""
    from assistant.automation.browser import automation as ba

    stale_page = MagicMock()
    stale_page.is_closed.return_value = True
    stale_ctx = AsyncMock()
    ba._planner_page = stale_page
    ba._planner_context = stale_ctx

    fresh_page = AsyncMock()
    fresh_page.url = "https://example.com"
    fresh_page.is_closed = MagicMock(return_value=False)
    fresh_page.goto = AsyncMock()

    fresh_ctx = AsyncMock()
    fresh_ctx.new_page = AsyncMock(return_value=fresh_page)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=fresh_ctx)

    steps = [{"action": "navigate", "params": {"url": "https://example.com"}}]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=True)

    # Stale page should be replaced
    assert ba._planner_page is fresh_page
    assert ba._planner_context is fresh_ctx
    # Stale context should have been cleaned up
    stale_ctx.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_page_full_lifecycle():
    """Simulate: step 1 creates page → step 2 reuses → cleanup closes."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "https://wikipedia.org"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="Wikipedia content")
    fill_locator = AsyncMock()
    fill_locator.first = AsyncMock()
    fill_locator.first.fill = AsyncMock()
    fill_locator.first.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=fill_locator)
    mock_page.title = AsyncMock(return_value="Wikipedia")

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    ok_result = MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=ok_result), \
         patch('assistant.automation.verification.post_verify', return_value=ok_result):

        # Step 1: navigate to Wikipedia
        steps1 = [{"action": "navigate", "params": {"url": "https://wikipedia.org"}}]
        await ba.run_browser_steps(steps1, _from_planner=True)
        assert ba._planner_page is mock_page
        assert mock_page in ba._pages

        # Step 2: fill search (reuses planner page — no navigate)
        steps2 = [{"action": "fill", "params": {"selector": "#searchInput", "value": "Python"}}]
        await ba.run_browser_steps(steps2, _from_planner=True)
        # Should still be the same page
        assert ba._planner_page is mock_page
        # browser.new_context called only once (step 1)
        assert mock_browser.new_context.await_count == 1

        # Cleanup
        await ba.close_planner_page()
        assert ba._planner_page is None
        assert ba._planner_context is None
        assert mock_page not in ba._pages
        mock_ctx.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_page_info_between_steps():
    """get_planner_page_info should return the URL from the page stored in step 1."""
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org/wiki/Main_Page"
    mock_page.title = AsyncMock(return_value="Wikipedia, the free encyclopedia")

    ba._planner_page = mock_page

    info = await ba.get_planner_page_info()
    assert info["url"] == "https://en.wikipedia.org/wiki/Main_Page"
    assert info["title"] == "Wikipedia, the free encyclopedia"

    ba._planner_page = None


# ─── Two-pass DOM scan tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_pass_triggers_when_navigate_plus_interact():
    """When LLM plans navigate+fill without DOM context, two-pass should
    execute navigate first, scan DOM, re-plan remaining steps."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    call_count = 0
    async def mock_llm(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '[{"action": "navigate", "params": {"url": "https://youtube.com"}}, {"action": "fill", "params": {"selector": "input#search", "value": "lofi"}}]'
        return '[{"action": "fill", "params": {"selector": "input#real-search", "value": "lofi"}}]'

    mock_elements = [
        {"s": "input#real-search", "tag": "input", "type": "text", "ph": "Search", "al": "", "text": ""},
    ]

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done") as mock_run, \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock, return_value=mock_elements), \
         patch('assistant.automation.browser.automation.get_planner_page_info', new_callable=AsyncMock, side_effect=[None, {"url": "https://youtube.com", "title": "YouTube"}]), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "open youtube and search for lofi", mock_llm, _from_planner=True
        )

    assert call_count == 2
    assert mock_run.await_count == 2
    first_call_steps = mock_run.call_args_list[0][0][0]
    assert len(first_call_steps) == 1
    assert first_call_steps[0]["action"] == "navigate"
    second_call_steps = mock_run.call_args_list[1][0][0]
    assert any(s.get("action") == "fill" for s in second_call_steps)


@pytest.mark.asyncio
async def test_two_pass_skipped_when_elements_already_injected():
    """Two-pass should NOT trigger when DOM elements were already available."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org"
    mock_page.title = AsyncMock(return_value="Wikipedia")
    ba._planner_page = mock_page

    call_count = 0
    async def mock_llm(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        return '[{"action": "navigate", "params": {"url": "https://en.wikipedia.org"}}, {"action": "fill", "params": {"selector": "input#searchInput", "value": "test"}}]'

    mock_elements = [
        {"s": "input#searchInput", "tag": "input", "type": "text", "ph": "Search Wikipedia", "al": "", "text": ""},
    ]

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done") as mock_run, \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock, return_value=mock_elements), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "search for test on wikipedia", mock_llm, _from_planner=True
        )

    assert call_count == 1
    assert mock_run.await_count == 1


@pytest.mark.asyncio
async def test_two_pass_skipped_for_navigate_only():
    """Two-pass should NOT trigger when the plan is just navigate (no interact)."""
    from assistant.automation import router

    call_count = 0
    async def mock_llm(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        return '[{"action": "navigate", "params": {"url": "https://example.com"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Navigated") as mock_run, \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "go to example.com", mock_llm, _from_planner=True
        )

    assert call_count == 1
    assert mock_run.await_count == 1


# ─── Stale planner page domain mismatch tests ─────────────────────────────

def test_extract_domain_full_url():
    from assistant.automation.router import _extract_domain
    assert _extract_domain("https://www.youtube.com/watch?v=abc") == "youtube.com"
    assert _extract_domain("https://github.com/search") == "github.com"
    assert _extract_domain("https://en.wikipedia.org/wiki/Main") == "en.wikipedia.org"

def test_extract_domain_bare():
    from assistant.automation.router import _extract_domain
    assert _extract_domain("open youtube.com and search") == "youtube.com"
    assert _extract_domain("go to github.com") == "github.com"

def test_extract_domain_none():
    from assistant.automation.router import _extract_domain
    assert _extract_domain("search for Python") is None
    assert _extract_domain("fill in the form") is None

def test_extract_domain_subdomains():
    from assistant.automation.router import _extract_domain
    assert _extract_domain("https://en.wikipedia.org/wiki/Main") == "en.wikipedia.org"
    assert _extract_domain("https://m.youtube.com") == "m.youtube.com"


@pytest.mark.asyncio
async def test_context_skipped_when_planner_page_domain_mismatches_goal():
    """When GitHub page is open but goal targets YouTube, stale context must NOT be injected."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://github.com/search"
    mock_page.title = AsyncMock(return_value="GitHub")
    ba._planner_page = mock_page

    captured_prompt = None
    async def mock_llm(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '[{"action": "navigate", "params": {"url": "https://youtube.com"}}, {"action": "fill", "params": {"selector": "input#search", "value": "lofi"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done") as mock_run, \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock) as mock_elements, \
         patch('assistant.automation.browser.automation.get_planner_page_info', new_callable=AsyncMock, side_effect=[
             {"url": "https://github.com/search", "title": "GitHub"},
             {"url": "https://youtube.com", "title": "YouTube"},
         ]), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "open youtube.com and search for lofi music", mock_llm, _from_planner=True
        )

    assert "[Current browser:" not in captured_prompt
    assert "GitHub" not in captured_prompt

    ba._planner_page = None


@pytest.mark.asyncio
async def test_context_injected_when_planner_page_domain_matches_goal():
    """When Wikipedia page is open and goal targets Wikipedia, context should be injected."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org/wiki/Main_Page"
    mock_page.title = AsyncMock(return_value="Wikipedia")
    ba._planner_page = mock_page

    captured_prompt = None
    async def mock_llm(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '[{"action": "fill", "params": {"selector": "#searchInput", "value": "Python"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done"), \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock, return_value=None), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "search for Python on wikipedia.org", mock_llm, _from_planner=True
        )

    assert "[Current browser:" in captured_prompt
    assert "Wikipedia" in captured_prompt

    ba._planner_page = None


@pytest.mark.asyncio
async def test_context_injected_when_goal_has_no_url():
    """When goal has no URL (e.g. 'search for Python'), context should still be injected."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org"
    mock_page.title = AsyncMock(return_value="Wikipedia")
    ba._planner_page = mock_page

    captured_prompt = None
    async def mock_llm(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '[{"action": "fill", "params": {"selector": "#search", "value": "Python"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done"), \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock, return_value=None), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "search for Python", mock_llm, _from_planner=True
        )

    assert "[Current browser:" in captured_prompt
    assert "Wikipedia" in captured_prompt

    ba._planner_page = None


@pytest.mark.asyncio
async def test_context_injected_when_subdomain_matches():
    """en.wikipedia.org page should match goal targeting wikipedia.org."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://en.wikipedia.org/wiki/Main_Page"
    mock_page.title = AsyncMock(return_value="Wikipedia")
    ba._planner_page = mock_page

    captured_prompt = None
    async def mock_llm(prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '[{"action": "fill", "params": {"selector": "#search", "value": "test"}}]'

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done"), \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock, return_value=None), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "search for test on wikipedia.org", mock_llm, _from_planner=True
        )

    assert "[Current browser:" in captured_prompt
    assert "Wikipedia" in captured_prompt

    ba._planner_page = None


@pytest.mark.asyncio
async def test_domain_mismatch_triggers_two_pass():
    """When domain mismatches, stale context is skipped → _had_elements stays False → two-pass kicks in."""
    from assistant.automation import router
    from assistant.automation.browser import automation as ba

    mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_page.url = "https://github.com/search"
    mock_page.title = AsyncMock(return_value="GitHub")
    ba._planner_page = mock_page

    call_count = 0
    async def mock_llm(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '[{"action": "navigate", "params": {"url": "https://youtube.com"}}, {"action": "fill", "params": {"selector": "input#search", "value": "lofi"}}]'
        return '[{"action": "fill", "params": {"selector": "input#real-search", "value": "lofi"}}]'

    mock_elements = [
        {"s": "input#real-search", "tag": "input", "type": "text", "ph": "Search", "al": "", "text": ""},
    ]

    with patch('assistant.automation.browser.automation.run_browser_steps', new_callable=AsyncMock, return_value="Done") as mock_run, \
         patch('assistant.automation.browser.automation.get_interactive_elements', new_callable=AsyncMock, return_value=mock_elements), \
         patch('assistant.automation.browser.automation.get_planner_page_info', new_callable=AsyncMock, side_effect=[
             {"url": "https://github.com/search", "title": "GitHub"},
             {"url": "https://youtube.com", "title": "YouTube"},
         ]), \
         patch('assistant.automation.step_cache.load_cached_steps', return_value=None), \
         patch('assistant.automation.step_cache.save_cached_steps'):
        result = await router._execute_browser_task(
            "open youtube.com and search for lofi", mock_llm, _from_planner=True
        )

    assert call_count == 2, "Two-pass should re-plan with real DOM elements"
    assert mock_run.await_count == 2, "Two-pass should call run_browser_steps twice (navigate, then interact)"

    ba._planner_page = None


# ─── Post-click overlay retry tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_post_click_retry_on_overlay():
    """After a click, if next step's pre-check fails, wait and retry before giving up."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "https://github.com"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()

    click_locator = AsyncMock()
    click_locator.first = AsyncMock()
    click_locator.first.click = AsyncMock()

    fill_locator = AsyncMock()
    fill_locator.first = AsyncMock()
    fill_locator.first.fill = AsyncMock()

    def make_locator(selector):
        if "search-button" in selector:
            return click_locator
        return fill_locator
    mock_page.locator = MagicMock(side_effect=make_locator)

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    pre_check_calls = 0
    pre_check_fail_first = True

    async def mock_pre_check(step, *, page=None):
        nonlocal pre_check_calls, pre_check_fail_first
        pre_check_calls += 1
        action = step.get("action")
        if action == "fill" and pre_check_fail_first:
            pre_check_fail_first = False
            return MagicMock(ok=False, confidence=1.0, observation="target 'input#overlay-search' not visible before fill")
        return MagicMock(ok=True, confidence=0.0)

    steps = [
        {"action": "click", "params": {"selector": "button#search-button"}},
        {"action": "fill", "params": {"selector": "input#overlay-search", "value": "test"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', side_effect=mock_pre_check), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=True)

    assert "VERIFY_FAILED" not in result
    assert "Clicked" in result
    assert "Filled" in result
    mock_page.wait_for_timeout.assert_awaited_once_with(800)
    assert pre_check_calls == 3  # click pre-check + fill fail + fill retry


@pytest.mark.asyncio
async def test_post_click_retry_still_fails_when_element_absent():
    """If retry after click also fails, VERIFY_FAILED should still be returned."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "https://github.com"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.locator = MagicMock(return_value=AsyncMock())

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    async def mock_pre_check(step, *, page=None):
        action = step.get("action")
        if action == "fill":
            return MagicMock(ok=False, confidence=1.0, observation="target not visible")
        return MagicMock(ok=True, confidence=0.0)

    steps = [
        {"action": "click", "params": {"selector": "button#open-dialog"}},
        {"action": "fill", "params": {"selector": "input#nonexistent", "value": "x"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', side_effect=mock_pre_check), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=True)

    assert "VERIFY_FAILED" in result
    mock_page.wait_for_timeout.assert_awaited_once_with(800)


@pytest.mark.asyncio
async def test_no_retry_when_previous_step_not_click():
    """Pre-check failure after a navigate should NOT trigger the overlay retry."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.locator = MagicMock(return_value=AsyncMock())

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    async def mock_pre_check(step, *, page=None):
        action = step.get("action")
        if action == "fill":
            return MagicMock(ok=False, confidence=1.0, observation="target not visible")
        return MagicMock(ok=True, confidence=0.0)

    steps = [
        {"action": "navigate", "params": {"url": "https://example.com"}},
        {"action": "fill", "params": {"selector": "input#missing", "value": "x"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', side_effect=mock_pre_check), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=True)

    assert "VERIFY_FAILED" in result
    mock_page.wait_for_timeout.assert_not_awaited()


# ─── Bypass VERIFY_FAILED test ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bypass_verify_failed_is_user_friendly():
    """VERIFY_FAILED from a bypass call should be converted to a friendly message."""
    import assistant.actions as _act
    from assistant.actions.da_handlers import handle_planner

    async def mock_execute(intent, params, llm_resp, bridge, _from_planner=False):
        return "VERIFY_FAILED|step=2|tier=pre_check|obs=target 'input#search' not visible before fill"

    with patch.object(_act, 'execute', side_effect=mock_execute):
        from assistant.actions.planner import planner
        with patch.object(planner, 'execute_plan', new_callable=AsyncMock, return_value="__BYPASS__:browser_action"):
            result = await handle_planner(
                {"goal": "search for lofi on youtube"}, "", None
            )

    assert "VERIFY_FAILED" not in result
    assert "not visible" in result or "didn't take" in result


@pytest.mark.asyncio
async def test_bypass_success_is_synthesized():
    """Raw step output from a bypass call should be synthesized, not sent to TTS raw."""
    import assistant.actions as _act
    from assistant.actions.da_handlers import handle_planner

    raw_result = "Navigated to https://www.wikipedia.org/\nFilled input#searchInput\nExtracted Text: Python is a..."

    async def mock_execute(intent, params, llm_resp, bridge, _from_planner=False):
        return raw_result

    with patch.object(_act, 'execute', side_effect=mock_execute):
        from assistant.actions.planner import planner
        with patch.object(planner, 'execute_plan', new_callable=AsyncMock, return_value="__BYPASS__:browser_action"), \
             patch('assistant.llm.contracts.ask_for_synthesis', new_callable=AsyncMock, return_value="I searched Wikipedia for Python programming.") as mock_synth:
            result = await handle_planner(
                {"goal": "go to Wikipedia and search for Python programming"}, "", None
            )

    assert result == "I searched Wikipedia for Python programming."
    mock_synth.assert_awaited_once()
    assert raw_result not in result


# ─── URL sanitization: strip trailing brackets from LLM output ─────────────


@pytest.mark.asyncio
async def test_url_trailing_bracket_stripped():
    """LLM sometimes emits URLs with trailing ] or ) — must be stripped before navigation."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "about:blank"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    steps = [
        {"action": "navigate", "params": {"url": "https://www.fandango.com/]"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        await ba.run_browser_steps(steps, _from_planner=True)

    actual_url = mock_page.goto.call_args[0][0]
    assert actual_url == "https://www.fandango.com/", f"Trailing ] not stripped: {actual_url}"


@pytest.mark.asyncio
async def test_url_trailing_paren_stripped():
    """Trailing ) in URL must be stripped."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "about:blank"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    steps = [
        {"action": "navigate", "params": {"url": "https://www.amctheatres.com)"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        await ba.run_browser_steps(steps, _from_planner=True)

    actual_url = mock_page.goto.call_args[0][0]
    assert actual_url == "https://www.amctheatres.com", f"Trailing ) not stripped: {actual_url}"


@pytest.mark.asyncio
async def test_clean_url_unchanged():
    """A well-formed URL must pass through untouched."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.url = "about:blank"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    steps = [
        {"action": "navigate", "params": {"url": "https://www.fandango.com/movies"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        await ba.run_browser_steps(steps, _from_planner=True)

    actual_url = mock_page.goto.call_args[0][0]
    assert actual_url == "https://www.fandango.com/movies", f"Clean URL was modified: {actual_url}"


# ─── Consent banner dismissal after navigation ────────────────────────────


@pytest.mark.asyncio
async def test_consent_banner_dismissed_after_navigate():
    """After navigating, consent banners must be dismissed before continuing."""
    from assistant.automation.browser import automation as ba

    mock_btn = AsyncMock()
    mock_btn.is_visible = AsyncMock(return_value=True)
    mock_btn.click = AsyncMock()

    mock_locator = MagicMock()
    mock_locator.first = mock_btn

    mock_page = AsyncMock()
    mock_page.url = "about:blank"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    steps = [
        {"action": "navigate", "params": {"url": "https://www.fandango.com"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        await ba.run_browser_steps(steps, _from_planner=True)

    mock_btn.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_consent_banner_no_error():
    """When no consent banner exists, navigation proceeds without error."""
    from assistant.automation.browser import automation as ba

    mock_btn = AsyncMock()
    mock_btn.is_visible = AsyncMock(return_value=False)

    mock_locator = MagicMock()
    mock_locator.first = mock_btn

    mock_page = AsyncMock()
    mock_page.url = "about:blank"
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.evaluate = AsyncMock(return_value=False)

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)

    steps = [
        {"action": "navigate", "params": {"url": "https://example.com"}},
    ]

    with patch.object(ba, 'ensure_browser', return_value=mock_browser), \
         patch('assistant.automation.verification.pre_check', return_value=MagicMock(ok=True, confidence=0.0)), \
         patch('assistant.automation.verification.post_verify', return_value=MagicMock(ok=True, confidence=0.0, tier="ok", skipped=False)):
        result = await ba.run_browser_steps(steps, _from_planner=True)

    assert "Navigated to" in result
    mock_btn.click.assert_not_awaited()


# ─── get_page() URL sanitization ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_page_strips_trailing_bracket():
    """get_page() must strip trailing ] from URLs before navigation."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.close = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)
    mock_browser.is_connected = MagicMock(return_value=True)

    ba._browser = mock_browser
    ba._browser_headless = False

    with patch.object(ba, 'ensure_browser', return_value=mock_browser):
        page = await ba.get_page("https://www.fandango.com/]")

    actual_url = mock_page.goto.call_args[0][0]
    assert actual_url == "https://www.fandango.com/", f"Trailing ] not stripped in get_page: {actual_url}"


@pytest.mark.asyncio
async def test_get_page_clean_url_unchanged():
    """get_page() must not modify well-formed URLs."""
    from assistant.automation.browser import automation as ba

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)

    mock_ctx = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)
    mock_browser.is_connected = MagicMock(return_value=True)

    ba._browser = mock_browser
    ba._browser_headless = True

    with patch.object(ba, 'ensure_browser', return_value=mock_browser):
        await ba.get_page("https://www.example.com/path")

    actual_url = mock_page.goto.call_args[0][0]
    assert actual_url == "https://www.example.com/path"


# ─── get_page() headless mode preservation ───────────────────────────────────


@pytest.mark.asyncio
async def test_get_page_preserves_headed_mode():
    """get_page() must not flip a headed browser to headless."""
    from assistant.automation.browser import automation as ba

    mock_browser = AsyncMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_browser.new_context = AsyncMock(return_value=AsyncMock(
        new_page=AsyncMock(return_value=AsyncMock(
            goto=AsyncMock(), is_closed=MagicMock(return_value=False)
        ))
    ))

    ba._browser = mock_browser
    ba._browser_headless = False

    mock_ensure = AsyncMock(return_value=mock_browser)
    with patch.object(ba, 'ensure_browser', mock_ensure):
        await ba.get_page("https://example.com")

    mock_ensure.assert_awaited_once_with(headless=False)


@pytest.mark.asyncio
async def test_get_page_defaults_headless_when_no_browser():
    """get_page() defaults to headless=True when no browser is running."""
    from assistant.automation.browser import automation as ba

    mock_browser = AsyncMock()
    mock_browser.is_connected = MagicMock(return_value=True)
    mock_browser.new_context = AsyncMock(return_value=AsyncMock(
        new_page=AsyncMock(return_value=AsyncMock(
            goto=AsyncMock(), is_closed=MagicMock(return_value=False)
        ))
    ))

    ba._browser = None
    ba._browser_headless = None

    mock_ensure = AsyncMock(return_value=mock_browser)
    with patch.object(ba, 'ensure_browser', mock_ensure):
        await ba.get_page("https://example.com")

    mock_ensure.assert_awaited_once_with(headless=True)


# ─── Consent banner JS fallback ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consent_banner_js_fallback_removes_overlay():
    """When no consent button is visible, JS fallback removes overlay containers."""
    from assistant.automation.browser import automation as ba

    mock_btn = AsyncMock()
    mock_btn.is_visible = AsyncMock(return_value=False)

    mock_locator = MagicMock()
    mock_locator.first = mock_btn

    mock_page = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.evaluate = AsyncMock(return_value=True)

    await ba._dismiss_consent_banner(mock_page)

    mock_page.evaluate.assert_awaited_once()
    mock_btn.click.assert_not_awaited()
