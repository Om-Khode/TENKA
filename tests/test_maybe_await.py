"""Tests for D8: _maybe_await helper in automation/router.py.

Verifies:
  - Async functions are awaited
  - Sync functions are called directly
  - Args and kwargs are forwarded correctly
"""

import asyncio
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_router_deps():
    """Stub heavy dependencies so we can import router.py without side effects."""
    stubs = {}
    for mod_name in [
        "assistant.automation.browser.automation",
        "assistant.automation.browser.cdp",
        "assistant.automation.browser.dom_orchestrator",
        "assistant.automation.native",
        "assistant.automation.vision",
        "assistant.automation.vision.agent",
        "assistant.io.screen",
        "assistant.llm",
        "assistant.intent",
        "playwright.async_api",
        "pyautogui",
    ]:
        if mod_name not in sys.modules:
            stubs[mod_name] = types.ModuleType(mod_name)
            sys.modules[mod_name] = stubs[mod_name]
    yield
    for mod_name, mod in stubs.items():
        if sys.modules.get(mod_name) is mod:
            del sys.modules[mod_name]


def test_maybe_await_calls_sync():
    from assistant.automation.router import _maybe_await

    def sync_fn(a, b, kw=None):
        return (a, b, kw)

    result = asyncio.get_event_loop().run_until_complete(_maybe_await(sync_fn, 1, 2, kw="x"))
    assert result == (1, 2, "x")


def test_maybe_await_awaits_async():
    from assistant.automation.router import _maybe_await

    async def async_fn(a, b, kw=None):
        return (a, b, kw)

    result = asyncio.get_event_loop().run_until_complete(_maybe_await(async_fn, 1, 2, kw="y"))
    assert result == (1, 2, "y")


def test_maybe_await_no_args():
    from assistant.automation.router import _maybe_await

    def no_args():
        return "ok"

    result = asyncio.get_event_loop().run_until_complete(_maybe_await(no_args))
    assert result == "ok"


def test_no_duplicate_iscoroutinefunction_in_callers():
    """Ensure no call sites still use the inline iscoroutinefunction pattern."""
    import inspect
    from assistant.automation import router
    source = inspect.getsource(router)
    lines = source.splitlines()
    for i, line in enumerate(lines, 1):
        if "iscoroutinefunction" in line and "def _maybe_await" not in lines[i - 2 : i + 1]:
            if "def _maybe_await" not in line and "async def _maybe_await" not in line:
                assert "if asyncio.iscoroutinefunction(func):" in line, \
                    f"Unexpected iscoroutinefunction usage at line {i}: {line.strip()}"
