"""Tests for D5+D6: Browser-name and window-title parsing consolidation.

Verifies:
  - config.BROWSER_NAMES exists and is a frozenset
  - All expected browsers are in the set
  - router._BROWSER_NAMES regex matches all config.BROWSER_NAMES entries
  - router._extract_doc_part correctly splits window titles
  - No inline browser-name lists remain in agent.py
"""

import ast
import inspect
import re
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_heavy_deps():
    stubs = {}
    for mod_name in [
        "assistant.automation.browser.automation",
        "assistant.automation.browser.cdp",
        "assistant.automation.browser.dom_orchestrator",
        "assistant.automation.native",
        "assistant.automation.vision",
        "assistant.automation.vision.agent",
        "assistant.automation.vision.verifier",
        "assistant.automation.vision.todo_classifier",
        "assistant.automation.vision._parsing",
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


# --- config.BROWSER_NAMES ---

def test_browser_names_is_frozenset():
    from assistant import config
    assert isinstance(config.BROWSER_NAMES, frozenset)


def test_browser_names_contains_expected():
    from assistant import config
    expected = {"chrome", "firefox", "edge", "brave", "opera", "safari", "vivaldi", "browser"}
    assert config.BROWSER_NAMES == expected


# --- router._BROWSER_NAMES regex derived from config ---

def test_router_regex_matches_all_config_names():
    from assistant import config
    from assistant.automation.router import _BROWSER_NAMES
    for name in config.BROWSER_NAMES:
        assert _BROWSER_NAMES.search(name), f"regex should match '{name}'"


def test_router_regex_case_insensitive():
    from assistant.automation.router import _BROWSER_NAMES
    assert _BROWSER_NAMES.search("Chrome")
    assert _BROWSER_NAMES.search("FIREFOX")
    assert _BROWSER_NAMES.search("eDgE")


# --- _extract_doc_part (D6) ---

def test_extract_doc_part_standard_title():
    from assistant.automation.router import _extract_doc_part
    doc, clean = _extract_doc_part("Untitled - Notepad")
    assert doc == "Untitled"
    assert clean == "Untitled - Notepad"


def test_extract_doc_part_unsaved_indicator():
    from assistant.automation.router import _extract_doc_part
    doc, clean = _extract_doc_part("*myfile.txt - Notepad")
    assert doc == "myfile.txt"
    assert clean == "myfile.txt - Notepad"


def test_extract_doc_part_no_separator():
    from assistant.automation.router import _extract_doc_part
    doc, clean = _extract_doc_part("Calculator")
    assert doc == "Calculator"
    assert clean == "Calculator"


def test_extract_doc_part_multiple_separators():
    from assistant.automation.router import _extract_doc_part
    doc, clean = _extract_doc_part("My Doc - v2 - Word")
    assert doc == "My Doc - v2"
    assert clean == "My Doc - v2 - Word"


# --- No inline browser lists in agent.py ---

def test_agent_no_inline_browser_lists():
    """agent.py should not contain hardcoded browser name lists/tuples."""
    from assistant.automation.vision import agent
    source = inspect.getsource(agent)
    forbidden = [
        'browser_names = [',
        'browser_keywords = [',
        '"chrome", "firefox", "edge", "brave"',
        '"firefox", "chrome", "brave", "edge"',
    ]
    for pattern in forbidden:
        assert pattern not in source, f"agent.py still contains inline: {pattern}"
