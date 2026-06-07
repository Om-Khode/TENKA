"""Tests for D1+D9: core/json_utils.py — consolidated JSON extraction.

Verifies:
  - sanitize_json: fences, think tags, unicode, trailing commas
  - recover_truncated_json: unterminated strings, unbalanced braces
  - extract_json_object: code fences, brace-depth, repair mode
  - extract_json_array: direct parse, fences, [{ preference, trailing commas
  - No local JSON extractors remain in consumer modules
"""

import json
import inspect

from assistant.core.json_utils import (
    sanitize_json,
    recover_truncated_json,
    extract_json_object,
    extract_json_array,
)


# --- sanitize_json ---

def test_sanitize_strips_code_fence():
    assert sanitize_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_sanitize_strips_open_fence():
    assert sanitize_json('```json\n{"a": 1}') == '{"a": 1}'


def test_sanitize_strips_think_tags():
    result = sanitize_json('<think>reasoning</think>{"a": 1}')
    assert result == '{"a": 1}'


def test_sanitize_fixes_unicode_quotes():
    result = sanitize_json('“hello”')
    assert result == '"hello"'


def test_sanitize_fixes_unicode_dashes():
    result = sanitize_json('key—value')
    assert result == 'key-value'


def test_sanitize_removes_trailing_commas():
    result = sanitize_json('{"a": 1, "b": 2,}')
    assert result == '{"a": 1, "b": 2}'


def test_sanitize_empty():
    assert sanitize_json("") == ""
    assert sanitize_json(None) is None


# --- recover_truncated_json ---

def test_recover_balanced_unchanged():
    assert recover_truncated_json('{"a": 1}') == '{"a": 1}'


def test_recover_unclosed_brace():
    result = recover_truncated_json('{"a": 1')
    assert result.endswith("}")
    json.loads(result)


def test_recover_unclosed_string():
    result = recover_truncated_json('{"a": "hello')
    assert '"hello"' in result
    assert result.endswith("}")
    json.loads(result)


def test_recover_unclosed_array():
    result = recover_truncated_json('[1, 2, 3')
    assert result.endswith("]")
    json.loads(result)


def test_recover_trailing_comma():
    result = recover_truncated_json('{"a": 1,')
    assert result.endswith("}")
    json.loads(result)


def test_recover_nested():
    result = recover_truncated_json('{"a": {"b": [1, 2')
    assert result.count("}") >= 2
    assert result.count("]") >= 1
    json.loads(result)


def test_recover_empty():
    assert recover_truncated_json("") == ""
    assert recover_truncated_json(None) is None


# --- extract_json_object ---

def test_extract_object_direct():
    result = extract_json_object('{"intent": "open_browser"}')
    assert result is not None
    assert json.loads(result)["intent"] == "open_browser"


def test_extract_object_with_prose():
    result = extract_json_object('Here is the result: {"a": 1} and more text')
    assert result is not None
    assert json.loads(result) == {"a": 1}


def test_extract_object_code_fence():
    result = extract_json_object('```json\n{"a": 1}\n```')
    assert result is not None
    assert json.loads(result) == {"a": 1}


def test_extract_object_nested():
    result = extract_json_object('{"a": {"b": {"c": 1}}}')
    assert result is not None
    assert json.loads(result)["a"]["b"]["c"] == 1


def test_extract_object_none_for_no_json():
    assert extract_json_object("no json here") is None
    assert extract_json_object("") is None


def test_extract_object_with_sanitize():
    text = '```json\n{“key”: “value”}\n```'
    result = extract_json_object(text, sanitize=True)
    assert result is not None
    assert json.loads(result)["key"] == "value"


def test_extract_object_with_repair():
    result = extract_json_object('{"key": "truncated value', repair=True)
    assert result is not None
    parsed = json.loads(result)
    assert "key" in parsed


def test_extract_object_repair_off_returns_none():
    result = extract_json_object('{"key": "truncated value')
    assert result is None


# --- extract_json_array ---

def test_extract_array_direct():
    result = extract_json_array('[{"action": "click"}]')
    assert result == [{"action": "click"}]


def test_extract_array_code_fence():
    result = extract_json_array('```json\n[1, 2, 3]\n```')
    assert result == [1, 2, 3]


def test_extract_array_with_prose():
    result = extract_json_array('Here: [{"a": 1}] done')
    assert result == [{"a": 1}]


def test_extract_array_prefers_object_array():
    result = extract_json_array('[sarcastic] here is [{"step": 1}]')
    assert result == [{"step": 1}]


def test_extract_array_trailing_commas():
    result = extract_json_array('[{"a": 1,}, {"b": 2,},]')
    assert len(result) == 2


def test_extract_array_empty_for_no_json():
    assert extract_json_array("no json") == []
    assert extract_json_array("") == []


def test_extract_array_with_sanitize():
    text = '<think>planning</think>[{"tool": "web_search"}]'
    result = extract_json_array(text, sanitize=True)
    assert result == [{"tool": "web_search"}]


# --- No local extractors in consumers ---

def test_intent_no_local_extract_json():
    from assistant import intent
    source = inspect.getsource(intent)
    assert "def _extract_json(" not in source


def test_router_no_local_extract_json_array():
    import sys, types
    for mod in ["assistant.automation.browser.automation", "assistant.automation.browser.cdp",
                "assistant.automation.browser.dom_orchestrator", "assistant.automation.native",
                "assistant.automation.vision", "assistant.automation.vision.agent",
                "assistant.io.screen", "assistant.llm", "assistant.intent",
                "playwright.async_api", "pyautogui"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    from assistant.automation import router
    source = inspect.getsource(router)
    assert "def _extract_json_array(" not in source


def test_code_executor_no_local_extract():
    from assistant import code_executor
    source = inspect.getsource(code_executor)
    assert "def _extract_json_from_llm(" not in source
    assert "def _sanitize_llm_json(" not in source
