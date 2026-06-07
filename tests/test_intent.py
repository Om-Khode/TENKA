"""
test_intent.py — Unit tests for intent post-correction guards.

Run: python -m pytest tests/test_intent.py -v
"""

import sys
import types

# ─── Stubs ──────────────────────────────────────────────────────────────────
# Stub heavy dependencies so the real intent.py can import without side effects.
# We must stub config and llm BEFORE importing intent, since intent does
# `from . import config` and `from . import llm` at module level.

_orig_config = sys.modules.get("assistant.config")
_orig_llm = sys.modules.get("assistant.llm")

_config_stub = types.ModuleType("assistant.config")
_config_stub.BROWSER_NAMES = frozenset({
    "chrome", "firefox", "edge", "brave", "opera", "safari", "vivaldi", "browser",
})
_config_stub.INTENTS = frozenset({
    "small_talk", "unknown", "computer_task", "code_executor",
    "find_and_click", "browse_url", "web_search", "get_time",
    "set_reminder", "open_browser", "read_screen",
    "memory_query", "file_task", "store_memory",
})
sys.modules["assistant.config"] = _config_stub

_llm_stub = types.ModuleType("assistant.llm")
sys.modules["assistant.llm"] = _llm_stub

import assistant.intent as intent_mod

IntentResult = intent_mod.IntentResult
_post_correct = intent_mod._post_correct_intent

# ─── Restore sys.modules so other test files import the real modules ─────────
# intent_mod's local `config`/`llm` bindings were resolved at the import above,
# so restoring sys.modules now does NOT change behavior inside this file —
# tests below still see the stubbed config via intent_mod. But sibling test
# files that pytest collects after this one (test_da_url_routing, etc.) now
# import the real assistant.config + assistant.llm.
if _orig_config is None:
    sys.modules.pop("assistant.config", None)
else:
    sys.modules["assistant.config"] = _orig_config
if _orig_llm is None:
    sys.modules.pop("assistant.llm", None)
else:
    sys.modules["assistant.llm"] = _orig_llm


def _make(intent: str, text: str, params: dict | None = None) -> IntentResult:
    return IntentResult(intent=intent, response=text, params=params or {"goal": text})


# ─── Guard 3: explicit browser mention → computer_task ──────────────────────

def test_guard3_search_weather_on_chrome():
    r = _post_correct(_make("code_executor", "Search weather in Berlin on Chrome"), "Search weather in Berlin on Chrome")
    assert r.intent == "computer_task"


def test_guard3_search_for_on_firefox():
    r = _post_correct(_make("code_executor", "search for recipes on firefox"), "search for recipes on firefox")
    assert r.intent == "computer_task"


def test_guard3_check_scores_in_edge():
    r = _post_correct(_make("code_executor", "check cricket scores in edge"), "check cricket scores in edge")
    assert r.intent == "computer_task"


def test_guard3_browse_url_on_brave():
    r = _post_correct(_make("browse_url", "open news on brave"), "open news on brave")
    assert r.intent == "computer_task"


def test_guard3_no_browser_stays_code_executor():
    """Without a browser mention, code_executor should stay as-is."""
    r = _post_correct(_make("code_executor", "what is my CPU usage"), "what is my CPU usage")
    assert r.intent == "code_executor"


def test_guard3_search_no_browser_stays():
    r = _post_correct(_make("code_executor", "search weather in Berlin"), "search weather in Berlin")
    assert r.intent == "code_executor"


# ─── Guard 1: find_and_click + app → computer_task ──────────────────────────

def test_guard1_find_and_click_with_app():
    r = _post_correct(_make("find_and_click", "click play on spotify"), "click play on spotify")
    assert r.intent == "computer_task"


# ─── Guard 2: code_executor + GUI verb + app → computer_task ────────────────

def test_guard2_click_play_on_spotify():
    r = _post_correct(_make("code_executor", "click play on spotify"), "click play on spotify")
    assert r.intent == "computer_task"


def test_guard2_no_gui_verb_stays():
    r = _post_correct(_make("code_executor", "play music on spotify"), "play music on spotify")
    assert r.intent == "code_executor"


# ─── Guard 4: system query → code_executor (Bug 6) ────────────────────────

def test_guard4_list_wifis():
    r = _post_correct(_make("file_task", "list all the wifis I have connected with"), "list all the wifis I have connected with")
    assert r.intent == "code_executor"


def test_guard4_show_bluetooth_devices():
    r = _post_correct(_make("file_task", "show bluetooth devices"), "show bluetooth devices")
    assert r.intent == "code_executor"


def test_guard4_check_battery():
    r = _post_correct(_make("computer_task", "check battery level"), "check battery level")
    assert r.intent == "code_executor"


def test_guard4_scan_wifi_networks():
    r = _post_correct(_make("unknown", "scan for wifi networks"), "scan for wifi networks")
    assert r.intent == "code_executor"


def test_guard4_no_false_positive_on_file_wifi():
    """'find the file wifi_config.txt' should stay file_task."""
    r = _post_correct(_make("file_task", "find the file wifi_config.txt"), "find the file wifi_config.txt")
    assert r.intent == "file_task"


# ─── Guard 5: personal recall → memory_query (Bug 9) ──────────────────────

def test_guard5_food_restrictions():
    r = _post_correct(_make("web_search", "do I have any food restrictions"), "do I have any food restrictions")
    assert r.intent == "memory_query"


def test_guard5_wifi_password_recall():
    r = _post_correct(_make("code_executor", "what's my wifi password"), "what's my wifi password")
    assert r.intent == "memory_query"


def test_guard5_what_did_i_tell_you():
    r = _post_correct(_make("web_search", "what did I tell you about my diet"), "what did I tell you about my diet")
    assert r.intent == "memory_query"


def test_guard5_no_override_time_sensitive():
    """'what's the current weather' should stay web_search even with 'my' nearby."""
    r = _post_correct(_make("web_search", "what's the current weather in my city"), "what's the current weather in my city")
    assert r.intent == "web_search"


def test_guard5_no_override_real_code_task():
    """'what's my IP address' is a live system query, not memory recall."""
    r = _post_correct(_make("code_executor", "what's my IP address"), "what's my IP address")
    assert r.intent == "code_executor"


# ─── Bug 10: store_memory in planner TOOL_MANIFEST ─────────────────────────

def test_store_memory_in_planner_manifest():
    from assistant.actions.planner.planner import TOOL_MANIFEST
    assert "store_memory" in TOOL_MANIFEST
    assert TOOL_MANIFEST["store_memory"]["param_key"] == "content"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
