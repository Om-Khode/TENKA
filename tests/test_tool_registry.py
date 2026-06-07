"""Integration tests — verify all intent handlers register correctly."""
import pytest

from assistant.actions.registry import tool_registry


@pytest.fixture(autouse=True)
def _snapshot_registry():
    snapshot = tool_registry.list_all()
    yield
    tool_registry.reset()
    for k, v in snapshot.items():
        tool_registry.register(k, v)


_EXPECTED_INTENTS = [
    "manage_shortcut", "manage_procedure", "manage_schedule", "manage_monitor",
    "create_note", "open_browser", "get_time", "web_search", "browse_url",
    "file_task", "small_talk", "unknown", "computer_task", "read_screen",
    "find_and_click", "code_executor", "memory_query", "store_memory",
    "set_reminder", "cancel_reminder", "planner", "browser_action",
    "app_action", "enroll_voice", "forget_voice", "browser_cdp_setup",
    "camera_look", "meet_face", "recognize_face",
    "forget_face", "start_recording", "stop_recording", "get_recording",
    "summarize_recording", "hide_avatar", "show_avatar",
]


def test_all_intents_registered():
    registered = tool_registry.list_all()
    missing = [i for i in _EXPECTED_INTENTS if i not in registered]
    assert not missing, f"Missing intents: {missing}"


def test_no_unexpected_intents():
    registered = set(tool_registry.keys())
    expected = set(_EXPECTED_INTENTS)
    extra = registered - expected
    assert not extra, f"Unexpected intents: {extra}"


def test_all_handlers_are_callable():
    for name, handler in tool_registry.list_all().items():
        assert callable(handler), f"Handler for '{name}' is not callable"


def test_dispatch_falls_back_to_unknown():
    handler = tool_registry.get("totally_fake_intent")
    assert handler is None
