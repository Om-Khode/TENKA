"""Tests that manage_backup is fully wired into intent detection."""


def test_manage_backup_in_intents_list():
    from assistant import config
    assert "manage_backup" in config.INTENTS


def test_manage_backup_has_registered_handler():
    import assistant.actions  # noqa: F401 — triggers handler registration
    from assistant.actions.registry import tool_registry
    assert tool_registry.get("manage_backup") is not None


def test_manage_backup_in_intent_system_prompt_catalogue():
    from assistant import config
    assert "manage_backup" in config.INTENT_SYSTEM_PROMPT
