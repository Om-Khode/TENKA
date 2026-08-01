"""Tests verifying documentation and config consistency."""


def test_all_handlers_have_intents():
    """Every registered handler must have a corresponding INTENTS entry
    (or be a known alias / internal routing target)."""
    from assistant.config import INTENTS
    import assistant.actions  # noqa: F401  (import triggers handler registration)
    from assistant.actions.registry import tool_registry

    # browser_action, app_action: internal routing targets used by da_handlers,
    #   not user-facing intents (dispatched by automation/router.py)
    known_exceptions = {"browser_action", "app_action"}

    missing = [
        name for name in tool_registry.keys()
        if name not in INTENTS and name not in known_exceptions
    ]

    assert not missing, (
        f"Handlers exist but intents are missing from config.INTENTS: {missing}. "
        f"Add them to INTENTS or document as internal-only."
    )


def test_all_intents_have_handlers():
    """Every INTENTS entry should have a handler registered."""
    from assistant.config import INTENTS
    import assistant.actions  # noqa: F401  (import triggers handler registration)
    from assistant.actions.registry import tool_registry

    # shutdown: intercepted in main.py's pipeline (sets _shutdown_event) before
    #   handler dispatch is reached, so it has no tool_registry entry by design.
    pipeline_intercepted = {"shutdown"}

    missing = [
        i for i in INTENTS
        if not tool_registry.has(i) and i not in pipeline_intercepted
    ]

    assert not missing, (
        f"Intents in config.INTENTS have no handler: {missing}. "
        f"Either add the handler or remove from INTENTS."
    )
