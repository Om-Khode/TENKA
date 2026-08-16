"""Every pending handler must be reachable from main.py's dispatch table.

Registering a PendingState in actions/__init__.py and writing its handler is
only two thirds of the wiring — main.py's _PENDING_HANDLERS table is what the
dispatch loop actually walks. A handler missing from it is never called: the
user's reply falls through to normal intent classification and the pending
state silently expires. That is exactly how backup onboarding shipped
unreachable, so it gets a test rather than a convention.
"""
import inspect

import assistant.actions as actions_module
import assistant.main as main_module


# Handlers deliberately dispatched outside the table, with where they run.
# Each one is still asserted to be referenced somewhere in main.py.
_DISPATCHED_ELSEWHERE = {
    "handle_pending_teaching": "called directly in process_text_from_queue()",
}


def _exported_pending_handlers() -> dict:
    return {
        name: obj
        for name, obj in vars(actions_module).items()
        if name.startswith("handle_pending_") and callable(obj)
    }


def test_every_pending_handler_is_dispatched():
    table_funcs = {entry[0] for entry in main_module._PENDING_HANDLERS}
    main_source = inspect.getsource(main_module)

    unreachable = []
    for name, func in _exported_pending_handlers().items():
        if func in table_funcs:
            continue
        if name in _DISPATCHED_ELSEWHERE and name in main_source:
            continue
        unreachable.append(name)

    assert not unreachable, (
        f"pending handlers never reached by main.py's dispatch: {unreachable}. "
        "Add a tuple to main._PENDING_HANDLERS."
    )


def test_backup_handlers_are_in_the_table():
    """The specific regression: backup onboarding could never complete."""
    names = {entry[0].__name__ for entry in main_module._PENDING_HANDLERS}
    assert {
        "handle_pending_backup_confirm_phrase",
        "handle_pending_backup_oauth",
        "handle_pending_backup_unlock_phrase",
        "handle_pending_backup_restore_phrase",
    } <= names


def test_pending_table_entries_are_well_formed():
    from assistant.core.capabilities import Capability

    for entry in main_module._PENDING_HANDLERS:
        handler, label, mem_intent, needs_bridge, required = entry
        assert inspect.iscoroutinefunction(handler), handler
        assert isinstance(label, str) and label
        assert isinstance(mem_intent, str) and mem_intent
        assert isinstance(needs_bridge, bool)
        # The fifth column is what the handler's effect costs; the dispatch
        # loop skips a row the turn's grants do not cover.
        assert isinstance(required, Capability), (handler.__name__, required)

        # needs_bridge must match the handler's real signature: the dispatch
        # loop calls handler(text) or handler(text, bridge), and a mismatch is
        # a TypeError at runtime, on a live turn.
        sig = inspect.signature(handler)
        args = ("text", None) if needs_bridge else ("text",)
        try:
            sig.bind(*args)
        except TypeError as exc:
            raise AssertionError(
                f"{handler.__name__} is registered with needs_bridge="
                f"{needs_bridge} but its signature {sig} rejects that call: {exc}"
            ) from exc


def test_memory_intents_are_real_intents():
    """The dispatch loop writes mem_intent into the conversations table; a
    typo there quietly corrupts intent history."""
    from assistant import config

    extra = {"oauth_setup", "device_auth", "messaging_disambig", "messaging_send",
             "incoming_message", "knowledge_approval"}
    for _handler, _label, mem_intent, _needs_bridge, _cap in main_module._PENDING_HANDLERS:
        assert mem_intent in config.INTENTS or mem_intent in extra, mem_intent
