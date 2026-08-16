"""The concrete runtime adapts sync facades onto the async protocols."""
import pytest

from assistant.actions.studio_runtime import build_studio_runtime
from assistant.core.capabilities import Capability
from assistant.io.api import runtime as rt

CHAT_ONLY = frozenset({Capability.CHAT_SEND})
A_DEVICE = "device:phone"


class StubDispatch:
    def __init__(self, accept=True, busy=False):
        self.accept = accept
        self.submitted = []
        self.granted = []
        self.principals = []
        self.busy = busy

    async def submit(self, text: str, grants: frozenset, principal: str):
        self.submitted.append(text)
        self.granted.append(grants)
        self.principals.append(principal)
        if not self.accept:
            return ("", "", False, "busy")
        return (f"t{len(self.submitted)}", "c1", True, "")

    async def abort(self) -> bool:
        return True


@pytest.fixture()
def runtime():
    return build_studio_runtime(StubDispatch())


def test_bundle_satisfies_every_protocol(runtime):
    assert isinstance(runtime.chat, rt.ChatRuntime)
    assert isinstance(runtime.memory, rt.MemoryRuntime)
    assert isinstance(runtime.settings, rt.SettingsRuntime)
    assert isinstance(runtime.personality, rt.PersonalityRuntime)


@pytest.mark.asyncio
async def test_chat_send_goes_through_the_dispatch(runtime):
    ref = await runtime.chat.send("what is on my calendar", CHAT_ONLY, A_DEVICE)
    assert ref.accepted is True
    assert ref.turn_id == "t1"


@pytest.mark.asyncio
async def test_chat_send_hands_the_grant_set_to_the_dispatch_unchanged():
    """The runtime is a pass-through here on purpose: narrowing already
    happened in `authenticate()`, and narrowing twice is a second chance to
    narrow differently."""
    dispatch = StubDispatch()
    runtime = build_studio_runtime(dispatch)
    await runtime.chat.send("hello", CHAT_ONLY, A_DEVICE)
    assert dispatch.granted == [CHAT_ONLY]
    # The identity is a pass-through for the same reason and with a
    # sharper consequence: it decides whose confirmations this turn may
    # answer, so rewriting it here would be rewriting who is asking.
    assert dispatch.principals == [A_DEVICE]


@pytest.mark.asyncio
async def test_chat_send_reports_a_refusal_without_raising():
    runtime = build_studio_runtime(StubDispatch(accept=False))
    ref = await runtime.chat.send("hello", CHAT_ONLY, A_DEVICE)
    assert ref.accepted is False
    assert ref.reason == "busy"


@pytest.mark.asyncio
async def test_preferences_map_onto_records_with_history(monkeypatch):
    import assistant.preferences as preferences
    monkeypatch.setattr(preferences, "get_all_preferences", lambda: [
        {"key": "reading_pace", "value": "1.5x", "category": "pacing",
         "confidence": 0.71, "updated_at": "2026-07-02T08:00:00Z"},
    ])
    monkeypatch.setattr(preferences, "get_preference_history", lambda days=365: [
        {"key": "reading_pace", "value": "1.25x", "changed_at": "2026-06-10T08:00:00Z"},
        {"key": "other_key", "value": "x", "changed_at": "2026-06-11T08:00:00Z"},
    ])
    runtime = build_studio_runtime(StubDispatch())
    records = await runtime.memory.preferences()
    assert records[0].key == "reading_pace"
    assert [h.value for h in records[0].history] == ["1.25x"], (
        "history was not filtered to this key"
    )


@pytest.mark.asyncio
async def test_forget_rejects_an_unknown_scope(runtime):
    with pytest.raises(ValueError):
        await runtime.memory.forget("nonsense", "1")


@pytest.mark.asyncio
async def test_forget_knowledge_with_a_non_numeric_id_reports_false_not_raise(runtime):
    """int(item_id) used to raise ValueError straight out of this method for
    a non-numeric route parameter, reaching DELETE /v1/memory/knowledge/{id}
    as a bare 500 -- a bad id is exactly as absent as an id that parses but
    doesn't exist, so it is reported the same way: False, not an exception.
    """
    assert await runtime.memory.forget("knowledge", "not-a-number") is False


@pytest.mark.asyncio
async def test_forget_procedure_with_a_non_numeric_id_reports_false_not_raise(runtime):
    assert await runtime.memory.forget("procedures", "not-a-number") is False


# ─── StatusInfo.busy ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_status_busy_reflects_the_dispatch_not_a_hardcoded_false():
    idle = build_studio_runtime(StubDispatch(busy=False))
    assert (await idle.system.status()).busy is False

    busy_runtime = build_studio_runtime(StubDispatch(busy=True))
    assert (await busy_runtime.system.status()).busy is True


@pytest.mark.asyncio
async def test_status_has_no_version_field(runtime):
    """No real app version constant exists anywhere in the codebase; the
    field was dropped from the wire rather than shipping a fake one."""
    info = await runtime.system.status()
    assert not hasattr(info, "version")


# ─── restore_backup must report a real failure, not always True ───────────
@pytest.mark.asyncio
async def test_restore_backup_reports_false_when_run_restore_raises(runtime, monkeypatch):
    from assistant.io.backup import crypto, orchestrator

    monkeypatch.setattr(crypto, "is_valid_recovery_phrase", lambda phrase: True)

    def _boom(phrase):
        raise RuntimeError("archive is unreadable")

    monkeypatch.setattr(orchestrator, "run_restore", _boom)

    ok = await runtime.system.restore_backup("whatever the phrase is")
    assert ok is False


@pytest.mark.asyncio
async def test_restore_backup_reports_true_when_run_restore_succeeds(runtime, monkeypatch):
    from assistant.io.backup import crypto, orchestrator

    monkeypatch.setattr(crypto, "is_valid_recovery_phrase", lambda phrase: True)
    monkeypatch.setattr(orchestrator, "run_restore", lambda phrase: None)

    ok = await runtime.system.restore_backup("whatever the phrase is")
    assert ok is True


@pytest.mark.asyncio
async def test_an_env_var_owns_a_row_only_when_the_db_has_none(monkeypatch):
    """Precedence is DB -> env -> default. Read core/runtime_config.py before
    changing this: the environment is a fallback here, not an override."""
    from assistant.core import runtime_config
    import assistant.settings as settings_facade

    monkeypatch.setitem(runtime_config.REGISTRY, "camera_enabled", {
        "default": False, "cast": bool,
        "description": "Allow the camera to be opened on request.",
        "needs_restart": True,
    })
    monkeypatch.setenv("CAMERA_ENABLED", "true")

    monkeypatch.setattr(settings_facade, "list_all", lambda: {})
    runtime = build_studio_runtime(StubDispatch())
    row = [r for r in await runtime.settings.all() if r.key == "camera_enabled"][0]
    assert row.source == "env"

    monkeypatch.setattr(settings_facade, "list_all", lambda: {"camera_enabled": False})
    row = [r for r in await runtime.settings.all() if r.key == "camera_enabled"][0]
    assert row.source == "db", "a stored value must win over the environment"


@pytest.mark.asyncio
async def test_a_row_sourced_from_env_is_still_saveable(monkeypatch):
    from assistant.core import runtime_config
    import assistant.settings as settings_facade

    monkeypatch.setitem(runtime_config.REGISTRY, "camera_enabled", {
        "default": False, "cast": bool, "description": "", "needs_restart": True,
    })
    monkeypatch.setenv("CAMERA_ENABLED", "true")
    written: dict = {}
    monkeypatch.setattr(settings_facade, "set",
                        lambda key, value, source="user": written.update({key: value}))

    runtime = build_studio_runtime(StubDispatch())
    outcome = await runtime.settings.save({"camera_enabled": False})
    assert outcome.saved == ["camera_enabled"]
    assert written == {"camera_enabled": False}
    assert outcome.restart_required == ["camera_enabled"]


@pytest.mark.asyncio
async def test_settings_save_refuses_an_unregistered_key(runtime):
    outcome = await runtime.settings.save({"not_a_real_setting": 1})
    assert outcome.rejected["not_a_real_setting"]
    assert outcome.saved == []


# ─── fix wave: env-sourced rows are cast, saved values are type-checked ───
@pytest.mark.asyncio
async def test_an_env_sourced_bool_row_carries_a_real_bool_not_the_raw_string(monkeypatch):
    """The raw os.getenv() string used to ride straight into SettingRow.value
    while `default` kept its real type -- Studio would see `value: "true"`
    next to `default: false` for the same row."""
    from assistant.core import runtime_config
    import assistant.settings as settings_facade

    monkeypatch.setitem(runtime_config.REGISTRY, "camera_enabled", {
        "default": False, "cast": bool,
        "description": "Allow the camera to be opened on request.",
        "needs_restart": True,
    })
    monkeypatch.setenv("CAMERA_ENABLED", "true")
    monkeypatch.setattr(settings_facade, "list_all", lambda: {})

    runtime = build_studio_runtime(StubDispatch())
    row = [r for r in await runtime.settings.all() if r.key == "camera_enabled"][0]
    assert row.source == "env"
    assert row.value is True
    assert isinstance(row.value, bool)


@pytest.mark.asyncio
async def test_an_env_value_that_fails_its_cast_falls_back_to_the_default(monkeypatch):
    from assistant.core import runtime_config
    import assistant.settings as settings_facade

    monkeypatch.setitem(runtime_config.REGISTRY, "followup_timer", {
        "default": 3.0, "cast": float, "description": "", "needs_restart": False,
    })
    monkeypatch.setenv("FOLLOWUP_TIMER", "not-a-number")
    monkeypatch.setattr(settings_facade, "list_all", lambda: {})

    runtime = build_studio_runtime(StubDispatch())
    row = [r for r in await runtime.settings.all() if r.key == "followup_timer"][0]
    assert row.source == "default"
    assert row.value == 3.0


@pytest.mark.asyncio
async def test_save_rejects_a_value_that_does_not_match_the_settings_cast(monkeypatch):
    from assistant.core import runtime_config
    import assistant.settings as settings_facade

    monkeypatch.setitem(runtime_config.REGISTRY, "followup_timer", {
        "default": 3.0, "cast": float, "description": "", "needs_restart": False,
    })
    written: dict = {}
    monkeypatch.setattr(settings_facade, "set",
                        lambda key, value, source="user": written.update({key: value}))

    runtime = build_studio_runtime(StubDispatch())
    outcome = await runtime.settings.save({"followup_timer": "banana"})

    assert outcome.saved == []
    assert "followup_timer" in outcome.rejected
    assert written == {}, "a value that fails its cast must never reach the store"


@pytest.mark.asyncio
async def test_save_rejects_a_string_for_a_toggle_setting(monkeypatch):
    """bool("anything non-empty") never raises -- a bare try/except around
    the cast would accept any non-empty string for a toggle. Must be
    rejected by type, not by whether the cast call itself raises."""
    from assistant.core import runtime_config
    import assistant.settings as settings_facade

    monkeypatch.setitem(runtime_config.REGISTRY, "camera_enabled", {
        "default": False, "cast": bool, "description": "", "needs_restart": True,
    })
    written: dict = {}
    monkeypatch.setattr(settings_facade, "set",
                        lambda key, value, source="user": written.update({key: value}))

    runtime = build_studio_runtime(StubDispatch())
    outcome = await runtime.settings.save({"camera_enabled": "banana"})

    assert outcome.saved == []
    assert "camera_enabled" in outcome.rejected
    assert written == {}


@pytest.mark.asyncio
async def test_no_facade_is_called_on_the_event_loop_thread(monkeypatch):
    """Sync facades must be wrapped; a direct call would block the loop."""
    import threading
    import assistant.preferences as preferences

    calling_threads = []

    def spy():
        calling_threads.append(threading.current_thread().name)
        return []

    monkeypatch.setattr(preferences, "get_all_preferences", spy)
    runtime = build_studio_runtime(StubDispatch())
    await runtime.memory.preferences()
    assert calling_threads and "MainThread" not in calling_threads[0], (
        f"facade ran on {calling_threads}; it must go through asyncio.to_thread"
    )
