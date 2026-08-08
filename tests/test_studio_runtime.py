"""The concrete runtime adapts sync facades onto the async protocols."""
import pytest

from assistant.actions.studio_runtime import build_studio_runtime
from assistant.io.api import runtime as rt


class StubDispatch:
    def __init__(self, accept=True):
        self.accept = accept
        self.submitted = []

    async def submit(self, text: str):
        self.submitted.append(text)
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
    ref = await runtime.chat.send("what is on my calendar")
    assert ref.accepted is True
    assert ref.turn_id == "t1"


@pytest.mark.asyncio
async def test_chat_send_reports_a_refusal_without_raising():
    runtime = build_studio_runtime(StubDispatch(accept=False))
    ref = await runtime.chat.send("hello")
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
    await runtime.memory.list("preferences")
    assert calling_threads and "MainThread" not in calling_threads[0], (
        f"facade ran on {calling_threads}; it must go through asyncio.to_thread"
    )
