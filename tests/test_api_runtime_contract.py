"""The fake runtime must satisfy the protocols the routes are written against."""
import inspect

import pytest

from assistant.io.api import runtime as rt
from tests.fakes.studio_runtime import build_fake_runtime

PROTOCOLS = {
    "chat": rt.ChatRuntime,
    "memory": rt.MemoryRuntime,
    "settings": rt.SettingsRuntime,
    "personality": rt.PersonalityRuntime,
    "files": rt.FileRuntime,
    "commands": rt.CommandRuntime,
    "system": rt.SystemRuntime,
}


def test_bundle_exposes_every_domain():
    fake = build_fake_runtime()
    for name in PROTOCOLS:
        assert hasattr(fake, name), f"StudioRuntime is missing '{name}'"


@pytest.mark.parametrize("name,protocol", sorted(PROTOCOLS.items()))
def test_fake_implements_every_protocol_method(name, protocol):
    impl = getattr(build_fake_runtime(), name)
    for method in [m for m in dir(protocol) if not m.startswith("_")]:
        assert hasattr(impl, method), f"fake {name} is missing {method}()"


@pytest.mark.parametrize("name,protocol", sorted(PROTOCOLS.items()))
def test_fake_signatures_match_the_protocol(name, protocol):
    impl = getattr(build_fake_runtime(), name)
    for method in [m for m in dir(protocol) if not m.startswith("_")]:
        expected = inspect.signature(getattr(protocol, method))
        actual = inspect.signature(getattr(impl, method))
        assert list(actual.parameters)[0:] == list(expected.parameters)[1:], (
            f"{name}.{method}: fake takes {list(actual.parameters)}, "
            f"protocol declares {list(expected.parameters)[1:]}"
        )


@pytest.mark.parametrize("name,protocol", sorted(PROTOCOLS.items()))
def test_every_protocol_method_is_async(name, protocol):
    """Facades are sync and get wrapped; the protocol surface is uniformly async."""
    for method in [m for m in dir(protocol) if not m.startswith("_")]:
        assert inspect.iscoroutinefunction(getattr(protocol, method)), (
            f"{name}.{method} must be async — routes await everything"
        )


@pytest.mark.asyncio
async def test_fake_memory_forget_removes_the_entity_and_its_facts():
    fake = build_fake_runtime()
    graph = await fake.memory.knowledge()
    victim = graph.entities[0].id
    assert await fake.memory.forget("knowledge", str(victim)) is True
    after = await fake.memory.knowledge()
    assert victim not in [e.id for e in after.entities]
    assert victim not in [f.subject_id for f in after.facts]


@pytest.mark.asyncio
async def test_fake_memory_forget_reports_false_for_unknown_id():
    fake = build_fake_runtime()
    assert await fake.memory.forget("knowledge", "no-such-id") is False


@pytest.mark.asyncio
async def test_fake_graph_keeps_a_superseded_fact_and_a_dangling_edge():
    """Both are properties the Memory page renders. A clean fixture proves nothing."""
    graph = await build_fake_runtime().memory.knowledge()
    assert any(f.invalid_at is not None for f in graph.facts)
    known = {e.id for e in graph.entities}
    assert any(r.to_id not in known for r in graph.relationships)


@pytest.mark.asyncio
async def test_fake_settings_save_persists_within_the_fake():
    fake = build_fake_runtime()
    rows = await fake.settings.all()
    key = rows[0].key
    await fake.settings.save({key: "changed"})
    assert [r for r in await fake.settings.all() if r.key == key][0].value == "changed"


# ─── Files ───────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fake_files_rename_returns_the_entry_and_updates_the_listing():
    fake = build_fake_runtime()
    renamed = await fake.files.rename("desktop/notes.md", "todo.md")
    assert renamed.path == "desktop/todo.md"
    assert renamed.name == "todo.md"
    listing = await fake.files.listing("desktop")
    assert "desktop/todo.md" in [e.path for e in listing]
    assert "desktop/notes.md" not in [e.path for e in listing]


@pytest.mark.asyncio
async def test_fake_files_rename_absent_path_raises():
    fake = build_fake_runtime()
    with pytest.raises(FileNotFoundError):
        await fake.files.rename("desktop/missing.md", "todo.md")


@pytest.mark.asyncio
async def test_fake_files_delete_removes_it_from_the_listing():
    fake = build_fake_runtime()
    assert await fake.files.delete("desktop/notes.md") is True
    listing = await fake.files.listing("desktop")
    assert "desktop/notes.md" not in [e.path for e in listing]


@pytest.mark.asyncio
async def test_fake_files_delete_reports_false_for_a_path_that_was_not_there():
    fake = build_fake_runtime()
    assert await fake.files.delete("desktop/missing.md") is False


# ─── Personality ─────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fake_personality_set_base_changes_state():
    fake = build_fake_runtime()
    changed = await fake.personality.set_base("dry")
    assert changed.base == "dry"
    assert (await fake.personality.state()).base == "dry"


@pytest.mark.asyncio
async def test_fake_personality_reset_puts_the_traits_back():
    fake = build_fake_runtime()
    before = await fake.personality.state()
    assert any(v != 0.5 for v in before.traits.values())
    await fake.personality.set_base("dry")
    after = await fake.personality.reset()
    assert after.base == "warm"
    assert all(v == 0.5 for v in after.traits.values())


# ─── Commands ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fake_commands_run_records_the_run_and_reports_ok():
    fake = build_fake_runtime()
    outcome = await fake.commands.run("volume_up")
    assert outcome.ok is True
    assert fake.commands.ran == ["volume_up"]


@pytest.mark.asyncio
async def test_fake_commands_run_unknown_command_reports_failure_without_recording():
    fake = build_fake_runtime()
    outcome = await fake.commands.run("no_such_command")
    assert outcome.ok is False
    assert fake.commands.ran == []


# ─── System ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fake_system_run_backup_advances_the_state_backup_state_returns():
    fake = build_fake_runtime()
    before = await fake.system.backup_state()
    ran = await fake.system.run_backup()
    assert ran.last_backup_at != before.last_backup_at
    assert (await fake.system.backup_state()).last_backup_at == ran.last_backup_at


@pytest.mark.asyncio
async def test_fake_system_restore_backup_records_the_phrase_and_rejects_wrong_length():
    fake = build_fake_runtime()
    eight_words = "one two three four five six seven eight"
    assert await fake.system.restore_backup(eight_words) is True
    assert await fake.system.restore_backup("too short") is False
    assert fake.system.restored_with == [eight_words, "too short"]


@pytest.mark.asyncio
async def test_fake_system_forget_enrolled_removes_the_voice():
    fake = build_fake_runtime()
    assert await fake.system.forget_enrolled("voice", "v1") is True
    remaining = await fake.system.enrollment()
    assert "v1" not in [v.item_id for v in remaining.voices]


@pytest.mark.asyncio
async def test_fake_system_forget_enrolled_removes_the_face():
    fake = build_fake_runtime()
    assert await fake.system.forget_enrolled("face", "f1") is True
    remaining = await fake.system.enrollment()
    assert "f1" not in [f.item_id for f in remaining.faces]


@pytest.mark.asyncio
async def test_fake_system_forget_enrolled_reports_false_for_unknown_id():
    fake = build_fake_runtime()
    assert await fake.system.forget_enrolled("voice", "no-such-id") is False


@pytest.mark.asyncio
async def test_fake_system_forget_enrolled_reports_false_for_unknown_kind():
    fake = build_fake_runtime()
    assert await fake.system.forget_enrolled("gadget", "v1") is False
