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
