"""Tests for the event source registry."""
import pytest
from assistant.automation.event_sources import source_registry
from assistant.automation.event_sources.base import EventSource


@pytest.fixture(autouse=True)
def _snapshot_registry():
    snapshot = source_registry.list_all()
    yield
    source_registry.reset()
    for k, v in snapshot.items():
        source_registry.register(k, v)


def test_media_source_registered():
    assert source_registry.has("smtc")


def test_window_source_registered():
    assert source_registry.has("window")


def test_sources_implement_protocol():
    for name, source in source_registry.list_all().items():
        assert isinstance(source, EventSource), f"{name} doesn't implement EventSource"
        assert hasattr(source, "name")
        assert hasattr(source, "event_types")
        assert isinstance(source.event_types, frozenset)


def test_event_types_correct():
    smtc = source_registry.require("smtc")
    assert smtc.event_types == frozenset({"media_changed"})
    window = source_registry.require("window")
    assert window.event_types == frozenset({"window_focus", "window_title"})


def test_all_event_types():
    all_types = set()
    for source in source_registry.list_all().values():
        all_types |= source.event_types
    assert all_types == {"media_changed", "window_focus", "window_title"}
