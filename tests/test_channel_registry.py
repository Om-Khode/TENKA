"""Tests for the messaging channel registry."""
import pytest
from assistant.io.channels import channel_registry
from assistant.io.channels.base import Channel


@pytest.fixture(autouse=True)
def _snapshot_registry():
    snapshot = channel_registry.list_all()
    yield
    channel_registry.reset()
    for k, v in snapshot.items():
        channel_registry.register(k, v)


def test_whatsapp_registered():
    assert channel_registry.has("whatsapp")


def test_whatsapp_implements_protocol():
    wa = channel_registry.require("whatsapp")
    assert isinstance(wa, Channel)
    assert wa.name == "whatsapp"


def test_channel_has_required_methods():
    wa = channel_registry.require("whatsapp")
    assert callable(getattr(wa, "send", None))
    assert callable(getattr(wa, "start", None))
    assert callable(getattr(wa, "stop", None))
    assert callable(getattr(wa, "execute", None))
