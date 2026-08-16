# tests/test_6b_transport_adapters.py
"""Milestone 6b Task 6 -- the transport registry and adapter contract.

This file's registry half is self-contained: no provider module
(`tailscale.py`, `cloudflare.py`) exists yet, so every test here drives the
registry directly against a test-double adapter rather than a real one.
Tasks 7 and 8 extend this file with the provider-specific tests named in
their own briefs.
"""
from __future__ import annotations

import pytest

from assistant.io.api.transports import TransportRegistry, transport_registry
from assistant.io.api.transports.base import TransportAdapter


class _FakeAdapter:
    """A minimal `TransportAdapter` double: just enough to register."""

    def __init__(self, name: str) -> None:
        self.name = name

    def command(self, port: int) -> list[str]:
        return ["true", str(port)]

    def hostname_from(self, line: str) -> str | None:
        return None

    def preflight(self, port: int) -> str | None:
        return None

    def stop(self, port: int) -> list[str] | None:
        return None


@pytest.fixture(autouse=True)
def _snapshot_registry():
    """Every test gets a clean registry and leaves none of its own fakes
    behind -- the same isolation `test_provider_registry.py` uses for
    `provider_registry`."""
    snapshot = transport_registry.list_all()
    transport_registry.reset()
    yield
    transport_registry.reset()
    for name, adapter in snapshot.items():
        transport_registry.register(name, adapter)


def test_an_adapter_registers_under_its_own_policy_name():
    assert isinstance(transport_registry, TransportRegistry)
    adapter = _FakeAdapter("tailnet")
    transport_registry.register("tailnet", adapter)
    assert transport_registry.get("tailnet") is adapter
    assert isinstance(adapter, TransportAdapter)


def test_registering_a_name_that_is_not_a_policy_is_refused():
    with pytest.raises(ValueError):
        transport_registry.register("carrier_pigeon", _FakeAdapter("carrier_pigeon"))
    assert transport_registry.get("carrier_pigeon") is None


def test_registering_the_same_name_twice_is_refused():
    transport_registry.register("funnel", _FakeAdapter("funnel"))
    with pytest.raises(ValueError):
        transport_registry.register("funnel", _FakeAdapter("funnel"))


def test_local_can_never_be_registered_as_a_transport():
    """KI-17 sibling: `local` is the loopback listener and has no tunnel. An
    adapter claiming that name would be handed local's port -- and local's
    full policy, admin and bearer and every capability included."""
    with pytest.raises(ValueError):
        transport_registry.register("local", _FakeAdapter("local"))
    assert transport_registry.get("local") is None


def test_an_unknown_name_resolves_to_nothing():
    assert transport_registry.get("nope") is None


def test_names_are_stable_and_sorted():
    transport_registry.register("quick", _FakeAdapter("quick"))
    transport_registry.register("funnel", _FakeAdapter("funnel"))
    transport_registry.register("tailnet", _FakeAdapter("tailnet"))
    assert transport_registry.names() == ["funnel", "quick", "tailnet"]
    # Stable: calling again in a different registration order still sorts.
    assert transport_registry.names() == sorted(transport_registry.names())
