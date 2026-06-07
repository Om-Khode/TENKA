"""Sanity tests for assistant.pending — PendingState + PendingRegistry."""
import time

import pytest

from assistant.pending import PendingRegistry, PendingState


def test_inactive_by_default():
    s = PendingState[dict]("test", timeout=10.0)
    assert s.payload is None
    assert s.active is False
    assert s.age == 0.0


def test_set_then_payload_returned():
    s = PendingState[dict]("test", timeout=10.0)
    s.set({"key": "value"})
    assert s.active is True
    assert s.payload == {"key": "value"}
    assert s.age >= 0.0
    assert s.age < 0.5  # just set


def test_clear_resets_state():
    s = PendingState[dict]("test", timeout=10.0)
    s.set({"k": 1})
    s.clear()
    assert s.payload is None
    assert s.active is False


def test_expiry_auto_clears_on_read():
    s = PendingState[dict]("test", timeout=0.05)
    s.set({"k": 1})
    assert s.active is True
    time.sleep(0.1)
    # Reading after expiry returns None and clears internally.
    assert s.payload is None
    assert s.active is False
    assert s._payload is None  # internal state was cleared


def test_touch_extends_lifetime():
    s = PendingState[dict]("test", timeout=0.1)
    s.set({"k": 1})
    time.sleep(0.06)
    s.touch()
    time.sleep(0.06)
    # Without touch this would have expired (total 0.12s > 0.1 timeout);
    # with touch the second sleep is only 0.06s past the touch.
    assert s.active is True


def test_touch_does_nothing_when_inactive():
    s = PendingState[dict]("test", timeout=10.0)
    s.touch()  # should not raise or set anything
    assert s.active is False
    assert s._ts == 0.0


def test_payload_mutation_persists():
    s = PendingState[dict]("test", timeout=10.0)
    s.set({"k": 1})
    s.payload["k"] = 2  # in-place dict mutation
    assert s.payload == {"k": 2}


def test_registry_register_and_get():
    reg = PendingRegistry()
    s = PendingState[dict]("a", timeout=5.0)
    reg.register(s)
    assert reg.get("a") is s
    assert reg.get("missing") is None


def test_registry_register_returns_state_for_chaining():
    reg = PendingRegistry()
    s = reg.register(PendingState[dict]("a", timeout=5.0))
    assert isinstance(s, PendingState)
    assert s.name == "a"


def test_registry_rejects_duplicates():
    reg = PendingRegistry()
    reg.register(PendingState[dict]("a", timeout=5.0))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(PendingState[dict]("a", timeout=5.0))


def test_registry_snapshot_reflects_active_state():
    reg = PendingRegistry()
    a = reg.register(PendingState[dict]("a", timeout=5.0))
    b = reg.register(PendingState[dict]("b", timeout=5.0))
    assert reg.snapshot() == {"a": False, "b": False}
    a.set({"x": 1})
    assert reg.snapshot() == {"a": True, "b": False}


def test_registry_names():
    reg = PendingRegistry()
    reg.register(PendingState[dict]("alpha", timeout=1.0))
    reg.register(PendingState[dict]("beta", timeout=1.0))
    assert set(reg.names()) == {"alpha", "beta"}
