"""Milestone 6b Task 2, spec §3.2-3.3 — the in-memory record a ceiling raise

lives in. `RaiseStore` supplies the `raised` argument `effective()` (Task 1)
folds in; this file never touches `policy.py` or `effective()` itself, only
the store's own grant/read/drop lifecycle.
"""
import logging
import time

import pytest

from assistant.core.capabilities import Capability
from assistant.io.api import raises as raises_module
from assistant.io.api.raises import MAX_RAISE_SECONDS, RaiseGrant, RaiseStore


def _fake_clock(monkeypatch, start=1_000.0):
    """Replace `raises.time.monotonic` with a controllable counter."""
    box = {"now": start}
    monkeypatch.setattr(raises_module.time, "monotonic", lambda: box["now"])
    return box


def test_a_granted_raise_is_readable_until_it_expires(monkeypatch):
    clock = _fake_clock(monkeypatch)
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=60, granted_by="admin-device", reason="fixing the printer",
    )

    assert store.capabilities_for("device-1", "tailnet") == frozenset({Capability.EXECUTE})

    clock["now"] += 59  # one second short of expiry
    assert store.capabilities_for("device-1", "tailnet") == frozenset({Capability.EXECUTE})


def test_reading_an_expired_raise_returns_nothing_and_forgets_it(monkeypatch):
    clock = _fake_clock(monkeypatch)
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=60, granted_by="admin-device", reason="fixing the printer",
    )

    clock["now"] += 61  # past expiry
    assert store.capabilities_for("device-1", "tailnet") == frozenset()

    # Forgotten, not merely masked: a later read at the same instant still
    # finds nothing, and the record no longer shows up in a full listing.
    assert store.capabilities_for("device-1", "tailnet") == frozenset()
    assert store.active() == {}


def test_duration_is_clamped_to_the_seven_day_cap():
    store = RaiseStore()
    before = time.monotonic()
    grant = store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=MAX_RAISE_SECONDS * 10, granted_by="admin-device", reason="testing the cap",
    )
    after = time.monotonic()

    assert before + MAX_RAISE_SECONDS <= grant.expires_at <= after + MAX_RAISE_SECONDS


def test_a_zero_or_negative_duration_is_refused():
    store = RaiseStore()
    with pytest.raises(ValueError):
        store.grant(
            "device-1", "tailnet", frozenset({Capability.EXECUTE}),
            seconds=0, granted_by="admin-device", reason="nope",
        )
    with pytest.raises(ValueError):
        store.grant(
            "device-1", "tailnet", frozenset({Capability.EXECUTE}),
            seconds=-30, granted_by="admin-device", reason="nope",
        )


def test_a_raise_is_scoped_to_one_device_and_one_policy():
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="scoping check",
    )

    # The granted pair.
    assert store.capabilities_for("device-1", "tailnet") == frozenset({Capability.EXECUTE})
    # Same device, another policy.
    assert store.capabilities_for("device-1", "funnel") == frozenset()
    # Another device, the same policy.
    assert store.capabilities_for("device-2", "tailnet") == frozenset()
    # A wholly unrelated pair.
    assert store.capabilities_for("device-9", "local") == frozenset()


def test_revoke_removes_exactly_one_record():
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="a",
    )
    store.grant(
        "device-2", "tailnet", frozenset({Capability.SYSTEM_CONTROL}),
        seconds=3600, granted_by="admin-device", reason="b",
    )

    assert store.revoke("device-1", "tailnet") is True

    assert store.capabilities_for("device-1", "tailnet") == frozenset()
    assert store.capabilities_for("device-2", "tailnet") == frozenset({Capability.SYSTEM_CONTROL})


def test_drop_device_removes_every_policy_for_that_device():
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="a",
    )
    store.grant(
        "device-1", "funnel", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="b",
    )
    store.grant(
        "device-2", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="c",
    )

    store.drop_device("device-1")

    assert store.capabilities_for("device-1", "tailnet") == frozenset()
    assert store.capabilities_for("device-1", "funnel") == frozenset()
    assert store.capabilities_for("device-2", "tailnet") == frozenset({Capability.EXECUTE})


def test_drop_policy_removes_every_device_on_that_transport():
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="a",
    )
    store.grant(
        "device-2", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="b",
    )
    store.grant(
        "device-1", "funnel", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="c",
    )

    store.drop_policy("tailnet")

    assert store.capabilities_for("device-1", "tailnet") == frozenset()
    assert store.capabilities_for("device-2", "tailnet") == frozenset()
    assert store.capabilities_for("device-1", "funnel") == frozenset({Capability.EXECUTE})


def test_clear_removes_everything():
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="a",
    )
    store.grant(
        "device-2", "funnel", frozenset({Capability.EXECUTE}),
        seconds=3600, granted_by="admin-device", reason="b",
    )

    store.clear()

    assert store.active() == {}
    assert store.capabilities_for("device-1", "tailnet") == frozenset()
    assert store.capabilities_for("device-2", "funnel") == frozenset()


def test_active_never_lists_an_expired_record(monkeypatch):
    clock = _fake_clock(monkeypatch)
    store = RaiseStore()
    store.grant(
        "device-1", "tailnet", frozenset({Capability.EXECUTE}),
        seconds=30, granted_by="admin-device", reason="short-lived",
    )

    live = store.active()
    assert ("device-1", "tailnet") in live
    assert isinstance(live[("device-1", "tailnet")], RaiseGrant)

    clock["now"] += 31
    assert store.active() == {}


def test_an_expiring_record_is_logged_once_and_only_once(caplog):
    """Spec §3.6 asks for a log line on mint **and on expiry**.

    Expiry has no timer to hang one off -- a record dies on the read that
    notices it -- so the line comes from the delete-on-read paths. The
    once-only half is the part worth pinning: `capabilities_for` is called on
    every authenticated request, so a line emitted per *read* rather than per
    *record* would turn a lapsed raise into a log flood for as long as the
    device kept talking.
    """
    box = {"now": 1_000.0}
    original = raises_module.time.monotonic
    raises_module.time.monotonic = lambda: box["now"]
    try:
        store = RaiseStore()
        store.grant(
            device_id="device-1", policy_name="tailnet",
            capabilities=frozenset({Capability.EXECUTE}),
            seconds=30, granted_by="admin-device", reason="fixing the printer",
        )
        box["now"] += 31

        with caplog.at_level(logging.INFO, logger=raises_module.__name__):
            # Five reads over one lapsed record, through both delete-on-read
            # paths: three `capabilities_for`, one `get`, one `active`.
            assert store.capabilities_for("device-1", "tailnet") == frozenset()
            assert store.capabilities_for("device-1", "tailnet") == frozenset()
            assert store.capabilities_for("device-1", "tailnet") == frozenset()
            assert store.get("device-1", "tailnet") is None
            assert store.active() == {}

        lines = [r.getMessage() for r in caplog.records if "expired" in r.getMessage()]
        assert len(lines) == 1, lines
        assert "device-1" in lines[0] and "tailnet" in lines[0]
        # Ids and the transport only -- the reason is on the mint line and has
        # no business in a log that outlives the raise.
        assert "fixing the printer" not in lines[0]
    finally:
        raises_module.time.monotonic = original


def test_the_expiry_line_also_fires_when_active_is_the_first_read(caplog):
    """The other delete-on-read path, reached first. `GET /v1/devices` sweeps
    with `active()`, so a raise that lapses while nobody is talking is still
    announced by the next admin listing rather than vanishing silently."""
    box = {"now": 1_000.0}
    original = raises_module.time.monotonic
    raises_module.time.monotonic = lambda: box["now"]
    try:
        store = RaiseStore()
        store.grant(
            device_id="device-9", policy_name="tailnet",
            capabilities=frozenset({Capability.SYSTEM_CONTROL}),
            seconds=30, granted_by="admin-device", reason="quiet",
        )
        box["now"] += 31

        with caplog.at_level(logging.INFO, logger=raises_module.__name__):
            assert store.active() == {}
            assert store.capabilities_for("device-9", "tailnet") == frozenset()

        lines = [r.getMessage() for r in caplog.records if "expired" in r.getMessage()]
        assert len(lines) == 1, lines
        assert "device-9" in lines[0]
    finally:
        raises_module.time.monotonic = original


def test_the_store_has_no_serialisation_surface():
    # `save`/`load`/`to_json`/`from_json` are never defined anywhere here, on
    # the module or either class -- a plain `hasattr` is enough for those.
    for name in ("save", "load", "to_json", "from_json"):
        assert not hasattr(raises_module, name), name
        assert not hasattr(raises_module.RaiseStore, name), name
        assert not hasattr(raises_module.RaiseGrant, name), name

    # `__getstate__` is different: Python 3.11 gave `object` a default
    # implementation, so `hasattr(..., "__getstate__")` is true of *every*
    # object and would make this assertion meaningless. What must be absent
    # is a `__getstate__` this module wrote itself.
    assert "__getstate__" not in vars(raises_module.RaiseStore)
    assert "__getstate__" not in vars(raises_module.RaiseGrant)

    import inspect
    source = inspect.getsource(raises_module)
    assert "import json" not in source
    assert "import pathlib" not in source
    assert "from json" not in source
    assert "from pathlib" not in source
