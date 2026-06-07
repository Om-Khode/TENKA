# tests/test_status_broadcaster.py
import io
import json
import time
import pytest
from assistant.io.status_broadcaster import StatusBroadcaster, StatusPhase


def test_set_writes_json_to_attached_writer():
    sink = io.StringIO()
    b = StatusBroadcaster()
    b.attach_ipc(sink)
    b.set(StatusPhase.CLICKING, detail="Send", cursor_follows=True)
    line = sink.getvalue().strip()
    payload = json.loads(line)
    assert payload["v"] == 2  # protocol v2 (adds step + tier)
    assert payload["type"] == "status"
    assert payload["phase"] == "CLICKING"
    assert payload["detail"] == "Send"
    assert payload["cursor_follows"] is True
    assert payload["step"] is None
    assert payload["tier"] is None
    assert "ts" in payload


def test_set_emits_step_and_tier_when_provided():
    sink = io.StringIO()
    b = StatusBroadcaster()
    b.attach_ipc(sink)
    b.set(StatusPhase.READING, detail="OCR", step=(2, 3), tier="vision")
    payload = json.loads(sink.getvalue().strip())
    assert payload["step"] == [2, 3]
    assert payload["tier"] == "vision"


def test_set_rejects_invalid_tier():
    b = StatusBroadcaster()
    try:
        b.set(StatusPhase.CLICKING, tier="bogus")
        assert False, "should have raised"
    except ValueError:
        pass


def test_set_rejects_malformed_step():
    b = StatusBroadcaster()
    try:
        b.set(StatusPhase.CLICKING, step=(1, 2, 3))
        assert False, "should have raised"
    except TypeError:
        pass


def test_dedupe_same_phase_detail_skipped():
    sink = io.StringIO()
    b = StatusBroadcaster()
    b.attach_ipc(sink)
    b.set(StatusPhase.THINKING, detail="x")
    b.set(StatusPhase.THINKING, detail="x")
    assert sink.getvalue().count("\n") == 1


def test_rate_limit_drops_writes_under_50ms():
    sink = io.StringIO()
    b = StatusBroadcaster()
    b.attach_ipc(sink)
    b.set(StatusPhase.CLICKING, detail="a")
    b.set(StatusPhase.CLICKING, detail="b")  # different detail, but <50ms
    # Second should be rate-limited
    assert sink.getvalue().count("\n") == 1


def test_idle_always_fires_regardless_of_rate_limit():
    sink = io.StringIO()
    b = StatusBroadcaster()
    b.attach_ipc(sink)
    b.set(StatusPhase.CLICKING, detail="x")
    b.set(StatusPhase.IDLE)
    assert sink.getvalue().count("\n") == 2


def test_in_proc_subscribers_fire():
    received = []
    b = StatusBroadcaster()
    b.subscribe(lambda evt: received.append(evt))
    b.set(StatusPhase.THINKING, detail="planning")
    assert len(received) == 1
    assert received[0]["phase"] == "THINKING"


def test_broken_pipe_triggers_dead_callback():
    class BrokenSink:
        def write(self, _): raise BrokenPipeError()
        def flush(self): pass
    dead_calls = []
    b = StatusBroadcaster()
    b.set_on_overlay_dead(lambda: dead_calls.append(1))
    b.attach_ipc(BrokenSink())
    b.set(StatusPhase.THINKING)
    assert dead_calls == [1]


def test_detach_ipc_stops_writes_but_keeps_subscribers():
    sink = io.StringIO()
    received = []
    b = StatusBroadcaster()
    b.attach_ipc(sink)
    b.subscribe(lambda e: received.append(e))
    b.detach_ipc()
    b.set(StatusPhase.IDLE)
    assert sink.getvalue() == ""
    assert len(received) == 1


def test_unknown_phase_label_handled():
    # set() takes StatusPhase enum; non-enum input is a programming error
    b = StatusBroadcaster()
    with pytest.raises((TypeError, AttributeError)):
        b.set("CLICKING")
