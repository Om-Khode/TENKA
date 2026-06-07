# tests/test_overlay_protocol.py
import json
import pytest
from assistant.io.overlay.ipc import parse_event, ParseResult


def _line(payload):
    return json.dumps(payload) + "\n"


def test_valid_status_event_parsed():
    r = parse_event(_line({"v": 1, "type": "status", "phase": "CLICKING",
                           "detail": "Send", "cursor_follows": True, "ts": 1.0}))
    assert r.ok is True
    assert r.event["phase"] == "CLICKING"


def test_valid_cmd_quit_parsed():
    r = parse_event(_line({"v": 1, "type": "cmd", "cmd": "quit"}))
    assert r.ok is True
    assert r.event["cmd"] == "quit"


def test_malformed_json_rejected():
    r = parse_event("not-json\n")
    assert r.ok is False
    assert "json" in r.error.lower()


def test_missing_v_rejected():
    r = parse_event(_line({"type": "status", "phase": "IDLE"}))
    assert r.ok is False
    assert "v" in r.error.lower()


def test_version_mismatch_rejected():
    r = parse_event(_line({"v": 999, "type": "status", "phase": "IDLE"}))
    assert r.ok is False
    assert "version" in r.error.lower()


def test_missing_type_rejected():
    r = parse_event(_line({"v": 1, "phase": "IDLE"}))
    assert r.ok is False


def test_unknown_phase_accepted_with_warning():
    # Per design: unknown phase falls back to "Working…", does not reject
    r = parse_event(_line({"v": 1, "type": "status", "phase": "NONSENSE",
                           "detail": "", "cursor_follows": False, "ts": 1.0}))
    assert r.ok is True
    assert r.warning is not None
    assert "phase" in r.warning.lower()


def test_status_missing_required_fields_filled_with_defaults():
    r = parse_event(_line({"v": 1, "type": "status", "phase": "IDLE"}))
    assert r.ok is True
    assert r.event.get("detail", "") == ""
    assert r.event.get("cursor_follows", False) is False


def test_v2_step_and_tier_pass_through():
    r = parse_event(_line({"v": 2, "type": "status", "phase": "READING",
                           "detail": "OCR", "cursor_follows": False,
                           "step": [2, 3], "tier": "vision", "ts": 1.0}))
    assert r.ok is True
    assert r.event["step"] == [2, 3]
    assert r.event["tier"] == "vision"


def test_v1_event_gets_step_tier_defaulted_to_none():
    r = parse_event(_line({"v": 1, "type": "status", "phase": "CLICKING"}))
    assert r.ok is True
    assert r.event["step"] is None
    assert r.event["tier"] is None
