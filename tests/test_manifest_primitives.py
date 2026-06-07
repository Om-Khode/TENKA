"""Tests for manifest-based primitive backends (hotkey, uia)."""

from assistant.automation.manifest_schema import Selector
from assistant.automation.manifest_primitives import (
    execute_primitive, PrimitiveResult,
)


def test_hotkey_primitive_sends_key(fake_terminator):
    sel = Selector(kind="hotkey", keys="Space")
    result = execute_primitive(sel, terminator=fake_terminator, active_window="TestApp")
    assert result.ok is True
    assert fake_terminator.last_call == ("send_key", "Space")


def test_uia_primitive_clicks_element(fake_terminator):
    fake_terminator.elements["play-pause"] = {"automation_id": "play-pause"}
    sel = Selector(kind="uia", control_type="Button", automation_id="play-pause")
    result = execute_primitive(sel, terminator=fake_terminator, active_window="TestApp")
    assert result.ok is True
    assert fake_terminator.last_call == ("click", "play-pause")


def test_uia_primitive_returns_failure_on_missing_element(fake_terminator):
    sel = Selector(kind="uia", control_type="Button", automation_id="missing")
    result = execute_primitive(sel, terminator=fake_terminator, active_window="TestApp")
    assert result.ok is False
    assert "find_element failed" in (result.error or "")


def test_vision_reground_is_placeholder_v1(fake_terminator):
    """Vision-reground is filled in session 4 — for now it's a recognized stub."""
    sel = Selector(kind="vision_reground", query="play button")
    result = execute_primitive(sel, terminator=fake_terminator, active_window="TestApp")
    assert result.ok is False
    assert "vision_reground not implemented" in (result.error or "")


# ─── F8: UIA dispatch by accessible name (no automation_id) ─────────────

def test_uia_primitive_clicks_by_name_when_no_automation_id(fake_terminator):
    """F8 happy path: selector has only name_hint → primitive dispatches by name.

    This is the Spotify case — the promoter writes name-only UIA selectors
    because the underlying app exposes no automation_id. Before F8 the
    primitive bailed with 'missing automation_id'; after F8 it walks the
    tree by accessible name.
    """
    fake_terminator.elements_by_name["Play"] = {"automation_id": "", "name": "Play"}
    sel = Selector(kind="uia", control_type="Button", name_hint="Play")
    result = execute_primitive(sel, terminator=fake_terminator, active_window="TestApp")
    assert result.ok is True
    assert fake_terminator.last_call == ("click", "Play")


def test_uia_primitive_prefers_automation_id_when_both_present(fake_terminator):
    """When a selector carries BOTH automation_id and name_hint, automation_id wins.

    automation_id is the more stable identifier — a renamed button keeps
    its id but loses its display name, so the dispatcher prefers id.
    """
    fake_terminator.elements["play-id"] = {"automation_id": "play-id", "name": "WrongName"}
    fake_terminator.elements_by_name["Play"] = {"automation_id": "", "name": "Play"}
    sel = Selector(
        kind="uia", control_type="Button",
        automation_id="play-id", name_hint="Play",
    )
    result = execute_primitive(sel, terminator=fake_terminator, active_window="TestApp")
    assert result.ok is True
    # The click went to the automation_id-keyed element, not the name-keyed one.
    assert fake_terminator.last_call == ("click", "play-id")


def test_uia_primitive_rejects_empty_selector(fake_terminator):
    """When BOTH automation_id and name_hint are missing, primitive returns failure."""
    sel = Selector(kind="uia", control_type="Button")
    result = execute_primitive(sel, terminator=fake_terminator, active_window="TestApp")
    assert result.ok is False
    assert "missing both 'automation_id' and 'name_hint'" in (result.error or "")
