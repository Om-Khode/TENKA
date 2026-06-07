"""Tests for manifest schema (Pydantic models + YAML round-trip)."""

import pydantic
import pytest
import yaml

from assistant.automation.manifest_schema import (
    AppManifest, Intent, Selector, Match, Captured, Health,
    load_manifest_from_yaml, dump_manifest_to_yaml,
)


# ─── Cluster 1: schema construction ────────────────────────────────────────

def test_minimal_manifest_constructs():
    am = AppManifest(
        schema_version=1,
        app_id="test_app.desktop",
        display_name="Test App",
        match=Match(process_names=["TestApp.exe"]),
        intents=[],
    )
    assert am.app_id == "test_app.desktop"
    assert am.match.url_patterns == []  # default


def test_selector_kinds_validated():
    Selector(kind="hotkey", keys="Space")
    Selector(kind="uia", control_type="Button", automation_id="play")
    Selector(kind="vision_reground", query="play button")
    with pytest.raises(pydantic.ValidationError):
        Selector(kind="invalid_kind")


def test_extra_fields_forbidden():
    with pytest.raises(pydantic.ValidationError):
        AppManifest(
            schema_version=1,
            app_id="x",
            display_name="X",
            match=Match(process_names=["x"]),
            intents=[],
            unknown_field="oops",
        )


# ─── Cluster 2: YAML round-trip ────────────────────────────────────────────

def test_yaml_round_trip(tmp_path):
    am = AppManifest(
        schema_version=1,
        app_id="spotify.desktop",
        display_name="Spotify",
        match=Match(process_names=["Spotify.exe"], window_title_patterns=["^Spotify"]),
        intents=[
            Intent(
                id="play", display_name="Play",
                phrases=["play music"],
                handler_selectors=[Selector(kind="hotkey", keys="Space")],
                captured=Captured(timestamp="2026-05-30T10:00:00Z"),
            )
        ],
    )
    p = tmp_path / "spotify.desktop.yaml"
    dump_manifest_to_yaml(am, p)
    loaded = load_manifest_from_yaml(p)
    assert loaded.app_id == am.app_id
    assert loaded.intents[0].phrases == ["play music"]
    assert loaded.intents[0].handler_selectors[0].keys == "Space"
    assert loaded.schema_version == 1
    assert loaded.intents[0].version == 1
    assert loaded.intents[0].captured.timestamp == "2026-05-30T10:00:00Z"
    assert loaded.intents[0].handler_selectors[0].weight == 1.0
    assert loaded.intents[0].handler_selectors[0].successes == 0
    assert loaded.intents[0].health.consecutive_failures == 0


def test_migration_identity_at_v1(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text(
        "schema_version: 1\napp_id: x\ndisplay_name: X\n"
        "match:\n  process_names: [x]\nintents: []\n",
        encoding="utf-8",
    )
    am = load_manifest_from_yaml(p)
    assert am.app_id == "x"


def test_migration_forward_version_refuses(tmp_path):
    """A manifest from a future TENKA build must be refused, not silently passed through."""
    p = tmp_path / "future.yaml"
    p.write_text(
        "schema_version: 99\napp_id: x\ndisplay_name: X\n"
        "match:\n  process_names: [x]\nintents: []\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"newer than this build"):
        load_manifest_from_yaml(p)
