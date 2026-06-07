"""Tests for manifest_registry — registry typed registry over the manifest store."""

import pytest

from assistant.automation.manifest_schema import (
    AppManifest, Intent, Selector, Match, Captured, dump_manifest_to_yaml,
)
from assistant.automation.manifest_store import ManifestStore
from assistant.automation.manifest_registry import ManifestRegistry
from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo


@pytest.fixture
def reg(tmp_path):
    _reset_for_testing()
    db_dir = tmp_path / "_db"
    db_dir.mkdir()
    db = Database(db_dir / "test.db")
    repo = AppManifestIndexRepo(db._conn)
    store = ManifestStore(manifests_dir=tmp_path, index_repo=repo)

    am = AppManifest(
        schema_version=1, app_id="test_app.desktop", display_name="Test App",
        match=Match(process_names=["TestApp.exe"], window_title_patterns=["^Test App"]),
        intents=[Intent(
            id="play", display_name="Play",
            phrases=["play music", "start playing"],
            handler_selectors=[Selector(kind="hotkey", keys="Space")],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    dump_manifest_to_yaml(am, tmp_path / "test_app.desktop.yaml")
    store.scan_and_index()

    r = ManifestRegistry(store=store, index_repo=repo)
    yield r
    db.close()
    _reset_for_testing()


def test_lookup_phrase_exact(reg):
    hits = reg.lookup_phrase("play music")
    assert len(hits) == 1
    assert hits[0] == ("test_app.desktop", "play")


def test_lookup_phrase_miss(reg):
    assert reg.lookup_phrase("dance the tango") == []


def test_get_for_active_app(reg):
    active = {"process_names": ["TestApp.exe"], "window_title": "Test App Premium"}
    am = reg.get_for_active_app(active)
    assert am is not None
    assert am.app_id == "test_app.desktop"


def test_get_for_active_app_no_match(reg):
    active = {"process_names": ["Other.exe"], "window_title": "Untitled"}
    assert reg.get_for_active_app(active) is None


def test_get_all_for_active_app_returns_list(reg):
    """New method returns list[AppManifest], not single — even for one match."""
    active = {"process_names": ["TestApp.exe"], "window_title": "Test App Premium"}
    matches = reg.get_all_for_active_app(active)
    assert len(matches) == 1
    assert matches[0].app_id == "test_app.desktop"


def test_get_all_for_active_app_empty_when_no_match(reg):
    active = {"process_names": ["Other.exe"], "window_title": "Untitled"}
    assert reg.get_all_for_active_app(active) == []


# ─── F12: multiple manifests sharing process_names ──────────────────────

def test_f12_get_all_for_active_app_returns_every_match(tmp_path):
    """F12 regression: when TWO manifests both match the active process,
    BOTH are returned (not just whichever was scanned first).

    Before F12, get_for_active_app returned at the first match and exited.
    A phrase belonging to manifest B was silently invisible whenever
    manifest A happened to load first — net effect: manifest-based routing dropped
    valid manifests, falling through to the LLM classifier and routing
    to whatever generic intent the classifier guessed. Live-test
    Scenario 6 hit this when spotify.desktop and test_collision_a.desktop
    both matched spotify.exe.
    """
    _reset_for_testing()
    db_dir = tmp_path / "_db"
    db_dir.mkdir()
    db = Database(db_dir / "test.db")
    repo = AppManifestIndexRepo(db._conn)
    store = ManifestStore(manifests_dir=tmp_path, index_repo=repo)

    # Two manifests, both claim "TestApp.exe"
    am_first = AppManifest(
        schema_version=1, app_id="test_app_first.desktop",
        display_name="First Test App",
        match=Match(process_names=["TestApp.exe"]),
        intents=[Intent(
            id="alpha", display_name="Alpha",
            phrases=["first only phrase"],
            handler_selectors=[Selector(kind="hotkey", keys="F1")],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    am_second = AppManifest(
        schema_version=1, app_id="test_app_second.desktop",
        display_name="Second Test App",
        match=Match(process_names=["TestApp.exe"]),
        intents=[Intent(
            id="beta", display_name="Beta",
            phrases=["second only phrase"],
            handler_selectors=[Selector(kind="hotkey", keys="F2")],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    dump_manifest_to_yaml(am_first, tmp_path / "test_app_first.desktop.yaml")
    dump_manifest_to_yaml(am_second, tmp_path / "test_app_second.desktop.yaml")
    store.scan_and_index()
    reg = ManifestRegistry(store=store, index_repo=repo)

    try:
        active = {"process_names": ["TestApp.exe"], "window_title": "anything"}
        matches = reg.get_all_for_active_app(active)
        app_ids = {m.app_id for m in matches}
        assert app_ids == {"test_app_first.desktop", "test_app_second.desktop"}, (
            f"F12 regression: both manifests must surface, got {app_ids}"
        )
    finally:
        db.close()
        _reset_for_testing()
