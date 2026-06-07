"""Test manifest phrases integrating with regex_router."""

import pytest
from unittest.mock import patch

from assistant.automation.manifest_schema import (
    AppManifest, Intent, Selector, Match, Captured, dump_manifest_to_yaml,
)
from assistant.automation.manifest_store import ManifestStore
from assistant.automation.manifest_registry import (
    ManifestRegistry, init_singleton,
)
from assistant import regex_router
from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo


@pytest.fixture
def reg_setup(tmp_path):
    from assistant.storage.db import Database, _reset_for_testing
    _reset_for_testing()
    db_dir = tmp_path / "_db"
    db_dir.mkdir()
    db = Database(db_dir / "test.db")
    repo = AppManifestIndexRepo(db._conn)
    store = ManifestStore(manifests_dir=tmp_path, index_repo=repo)
    am = AppManifest(
        schema_version=1, app_id="test_app.desktop", display_name="Test App",
        match=Match(process_names=["TestApp.exe"]),
        intents=[Intent(
            id="play", display_name="Play",
            phrases=["play music", "press play"],
            handler_selectors=[Selector(kind="hotkey", keys="Space")],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    dump_manifest_to_yaml(am, tmp_path / "test_app.desktop.yaml")
    store.scan_and_index()
    init_singleton(store=store, index_repo=repo)
    yield
    db.close()
    _reset_for_testing()
    # Also reset the manifest registry singleton so test isolation holds
    from assistant.automation import manifest_registry as _mr
    _mr._singleton = None


def test_regex_router_dispatches_to_manifest_intent_when_active_app_matches(reg_setup):
    # Use "press play" — "play music" collides with _PLAY_RE further down
    # the router; "press play" matches no other pre-route pattern, so the
    # negative test below can assert `result is None` cleanly.
    with patch("assistant.automation.router.detect_active_app",
               return_value={"process_names": ["TestApp.exe"],
                             "window_title": "Test App Window",
                             "active_url": ""}):
        result = regex_router.pre_route("press play")
        assert result is not None
        assert result.intent == "manifest_dispatch"
        assert result.params["app_id"] == "test_app.desktop"
        assert result.params["intent_id"] == "play"


def test_regex_router_no_match_when_app_not_active(reg_setup):
    with patch("assistant.automation.router.detect_active_app",
               return_value={"process_names": ["Notepad.exe"],
                             "window_title": "Untitled",
                             "active_url": ""}):
        result = regex_router.pre_route("press play")
        # No manifest match AND "press play" is not covered by any other
        # pre-route pattern, so result must be None.
        assert result is None


# ─── F12: two manifests, same process, phrase belongs to second ─────────

@pytest.fixture
def reg_two_manifests_same_process(tmp_path):
    """Two manifests both matching TestApp.exe. The phrase only belongs to
    the SECOND one (loaded after the first). Before F12, the regex_router
    would fetch the first match from get_for_active_app and never see the
    second — the phrase candidate would never line up with the active app
    and routing would fall through to None.
    """
    from assistant.storage.db import Database, _reset_for_testing
    _reset_for_testing()
    db_dir = tmp_path / "_db"
    db_dir.mkdir()
    db = Database(db_dir / "test.db")
    repo = AppManifestIndexRepo(db._conn)
    store = ManifestStore(manifests_dir=tmp_path, index_repo=repo)

    am_first = AppManifest(
        schema_version=1, app_id="a_first.desktop", display_name="First",
        match=Match(process_names=["TestApp.exe"]),
        intents=[Intent(
            id="alpha", display_name="Alpha",
            phrases=["alpha only"],
            handler_selectors=[Selector(kind="hotkey", keys="F1")],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    am_second = AppManifest(
        schema_version=1, app_id="b_second.desktop", display_name="Second",
        match=Match(process_names=["TestApp.exe"]),
        intents=[Intent(
            id="beta", display_name="Beta",
            phrases=["beta only"],
            handler_selectors=[Selector(kind="hotkey", keys="F2")],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    # 'a_first' sorts before 'b_second' so the store scans it first —
    # mimicking the real Spotify-vs-test_collision_a load order in the
    # live trace that exposed F12.
    dump_manifest_to_yaml(am_first, tmp_path / "a_first.desktop.yaml")
    dump_manifest_to_yaml(am_second, tmp_path / "b_second.desktop.yaml")
    store.scan_and_index()
    init_singleton(store=store, index_repo=repo)
    yield
    db.close()
    _reset_for_testing()
    from assistant.automation import manifest_registry as _mr
    _mr._singleton = None


def test_f12_regex_router_routes_to_second_manifest_when_phrase_belongs_to_it(
    reg_two_manifests_same_process,
):
    """F12 regression: routing must pick the manifest that OWNS the phrase,
    not just whichever happens to be first in the active-app scan."""
    with patch("assistant.automation.router.detect_active_app",
               return_value={"process_names": ["TestApp.exe"],
                             "window_title": "anything",
                             "active_url": ""}):
        # "alpha only" belongs to the FIRST manifest — straightforward case.
        result_first = regex_router.pre_route("alpha only")
        assert result_first is not None
        assert result_first.intent == "manifest_dispatch"
        assert result_first.params["app_id"] == "a_first.desktop"

        # "beta only" belongs to the SECOND manifest — before F12 this
        # would silently fall through because get_for_active_app returned
        # only the first match.
        result_second = regex_router.pre_route("beta only")
        assert result_second is not None
        assert result_second.intent == "manifest_dispatch"
        assert result_second.params["app_id"] == "b_second.desktop", (
            "F12 regression: phrase 'beta only' belongs to b_second.desktop "
            "and must route there even though a_first.desktop matches the "
            "active app first"
        )
