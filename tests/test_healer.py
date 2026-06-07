"""Tests for manifest-based healer — tier-1 AT fingerprint heal."""

import json
import pytest
from pathlib import Path

from assistant.automation.manifest_schema import (
    AppManifest, Intent, Selector, Match, Captured, dump_manifest_to_yaml,
)
from assistant.automation.manifest_store import ManifestStore
from assistant.automation.healer import Healer, HealResult
from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo


FIXTURES = Path(__file__).parent / "fixtures" / "at_trees"


class FakeATTerminator:
    """Walks a flat AT-tree fixture and exposes the healer's surface."""

    def __init__(self, tree_path: Path):
        self.tree = json.loads(tree_path.read_text(encoding="utf-8"))

    def enumerate_descendants(
        self, *, parent_window: str, max_depth: int = 4,
    ) -> list[dict]:
        return list(self.tree["elements"].values())

    def screenshot(self) -> bytes:
        return b"PNG_BYTES"

    def element_at_point(self, x: int, y: int) -> dict | None:
        return None


class EmptyATTerminator:
    """No matching candidates — tier-1 must score below threshold."""

    def enumerate_descendants(self, **_) -> list[dict]:
        return [{
            "automation_id": "x", "control_type": "Edit",
            "name": "Other", "parent_chain": [], "sibling_count": 0,
        }]

    def screenshot(self) -> bytes:
        return b""

    def element_at_point(self, x: int, y: int) -> dict | None:
        return None


@pytest.fixture
def setup(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    repo = AppManifestIndexRepo(db._conn)
    store = ManifestStore(manifests_dir=tmp_path, index_repo=repo)
    am = AppManifest(
        schema_version=1, app_id="test_app.desktop", display_name="TestApp",
        match=Match(process_names=["TestApp.exe"]),
        intents=[Intent(
            id="play", display_name="Play", phrases=["play music"],
            handler_selectors=[
                Selector(
                    kind="uia", control_type="Button",
                    automation_id="play-pause-button",
                    parent_chain=["Window[Name~'TestApp']",
                                  "Pane[ClassName~'Chrome_WidgetWin']"],
                    name_hint="Play",
                ),
            ],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    dump_manifest_to_yaml(am, tmp_path / "test_app.desktop.yaml")
    store.scan_and_index()
    yield {"store": store, "tmp_path": tmp_path, "db": db}
    db.close()
    _reset_for_testing()


def test_tier1_heals_when_automation_id_renamed(setup):
    """The renamed-automation_id scenario: tier-1 fingerprint patches in place."""
    term = FakeATTerminator(FIXTURES / "test_app_renamed.json")
    healer = Healer(
        store=setup["store"],
        terminator_provider=lambda: term,
        vision_cap=None,  # tier-2 won't fire in this test
    )
    am = setup["store"].get("test_app.desktop")
    intent = am.intents[0]
    version_before = intent.version
    result = healer.try_heal(
        manifest=am, intent=intent, selector_index=0,
        active_window="Test App Window",
    )
    assert result.ok is True
    assert result.tier == 1
    assert result.new_automation_id == "play_btn_v2"
    # Manifest patched in place
    assert intent.handler_selectors[0].automation_id == "play_btn_v2"
    assert intent.version == version_before + 1


def test_tier1_fails_when_no_candidate_above_threshold(setup):
    """An AT tree with no Button control — type mismatch scores 0, tier-1 rejects below threshold."""
    healer = Healer(
        store=setup["store"],
        terminator_provider=lambda: EmptyATTerminator(),
        vision_cap=None,
    )
    am = setup["store"].get("test_app.desktop")
    intent = am.intents[0]
    result = healer.try_heal(
        manifest=am, intent=intent, selector_index=0,
        active_window="Test App Window",
    )
    assert result.ok is False
    # Manifest unchanged
    assert intent.handler_selectors[0].automation_id == "play-pause-button"
    assert intent.version == 1


def test_try_heal_skips_non_uia_selector(setup):
    """Hotkey/vision selectors are out of scope for the healer — early exit, no AT enumerate."""
    am = setup["store"].get("test_app.desktop")
    # Swap the saved UIA selector for a hotkey selector
    am.intents[0].handler_selectors[0] = Selector(kind="hotkey", keys="ctrl+p")

    class NeverCalledTerm:
        def enumerate_descendants(self, **_):
            raise AssertionError("enumerate must not run for non-uia selectors")
        def screenshot(self):
            raise AssertionError("screenshot must not run for non-uia selectors")
        def element_at_point(self, x, y):
            return None

    healer = Healer(
        store=setup["store"],
        terminator_provider=lambda: NeverCalledTerm(),
        vision_cap=None,
    )
    result = healer.try_heal(
        manifest=am, intent=am.intents[0], selector_index=0,
        active_window="Test App Window",
    )
    assert result.ok is False
    assert result.tier == 0
    assert "uia" in (result.error or "")


def test_tier2_fires_when_tier1_empty(setup, monkeypatch):
    """No UIA elements at all → tier-1 scores nothing, tier-2 vision takes over."""
    import asyncio

    class NonUIATerm:
        def enumerate_descendants(self, **_):
            return []  # empty AT tree forces tier-1 to find no candidates

        def screenshot(self):
            return b"PNG_BYTES"

        def element_at_point(self, x, y):
            return {
                "automation_id": "vision_resolved_id",
                "parent_chain": ["Window[Name~'TestApp']"],
            }

    async def _fake_vision(*, crop_bytes, query, crop_origin):
        return {"x": 400, "y": 300, "confidence": 0.9}

    def _run_coro_sync(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(
        "assistant.llm.contracts.ask_for_vision_ground_coords", _fake_vision,
    )
    monkeypatch.setattr(
        "assistant.core.asyncio_utils.call_async", _run_coro_sync,
    )

    import sys
    fake_pag = type("FakePAG", (), {"click": staticmethod(lambda x, y: None)})
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pag)

    from assistant.automation.vision_cap import VisionCapTracker
    cap = VisionCapTracker(setup["db"]._conn)

    healer = Healer(
        store=setup["store"],
        terminator_provider=lambda: NonUIATerm(),
        vision_cap=cap,
    )
    am = setup["store"].get("test_app.desktop")
    intent = am.intents[0]
    version_before = intent.version

    result = healer.try_heal(
        manifest=am, intent=intent, selector_index=0,
        active_window="Test App Window",
    )
    assert result.ok is True
    assert result.tier == 2
    # Manifest was patched with the new automation_id from element_at_point
    assert intent.handler_selectors[0].automation_id == "vision_resolved_id"
    assert intent.version == version_before + 1


def test_tier2_blocked_when_cap_reached(setup, monkeypatch):
    """Daily vision cap reached → tier-2 returns ok=False without calling vision."""
    from assistant.automation.vision_cap import VisionCapTracker, DEFAULT_DAILY_CAP

    cap = VisionCapTracker(setup["db"]._conn)
    # Burn the cap
    for _ in range(DEFAULT_DAILY_CAP):
        assert cap.try_increment() is True
    # Next try_increment must return False (cap reached)
    assert cap.try_increment() is False

    # Track whether the vision contract is called — it must NOT be
    vision_calls = []

    async def _fake_vision_should_not_run(*, crop_bytes, query, crop_origin):
        vision_calls.append((crop_bytes, query, crop_origin))
        return {"x": 0, "y": 0, "confidence": 1.0}

    monkeypatch.setattr(
        "assistant.llm.contracts.ask_for_vision_ground_coords",
        _fake_vision_should_not_run,
    )

    # enumerate_descendants returns [] so tier-1 finds no candidates and
    # escalates to tier-2, where the cap check fires before the vision
    # contract is ever invoked. If a future change scores tier-1 candidates,
    # this test would silently bypass the cap-blocked assertion.
    class NonUIATerm:
        def enumerate_descendants(self, **_):
            return []
        def screenshot(self):
            return b""
        def element_at_point(self, x, y):
            return None

    healer = Healer(
        store=setup["store"],
        terminator_provider=lambda: NonUIATerm(),
        vision_cap=cap,
    )
    am = setup["store"].get("test_app.desktop")
    intent = am.intents[0]
    version_before = intent.version

    result = healer.try_heal(
        manifest=am, intent=intent, selector_index=0,
        active_window="Test App Window",
    )
    assert result.ok is False
    assert result.tier == 0  # both tiers exhausted; cap blocks tier-2
    # Manifest unchanged
    assert intent.version == version_before
    # Vision was never called
    assert vision_calls == []
