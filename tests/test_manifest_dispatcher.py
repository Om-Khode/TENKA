"""Tests for manifest-based dispatcher — selector chain walk + health counter updates."""

import pytest

from assistant.automation.manifest_dispatcher import ManifestDispatcher
from assistant.automation.manifest_registry import ManifestRegistry
from assistant.automation.manifest_schema import (
    AppManifest, Captured, Intent, Match, Selector, dump_manifest_to_yaml,
)
from assistant.automation.manifest_store import ManifestStore
from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo


def _build_test_manifest():
    return AppManifest(
        schema_version=1, app_id="test_app.desktop", display_name="Test App",
        match=Match(process_names=["TestApp.exe"]),
        intents=[Intent(
            id="play", display_name="Play",
            phrases=["play music"],
            handler_selectors=[
                Selector(kind="hotkey", keys="Space"),
                Selector(kind="uia", control_type="Button",
                         automation_id="play-pause-button"),
            ],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )


@pytest.fixture
def setup(tmp_path, fake_terminator):
    _reset_for_testing()
    db_dir = tmp_path / "_db"
    db_dir.mkdir()
    db = Database(db_dir / "test.db")
    repo = AppManifestIndexRepo(db._conn)
    store = ManifestStore(manifests_dir=tmp_path, index_repo=repo)
    am = _build_test_manifest()
    dump_manifest_to_yaml(am, tmp_path / "test_app.desktop.yaml")
    store.scan_and_index()
    reg = ManifestRegistry(store=store, index_repo=repo)
    disp = ManifestDispatcher(
        registry=reg, store=store, terminator_provider=lambda: fake_terminator,
    )
    yield {"store": store, "registry": reg, "term": fake_terminator, "disp": disp}
    db.close()
    _reset_for_testing()


def test_dispatch_first_selector_wins(setup):
    result = setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    assert result.ok is True
    assert setup["term"].last_call == ("send_key", "Space")
    am = setup["store"].get("test_app.desktop")
    assert am.intents[0].handler_selectors[0].successes == 1


def test_dispatch_falls_through_to_next_on_miss(setup):
    def boom(k):
        raise RuntimeError("simulated send_key failure")
    setup["term"].send_key = boom
    setup["term"].elements["play-pause-button"] = {
        "automation_id": "play-pause-button",
    }
    result = setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    assert result.ok is True
    assert setup["term"].last_call == ("click", "play-pause-button")


def test_dispatch_escalates_when_all_selectors_miss(setup):
    def boom_k(k):
        raise RuntimeError("k")
    setup["term"].send_key = boom_k
    # Don't seed any elements — uia call will also fail.
    result = setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    assert result.ok is False
    assert result.escalate_to_dispatch is True


def test_dispatch_missing_manifest_escalates(setup):
    result = setup["disp"].dispatch(
        app_id="unknown.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    assert result.ok is False
    assert result.escalate_to_dispatch is True
    assert "manifest not found" in (result.error or "")


def test_dispatch_missing_intent_escalates(setup):
    result = setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="dance",
        slots={}, active_window="TestApp",
    )
    assert result.ok is False
    assert result.escalate_to_dispatch is True
    assert "intent not found" in (result.error or "")


def test_failure_bumps_failure_counter(setup):
    def boom(k):
        raise RuntimeError("nope")
    setup["term"].send_key = boom
    setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    am = setup["store"].get("test_app.desktop")
    assert am.intents[0].handler_selectors[0].failures == 1
    assert am.intents[0].health.consecutive_failures == 1


def test_voyager_demotion_after_three_consecutive_failures(setup):
    def boom(k):
        raise RuntimeError("nope")
    setup["term"].send_key = boom
    am_before = setup["store"].get("test_app.desktop")
    primary_before = am_before.intents[0].handler_selectors[0]
    secondary_before = am_before.intents[0].handler_selectors[1]
    assert primary_before.kind == "hotkey"
    assert secondary_before.kind == "uia"

    for _ in range(3):
        setup["disp"].dispatch(
            app_id="test_app.desktop", intent_id="play",
            slots={}, active_window="TestApp",
        )

    am_after = setup["store"].get("test_app.desktop")
    # Primary and secondary swapped after 3 consecutive primary failures.
    assert am_after.intents[0].handler_selectors[0].kind == "uia"
    assert am_after.intents[0].handler_selectors[1].kind == "hotkey"
    # consecutive_failures reset after the swap.
    assert am_after.intents[0].health.consecutive_failures == 0


def test_success_resets_consecutive_failures_and_bumps_dispatches(setup):
    def boom(k):
        raise RuntimeError("nope")
    setup["term"].send_key = boom
    setup["term"].elements["play-pause-button"] = {
        "automation_id": "play-pause-button",
    }
    result = setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    assert result.ok is True
    am = setup["store"].get("test_app.desktop")
    # Primary (hotkey) failed; secondary (uia) succeeded.
    assert am.intents[0].handler_selectors[0].failures == 1
    assert am.intents[0].handler_selectors[1].successes == 1
    # health updated by the success path.
    assert am.intents[0].health.consecutive_failures == 0
    assert am.intents[0].health.total_dispatches == 1


def test_flush_now_writes_pending_health_to_disk(tmp_path, setup):
    setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    yaml_path = tmp_path / "test_app.desktop.yaml"
    text_before = yaml_path.read_text(encoding="utf-8")
    assert "successes: 1" not in text_before

    setup["disp"].flush_now()
    text_after = yaml_path.read_text(encoding="utf-8")
    assert "successes: 1" in text_after


def test_correction_signal_increments_failure_counter(setup):
    """record_correction bumps the primary selector's failure counter."""
    # Dispatch once to populate health + set _last_dispatch
    setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    before = setup["store"].get("test_app.desktop").intents[0].handler_selectors[0].failures

    setup["disp"].record_correction(
        app_id="test_app.desktop", intent_id="play",
    )

    after = setup["store"].get("test_app.desktop").intents[0].handler_selectors[0].failures
    assert after == before + 1


def test_record_last_dispatch_correction_proxies_to_cached_params(setup):
    """record_last_dispatch_correction uses the cached (app_id, intent_id) from the last dispatch."""
    setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    before = setup["store"].get("test_app.desktop").intents[0].handler_selectors[0].failures
    setup["disp"].record_last_dispatch_correction()
    after = setup["store"].get("test_app.desktop").intents[0].handler_selectors[0].failures
    assert after == before + 1


def test_record_last_dispatch_correction_noop_when_never_dispatched(setup):
    """record_last_dispatch_correction silently no-ops if nothing's been dispatched yet."""
    # No dispatch call before record_correction — should not raise, should not change state
    setup["disp"].record_last_dispatch_correction()
    # No assertion needed beyond "didn't raise" — verify by getting a fresh selector
    sel = setup["store"].get("test_app.desktop").intents[0].handler_selectors[0]
    assert sel.failures == 0


# ─── F10: corrections accumulate across successful dispatches ──────────

def test_correction_signal_bumps_consecutive_corrections_not_failures(setup):
    """record_correction must increment consecutive_corrections, NOT consecutive_failures.

    The two counters track different signals: failures = dispatch-level
    selector miss, corrections = T1-detected user feedback. They must
    accumulate independently so a successful dispatch (which resets
    consecutive_failures) does not erase user-feedback pressure.
    """
    setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    health_before = setup["store"].get("test_app.desktop").intents[0].health
    assert health_before.consecutive_failures == 0
    assert health_before.consecutive_corrections == 0

    setup["disp"].record_correction(
        app_id="test_app.desktop", intent_id="play",
    )

    health_after = setup["store"].get("test_app.desktop").intents[0].health
    # The user-signal counter went up; the dispatch-failure counter did NOT.
    assert health_after.consecutive_corrections == 1
    assert health_after.consecutive_failures == 0


def test_successful_dispatch_does_not_reset_consecutive_corrections(setup):
    """F10 core: a success path must NOT erase accrued correction signals.

    The original bug: every successful dispatch reset consecutive_failures
    to 0, and corrections used the same counter, so 'dispatch → correction
    → dispatch → correction' never accumulated past 1. After F10, the
    corrections counter is tracked separately and survives a success.
    """
    # Cycle 1: dispatch ok, correction
    setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    setup["disp"].record_correction(
        app_id="test_app.desktop", intent_id="play",
    )
    # Cycle 2: another successful dispatch
    setup["disp"].dispatch(
        app_id="test_app.desktop", intent_id="play",
        slots={}, active_window="TestApp",
    )
    health = setup["store"].get("test_app.desktop").intents[0].health
    # The correction from cycle 1 must still be counted.
    assert health.consecutive_corrections == 1
    # The dispatch-failure counter IS reset by the success path — that
    # part of the design is intentional and orthogonal to the F10 fix.
    assert health.consecutive_failures == 0


def test_three_corrections_across_successes_demote_primary(setup):
    """F10 end-to-end: 3 correction signals interleaved with successful
    dispatches must drive the Voyager demotion.

    Before F10, the realistic user flow (click → correct → click → correct
    → click → correct) couldn't trigger demotion because intervening
    successes reset the counter. After F10, the corrections counter is
    separate and accumulates.
    """
    am_before = setup["store"].get("test_app.desktop")
    primary_before = am_before.intents[0].handler_selectors[0]
    secondary_before = am_before.intents[0].handler_selectors[1]
    assert primary_before.kind == "hotkey"
    assert secondary_before.kind == "uia"

    # 3 cycles of dispatch-then-correct — the realistic user flow.
    for _ in range(3):
        setup["disp"].dispatch(
            app_id="test_app.desktop", intent_id="play",
            slots={}, active_window="TestApp",
        )
        setup["disp"].record_correction(
            app_id="test_app.desktop", intent_id="play",
        )

    am_after = setup["store"].get("test_app.desktop")
    # Demotion fired: secondary (uia) moved to index 0, primary (hotkey) to 1.
    assert am_after.intents[0].handler_selectors[0].kind == "uia"
    assert am_after.intents[0].handler_selectors[1].kind == "hotkey"
    # Both counters reset by the demotion.
    assert am_after.intents[0].health.consecutive_corrections == 0
    assert am_after.intents[0].health.consecutive_failures == 0


def test_demotion_resets_both_counters(setup):
    """When demotion fires from EITHER counter reaching 3, BOTH reset.

    Locks the v1 behaviour: a flip-flop driven by user corrections should
    not leave latent failure-counter pressure that would re-fire the swap
    on the next miss — and vice versa.
    """
    # Stuff the failure counter to 2 via two hard dispatch-level failures
    # (BOTH selectors must fail so the success path never resets the counter).
    setup["term"].send_key = lambda k: (_ for _ in ()).throw(RuntimeError("nope"))
    # Don't seed elements — the uia secondary also fails.
    for _ in range(2):
        setup["disp"].dispatch(
            app_id="test_app.desktop", intent_id="play",
            slots={}, active_window="TestApp",
        )
    health_mid = setup["store"].get("test_app.desktop").intents[0].health
    assert health_mid.consecutive_failures == 2
    assert health_mid.consecutive_corrections == 0

    # Now drive demotion from the corrections counter.
    for _ in range(3):
        setup["disp"].record_correction(
            app_id="test_app.desktop", intent_id="play",
        )

    health_after = setup["store"].get("test_app.desktop").intents[0].health
    # Demotion swept BOTH counters to 0, not just the one that triggered.
    assert health_after.consecutive_failures == 0
    assert health_after.consecutive_corrections == 0
