"""Tests for manifest-based promoter — N=2 gate, clustering, verifier, YAML writer."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from assistant.automation.manifest_schema import load_manifest_from_yaml
from assistant.automation.manifest_store import ManifestStore
from assistant.automation.promoter import (
    Promoter,
    _entry_epoch,
    count_distinct_buckets,
    group_automation_cache_entries_by_app,
)
from assistant.storage.db import Database
from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo
from assistant.storage.repos.automation_cache import AutomationCacheRepo


# ─── Async test helper ────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


# ─── Pure helpers ─────────────────────────────────────────────────────────

def test_count_distinct_buckets_collapses_same_bucket():
    """Two entries in the same 30-min bucket collapse to one bucket."""
    # ts=1000.0 (16:46:40 UTC bucket 0) and ts=1010.0 share a bucket;
    # ts=5000.0 (~1:23:20 later) is bucket 2.
    entries = [
        {"goal_slug": "play_music",
         "created_at": datetime.fromtimestamp(1000.0).isoformat()},
        {"goal_slug": "play_music",
         "created_at": datetime.fromtimestamp(1010.0).isoformat()},
        {"goal_slug": "play_music",
         "created_at": datetime.fromtimestamp(5000.0).isoformat()},
    ]
    assert count_distinct_buckets(entries) == 2


def test_count_distinct_buckets_distinct_slugs():
    """Same time bucket but distinct slugs → 2 buckets."""
    entries = [
        {"goal_slug": "play_music",
         "created_at": datetime.fromtimestamp(1000.0).isoformat()},
        {"goal_slug": "music_play_song",
         "created_at": datetime.fromtimestamp(1010.0).isoformat()},
    ]
    assert count_distinct_buckets(entries) == 2


def test_count_distinct_buckets_handles_numeric_created_at():
    """Defensive: accept legacy float epoch in created_at."""
    entries = [
        {"goal_slug": "a", "created_at": 1000.0},
        {"goal_slug": "a", "created_at": 99999.0},
    ]
    assert count_distinct_buckets(entries) == 2


def test_count_distinct_buckets_skips_malformed_created_at():
    """Defensive: malformed values don't crash; they fall to bucket 0."""
    entries = [
        {"goal_slug": "a", "created_at": "not-a-date"},
        {"goal_slug": "a", "created_at": "also-bad"},
    ]
    # Both fall to epoch 0.0 → same (slug, bucket); collapses to 1.
    assert count_distinct_buckets(entries) == 1


def test_entry_epoch_iso_string():
    # Use tz-aware ISO so the round-trip is unambiguous on all platforms.
    # Naive ISO would have local-tz semantics on Windows and UTC semantics
    # under the post-fix _entry_epoch — pinning UTC removes that ambiguity.
    iso = datetime.fromtimestamp(1234.0, tz=timezone.utc).isoformat()
    assert _entry_epoch({"created_at": iso}) == pytest.approx(1234.0)


def test_entry_epoch_numeric_passthrough():
    assert _entry_epoch({"created_at": 42.0}) == 42.0
    assert _entry_epoch({"created_at": 99}) == 99.0


def test_entry_epoch_missing_or_bad_returns_zero():
    assert _entry_epoch({}) == 0.0
    assert _entry_epoch({"created_at": "garbage"}) == 0.0
    assert _entry_epoch({"created_at": None}) == 0.0


def test_group_automation_cache_entries_by_app():
    """Filters by allowed backend; groups by app_name."""
    entries = [
        {"backend": "native", "app_name": "media_player", "goal_slug": "a"},
        {"backend": "native", "app_name": "media_player", "goal_slug": "b"},
        {"backend": "browser", "app_name": "example.com", "goal_slug": "c"},
        {"backend": "native", "app_name": "text_editor", "goal_slug": "d"},
    ]
    grouped = group_automation_cache_entries_by_app(entries, allowed_backends={"native"})
    assert set(grouped.keys()) == {"media_player", "text_editor"}
    assert len(grouped["media_player"]) == 2
    assert len(grouped["text_editor"]) == 1


# ─── Full cycle fixture ───────────────────────────────────────────────────

@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Real Database + repos + ManifestStore + Promoter with fake LLM contracts."""
    db = Database(tmp_path / "test.db")
    ac1 = AutomationCacheRepo(db)
    idx = AppManifestIndexRepo(db._conn)
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    store = ManifestStore(manifests_dir=manifests_dir, index_repo=idx)
    store.scan_and_index()
    promoter = Promoter(automation_cache_repo=ac1, manifest_store=store)

    canned: dict = {}

    async def _fake_cluster(*, app, goals):
        return canned.get(("cluster", app), [])

    async def _fake_verify(*, traces):
        return canned.get("verify", {
            "primary_primitive": {"kind": "hotkey", "keys": "Space"},
            "alternatives": [],
            "confidence": "high",
            "diff_notes": "",
        })

    async def _fake_synth(*, intent_id, originals):
        return canned.get(("synth", intent_id), ["press play"])

    monkeypatch.setattr(
        "assistant.automation.promoter.ask_for_intent_clustering", _fake_cluster
    )
    monkeypatch.setattr(
        "assistant.automation.promoter.ask_for_trace_diff_verification", _fake_verify
    )
    monkeypatch.setattr(
        "assistant.automation.promoter.ask_for_phrase_synthesis", _fake_synth
    )

    yield {
        "ac1": ac1, "db": db, "store": store, "promoter": promoter,
        "canned": canned, "manifests_dir": manifests_dir,
    }
    db.close()


def _seed_ac1(db, *, backend: str, app_name: str, slug: str, ts: float,
              steps: list[dict]) -> None:
    """Insert directly into automation_cache to control created_at exactly."""
    created_at = datetime.fromtimestamp(ts).isoformat()
    db.execute(
        """INSERT INTO automation_cache
           (backend, app_name, goal_slug, goal_text, steps_json,
            hit_count, created_at, last_hit_at, version)
           VALUES (?, ?, ?, ?, ?, 0, ?, ?, 1)""",
        (backend, app_name, slug, slug.replace("_", " "),
         json.dumps(steps), created_at, created_at),
    )
    db.commit()


# ─── Full cycle: gate, write, low-conf skip ───────────────────────────────

def test_promotion_skips_when_below_threshold(setup):
    """Single entry → cannot meet N=2 bucket gate → no promotion."""
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="play_music", ts=1000.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])

    summary = _run(setup["promoter"].run_once())
    assert summary["intents_promoted"] == 0


def test_promotion_writes_yaml_when_high_conf_cluster(setup):
    """2 entries in distinct buckets, high-conf cluster → YAML written."""
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="play_music", ts=1000.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="music_play_start", ts=99999.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])

    setup["canned"][("cluster", "test_app")] = [{
        "intent_id": "play",
        "members": ["play_music", "music_play_start"],
        "phrases": ["play music", "start music"],
        "confidence": "high",
    }]

    summary = _run(setup["promoter"].run_once())
    assert summary["intents_promoted"] == 1
    assert summary["apps_processed"] == 1

    yaml_path = setup["manifests_dir"] / "test_app.desktop.yaml"
    assert yaml_path.exists()

    am = load_manifest_from_yaml(yaml_path)
    assert am.app_id == "test_app.desktop"
    assert len(am.intents) == 1
    intent = am.intents[0]
    assert intent.id == "play"
    assert "press play" in intent.phrases     # synthesized
    assert "play music" in intent.phrases     # original
    assert "start music" in intent.phrases    # original

    # automation-cache rows are claimed.
    assert setup["ac1"].find_unpromoted() == []


def test_promotion_skips_low_confidence(setup):
    """confidence='low' cluster → no promotion, no YAML."""
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="a", ts=1.0, steps=[])
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="b", ts=99999.0, steps=[])

    setup["canned"][("cluster", "test_app")] = [{
        "intent_id": "a", "members": ["a", "b"], "phrases": ["a"],
        "confidence": "low",
    }]

    summary = _run(setup["promoter"].run_once())
    assert summary["intents_promoted"] == 0

    yaml_path = setup["manifests_dir"] / "test_app.desktop.yaml"
    assert not yaml_path.exists()
    # automation-cache entries remain unclaimed for later attempts.
    assert len(setup["ac1"].find_unpromoted()) == 2


def test_promotion_skips_non_native_backend(setup):
    """Browser-backend entries are ignored (v1: native only)."""
    _seed_ac1(setup["db"], backend="browser", app_name="example.com",
              slug="a", ts=1.0, steps=[])
    _seed_ac1(setup["db"], backend="browser", app_name="example.com",
              slug="b", ts=99999.0, steps=[])

    summary = _run(setup["promoter"].run_once())
    assert summary["intents_promoted"] == 0
    assert summary["apps_processed"] == 0


def test_promotion_per_app_exception_does_not_block_cycle(setup, monkeypatch):
    """A failure in one app's _process_app must not stop other apps."""
    _seed_ac1(setup["db"], backend="native", app_name="bad_app",
              slug="a", ts=1.0, steps=[])
    _seed_ac1(setup["db"], backend="native", app_name="bad_app",
              slug="b", ts=99999.0, steps=[])
    _seed_ac1(setup["db"], backend="native", app_name="good_app",
              slug="play_music", ts=1.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    _seed_ac1(setup["db"], backend="native", app_name="good_app",
              slug="music_play_start", ts=99999.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])

    setup["canned"][("cluster", "good_app")] = [{
        "intent_id": "play",
        "members": ["play_music", "music_play_start"],
        "phrases": ["play music"],
        "confidence": "high",
    }]

    # bad_app crashes the cluster contract.
    async def _selective_cluster(*, app, goals):
        if app == "bad_app":
            raise RuntimeError("simulated LLM blowup")
        return setup["canned"].get(("cluster", app), [])

    monkeypatch.setattr(
        "assistant.automation.promoter.ask_for_intent_clustering",
        _selective_cluster,
    )

    summary = _run(setup["promoter"].run_once())
    # good_app still promotes.
    assert summary["intents_promoted"] == 1
    yaml_path = setup["manifests_dir"] / "good_app.desktop.yaml"
    assert yaml_path.exists()


def test_promotion_verifier_no_primary_skips_intent(setup):
    """If trace verifier returns no primary primitive, intent is skipped."""
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="play_music", ts=1.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="music_play_start", ts=99999.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])

    setup["canned"][("cluster", "test_app")] = [{
        "intent_id": "play",
        "members": ["play_music", "music_play_start"],
        "phrases": ["play music"],
        "confidence": "high",
    }]
    setup["canned"]["verify"] = {
        "primary_primitive": None, "alternatives": [],
        "confidence": "low", "diff_notes": "no convergence",
    }

    summary = _run(setup["promoter"].run_once())
    assert summary["intents_promoted"] == 0


def test_promotion_merges_into_existing_manifest(setup):
    """Re-running with a new intent appends rather than replacing."""
    # First cycle: promote intent "play".
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="play_music", ts=1.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="music_play_start", ts=99999.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    setup["canned"][("cluster", "test_app")] = [{
        "intent_id": "play",
        "members": ["play_music", "music_play_start"],
        "phrases": ["play music"],
        "confidence": "high",
    }]
    _run(setup["promoter"].run_once())

    # Second cycle: a new pause intent on the same app.
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="pause_music", ts=200000.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="music_stop", ts=300000.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    setup["canned"][("cluster", "test_app")] = [{
        "intent_id": "pause",
        "members": ["pause_music", "music_stop"],
        "phrases": ["pause music"],
        "confidence": "high",
    }]
    _run(setup["promoter"].run_once())

    am = load_manifest_from_yaml(
        setup["manifests_dir"] / "test_app.desktop.yaml"
    )
    intent_ids = {i.id for i in am.intents}
    assert intent_ids == {"play", "pause"}


def test_merge_intent_bumps_version_on_repromote(setup):
    """Re-promoting the same intent_id bumps Intent.version on the manifest.

    Seeds 2 distinct-bucket entries → promote → version=1. Then resets
    promoted_intent_id back to NULL so the rows look unpromoted again
    and re-runs. The merge path should detect the existing intent and
    bump version to 2.
    """
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="play_music", ts=1000.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])
    _seed_ac1(setup["db"], backend="native", app_name="test_app",
              slug="music_play_start", ts=99999.0,
              steps=[{"action": "press_key", "params": {"key": "Space"}}])

    setup["canned"][("cluster", "test_app")] = [{
        "intent_id": "play",
        "members": ["play_music", "music_play_start"],
        "phrases": ["play music"],
        "confidence": "high",
    }]

    summary = _run(setup["promoter"].run_once())
    assert summary["intents_promoted"] == 1

    am = load_manifest_from_yaml(
        setup["manifests_dir"] / "test_app.desktop.yaml"
    )
    assert len(am.intents) == 1
    assert am.intents[0].id == "play"
    assert am.intents[0].version == 1

    # Reset promoted_intent_id so find_unpromoted picks them up again.
    setup["db"].execute(
        "UPDATE automation_cache SET promoted_intent_id = NULL "
        "WHERE app_name = ?",
        ("test_app",),
    )
    setup["db"].commit()
    assert len(setup["ac1"].find_unpromoted()) == 2

    summary2 = _run(setup["promoter"].run_once())
    assert summary2["intents_promoted"] == 1

    am2 = load_manifest_from_yaml(
        setup["manifests_dir"] / "test_app.desktop.yaml"
    )
    assert len(am2.intents) == 1
    assert am2.intents[0].id == "play"
    assert am2.intents[0].version == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
