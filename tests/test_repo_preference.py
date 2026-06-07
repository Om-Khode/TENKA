"""Tests for storage/repos/preference.py — PreferenceRepo."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.preference import (
    PreferenceRepo,
    CONFIDENCE_SILENT,
    CONFIDENCE_ASK,
    CONFIDENCE_APPLIED_NO_COMPLAINT,
    CONFIDENCE_APPLIED_OVERRIDDEN,
    DECAY_AMOUNT,
    MIN_CONFIDENCE_BEFORE_PRUNE,
)


@pytest.fixture
def repo(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield PreferenceRepo(db)
    db.close()
    _reset_for_testing()


# --- set / get ---


def test_set_and_get(repo):
    repo.set_preference("music_app", "spotify", "app_routing", 0.5, "reflection", "Observed 3x")
    pref = repo.get_preference("music_app")
    assert pref is not None
    assert pref["key"] == "music_app"
    assert pref["value"] == "spotify"
    assert pref["category"] == "app_routing"
    assert pref["confidence"] == 0.5
    assert pref["source"] == "reflection"


def test_get_missing_returns_none(repo):
    assert repo.get_preference("nonexistent") is None


def test_get_with_category_filter(repo):
    repo.set_preference("music_app", "spotify", "app_routing", 0.5, "reflection", "r")
    repo.set_preference("music_app", "vlc", "media", 0.6, "explicit", "r")
    pref = repo.get_preference("music_app", category="media")
    # category filter only applies when key+category both match;
    # but key is PRIMARY KEY so second set_preference overwrites the first
    # (key is unique regardless of category).
    # The result depends on upsert behavior: last write wins.
    assert pref is None  # category changed to "media", so "app_routing" filter returns None
    pref2 = repo.get_preference("music_app", category="media")
    assert pref2 is not None
    assert pref2["value"] == "vlc"


def test_upsert_updates_existing(repo):
    repo.set_preference("browser", "chrome", "app_routing", 0.4, "reflection", "first")
    repo.set_preference("browser", "firefox", "app_routing", 0.8, "correction", "user corrected")
    pref = repo.get_preference("browser")
    assert pref["value"] == "firefox"
    assert pref["confidence"] == 0.8
    assert pref["source"] == "correction"


# --- get_preferences_by_category ---


def test_get_by_category_sorted(repo):
    repo.set_preference("a", "v1", "app_routing", 0.3, "reflection", "r")
    repo.set_preference("b", "v2", "app_routing", 0.9, "explicit", "r")
    repo.set_preference("c", "v3", "app_routing", 0.6, "reflection", "r")
    results = repo.get_preferences_by_category("app_routing")
    assert len(results) == 3
    # Sorted by confidence descending
    confidences = [r["confidence"] for r in results]
    assert confidences == sorted(confidences, reverse=True)


def test_get_by_category_empty(repo):
    assert repo.get_preferences_by_category("nonexistent") == []


# --- get_active_preferences ---


def test_get_active_default_threshold(repo):
    repo.set_preference("a", "v1", "cat", 0.3, "reflection", "r")
    repo.set_preference("b", "v2", "cat", 0.7, "explicit", "r")
    repo.set_preference("c", "v3", "cat", 0.9, "explicit", "r")
    active = repo.get_active_preferences()  # default min_confidence=0.7
    assert len(active) == 2
    keys = {p["key"] for p in active}
    assert keys == {"b", "c"}


def test_get_active_custom_threshold(repo):
    repo.set_preference("a", "v1", "cat", 0.3, "reflection", "r")
    repo.set_preference("b", "v2", "cat", 0.5, "reflection", "r")
    active = repo.get_active_preferences(min_confidence=0.4)
    assert len(active) == 1
    assert active[0]["key"] == "b"


# --- get_all_preferences ---


def test_get_all(repo):
    repo.set_preference("x", "1", "catA", 0.5, "reflection", "r")
    repo.set_preference("y", "2", "catB", 0.3, "reflection", "r")
    all_prefs = repo.get_all_preferences()
    assert len(all_prefs) == 2


# --- bump_confidence ---


def test_bump_confidence_normal(repo):
    repo.set_preference("k", "v", "cat", 0.5, "reflection", "r")
    new_conf = repo.bump_confidence("k", delta=0.1)
    assert new_conf == pytest.approx(0.6, abs=0.001)
    pref = repo.get_preference("k")
    assert pref["confidence"] == pytest.approx(0.6, abs=0.001)


def test_bump_confidence_cap_at_1(repo):
    repo.set_preference("k", "v", "cat", 0.95, "reflection", "r")
    new_conf = repo.bump_confidence("k", delta=0.2)
    assert new_conf == pytest.approx(1.0, abs=0.001)


def test_bump_confidence_missing_key(repo):
    result = repo.bump_confidence("nonexistent")
    assert result is None


def test_bump_confidence_no_change(repo):
    repo.set_preference("k", "v", "cat", 1.0, "reflection", "r")
    result = repo.bump_confidence("k", delta=0.0)
    assert result == pytest.approx(1.0, abs=0.001)


# --- decay_preference ---


def test_decay_normal(repo):
    repo.set_preference("k", "v", "cat", 0.7, "reflection", "r")
    new_conf = repo.decay_preference("k", delta=0.1)
    assert new_conf == pytest.approx(0.6, abs=0.001)


def test_decay_floor(repo):
    repo.set_preference("k", "v", "cat", 0.2, "reflection", "r")
    new_conf = repo.decay_preference("k", delta=0.5)
    assert new_conf == pytest.approx(MIN_CONFIDENCE_BEFORE_PRUNE, abs=0.001)


def test_decay_missing_key(repo):
    assert repo.decay_preference("nonexistent") is None


def test_decay_no_change_at_floor(repo):
    repo.set_preference("k", "v", "cat", MIN_CONFIDENCE_BEFORE_PRUNE, "reflection", "r")
    result = repo.decay_preference("k", delta=0.05)
    # Already at floor, no change
    assert result == pytest.approx(MIN_CONFIDENCE_BEFORE_PRUNE, abs=0.001)


# --- record_preference_used ---


def test_record_used(repo):
    repo.set_preference("k", "v", "cat", 0.5, "reflection", "r")
    repo.record_preference_used("k")
    pref = repo.get_preference("k")
    assert pref["times_used"] == 1
    # Confidence should have bumped by CONFIDENCE_APPLIED_NO_COMPLAINT (0.05)
    assert pref["confidence"] == pytest.approx(0.55, abs=0.001)


def test_record_used_missing_key(repo):
    # Should not raise
    repo.record_preference_used("nonexistent")


# --- record_preference_overridden ---


def test_record_overridden(repo):
    repo.set_preference("k", "v", "cat", 0.8, "reflection", "r")
    repo.record_preference_overridden("k")
    pref = repo.get_preference("k")
    assert pref["times_overridden"] == 1
    # Confidence should have dropped by CONFIDENCE_APPLIED_OVERRIDDEN (-0.2)
    assert pref["confidence"] == pytest.approx(0.6, abs=0.001)


def test_record_overridden_missing_key(repo):
    # Should not raise
    repo.record_preference_overridden("nonexistent")


# --- get_preference_context_block ---


def test_context_block_empty(repo):
    assert repo.get_preference_context_block() == ""


def test_context_block_with_data(repo):
    repo.set_preference("browser", "chrome", "app_routing", 0.8, "explicit", "r")
    repo.set_preference("lang", "en", "response_style", 0.5, "reflection", "r")
    repo.set_preference("ignored", "x", "other", 0.2, "reflection", "r")  # below ASK
    block = repo.get_preference_context_block()
    assert "--- User Preferences ---" in block
    assert "browser: chrome (confirmed)" in block
    assert "lang: en (probable)" in block
    assert "ignored" not in block  # below CONFIDENCE_ASK


def test_context_block_categories_grouped(repo):
    repo.set_preference("a", "v1", "catA", 0.8, "explicit", "r")
    repo.set_preference("b", "v2", "catB", 0.5, "reflection", "r")
    block = repo.get_preference_context_block()
    assert "[catA]" in block
    assert "[catB]" in block


# --- get_preference_history ---


def test_history_returns_recent(repo):
    repo.set_preference("k", "v", "cat", 0.5, "reflection", "first")
    repo.set_preference("k", "v2", "cat", 0.8, "correction", "updated")
    history = repo.get_preference_history(days=30)
    assert len(history) == 2
    # Newest first
    assert history[0]["new_value"] == "v2"
    assert history[1]["new_value"] == "v"


def test_history_empty(repo):
    assert repo.get_preference_history() == []


# --- decay_unused_preferences ---


def test_decay_unused_skips_recent(repo):
    repo.set_preference("k", "v", "cat", 0.7, "reflection", "r")
    # Just created, so updated_at is now — should not decay
    decayed = repo.decay_unused_preferences()
    assert decayed == 0


def test_decay_unused_decays_old(repo):
    repo.set_preference("k", "v", "cat", 0.7, "reflection", "r")
    # Manually backdate the updated_at to 31 days ago
    old_date = (datetime.now() - timedelta(days=31)).isoformat()
    repo._db.execute(
        "UPDATE user_preferences SET updated_at = ? WHERE key = ?",
        (old_date, "k"),
    )
    repo._db.commit()
    decayed = repo.decay_unused_preferences()
    assert decayed == 1
    pref = repo.get_preference("k")
    assert pref["confidence"] == pytest.approx(0.65, abs=0.001)


def test_decay_unused_skips_already_at_floor(repo):
    repo.set_preference("k", "v", "cat", MIN_CONFIDENCE_BEFORE_PRUNE, "reflection", "r")
    old_date = (datetime.now() - timedelta(days=31)).isoformat()
    repo._db.execute(
        "UPDATE user_preferences SET updated_at = ? WHERE key = ?",
        (old_date, "k"),
    )
    repo._db.commit()
    decayed = repo.decay_unused_preferences()
    assert decayed == 0  # already at or below floor


# --- reset ---


def test_reset(repo):
    repo.set_preference("a", "v1", "cat", 0.5, "reflection", "r")
    repo.set_preference("b", "v2", "cat", 0.8, "explicit", "r")
    repo.reset_preferences()
    assert repo.get_all_preferences() == []
    assert repo.get_preference_history(days=3650) == []


# --- confidence clamping ---


def test_set_preference_clamps_above_1(repo):
    repo.set_preference("k", "v", "cat", 1.5, "reflection", "r")
    pref = repo.get_preference("k")
    assert pref["confidence"] == 1.0


def test_set_preference_clamps_below_0(repo):
    repo.set_preference("k", "v", "cat", -0.5, "reflection", "r")
    pref = repo.get_preference("k")
    assert pref["confidence"] == 0.0
