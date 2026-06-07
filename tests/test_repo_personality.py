"""Tests for storage/repos/personality.py — PersonalityRepo."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.personality import (
    PersonalityRepo,
    TRAIT_DEFAULTS,
    MAX_DELTA_PER_CYCLE,
    MAX_DELTA_PER_EVENT,
)


@pytest.fixture
def repo(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield PersonalityRepo(db)
    db.close()
    _reset_for_testing()


# --- Seeding ---


def test_seed_defaults_on_first_run(repo):
    """Constructor seeds all 6 default traits."""
    traits = repo.get_current_traits()
    assert len(traits) == 6
    for name, vals in TRAIT_DEFAULTS.items():
        assert name in traits
        assert traits[name] == round(vals["initial"], 4)


def test_seed_conversation_counter(repo):
    """Constructor seeds conversation_count metadata to 0."""
    assert repo.get_conversation_count() == 0


def test_seed_last_reflection_at(repo):
    """Constructor seeds last_reflection_at metadata."""
    val = repo.get_metadata("last_reflection_at")
    assert val is not None
    # Should be a valid ISO timestamp
    datetime.fromisoformat(val)


def test_seed_idempotent(tmp_path):
    """Creating a second PersonalityRepo on the same DB does not duplicate rows."""
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    repo1 = PersonalityRepo(db)
    repo2 = PersonalityRepo(db)
    traits = repo2.get_current_traits()
    assert len(traits) == 6
    assert repo2.get_conversation_count() == 0
    db.close()
    _reset_for_testing()


# --- Read Operations ---


def test_get_current_traits(repo):
    """get_current_traits returns dict of name→float."""
    traits = repo.get_current_traits()
    assert isinstance(traits, dict)
    assert "trust" in traits
    assert isinstance(traits["trust"], float)


def test_get_full_trait_info(repo):
    """get_full_trait_info returns value, floor, ceiling per trait."""
    info = repo.get_full_trait_info()
    assert len(info) == 6
    for name, detail in info.items():
        assert "value" in detail
        assert "floor" in detail
        assert "ceiling" in detail
        assert detail["floor"] == TRAIT_DEFAULTS[name]["floor"]
        assert detail["ceiling"] == TRAIT_DEFAULTS[name]["ceiling"]


# --- update_traits ---


def test_update_traits_normal(repo):
    """Small delta within bounds applies correctly."""
    changed = repo.update_traits({"trust": 0.03}, "test reason")
    assert "trust" in changed
    expected = round(TRAIT_DEFAULTS["trust"]["initial"] + 0.03, 4)
    assert changed["trust"] == expected
    # Persisted
    assert repo.get_current_traits()["trust"] == expected


def test_update_traits_clamp_delta_cycle(repo):
    """Delta exceeding MAX_DELTA_PER_CYCLE is clamped."""
    changed = repo.update_traits({"trust": 0.5}, "big delta")
    assert "trust" in changed
    expected = round(TRAIT_DEFAULTS["trust"]["initial"] + MAX_DELTA_PER_CYCLE, 4)
    assert changed["trust"] == expected


def test_update_traits_clamp_delta_event(repo):
    """Event trigger uses MAX_DELTA_PER_EVENT cap."""
    changed = repo.update_traits(
        {"trust": 0.5}, "event bump", trigger="event"
    )
    assert "trust" in changed
    expected = round(TRAIT_DEFAULTS["trust"]["initial"] + MAX_DELTA_PER_EVENT, 4)
    assert changed["trust"] == expected


def test_update_traits_respect_ceiling(repo):
    """Value cannot exceed ceiling even with repeated bumps."""
    ceiling = TRAIT_DEFAULTS["sass"]["ceiling"]
    # Sass starts at 0.75, ceiling 0.95 — push it way up
    for _ in range(20):
        repo.update_traits({"sass": MAX_DELTA_PER_CYCLE}, "push up")
    current = repo.get_current_traits()["sass"]
    assert current <= ceiling


def test_update_traits_respect_floor(repo):
    """Value cannot drop below floor even with repeated decrements."""
    floor = TRAIT_DEFAULTS["openness"]["floor"]
    # Openness starts at 0.20, floor 0.05 — push it way down
    for _ in range(20):
        repo.update_traits({"openness": -MAX_DELTA_PER_CYCLE}, "push down")
    current = repo.get_current_traits()["openness"]
    assert current >= floor


def test_update_traits_ignore_unknown(repo):
    """Unknown trait names are silently ignored."""
    changed = repo.update_traits({"nonexistent": 0.03}, "test")
    assert changed == {}


def test_update_traits_logging(repo):
    """Each change is recorded in trait history."""
    repo.update_traits({"trust": 0.03}, "test log", trigger="event")
    history = repo.get_trait_history(days=1)
    assert len(history) == 1
    entry = history[0]
    assert entry["trait"] == "trust"
    assert entry["reason"] == "test log"
    assert entry["trigger"] == "event"


def test_update_traits_zero_delta_skipped(repo):
    """Zero delta produces no change and no log entry."""
    changed = repo.update_traits({"trust": 0.0}, "zero")
    assert changed == {}
    assert repo.get_trait_history(days=1) == []


def test_update_traits_negative_delta(repo):
    """Negative delta decreases trait value."""
    changed = repo.update_traits({"trust": -0.03}, "decrease")
    assert "trust" in changed
    expected = round(TRAIT_DEFAULTS["trust"]["initial"] - 0.03, 4)
    assert changed["trust"] == expected


# --- reset_traits ---


def test_reset_traits(repo):
    """Reset restores all traits to initial values and logs changes."""
    # Move traits away from defaults
    repo.update_traits({"trust": 0.05, "sass": -0.05}, "shift")
    repo.reset_traits()
    traits = repo.get_current_traits()
    for name, vals in TRAIT_DEFAULTS.items():
        assert traits[name] == round(vals["initial"], 4)
    # Should have log entries for the reset
    history = repo.get_trait_history(days=1)
    manual_entries = [e for e in history if e["trigger"] == "manual"]
    assert len(manual_entries) >= 1


# --- Conversation Counter ---


def test_increment_conversation_count(repo):
    """Increment returns new count."""
    assert repo.increment_conversation_count() == 1
    assert repo.increment_conversation_count() == 2
    assert repo.get_conversation_count() == 2


def test_reset_conversation_count(repo):
    """Reset sets counter back to 0."""
    repo.increment_conversation_count()
    repo.increment_conversation_count()
    repo.reset_conversation_count()
    assert repo.get_conversation_count() == 0


# --- Metadata ---


def test_set_and_get_metadata(repo):
    """set_metadata + get_metadata round-trips."""
    repo.set_metadata("test_key", "test_value")
    assert repo.get_metadata("test_key") == "test_value"


def test_metadata_upsert(repo):
    """set_metadata overwrites existing value."""
    repo.set_metadata("test_key", "v1")
    repo.set_metadata("test_key", "v2")
    assert repo.get_metadata("test_key") == "v2"


def test_metadata_missing_returns_none(repo):
    """get_metadata returns None for missing key."""
    assert repo.get_metadata("nonexistent") is None


# --- History ---


def test_history_empty(repo):
    """History is empty when no changes have been made."""
    assert repo.get_trait_history(days=1) == []


def test_history_returns_recent(repo):
    """History returns entries from within the time window."""
    repo.update_traits({"trust": 0.02}, "recent change")
    history = repo.get_trait_history(days=1)
    assert len(history) == 1
    assert history[0]["trait"] == "trust"
    assert history[0]["reason"] == "recent change"
