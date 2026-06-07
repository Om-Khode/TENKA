"""Tests for manifest-based daily vision call counter."""

import pytest

from assistant.automation.vision_cap import (
    VisionCapTracker,
    DEFAULT_DAILY_CAP,
)
from assistant.storage.db import Database, _reset_for_testing


@pytest.fixture
def tracker(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield VisionCapTracker(db._conn)
    db.close()
    _reset_for_testing()


def test_increments_within_cap(tracker):
    assert tracker.calls_today() == 0
    assert tracker.try_increment() is True
    assert tracker.calls_today() == 1


def test_blocks_at_cap(tracker):
    # Burn through the cap
    for _ in range(DEFAULT_DAILY_CAP):
        assert tracker.try_increment() is True
    # Next call must be blocked
    assert tracker.try_increment() is False
    assert tracker.calls_today() == DEFAULT_DAILY_CAP


def test_reset_purges_stale_day_rows_but_preserves_today(tracker):
    """reset_for_new_day clears yesterday and older rows; today's row is left alone.

    The production reset fires at midnight after a date rollover, so the
    'today' row hasn't been written yet — the function's job is to purge
    accumulated stale rows from prior days, not to reset the in-progress
    counter for the current day.
    """
    # Insert a fake "yesterday" row directly via the tracker's connection
    tracker._db.execute(
        "INSERT INTO vision_calls (day, count) VALUES (?, ?)",
        ("2026-05-30", 42),
    )
    # Also increment today (creates today's row at count=1)
    assert tracker.try_increment() is True
    assert tracker.calls_today() == 1

    # Verify both rows are present pre-reset
    rows_before = tracker._db.execute(
        "SELECT day, count FROM vision_calls ORDER BY day"
    ).fetchall()
    assert len(rows_before) == 2

    tracker.reset_for_new_day()

    # Yesterday gone; today preserved
    rows_after = tracker._db.execute(
        "SELECT day, count FROM vision_calls ORDER BY day"
    ).fetchall()
    assert len(rows_after) == 1
    assert rows_after[0][0] == tracker._today_key()
    assert tracker.calls_today() == 1


def test_reset_is_noop_with_no_row(tracker):
    """reset_for_new_day on a fresh DB must not raise; calls_today stays 0."""
    assert tracker.calls_today() == 0
    tracker.reset_for_new_day()
    assert tracker.calls_today() == 0
