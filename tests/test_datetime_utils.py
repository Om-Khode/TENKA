"""Tests for core/datetime_utils.py's humanize_relative()."""
from datetime import datetime, timedelta, timezone

from assistant.core.datetime_utils import humanize_relative


def _iso(ago: timedelta) -> str:
    return (datetime.now(timezone.utc) - ago).isoformat()


def test_just_now():
    assert humanize_relative(_iso(timedelta(seconds=30))) == "just now"


def test_minutes_ago():
    assert humanize_relative(_iso(timedelta(minutes=10))) == "10 minutes ago"


def test_hours_ago():
    assert humanize_relative(_iso(timedelta(hours=3))) == "3 hours ago"


def test_days_ago():
    assert humanize_relative(_iso(timedelta(days=2))) == "2 days ago"


def test_naive_timestamp_treated_as_utc():
    naive = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None).isoformat()
    assert humanize_relative(naive) == "2 hours ago"


def test_garbage_input_does_not_raise():
    assert humanize_relative("not a timestamp") == "at an unknown time"


def test_none_input_does_not_raise():
    assert humanize_relative(None) == "at an unknown time"
