"""Tests for storage/repos/settings.py — SettingsRepo."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.settings import SettingsRepo


@pytest.fixture
def repo(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield SettingsRepo(db)
    db.close()
    _reset_for_testing()


def test_get_missing_returns_default(repo):
    assert repo.get("nonexistent") is None
    assert repo.get("nonexistent", 42) == 42


def test_set_and_get_string(repo):
    repo.set("theme", "dark")
    assert repo.get("theme") == "dark"


def test_set_and_get_bool(repo):
    repo.set("verbose", True)
    assert repo.get("verbose") is True


def test_set_and_get_int(repo):
    repo.set("timeout", 30)
    assert repo.get("timeout") == 30


def test_set_and_get_float(repo):
    repo.set("rate", 1.5)
    assert repo.get("rate") == 1.5


def test_set_and_get_list(repo):
    repo.set("items", [1, 2, 3])
    assert repo.get("items") == [1, 2, 3]


def test_set_and_get_dict(repo):
    repo.set("config", {"a": 1})
    assert repo.get("config") == {"a": 1}


def test_upsert_overwrites(repo):
    repo.set("key", "old")
    repo.set("key", "new")
    assert repo.get("key") == "new"


def test_delete_existing(repo):
    repo.set("key", "value")
    assert repo.delete("key") is True
    assert repo.get("key") is None


def test_delete_nonexistent(repo):
    assert repo.delete("nope") is False


def test_list_all_empty(repo):
    assert repo.list_all() == {}


def test_list_all(repo):
    repo.set("a", 1)
    repo.set("b", "two")
    result = repo.list_all()
    assert result == {"a": 1, "b": "two"}


def test_corrupt_value_returns_default(repo):
    repo._db.execute(
        "INSERT INTO runtime_settings (key, value, updated_at) VALUES (?, ?, ?)",
        ("bad", "not-json{{{", "2026-01-01"),
    )
    repo._db.commit()
    assert repo.get("bad", "fallback") == "fallback"
