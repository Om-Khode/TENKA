"""Tests for storage/repos/shortcut.py — ShortcutRepo."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.shortcut import ShortcutRepo

_TEST_INTENTS = ["planner", "camera_look", "web_search", "code_executor"]


@pytest.fixture
def repo(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield ShortcutRepo(db, assistant_name_lower="tenka", intents=_TEST_INTENTS)
    db.close()
    _reset_for_testing()


def test_create_and_get(repo):
    assert repo.create_shortcut("setup", "planner", {"goal": "open vscode"}, "dev setup")
    result = repo.get_shortcut("setup")
    assert result is not None
    assert result["intent"] == "planner"
    assert result["params"] == {"goal": "open vscode"}


def test_match_exact(repo):
    repo.create_shortcut("pikachu", "camera_look")
    match = repo.match_shortcut("pikachu")
    assert match is not None
    assert match["intent"] == "camera_look"


def test_match_with_filler(repo):
    repo.create_shortcut("setup", "planner")
    assert repo.match_shortcut("setup please") is not None
    assert repo.match_shortcut("please setup") is not None


def test_match_case_insensitive(repo):
    repo.create_shortcut("Setup", "planner")
    assert repo.match_shortcut("SETUP") is not None


def test_no_match(repo):
    repo.create_shortcut("setup", "planner")
    assert repo.match_shortcut("something else") is None


def test_match_increments_usage(repo):
    repo.create_shortcut("test", "planner")
    repo.match_shortcut("test")
    result = repo.get_shortcut("test")
    assert result["times_used"] == 1


def test_reject_reserved(repo):
    assert repo.create_shortcut("tenka", "planner") is False
    assert repo.create_shortcut("help", "planner") is False


def test_reject_too_short(repo):
    assert repo.create_shortcut("a", "planner") is False


def test_reject_unknown_intent(repo):
    assert repo.create_shortcut("test", "nonexistent_intent") is False


def test_delete(repo):
    repo.create_shortcut("test", "planner")
    assert repo.delete_shortcut("test") is True
    assert repo.get_shortcut("test") is None


def test_delete_nonexistent(repo):
    assert repo.delete_shortcut("nope") is False


def test_list_shortcuts(repo):
    repo.create_shortcut("alpha", "planner")
    repo.create_shortcut("beta", "camera_look")
    result = repo.list_shortcuts()
    assert len(result) == 2


def test_upsert(repo):
    repo.create_shortcut("test", "planner")
    repo.create_shortcut("test", "camera_look")
    result = repo.get_shortcut("test")
    assert result["intent"] == "camera_look"


def test_reset(repo):
    repo.create_shortcut("a", "planner")
    repo.create_shortcut("b", "planner")
    repo.reset_shortcuts()
    assert repo.list_shortcuts() == []


def test_match_empty_returns_none(repo):
    assert repo.match_shortcut("") is None
    assert repo.match_shortcut(" ") is None
    assert repo.match_shortcut(None) is None
