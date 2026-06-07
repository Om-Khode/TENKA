"""Tests for AppManifestIndexRepo — SQLite index over per-app manifest YAMLs."""

import pytest

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo


@pytest.fixture
def repo(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield AppManifestIndexRepo(db._conn)
    db.close()
    _reset_for_testing()


def test_upsert_then_get(repo):
    repo.upsert_manifest(
        app_id="test_app.desktop",
        file_path="/x/test_app.desktop.yaml",
        file_mtime=12345.6,
        process_names=["TestApp.exe"],
        window_patterns=["^TestApp"],
        intent_count=2,
    )
    row = repo.get("test_app.desktop")
    assert row["process_names"] == ["TestApp.exe"]
    assert row["intent_count"] == 2


def test_upsert_phrases(repo):
    repo.upsert_manifest(
        app_id="x", file_path="/x.yaml", file_mtime=1.0,
        process_names=["x"], window_patterns=[], intent_count=1,
    )
    repo.replace_phrases("x", [
        ("play music", "play", False),
        ("resume", "play", True),
    ])
    rows = repo.find_phrase("play music")
    assert len(rows) == 1
    assert rows[0]["app_id"] == "x"
    assert rows[0]["intent_id"] == "play"


def test_replace_phrases_idempotent(repo):
    repo.upsert_manifest(
        app_id="x", file_path="/x.yaml", file_mtime=1.0,
        process_names=["x"], window_patterns=[], intent_count=1,
    )
    repo.replace_phrases("x", [("play", "play", False)])
    repo.replace_phrases("x", [("pause", "pause", False)])  # replaces, not appends
    assert repo.find_phrase("play") == []
    assert len(repo.find_phrase("pause")) == 1


def test_all_apps_and_delete(repo):
    repo.upsert_manifest(
        app_id="a", file_path="/a.yaml", file_mtime=1.0,
        process_names=["a"], window_patterns=[], intent_count=1,
    )
    repo.upsert_manifest(
        app_id="b", file_path="/b.yaml", file_mtime=1.0,
        process_names=["b"], window_patterns=[], intent_count=1,
    )
    repo.replace_phrases("a", [("foo", "foo", False)])
    assert len(repo.all_apps()) == 2
    repo.delete("a")
    assert len(repo.all_apps()) == 1
    assert repo.find_phrase("foo") == []  # cascade-cleaned phrases
