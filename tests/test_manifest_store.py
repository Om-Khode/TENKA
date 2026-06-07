"""Tests for manifest_store — scan/cache/invalidate per-app YAML."""

import pytest

from assistant.automation.manifest_schema import (
    AppManifest, Intent, Selector, Match, Captured, dump_manifest_to_yaml,
)
from assistant.automation.manifest_store import ManifestStore
from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.app_manifest_index import AppManifestIndexRepo


@pytest.fixture
def store(tmp_path):
    _reset_for_testing()
    db_dir = tmp_path / "_db"
    db_dir.mkdir()
    db = Database(db_dir / "test.db")
    repo = AppManifestIndexRepo(db._conn)
    s = ManifestStore(manifests_dir=tmp_path, index_repo=repo)
    yield s
    db.close()
    _reset_for_testing()


def _write_minimal(tmp_path, app_id):
    am = AppManifest(
        schema_version=1, app_id=app_id, display_name=app_id.title(),
        match=Match(process_names=[f"{app_id}.exe"]),
        intents=[Intent(
            id="play", display_name="Play", phrases=["play"],
            handler_selectors=[Selector(kind="hotkey", keys="Space")],
            captured=Captured(timestamp="2026-05-30T10:00:00Z"),
        )],
    )
    dump_manifest_to_yaml(am, tmp_path / f"{app_id}.yaml")
    return am


def test_scan_loads_existing_yaml(store, tmp_path):
    _write_minimal(tmp_path, "test_app.desktop")
    store.scan_and_index()
    assert store.get("test_app.desktop").app_id == "test_app.desktop"


def test_cache_hit_after_scan(store, tmp_path):
    _write_minimal(tmp_path, "x")
    store.scan_and_index()
    am1 = store.get("x")
    am2 = store.get("x")
    assert am1 is am2  # same object — cache hit


def test_mtime_invalidates(store, tmp_path):
    _write_minimal(tmp_path, "x")
    store.scan_and_index()
    am1 = store.get("x")
    # Mutate file
    import time; time.sleep(0.01)
    _write_minimal(tmp_path, "x")  # rewrites, new mtime
    store.refresh_if_stale("x")
    am2 = store.get("x")
    assert am1 is not am2


def test_corrupt_yaml_quarantines(store, tmp_path):
    (tmp_path / "broken.yaml").write_text("not: valid: yaml: [", encoding="utf-8")
    store.scan_and_index()  # must not raise
    quarantined = list(tmp_path.glob("broken.yaml.corrupt.*"))
    assert len(quarantined) == 1
    assert (tmp_path / "broken.yaml").exists() is False
