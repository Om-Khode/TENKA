"""Tests for io/backup/orchestrator.py."""
import tarfile

import pytest

from assistant.storage.db import init_db, _reset_for_testing


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point config.SANDBOX_DIR at a tmp dir with a real DB + fake extras."""
    _reset_for_testing()
    sandbox_dir = tmp_path / "TENKA"
    memory_dir = sandbox_dir / "memory"
    memory_dir.mkdir(parents=True)
    db = init_db(memory_dir / "tenka.db")

    (memory_dir / "voiceprint.npz").write_bytes(b"fake-voiceprint")
    (sandbox_dir / "manifests").mkdir()
    (sandbox_dir / "manifests" / "notepad.yaml").write_text("app: notepad")
    (sandbox_dir / "Notes").mkdir()
    (sandbox_dir / "Notes" / "todo.md").write_text("- buy milk")
    (sandbox_dir / "browser-cache").mkdir()
    (sandbox_dir / "browser-cache" / "chromium.bin").write_bytes(b"should-not-be-backed-up")

    from assistant import config
    monkeypatch.setattr(config, "SANDBOX_DIR", sandbox_dir)

    yield sandbox_dir
    db.close()
    _reset_for_testing()


def test_build_archive_includes_expected_paths_excludes_browser_cache(sandbox, tmp_path):
    from assistant.io.backup.orchestrator import _build_archive

    archive_path = tmp_path / "backup.tar"
    _build_archive(archive_path)

    with tarfile.open(archive_path, "r") as tar:
        names = tar.getnames()

    assert "memory/tenka.db" in names
    assert "memory/voiceprint.npz" in names
    assert "manifests/notepad.yaml" in names
    assert "Notes/todo.md" in names
    assert not any("browser-cache" in n for n in names)


def test_build_archive_excludes_live_wal_sidecar_files(sandbox, tmp_path):
    """WAL-mode sidecars (tenka.db-wal, tenka.db-shm) sit next to the live
    tenka.db while the DB connection is open — Database.backup_to() already
    produces a clean snapshot, so the raw sidecars must never be tarred in
    raw; that would reintroduce the exact mid-write-state risk backup_to()
    exists to avoid.

    The sandbox fixture's DB connection is left open (matching how a real
    backup runs) and was opened under PRAGMA journal_mode=WAL, so SQLite
    itself has already created real tenka.db-wal / tenka.db-shm sidecars
    on disk by the time this test runs — no need to fabricate them (and on
    Windows, tenka.db-shm is memory-mapped by the live connection, so a
    direct overwrite from the test would hit a file-locking error anyway).
    """
    from assistant.io.backup.orchestrator import _build_archive

    memory_dir = sandbox / "memory"
    # Precondition: the live connection really did leave sidecars behind.
    assert (memory_dir / "tenka.db-wal").exists()
    assert (memory_dir / "tenka.db-shm").exists()

    archive_path = tmp_path / "backup.tar"
    _build_archive(archive_path)

    with tarfile.open(archive_path, "r") as tar:
        names = tar.getnames()

    assert "memory/tenka.db" in names
    assert "memory/tenka.db-wal" not in names
    assert "memory/tenka.db-shm" not in names
    assert not any(n.endswith((".db-wal", ".db-shm")) for n in names)
