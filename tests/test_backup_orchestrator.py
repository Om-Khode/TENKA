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

    # Data directories that used to be silently omitted from every backup.
    (sandbox_dir / "faces").mkdir()
    (sandbox_dir / "faces" / "omkho.npy").write_bytes(b"fake-embedding")
    (sandbox_dir / "scripts").mkdir()
    (sandbox_dir / "scripts" / "open_thing.py").write_text("print('hi')")
    (sandbox_dir / "knowledge").mkdir()
    (sandbox_dir / "knowledge" / "some_service.json").write_text("{}")
    (sandbox_dir / "service_data").mkdir(parents=True)
    (sandbox_dir / "service_data" / "some_service").mkdir()
    (sandbox_dir / "service_data" / "some_service" / "session.json").write_text("{}")

    # Excluded, and deliberately nested inside a directory that IS backed up —
    # the top-level allowlist alone would not skip these.
    (sandbox_dir / "Sessions" / "captures").mkdir(parents=True)
    (sandbox_dir / "Sessions" / "captures" / "frame.png").write_bytes(b"regenerable")
    (sandbox_dir / "Notes" / "browser-cache").mkdir()
    (sandbox_dir / "Notes" / "browser-cache" / "blob.bin").write_bytes(b"regenerable")

    # Machine-scoped secrets — never archived.
    (sandbox_dir / "credentials").mkdir()
    (sandbox_dir / "credentials" / "google_drive.json").write_text('{"token": "secret"}')

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


def test_build_archive_includes_every_durable_data_directory(sandbox, tmp_path):
    """faces/, scripts/, knowledge/ and service_data/ are as unrecoverable as
    Notes/ — a backup that omits them is not a backup."""
    from assistant.io.backup.orchestrator import _build_archive

    archive_path = tmp_path / "backup.tar"
    _build_archive(archive_path)

    with tarfile.open(archive_path, "r") as tar:
        names = tar.getnames()

    assert "faces/omkho.npy" in names
    assert "scripts/open_thing.py" in names
    assert "knowledge/some_service.json" in names
    assert "service_data/some_service/session.json" in names


def test_build_archive_prunes_excluded_dirs_nested_under_included_ones(sandbox, tmp_path):
    """The exclusion set is walk-level, not just a top-level allowlist gap:
    an excluded name anywhere in a backed-up tree is skipped."""
    from assistant.io.backup.orchestrator import _build_archive

    archive_path = tmp_path / "backup.tar"
    _build_archive(archive_path)

    with tarfile.open(archive_path, "r") as tar:
        names = tar.getnames()

    # Both live inside directories that ARE archived.
    assert "Sessions/captures/frame.png" not in names
    assert "Notes/browser-cache/blob.bin" not in names
    assert not any("captures" in n for n in names)


def test_build_archive_never_includes_credentials(sandbox, tmp_path):
    """OAuth tokens are machine-scoped and cheap to re-obtain; shipping them
    inside a cloud-hosted archive is not a trade worth making."""
    from assistant.io.backup.orchestrator import _build_archive

    archive_path = tmp_path / "backup.tar"
    _build_archive(archive_path)

    with tarfile.open(archive_path, "r") as tar:
        names = tar.getnames()

    assert not any("credentials" in n for n in names)


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


def test_is_unlocked_false_by_default():
    from assistant.io.backup import orchestrator
    orchestrator.set_unlocked_key(None)
    assert orchestrator.is_unlocked() is False


def test_set_and_get_unlocked_key():
    from assistant.io.backup import orchestrator
    orchestrator.set_unlocked_key(b"0" * 32)
    assert orchestrator.is_unlocked() is True
    assert orchestrator.get_unlocked_key() == b"0" * 32
    orchestrator.set_unlocked_key(None)


def test_run_backup_raises_when_locked(sandbox):
    from assistant.io.backup import orchestrator
    orchestrator.set_unlocked_key(None)
    with pytest.raises(RuntimeError, match="unlocked"):
        orchestrator.run_backup()


def test_run_backup_uploads_and_applies_retention(sandbox, monkeypatch):
    from assistant.io.backup import orchestrator, crypto, backup_provider_registry
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo

    key = crypto.derive_key(crypto.generate_recovery_phrase())
    orchestrator.set_unlocked_key(key)

    class _FakeProvider:
        name = "google_drive"
        def __init__(self):
            self.uploads: dict[str, bytes] = {}
        def is_connected(self): return True
        def upload(self, blob, label): self.uploads[label] = blob
        def list_versions(self): return sorted(self.uploads.keys(), reverse=True)
        def download(self, label): return self.uploads[label]
        def delete(self, label): del self.uploads[label]

    fake = _FakeProvider()
    monkeypatch.setattr(backup_provider_registry, "_entries", {"google_drive": fake})

    for i in range(4):
        orchestrator.run_backup()

    assert len(fake.uploads) == 3  # retention kept only the last 3

    db = get_db()
    status = SettingsRepo(db).get("backup_last_backup_status")
    assert status == "success"

    orchestrator.set_unlocked_key(None)


def test_run_backup_records_failed_status_and_reraises_on_upload_error(sandbox, monkeypatch):
    from assistant.io.backup import orchestrator, crypto, backup_provider_registry
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo

    key = crypto.derive_key(crypto.generate_recovery_phrase())
    orchestrator.set_unlocked_key(key)

    class _FailingProvider:
        name = "google_drive"
        def is_connected(self): return True
        def upload(self, blob, label): raise RuntimeError("boom")
        def list_versions(self): return []
        def download(self, label): raise AssertionError("not reached")
        def delete(self, label): raise AssertionError("not reached")

    monkeypatch.setattr(backup_provider_registry, "_entries", {"google_drive": _FailingProvider()})

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.run_backup()

    db = get_db()
    status = SettingsRepo(db).get("backup_last_backup_status")
    assert status == "failed"

    orchestrator.set_unlocked_key(None)


def test_run_restore_extracts_archive(sandbox, tmp_path, monkeypatch):
    from assistant.io.backup import orchestrator, crypto, backup_provider_registry
    from assistant import config

    phrase = crypto.generate_recovery_phrase()
    key = crypto.derive_key(phrase)
    orchestrator.set_unlocked_key(key)

    class _FakeProvider:
        name = "google_drive"
        def __init__(self):
            self.uploads: dict[str, bytes] = {}
        def is_connected(self): return True
        def upload(self, blob, label): self.uploads[label] = blob
        def list_versions(self): return sorted(self.uploads.keys(), reverse=True)
        def download(self, label): return self.uploads[label]
        def delete(self, label): del self.uploads[label]

    fake = _FakeProvider()
    monkeypatch.setattr(backup_provider_registry, "_entries", {"google_drive": fake})

    orchestrator.run_backup()

    restore_target = tmp_path / "restored_TENKA"
    monkeypatch.setattr(config, "SANDBOX_DIR", restore_target)

    orchestrator.run_restore(phrase)

    assert (restore_target / "memory" / "tenka.db").exists()
    assert (restore_target / "Notes" / "todo.md").exists()

    orchestrator.set_unlocked_key(None)


def test_run_restore_wrong_phrase_raises(sandbox, tmp_path, monkeypatch):
    from assistant.io.backup import orchestrator, crypto, backup_provider_registry

    phrase = crypto.generate_recovery_phrase()
    orchestrator.set_unlocked_key(crypto.derive_key(phrase))

    class _FakeProvider:
        name = "google_drive"
        def __init__(self):
            self.uploads: dict[str, bytes] = {}
        def is_connected(self): return True
        def upload(self, blob, label): self.uploads[label] = blob
        def list_versions(self): return sorted(self.uploads.keys(), reverse=True)
        def download(self, label): return self.uploads[label]
        def delete(self, label): del self.uploads[label]

    fake = _FakeProvider()
    monkeypatch.setattr(backup_provider_registry, "_entries", {"google_drive": fake})
    orchestrator.run_backup()

    wrong_phrase = crypto.generate_recovery_phrase()
    with pytest.raises(RuntimeError, match="incorrect"):
        orchestrator.run_restore(wrong_phrase)

    orchestrator.set_unlocked_key(None)


def test_run_restore_no_backups_raises(sandbox, monkeypatch):
    from assistant.io.backup import orchestrator, backup_provider_registry

    class _EmptyProvider:
        name = "google_drive"
        def is_connected(self): return True
        def upload(self, blob, label): pass
        def list_versions(self): return []
        def download(self, label): raise KeyError(label)
        def delete(self, label): pass

    monkeypatch.setattr(backup_provider_registry, "_entries", {"google_drive": _EmptyProvider()})

    with pytest.raises(RuntimeError, match="No backups found"):
        orchestrator.run_restore("any twelve word phrase used only to derive a key here now")


def test_run_restore_bad_archive_leaves_sandbox_clean(sandbox, tmp_path, monkeypatch):
    """A blob that decrypts fine (right phrase, valid AES-GCM tag) but isn't
    a structurally valid tar must never leak partial content into the live
    SANDBOX_DIR — the extract-to-staging-then-copy design means the sandbox
    should be untouched entirely, not partially populated."""
    from assistant.io.backup import orchestrator, crypto, backup_provider_registry
    from assistant import config

    phrase = crypto.generate_recovery_phrase()
    key = crypto.derive_key(phrase)
    orchestrator.set_unlocked_key(key)

    # Valid ciphertext (decrypts cleanly), but the plaintext is not a tar
    # archive at all — simulates a corrupted/malformed archive that still
    # passes AES-GCM authentication.
    bad_blob = crypto.encrypt(b"this is not a tar archive", key)

    class _FakeProvider:
        name = "google_drive"
        def __init__(self):
            self.uploads = {"20260101T000000.000000Z": bad_blob}
        def is_connected(self): return True
        def upload(self, blob, label): self.uploads[label] = blob
        def list_versions(self): return sorted(self.uploads.keys(), reverse=True)
        def download(self, label): return self.uploads[label]
        def delete(self, label): del self.uploads[label]

    monkeypatch.setattr(backup_provider_registry, "_entries", {"google_drive": _FakeProvider()})

    restore_target = tmp_path / "restored_TENKA"
    restore_target.mkdir(parents=True)
    (restore_target / "pre_existing.txt").write_text("keep me")
    monkeypatch.setattr(config, "SANDBOX_DIR", restore_target)

    with pytest.raises(RuntimeError, match="corrupted"):
        orchestrator.run_restore(phrase)

    # Pre-existing content untouched, and nothing from the bad archive
    # was ever written into the live sandbox.
    assert (restore_target / "pre_existing.txt").read_text() == "keep me"
    assert list(restore_target.iterdir()) == [restore_target / "pre_existing.txt"]

    orchestrator.set_unlocked_key(None)


def test_start_stop_thread_lifecycle():
    from assistant.io.backup import orchestrator
    import threading

    orchestrator.start()
    assert orchestrator._backup_thread is not None
    assert orchestrator._backup_thread.is_alive()

    orchestrator.stop()
    orchestrator._backup_thread.join(timeout=2)
    assert not orchestrator._backup_thread.is_alive()


def test_maybe_run_scheduled_backup_skips_when_locked(sandbox, monkeypatch):
    """The 'key not unlocked this session' gate is the critical security
    constraint — a freshly-restarted process must never auto-backup."""
    from assistant.io.backup import orchestrator
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo

    orchestrator.set_unlocked_key(None)
    settings = SettingsRepo(get_db())
    settings.set("backup_enabled", True, source="test")

    calls = []
    monkeypatch.setattr(orchestrator, "run_backup", lambda provider_name: calls.append(provider_name))

    orchestrator._maybe_run_scheduled_backup()

    assert calls == []


def test_maybe_run_scheduled_backup_skips_when_disabled(sandbox, monkeypatch):
    from assistant.io.backup import orchestrator, crypto
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo

    orchestrator.set_unlocked_key(crypto.derive_key(crypto.generate_recovery_phrase()))
    SettingsRepo(get_db())  # backup_enabled left unset -> defaults False

    calls = []
    monkeypatch.setattr(orchestrator, "run_backup", lambda provider_name: calls.append(provider_name))

    orchestrator._maybe_run_scheduled_backup()

    assert calls == []
    orchestrator.set_unlocked_key(None)


def test_maybe_run_scheduled_backup_skips_when_interval_not_elapsed(sandbox, monkeypatch):
    from assistant.io.backup import orchestrator, crypto
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo
    from datetime import datetime, timezone

    orchestrator.set_unlocked_key(crypto.derive_key(crypto.generate_recovery_phrase()))
    settings = SettingsRepo(get_db())
    settings.set("backup_enabled", True, source="test")
    settings.set("backup_last_backup_at", datetime.now(timezone.utc).isoformat(), source="test")

    calls = []
    monkeypatch.setattr(orchestrator, "run_backup", lambda provider_name: calls.append(provider_name))

    orchestrator._maybe_run_scheduled_backup()

    assert calls == []
    orchestrator.set_unlocked_key(None)


def test_maybe_run_scheduled_backup_runs_when_all_gates_pass(sandbox, monkeypatch):
    from assistant.io.backup import orchestrator, crypto
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo
    from datetime import datetime, timedelta, timezone

    orchestrator.set_unlocked_key(crypto.derive_key(crypto.generate_recovery_phrase()))
    settings = SettingsRepo(get_db())
    settings.set("backup_enabled", True, source="test")
    stale = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    settings.set("backup_last_backup_at", stale, source="test")
    settings.set("backup_provider", "google_drive", source="test")

    calls = []
    monkeypatch.setattr(orchestrator, "run_backup", lambda provider_name: calls.append(provider_name))

    orchestrator._maybe_run_scheduled_backup()

    assert calls == ["google_drive"]
    orchestrator.set_unlocked_key(None)


def test_maybe_run_scheduled_backup_runs_on_corrupted_timestamp(sandbox, monkeypatch):
    """A corrupted/unparseable backup_last_backup_at must not permanently
    wedge the scheduler into never backing up again — run anyway."""
    from assistant.io.backup import orchestrator, crypto
    from assistant.storage.db import get_db
    from assistant.storage.repos.settings import SettingsRepo

    orchestrator.set_unlocked_key(crypto.derive_key(crypto.generate_recovery_phrase()))
    settings = SettingsRepo(get_db())
    settings.set("backup_enabled", True, source="test")
    settings.set("backup_last_backup_at", "not-a-valid-timestamp", source="test")

    calls = []
    monkeypatch.setattr(orchestrator, "run_backup", lambda provider_name: calls.append(provider_name))

    orchestrator._maybe_run_scheduled_backup()

    assert calls == ["google_drive"]  # default provider, backup_provider unset
    orchestrator.set_unlocked_key(None)
