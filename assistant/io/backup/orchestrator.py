"""
io/backup/orchestrator.py — Archive, encrypt, upload, and restore.

run_backup() builds an archive, encrypts it with the in-memory unlocked
key, uploads it via the configured provider, and applies a fixed-count
retention policy. run_restore() downloads, decrypts, and extracts a
backup into SANDBOX_DIR. Background scheduling lands in Task 8.
"""
import logging
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("backup.orchestrator")

_EXCLUDED_TOP_LEVEL_DIRS = {"browser-cache"}

_DB_SIDECAR_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal")

_RETENTION_COUNT = 3

# In-memory only — the recovery phrase (and the key derived from it) is
# NEVER written to disk. This means scheduled backups only run in a
# process that has had the phrase entered at least once since it
# started; that's the direct, intended consequence of "never persisted".
_unlocked_key: bytes | None = None


def _build_archive(dest_path: Path) -> None:
    """Tar up everything under SANDBOX_DIR a restore needs.

    The SQLite DB is snapshotted via Database.backup_to() into a temp
    file first, rather than tarring the live .db file directly — WAL
    mode means the .db file alone can be mid-write. browser-cache/ is
    skipped entirely: it's Playwright's bundled Chromium profile,
    purely regenerable, zero backup value.
    """
    from ... import config
    from ...storage.db import get_db

    db = get_db()
    if db is None:
        raise RuntimeError("DB not initialized — cannot build backup archive.")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db_snapshot = Path(tmp) / "tenka.db"
        db.backup_to(db_snapshot)

        with tarfile.open(dest_path, "w") as tar:
            tar.add(db_snapshot, arcname="memory/tenka.db")

            for top_level in ("memory", "manifests", "Notes", "Sessions"):
                src_dir = config.SANDBOX_DIR / top_level
                if not src_dir.exists():
                    continue
                for item in src_dir.rglob("*"):
                    if item.is_dir():
                        continue
                    if item.name.endswith(_DB_SIDECAR_SUFFIXES):
                        continue  # already snapshotted above (or a live WAL-mode sidecar)
                    rel = item.relative_to(config.SANDBOX_DIR)
                    if any(part in _EXCLUDED_TOP_LEVEL_DIRS for part in rel.parts):
                        continue
                    tar.add(item, arcname=str(rel))

    logger.info(f"[BACKUP] Archive built at {dest_path} ({dest_path.stat().st_size} bytes)")


def set_unlocked_key(key: bytes | None) -> None:
    """Cache the derived AES key in process memory for this session."""
    global _unlocked_key
    _unlocked_key = key


def get_unlocked_key() -> bytes | None:
    return _unlocked_key


def is_unlocked() -> bool:
    return _unlocked_key is not None


def _apply_retention(provider, keep: int = _RETENTION_COUNT) -> None:
    versions = provider.list_versions()  # newest first
    for stale_label in versions[keep:]:
        provider.delete(stale_label)


def run_backup(provider_name: str = "google_drive") -> None:
    """Build, encrypt, and upload one backup snapshot.

    Raises RuntimeError if no key is unlocked, or BackupProviderError /
    whatever the provider raises on upload failure. Callers (the
    scheduler loop, the manage_backup handler) decide how to report it.
    """
    from . import crypto, backup_provider_registry
    from ...storage.db import get_db
    from ...storage.repos.settings import SettingsRepo

    if not is_unlocked():
        raise RuntimeError("Backup key is not unlocked — provide the recovery phrase first.")

    provider = backup_provider_registry.require(provider_name)

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "tenka_backup.tar"
        _build_archive(archive_path)
        encrypted = crypto.encrypt(archive_path.read_bytes(), _unlocked_key)

    # Microsecond resolution: two backups in the same wall-clock second
    # (e.g. rapid manual retries, or a tight retention test loop) would
    # otherwise collide onto one label and silently overwrite each other.
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    db = get_db()
    settings = SettingsRepo(db) if db is not None else None

    try:
        provider.upload(encrypted, label)
        _apply_retention(provider)
    except Exception:
        if settings is not None:
            settings.set("backup_last_backup_status", "failed", source="backup_run")
        raise

    if settings is not None:
        settings.set("backup_last_backup_at", datetime.now(timezone.utc).isoformat(), source="backup_run")
        settings.set("backup_last_backup_status", "success", source="backup_run")

    logger.info(f"[BACKUP] Uploaded version '{label}' to {provider_name}")


def run_restore(
    recovery_phrase: str,
    provider_name: str = "google_drive",
    label: str | None = None,
) -> None:
    """Download, decrypt, and extract a backup into SANDBOX_DIR.

    Raises RuntimeError with a user-facing message on any failure —
    wrong phrase, corrupted blob, no backups found, or a structurally
    bad archive. Never applies a partial or corrupt extraction: the
    archive is extracted into a scratch staging directory first (with
    tarfile's 'data' filter, guarding against path traversal via
    malicious member paths/symlinks — PEP 706), and only copied into
    the live SANDBOX_DIR after extraction fully succeeds.
    """
    from cryptography.exceptions import InvalidTag

    from . import crypto, backup_provider_registry
    from ... import config

    provider = backup_provider_registry.require(provider_name)

    versions = provider.list_versions()
    if not versions:
        raise RuntimeError("No backups found for this provider.")
    target_label = label or versions[0]

    encrypted = provider.download(target_label)
    key = crypto.derive_key(recovery_phrase)

    try:
        archive_bytes = crypto.decrypt(encrypted, key)
    except InvalidTag:
        raise RuntimeError("Recovery phrase is incorrect, or the backup is corrupted.")

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "restore.tar"
        archive_path.write_bytes(archive_bytes)

        staging_dir = Path(tmp) / "extracted"
        staging_dir.mkdir()
        try:
            with tarfile.open(archive_path, "r") as tar:
                tar.extractall(staging_dir, filter="data")
        except (tarfile.TarError, OSError) as exc:
            raise RuntimeError(
                "Backup archive is corrupted and could not be extracted."
            ) from exc

        # Only touch the live sandbox once the archive is proven fully
        # extractable — nothing above this line has written to it.
        config.SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging_dir, config.SANDBOX_DIR, dirs_exist_ok=True)

    logger.info(f"[BACKUP] Restored version '{target_label}' from {provider_name}")
