"""
io/backup/orchestrator.py — Archive, encrypt, upload, and restore.

run_backup() builds an archive, encrypts it with the in-memory unlocked
key, uploads it via the configured provider, and applies a fixed-count
retention policy. run_restore() downloads, decrypts, and extracts a
backup into SANDBOX_DIR. start()/stop() run a background scheduler
thread that checks periodically and calls run_backup() when a backup
is due (see _maybe_run_scheduled_backup()).
"""
import logging
import shutil
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("backup.orchestrator")

# Directories under SANDBOX_DIR that are never archived, at any depth.
# Checked against every path part of every candidate file, so a nested
# copy (Sessions/captures/, memory/debug/) is skipped too.
_EXCLUDED_DIRS = {
    "credentials",     # OAuth tokens + API keys: machine-scoped secrets.
                       # Reconnecting a provider is cheap; putting live
                       # credentials inside a cloud-hosted archive is not.
    "captures",        # vision-loop screenshots — regenerable, large
    "debug",           # code_executor code dumps — dev artifacts only
    "debug_captures",  # vision debug frames — dev artifacts only
    "browser-cache",   # Playwright's bundled Chromium profile — pure cache
}

_DB_SIDECAR_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal")

_RETENTION_COUNT = 3

_BACKUP_CHECK_SECONDS = 6 * 60 * 60   # check every 6 hours
_BACKUP_INTERVAL_HOURS = 24           # run at most once per 24h

_backup_thread: Optional[threading.Thread] = None
_backup_stop_event = threading.Event()

# In-memory only — the recovery phrase (and the key derived from it) is
# NEVER written to disk. This means scheduled backups only run in a
# process that has had the phrase entered at least once since it
# started; that's the direct, intended consequence of "never persisted".
_unlocked_key: bytes | None = None


def _backed_up_dirs() -> tuple[str, ...]:
    """Top-level SANDBOX_DIR directories a restore needs, by name.

    config.py owns constants for only three of these; the rest are
    created by their owning module, cited below. Anything not listed
    here is not backed up — _EXCLUDED_DIRS names the ones that is
    deliberate for. Add a line here when a new durable data directory
    appears; the alternative is silently losing it.
    """
    from ... import config
    return (
        "memory",                  # SQLite DB + FAISS index/ID-map (storage/db.py)
        config.MANIFESTS_DIR.name,
        config.NOTES_DIR.name,
        config.SESSIONS_DIR.name,
        "faces",                   # face embeddings (faces.py:19)
        "scripts",                 # saved generated scripts (code_executor/templates.py:17)
        "knowledge",               # per-service knowledge JSON (knowledge.py:69)
        "service_data",            # messaging session data (io/messaging_bridge.py:78)
    )


def _build_archive(dest_path: Path) -> None:
    """Tar up everything under SANDBOX_DIR a restore needs.

    The SQLite DB is snapshotted via Database.backup_to() into a temp
    file first, rather than tarring the live .db file directly — WAL
    mode means the .db file alone can be mid-write. The directory list
    comes from _backed_up_dirs(); _EXCLUDED_DIRS prunes regenerable
    caches and machine-scoped secrets wherever they appear inside it.
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

            for top_level in _backed_up_dirs():
                src_dir = config.SANDBOX_DIR / top_level
                if not src_dir.exists():
                    continue
                for item in src_dir.rglob("*"):
                    if item.is_dir():
                        continue
                    if item.name.endswith(_DB_SIDECAR_SUFFIXES):
                        continue  # already snapshotted above (or a live WAL-mode sidecar)
                    rel = item.relative_to(config.SANDBOX_DIR)
                    if any(part in _EXCLUDED_DIRS for part in rel.parts):
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
        settings.set("backup_last_backup_size_bytes", len(encrypted), source="backup_run")

    logger.info(f"[BACKUP] Uploaded version '{label}' to {provider_name}")


def get_state() -> dict:
    """Read the current backup configuration and most recent run outcome.

    Settings, not provider.list_versions(), because run_backup() and
    _maybe_run_scheduled_backup() already write here on every run --
    scheduled or manual, from this process or a prior one -- and
    BackupProvider has no size-reporting method to fall back on anyway.
    A missing DB (not yet initialized) degrades to all-defaults rather
    than raising, since this is a read a status panel polls freely.
    """
    from ...storage.db import get_db
    from ...storage.repos.settings import SettingsRepo

    db = get_db()
    if db is None:
        return {
            "enabled": False, "provider": "google_drive",
            "last_backup_at": "", "last_result": "", "size_bytes": 0,
        }
    settings = SettingsRepo(db)
    return {
        "enabled": bool(settings.get("backup_enabled", False)),
        "provider": str(settings.get("backup_provider", "google_drive")),
        "last_backup_at": str(settings.get("backup_last_backup_at") or ""),
        "last_result": str(settings.get("backup_last_backup_status") or ""),
        "size_bytes": int(settings.get("backup_last_backup_size_bytes", 0) or 0),
    }


def run_restore(
    recovery_phrase: str,
    provider_name: str = "google_drive",
    label: str | None = None,
) -> None:
    """Download, decrypt, and extract a backup into SANDBOX_DIR.

    Raises RuntimeError with a user-facing message on any failure —
    wrong phrase, corrupted blob, no backups found, or a structurally
    bad archive. Never applies a partial or corrupt extraction: each
    candidate archive is extracted into a scratch staging directory
    first (with tarfile's 'data' filter, guarding against path
    traversal via malicious member paths/symlinks — PEP 706), and only
    copied into the live SANDBOX_DIR after extraction fully succeeds.

    When label is None, every stored version is a fallback candidate,
    newest first: a decrypt (InvalidTag) or extract failure on one
    version moves on to the next rather than failing the whole restore.
    This is safe even though InvalidTag can't distinguish "wrong
    phrase" from "corrupted blob" — a wrong phrase fails identically
    against every version, so falling through the whole list still
    ends in the same correct error; a corrupted latest version with a
    correct phrase now recovers via the next-older one instead of
    failing outright. An explicit label is a specific request, so it
    is tried alone with no fallback.
    """
    from cryptography.exceptions import InvalidTag

    from . import crypto, backup_provider_registry
    from ... import config

    provider = backup_provider_registry.require(provider_name)

    versions = provider.list_versions()
    if not versions:
        raise RuntimeError("No backups found for this provider.")
    candidates = [label] if label else versions

    key = crypto.derive_key(recovery_phrase)

    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = None
        target_label = None
        last_error: RuntimeError = RuntimeError("No backups found for this provider.")

        for candidate in candidates:
            encrypted = provider.download(candidate)
            try:
                archive_bytes = crypto.decrypt(encrypted, key)
            except InvalidTag:
                last_error = RuntimeError("Recovery phrase is incorrect, or the backup is corrupted.")
                continue

            archive_path = Path(tmp) / f"restore_{candidate}.tar"
            archive_path.write_bytes(archive_bytes)

            candidate_staging = Path(tmp) / f"extracted_{candidate}"
            candidate_staging.mkdir()
            try:
                with tarfile.open(archive_path, "r") as tar:
                    tar.extractall(candidate_staging, filter="data")
            except (tarfile.TarError, OSError) as exc:
                last_error = RuntimeError("Backup archive is corrupted and could not be extracted.")
                logger.warning(f"[BACKUP] Version '{candidate}' is unusable, trying older version: {exc}")
                continue

            staging_dir = candidate_staging
            target_label = candidate
            break

        if staging_dir is None:
            raise last_error

        # Only touch the live sandbox once the archive is proven fully
        # extractable — nothing above this line has written to it.
        config.SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

        # The live process's own tenka.db connection must not survive
        # having its file replaced out from under it: if anything writes
        # through the stale connection afterward, SQLite writes new WAL
        # frames against a file whose physical structure no longer matches
        # what the connection last read — real corruption, not just a
        # missed restore. Closing it first (close_for_restore()) prevents
        # that.
        #
        # Deliberately NOT reopened afterward: storage/db.py's own
        # singleton isn't the only reference in play — memory.py (and the
        # rest of the 13-repo convention) each cache their own repo object
        # at startup instead of calling get_db() fresh every time, so a
        # new storage/db.py singleton wouldn't reach them anyway. A real
        # process restart is the only thing that rebuilds every one of
        # those caches correctly — this is why run_restore()'s caller
        # (actions/backup_pending.py) requests a full shutdown rather than
        # trying to keep the current process alive.
        from ...storage.db import close_for_restore
        close_for_restore()

        # Backups never contain -wal/-shm/-journal sidecars (excluded by
        # _build_archive — the DB is checkpointed into a clean single file
        # via Database.backup_to() before archiving). copytree only
        # overwrites files the archive actually has, so a stale sidecar
        # left over from the live/pre-restore database would otherwise sit
        # untouched next to the freshly restored .db file — and in WAL
        # mode, SQLite replays an existing -wal file over the main file's
        # content on next open, silently discarding the restore. Now that
        # the connection above is closed, SQLite has released its lock on
        # these, so the unlink can't hit Windows' WinError 32 the way it
        # would against a still-open handle — but stay defensive anyway.
        for db_file in staging_dir.rglob("*.db"):
            rel = db_file.relative_to(staging_dir)
            live_db = config.SANDBOX_DIR / rel
            for suffix in ("-wal", "-shm", "-journal"):
                stale = live_db.with_name(live_db.name + suffix)
                try:
                    stale.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(f"[BACKUP] Could not remove {stale.name}: {exc}")

        shutil.copytree(staging_dir, config.SANDBOX_DIR, dirs_exist_ok=True)

    logger.info(f"[BACKUP] Restored version '{target_label}' from {provider_name}")


def start() -> None:
    """Start the background backup scheduler thread."""
    global _backup_thread

    if _backup_thread and _backup_thread.is_alive():
        logger.debug("[BACKUP] Scheduler already running")
        return

    _backup_stop_event.clear()
    _backup_thread = threading.Thread(
        target=_backup_loop,
        name="cloud-backup-scheduler",
        daemon=True,
    )
    _backup_thread.start()
    logger.info("[BACKUP] Scheduler started")


def stop() -> None:
    """Stop the background backup scheduler thread."""
    _backup_stop_event.set()
    if _backup_thread:
        _backup_thread.join(timeout=5)
    logger.info("[BACKUP] Scheduler stopped")


def _backup_loop() -> None:
    # Let everything else finish initializing first. Waited on the stop
    # event (not a plain time.sleep) so stop() can interrupt this delay
    # immediately instead of blocking for up to 30s.
    if _backup_stop_event.wait(timeout=30.0):
        return

    while not _backup_stop_event.is_set():
        try:
            _maybe_run_scheduled_backup()
        except Exception as e:
            logger.warning(f"[BACKUP] Error in scheduled backup check: {e}")

        for _ in range(_BACKUP_CHECK_SECONDS):
            if _backup_stop_event.is_set():
                return
            time.sleep(1)


def _maybe_run_scheduled_backup() -> None:
    from ...storage.db import get_db
    from ...storage.repos.settings import SettingsRepo

    db = get_db()
    if db is None:
        return
    settings = SettingsRepo(db)

    if not settings.get("backup_enabled", False):
        return
    if not is_unlocked():
        logger.debug("[BACKUP] Skipping scheduled backup — key not unlocked this session")
        return

    last_at = settings.get("backup_last_backup_at")
    if last_at:
        try:
            last_dt = datetime.fromisoformat(last_at)
            hours_elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            if hours_elapsed < _BACKUP_INTERVAL_HOURS:
                return
        except (ValueError, TypeError):
            pass  # corrupted timestamp -> run anyway

    provider_name = settings.get("backup_provider", "google_drive")
    logger.info("[BACKUP] Running scheduled backup")
    run_backup(provider_name)
