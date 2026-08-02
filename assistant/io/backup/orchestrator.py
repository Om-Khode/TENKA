"""
io/backup/orchestrator.py — Archive, encrypt, upload, and restore.

Background scheduling and the public run_backup/run_restore entry points
land in Tasks 6-8; this file starts with the archive builder they share.
"""
import logging
import tarfile
from pathlib import Path

logger = logging.getLogger("backup.orchestrator")

_EXCLUDED_TOP_LEVEL_DIRS = {"browser-cache"}

_DB_SIDECAR_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal")


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
