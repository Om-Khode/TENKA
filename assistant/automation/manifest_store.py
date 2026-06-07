"""manifest_store.py — scan, cache, and invalidate per-app manifest YAMLs.

Source of truth: <sandbox>/manifests/*.yaml. This module scans on startup,
caches loaded AppManifest objects in memory, and re-reads on mtime drift.
Corrupt YAML is quarantined (renamed .corrupt.<ts>), never deleted.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from threading import RLock

from .manifest_schema import AppManifest, load_manifest_from_yaml
from ..storage.repos.app_manifest_index import AppManifestIndexRepo

logger = logging.getLogger("manifest")


class ManifestStore:
    def __init__(
        self, manifests_dir: Path, index_repo: AppManifestIndexRepo,
    ) -> None:
        self._dir = manifests_dir
        self._repo = index_repo
        self._cache: dict[str, AppManifest] = {}
        self._mtimes: dict[str, float] = {}
        self._lock = RLock()
        self._invalidation_listeners: list[Callable[[str], None]] = []

    # ─── Public API ────────────────────────────────────────────────────────

    def scan_and_index(self) -> None:
        """Walk manifests_dir, load every *.yaml, populate cache + SQLite index.

        Quarantines corrupt YAMLs as .corrupt.<ts>; logs without raising.
        """
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(self._dir.glob("*.yaml")):
                self._load_one(path)

    def get(self, app_id: str) -> AppManifest | None:
        with self._lock:
            return self._cache.get(app_id)

    def all(self) -> list[AppManifest]:
        with self._lock:
            return list(self._cache.values())

    def refresh_if_stale(self, app_id: str) -> None:
        with self._lock:
            path = self._dir / f"{app_id}.yaml"
            if not path.exists():
                self._cache.pop(app_id, None)
                self._mtimes.pop(app_id, None)
                self._repo.delete(app_id)
                self._fire_invalidate(app_id)
                return
            mt = path.stat().st_mtime
            if mt != self._mtimes.get(app_id):
                self._load_one(path)
                self._fire_invalidate(app_id)

    def write_and_invalidate(self, manifest: AppManifest) -> None:
        """Used by the promoter. Writes YAML, refreshes cache + index, fires listeners."""
        from .manifest_schema import dump_manifest_to_yaml
        path = self._dir / f"{manifest.app_id}.yaml"
        dump_manifest_to_yaml(manifest, path)
        with self._lock:
            self._load_one(path)
        self._fire_invalidate(manifest.app_id)

    def register_invalidation_listener(self, fn: Callable[[str], None]) -> None:
        self._invalidation_listeners.append(fn)

    # ─── Internals ─────────────────────────────────────────────────────────

    def _load_one(self, path: Path) -> None:
        try:
            am = load_manifest_from_yaml(path)
        except Exception as e:
            self._quarantine(path, e)
            return
        self._cache[am.app_id] = am
        self._mtimes[am.app_id] = path.stat().st_mtime
        # Denormalize for the index repo.
        self._repo.upsert_manifest(
            app_id=am.app_id,
            file_path=str(path),
            file_mtime=self._mtimes[am.app_id],
            process_names=am.match.process_names,
            window_patterns=am.match.window_title_patterns,
            intent_count=len(am.intents),
        )
        phrases = [
            (p, intent.id, False) for intent in am.intents for p in intent.phrases
        ]
        self._repo.replace_phrases(am.app_id, phrases)

    def _quarantine(self, path: Path, error: Exception) -> None:
        ts = int(time.time())
        quarantine_path = path.with_suffix(path.suffix + f".corrupt.{ts}")
        logger.warning(
            f"[manifest] Corrupt manifest {path.name}: {error}; "
            f"moved to {quarantine_path.name}"
        )
        path.rename(quarantine_path)

    def _fire_invalidate(self, app_id: str) -> None:
        for fn in list(self._invalidation_listeners):
            try:
                fn(app_id)
            except Exception as e:
                logger.warning(f"[manifest] Invalidation listener raised: {e}")
