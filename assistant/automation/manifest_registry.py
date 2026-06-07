"""manifest_registry.py — registry typed registry over the manifest store.

Exposes phrase lookup and active-app matching. Underlying state lives
in ManifestStore (in-memory cache) + AppManifestIndexRepo (SQLite).
"""
from __future__ import annotations

import re
from typing import Any

from .manifest_schema import AppManifest
from .manifest_store import ManifestStore
from ..storage.repos.app_manifest_index import AppManifestIndexRepo


class ManifestRegistry:
    def __init__(
        self, *, store: ManifestStore, index_repo: AppManifestIndexRepo,
    ) -> None:
        self._store = store
        self._repo = index_repo

    # ─── Public accessors ─────────────────────────────────────────────────

    @property
    def store(self) -> ManifestStore:
        """Underlying manifest store (in-memory cache + YAML I/O)."""
        return self._store

    @property
    def index_repo(self) -> AppManifestIndexRepo:
        """Underlying SQLite index repo for app_manifest_index."""
        return self._repo

    def lookup_phrase(self, phrase: str) -> list[tuple[str, str]]:
        """Returns list of (app_id, intent_id) for an exact phrase match."""
        rows = self._repo.find_phrase(phrase.strip().lower())
        return [(r["app_id"], r["intent_id"]) for r in rows]

    def get_all_for_active_app(self, active: dict[str, Any]) -> list[AppManifest]:
        """Returns ALL manifests whose match clause matches the active window.

        active is the dict produced by automation/router.py active-app detection,
        with keys 'process_names' (list[str]) and 'window_title' (str).

        Why list and not Optional[AppManifest]: a single process can have
        multiple manifests (e.g., a user with separate 'edit code' and
        'lock journal' manifests for notepad.exe). Returning only the
        first match silently hides the others from the regex_router phrase
        lookup — see F12 in the Session-5 post-mortem. Callers walk the
        returned list and intersect with phrase candidates to find the
        manifest that BOTH matches the active app AND owns the phrase.
        """
        proc_set = {p.lower() for p in active.get("process_names", [])}
        title = active.get("window_title", "") or ""
        out: list[AppManifest] = []
        for am in self._store.all():
            am_procs = {p.lower() for p in am.match.process_names}
            if not (am_procs & proc_set):
                continue
            if not am.match.window_title_patterns:
                out.append(am)
                continue
            for pat in am.match.window_title_patterns:
                if re.search(pat, title, re.IGNORECASE):
                    out.append(am)
                    break
        return out

    def get_for_active_app(self, active: dict[str, Any]) -> AppManifest | None:
        """Back-compat first-match shim. Prefer get_all_for_active_app for
        anything that intersects active-app match with phrase candidates —
        F12 showed first-match silently hides collisions.
        """
        matches = self.get_all_for_active_app(active)
        return matches[0] if matches else None

    def all_manifests(self) -> list[AppManifest]:
        return self._store.all()


# ─── Module-level singleton (lazy init from main.py) ──────────────────────

_singleton: ManifestRegistry | None = None


def init_singleton(
    *, store: ManifestStore, index_repo: AppManifestIndexRepo,
) -> ManifestRegistry:
    global _singleton
    _singleton = ManifestRegistry(store=store, index_repo=index_repo)
    return _singleton


def get_singleton() -> ManifestRegistry | None:
    return _singleton
