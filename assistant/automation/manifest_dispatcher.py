"""manifest_dispatcher.py — manifest-based hot-path dispatch.

Resolves an (app_id, intent_id) to a selector chain and walks it primary-first,
escalating to the healer (session 4) on miss. Health counters update in place
on the cached AppManifest object; writes back to disk are debounced.

The healer hook is wired in session 4. In session 3 a miss simply falls through
to the next selector; if all selectors miss, the dispatcher returns
escalate_to_dispatch=True and the caller routes through computer_task.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from .manifest_primitives import execute_primitive
from .manifest_registry import ManifestRegistry
from .manifest_schema import AppManifest, Intent
from .manifest_store import ManifestStore

logger = logging.getLogger("manifest")

_HEALTH_FLUSH_INTERVAL_S = 30.0  # Debounce writes per spec §3.1


# TODO(manifest-based post-v1): split escalate_to_dispatch into routing_error vs operational_failure.
@dataclass(frozen=True)
class DispatchResult:
    ok: bool
    selector_used_index: int | None = None
    escalate_to_dispatch: bool = False
    error: str | None = None


class ManifestDispatcher:
    def __init__(
        self,
        *,
        registry: ManifestRegistry,
        store: ManifestStore,
        terminator_provider: Callable[[], Any],
        healer: Any = None,  # filled in session 4
    ) -> None:
        # Reserved — healer (Session 4) uses this for re-lookup after healing.
        self._reg = registry
        self._store = store
        self._term_provider = terminator_provider
        self._healer = healer
        self._last_dispatch: tuple[str, str] | None = None
        self._dirty_apps: set[str] = set()
        # Starts at now: first 30s of operation is debounced. flush_now() forces.
        self._last_flush = time.time()
        # RLock: _on_selector_* hold the lock then call _maybe_flush which re-acquires.
        self._lock = RLock()

    def dispatch(
        self,
        *,
        app_id: str,
        intent_id: str,
        slots: dict[str, str],
        active_window: str,
    ) -> DispatchResult:
        with self._lock:
            self._last_dispatch = (app_id, intent_id)
        # slots: reserved — Task 15/16 will thread slot bindings into selector args
        am = self._store.get(app_id)
        if am is None:
            # Routing error — caller addressed wrong dispatcher.
            return DispatchResult(
                ok=False, escalate_to_dispatch=True,
                error=f"manifest not found: {app_id}",
            )
        intent = next((i for i in am.intents if i.id == intent_id), None)
        if intent is None:
            # Routing error — manifest exists but intent missing.
            return DispatchResult(
                ok=False, escalate_to_dispatch=True,
                error=f"intent not found: {intent_id}",
            )

        term = self._term_provider()
        for idx, selector in enumerate(intent.handler_selectors):
            from assistant.core.abort import abort, UserAborted
            if abort.is_aborted():
                raise UserAborted(abort.reason)
            from assistant.io.status_broadcaster import status, StatusPhase
            # Detail uses the intent id (e.g. "play_song") replacing underscores
            # with spaces. Falls back to empty if missing — step chip carries the count.
            _intent_id = getattr(intent, "id", "") or ""
            _detail = str(_intent_id).replace("_", " ")[:32]
            status.set(StatusPhase.CLICKING,
                       detail=_detail,
                       cursor_follows=True,
                       step=(idx + 1, len(intent.handler_selectors)),
                       tier="native")
            result = execute_primitive(
                selector, terminator=term, active_window=active_window,
            )
            if result.ok:
                self._on_selector_success(am, intent, idx)
                return DispatchResult(ok=True, selector_used_index=idx)
            # Capture the failing selector BEFORE bookkeeping, which may
            # swap selectors on demotion. We need the original Python object
            # to retry the same one (not whatever now sits at the same idx).
            failing_selector = selector
            self._on_selector_failure(am, intent, idx, result.error or "")
            # Healer hook (session 4): try to heal in place before falling through.
            if self._healer is not None:
                healed = self._healer.try_heal(
                    manifest=am, intent=intent, selector_index=idx,
                    active_window=active_window,
                )
                if healed.ok:
                    retry = execute_primitive(
                        failing_selector,
                        terminator=term, active_window=active_window,
                    )
                    if retry.ok:
                        # Locate failing_selector by identity (`is`), not by
                        # value — Pydantic __eq__ compares fields, so two
                        # distinct selectors with identical fields would
                        # mis-match. Identity walk is unambiguous.
                        new_idx = next(
                            (i for i, s in enumerate(intent.handler_selectors)
                             if s is failing_selector),
                            idx,
                        )
                        self._on_selector_success(am, intent, new_idx)
                        return DispatchResult(ok=True, selector_used_index=new_idx)

        # Operational failure — selectors tried and failed.
        return DispatchResult(
            ok=False, escalate_to_dispatch=True,
            error="all selectors exhausted",
        )

    # ─── Health bookkeeping ───────────────────────────────────────────────

    # Bookkeeping mutates shared state — wrap in self._lock.
    def _on_selector_success(
        self, am: AppManifest, intent: Intent, idx: int,
    ) -> None:
        with self._lock:
            intent.handler_selectors[idx].successes += 1
            intent.health.total_dispatches += 1
            intent.health.consecutive_failures = 0
            self._dirty_apps.add(am.app_id)
            self._maybe_flush()

    def _on_selector_failure(
        self, am: AppManifest, intent: Intent, idx: int, error: str,
    ) -> None:
        with self._lock:
            intent.handler_selectors[idx].failures += 1
            if idx == 0:
                intent.health.consecutive_failures += 1
                self._maybe_demote(am, intent, reason="3 consecutive failures")
            self._dirty_apps.add(am.app_id)
            self._maybe_flush()

    def _maybe_demote(self, am: AppManifest, intent: Intent, *, reason: str) -> None:
        """Check both demotion gates (selector failures + correction signals).

        Caller must hold ``self._lock``. Demotion swaps selectors[0] ↔ [1]
        and resets BOTH counters — a flip-flop on either signal is the
        expected v1 design.
        """
        if len(intent.handler_selectors) <= 1:
            return
        if (
            intent.health.consecutive_failures < 3
            and intent.health.consecutive_corrections < 3
        ):
            return
        intent.handler_selectors[0], intent.handler_selectors[1] = (
            intent.handler_selectors[1],
            intent.handler_selectors[0],
        )
        intent.health.consecutive_failures = 0
        intent.health.consecutive_corrections = 0
        logger.info(
            f"[manifest] {am.app_id}:{intent.id} demoted primary after {reason}"
        )

    def _maybe_flush(self) -> None:
        with self._lock:
            now = time.time()
            if now - self._last_flush < _HEALTH_FLUSH_INTERVAL_S:
                return
            to_flush = set(self._dirty_apps)
            self._dirty_apps.clear()
            self._last_flush = now
        # I/O outside the lock — write_and_invalidate hits disk + SQLite.
        for app_id in to_flush:
            am = self._store.get(app_id)
            if am is not None:
                self._store.write_and_invalidate(am)

    def flush_now(self) -> None:
        """Force flush of pending health updates. Called from shutdown handler."""
        with self._lock:
            self._last_flush = 0.0
        self._maybe_flush()

    # ─── Correction signal (called by T1 telemetry) ───────────────────────

    def record_correction(self, *, app_id: str, intent_id: str) -> None:
        """T1 detected a correction within 30s of our dispatch.

        Bumps the primary selector's failure count AND the intent's
        consecutive_corrections counter, then checks the demotion gate.
        Tracked separately from consecutive_failures so a successful
        dispatch between corrections doesn't erase the user signal — see
        F10 in the Session-5 post-mortem.

        v1 design: correction signals always penalize the primary
        (selector index 0), regardless of which selector actually won
        the last dispatch. Tracking the winning index per dispatch is a
        v1.1 improvement — flag in [[me1-session5-followup]] when written.
        """
        am = self._store.get(app_id)
        if am is None:
            return
        intent = next((i for i in am.intents if i.id == intent_id), None)
        if intent is None or not intent.handler_selectors:
            return
        with self._lock:
            intent.handler_selectors[0].failures += 1
            intent.health.consecutive_corrections += 1
            self._maybe_demote(am, intent, reason="3 consecutive corrections")
            self._dirty_apps.add(am.app_id)
            self._maybe_flush()
        logger.info(
            f"[manifest] correction signal recorded for {app_id}:{intent_id}"
        )

    def record_last_dispatch_correction(self) -> None:
        """Telemetry-side entry point — looks up the most recent (app_id,
        intent_id) pair we dispatched and proxies to record_correction.

        Used by assistant.telemetry.check_correction so it doesn't need to
        carry manifest-based params in the event row (they aren't in the schema).
        No-op when no dispatch has happened yet this process lifetime.
        """
        with self._lock:
            last = self._last_dispatch
        if last is None:
            return
        app_id, intent_id = last
        self.record_correction(app_id=app_id, intent_id=intent_id)
