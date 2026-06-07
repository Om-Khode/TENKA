"""step_cache.py — Automation step cache facade.

Same pattern as code_executor/templates.py: save on success, load on
repeat, delete on failure. Backed by the automation_cache DB table
instead of filesystem scripts.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("automation")

_STEPS_VERSION = 1

# ─── manifest-based promotion counter ─────────────────────────────────────────────────
# Schedules a background Promoter.run_once() every Nth successful save.
# _save_counter is event-loop-only: all callers run on TENKA's single asyncio
# event loop. Do NOT increment from a thread (no lock) — use asyncio.to_thread
# wrappers if a future caller is threaded.
_save_counter = 0
_PROMOTE_EVERY_N_SAVES = 50

# Strong references to scheduled background tasks. asyncio's _all_tasks is a
# WeakSet, so without our own set + discard callback a fire-and-forget
# create_task() result can be GC'd mid-cycle, silently killing the run.
_background_tasks: set = set()

_STOP = frozenset({
    "my", "me", "i", "the", "a", "an", "to", "of", "in", "on",
    "do", "have", "is", "are", "can", "get", "what", "how",
    "please", "just", "from", "for", "and", "or", "all",
    "it", "that", "this", "with", "using",
})


def _make_goal_slug(goal: str) -> str:
    words = re.sub(r"[^\w\s]", "", goal.lower()).split()
    keywords = sorted(w for w in words if len(w) > 1 and w not in _STOP)
    return "_".join(keywords) if keywords else "unknown"


def _goal_matches_cached(current_goal: str, stored_goal: str) -> bool:
    if not stored_goal:
        return True

    def _keywords(text: str) -> set[str]:
        return {w for w in re.sub(r"[^\w\s]", "", text.lower()).split()
                if len(w) > 1 and w not in _STOP}

    cur = _keywords(current_goal)
    stored = _keywords(stored_goal)

    if not cur or not stored:
        return True

    overlap = len(cur & stored)
    total = min(len(cur), len(stored))
    ratio = overlap / total if total > 0 else 0
    logger.debug(f"[AC] Goal match: cur={cur}, stored={stored}, ratio={ratio:.2f}")
    return ratio >= 0.40


def _get_repo():
    from ..storage.db import get_db
    from ..storage.repos.automation_cache import AutomationCacheRepo
    db = get_db()
    if db is None:
        return None
    return AutomationCacheRepo(db)


def load_cached_steps(
    backend: str, app_name: str, goal: str,
) -> list[dict] | None:
    app_key = app_name.lower().strip()
    slug = _make_goal_slug(goal)
    repo = _get_repo()
    if repo is None:
        return None
    entry = repo.get(backend, app_key, slug)
    if entry is None:
        return None
    if entry.get("version", 0) != _STEPS_VERSION:
        logger.info(f"[AC] Stale version {entry.get('version')} (want {_STEPS_VERSION}), deleting")
        repo.delete(backend, app_key, slug)
        return None
    if not _goal_matches_cached(goal, entry["goal_text"]):
        logger.info(f"[AC] Cache hit but goal mismatch: '{goal}' vs '{entry['goal_text']}'")
        return None
    try:
        steps = json.loads(entry["steps_json"])
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[AC] Corrupt steps_json for {backend}/{app_key}/{slug}: {e} — deleting")
        repo.delete(backend, app_key, slug)
        return None
    logger.info(
        f"[AC] Cache HIT: {backend}/{app_key}/{slug} "
        f"(hits={entry['hit_count']}, goal='{entry['goal_text'][:60]}')"
    )
    repo.record_hit(backend, app_key, slug)
    return steps


def save_cached_steps(
    backend: str, app_name: str, goal: str, steps: list[dict],
) -> None:
    app_key = app_name.lower().strip()
    slug = _make_goal_slug(goal)
    repo = _get_repo()
    if repo is None:
        return
    repo.save(backend, app_key, slug, goal, steps)
    logger.info(f"[AC] Cached {len(steps)} steps: {backend}/{app_key}/{slug}")
    _maybe_schedule_promotion()


# ─── manifest-based promotion scheduling ─────────────────────────────────────────────


def _maybe_schedule_promotion() -> None:
    """Schedule a Promoter.run_once() every Nth successful save.

    Fire-and-forget on the running event loop. If no loop is running
    (tests, ad-hoc CLI), skip silently — never crash the save path.
    """
    global _save_counter
    _save_counter += 1
    if _save_counter % _PROMOTE_EVERY_N_SAVES != 0:
        return

    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — skip. The shutdown hook and /promote remain.
        return

    try:
        from .manifest_registry import get_singleton
        from . import promoter as promoter_mod
        from .promoter import Promoter
        from ..storage.db import get_db
        from ..storage.repos.automation_cache import AutomationCacheRepo

        registry = get_singleton()
        db = get_db()
        if registry is None or db is None:
            return

        # Debounce: skip if /promote (or a prior auto-cycle) is already
        # running. Shared flag in promoter.py is the single source of truth.
        if promoter_mod.is_promotion_in_flight():
            logger.debug(
                "[manifest promote] auto cycle skipped — promotion already in progress"
            )
            return

        # Snapshot the counter NOW so the log line reports the scheduling
        # count, not whatever value _save_counter has when the cycle ends.
        count_at_schedule = _save_counter

        promoter_mod._set_in_flight(True)
        promoter = Promoter(
            automation_cache_repo=AutomationCacheRepo(db),
            manifest_store=registry.store,
        )

        def _log_result(task: "asyncio.Task") -> None:
            promoter_mod._set_in_flight(False)
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.warning(
                    f"[manifest promote] auto cycle failed: {exc}", exc_info=exc,
                )
                return
            logger.info(
                f"[manifest promote] auto cycle ({count_at_schedule} saves) "
                f"summary: {task.result()}"
            )

        task = loop.create_task(promoter.run_once())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_log_result)
    except Exception as e:
        # Promotion is best-effort; never block automation-cache saves on its failure.
        # Release the debounce on the failure path — _log_result never runs
        # if the task was never scheduled.
        try:
            from . import promoter as promoter_mod
            promoter_mod._set_in_flight(False)
        except Exception:
            pass
        logger.debug(f"[manifest promote] auto schedule skipped: {e}")


def delete_cached_steps(
    backend: str, app_name: str, goal: str,
) -> None:
    app_key = app_name.lower().strip()
    slug = _make_goal_slug(goal)
    repo = _get_repo()
    if repo is None:
        return
    repo.delete(backend, app_key, slug)


def cleanup_expired(max_age_days: int = 30) -> int:
    repo = _get_repo()
    if repo is None:
        return 0
    return repo.cleanup_expired(max_age_days)
