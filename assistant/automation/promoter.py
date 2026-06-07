"""promoter.py — manifest-based cold-path mining pipeline.

Reads unpromoted automation-cache successes, clusters by app, gates at N=2 distinct
(slug × 30-min time-window) buckets, asks Flash-Lite to verify trace
convergence, synthesizes paraphrase phrases, and writes the manifest YAML.

Runs at idle/shutdown, after N automation-cache saves, or via /promote slash command.
"""
from __future__ import annotations

import calendar
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

from .manifest_schema import (
    CURRENT_SCHEMA_VERSION,
    AppManifest,
    Captured,
    Health,
    Intent,
    Match,
    Selector,
)
from .manifest_store import ManifestStore
from ..llm.contracts import (
    ask_for_intent_clustering,
    ask_for_phrase_synthesis,
    ask_for_trace_diff_verification,
)
from ..storage.repos.automation_cache import AutomationCacheRepo

logger = logging.getLogger("manifest")

# Spec §promotion-algorithm: 30-minute time-window per bucket.
_BUCKET_SECONDS = 30 * 60
# Spec §promotion-algorithm: N=2 distinct (slug × bucket) buckets required.
_PROMOTION_THRESHOLD = 2
# Matches the backend string router.py writes via step_cache.save_cached_steps
# on the app_action path ("native"). Not "terminator" — that names the library,
# while automation-cache rows use "native" for the abstract "desktop UI automation" tier.
_ALLOWED_BACKENDS: set[str] = {"native"}


# ─── Shared in-flight flag ─────────────────────────────────────────────────
# Single source of truth so /promote and the 50-save auto-trigger debounce
# against each other. Two concurrent cycles would call find_unpromoted()
# before either calls mark_promoted() — duplicating Groq/Cerebras spend.
# The data side is idempotent (version-bump merge), so we only debounce to
# protect the free-tier LLM quota.
_in_flight: bool = False


def is_promotion_in_flight() -> bool:
    """True if a Promoter.run_once() cycle is currently scheduled or running."""
    return _in_flight


def _set_in_flight(value: bool) -> None:
    """Set the shared debounce flag. Module-private — callers use the slash
    or the auto-scheduler, not this directly."""
    global _in_flight
    _in_flight = value


# ─── Pure helpers ──────────────────────────────────────────────────────────

def _entry_epoch(entry: dict) -> float:
    """Parse an automation-cache entry's ``created_at`` to a POSIX epoch float.

    automation-cache stores ``created_at`` as an ISO-8601 string (``datetime.now().isoformat()``).
    A numeric value is also tolerated for legacy / direct-insert callers.
    Anything else falls back to ``0.0`` so a malformed row cannot crash the gate.

    Naive ISO strings (no tz) are interpreted as UTC via ``calendar.timegm``
    rather than ``.timestamp()``. The latter routes through Windows' ``mktime``,
    which raises ``OSError: [Errno 22]`` for dates the local timezone can't
    represent (notably anything near 1970-01-01). Bucketing only cares about
    deltas, so a constant UTC-vs-local shift preserves bucket distinctness.
    """
    raw = entry.get("created_at")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return 0.0
        if dt.tzinfo is None:
            return float(calendar.timegm(dt.timetuple())) + dt.microsecond / 1_000_000
        try:
            return dt.timestamp()
        except OSError:
            return 0.0
    return 0.0


def count_distinct_buckets(entries: list[dict]) -> int:
    """N=2 gate primitive: count distinct (goal_slug × 30-min window) buckets."""
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        slug = entry.get("goal_slug")
        if not slug:
            continue
        bucket = int(_entry_epoch(entry) // _BUCKET_SECONDS)
        seen.add((slug, bucket))
    return len(seen)


def group_automation_cache_entries_by_app(
    entries: list[dict], allowed_backends: set[str],
) -> dict[str, list[dict]]:
    """Filter to ``allowed_backends`` and group entries by ``app_name``."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        if entry.get("backend") not in allowed_backends:
            continue
        app_name = entry.get("app_name")
        if not app_name:
            continue
        grouped[app_name].append(entry)
    return dict(grouped)


# ─── Promoter ──────────────────────────────────────────────────────────────

class Promoter:
    """One-shot promotion cycle: automation-cache → LLM-clustered → verified → manifest."""

    def __init__(
        self, *, automation_cache_repo: AutomationCacheRepo, manifest_store: ManifestStore,
    ) -> None:
        self._automation_cache = automation_cache_repo
        self._store = manifest_store

    async def run_once(self) -> dict:
        """Run a single promotion cycle.

        Returns a summary dict suitable for logging / UX surface:
        ``{"apps_processed": int, "intents_promoted": int,
        "skipped_low_conf": int, "skipped_build_failure": int}``.

        Per-app failures are caught and logged so one bad app cannot block
        the cycle for the others.
        """
        summary = {
            "apps_processed": 0,
            "intents_promoted": 0,
            "skipped_low_conf": 0,
            "skipped_build_failure": 0,
        }

        unpromoted = self._automation_cache.find_unpromoted()
        grouped = group_automation_cache_entries_by_app(
            unpromoted, allowed_backends=_ALLOWED_BACKENDS,
        )

        for app_name, entries in grouped.items():
            try:
                result = await self._process_app(app_name, entries)
                summary["intents_promoted"] += result["promoted"]
                summary["skipped_low_conf"] += result["low_conf"]
                summary["skipped_build_failure"] += result["build_failures"]
                summary["apps_processed"] += 1
            except Exception as e:
                logger.warning(
                    f"[manifest promote] App '{app_name}' failed: {e}",
                    exc_info=True,
                )
                # Other apps continue.

        return summary

    async def _process_app(
        self, app_name: str, entries: list[dict],
    ) -> dict:
        """Process one app's unpromoted entries.

        Returns ``{"promoted": int, "low_conf": int, "build_failures": int}``.
        """
        zero = {"promoted": 0, "low_conf": 0, "build_failures": 0}
        if count_distinct_buckets(entries) < _PROMOTION_THRESHOLD:
            return zero

        goals = sorted({e["goal_slug"] for e in entries if e.get("goal_slug")})
        clusters = await ask_for_intent_clustering(app=app_name, goals=goals)
        if not clusters:
            return zero

        # Filter to high-confidence clusters that themselves meet the bucket
        # gate when restricted to their member goals.
        eligible: list[dict] = []
        low_conf_count = 0
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            if cluster.get("confidence") != "high":
                low_conf_count += 1
                continue
            members = set(cluster.get("members") or [])
            if not members:
                continue
            member_entries = [e for e in entries if e["goal_slug"] in members]
            if count_distinct_buckets(member_entries) >= _PROMOTION_THRESHOLD:
                eligible.append({**cluster, "_member_entries": member_entries})

        if not eligible:
            return {"promoted": 0, "low_conf": low_conf_count, "build_failures": 0}

        app_id = self._derive_app_id(app_name)
        existing = self._store.get(app_id)
        manifest = existing or self._fresh_manifest(app_id, app_name)

        promoted_count = 0
        build_failures = 0
        # Track (member_entries, intent_id) so we can claim automation-cache rows
        # ONLY after the YAML write succeeds. If write_and_invalidate
        # throws, the next run_once re-attempts; the _merge_intent
        # version-bump path makes the retry idempotent.
        pending_claims: list[tuple[list[dict], str]] = []
        for cluster in eligible:
            intent = await self._build_intent_from_cluster(cluster)
            if intent is None:
                build_failures += 1
                continue
            self._merge_intent(manifest, intent)
            pending_claims.append((cluster["_member_entries"], intent.id))
            promoted_count += 1

        if promoted_count > 0:
            # Durable artifact first — if this throws, nothing is claimed
            # and the next cycle retries cleanly.
            self._store.write_and_invalidate(manifest)
            # YAML is durable; now claim the automation-cache rows. A failure here
            # only leaves rows un-promoted (re-evaluated next cycle,
            # which is harmless because version-bump is idempotent).
            for member_entries, intent_id in pending_claims:
                for e in member_entries:
                    try:
                        self._automation_cache.mark_promoted(
                            e["backend"], e["app_name"], e["goal_slug"],
                            f"{app_id}:{intent_id}",
                        )
                    except Exception as mark_exc:
                        logger.warning(
                            f"[manifest promote] mark_promoted failed for "
                            f"{e.get('backend')}/{e.get('app_name')}/"
                            f"{e.get('goal_slug')}: {mark_exc}",
                            exc_info=True,
                        )
            logger.info(
                f"[manifest promote] {app_id}: promoted {promoted_count} intent(s)"
            )
        return {
            "promoted": promoted_count,
            "low_conf": low_conf_count,
            "build_failures": build_failures,
        }

    async def _build_intent_from_cluster(
        self, cluster: dict,
    ) -> Intent | None:
        """Build an Intent from a cluster — verifier + phrase synthesis."""
        intent_id = cluster.get("intent_id")
        if not intent_id:
            return None
        originals = list(cluster.get("phrases") or [])

        member_entries = cluster["_member_entries"]
        traces = []
        for entry in member_entries[:2]:
            try:
                traces.append(json.loads(entry["steps_json"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        if len(traces) < 2:
            logger.info(
                f"[manifest promote] Intent '{intent_id}': insufficient parseable "
                f"traces ({len(traces)}/2); skipping"
            )
            return None

        verdict = await ask_for_trace_diff_verification(traces=traces)
        primary = verdict.get("primary_primitive")
        if primary is None:
            logger.info(
                f"[manifest promote] Verifier returned no primary primitive for "
                f"'{intent_id}'; skipping"
            )
            return None

        # vision_reground is a session-4 dispatcher concept; no manifest
        # dispatcher in v1 (Task 14) handles kind="vision_reground" as the
        # primary primitive. It's still tolerated as a fallback selector
        # in `alternatives` (weight=0.5).
        if isinstance(primary, dict) and primary.get("kind") == "vision_reground":
            logger.info(
                f"[manifest promote] Verifier returned vision_reground as primary "
                f"for '{intent_id}'; not dispatchable in v1, skipping"
            )
            return None

        selectors: list[Selector] = []
        try:
            selectors.append(self._primitive_to_selector(primary, weight=1.0))
        except ValueError as e:
            logger.info(
                f"[manifest promote] Unknown primary primitive for '{intent_id}': {e}"
            )
            return None
        for alt in verdict.get("alternatives") or []:
            try:
                selectors.append(self._primitive_to_selector(alt, weight=0.5))
            except ValueError:
                continue

        synthesized = await ask_for_phrase_synthesis(
            intent_id=intent_id, originals=originals,
        )
        # STT input is lowercased before regex_router lookup; phrases must
        # be stored lowercased + stripped so utterances actually hit the
        # phrase index.
        phrases = sorted(
            {p.strip().lower() for p in originals if p and p.strip()}
            | {p.strip().lower() for p in synthesized if p and p.strip()}
        )

        return Intent(
            id=intent_id,
            display_name=intent_id.replace("_", " ").title(),
            phrases=phrases,
            version=1,
            handler_selectors=selectors,
            timeout_ms=3000,
            captured=Captured(
                tool="tenka_promotion",
                timestamp=datetime.now(timezone.utc).isoformat(),
                promoted_from_trace_ids=[
                    f"{e['app_name']}/{e['goal_slug']}@{int(_entry_epoch(e))}"
                    for e in member_entries[:2]
                ],
            ),
            health=Health(),
        )

    # ─── Sync helpers ──────────────────────────────────────────────────────

    def _derive_app_id(self, app_name: str) -> str:
        """app_name is lowercased + ``.desktop``-suffixed for the v1 namespace.

        Browser-side promotions will use a different suffix later; keeping
        the suffix explicit avoids future collisions.
        """
        slug = app_name.strip().lower().replace(" ", "_").replace("-", "_")
        return f"{slug}.desktop"

    def _fresh_manifest(self, app_id: str, app_name: str) -> AppManifest:
        """Build a minimal new AppManifest for a never-seen-before app.

        Uses the normalized slug (not the raw ``app_name``) for the exe
        process name so multi-word apps like ``"Media Player"`` get a
        matchable ``"media_player.exe"`` rather than ``"Media Player.exe"``.
        ``display_name`` keeps word boundaries — it's a human-readable label.
        """
        slug = app_id[:-len(".desktop")] if app_id.endswith(".desktop") else app_id
        return AppManifest(
            schema_version=CURRENT_SCHEMA_VERSION,
            app_id=app_id,
            display_name=app_name.replace("_", " ").title(),
            match=Match(
                process_names=[f"{slug}.exe"],
                window_title_patterns=[],
                url_patterns=[],
            ),
            intents=[],
        )

    def _primitive_to_selector(self, prim: dict, *, weight: float) -> Selector:
        """Translate the verifier's primitive dict into a typed Selector.

        Generic dispatch by ``kind`` — no app-specific logic.
        """
        if not isinstance(prim, dict):
            raise ValueError(f"primitive must be a dict, got {type(prim).__name__}")
        kind = prim.get("kind", "uia")
        if kind == "hotkey":
            return Selector(
                kind="hotkey",
                keys=prim.get("keys") or "",
                weight=weight,
            )
        if kind == "uia":
            return Selector(
                kind="uia",
                control_type=prim.get("control_type"),
                automation_id=prim.get("automation_id"),
                parent_chain=list(prim.get("parent_chain") or []),
                name_hint=prim.get("name_hint"),
                weight=weight,
            )
        if kind == "vision_reground":
            return Selector(
                kind="vision_reground",
                query=prim.get("query"),
                weight=weight,
            )
        raise ValueError(f"Unknown primitive kind: {kind!r}")

    def _merge_intent(self, manifest: AppManifest, intent: Intent) -> None:
        """If the intent_id already exists, bump version and replace; else append."""
        for i, existing in enumerate(manifest.intents):
            if existing.id == intent.id:
                intent.version = existing.version + 1
                manifest.intents[i] = intent
                return
        manifest.intents.append(intent)
