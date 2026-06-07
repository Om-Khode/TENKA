"""healer.py — manifest-based self-healing layer (tier-1 AT fingerprint + tier-2 vision).

Tier-1: deterministic AT-tree priority-ladder rescore. Walks the live AT
tree under the saved parent_chain, scores each candidate via
at_fingerprint.fingerprint_score(), accepts the best if >= 0.50, patches
the manifest in place (Voyager pattern: intent.version += 1).

Tier-2: Gemini Flash-Lite vision crop+ground. Calls the (async) vision
contract via call_async (we are on a worker thread spun up by
asyncio.to_thread from handle_manifest_dispatch). Asks the model for coords,
clicks via pyautogui, re-resolves AT under the click point, patches the
manifest with the new parent_chain.

If both tiers miss, healer returns ok=False and the dispatcher demotes
the selector + falls through to computer_task.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .at_fingerprint import (
    FingerprintCandidate, find_best_match,
)
from .manifest_schema import AppManifest, Intent
from .manifest_store import ManifestStore
from .vision_cap import VisionCapTracker

logger = logging.getLogger("manifest")

_ACCEPT_THRESHOLD = 0.50          # Per spec §2.4
_VISION_CONFIDENCE_THRESHOLD = 0.7  # Per spec §3.3


@dataclass(frozen=True)
class HealResult:
    ok: bool
    tier: int                # 1 (AT fingerprint), 2 (vision), 0 (neither attempted)
    new_automation_id: str | None = None
    error: str | None = None


class Healer:
    def __init__(
        self,
        *, store: ManifestStore,
        terminator_provider: Callable[[], Any],
        vision_cap: VisionCapTracker | None,
    ) -> None:
        """Construct the healer.

        Args:
            store: ManifestStore that owns the YAML cache and write path.
            terminator_provider: Zero-arg callable returning a Terminator-like
                handle. Expected to return the SAME singleton each call (it
                is invoked once per tier, so a factory that opens a fresh
                connection per call would leak two connections per heal).
                In production this is ``lambda: terminator_singleton``.
            vision_cap: Daily Gemini-vision call counter, or None to disable
                tier-2 entirely (used by tier-1-only tests).
        """
        self._store = store
        self._term_provider = terminator_provider
        self._vision_cap = vision_cap

    def try_heal(
        self, *, manifest: AppManifest, intent: Intent,
        selector_index: int, active_window: str,
    ) -> HealResult:
        selector = intent.handler_selectors[selector_index]
        if selector.kind != "uia":
            return HealResult(ok=False, tier=0, error="only uia selectors heal")

        tier1 = self._try_tier1(manifest, intent, selector_index, active_window)
        if tier1.ok:
            return tier1

        if self._vision_cap is not None:
            tier2 = self._try_tier2(manifest, intent, selector_index, active_window)
            if tier2.ok:
                return tier2

        return HealResult(ok=False, tier=0, error="both tiers exhausted")

    # ─── Tier 1 — AT-tree fingerprint ──────────────────────────────────────

    def _try_tier1(
        self, manifest: AppManifest, intent: Intent,
        selector_index: int, active_window: str,
    ) -> HealResult:
        selector = intent.handler_selectors[selector_index]
        term = self._term_provider()
        try:
            raw_candidates = term.enumerate_descendants(
                parent_window=active_window, max_depth=4,
            )
        except Exception as e:
            logger.info(
                f"[manifest heal] tier-1 enumerate failed for {manifest.app_id}:"
                f"{intent.id}: {e}"
            )
            return HealResult(ok=False, tier=1, error=f"enumerate failed: {e}")

        candidates = [
            FingerprintCandidate(
                automation_id=c.get("automation_id", ""),
                control_type=c.get("control_type", ""),
                name=c.get("name", ""),
                parent_chain=tuple(c.get("parent_chain", [])),
                sibling_count=c.get("sibling_count", 0),
            )
            for c in raw_candidates
        ]
        if not candidates:
            logger.info(
                f"[manifest heal] tier-1 returned 0 candidates for {manifest.app_id}:"
                f"{intent.id} (window={active_window!r})"
            )
            return HealResult(
                ok=False, tier=1, error="enumerate returned 0 candidates",
            )
        best, score = find_best_match(
            selector, candidates,
            saved_sibling_count=0,  # we don't store saved sibling_count in v1
        )
        if best is None or score < _ACCEPT_THRESHOLD:
            best_desc = (
                f"name={best.name!r} aid={best.automation_id!r} "
                f"control_type={best.control_type!r}"
            ) if best is not None else "no candidate"
            logger.info(
                f"[manifest heal] tier-1 best score {score:.2f} < {_ACCEPT_THRESHOLD} "
                f"for {manifest.app_id}:{intent.id} "
                f"(searched {len(candidates)} candidates; best={best_desc}; "
                f"saved control_type={selector.control_type!r} "
                f"name_hint={selector.name_hint!r} "
                f"aid={selector.automation_id!r})"
            )
            return HealResult(
                ok=False, tier=1,
                error=f"best score {score:.2f} < {_ACCEPT_THRESHOLD}",
            )

        selector.automation_id = best.automation_id
        if best.parent_chain:
            selector.parent_chain = list(best.parent_chain)  # tuple → list for Pydantic
        if best.name:
            selector.name_hint = best.name
        intent.version += 1
        self._store.write_and_invalidate(manifest)
        logger.info(
            f"[manifest heal] tier-1 patched {manifest.app_id}:{intent.id} "
            f"automation_id → {best.automation_id} (score={score:.2f})"
        )
        return HealResult(ok=True, tier=1, new_automation_id=best.automation_id)

    # ─── Tier 2 — Gemini vision crop+ground ────────────────────────────────

    def _try_tier2(
        self, manifest: AppManifest, intent: Intent,
        selector_index: int, active_window: str,
    ) -> HealResult:
        if not self._vision_cap.try_increment():
            logger.info(
                f"[manifest heal] vision cap reached for {manifest.app_id}:{intent.id}"
            )
            return HealResult(ok=False, tier=2, error="daily vision cap reached")

        selector = intent.handler_selectors[selector_index]
        term = self._term_provider()
        try:
            screenshot = term.screenshot()
        except Exception as e:
            return HealResult(ok=False, tier=2, error=f"screenshot failed: {e}")

        from ..core.asyncio_utils import call_async
        from ..llm.contracts import ask_for_vision_ground_coords
        query = (
            (intent.display_name or intent.id)
            + " in " + (manifest.display_name or manifest.app_id)
        )
        try:
            coords = call_async(ask_for_vision_ground_coords(
                crop_bytes=screenshot, query=query, crop_origin=(0, 0),
            ))
        except Exception as e:
            return HealResult(ok=False, tier=2, error=f"vision call failed: {e}")
        if coords.get("confidence", 0.0) < _VISION_CONFIDENCE_THRESHOLD:
            return HealResult(ok=False, tier=2, error="vision low confidence")

        try:
            import pyautogui
            pyautogui.click(coords["x"], coords["y"])
        except Exception as e:
            return HealResult(ok=False, tier=2, error=f"click failed: {e}")

        try:
            elem = term.element_at_point(coords["x"], coords["y"])
            if elem and elem.get("automation_id"):
                selector.automation_id = elem["automation_id"]
                if elem.get("parent_chain"):
                    selector.parent_chain = list(elem["parent_chain"])
                intent.version += 1
                try:
                    self._store.write_and_invalidate(manifest)
                    logger.info(
                        f"[manifest heal] tier-2 AT re-resolve patched "
                        f"{manifest.app_id}:{intent.id} "
                        f"automation_id → {elem['automation_id']}"
                    )
                except Exception as write_err:
                    logger.warning(
                        f"[manifest heal] tier-2 AT re-resolve succeeded but persist "
                        f"failed for {manifest.app_id}:{intent.id}: {write_err}"
                    )
            else:
                # Common case after a navigation/close click: the post-click
                # window is gone (or different) so element_at_point returns
                # None or hits a node with no automation_id. The vision click
                # already produced its effect, but the selector stays
                # un-patched. Log so live-test can distinguish this from a
                # silent infra failure.
                logger.info(
                    f"[manifest heal] tier-2 AT re-resolve found no usable "
                    f"automation_id for {manifest.app_id}:{intent.id} "
                    f"at ({coords['x']}, {coords['y']}); selector not patched"
                )
        except Exception as e:
            logger.info(
                f"[manifest heal] tier-2 AT re-resolve skipped for "
                f"{manifest.app_id}:{intent.id}: {e}"
            )

        logger.info(
            f"[manifest heal] tier-2 vision succeeded for {manifest.app_id}:{intent.id}"
        )
        return HealResult(
            ok=True, tier=2,
            new_automation_id=selector.automation_id,
        )
