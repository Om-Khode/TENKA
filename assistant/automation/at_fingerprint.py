"""at_fingerprint.py — AT-tree priority-ladder fingerprint scoring.

Ported from arxiv 2603.20358 (100% recovery on 31 web AT trees) with
weights adapted to Windows UIA per manifest-based design spec §2.4:

  control_type match    : 0.30 (REQUIRED — if missing, score=0)
  automation_id match   : 0.40
  name match            : 0.10
  parent chain match    : 0.15
  sibling count match   : 0.05

Acceptance threshold (per spec): score >= 0.50.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .manifest_schema import Selector


@dataclass(frozen=True)
class FingerprintCandidate:
    automation_id: str
    control_type: str
    name: str
    parent_chain: tuple[str, ...]
    sibling_count: int


def fingerprint_score(
    saved: Selector, cand: FingerprintCandidate, *, saved_sibling_count: int,
) -> float:
    """Score a candidate against a saved selector.

    Returns 0.0 if control_type does not match (required gate).
    Otherwise returns the sum of weighted feature matches.
    """
    if (saved.control_type or "") != cand.control_type:
        return 0.0

    score = 0.30  # control_type matched (required gate cleared)

    if saved.automation_id and saved.automation_id == cand.automation_id:
        score += 0.40

    if saved.name_hint and saved.name_hint == cand.name:
        score += 0.10

    # Empty saved.parent_chain semantically means "no chain constraint" —
    # _parent_chain_match returns True in that case (no segments to violate).
    # The earlier `saved.parent_chain and ...` guard short-circuited on the
    # falsy empty list and dropped 0.15 from the score, making 0.50 unreachable
    # for any v1-promoted manifest (the promoter writes parent_chain=[]).
    # See F14b in the Session-5 post-mortem.
    if _parent_chain_match(saved.parent_chain, cand.parent_chain):
        score += 0.15

    if saved_sibling_count == cand.sibling_count:
        score += 0.05

    return score


def find_best_match(
    saved: Selector,
    candidates: list[FingerprintCandidate],
    *, saved_sibling_count: int,
) -> tuple[FingerprintCandidate | None, float]:
    """Walks candidates, returns (best_candidate, score) or (None, 0.0)."""
    best: FingerprintCandidate | None = None
    best_score = 0.0
    for cand in candidates:
        s = fingerprint_score(saved, cand, saved_sibling_count=saved_sibling_count)
        if s > best_score:
            best = cand
            best_score = s
    return best, best_score


def _parent_chain_match(saved_chain: Sequence[str], cand_chain: Sequence[str]) -> bool:
    """Soft chain match: every saved_chain segment must appear in order in cand_chain."""
    if not saved_chain:
        return True
    si = 0
    for c in cand_chain:
        if si < len(saved_chain) and saved_chain[si] == c:
            si += 1
    return si == len(saved_chain)
