"""Tests for manifest-based AT-tree fingerprint scorer."""

import json
from pathlib import Path

from assistant.automation.at_fingerprint import (
    FingerprintCandidate, fingerprint_score, find_best_match,
)
from assistant.automation.manifest_schema import Selector


FIXTURES = Path(__file__).parent / "fixtures" / "at_trees"


def _to_candidate(d):
    return FingerprintCandidate(
        automation_id=d.get("automation_id", ""),
        control_type=d.get("control_type", ""),
        name=d.get("name", ""),
        parent_chain=tuple(d.get("parent_chain", [])),
        sibling_count=d.get("sibling_count", 0),
    )


def test_exact_match_scores_1():
    saved = Selector(
        kind="uia", control_type="Button",
        automation_id="play-pause-button",
        parent_chain=["Window[Name~'TestApp']", "Pane[ClassName~'Chrome_WidgetWin']"],
        name_hint="Play",
    )
    cand = FingerprintCandidate(
        automation_id="play-pause-button", control_type="Button",
        name="Play",
        parent_chain=("Window[Name~'TestApp']", "Pane[ClassName~'Chrome_WidgetWin']"),
        sibling_count=5,
    )
    score = fingerprint_score(saved, cand, saved_sibling_count=5)
    assert score >= 1.0 - 1e-6


def test_control_type_mismatch_scores_0():
    saved = Selector(kind="uia", control_type="Button", automation_id="x")
    cand = FingerprintCandidate(
        automation_id="x", control_type="Edit", name="", parent_chain=(), sibling_count=0,
    )
    assert fingerprint_score(saved, cand, saved_sibling_count=0) == 0.0


def test_renamed_id_with_matching_name_chain_passes_threshold():
    """Same name + chain, different automation_id."""
    saved = Selector(
        kind="uia", control_type="Button",
        automation_id="play-pause-button",
        parent_chain=["Window[Name~'TestApp']", "Pane[ClassName~'Chrome_WidgetWin']"],
        name_hint="Play",
    )
    cand = FingerprintCandidate(
        automation_id="play_btn_v2",
        control_type="Button", name="Play",
        parent_chain=("Window[Name~'TestApp']", "Pane[ClassName~'Chrome_WidgetWin']"),
        sibling_count=5,
    )
    score = fingerprint_score(saved, cand, saved_sibling_count=5)
    # 0.30 (control_type) + 0.10 (name) + 0.15 (parent) + 0.05 (siblings) = 0.60
    assert score >= 0.5
    assert score < 1.0  # automation_id mismatch (0.40 missing)


def test_find_best_match_picks_top_candidate():
    saved = Selector(
        kind="uia", control_type="Button",
        automation_id="play-pause-button",
        parent_chain=["Window[Name~'TestApp']"],
        name_hint="Play",
    )
    tree = json.loads((FIXTURES / "test_app_renamed.json").read_text())
    candidates = [_to_candidate(e) for e in tree["elements"].values()]
    best, score = find_best_match(saved, candidates, saved_sibling_count=2)
    assert best.automation_id == "play_btn_v2"
    assert score >= 0.5


def test_find_best_match_empty_candidates_returns_none():
    """Empty candidate list → (None, 0.0). Contract for the healer's no-AT-tree path."""
    saved = Selector(kind="uia", control_type="Button")
    best, score = find_best_match(saved, [], saved_sibling_count=0)
    assert best is None
    assert score == 0.0


def test_f14b_empty_saved_parent_chain_still_awards_chain_bonus():
    """F14b regression: a v1-promoted manifest's empty parent_chain (the
    common case — the promoter writes [] when automation-cache trace doesn't carry
    chain info) must still pick up the 0.15 chain bonus on score.

    Without this, the maximum possible score for a renamed-aid heal
    when name + control_type match is 0.40 (control_type 0.30 + name 0.10),
    which is below the 0.50 acceptance threshold. Tier-1 heal becomes
    effectively dead for all naturally-promoted manifests. Caught live
    on Notepad's Close button during Scenario-3.
    """
    saved = Selector(
        kind="uia", control_type="Button",
        automation_id="heal_demo_broken_aid_v1",
        parent_chain=[],            # ← empty — the v1 promoter default
        name_hint="Close",
    )
    cand = FingerprintCandidate(
        automation_id="real_aid_for_close",
        control_type="Button", name="Close",
        parent_chain=("Notepad", "TitleBar", "MinMaxCloseGroup"),
        sibling_count=3,
    )
    # 0.30 (control_type) + 0.10 (name) + 0.15 (empty saved chain ⇒ no
    # constraint ⇒ trivially satisfied) = 0.55, which clears 0.50.
    score = fingerprint_score(saved, cand, saved_sibling_count=0)
    assert score >= 0.5, (
        f"Empty saved parent_chain must contribute 0.15 (no constraint = "
        f"trivially satisfied) so that name + control_type matches reach the "
        f"0.50 heal threshold for naturally-promoted manifests. Got {score:.2f}."
    )
    assert abs(score - 0.55) < 1e-6


def test_f14b_non_empty_saved_chain_still_requires_match():
    """F14b counterpart: a saved chain that's set MUST actually match the
    candidate's chain in order. Drops to no-bonus on mismatch.
    """
    saved = Selector(
        kind="uia", control_type="Button",
        automation_id="x", parent_chain=["RequiredAncestor"], name_hint="Close",
    )
    cand = FingerprintCandidate(
        automation_id="x", control_type="Button", name="Close",
        parent_chain=("DifferentAncestor", "Other"),
        sibling_count=3,  # non-zero so the +0.05 sibling bonus doesn't fire
    )
    # 0.30 + 0.40 (aid match) + 0.10 (name) = 0.80 — chain bonus NOT added
    # because "RequiredAncestor" doesn't appear in the candidate's chain.
    score = fingerprint_score(saved, cand, saved_sibling_count=0)
    assert abs(score - 0.80) < 1e-6
