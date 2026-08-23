"""A model's proposal cannot steer execution on its own say-so.

`automation/router.py:_check_routing_preference` is routing **priority 1** --
ahead of URL detection, running process, everything. What it accepts decides
how a goal executes before anything else gets a vote.

Two things met badly:

* `reflection.py` passes the *model's own* `confidence` through to
  `set_preference`, and its prompt offers 0.8 for "the user explicitly stated
  this". So one nightly cycle could write a preference above
  `CONFIDENCE_SILENT` on the model's assessment of its own evidence.
* the consumer accepted a bare `>= 0.4` -- `CONFIDENCE_ASK`, the bar for
  "apply, but mention it". Routing is silent.

The ladder in `storage/repos/preference.py` already encoded the right design:
discover at 0.4, +0.15 per re-observation, 0.7 to act silently. Repetition
earns trust. Reflection walked around it, so the design was never wrong -- it
was just unenforced.

Both halves are pinned here, and both directions of each: a fix that refused
every inferred preference would make reflection pointless, and one that
refused user-stated preferences would make the whole feature pointless.

Run with:  py -3.11 -m pytest tests/test_preference_provenance.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.storage.repos.preference import (  # noqa: E402
    CONFIDENCE_ASK, CONFIDENCE_FIRST_OBSERVATION, CONFIDENCE_REOBSERVED,
    CONFIDENCE_SILENT, USER_STATED_SOURCES, PreferenceRepo,
)


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """The facade, not the repo: the clamp lives at the facade because that is
    where every production writer goes through, and the repo stays a faithful
    storage layer. Testing the repo directly would test the wrong boundary."""
    from assistant import preferences as prefs
    from assistant.storage.db import Database

    db = Database(tmp_path / "t.db")
    real = PreferenceRepo(db)
    monkeypatch.setattr(prefs, "_repo", real, raising=False)
    monkeypatch.setattr(prefs, "_get_repo", lambda: real, raising=False)
    try:
        yield prefs
    finally:
        db._conn.close()


# ─── the write boundary: a proposal is not evidence ──────────────────────────

@pytest.mark.parametrize("source", ["reflection", "inference", "assistant"])
def test_a_model_cannot_grade_its_own_evidence(repo, source):
    """The model asks for 0.9. It gets the discovery floor."""
    repo.set_preference("automation_music", "app_a", "app_routing",
                        confidence=0.9, source=source, reason="model says so")

    got = repo.get_preference("automation_music")
    assert got["confidence"] == CONFIDENCE_FIRST_OBSERVATION, (
        f"a {source}-written preference kept confidence {got['confidence']}. "
        f"The model graded its own evidence, and routing reads this."
    )


def test_an_unrecognised_writer_is_capped_too(repo):
    """Fail-closed on the permissive side. Keying the clamp on a list of
    known-model sources would let a writer nobody classified start wherever
    it liked -- the same silence `DEFAULT_REQUIRED` exists to prevent."""
    repo.set_preference("automation_music", "app_a", "app_routing",
                        confidence=0.95, source="some_new_subsystem",
                        reason="")

    got = repo.get_preference("automation_music")
    assert got["confidence"] == CONFIDENCE_FIRST_OBSERVATION, (
        "an unclassified writer set its own confidence"
    )


@pytest.mark.parametrize("source", sorted(USER_STATED_SOURCES))
def test_the_user_may_start_high(repo, source):
    """The other direction. They said it -- that is the evidence."""
    repo.set_preference("automation_music", "app_a", "app_routing",
                        confidence=0.9, source=source, reason="user said so")

    got = repo.get_preference("automation_music")
    assert got["confidence"] == 0.9, (
        f"a {source}-sourced preference was capped to {got['confidence']}; "
        f"the user's own statement is being treated as a guess"
    )


def test_a_low_proposal_is_left_alone(repo):
    """The clamp is a ceiling, not an assignment. A model proposing 0.2 must
    not be promoted to 0.4 by the thing meant to restrain it."""
    repo.set_preference("automation_music", "app_a", "app_routing",
                        confidence=0.2, source="reflection", reason="")
    assert repo.get_preference("automation_music")["confidence"] == 0.2


def test_repetition_is_still_the_way_up(repo):
    """The ladder has to remain walkable, or the clamp has just disabled
    reflection. Two re-observations carry a discovery to the silent bar."""
    repo.set_preference("automation_music", "app_a", "app_routing",
                        confidence=0.9, source="reflection", reason="")
    repo.bump_confidence("automation_music", delta=CONFIDENCE_REOBSERVED)
    final = repo.bump_confidence("automation_music", delta=CONFIDENCE_REOBSERVED)

    assert final >= CONFIDENCE_SILENT, (
        f"after two re-observations the preference is at {final}, still below "
        f"{CONFIDENCE_SILENT}. Reflection can no longer earn a routing "
        f"decision at all, which is not the fix -- it is the feature removed."
    )


# ─── the read boundary: routing checks where it came from ────────────────────

def _route(monkeypatch, *, source, confidence):
    """Ask `_check_routing_preference` about a goal naming one app."""
    from assistant.automation import router

    class _Prefs:
        @staticmethod
        def get_preference(key, category=None):
            if key == "automation_someapp":
                return {"key": key, "value": "native",
                        "confidence": confidence, "source": source}
            return None

    # `_check_routing_preference` does `from .. import preferences`, which
    # reads the ATTRIBUTE on the `assistant` package -- not `sys.modules`.
    # Patching sys.modules worked in isolation (nothing had imported it yet)
    # and silently stopped working once any earlier test did, which is a test
    # that passes alone and fails in company. Patch the attribute.
    import assistant
    monkeypatch.setattr(assistant, "preferences", _Prefs, raising=False)
    return router._check_routing_preference("do the thing on someapp")


def test_routing_ignores_a_fresh_inference(monkeypatch):
    """0.4 is `CONFIDENCE_ASK` -- apply but mention it. Routing is silent, so
    a fresh model discovery must not decide it."""
    assert _route(monkeypatch, source="reflection",
                  confidence=CONFIDENCE_FIRST_OBSERVATION) is None, (
        "a preference the model proposed last night steered routing ahead of "
        "URL detection, on its first observation"
    )


def test_routing_accepts_a_corroborated_inference(monkeypatch):
    """Repetition TENKA counted itself. This is what reflection is for."""
    assert _route(monkeypatch, source="reflection",
                  confidence=CONFIDENCE_SILENT) == "native"


def test_routing_accepts_a_user_stated_preference_at_the_lower_bar(monkeypatch):
    """They said it once. That is enough -- the bar exists to discount
    guesses, not statements."""
    assert _route(monkeypatch, source="user",
                  confidence=CONFIDENCE_ASK) == "native"


def test_routing_ignores_an_unknown_source_below_the_silent_bar(monkeypatch):
    assert _route(monkeypatch, source="mystery",
                  confidence=CONFIDENCE_ASK) is None


def test_the_provenance_sets_are_not_empty():
    """Anti-vacuity: an empty `USER_STATED_SOURCES` would make the clamp
    universal and every test above would still pass for the wrong reason."""
    assert USER_STATED_SOURCES, "no user-stated sources -- clamp is universal"
    assert CONFIDENCE_SILENT > CONFIDENCE_ASK, "the two bars are not distinct"
    assert CONFIDENCE_FIRST_OBSERVATION < CONFIDENCE_SILENT, (
        "a discovery already meets the silent bar; the clamp buys nothing"
    )
