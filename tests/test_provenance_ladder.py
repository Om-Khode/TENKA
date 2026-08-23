"""A model agreeing with itself is not evidence, and provenance gates the read.

Three rules, from TENKA-v2 §10.

**D3 — a single inference never becomes an unattended behaviour change.** The
obvious version of this rule is useless, and the tree had the useless version.
The reflection prompt says *"minimum 3 occurrences of a pattern"* and nothing
ever checked; `source="reflection"` records the **provider**, not the
**evidence**. Writes were clamped to `CONFIDENCE_FIRST_OBSERVATION` (0.4), but
`bump_confidence` added +0.15 per re-proposal with nothing verifying the count.
So:

    night 1  model proposes music_app=X   -> 0.40  (clamped)
    night 2  model proposes it again      -> 0.55
    night 3  model proposes it again      -> 0.70  == CONFIDENCE_SILENT

At `CONFIDENCE_SILENT` two things change at once: the preference is applied
*silently*, and it becomes eligible for `actions._build_goal_hints`, which
appends it to the `code_executor` and `planner` prompts. The output of those
prompts runs in a subprocess. Three nights of one model agreeing with itself is
the entire evidence chain.

**D2 — provenance is consulted on read**, and the strictest consumer is the one
with the worst blast radius. `automation/router.py` accepts a model-proposed
preference at a lower floor because the worst case is opening the wrong
application. `_build_goal_hints` accepts none, at any confidence, because its
destination is a code generator.

**§10.4 — `save_fact`'s `source` no longer defaults**, and expired facts are not
returned on any read path.

Both directions throughout. A cap that froze every preference forever, or a
filter that returned nothing, would satisfy every "does not overclaim"
assertion here while breaking the feature it guards.

Run with:  py -3.11 -m pytest tests/test_provenance_ladder.py -v
"""
import inspect
import pathlib
import sys
from datetime import datetime, timedelta

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.storage.repos.preference import (  # noqa: E402
    CONFIDENCE_APPLIED_NO_COMPLAINT, CONFIDENCE_ASK, CONFIDENCE_SILENT,
    MODEL_PROPOSED_CEILING, PreferenceRepo, USER_STATED_SOURCES,
)


@pytest.fixture()
def prefs(tmp_path):
    from assistant.storage.db import Database
    db = Database(tmp_path / "p.db")
    try:
        yield PreferenceRepo(db)
    finally:
        db._conn.close()


@pytest.fixture()
def facts(tmp_path):
    from assistant.storage.db import Database
    from assistant.storage.repos.memory import MemoryRepo
    db = Database(tmp_path / "m.db")
    try:
        yield MemoryRepo(db, tmp_path), db
    finally:
        db._conn.close()


# ─── D3: the model does not count its own evidence ───────────────────────────

def test_a_model_reassertion_cannot_pass_the_ceiling(prefs):
    """The three-night walk, run as three bumps. Each is the reflection cycle
    reporting that its own earlier proposal showed up again."""
    prefs.set_preference("music_app", "X", "app_routing", 0.4, "reflection", "r")
    for _ in range(3):
        prefs.bump_confidence("music_app", delta=0.15, counted_by_tenka=False)

    got = prefs.get_preference("music_app")["confidence"]
    assert got <= MODEL_PROPOSED_CEILING, (
        f"a model re-asserting its own proposal three times reached {got}. "
        f"At {CONFIDENCE_SILENT} it is applied silently and enters a "
        f"code-generation prompt."
    )


def test_the_ceiling_sits_below_the_silent_threshold(prefs):
    """The number has to be *load-bearing*, not merely lower. If the ceiling
    ever rose to or above `CONFIDENCE_SILENT` the cap would still 'work' while
    permitting exactly the outcome it exists to prevent."""
    assert MODEL_PROPOSED_CEILING < CONFIDENCE_SILENT
    assert MODEL_PROPOSED_CEILING == CONFIDENCE_ASK, (
        "the ceiling should be the level at which a preference is applied AND "
        "mentioned -- a wrong guess the user can see and correct"
    )


def test_an_observed_turn_may_raise_a_preference_past_the_ceiling(prefs):
    """**The other direction, and the one that matters.** A cap that applied to
    everything would freeze every preference at 0.4 forever and quietly delete
    the learning feature. `record_preference_used` fires after a preference
    was really used on a real turn and the user did not override it -- a fact
    about the world, so it counts."""
    prefs.set_preference("music_app", "X", "app_routing", MODEL_PROPOSED_CEILING,
                         "reflection", "r")
    for _ in range(20):
        prefs.record_preference_used("music_app")

    got = prefs.get_preference("music_app")["confidence"]
    assert got > MODEL_PROPOSED_CEILING, (
        f"observed successful use could not raise the preference past the "
        f"model ceiling ({got}) -- confidence is now unreachable and the "
        f"preference system cannot learn"
    )
    assert CONFIDENCE_APPLIED_NO_COMPLAINT > 0, "the observation delta is inert"


def test_a_model_reassertion_never_lowers_an_earned_confidence(prefs):
    """A preference already above the ceiling earned that from real evidence.
    A model re-assertion must not move it in *either* direction -- capping
    downward would let the reflection cycle demote what observation established.
    """
    prefs.set_preference("music_app", "X", "app_routing", 0.9, "user", "r")
    prefs.bump_confidence("music_app", delta=0.15, counted_by_tenka=False)
    assert prefs.get_preference("music_app")["confidence"] == pytest.approx(0.9)


def test_reflection_is_the_only_caller_that_relays_a_model_claim():
    """Source-level. `counted_by_tenka` defaults to True, so a future caller
    that forgets it is treated as evidence -- the unsafe direction. What keeps
    that safe is that there is exactly one relay of a model's claim, and it
    says so. A second one appearing without the keyword should be noticed here.
    """
    import ast

    # AST rather than a line scan: the reflection call is wrapped across two
    # lines, so a per-line regex sees `bump_confidence(` on one line and the
    # keyword on the next and reports a violation that is not there. The first
    # version of this test did exactly that.
    hits = []
    for path in sorted((_ROOT / "assistant").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:      # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name != "bump_confidence":
                continue
            declared = any(k.arg == "counted_by_tenka" for k in node.keywords)
            hits.append((path.name, node.lineno, declared))

    assert hits, "no bump_confidence calls found -- the sweep walks nothing"

    reflection_calls = [h for h in hits if h[0] == "reflection.py"]
    assert reflection_calls, "reflection.py no longer bumps -- move this check"
    undeclared = [(n, ln) for n, ln, declared in reflection_calls if not declared]
    assert not undeclared, (
        f"reflection relays a model's claim without declaring it: {undeclared}. "
        f"`counted_by_tenka` defaults to True, so an omission here silently "
        f"restores the three-nights-to-0.70 walk."
    )


def test_the_reflection_bump_declares_itself_uncounted():
    """The specific call, read off the source, because the argument travels two
    frames (reflection -> facade -> repo) and a default of True at either hop
    would silently restore the old behaviour."""
    from assistant import reflection
    src = inspect.getsource(reflection)
    assert "counted_by_tenka=False" in src, (
        "reflection's bump no longer declares itself uncounted, so a model "
        "re-proposal is being treated as evidence TENKA gathered"
    )


# ─── D2: the code-generation prompt takes user-stated provenance only ────────

def _hints_with(monkeypatch, rows):
    import assistant.actions as A
    import assistant.preferences as preferences_mod
    monkeypatch.setattr(preferences_mod, "get_active_preferences",
                        lambda **kw: rows)
    return A._build_goal_hints()


def test_a_model_proposed_preference_never_reaches_the_code_prompt(monkeypatch):
    """Even at confidence 1.0. This is the destination rule, not the confidence
    rule: a value no human stated does not get appended to a prompt whose
    output is executed."""
    hints = _hints_with(monkeypatch, [{
        "key": "music_app", "value": "rm -rf /", "category": "app_routing",
        "confidence": 1.0, "source": "reflection",
    }])
    assert hints == "", (
        f"a reflection-written preference reached the code-generation prompt: "
        f"{hints!r}"
    )


def test_a_user_stated_preference_still_reaches_the_code_prompt(monkeypatch):
    """**The answer, not the refusal.** A filter that dropped everything would
    pass the test above while silently removing preference-aware code
    generation -- which looks like the model being unhelpful, not like a bug."""
    hints = _hints_with(monkeypatch, [{
        "key": "music_app", "value": "spotify", "category": "app_routing",
        "confidence": CONFIDENCE_SILENT, "source": "user",
    }])
    assert "music_app=spotify" in hints, f"a user-stated preference was dropped: {hints!r}"


@pytest.mark.parametrize("source", sorted(USER_STATED_SOURCES))
def test_every_user_stated_spelling_is_accepted(monkeypatch, source):
    """`USER_STATED_SOURCES` has four spellings and all four mean the user said
    it. A filter that honoured only `"user"` would silently drop corrections --
    the highest-trust provenance there is."""
    hints = _hints_with(monkeypatch, [{
        "key": "k", "value": "v", "category": "app_routing",
        "confidence": 0.9, "source": source,
    }])
    assert "k=v" in hints, f"source {source!r} was rejected"


def test_a_row_with_no_source_is_rejected(monkeypatch):
    """Fail closed. Absent provenance is not user-stated provenance."""
    assert _hints_with(monkeypatch, [{
        "key": "k", "value": "v", "category": "app_routing", "confidence": 1.0,
    }]) == ""


def test_the_router_is_deliberately_less_strict_than_the_code_prompt():
    """The two consumers differ on purpose, and the difference is the argument.
    Collapsing them either lets model-proposed values into a code generator or
    stops TENKA ever acting on a learned routing preference. Pinned so a future
    tidy-up has to read the reason first."""
    from assistant.automation import router
    src = inspect.getsource(router._check_routing_preference)
    assert "USER_STATED_SOURCES" in src, "the router stopped reading provenance"
    assert "CONFIDENCE_SILENT" in src and "CONFIDENCE_ASK" in src, (
        "the router no longer varies its floor by provenance -- it either "
        "trusts everything or nothing"
    )


# ─── §10.4: required provenance, and expiry honoured on read ─────────────────

def test_save_fact_requires_a_source():
    """It defaulted to `"user"`, the highest trust tier, so a forgotten
    argument manufactured an explicit user statement."""
    from assistant import memory
    from assistant.storage.repos.memory import MemoryRepo

    for fn in (memory.save_fact, MemoryRepo.save_fact):
        sig = inspect.signature(fn)
        assert sig.parameters["source"].default is inspect.Parameter.empty, (
            f"{fn.__qualname__} still defaults `source`"
        )


def test_an_expired_fact_is_not_returned_by_search(facts):
    """`main._build_facts_context()` calls `search_facts("user_")` and puts the
    result into the system prompt as KNOWN FACTS ABOUT THE USER, every turn. An
    expired row was still asserted as currently true."""
    repo, _ = facts
    past = (datetime.now() - timedelta(days=1)).isoformat()
    future = (datetime.now() + timedelta(days=1)).isoformat()
    repo.save_typed_fact("user_city", "Delhi", "user", "fact", expires_at=past)
    repo.save_typed_fact("user_name", "Om", "user", "fact", expires_at=future)

    keys = {f["key"] for f in repo.search_facts("user_")}
    assert "user_name" in keys, "a live fact was dropped -- the filter is too broad"
    assert "user_city" not in keys, (
        "an expired fact was returned and would be asserted as current fact in "
        "the system prompt"
    )


def test_a_fact_with_no_expiry_is_always_returned(facts):
    """`identity` and `preference` types get no `expires_at` at all. A filter
    that dropped NULLs would erase everything TENKA knows permanently."""
    repo, _ = facts
    repo.save_typed_fact("user_name", "Om", "user", "identity")
    assert [f["key"] for f in repo.search_facts("user_")] == ["user_name"]


def test_the_degraded_hybrid_path_also_filters_expiry(facts):
    """The fused path filtered and the LIKE fallback did not -- correct on the
    normal route, lapsed on the degraded one. That fallback runs precisely when
    semantic and FTS both came back empty, so it is the path taken when things
    are already going badly."""
    repo, _ = facts
    past = (datetime.now() - timedelta(days=1)).isoformat()
    repo.save_typed_fact("user_city", "Delhi", "user", "fact", expires_at=past)

    # Force the fallback: no semantic index and no FTS hit for this query.
    results = repo.hybrid_search_facts("Delhi")
    assert all(r["key"] != "user_city" for r in results), (
        f"the LIKE fallback returned an expired fact: {results}"
    )
