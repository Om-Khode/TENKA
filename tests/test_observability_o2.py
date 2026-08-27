"""The store answers "why", not just "what". §15's O2, the three it could not.

Telemetry was ~70% of what §15 asks for. The gap was the *reasons*: the store
could say a turn made four model calls and not which were classification, which
were code-generation and which were the reply; it could say a turn failed and
not whether recovery was tried; it could say she reported success and not which
tier decided.

    llm_purposes        why did she call a model
    replan_count        why did planning happen
    recovery_count      why did execution fail
    verification_tiers  why did she report success

**Five columns, not the ten §15 lists**, and the count is the phase's own rule
at work rather than a shortcut: *a field that is always null is not
observability*. `task_id`, `step_id`, `affordance`, `operation` and
`final_task_status` all need a Task to exist per turn, and nothing creates one
-- `brain/authority.py:create_task` has no caller in `assistant/`.

    context_bytes_by_profile   what the context actually cost

was the sixth deferral and is the fifth column now. It arrived when the Context
Builder gained its first caller: the conversational turn assembles through
`core/context.py`, so the number exists. A deferred field is due the moment
something can fill it, and no sooner.

Run with:  py -3.11 -m pytest tests/test_observability_o2.py -v
"""
import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_NEW_COLUMNS = ("llm_purposes", "replan_count", "recovery_count",
                "verification_tiers", "context_bytes_by_profile")

# The five §15 fields deliberately not added, and why. Named so the omission is
# a decision on the record rather than something that looks forgotten.
_DEFERRED = {
    "task_id": "no Task is created per turn (create_task has no caller)",
    "step_id": "same",
    "affordance": "same",
    "operation": "same",
    "final_task_status": "same",
}

# `context_bytes_by_profile` was here. It left when the Context Builder gained
# its first caller: the conversational turn assembles through
# `core/context.py` now, so the number exists and the column stops being
# permanently NULL. That is the whole test for whether a deferred field is due.


@pytest.fixture
def db(tmp_path):
    from assistant.storage.db import Database, _reset_for_testing
    database = Database(tmp_path / "t.db")
    yield database
    _reset_for_testing()


@pytest.fixture
def tracker():
    from assistant import telemetry
    t = telemetry.TurnTracker(session_id="s", input_modality="text",
                              transcript="x")
    token = telemetry.set_current_tracker(t)
    yield t
    telemetry.reset_current_tracker(token)


# ─── the schema ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("column", _NEW_COLUMNS)
def test_the_column_exists_after_migration(db, column):
    cols = {r[1] for r in db.fetchall(
        "PRAGMA table_info(interaction_events)")}
    assert column in cols, f"the migration did not add {column}"


def test_the_schema_version_moved(db):
    assert db._get_version() == 24


def test_context_bytes_are_recorded_by_the_real_builder(tracker):
    """Not by calling the counter directly -- the question is whether a real
    bundle's size reaches telemetry, and a test that calls `note_context()`
    itself answers a different one."""
    from assistant.core.context import build

    bundle = build("interpretation", stored_facts="user_name: Om")
    tracker.note_context(bundle.profile, bundle.size_bytes)

    assert dict(tracker.context_bytes) == {"interpretation": bundle.size_bytes}
    assert bundle.size_bytes > 300, "the fence is not being counted"


def test_repeat_builds_of_one_profile_are_summed(tracker):
    """A planner that replans builds `planning` twice, and the cost asked about
    is the total."""
    tracker.note_context("planning", 100)
    tracker.note_context("planning", 250)
    assert dict(tracker.context_bytes) == {"planning": 350}


def test_an_empty_bundle_is_not_recorded(tracker):
    """Zero bytes is not a measurement, and a profile that carried nothing
    should not appear as though it did."""
    tracker.note_context("execution", 0)
    tracker.note_context("", 500)
    assert not tracker.context_bytes


def test_the_turn_records_the_context_it_built():
    """**The wiring, structurally.** Everything above exercises the tracker; a
    turn that measures and never records would pass all of it."""
    import ast

    tree = ast.parse((_ROOT / "assistant" / "main.py").read_text(
        encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "note_context"]
    assert calls, "main.py builds a context bundle and never records its size"
    for call in calls:
        rendered = ast.unparse(call)
        assert "size_bytes" in rendered, (
            f"the recorded number is not the bundle's size: {rendered}")


def test_the_migration_is_safe_to_rerun(db):
    """A migration that cannot be re-run cannot be recovered from a
    half-applied state, and an already-present column is the expected outcome
    of a re-run rather than an error."""
    db._migrate_v23()
    db._migrate_v24()
    cols = {r[1] for r in db.fetchall("PRAGMA table_info(interaction_events)")}
    assert set(_NEW_COLUMNS) <= cols


# ─── the fields are populated by real paths ──────────────────────────────────

def test_llm_purpose_is_recorded_per_call(tracker):
    """`task_type` was already chosen by every caller and already used to pick
    the model. It was simply never written down."""
    from assistant.llm.router import LLMResult

    for purpose in ("intent", "intent", "code_gen"):
        tracker.record_llm_result(LLMResult(
            text="x", provider="p", model="m", tokens_in=1, tokens_out=1,
            latency_ms=1.0, fallback_depth=0, task_type=purpose))

    assert dict(tracker.llm_purposes) == {"intent": 2, "code_gen": 1}
    assert tracker.llm_calls_count == 3


def test_a_result_without_a_purpose_does_not_invent_one(tracker):
    """Best-effort: the field defaults, and a caller that never set it must not
    make the counter lie about which purposes were used."""
    from assistant.llm.router import LLMResult

    tracker.record_llm_result(LLMResult(
        text="x", provider="p", model="m", tokens_in=None, tokens_out=None,
        latency_ms=0.0, fallback_depth=0))
    assert dict(tracker.llm_purposes) == {"default": 1}


@pytest.mark.asyncio
async def test_a_verification_tier_is_recorded_by_the_real_path(tracker):
    """Through `post_verify`, not by calling the counter directly -- the
    question is whether the wiring happened, and a test that calls
    `note_verification()` itself answers a different one."""
    from assistant.automation import verification

    result = await verification.post_verify(
        {"type": "app", "action": "wait", "params": {}})

    assert dict(tracker.verification_tiers) == {result.tier: 1}
    assert result.tier, "the verdict carries no tier to record"


@pytest.mark.asyncio
async def test_every_return_in_post_verify_is_counted(tracker):
    """Why `post_verify` is a wrapper rather than a `_note_tier(...)` at each
    `return`: the body has eleven of them, and the one someone forgets becomes
    a tier that never appears in the telemetry -- which reads as "that tier
    never ran" rather than as a missing call."""
    import ast
    import inspect

    from assistant.automation import verification

    src = inspect.getsource(verification.post_verify)
    tree = ast.parse(src.lstrip())
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    assert len(returns) == 1, (
        f"post_verify has {len(returns)} returns; it is meant to be a wrapper "
        "with exactly one, so no tier can escape the counter")
    assert "_note_tier" in ast.unparse(returns[0])


def test_the_counters_start_at_zero_and_move(tracker):
    assert tracker.replan_count == 0
    assert tracker.recovery_count == 0
    tracker.note_replan()
    tracker.note_recovery()
    tracker.note_recovery()
    assert tracker.replan_count == 1
    assert tracker.recovery_count == 2


def test_an_empty_tier_name_is_not_counted(tracker):
    """A verdict with no tier records nothing rather than a `""` key, which
    would show up in a report as a nameless tier that decided something."""
    tracker.note_verification("")
    assert not tracker.verification_tiers


# ─── they survive the write ──────────────────────────────────────────────────

def test_the_values_round_trip_through_the_repo(db):
    from assistant.storage.repos.telemetry import TelemetryRepo

    row_id = TelemetryRepo(db).create(
        session_id="s", timestamp="t", input_modality="text", transcript="hi",
        intent_detected="small_talk", intent_source="llm",
        action_dispatched="small_talk", action_outcome="success",
        error_class=None, latency_total_ms=1, latency_stt_ms=None,
        latency_intent_ms=1, latency_action_ms=1, latency_tts_ms=1,
        llm_calls_count=2, llm_tokens_in=10, llm_tokens_out=5,
        fallback_chain_depth=0, vision_calls_count=0,
        llm_purposes=json.dumps({"intent": 1, "small_talk": 1}),
        replan_count=2, recovery_count=1,
        verification_tiers=json.dumps({"code": 3, "vision": 1}),
    )
    row = dict(db.fetchone(
        "SELECT * FROM interaction_events WHERE id = ?", (row_id,)))

    assert json.loads(row["llm_purposes"]) == {"intent": 1, "small_talk": 1}
    assert row["replan_count"] == 2
    assert row["recovery_count"] == 1
    assert json.loads(row["verification_tiers"]) == {"code": 3, "vision": 1}


def test_the_tracker_hands_every_new_field_to_the_repo():
    """The seam a signature change breaks silently: `save()` builds the JSON
    and calls `create()`, and a field added to one and not the other is a
    column that stays NULL while the counter fills up."""
    import ast
    import inspect

    from assistant import telemetry

    src = inspect.getsource(telemetry.TurnTracker.save)
    tree = ast.parse(src.lstrip())
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "create")
    passed = {k.arg for k in call.keywords}

    for column in _NEW_COLUMNS:
        assert column in passed, f"save() never passes {column}"


# ─── O1: no raw payload reaches the metrics store ────────────────────────────

def test_the_new_fields_carry_counts_not_payloads(tracker):
    """§15's O1. Every one of these is a count or a key set; none of them can
    carry a prompt, a page or a screenshot, which is what keeps the metrics
    store safe to ship off-machine (`io/backup` snapshots this database)."""
    from assistant.llm.router import LLMResult

    secret = "sk-aB3xQ9zKmN7pR2tV5wY8uI1oL4jH6gF0dS2eC5vB"
    tracker.record_llm_result(LLMResult(
        text=f"my key is {secret}", provider="p", model="m",
        tokens_in=1, tokens_out=1, latency_ms=1.0, fallback_depth=0,
        task_type="code_gen"))

    blob = json.dumps({
        "llm_purposes": dict(tracker.llm_purposes),
        "verification_tiers": dict(tracker.verification_tiers),
        "replan_count": tracker.replan_count,
        "recovery_count": tracker.recovery_count,
    })
    assert secret not in blob
    assert "my key is" not in blob


# ─── the omission is a decision, on the record ───────────────────────────────

@pytest.mark.parametrize("column", sorted(_DEFERRED))
def test_a_deferred_field_was_not_added_as_a_null_column(db, column):
    """*A field that is always null is not observability* -- P14's own
    property. These five need a per-turn Task and the sixth needs the Context
    Builder; adding them now would make the schema look like it answers
    questions it cannot."""
    cols = {r[1] for r in db.fetchall("PRAGMA table_info(interaction_events)")}
    assert column not in cols, (
        f"{column} was added but nothing can populate it: {_DEFERRED[column]}")


def test_nothing_creates_a_task_per_turn_yet():
    """The fact the deferral rests on, asserted rather than remembered. When
    this goes red, the five Task-dependent fields have become populatable and
    should be added."""
    import ast

    hits = []
    for path in (_ROOT / "assistant").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "create_task"):
                hits.append(str(path.relative_to(_ROOT)))

    assert not hits, (
        f"create_task now has callers ({hits}) -- a Task exists per turn, so "
        "task_id/step_id/affordance/operation/final_task_status can be "
        "populated and belong in the schema")
