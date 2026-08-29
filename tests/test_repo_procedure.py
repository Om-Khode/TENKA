"""Tests for storage/repos/procedure.py — ProcedureRepo."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.storage.db import Database, _reset_for_testing
from assistant.storage.repos.procedure import ProcedureRepo


@pytest.fixture
def repo(tmp_path):
    _reset_for_testing()
    db = Database(tmp_path / "test.db")
    yield ProcedureRepo(db, assistant_name_lower="tenka")
    db.close()
    _reset_for_testing()


_STEPS = [{"type": "app", "action": "open", "params": {"name": "notepad"}}]
_TWO_STEPS = [
    {"type": "app", "action": "open", "params": {"name": "notepad"}},
    {"type": "app", "action": "type", "params": {"text": "hello"}},
]


# --- CRUD ---


def test_create_and_get(repo):
    proc_id = repo.create_procedure("open editor", "Open Editor", _STEPS)
    assert proc_id > 0
    result = repo.get_procedure("open editor")
    assert result is not None
    assert result["name"] == "Open Editor"
    assert result["steps"] == _STEPS
    assert result["trigger"] == "open editor"


def test_get_by_id(repo):
    proc_id = repo.create_procedure("test proc", "Test", _STEPS)
    result = repo.get_procedure_by_id(proc_id)
    assert result is not None
    assert result["id"] == proc_id
    assert result["name"] == "Test"


def test_get_nonexistent(repo):
    assert repo.get_procedure("nope") is None
    assert repo.get_procedure_by_id(9999) is None


def test_update_name(repo):
    proc_id = repo.create_procedure("test", "Old Name", _STEPS)
    assert repo.update_procedure(proc_id, name="New Name")
    result = repo.get_procedure_by_id(proc_id)
    assert result["name"] == "New Name"


def test_update_trigger(repo):
    proc_id = repo.create_procedure("old trigger", "Proc", _STEPS)
    assert repo.update_procedure(proc_id, trigger="new trigger")
    assert repo.get_procedure("old trigger") is None
    assert repo.get_procedure("new trigger") is not None


def test_update_trigger_conflict(repo):
    repo.create_procedure("trigger a", "Proc A", _STEPS)
    proc_b = repo.create_procedure("trigger b", "Proc B", _STEPS)
    with pytest.raises(ValueError, match="already used"):
        repo.update_procedure(proc_b, trigger="trigger a")


def test_update_trigger_reserved(repo):
    proc_id = repo.create_procedure("my proc", "Test", _STEPS)
    with pytest.raises(ValueError, match="reserved"):
        repo.update_procedure(proc_id, trigger="help")


def test_soft_delete(repo):
    proc_id = repo.create_procedure("del me", "Delete Me", _STEPS)
    assert repo.delete_procedure(proc_id)
    # Should not appear in enabled-only queries
    assert repo.get_procedure("del me") is None
    # But should still be found by ID
    result = repo.get_procedure_by_id(proc_id)
    assert result is not None
    assert result["enabled"] == 0


def test_delete_nonexistent(repo):
    assert repo.delete_procedure(9999) is False


def test_list(repo):
    repo.create_procedure("aaa", "Proc A", _STEPS)
    repo.create_procedure("bbb", "Proc B", _TWO_STEPS)
    result = repo.list_procedures()
    assert len(result) == 2


def test_list_excludes_deleted(repo):
    proc_id = repo.create_procedure("aaa", "Proc A", _STEPS)
    repo.create_procedure("bbb", "Proc B", _STEPS)
    repo.delete_procedure(proc_id)
    result = repo.list_procedures(enabled_only=True)
    assert len(result) == 1
    assert result[0]["trigger"] == "bbb"


def test_list_includes_deleted(repo):
    proc_id = repo.create_procedure("aaa", "Proc A", _STEPS)
    repo.create_procedure("bbb", "Proc B", _STEPS)
    repo.delete_procedure(proc_id)
    result = repo.list_procedures(enabled_only=False)
    assert len(result) == 2


# --- Validation ---


def test_create_too_short_trigger(repo):
    with pytest.raises(ValueError, match="too short"):
        repo.create_procedure("ab", "Short", _STEPS)


def test_create_reserved_trigger(repo):
    with pytest.raises(ValueError, match="reserved"):
        repo.create_procedure("tenka", "Bad", _STEPS)
    with pytest.raises(ValueError, match="reserved"):
        repo.create_procedure("help", "Bad", _STEPS)


def test_create_empty_steps(repo):
    with pytest.raises(ValueError, match="at least one step"):
        repo.create_procedure("test", "Test", [])


def test_create_too_many_steps(repo):
    big_steps = [{"type": "app", "action": "wait", "params": {"seconds": 1}}] * 21
    with pytest.raises(ValueError, match="maximum"):
        repo.create_procedure("test", "Test", big_steps)


def test_create_duplicate_trigger(repo):
    repo.create_procedure("test proc", "First", _STEPS)
    with pytest.raises(ValueError, match="already exists"):
        repo.create_procedure("test proc", "Second", _STEPS)


# --- Trigger Matching ---


def test_match_exact(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    result = repo.match_trigger("open editor")
    assert result is not None
    assert result["trigger"] == "open editor"


def test_match_filler(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    # "please" should be stripped as filler
    result = repo.match_trigger("please open editor")
    assert result is not None
    assert result["trigger"] == "open editor"


def test_match_prefix(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    result = repo.match_trigger("open editor and do stuff")
    assert result is not None
    assert result["trigger"] == "open editor"


def test_match_no_match(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    assert repo.match_trigger("close everything") is None


def test_match_short_text(repo):
    assert repo.match_trigger("hi") is None
    assert repo.match_trigger("") is None
    assert repo.match_trigger(None) is None


def test_match_contained(repo):
    repo.create_procedure("save file", "Save", _STEPS)
    result = repo.match_trigger("quickly save file now")
    assert result is not None
    assert result["trigger"] == "save file"


# --- Usage Tracking ---


def test_record_usage(repo):
    proc_id = repo.create_procedure("test", "Test", _STEPS)
    repo.record_usage(proc_id)
    repo.record_usage(proc_id)
    result = repo.get_procedure_by_id(proc_id)
    assert result["use_count"] == 2
    assert result["last_used"] is not None


# --- Conflict Check ---


def test_conflict_procedure(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    conflict = repo.check_trigger_conflict("open editor")
    assert conflict is not None
    assert "procedure" in conflict.lower()


def test_conflict_reserved(repo):
    conflict = repo.check_trigger_conflict("help")
    assert conflict is not None
    assert "reserved" in conflict.lower()


def test_conflict_none(repo):
    assert repo.check_trigger_conflict("brand new trigger") is None


def test_conflict_shortcut(repo):
    # Create a shortcut in the same DB
    from datetime import datetime
    now = datetime.now().isoformat()
    repo._db.execute(
        "INSERT INTO user_shortcuts (trigger, intent, params_json, description, "
        "times_used, created_at, updated_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
        ("my shortcut", "planner", "{}", "", now, now),
    )
    repo._db.commit()
    conflict = repo.check_trigger_conflict("my shortcut")
    assert conflict is not None
    assert "shortcut" in conflict.lower()


# --- Find by Name or Trigger ---


def test_find_by_trigger(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    result = repo.find_by_name_or_trigger("open editor")
    assert result is not None
    assert result["trigger"] == "open editor"


def test_find_by_name(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    result = repo.find_by_name_or_trigger("Open Editor")
    assert result is not None
    assert result["name"] == "Open Editor"


def test_find_partial(repo):
    repo.create_procedure("open work setup", "Work Setup", _STEPS)
    result = repo.find_by_name_or_trigger("open work")
    assert result is not None
    assert result["trigger"] == "open work setup"


def test_find_no_match(repo):
    repo.create_procedure("open editor", "Open Editor", _STEPS)
    assert repo.find_by_name_or_trigger("completely different") is None


def test_find_empty(repo):
    assert repo.find_by_name_or_trigger("") is None
    assert repo.find_by_name_or_trigger(None) is None


# --- Static Helpers ---


def test_step_count_warning_none():
    steps = [{"type": "app"}] * 5
    assert ProcedureRepo.step_count_warning(steps) is None


def test_step_count_warning_soft():
    steps = [{"type": "app"}] * 10
    warning = ProcedureRepo.step_count_warning(steps)
    assert warning is not None
    assert "less reliable" in warning


def test_step_count_warning_max():
    steps = [{"type": "app"}] * 20
    warning = ProcedureRepo.step_count_warning(steps)
    assert warning is not None
    assert "maximum" in warning


def test_subsequence_remainder_match():
    result = ProcedureRepo.subsequence_remainder("open my editor", "open the my cool editor")
    assert result == "the cool"


def test_subsequence_remainder_no_match():
    result = ProcedureRepo.subsequence_remainder("open editor", "close everything")
    assert result == "close everything"


def test_subsequence_remainder_single_word():
    # Single-word triggers don't do subsequence matching
    result = ProcedureRepo.subsequence_remainder("open", "open editor")
    assert result == "open editor"


# --- KI-38: a trigger with filler words at either end ---
#
# `match_trigger` normalized the utterance and compared it against the stored
# trigger raw, so `run`, `go`, `please`, `now`, `hey`, `just`, `do it` and the
# assistant's own name each made a trigger permanently unmatchable -- on all
# four tiers, silently, with nothing refusing it at teach time.
#
# Found live: a procedure taught as "run the p13 check" never fired once, and
# the utterance fell through to the classifier, which answered a stored
# keystroke program with generated Python.

# From the repo's own filler list. Enumerated rather than sampled, because the
# defect was uniform across them and one example would not show the fix is too.
_LEADING_FILLERS = ["run", "go", "please", "now", "hey", "hi", "just",
                    "do it", "can you", "could you", "would you", "tenka"]


@pytest.mark.parametrize("filler", _LEADING_FILLERS)
def test_a_trigger_starting_with_any_filler_still_matches(repo, filler):
    trigger = f"{filler} the widget"
    repo.create_procedure(trigger, "Widget", _STEPS)
    m = repo.match_trigger(trigger)
    assert m is not None, f"a procedure taught as {trigger!r} can never be run"
    assert m["name"] == "Widget"


def test_a_trigger_ending_with_a_filler_still_matches(repo):
    # `_normalize` strips at both ends; only the leading case was seen live,
    # so this is the half that was never observed failing.
    repo.create_procedure("open the widget now", "Trailing", _STEPS)
    m = repo.match_trigger("open the widget now")
    assert m is not None
    assert m["name"] == "Trailing"


@pytest.mark.parametrize("said", [
    "daily report", "please daily report", "run daily report",
    "daily report now", "tenka daily report",
])
def test_the_utterance_may_still_carry_fillers_the_trigger_does_not(repo, said):
    # What `_normalize` existed for in the first place, and what the fix must
    # not cost.
    repo.create_procedure("daily report", "Report", _STEPS)
    m = repo.match_trigger(said)
    assert m is not None, f"{said!r} no longer reaches the trigger"
    assert m["name"] == "Report"


def test_an_exact_match_is_still_reported_as_exact(repo):
    # The tier drives `main.py:_weak_trigger_yields`. A strong match that
    # started reporting `contained` would begin yielding to `pre_route`, which
    # is a routing change wearing a matching fix's clothes.
    repo.create_procedure("run the widget", "Widget", _STEPS)
    assert repo.match_trigger("run the widget")["match_tier"] == "exact"


@pytest.mark.parametrize("said", [
    "what is the weather in pune", "please", "run", "open notepad",
])
def test_an_unrelated_utterance_still_matches_nothing(repo, said):
    # The control. Normalizing both sides must not turn matching into
    # something that claims everything.
    repo.create_procedure("run the widget", "Widget", _STEPS)
    assert repo.match_trigger(said) is None


def test_longest_match_is_measured_after_normalization(repo):
    """The sort has to rank by the same string the tiers compare.

    The first draft of this test was a green mutant: reverting the sort to raw
    length left it passing, because its two triggers landed in *different*
    tiers and tier precedence decided the winner regardless of order. Order
    only matters *within* a tier, so both candidates must reach the same one.

    Both of these are `contained` matches on the utterance, and their raw and
    normalized lengths rank in opposite directions:

        "can you run the widget"   raw 22  ->  normalized "the widget"      10
        "widget factory"           raw 14  ->  normalized "widget factory"  14

    Sorted raw, the first wins and the more specific match is discarded.
    """
    repo.create_procedure("can you run the widget", "Vague", _STEPS)
    repo.create_procedure("widget factory", "Specific", _STEPS)

    m = repo.match_trigger("open the widget factory today")
    assert m is not None
    assert m["match_tier"] == "contained", (
        f"both triggers must reach the same tier for order to be under test; "
        f"got {m['match_tier']}")
    assert m["name"] == "Specific", (
        "the shorter normalized trigger won, so the sort is ranking by raw "
        "length while the tiers compare normalized strings")


# --- KI-38: the all-filler case, which cannot occur ---
#
# The KI writeup proposed refusing a trigger that normalizes to the empty
# string, on the grounds that `"" in anything` would make it claim every turn.
# There is no such trigger. `_filler_re` strips a leading filler only when
# whitespace follows it, and a trailing one only when whitespace precedes it,
# so the first and last tokens always survive. This pins that, so nobody adds
# the guard back on the strength of the writeup.


def test_no_all_filler_phrase_normalizes_to_empty(repo):
    import itertools
    fillers = ["please", "now", "tenka", "go", "do it", "run", "hey", "hi",
               "can you", "could you", "would you", "just"]
    empties = [
        " ".join(c)
        for n in (1, 2, 3)
        for c in itertools.product(fillers, repeat=n)
        if not repo._normalize(" ".join(c))
    ]
    assert not empties, (
        f"{len(empties)} all-filler phrases normalize to empty, e.g. "
        f"{empties[:3]}. The contained tier would match them against every "
        f"utterance -- match_trigger now needs the guard KI-38 proposed."
    )
