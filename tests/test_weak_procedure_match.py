"""A one-word trigger must not claim every sentence containing that word.

Found by a live test that was checking something else entirely. A procedure was
taught with the trigger `scratchpad`, and then:

    'schedule the scratchpad procedure every minute'
    [PROCEDURES] Matched trigger='scratchpad' -> id=1 name='A New Thing'
    [PROC] Executing 'A New Thing' (1 steps)

It ran the procedure instead of scheduling it, and the schedule was never
created -- so the thing the live test was actually verifying never ran either.

Two mechanisms combined. `match_trigger`'s `contained` tier accepts a trigger
appearing *anywhere* in the utterance, and `main.py` calls it before shortcuts
and before intent routing so that a taught trigger beats the classifier. Both
are deliberate and neither is wrong alone. Together, a procedure named for a
common word ("notes", "email", "work") shadows a large share of ordinary
speech -- and it is `EXECUTE`-gated, which means nothing at the keyboard, where
every capability is held.

Resolved by match strength, not by a list of verbs meaning something else:
`exact` and `prefix` still outrank everything, and only the weak tiers ask
`pre_route` whether the sentence is a command about durable state.

Run with:  py -3.11 -m pytest tests/test_weak_procedure_match.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant import config, regex_router  # noqa: E402
from assistant.main import _NAMES_AN_OBJECT  # noqa: E402


@pytest.fixture()
def repo(tmp_path):
    from assistant.storage.db import Database
    from assistant.storage.repos.procedure import ProcedureRepo
    db = Database(tmp_path / "p.db")
    try:
        yield ProcedureRepo(db, assistant_name_lower="tenka")
    finally:
        db._conn.close()


def _teach(repo, trigger, name="A New Thing"):
    repo.create_procedure(
        trigger=trigger, name=name,
        steps=[{"type": "app", "action": "open",
                "params": {"name": "notepad"}}])


def _decision(repo, text):
    """What the turn pipeline will do, by calling the code that decides it.

    **Calls `_weak_trigger_yields`, does not re-implement it.** The first
    version of this helper mirrored `main.py`'s two conditions so the test could
    avoid standing up the turn loop -- and three mutations of the real logic
    (never defer, always defer, treat strong matches as weak) all passed,
    because the mirror was what was under test. The decision now lives in one
    function that both the turn loop and this call.
    """
    from assistant.main import _weak_trigger_yields

    match = repo.match_trigger(text)
    if match is None:
        return None
    competing = _weak_trigger_yields(match, text)
    if competing is not None:
        return f"yields:{competing.intent}"
    return f"runs:{match['match_tier']}"


# ─── the tier is reported at all ─────────────────────────────────────────────

@pytest.mark.parametrize("text,tier", [
    # `_normalize` strips leading and trailing filler, so several phrasings
    # that look weak are `exact` once normalised -- measured, not assumed. The
    # first draft of this list guessed "prefix" and "contained" for the middle
    # two and was wrong about both, which is worth keeping visible: the weak
    # tiers are reached less often than they look, and the observed defect
    # still went through one.
    ("scratchpad", "exact"),
    ("please run scratchpad now", "exact"),
    ("scratchpad and then close it", "prefix"),
    ("do the scratchpad thing", "contained"),
    ("schedule the scratchpad procedure every minute", "contained"),
])
def test_match_trigger_names_its_tier(repo, text, tier):
    """A caller cannot weigh a match it cannot measure. The tier used to be
    computed and thrown away."""
    _teach(repo, "scratchpad")
    match = repo.match_trigger(text)
    assert match is not None, f"no match at all for {text!r}"
    assert match["match_tier"] == tier


def test_a_missing_trigger_still_returns_none(repo):
    _teach(repo, "scratchpad")
    assert repo.match_trigger("what is the weather today") is None


# ─── the observed defect ─────────────────────────────────────────────────────

def test_the_reported_failure(repo):
    """Verbatim from the live test. This ran the procedure and never created
    the schedule."""
    _teach(repo, "scratchpad")
    assert _decision(repo, "schedule the scratchpad procedure every minute") == (
        "yields:manage_schedule"
    )


@pytest.mark.parametrize("text", [
    "schedule the scratchpad procedure every minute",
    "cancel the scratchpad schedule",
    "pause the scratchpad schedule",
])
def test_a_command_about_durable_state_wins(repo, text):
    """In each of these the procedure's name is the *object* of the command,
    never the command."""
    _teach(repo, "scratchpad")
    assert _decision(repo, text).startswith("yields:"), (
        f"{text!r} still runs the procedure"
    )


# ─── and the procedure still runs when it should ─────────────────────────────

@pytest.mark.parametrize("text", [
    "scratchpad",
    "run scratchpad",
    "do the scratchpad thing",
    "scratchpad now please",
])
def test_ordinary_invocation_still_runs_the_procedure(repo, text):
    """**The direction that matters more.** A fix that deferred whenever
    `pre_route` had any answer would satisfy every test above while making
    taught procedures unreachable -- and it would look like the classifier
    misbehaving, not like a bug here."""
    _teach(repo, "scratchpad")
    assert _decision(repo, text).startswith("runs:"), (
        f"{text!r} no longer runs the procedure"
    )


def test_an_ambiguous_action_still_goes_to_the_procedure(repo):
    """`pre_route("open scratchpad for me")` answers `computer_task`, and
    yielding there would launch an application called "scratchpad" instead of
    running what someone taught under that name. Genuinely ambiguous, and the
    procedure should win -- which is why `_NAMES_AN_OBJECT` is a short list and
    not "any intent pre_route can name"."""
    _teach(repo, "scratchpad")
    competing = regex_router.pre_route("open scratchpad for me")
    assert competing is not None and competing.intent == "computer_task", (
        "the premise moved; re-check whether this case should still defer"
    )
    assert _decision(repo, "open scratchpad for me").startswith("runs:")


def test_a_strong_match_never_consults_the_router(repo):
    """`exact` and `prefix` outrank the classifier, which is the whole reason
    this block runs before routing. Only the weak tiers defer."""
    _teach(repo, "list my schedules")     # a trigger that IS a routed phrase
    assert _decision(repo, "list my schedules") == "runs:exact", (
        "an exact trigger match yielded to the router; a taught trigger is "
        "meant to beat it"
    )


# ─── the object-naming set is real ───────────────────────────────────────────

def test_every_listed_intent_exists():
    """A renamed intent would leave a dead string here, silently granting
    precedence to nothing and restoring the defect for that one command."""
    unknown = _NAMES_AN_OBJECT - set(config.INTENTS)
    assert not unknown, f"not real intents: {unknown}"


def test_computer_task_is_deliberately_absent():
    """Pinned because adding it is the obvious "improvement" and it breaks
    `open scratchpad`."""
    assert "computer_task" not in _NAMES_AN_OBJECT
    assert "planner" not in _NAMES_AN_OBJECT


def test_the_turn_loop_applies_both_conditions():
    """Source-level, because `_decision` above is a mirror of `main.py` and a
    mirror that drifts tests itself. Both halves have to be present: the tier
    check, and the restriction to object-naming intents."""
    src = (_ROOT / "assistant" / "main.py").read_text(encoding="utf-8")
    block = src[src.index("_proc_match = procedures.match_trigger("):]
    block = block[:block.index("[PROC] Executing")]
    assert "_weak_trigger_yields" in block, (
        "the turn loop no longer consults the decision function, so every test "
        "in this file is measuring something the pipeline does not run"
    )
