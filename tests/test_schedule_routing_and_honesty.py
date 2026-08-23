"""Three defects from one live test, all in the same family.

The session that produced them, in order:

    'delete that schedule'          -> forget_memory, "I don't have anything
                                       about that"
    'call it scratchpad'            -> procedure stored under the trigger
                                       'call it scratchpad'
    [scheduler] Procedure not found: scratchpad procedure
    [scheduler] Notified: Task completed

Each is something claiming more than it should:

1. **`_FORGET_MEMORY_RE` claimed a schedule command.** Its leading verb matches
   "delete that schedule", and it was tested before the schedule block, so line
   order decided. `delete scratchpad schedule` worked and `delete that schedule`
   did not -- the only difference being whether the words after the verb
   happened to satisfy the forget pattern's qualifier. The monitor block had
   already been hoisted above the forget pattern for exactly this reason, with a
   comment saying so; the schedule block was left below it.
2. **The teach flow claimed the whole sentence as the trigger.** Asked what to
   call it, "call it scratchpad" became the trigger, so the procedure existed
   under a phrase nobody would ever say.
3. **The scheduler claimed success for work it never did.** Every not-run path
   returned `""`, and `result or "Task completed"` turned that into an
   announcement. It said "Task completed" on the line after "Procedure not
   found", once a minute.

Run with:  py -3.11 -m pytest tests/test_schedule_routing_and_honesty.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant import regex_router  # noqa: E402


# ─── 1. durable-state CRUD outranks the forget pattern ───────────────────────

@pytest.mark.parametrize("text", [
    "delete that schedule",
    "delete the schedule",
    "cancel that schedule",
    "remove that schedule",
    "delete scratchpad schedule",
    "pause that schedule",
])
def test_a_schedule_command_is_not_a_memory_command(text):
    """`delete that schedule` was the observed failure and answered "I don't
    have anything about that"."""
    result = regex_router.pre_route(text)
    assert result is not None, f"{text!r} routes nowhere"
    assert result.intent == "manage_schedule", (
        f"{text!r} routed to {result.intent}"
    )


@pytest.mark.parametrize("text", [
    "delete all monitors",
    "remove the song monitor",
    "show my monitors",
])
def test_the_monitor_half_still_works(text):
    """It was hoisted first, for the same reason. Moving the schedule block up
    beside it must not disturb it."""
    result = regex_router.pre_route(text)
    assert result is not None and result.intent == "manage_monitor"


@pytest.mark.parametrize("text", [
    "forget my birthday",
    "delete that memory",
    "forget what I said about pune",
    "erase that fact",
])
def test_memory_commands_still_reach_forget_memory(text):
    """**The direction the hoist could break.** Moving a block above the forget
    pattern narrows what reaches it, and narrowing too far would silently stop
    her forgetting anything -- which nothing would report."""
    result = regex_router.pre_route(text)
    assert result is not None, f"{text!r} routes nowhere"
    assert result.intent == "forget_memory", (
        f"{text!r} routed to {result.intent}; the hoist took too much"
    )


def test_schedule_creation_survived_the_move():
    """`^schedule ...` moved with its block. It cannot collide with a forget
    pattern, but a block split across two places is how the second half gets
    forgotten -- which is what happened here."""
    result = regex_router.pre_route("schedule a news search every morning")
    assert result is not None and result.intent == "manage_schedule"
    assert result.params.get("action") == "create"


def test_the_crud_block_precedes_the_forget_pattern_in_source():
    """Line order *is* the rule here, so it is asserted directly. A future edit
    that moves either one restores the defect with no test failing on behaviour
    alone if the phrasings happen to differ."""
    src = (_ROOT / "assistant" / "regex_router.py").read_text(encoding="utf-8")
    forget_at = src.index("_FORGET_MEMORY_RE.match(t)")
    for name in ("_SCHEDULE_CANCEL_RE.match", "_MONITOR_CRUD_RE.match"):
        assert src.index(name) < forget_at, (
            f"{name} is tested after the forget pattern, so a command whose "
            f"leading verb both claim goes to memory"
        )


# ─── 2. the trigger is the name, not the sentence ────────────────────────────

def _strip(text):
    """Calls the real derivation, does not re-implement it.

    The first version applied `_NAMING_PREAMBLE_RE` itself and rebuilt the
    surrounding strip-and-fallback here -- so removing the strip from the teach
    flow was a GREEN mutation. Third time in one session that a test mirrored
    the code it was testing.
    """
    from assistant.actions.teaching import _trigger_from_naming_reply
    return _trigger_from_naming_reply(text)


@pytest.mark.parametrize("said,trigger", [
    ("call it scratchpad", "scratchpad"),
    ("call it my morning routine", "my morning routine"),
    ("name it deploy", "deploy"),
    ("the trigger is start coding", "start coding"),
    ("it's called scratchpad", "scratchpad"),
    ("just call it notes", "notes"),
    ("let's call it notes", "notes"),
    ("I will call it notes", "notes"),
])
def test_a_naming_sentence_yields_the_name(said, trigger):
    """The observed case is the first row. It stored a procedure under 'call it
    scratchpad', so saying "scratchpad" ran nothing and a schedule pointing at
    it logged "Procedure not found" every minute."""
    assert _strip(said) == trigger


@pytest.mark.parametrize("said", [
    "scratchpad",
    "start my coding session",
    "call me a taxi",
    "name the thing",
    "call mom every morning",
])
def test_an_ordinary_name_is_left_alone(said):
    """**The direction that matters.** Over-stripping would silently rename
    procedures, and "call me a taxi" is a perfectly good trigger that begins
    with the same verb."""
    assert _strip(said) == said


def test_a_preamble_with_nothing_after_it_keeps_the_original():
    """Stripping to nothing would leave the length guard rejecting a name the
    person did say. Better to keep the raw text and let them see it."""
    assert _strip("call it") == "call it"


# ─── 3. a task that did not run does not say it did ──────────────────────────

def test_a_missing_procedure_reports_did_not_run():
    """Verbatim from the log:

        [scheduler] Procedure not found: scratchpad procedure
        [scheduler] Notified: Task completed

    Two adjacent lines, one of them false.
    """
    from assistant import scheduler

    task = {"id": 1, "name": "t", "notify_mode": "always",
            "condition_text": None, "last_result_hash": None}
    notify, summary = scheduler._should_notify_sync(task, "there was nothing to run")
    assert notify is True
    assert "completed" not in summary.lower(), (
        f"a task that never ran was reported as completed: {summary!r}"
    )
    assert "nothing to run" in summary


def test_an_empty_result_from_a_task_that_did_run_claims_nothing():
    """`result or "Task completed"` was the exact line. A task that genuinely
    ran and produced no text still must not assert that the work is done."""
    from assistant import scheduler

    task = {"id": 1, "name": "t", "notify_mode": "always",
            "condition_text": None, "last_result_hash": None}
    _notify, summary = scheduler._should_notify_sync(task, "")
    assert "completed" not in summary.lower(), (
        f"an empty result still claims completion: {summary!r}"
    )


def test_a_real_result_is_passed_through_untouched():
    """Both directions. A change that reported failure for everything would
    satisfy the two tests above and make every working schedule sound broken."""
    from assistant import scheduler

    task = {"id": 1, "name": "t", "notify_mode": "always",
            "condition_text": None, "last_result_hash": None}
    notify, summary = scheduler._should_notify_sync(task, "Stripe bought OpenRouter.")
    assert notify is True
    assert summary == "Stripe bought OpenRouter."


def test_run_handler_reports_whether_it_ran():
    """Two values, because one empty string cannot distinguish "ran and had
    nothing to say" from "never ran" -- and the second used to be reported as
    the first."""
    import inspect

    from assistant import scheduler
    src = inspect.getsource(scheduler._run_handler)
    assert "return False" in src, "_run_handler cannot report a failure to run"
    assert "return True" in src, "_run_handler cannot report a success"


def test_the_not_run_branch_is_wired_into_the_fire_path():
    """A `(ran, text)` pair nothing unpacks is worse than none: it reads as a
    fix."""
    import inspect

    from assistant import scheduler
    src = inspect.getsource(scheduler._fire_task)
    assert "ran, result = _run_handler(task)" in src, (
        "_fire_task does not read whether the task ran"
    )
    assert "if ran:" in src, "_fire_task ignores it after unpacking"
    assert "DID NOT RUN" in src, (
        "the log line cannot distinguish the two outcomes, which is how this "
        "went unnoticed for a whole session"
    )


def test_the_success_fallback_string_is_gone():
    """Pinned by absence, because re-adding it is a one-word edit that reads as
    a tidy-up."""
    src = (_ROOT / "assistant" / "scheduler.py").read_text(encoding="utf-8")
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    assert 'or "Task completed"' not in code, (
        "the invented success message is back"
    )


# ─── and the scheduler's own not-run mapping, called rather than mirrored ─────

def test_run_handler_maps_none_to_did_not_run(monkeypatch):
    """`_async_run_handler` returns `None` when there is nothing to run -- a
    missing procedure, an unknown task type. Returning `""` there was the
    original defect, and a test that only exercised `_should_notify_sync` with a
    hand-written string never touched this mapping: restoring the `""` was a
    GREEN mutation.
    """
    import asyncio

    from assistant import scheduler

    async def _nothing(task):
        return None

    monkeypatch.setattr(scheduler, "_async_run_handler", _nothing)
    loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(scheduler, "_loop", loop)
        import threading
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        ran, text = scheduler._run_handler({"name": "t"})
    finally:
        loop.call_soon_threadsafe(loop.stop)

    assert ran is False, "a task with nothing to run reported that it ran"
    assert "nothing to run" in text


def test_run_handler_maps_a_string_to_ran(monkeypatch):
    """The other direction, on the same function. A mapping that answered False
    for everything would satisfy the test above and make every working schedule
    report a failure."""
    import asyncio
    import threading

    from assistant import scheduler

    async def _something(task):
        return "Stripe bought OpenRouter."

    monkeypatch.setattr(scheduler, "_async_run_handler", _something)
    loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(scheduler, "_loop", loop)
        threading.Thread(target=loop.run_forever, daemon=True).start()
        ran, text = scheduler._run_handler({"name": "t"})
    finally:
        loop.call_soon_threadsafe(loop.stop)

    assert ran is True
    assert text == "Stripe bought OpenRouter."


def test_a_missing_procedure_returns_none_not_empty_string():
    """Source-level on the branch itself, because reaching it needs a real
    procedure store and a running loop. `""` and `None` are one character apart
    and mean opposite things to `_run_handler`."""
    import inspect

    from assistant import scheduler
    src = inspect.getsource(scheduler._async_run_handler)
    proc_half = src[src.index('task_type == "procedure"'):]
    not_found = proc_half[:proc_half.index("run_procedure")]
    assert "return None" in not_found, (
        f"the not-found branch does not report did-not-run: {not_found}"
    )
    assert 'return ""' not in not_found
