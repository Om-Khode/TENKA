"""The false-claim fix — a turn a security control skipped must not compose a
reply that asserts a state change which did not happen, and that reply must
not become durable session memory either.

Live-test defect this closes (`.superpowers/sdd/2026-08-16-milestone6b-
transports/live-test-session-2026-08-20.log`): a foreign-answer skip on
`pending_destructive` left the state armed, the turn fell through to the
small_talk/unknown branch, and that branch's LLM call — built from
`_build_conversation_messages()`, with no signal that an answer never landed
— invented "the deletion... has been cancelled". The false sentence was then
persisted verbatim as a session snapshot and replayed as fact next session.

Two halves, tested here:

1. `main.py`'s turn loop tracks `_security_skip_this_turn` across BOTH skip
   shapes in the pending-handler chain -- a foreign-owner mismatch and a
   capability shortfall -- and the small_talk/unknown branch short-circuits
   to a fixed, deterministic reply (`_SECURITY_SKIP_FALLBACK`) instead of
   ever building an LLM prompt, when either fires. This is a code branch,
   not a prompt addition: the LLM is never invoked for a flagged turn, so it
   cannot talk past a hint the way a prompt instruction could be ignored.
2. `session.py`'s `_exclude_security_skips()` drops any turn flagged
   `security_skip` (a new v20 column on `conversations`, set by
   `memory.save_turn(..., security_skip=...)`) before it ever reaches
   `ask_for_session_summary` — a deterministic filter, not the model judging
   its own output.

Every reply assertion pins the exact literal `_SECURITY_SKIP_FALLBACK`
rather than a substring like "not cancelled" — the fix guarantees that one
fixed string is used, so equality to it is the strongest and least
wording-dependent pin available: a differently-worded false claim, or a
differently-worded true claim, both fail it as they should.

Run with:  py -3.11 -m pytest tests/test_reply_cannot_contradict_the_machine.py -v
"""
import contextvars
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from assistant.core.capabilities import Capability

_A_DEVICE = "device:phone"


def _remote_grants() -> "frozenset[Capability]":
    """Funnel ceiling: {OBSERVE, RECALL, CHAT_SEND, SCREEN, FILES} — no
    EXECUTE, no SYSTEM_CONTROL. Built from policy.py so this file tracks the
    ceiling if it moves."""
    from assistant.io.api.policy import POLICIES
    return frozenset(POLICIES["funnel"].ceiling)


# ─────────────────────────────────────────────────────────────────────────
# Harness: one real turn through main.py, with a real grant set AND a real
# principal. Only the edges are stubbed — TTS, the bridge, the small-talk
# LLM call, streaming playback, SQLite, telemetry. The dispatch loop, the
# pending chain, the owner/capability checks, and the fallback branch under
# test are the real code.
#
# `llm.stream_for_small_talk` is stubbed to a canary reply ("Sure, all
# handled!") rather than left to reach the network. That is deliberate for
# the mutation check this file's tests document inline: with the fix
# removed, a flagged turn falls through to this stub and the canary shows up
# where the deterministic fallback should have been — a network-independent
# way to prove the branch under test actually fired, rather than to prove
# nothing by way of a swallowed connection error.
# ─────────────────────────────────────────────────────────────────────────

class _FakeBridge:
    def __init__(self):
        self.commands = []

    async def send_command(self, name, **kw):
        self.commands.append((name, kw))


class _FakeTracker:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return None

    def save(self):
        pass


_CANARY_REPLY = "Sure, all handled!"


@pytest.fixture
def turn(monkeypatch):
    from assistant import main as main_mod

    spoken: list = []
    saved: list = []
    tracker_box: dict = {}

    def _make_tracker(**kw):
        t = _FakeTracker(**kw)
        tracker_box["t"] = t
        return t

    monkeypatch.setattr(main_mod._telemetry, "TurnTracker", _make_tracker)
    monkeypatch.setattr(main_mod._telemetry, "check_correction", lambda *a, **k: None)
    monkeypatch.setattr(main_mod._telemetry, "set_current_tracker",
                        lambda t: contextvars.ContextVar("x", default=None).set(None))
    monkeypatch.setattr(main_mod._telemetry, "reset_current_tracker", lambda t: None)

    import assistant.session as session_mod
    monkeypatch.setattr(session_mod, "get_current_session_id", lambda: "probe-session")
    monkeypatch.setattr(session_mod, "record_turn", lambda *a, **k: None)

    def _fake_save_turn(*a, **k):
        saved.append((a, k))
        return 1
    monkeypatch.setattr(main_mod.memory, "save_turn", _fake_save_turn)
    # Reached only if a turn falls all the way through to the ordinary
    # small_talk/unknown branch -- `_build_conversation_messages()` needs it.
    monkeypatch.setattr(main_mod.memory, "get_recent", lambda *a, **k: [])

    async def _speak(text, *a, **k):
        spoken.append(text)
    monkeypatch.setattr(main_mod.tts, "speak", _speak)

    async def _finish(*a, **k):
        return None
    monkeypatch.setattr(main_mod, "_finish_turn", _finish)
    monkeypatch.setattr(main_mod, "_publish_turn_status", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_wake_listener", None)

    async def _chat(*a, **k):
        return _CANARY_REPLY
    monkeypatch.setattr(main_mod.llm, "chat", _chat)

    # The canary for the ordinary (non-skipped) small_talk/unknown path. See
    # the module docstring for why this is a fixed string rather than a real
    # network call.
    async def _canary_stream(*a, **k):
        yield _CANARY_REPLY
    monkeypatch.setattr(main_mod.llm, "stream_for_small_talk",
                        lambda *a, **k: _canary_stream())

    async def _fake_speak_streaming(token_stream, bridge, emotion="neutral"):
        text = "".join([chunk async for chunk in token_stream])
        spoken.append(text)
        return True, text
    import assistant.io.audio.streaming as streaming_mod
    monkeypatch.setattr(streaming_mod, "speak_streaming", _fake_speak_streaming)

    class _Topic:
        def push_turn(self, *a, **k): pass
        def resolve_query(self, q): return q
        def get_topic_hint(self): return None
    monkeypatch.setattr(main_mod, "_get_topic_tracker", lambda: _Topic())

    monkeypatch.setattr(main_mod.procedures, "match_trigger", lambda t: None)
    monkeypatch.setattr(main_mod.procedures, "find_by_name_or_trigger",
                        lambda *a, **k: None)
    monkeypatch.setattr(main_mod.shortcuts, "match_shortcut", lambda t: None)
    monkeypatch.setattr(main_mod.regex_router, "match_procedure_command",
                        lambda t: None)
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)

    class _Run:
        def __init__(self):
            self.spoken = spoken
            self.saved = saved
            self.tracker = None

        async def __call__(self, text, grants, principal,
                           source="studio", bridge=None, intent="unknown"):
            from assistant.intent import IntentResult

            async def _detect(*a, **k):
                return IntentResult(intent=intent, response=_CANARY_REPLY, params={})
            monkeypatch.setattr(main_mod, "detect_intent", _detect)

            bridge = bridge or _FakeBridge()
            await main_mod.process_text_from_queue(
                source, text, bridge, grants=grants, principal=principal)
            self.tracker = tracker_box.get("t")
            return self

    yield _Run()


def _arm(state_name: str, principal):
    from assistant.pending import pending_registry
    state = pending_registry.get(state_name)
    state.set({"op": "delete", "path": "asdjashdk.txt"}, principal=principal)
    return state


def _one_row(spy, state, capability):
    """A one-row `_PENDING_HANDLERS` table pointing at `state`. The real
    table's row shape is what the dispatch loop unpacks, so a change to it
    fails here rather than silently walking the wrong column."""
    return [(spy, "DESTRUCTIVE", "file_task", False, capability, state)]


def _spy():
    answered: list = []

    async def _handler(text):
        answered.append(text)
        return "Deleted."
    return answered, _handler


@pytest.fixture(autouse=True)
def _clean_pending():
    """Pending state is process-global; leftovers from one test must not
    bleed into the next one's owner/capability checks."""
    from assistant.pending import pending_registry

    def _clear_all():
        for name in pending_registry.names():
            pending_registry.get(name).clear()

    _clear_all()
    yield
    _clear_all()


def _saved_response(out) -> str:
    kw = next(k for a, k in out.saved if "response" in k)
    return kw["response"]


def _saved_security_skip(out) -> bool:
    kw = next(k for a, k in out.saved if "security_skip" in k)
    return kw["security_skip"]


# ═════════════════════════════════════════════════════════════════════════
# Half 1 — the reply itself must not claim a state change that did not
# happen, for BOTH skip shapes.
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["unknown", "small_talk"])
async def test_foreign_answer_skip_does_not_claim_a_state_change(
        turn, monkeypatch, intent):
    """KI-13's shape: the operator arms `pending_destructive`, a different
    principal's "cancel" is skipped (not refused), and the reply must not
    assert the delete was cancelled -- it is still armed."""
    from assistant import main as main_mod
    from assistant.actions import LOCAL_PRINCIPAL

    answered, spy = _spy()
    state = _arm("destructive", LOCAL_PRINCIPAL)
    # FILES is in the funnel ceiling, so this isolates the OWNER mismatch --
    # the caller holds the capability and is skipped anyway.
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS",
                        _one_row(spy, state, Capability.FILES))

    out = await turn("cancel", _remote_grants(), _A_DEVICE, intent=intent)

    assert answered == [], "a foreign 'cancel' drove the confirmation"
    assert state.active, "the confirmation was cleared by a skipped turn"
    assert state.payload["path"] == "asdjashdk.txt", (
        "the armed payload changed even though nothing answered it")

    assert _saved_response(out) == main_mod._SECURITY_SKIP_FALLBACK, (
        f"the skipped turn's reply was not the deterministic fallback: "
        f"{_saved_response(out)!r}")
    assert _saved_response(out) != _CANARY_REPLY, (
        "the skipped turn reached the LLM canary instead of short-circuiting")
    assert _saved_security_skip(out) is True, (
        "the turn was not flagged for the session-snapshot backstop")

    # Recorded, not spoken. This line used to assert the opposite --
    # `out.spoken == [_SECURITY_SKIP_FALLBACK]` -- and it was pinning a defect
    # rather than a decision: `source` here is "studio", and the response path
    # asked about the source nowhere, so every remote answer came out of the
    # speakers on the operator's machine. Live testing caught it, this
    # assertion had been holding it in place, and the control for the other
    # direction (a local turn still speaks) is in
    # `tests/test_remote_turns_are_not_spoken.py`, both response paths.
    assert out.spoken == [], (
        f"a foreign caller's skipped turn was spoken aloud: {out.spoken}")
    assert _saved_response(out) == main_mod._SECURITY_SKIP_FALLBACK, (
        "the fallback stopped being recorded -- Studio settles a turn by "
        "re-reading the transcript, so unspoken and unsaved is a lost turn")


@pytest.mark.asyncio
async def test_capability_skip_does_not_claim_a_state_change(turn, monkeypatch):
    """Not two-principal-specific: the SAME principal owns the state and
    sends this turn, but the turn's own grants lack what the row costs (a
    narrow device, or an EXECUTE-gated row met over a transport whose
    ceiling excludes it). Skipped, same as the foreign case, and the reply
    must be equally honest."""
    from assistant import main as main_mod

    answered, spy = _spy()
    # Owned by the SAME principal that is about to answer it.
    state = _arm("destructive", _A_DEVICE)
    # EXECUTE is NOT in the funnel ceiling, so the capability check fires
    # first -- ownership never even gets consulted.
    monkeypatch.setattr(main_mod, "_PENDING_HANDLERS",
                        _one_row(spy, state, Capability.EXECUTE))

    out = await turn("cancel", _remote_grants(), _A_DEVICE, intent="unknown")

    assert answered == [], "a capability-short caller drove the confirmation"
    assert state.active, "the confirmation was cleared by a capability skip"

    assert _saved_response(out) == main_mod._SECURITY_SKIP_FALLBACK, (
        f"the capability-skipped turn's reply was not the deterministic "
        f"fallback: {_saved_response(out)!r}")
    assert _saved_response(out) != _CANARY_REPLY
    assert _saved_security_skip(out) is True


@pytest.mark.asyncio
async def test_the_fallback_does_not_disclose_that_anything_is_pending(turn, monkeypatch):
    """The other property the fallback text has to hold, beyond "not false":
    it must not tell a caller who cannot answer a confirmation that one
    exists -- that is exactly what the pending chain's own silent-skip
    design (main.py's comments on the dispatch loop) is built to avoid."""
    from assistant import main as main_mod

    text = main_mod._SECURITY_SKIP_FALLBACK
    for leaky_word in ("wait", "pending", "someone else", "another device",
                       "cancel", "delet"):
        assert leaky_word not in text.lower(), (
            f"the fallback leaks {leaky_word!r}: {text!r}")


def test_the_fallback_is_safe_to_speak():
    """Same constraints every spoken string in this tree carries: under 120
    characters, no paths, no error codes."""
    from assistant import main as main_mod
    text = main_mod._SECURITY_SKIP_FALLBACK
    assert text and len(text) < 120, (len(text), text)
    assert "\\" not in text and "/" not in text, text
    assert not any(ch.isdigit() for ch in text), text


@pytest.mark.asyncio
async def test_a_normal_turn_still_reaches_the_llm(turn, monkeypatch):
    """The mechanism must not swallow ordinary conversation. Nothing is
    armed, so `_security_skip_this_turn` stays False and the small_talk
    branch takes its usual path -- proven by the canary reply showing up
    rather than the fallback."""
    out = await turn("what is the weather", _remote_grants(), _A_DEVICE,
                     intent="unknown")

    assert _saved_response(out) == _CANARY_REPLY, (
        "a normal turn was diverted to the security-skip fallback")
    assert _saved_security_skip(out) is False


# ═════════════════════════════════════════════════════════════════════════
# Mutation check, actually run (not merely argued) during development: with
# `if _security_skip_this_turn:` in `process_text_from_queue`'s small_talk/
# unknown branch changed to `if False and _security_skip_this_turn:`,
# `test_foreign_answer_skip_does_not_claim_a_state_change` (both intents) and
# `test_capability_skip_does_not_claim_a_state_change` all failed --
# `_saved_response(out)` came back as `_CANARY_REPLY` ("Sure, all handled!")
# instead of `_SECURITY_SKIP_FALLBACK`, because the skipped turn fell
# through to the (stubbed) LLM call exactly as the live-test session showed
# it doing before this fix. The mutation was then reverted and all tests in
# this file passed again. Not left as an automated toggle in this file,
# because flipping production code from within a test is its own hazard --
# this comment is the record of the check having been run, not a substitute
# for it.
# ═════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════
# Half 2 — a flagged turn must not become durable session memory, regardless
# of what its own reply text says (the deterministic backstop).
# ═════════════════════════════════════════════════════════════════════════

class TestSnapshotExcludesSecuritySkips:
    """Exercises `session._exclude_security_skips()` and its two callers
    directly -- no LLM, no main.py turn loop. `ask_for_session_summary` is
    mocked; the property under test is WHICH turns it was called with."""

    def _init_facade(self, tmp_path, monkeypatch):
        import assistant.session as session_mod
        from assistant.storage.db import init_db, _reset_for_testing

        _reset_for_testing()
        session_mod._repo = None
        session_mod._current_session_id = None
        init_db(tmp_path / "test.db")
        session_mod.init_session_db()
        return session_mod

    @pytest.mark.asyncio
    async def test_flagged_turn_excluded_from_the_summarizer_call(
            self, tmp_path, monkeypatch):
        import assistant.session as session_mod
        from unittest.mock import AsyncMock

        session = self._init_facade(tmp_path, monkeypatch)
        session.start_session()
        session.record_turn("file_task")
        session.record_turn("unknown")

        mock_summary = AsyncMock(
            return_value={"task_summary": "Deleted a file", "blocker": None})
        monkeypatch.setattr(
            "assistant.llm.contracts.ask_for_session_summary", mock_summary)

        turns = [
            {"user_input": "delete asdjashdk.txt",
             "response": "Are you sure you want to delete asdjashdk.txt?",
             "security_skip": False},
            {"user_input": "cancel",
             "response": session_mod.__dict__.get(
                 "_SECURITY_SKIP_FALLBACK",
                 "I'm not sure what you're asking. Could you try rephrasing that?"),
             "security_skip": True},
        ]
        await session.save_snapshot(turns)

        assert mock_summary.await_count == 1
        passed_turns = mock_summary.await_args.args[0]
        assert passed_turns == [turns[0]], (
            f"the flagged turn reached the summarizer anyway: {passed_turns}")

    @pytest.mark.asyncio
    async def test_all_turns_flagged_skips_the_snapshot_entirely(
            self, tmp_path, monkeypatch):
        import assistant.session as session_mod
        from unittest.mock import AsyncMock

        session = self._init_facade(tmp_path, monkeypatch)
        session.start_session()
        session.record_turn("unknown")
        session.record_turn("unknown")

        mock_summary = AsyncMock(
            return_value={"task_summary": "should not run", "blocker": None})
        monkeypatch.setattr(
            "assistant.llm.contracts.ask_for_session_summary", mock_summary)

        turns = [
            {"user_input": "cancel", "response": "fallback 1", "security_skip": True},
            {"user_input": "cancel", "response": "fallback 2", "security_skip": True},
        ]
        await session.save_snapshot(turns)

        mock_summary.assert_not_called()
        assert session._repo.get_last_snapshot() is None, (
            "a snapshot was saved even though every turn was security-skipped")

    @pytest.mark.asyncio
    async def test_a_normal_sessions_snapshot_is_unaffected(
            self, tmp_path, monkeypatch):
        """The conservative half of the ask: an ordinary session with no
        flagged turns must summarize exactly as it did before this fix."""
        import assistant.session as session_mod
        from unittest.mock import AsyncMock

        session = self._init_facade(tmp_path, monkeypatch)
        session.start_session()
        session.record_turn("web_search")
        session.record_turn("small_talk")

        mock_summary = AsyncMock(
            return_value={"task_summary": "Searched for GPUs", "blocker": None})
        monkeypatch.setattr(
            "assistant.llm.contracts.ask_for_session_summary", mock_summary)

        turns = [
            {"user_input": "find GPUs", "response": "Here are options"},
            {"user_input": "thanks", "response": "No problem!"},
        ]
        await session.save_snapshot(turns)

        mock_summary.assert_awaited_once_with(turns)
        snap = session._repo.get_last_snapshot()
        assert snap["task_summary"] == "Searched for GPUs"

    @pytest.mark.asyncio
    async def test_crash_recovery_also_excludes_flagged_turns(
            self, tmp_path, monkeypatch):
        """`recover_crashed_session` reads straight from `conversations` via
        raw SQL and does not go through `save_snapshot()` -- a second call
        site, so it needs its own coverage rather than trusting the first."""
        import assistant.session as session_mod
        from unittest.mock import AsyncMock

        session = self._init_facade(tmp_path, monkeypatch)
        session._repo.start_session("crashed-sess")
        session._repo.increment_turn_count("crashed-sess")
        session._repo.increment_turn_count("crashed-sess")

        db = session._repo._db
        db.execute(
            "INSERT INTO conversations "
            "(timestamp, user_input, intent, response, session_id, security_skip) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-08-20T09:35:00", "delete asdjashdk.txt", "file_task",
             "Are you sure?", "crashed-sess", 0),
        )
        db.execute(
            "INSERT INTO conversations "
            "(timestamp, user_input, intent, response, session_id, security_skip) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-08-20T09:35:41", "cancel", "unknown",
             "I'm not sure what you're asking. Could you try rephrasing that?",
             "crashed-sess", 1),
        )
        db.commit()

        mock_summary = AsyncMock(
            return_value={"task_summary": "Was deleting a file", "blocker": None})
        monkeypatch.setattr(
            "assistant.llm.contracts.ask_for_session_summary", mock_summary)

        session.start_session()
        await session.recover_crashed_session()

        assert mock_summary.await_count == 1
        passed_turns = mock_summary.await_args.args[0]
        assert len(passed_turns) == 1, (
            f"the flagged crash-recovery turn reached the summarizer: "
            f"{passed_turns}")
        assert passed_turns[0]["user_input"] == "delete asdjashdk.txt"

    def test_security_skip_defaults_to_false_for_ordinary_saves(self, tmp_path):
        """`memory.save_turn`'s new parameter must not change what an
        ordinary turn persists -- the conservative constraint on this half."""
        from assistant.storage.db import Database
        from assistant.storage.repos.memory import MemoryRepo

        db = Database(tmp_path / "test.db")
        repo = MemoryRepo(db, tmp_path)
        row_id = repo.save_turn("hi", "small_talk", "hello!", "sess-1")
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", (row_id,))
        assert row["security_skip"] == 0
        assert row["user_input"] == "hi"
        assert row["response"] == "hello!"
