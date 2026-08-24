"""A remote turn's answer must not reach the local speaker.

Found in live testing, not by a test. A Studio turn over `tailnet` asked for
`code_executor`, was correctly refused by `actions.execute()`, and the refusal
came out of the speakers on the operator's machine:

    20:59:04 [actions] Refused intent 'code_executor'
    20:59:04 [tts] Speaking: 'This device holds execute; raise it from the
                              keyboard to use it here.'

The property is not new -- the pre-dispatch branches and both pending paths
have guarded `tts.speak` on `_LOCAL_SOURCES` since 6a.5, and
`test_6a5_predispatch_gate.py::test_a_studio_refusal_is_recorded_and_never_spoken`
already states the stake: *a remote device that can make the local speaker talk
on demand has a standing way to interrupt the owner's room.*

What that test could not see is that it exercises `shutdown`, which is refused
**pre-dispatch**. Everything refused by `execute()` instead -- and every
ordinary Studio answer -- lands on the response path further down, which gated
only on `intent != "computer_task"` and asked about the source nowhere. Two
guarded doors and one open one, which is the shape §21.3 keeps warning about:
the check was easy, the enumeration of paths around it was not.

Both directions are asserted for each path. A guard that silences everything
passes the remote half perfectly.

Run with:  py -3.11 -m pytest tests/test_remote_turns_are_not_spoken.py -v
"""
import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tests"))

# The harness runs one real turn through main.py with only the edges stubbed.
from test_6a5_predispatch_gate import (  # noqa: E402,F401
    turn, _local_grants, _remote_grants,
)

_MAIN_PY = _ROOT / "assistant" / "main.py"


@pytest.fixture(autouse=True)
def _no_history(monkeypatch):
    """`small_talk` reaches `_build_conversation_messages`, which asks
    `memory.get_recent` for the last 25 turns and needs a live database. The
    6a.5 harness stubs `save_turn` but not the read side, because nothing that
    used it before got this far down the pipeline."""
    from assistant import main as main_mod
    monkeypatch.setattr(main_mod.memory, "get_recent", lambda *a, **k: [])

    # And no model. `small_talk` streams through `llm.stream_for_small_talk`,
    # which the 6a.5 harness does not stub -- it stubs `llm.chat`, the
    # non-streaming sibling. So these tests were making a live provider call:
    # a real Gemini request when run alone, and a failed Ollama one in
    # company, which is what surfaced it. A unit test has no business calling
    # a model, and the answer here is a fixture, not a network.
    async def _fake_stream(*a, **k):
        for chunk in ("[neutral] ", "Hey ", "there."):
            yield chunk

    monkeypatch.setattr(main_mod.llm, "stream_for_small_talk", _fake_stream)


@pytest.fixture
def streamed(monkeypatch):
    """Record `speak_streaming` instead of running it.

    The harness stubs `main_mod.tts.speak`, which is where `out.spoken` comes
    from -- and the streaming path does not use it. `_turn_pipeline` imports
    `speak_streaming` inside the function (`from .io.audio.streaming import
    speak_streaming`), so the name is resolved on the module at call time and
    patching the module attribute is what intercepts it.

    Without this the local control ran **real Kokoro synthesis** in a unit
    test -- 36 seconds of it -- while asserting on a list the streaming path
    never touches. A test that reaches the speaker to check that something
    reached the speaker is measuring the wrong thing twice.
    """
    from assistant.io.audio import streaming as _streaming

    played: list[str] = []

    async def _fake(token_stream, bridge=None, emotion="neutral"):
        parts = []
        async for chunk in token_stream:
            parts.append(chunk)
        text = "".join(parts)
        played.append(text)
        return True, text

    monkeypatch.setattr(_streaming, "speak_streaming", _fake)
    return played


def _detect_as(monkeypatch, intent, **params):
    from assistant import main as main_mod
    from assistant.intent import IntentResult

    async def _detect(*a, **k):
        return IntentResult(intent=intent, response="", params=params)

    monkeypatch.setattr(main_mod, "detect_intent", _detect)
    monkeypatch.setattr(main_mod.regex_router, "pre_route", lambda t: None)


# ─── the non-streaming path (tool results, and every `execute()` refusal) ────

@pytest.mark.asyncio
async def test_a_refusal_from_execute_is_not_spoken_to_a_remote_caller(
        turn, monkeypatch):
    """**The one that was live.** `code_executor` needs EXECUTE, the tailnet
    ceiling omits it, so `execute()` refuses -- past every pre-dispatch guard,
    onto the ordinary response path."""
    _detect_as(monkeypatch, "code_executor", goal="print hello")

    out = await turn("run a python script that prints hello",
                     _remote_grants(), source="studio")

    assert out.spoken == [], (
        f"a remote caller made the local speaker talk: {out.spoken}")
    assert out.saved, (
        "the turn left no transcript record -- Studio settles a turn by "
        "re-reading memory, so a silent turn that saves nothing is a lost one")


@pytest.mark.asyncio
async def test_the_same_refusal_is_still_spoken_to_the_person_here(
        turn, monkeypatch):
    """The control, and the one that matters more. Silencing everything
    satisfies the test above and breaks the assistant."""
    _detect_as(monkeypatch, "code_executor", goal="print hello")

    out = await turn("run a python script that prints hello",
                     _local_grants(), source="chat")

    assert out.spoken, "a local turn stopped reaching the speaker"


# ─── the streaming path (ordinary conversation) ──────────────────────────────

@pytest.mark.asyncio
async def test_an_ordinary_remote_answer_is_not_spoken(
        turn, streamed, monkeypatch):
    """Not only refusals. `small_talk` streams, and a remote "hello" was
    played aloud on the operator's machine the same way."""
    _detect_as(monkeypatch, "small_talk")

    out = await turn("hello", _remote_grants(), source="studio")

    assert streamed == [], (
        f"a remote conversational turn reached the audio pipeline: {streamed}")
    assert out.spoken == [], (
        f"a remote conversational turn was spoken aloud: {out.spoken}")


@pytest.mark.asyncio
async def test_a_remote_answer_is_still_recorded(
        turn, streamed, monkeypatch):
    """**The half a naive guard breaks.** `speak_streaming` both synthesises
    the audio *and* assembles the reply text out of the token stream, so
    skipping the call outright silences the turn and loses the answer with it.
    The remote branch has to drain the stream instead."""
    _detect_as(monkeypatch, "small_talk")

    out = await turn("hello", _remote_grants(), source="studio")

    assert out.saved, "the remote turn saved nothing -- the answer was lost"
    responses = [
        (kw.get("response") if kw.get("response") is not None
         else (a[2] if len(a) > 2 else ""))
        for a, kw in out.saved
    ]
    assert any(r and r.strip() for r in responses), (
        f"the remote turn was saved with an empty answer: {responses}. The "
        "stream was skipped rather than drained.")


@pytest.mark.asyncio
async def test_an_ordinary_local_answer_is_still_spoken(
        turn, streamed, monkeypatch):
    """The control for the streaming half. Asserted on `speak_streaming`, not
    on `out.spoken` -- the streaming path never calls `tts.speak`, so the
    obvious assertion here fails for a reason that has nothing to do with the
    guard."""
    _detect_as(monkeypatch, "small_talk")

    await turn("hello", _local_grants(), source="chat")

    assert streamed, "a local conversational turn stopped reaching the speaker"


# ─── structural: no unguarded speak survives on the response path ────────────

def test_no_speak_on_the_response_path_is_unguarded():
    """The enumeration, so the *next* speak site added down there fails here
    rather than in someone's room.

    Walks every `tts.speak` / `speak_streaming` call inside `_turn_pipeline`
    and requires each to sit under a test mentioning `_LOCAL_SOURCES` -- except
    the three error fallbacks, which are named. Those speak a fixed apology on
    a path where the turn has already failed, carry nothing a remote caller
    chose, and are listed rather than exempted silently.
    """
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "_turn_pipeline")

    # Line ranges of every `if`/`elif` whose test names _LOCAL_SOURCES --
    # *both* arms. `if source not in _LOCAL_SOURCES: save ... else: speak` is
    # as guarded as the positive spelling, and counting only the body reported
    # the negated form as a hole.
    guarded_spans = []
    for n in ast.walk(fn):
        if not (isinstance(n, ast.If)
                and "_LOCAL_SOURCES" in ast.unparse(n.test)):
            continue
        for arm in (n.body, n.orelse):
            if arm:
                guarded_spans.append(
                    (arm[0].lineno,
                     max(c.end_lineno or c.lineno for c in arm)))
    assert guarded_spans, "no _LOCAL_SOURCES guard found in the turn pipeline"

    _ALLOWED_UNGUARDED = {
        "Sorry, something went wrong.",
        "I didn't catch that. Could you try again?",
        "Sorry, something went wrong with recording.",
    }

    speaks = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        name = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
        if name not in ("speak", "speak_streaming"):
            continue
        if name == "speak" and getattr(
                getattr(n.func, "value", None), "id", None) != "tts":
            continue
        speaks.append(n)

    assert speaks, "walked nothing -- the pipeline no longer speaks at all"

    unguarded = []
    for call in speaks:
        if any(lo <= call.lineno <= hi for lo, hi in guarded_spans):
            continue
        first = call.args[0] if call.args else None
        literal = first.value if isinstance(first, ast.Constant) else None
        if literal in _ALLOWED_UNGUARDED:
            continue
        unguarded.append((call.lineno, ast.unparse(call)[:70]))

    assert not unguarded, (
        f"speech reachable by a remote caller: {unguarded}. Put it under "
        "`if source in _LOCAL_SOURCES:` -- a device that can make the local "
        "speaker talk on demand can interrupt the owner's room.")
