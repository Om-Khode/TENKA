"""After a personality switch, history carries topics but no voice to copy.

The live defect, one turn after `/personality warm_honest`:

    User:  what is your faviorite color?
    TENKA: (responded)

...spoken aloud by the TTS.

`_build_conversation_messages` masks assistant turns from before the switch, so
the new personality does not imitate the old one's tone. It masked them by
appending an assistant message whose content was the literal string
`"(responded)"` -- and immediately after a switch **every** turn is pre-switch,
so the only assistant behaviour anywhere in the history was that one string,
repeated ten times. The model did the obvious thing.

The mistake was putting a marker in the role reserved for things she actually
said. Anything plausible-looking there is an example to imitate, and a masked
turn is precisely the case where there is no example to give. The fix omits the
message instead: same signal to the model, no words handed over.

Both directions matter here. Dropping too much would lose the topical continuity
that makes "open it again" resolvable, which is why the *user* side of a
pre-switch turn is kept.

Run with:  py -3.11 -m pytest tests/test_switched_personality_history.py -v
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SWITCH_AT = "2026-08-23T18:55:49"


def _turn(n, ts, user, response):
    return {"id": n, "user_input": user, "response": response, "timestamp": ts}


@pytest.fixture()
def build(monkeypatch):
    """Drive the real `_build_conversation_messages` over supplied turns."""
    import assistant.main as main_mod
    import assistant.memory as memory_mod

    def _run(turns, switch_ts):
        monkeypatch.setattr(memory_mod, "get_recent", lambda *a, **kw: turns)
        monkeypatch.setattr(main_mod, "_get_personality_switch_ts",
                            lambda: switch_ts)
        import asyncio
        return asyncio.run(main_mod._build_conversation_messages())

    return _run


# ─── the defect ──────────────────────────────────────────────────────────────

def test_no_assistant_message_is_a_placeholder(build):
    """Every turn pre-switch -- the exact situation one turn after
    `/personality`. There must be nothing in an assistant slot for the model to
    copy."""
    turns = [
        _turn(1, "2026-08-23T18:39:00", "hi", "Hey. What do you need?"),
        _turn(2, "2026-08-23T18:40:00", "who are you", "I'm Tenka."),
    ]
    messages, _ = build(turns, _SWITCH_AT)

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert not assistant_msgs, (
        f"pre-switch turns still contribute assistant text: {assistant_msgs}"
    )
    blob = " ".join(m["content"] for m in messages)
    assert "(responded)" not in blob, "the placeholder is back"


def test_the_old_personalitys_words_do_not_survive_the_switch(build):
    """The masking's actual purpose. A tsundere reply reaching a warm_honest
    turn is a voice to imitate, which is what the switch exists to stop."""
    turns = [
        _turn(1, "2026-08-23T18:39:00", "hi", "Tch. What do you want, dummy?"),
    ]
    messages, _ = build(turns, _SWITCH_AT)
    blob = " ".join(m["content"] for m in messages)
    assert "dummy" not in blob and "Tch" not in blob


def test_what_the_user_said_before_the_switch_is_kept(build):
    """**The other direction.** Dropping the whole pair would lose the topic,
    so "open it again" stops resolving. Only the assistant side is masked."""
    turns = [
        _turn(1, "2026-08-23T18:39:00", "open the budget spreadsheet", "Opened."),
    ]
    messages, _ = build(turns, _SWITCH_AT)
    blob = " ".join(m["content"] for m in messages)
    assert "budget spreadsheet" in blob, (
        "the pre-switch topic was discarded along with the voice"
    )


# ─── post-switch turns are untouched ─────────────────────────────────────────

def test_turns_after_the_switch_keep_their_replies(build):
    """A mask that applied to everything would satisfy every assertion above
    while deleting conversational memory outright -- she would answer each turn
    as though it were the first."""
    turns = [
        _turn(1, "2026-08-23T18:39:00", "before", "old voice"),
        _turn(2, "2026-08-23T18:56:00", "after", "This one is mine."),
    ]
    messages, _ = build(turns, _SWITCH_AT)

    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    assert assistant_msgs == ["This one is mine."], (
        f"post-switch replies were masked too: {assistant_msgs}"
    )


def test_with_no_switch_every_reply_is_present(build):
    """`_get_personality_switch_ts` returns None when the personality has never
    been changed, which is the ordinary case."""
    turns = [
        _turn(1, "2026-08-23T18:39:00", "one", "first"),
        _turn(2, "2026-08-23T18:40:00", "two", "second"),
    ]
    messages, _ = build(turns, None)
    assert [m["content"] for m in messages if m["role"] == "assistant"] == [
        "first", "second"]


# ─── roles stay usable ───────────────────────────────────────────────────────

def test_adjacent_user_turns_are_merged(build):
    """Pre-switch turns contribute no assistant message, so several user turns
    end up adjacent. Gemini tolerates that; the OpenAI-shaped fallbacks are
    happier with alternation and there is no reason to depend on the
    difference."""
    turns = [
        _turn(1, "2026-08-23T18:39:00", "first thing", "a"),
        _turn(2, "2026-08-23T18:40:00", "second thing", "b"),
        _turn(3, "2026-08-23T18:41:00", "third thing", "c"),
    ]
    messages, _ = build(turns, _SWITCH_AT)

    roles = [m["role"] for m in messages]
    assert roles == ["user"], f"expected one merged user turn, got {roles}"
    assert all(t in messages[0]["content"]
               for t in ("first thing", "second thing", "third thing")), (
        "merging lost content"
    )


def test_roles_alternate_across_the_switch_boundary(build):
    """The mixed case, which is where a merge bug would show: masked turns
    before, real ones after."""
    turns = [
        _turn(1, "2026-08-23T18:39:00", "old one", "old reply"),
        _turn(2, "2026-08-23T18:40:00", "old two", "old reply"),
        _turn(3, "2026-08-23T18:56:00", "new one", "new reply"),
    ]
    messages, _ = build(turns, _SWITCH_AT)

    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant"], roles
    assert "old one" in messages[0]["content"] and "new one" in messages[0]["content"]
    assert messages[1]["content"] == "new reply"


def test_no_message_is_empty(build):
    """An empty part is rejected outright by some providers, and a merge that
    joined nothing would produce one."""
    turns = [_turn(1, "2026-08-23T18:39:00", "hello", "hi")]
    messages, _ = build(turns, _SWITCH_AT)
    assert messages, "no messages at all"
    for m in messages:
        assert m["content"].strip(), f"empty message: {m}"
