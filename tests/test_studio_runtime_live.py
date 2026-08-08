"""LivePersonalityRuntime and LiveChatRuntime against real, temporary SQLite --
the isinstance check in test_studio_runtime.py proves the shape of these
classes but never calls a method on them. These tests call state(),
set_base(), reset(), conversations() and conversation() for real, the same
tmp-dir-SQLite style as tests/test_kg_repo_listing.py.
"""
import pytest

from assistant.actions.studio_runtime import build_studio_runtime
from assistant.storage.db import init_db, _reset_for_testing


class StubDispatch:
    async def submit(self, text: str):
        return ("t1", "c1", True, "")

    async def abort(self) -> bool:
        return True


# ─── LivePersonalityRuntime ─────────────────────────────────────────────────
@pytest.fixture
def personality_db(tmp_path):
    import assistant.personality as personality_facade
    import assistant.personalities as personalities_module

    _reset_for_testing()
    personality_facade._repo = None
    personalities_module._active_loader = None

    init_db(tmp_path / "test.db")
    personality_facade.init_personality_db()

    yield personality_facade, personalities_module

    personality_facade._repo = None
    personalities_module._active_loader = None
    _reset_for_testing()


@pytest.mark.asyncio
async def test_state_reflects_the_real_active_personality(personality_db):
    personality_facade, personalities_module = personality_db
    runtime = build_studio_runtime(StubDispatch())

    state = await runtime.personality.state()

    assert state.base == personalities_module.get_active_personality_id()
    assert "warm_honest" in state.available
    assert "tsundere" in state.available
    assert state.traits, "get_current_traits() returned nothing for a fresh DB"


@pytest.mark.asyncio
async def test_set_base_switches_the_real_active_personality(personality_db):
    personality_facade, personalities_module = personality_db
    runtime = build_studio_runtime(StubDispatch())

    state = await runtime.personality.set_base("tsundere")

    assert state.base == "tsundere"
    assert personalities_module.get_active_personality_id() == "tsundere"


@pytest.mark.asyncio
async def test_set_base_rejects_an_unknown_personality(personality_db):
    runtime = build_studio_runtime(StubDispatch())
    state = await runtime.personality.set_base("does_not_exist")
    # switch_personality() reports the failure as a message rather than
    # raising; the active personality must not have moved.
    assert state.base == "warm_honest"


@pytest.mark.asyncio
async def test_reset_restores_the_pre_bump_traits(personality_db):
    personality_facade, _ = personality_db
    runtime = build_studio_runtime(StubDispatch())

    baseline = (await runtime.personality.state()).traits
    personality_facade.update_traits(
        {"warmth": 0.05}, reason="test bump", trigger="event",
    )
    bumped = (await runtime.personality.state()).traits
    assert bumped["warmth"] != baseline["warmth"], "the bump didn't take"

    reset_state = await runtime.personality.reset()
    assert reset_state.traits["warmth"] == baseline["warmth"]


# ─── LiveChatRuntime ─────────────────────────────────────────────────────────
@pytest.fixture
def memory_db(tmp_path):
    import assistant.memory as memory_facade
    from assistant.storage.repos.memory import MemoryRepo

    _reset_for_testing()
    memory_facade._repo = None

    db = init_db(tmp_path / "test.db")
    # Bind directly rather than through memory.init_memory(): that call
    # also spins up the FAISS vector store, which this runtime's plain SQL
    # reads (get_recent, list_conversation_sessions) never touch.
    memory_facade._repo = MemoryRepo(db, data_dir=tmp_path)

    yield memory_facade

    memory_facade._repo = None
    _reset_for_testing()


@pytest.mark.asyncio
async def test_conversations_lists_real_sessions_not_recording_sessions(memory_db):
    memory_facade = memory_db
    memory_facade.save_turn("hello", "small_talk", "hi", "sess_a")
    memory_facade.save_turn("do thing", "planner", "done", "sess_a")
    memory_facade.save_turn("other", "small_talk", "hey", "sess_b")

    runtime = build_studio_runtime(StubDispatch())
    conversations = await runtime.chat.conversations()

    by_id = {c.conversation_id: c for c in conversations}
    assert by_id["sess_a"].message_count == 2
    assert by_id["sess_b"].message_count == 1
    # Most recently active session first.
    assert conversations[0].conversation_id == "sess_b"


@pytest.mark.asyncio
async def test_conversations_is_empty_with_no_chat_history(memory_db):
    runtime = build_studio_runtime(StubDispatch())
    assert await runtime.chat.conversations() == []


@pytest.mark.asyncio
async def test_conversation_returns_user_and_assistant_messages_in_order(memory_db):
    memory_facade = memory_db
    memory_facade.save_turn("what's on my calendar", "memory_query", "Three things.", "sess_a")

    runtime = build_studio_runtime(StubDispatch())
    detail = await runtime.chat.conversation("sess_a")

    assert detail is not None
    assert [m.role for m in detail.messages] == ["user", "assistant"]
    assert detail.messages[0].text == "what's on my calendar"
    assert detail.messages[1].text == "Three things."


@pytest.mark.asyncio
async def test_conversation_returns_none_for_an_unknown_id(memory_db):
    runtime = build_studio_runtime(StubDispatch())
    assert await runtime.chat.conversation("no-such-session") is None
