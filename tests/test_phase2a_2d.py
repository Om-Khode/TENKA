"""
test_phase2a_2d.py — Verify Phase 2A (history expansion) and Phase 2D
(preference-hint relocation out of user message).

These tests patch memory + the LLM callable to avoid real API calls.
"""

import asyncio
import inspect
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assistant import code_executor, memory
from assistant.code_executor import routing


# ─── Phase 2A — memory.build_recent_context() ─────────────────────────────────


def test_build_recent_context_is_exported():
    assert hasattr(memory, "build_recent_context")
    assert callable(memory.build_recent_context)


# `memory.build_recent_context` used to compose the string itself out of
# `memory.get_recent`, so these tests patched `get_recent` and read the result.
# It now delegates one layer down to `MemoryRepo.build_recent_context`, which
# composes it *and* applies `redact_secrets_strict` on the way out -- so the
# patch stopped intercepting and every one of these died on "init_memory()
# called before init_db()" rather than on an assertion.
#
# Rewritten against the real formatter with a real (temporary) database. No
# patching of the thing under test, and the redaction that arrived with the
# move is covered too, since that is the half the old tests could not see.

@pytest.fixture
def mem_repo(tmp_path):
    from assistant.storage.db import Database, _reset_for_testing
    from assistant.storage.repos.memory import MemoryRepo

    repo = MemoryRepo(Database(tmp_path / "m.db"), tmp_path)
    yield repo
    _reset_for_testing()


def test_build_recent_context_formats_turns(mem_repo):
    mem_repo.save_turn("hello", "small_talk", "[neutral] hi", "s1")
    mem_repo.save_turn("weather?", "web_search", "[happy] 25C", "s1")

    out = mem_repo.build_recent_context(limit=10)
    assert "RECENT CONVERSATION HISTORY:" in out
    assert "User: hello" in out
    assert "Assistant: [neutral] hi" in out
    assert "User: weather?" in out
    assert "Assistant: [happy] 25C" in out


def test_build_recent_context_respects_custom_header(mem_repo):
    mem_repo.save_turn("x", "small_talk", "y", "s1")

    out = mem_repo.build_recent_context(limit=5, header="CUSTOM HEADER:")
    assert out.startswith("CUSTOM HEADER:")
    assert "RECENT CONVERSATION HISTORY:" not in out


def test_build_recent_context_empty_when_no_turns(mem_repo):
    assert mem_repo.build_recent_context(limit=25) == ""


def test_build_recent_context_swallows_errors(mem_repo):
    """A DB error gives an empty string, not a crash in the caller -- this
    string is built for a prompt, and a failed history is not a failed turn."""
    with patch.object(mem_repo, "get_recent", side_effect=RuntimeError("DB down")):
        assert mem_repo.build_recent_context(limit=25) == ""


def test_build_recent_context_passes_limit_through(mem_repo):
    """The caller's limit must reach `get_recent` -- callers pass 8 for
    reference resolution and 25 for conversation context, and the difference
    only means anything if it survives the hop."""
    captured = {}

    def fake_get_recent(n, session_id=""):
        captured["n"] = n
        return []

    with patch.object(mem_repo, "get_recent", side_effect=fake_get_recent):
        mem_repo.build_recent_context(limit=8)
    assert captured["n"] == 8


def test_build_recent_context_redacts_on_the_way_out(mem_repo):
    """**The half that arrived with the move.** This string is an egress
    boundary: three callers put it into a code-generation or plan-generation
    prompt. Write-side redaction (KI-29) only covers rows written since it
    shipped, so anything older, anything its patterns missed, and anything
    restored from an older backup arrives here unscrubbed."""
    # Two things this test had to get right, and the first draft got neither.
    #
    # **A run of 40 identical characters is not redacted.** The bare rule
    # carries an entropy floor -- that floor is what stops it eating long
    # ordinary words out of prose -- so the fixture has to look like a key.
    #
    # **It cannot go in through `save_turn`.** That applies the lenient
    # `redact_secrets` on write, so the row never held the secret and the test
    # passed with the egress redaction deleted: a sibling mechanism refusing
    # the same input. Inserting directly is the only way to produce the row
    # this guard exists for -- one written before KI-29 shipped, or restored
    # from a backup taken before it.
    secret = "sk-" + "aB3xQ9zKmN7pR2tV5wY8uI1oL4jH6gF0dS2eC5vB"
    mem_repo._db.execute(
        "INSERT INTO conversations "
        "(timestamp, user_input, intent, response, session_id, security_skip) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-01-01T00:00:00", "my key is " + secret, "small_talk",
         "noted", "s1", 0),
    )
    mem_repo._db.commit()
    assert secret in mem_repo.get_recent(5)[0]["user_input"], (
        "the fixture never got the unredacted row into the table, so this test "
        "would pass with the egress redaction removed"
    )

    out = mem_repo.build_recent_context(limit=5)
    assert secret not in out, "a stored secret reached a model-bound string"
    assert "my key is" in out, "redaction ate the whole turn"


def test_the_facade_delegates_to_the_repo():
    """`memory.build_recent_context` is a one-line facade, and the tests above
    are only about the app's behaviour if that hop actually happens."""
    src = inspect.getsource(memory.build_recent_context)
    assert "_get_repo().build_recent_context(" in src


# ─── Phase 2A — the conversation window is 25 turns ───────────────


def test_main_conversation_context_uses_limit_25():
    """`_build_conversation_context` became `_build_conversation_messages`: the
    history reaches the model as native multi-turn messages now instead of a
    text blob, so it calls `memory.get_recent` directly. The window is the part
    that was being pinned, and it survived the rewrite."""
    from assistant import main as main_mod

    captured = {}

    def fake_get_recent(n, *a, **kw):
        captured["limit"] = n
        return []

    with patch.object(memory, "get_recent", side_effect=fake_get_recent):
        asyncio.run(main_mod._build_conversation_messages())
    assert captured.get("limit") == 25


# ─── Phase 2D — preference hints in system prompt, not user message ───────────


def test_route_goal_preference_hints_move_to_system_prompt():
    """
    With preference_hints set, the captured call should have the hints in
    system_prompt, and the user message should NOT contain the IMPORTANT
    prefix (the removed hack).
    """
    captured = {}

    async def fake_llm(*args, **kwargs):
        # Accept either positional user message or keyword
        user_msg = args[0] if args else kwargs.get("user_message") or kwargs.get("prompt")
        captured["user_message"] = user_msg
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return '{"tier": 2, "template_slug": "foo", "requires": [], "params": {}}'

    asyncio.run(routing._route_goal(
        goal="play music",
        llm_func=fake_llm,
        preference_hints="music_app: spotify",
    ))

    # Hints live in the system prompt now
    assert "music_app: spotify" in captured["system_prompt"]
    # The user message carries the goal and nothing borrowed from the hints --
    # no IMPORTANT prefix hack. It is no longer *only* the goal: a literal
    # date/time line is prepended, because the model never computes datetimes
    # (`core/datetime_utils.py` does, and passes a string in). The property
    # being pinned is where the hints live, so assert that, not equality with
    # a message shape that has since gained an unrelated line.
    assert "Goal: play music" in captured["user_message"]
    assert "music_app" not in captured["user_message"]
    assert "IMPORTANT" not in captured["user_message"]


def test_route_goal_no_hints_leaves_system_prompt_unchanged():
    """If preference_hints is empty, system_prompt equals the dynamic router prompt."""
    from assistant.code_executor.prompts import get_router_system_prompt
    captured = {}

    async def fake_llm(*args, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return '{"tier": 1, "template_slug": null, "requires": [], "params": {}}'

    asyncio.run(routing._route_goal(
        goal="what time is it",
        llm_func=fake_llm,
        preference_hints="",
    ))

    assert captured["system_prompt"] == get_router_system_prompt()


# ─── Phase 2A — code_executor injects recent context into gen prompt ──────────


def test_code_executor_imports_memory_build_recent_context():
    """Static check: the gen_prompt construction calls memory.build_recent_context."""
    # `inspect.getsource(code_executor)` reads the package's `__init__.py`, and
    # the call moved into `orchestrator.py` when code_executor became a package.
    # The old form was asserting against a file that never had the call.
    from assistant.code_executor import orchestrator

    src = inspect.getsource(orchestrator)
    assert "memory.build_recent_context" in src, (
        "orchestrator.py must call memory.build_recent_context() for gen prompt injection"
    )
    # And specifically with limit=8 (reference-resolution size, not the 25-turn small_talk size)
    assert "limit=8" in src


# ─── Phase 2A — planner injects recent context into plan prompt ───────────────


def test_planner_imports_memory_build_recent_context():
    from assistant.actions.planner import planner
    src = inspect.getsource(planner)
    assert "memory.build_recent_context" in src, (
        "planner.py must call memory.build_recent_context() for plan prompt injection"
    )
    assert "limit=8" in src


# ─── Phase 2B — intent prompt diet ────────────────────────────────────────────


def test_intent_prompt_is_under_diet_target():
    """Roadmap target: ~120 lines. Count should be materially below the old ~240."""
    from assistant import config
    line_count = config.INTENT_SYSTEM_PROMPT.count("\n")
    # Target was ≤120; assert generously at 150 to avoid flaky line-count drift.
    assert line_count < 150, f"Intent prompt too long ({line_count} lines), target <150"
    # Lower bound sanity — if we accidentally gutted it, catch that too.
    assert line_count > 40, f"Intent prompt suspiciously short ({line_count} lines)"


# `test_intent_prompt_covers_all_registered_intents` was here, and it was a
# copy of `test_intent_prompt.py::test_all_intents_present_in_prompt` -- with no
# notion of an intent that is deliberately kept out of the classifier's
# catalogue. That is exactly how it went red: `manifest_dispatch` is synthetic,
# fired only by `regex_router.py`, and the canonical test records that in
# `_ALIAS_INTENTS` while this copy could not. Two tests for one property, with
# different policies, is worse than one. Deleted rather than re-synced; the
# canonical one also checks the catalogue row-by-row, which this never did.


def test_intent_prompt_preserves_critical_routing_rules():
    """Load-bearing disambiguators must survive the diet."""
    from assistant import config
    p = config.INTENT_SYSTEM_PROMPT
    checks = [
        "code_executor",              # the API-first rule
        "computer_task",              # GUI rule
        "find_and_click",             # already-visible rule
        "web_search",                 # current vs stable knowledge
        "browse_url",                 # specific page rule
        "planner",                    # multi-step rule
        # "read_file" / "DEPRECATED" stood here. `read_file` was never an
        # intent -- it was a callout telling the classifier to prefer
        # `file_task` -- and it went out with the prompt diet these tests were
        # written to guard. `file_task` now owns every file operation in one
        # catalogue row, so there is nothing left to deprecate and the two
        # checks were pinning the absence of a rule rather than a rule.
        "exact spoken words",         # param verbatim rule
        "infer",                      # URL inference rule
    ]
    missing = [c for c in checks if c not in p]
    assert not missing, f"Intent prompt lost critical text: {missing}"


# ─── Phase 2C — personality context summary ───────────────────────────────────


def test_personality_context_summary_includes_count_and_snippets():
    from unittest.mock import patch
    from assistant import personality
    from assistant.llm import prompts as _prompts

    fake_turns = [
        {"user_input": "what time is it", "response": "r1"},
        {"user_input": "play music",      "response": "r2"},
        {"user_input": "cancel reminder", "response": "r3"},
    ]
    with patch.object(personality, "get_conversation_count", return_value=17):
        with patch.object(memory, "get_recent", return_value=fake_turns):
            out = _prompts._build_personality_context_summary()

    assert "Relationship Context" in out
    assert "17 conversations" in out
    # Newest-first collection reversed to oldest-first for output → last item is newest
    assert "cancel reminder" in out
    assert "play music" in out


def test_personality_context_summary_empty_when_no_data():
    from unittest.mock import patch
    from assistant import personality
    from assistant.llm import prompts as _prompts

    with patch.object(personality, "get_conversation_count", return_value=0):
        with patch.object(memory, "get_recent", return_value=[]):
            assert _prompts._build_personality_context_summary() == ""


def test_personality_context_summary_truncates_long_utterances():
    from unittest.mock import patch
    from assistant import personality
    from assistant.llm import prompts as _prompts

    long_utt = "this is a really long message with plenty of words that should get chopped early"
    with patch.object(personality, "get_conversation_count", return_value=3):
        with patch.object(memory, "get_recent", return_value=[{"user_input": long_utt, "response": "r"}]):
            out = _prompts._build_personality_context_summary()
    # 8-word cap means the trailing words must NOT appear
    assert "chopped early" not in out
    # but the first few words DO
    assert "this is a really long message" in out


def test_personality_context_summary_dedupes_repeats():
    from unittest.mock import patch
    from assistant import personality
    from assistant.llm import prompts as _prompts

    repeats = [
        {"user_input": "play music", "response": "r"},
        {"user_input": "play music", "response": "r"},
        {"user_input": "play music", "response": "r"},
    ]
    with patch.object(personality, "get_conversation_count", return_value=5):
        with patch.object(memory, "get_recent", return_value=repeats):
            out = _prompts._build_personality_context_summary()
    # Only one occurrence in the snippet list
    assert out.count('"play music"') == 1


def test_personality_context_summary_survives_db_errors():
    from unittest.mock import patch
    from assistant import personality
    from assistant.llm import prompts as _prompts

    with patch.object(personality, "get_conversation_count", side_effect=RuntimeError("boom")):
        with patch.object(memory, "get_recent", side_effect=RuntimeError("boom")):
            # Should swallow and return empty, not propagate
            assert _prompts._build_personality_context_summary() == ""


# ─── Phase 2C — preference block enhancements ─────────────────────────────────


def test_preference_block_uses_humanized_fallback():
    """Unmapped preferences render as natural language, not 'key = value'."""
    from unittest.mock import patch
    from assistant import preferences
    from assistant.llm import prompts as _prompts

    fake_prefs = [
        {"key": "unknown_thing", "value": "special_mode", "confidence": 0.9},
    ]
    with patch.object(preferences, "get_preferences_by_category", return_value=fake_prefs):
        block = _prompts._build_preference_prompt_block()
    assert "unknown_thing = special_mode" not in block, "Old key=value fallback must be gone"
    assert "unknown thing" in block  # humanized key
    assert "special mode" in block   # humanized value


def test_trait_modifiers_enriched_with_concrete_examples():
    """
    Phase 2C roadmap: 18 trait modifier blocks should be richer — more concrete
    behavioral examples, verbal tics, and example phrasings. Assert each block
    is long enough to plausibly contain those cues.
    Modifiers now live in PersonalityLoader, loaded from modifiers.json.
    """
    from assistant.personalities import PersonalityLoader
    # Test all personalities that have modifiers
    for name in PersonalityLoader.BUILTIN:
        loader = PersonalityLoader(name)
        modifiers = loader.get_modifiers()
        if not modifiers:
            continue  # minimal has no modifiers, skip
        skimpy = []
        for trait, tiers in modifiers.items():
            for tier_name, text in tiers.items():
                if len(text) < 280:  # Pre-enrichment blocks were ~150 chars; enriched ~300+
                    skimpy.append(f"{name}/{trait}/{tier_name}={len(text)}c")
        assert not skimpy, f"Trait modifier blocks still too thin: {skimpy}"


def test_trait_modifiers_contain_concrete_phrasings():
    """At least some trait blocks must contain example utterances (quoted phrases)
    so the model has concrete patterns to mimic, not just abstract direction.
    Modifiers now live in PersonalityLoader, loaded from modifiers.json."""
    from assistant.personalities import PersonalityLoader
    # Check personalities that have modifiers
    for name in PersonalityLoader.BUILTIN:
        loader = PersonalityLoader(name)
        modifiers = loader.get_modifiers()
        if not modifiers:
            continue  # minimal has no modifiers, skip
        all_text = "\n".join(
            text
            for tiers in modifiers.values()
            for text in tiers.values()
        )
        # Enriched blocks should contain direct example utterances (quoted)
        quoted_count = all_text.count("'")
        assert quoted_count >= 40, (
            f"Expected >=40 quote chars in {name} trait blocks, got {quoted_count}"
        )


def test_preference_block_uses_updated_verbosity_text():
    """The new mappings should emit the punchier imperative phrasing."""
    from unittest.mock import patch
    from assistant import preferences
    from assistant.llm import prompts as _prompts

    fake_prefs = [
        {"key": "verbosity", "value": "brief", "confidence": 0.9},
    ]
    with patch.object(preferences, "get_preferences_by_category", return_value=fake_prefs):
        block = _prompts._build_preference_prompt_block()
    assert "Don't ramble" in block, "verbosity:brief should use the punchier phrasing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
