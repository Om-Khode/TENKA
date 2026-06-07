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


# ─── Phase 2A — memory.build_recent_context() ─────────────────────────────────


def test_build_recent_context_is_exported():
    assert hasattr(memory, "build_recent_context")
    assert callable(memory.build_recent_context)


def test_build_recent_context_formats_turns():
    fake_turns = [
        {"user_input": "hello", "response": "[neutral] hi"},
        {"user_input": "weather?", "response": "[happy] 25C"},
    ]
    with patch.object(memory, "get_recent", return_value=fake_turns):
        out = memory.build_recent_context(limit=10)
    assert "RECENT CONVERSATION HISTORY:" in out
    assert "User: hello" in out
    assert "Assistant: [neutral] hi" in out
    assert "User: weather?" in out
    assert "Assistant: [happy] 25C" in out


def test_build_recent_context_respects_custom_header():
    fake_turns = [{"user_input": "x", "response": "y"}]
    with patch.object(memory, "get_recent", return_value=fake_turns):
        out = memory.build_recent_context(limit=5, header="CUSTOM HEADER:")
    assert out.startswith("CUSTOM HEADER:")
    assert "RECENT CONVERSATION HISTORY:" not in out


def test_build_recent_context_empty_when_no_turns():
    with patch.object(memory, "get_recent", return_value=[]):
        assert memory.build_recent_context(limit=25) == ""


def test_build_recent_context_swallows_errors():
    """A DB error should give an empty string, not crash the caller."""
    with patch.object(memory, "get_recent", side_effect=RuntimeError("DB down")):
        assert memory.build_recent_context(limit=25) == ""


def test_build_recent_context_passes_limit_through():
    """Caller's limit value must reach get_recent()."""
    captured = {}

    def fake_get_recent(n):
        captured["n"] = n
        return []

    with patch.object(memory, "get_recent", side_effect=fake_get_recent):
        memory.build_recent_context(limit=8)
    assert captured["n"] == 8


# ─── Phase 2A — main.py bumped window to 25 ───────────────────────────────────


def test_main_conversation_context_uses_limit_25():
    """main._build_conversation_context should delegate with limit=25."""
    from assistant import main as main_mod

    captured_limit = {}

    def fake_build(limit=25, header="RECENT CONVERSATION HISTORY:"):
        captured_limit["limit"] = limit
        return ""

    with patch.object(memory, "build_recent_context", side_effect=fake_build):
        main_mod._build_conversation_context()
    assert captured_limit.get("limit") == 25


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

    asyncio.run(code_executor._route_goal(
        goal="play music",
        llm_func=fake_llm,
        preference_hints="music_app: spotify",
    ))

    # Hints live in the system prompt now
    assert "music_app: spotify" in captured["system_prompt"]
    # The user message is the naked goal — no IMPORTANT prefix hack
    assert captured["user_message"] == "Goal: play music"
    assert "IMPORTANT" not in captured["user_message"]


def test_route_goal_no_hints_leaves_system_prompt_unchanged():
    """If preference_hints is empty, system_prompt equals the dynamic router prompt."""
    from assistant.code_executor.prompts import get_router_system_prompt
    captured = {}

    async def fake_llm(*args, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return '{"tier": 1, "template_slug": null, "requires": [], "params": {}}'

    asyncio.run(code_executor._route_goal(
        goal="what time is it",
        llm_func=fake_llm,
        preference_hints="",
    ))

    assert captured["system_prompt"] == get_router_system_prompt()


# ─── Phase 2A — code_executor injects recent context into gen prompt ──────────


def test_code_executor_imports_memory_build_recent_context():
    """Static check: the gen_prompt construction calls memory.build_recent_context."""
    src = inspect.getsource(code_executor)
    assert "memory.build_recent_context" in src, (
        "code_executor.py must call memory.build_recent_context() for gen prompt injection"
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


def test_intent_prompt_covers_all_registered_intents():
    """Every intent name in config.INTENTS must appear in the prompt text."""
    from assistant import config
    missing = [name for name in config.INTENTS if name not in config.INTENT_SYSTEM_PROMPT]
    assert not missing, f"Intent prompt missing names: {missing}"


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
        "read_file",                  # deprecation callout
        "DEPRECATED",                 # explicit mark
        "exact spoken words",         # param verbatim rule
        "infer",                      # URL inference rule
    ]
    missing = [c for c in checks if c not in p]
    assert not missing, f"Intent prompt lost critical text: {missing}"


# ─── Phase 2C — personality context summary ───────────────────────────────────


def test_personality_context_summary_includes_count_and_snippets():
    from unittest.mock import patch
    from assistant import config, personality

    fake_turns = [
        {"user_input": "what time is it", "response": "r1"},
        {"user_input": "play music",      "response": "r2"},
        {"user_input": "cancel reminder", "response": "r3"},
    ]
    with patch.object(personality, "get_conversation_count", return_value=17):
        with patch.object(memory, "get_recent", return_value=fake_turns):
            out = config._build_personality_context_summary()

    assert "Relationship Context" in out
    assert "17 conversations" in out
    # Newest-first collection reversed to oldest-first for output → last item is newest
    assert "cancel reminder" in out
    assert "play music" in out


def test_personality_context_summary_empty_when_no_data():
    from unittest.mock import patch
    from assistant import config, personality

    with patch.object(personality, "get_conversation_count", return_value=0):
        with patch.object(memory, "get_recent", return_value=[]):
            assert config._build_personality_context_summary() == ""


def test_personality_context_summary_truncates_long_utterances():
    from unittest.mock import patch
    from assistant import config, personality

    long_utt = "this is a really long message with plenty of words that should get chopped early"
    with patch.object(personality, "get_conversation_count", return_value=3):
        with patch.object(memory, "get_recent", return_value=[{"user_input": long_utt, "response": "r"}]):
            out = config._build_personality_context_summary()
    # 8-word cap means the trailing words must NOT appear
    assert "chopped early" not in out
    # but the first few words DO
    assert "this is a really long message" in out


def test_personality_context_summary_dedupes_repeats():
    from unittest.mock import patch
    from assistant import config, personality

    repeats = [
        {"user_input": "play music", "response": "r"},
        {"user_input": "play music", "response": "r"},
        {"user_input": "play music", "response": "r"},
    ]
    with patch.object(personality, "get_conversation_count", return_value=5):
        with patch.object(memory, "get_recent", return_value=repeats):
            out = config._build_personality_context_summary()
    # Only one occurrence in the snippet list
    assert out.count('"play music"') == 1


def test_personality_context_summary_survives_db_errors():
    from unittest.mock import patch
    from assistant import config, personality

    with patch.object(personality, "get_conversation_count", side_effect=RuntimeError("boom")):
        with patch.object(memory, "get_recent", side_effect=RuntimeError("boom")):
            # Should swallow and return empty, not propagate
            assert config._build_personality_context_summary() == ""


# ─── Phase 2C — preference block enhancements ─────────────────────────────────


def test_preference_block_uses_humanized_fallback():
    """Unmapped preferences render as natural language, not 'key = value'."""
    from unittest.mock import patch
    from assistant import config, preferences

    fake_prefs = [
        {"key": "unknown_thing", "value": "special_mode", "confidence": 0.9},
    ]
    with patch.object(preferences, "get_preferences_by_category", return_value=fake_prefs):
        block = config._build_preference_prompt_block()
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
    from assistant import config, preferences

    fake_prefs = [
        {"key": "verbosity", "value": "brief", "confidence": 0.9},
    ]
    with patch.object(preferences, "get_preferences_by_category", return_value=fake_prefs):
        block = config._build_preference_prompt_block()
    assert "Don't ramble" in block, "verbosity:brief should use the punchier phrasing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
