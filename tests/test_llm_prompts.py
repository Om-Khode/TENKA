"""Tests for llm/prompts.py — personality and intent prompt builders."""

import os
from unittest.mock import patch

import pytest


def test_build_personality_prompt_returns_string():
    """build_personality_prompt always returns a non-empty string."""
    from assistant.llm.prompts import build_personality_prompt
    with patch("assistant.llm.prompts._get_personality_traits", return_value={}):
        with patch("assistant.llm.prompts._build_personality_context_summary", return_value=""):
            with patch("assistant.llm.prompts._build_preference_prompt_block", return_value=""):
                result = build_personality_prompt()
    assert isinstance(result, str)
    assert len(result) > 100


def test_build_personality_prompt_includes_name():
    """The prompt includes the assistant's display name."""
    from assistant.llm.prompts import build_personality_prompt
    from assistant import config
    with patch("assistant.llm.prompts._get_personality_traits", return_value={}):
        with patch("assistant.llm.prompts._build_personality_context_summary", return_value=""):
            with patch("assistant.llm.prompts._build_preference_prompt_block", return_value=""):
                result = build_personality_prompt()
    assert config.ASSISTANT_NAME_DISPLAY in result


def test_build_personality_prompt_fallback_on_error():
    """If _get_personality_traits raises, returns static base without crashing."""
    from assistant.llm.prompts import build_personality_prompt
    with patch("assistant.llm.prompts._get_personality_traits", side_effect=Exception("DB fail")):
        result = build_personality_prompt()
    assert isinstance(result, str)
    assert len(result) > 100


def test_build_intent_prompt_returns_string():
    """build_intent_prompt always returns a non-empty string."""
    from assistant.llm.prompts import build_intent_prompt
    with patch("assistant.llm.prompts._get_routing_preferences", return_value=[]):
        result = build_intent_prompt()
    assert isinstance(result, str)
    assert "intent classifier" in result.lower()


def test_build_intent_prompt_includes_preferences_when_available():
    """When routing preferences exist, they're appended."""
    from assistant.llm.prompts import build_intent_prompt
    mock_prefs = [
        {"category": "app_routing", "key": "music_app", "value": "spotify", "confidence": 0.9}
    ]
    with patch("assistant.llm.prompts._get_routing_preferences", return_value=mock_prefs):
        result = build_intent_prompt()
    assert "music_app" in result
    assert "spotify" in result


def test_trait_tier_mapping():
    """_get_trait_tier correctly maps float to low/mid/high."""
    from assistant.llm.prompts import _get_trait_tier
    assert _get_trait_tier(0.0) == "low"
    assert _get_trait_tier(0.33) == "low"
    assert _get_trait_tier(0.34) == "mid"
    assert _get_trait_tier(0.5) == "mid"
    assert _get_trait_tier(0.67) == "high"
    assert _get_trait_tier(1.0) == "high"


def test_build_personality_prompt_injects_modifiers():
    """When traits are available, modifiers are injected into the prompt."""
    from assistant.llm.prompts import build_personality_prompt
    mock_traits = {"trust": 0.8, "warmth": 0.2, "sass": 0.5}
    with patch("assistant.llm.prompts._get_personality_traits", return_value=mock_traits):
        with patch("assistant.llm.prompts._build_personality_context_summary", return_value=""):
            with patch("assistant.llm.prompts._build_preference_prompt_block", return_value=""):
                result = build_personality_prompt()
    assert "Current Behavioral State" in result


def test_preference_prompt_block_empty_when_no_prefs():
    """Returns empty string when no preferences qualify."""
    from assistant.llm.prompts import _build_preference_prompt_block
    with patch("assistant.llm.prompts._get_style_preferences", return_value=[]):
        result = _build_preference_prompt_block()
    assert result == ""


def test_personality_context_summary_empty_on_error():
    """Returns empty string when DB access fails."""
    from assistant.llm.prompts import _build_personality_context_summary
    with patch("assistant.llm.prompts._get_conversation_count", side_effect=Exception("fail")):
        result = _build_personality_context_summary()
    assert result == ""


def test_get_system_prompt_returns_personality():
    """get_system_prompt() returns the active personality base (no dynamic content)."""
    from assistant.llm.prompts import get_system_prompt
    from assistant import config
    prompt = get_system_prompt()
    assert config.ASSISTANT_NAME_DISPLAY in prompt
    assert "Current Behavioral State" not in prompt
