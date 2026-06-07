"""
test_intent_prompt.py — Validate INTENT_SYSTEM_PROMPT structure.

Ensures:
  1. Every intent in config.INTENTS appears in the prompt (except aliases).
  2. All few-shot JSON examples are valid and reference known intents.

Run: python -m pytest tests/test_intent_prompt.py -v
"""

import json
import re

from assistant import config


# Intents that are valid in config but intentionally not surfaced to the
# classifier prompt (currently none — kept as a structural hook for future
# aliases / internal-only intents).
_ALIAS_INTENTS: set[str] = set()


def test_all_intents_present_in_prompt():
    """Every intent from config.INTENTS (except aliases) appears in INTENT_SYSTEM_PROMPT."""
    prompt = config.INTENT_SYSTEM_PROMPT
    missing = []
    for intent in config.INTENTS:
        if intent in _ALIAS_INTENTS:
            continue
        if intent not in prompt:
            missing.append(intent)
    assert not missing, f"Intents missing from INTENT_SYSTEM_PROMPT: {missing}"


def test_few_shot_examples_are_valid_json():
    """Every → {...} example in the prompt parses as valid JSON with a known intent."""
    prompt = config.INTENT_SYSTEM_PROMPT
    # Match lines like: "some text" → {"intent":...}
    pattern = re.compile(r'→\s*(\{.*\})\s*$', re.MULTILINE)
    matches = pattern.findall(prompt)
    assert len(matches) >= 10, f"Expected ≥10 few-shot examples, found {len(matches)}"

    known_intents = set(config.INTENTS)
    for raw_json in matches:
        parsed = json.loads(raw_json)
        assert "intent" in parsed, f"Example missing 'intent' key: {raw_json}"
        assert "params" in parsed, f"Example missing 'params' key: {raw_json}"
        assert parsed["intent"] in known_intents, (
            f"Example intent '{parsed['intent']}' not in config.INTENTS"
        )


def test_prompt_starts_with_classifier_instruction():
    """Prompt opens with the output format instruction."""
    prompt = config.INTENT_SYSTEM_PROMPT
    assert prompt.startswith("You are an intent classifier")


def test_prompt_char_size_reasonable():
    """Prompt should stay under 7500 chars (compact pipe-table format)."""
    prompt = config.INTENT_SYSTEM_PROMPT
    assert len(prompt) < 7500, (
        f"Prompt too large: {len(prompt)} chars (limit 7500). "
        f"Did someone add verbose examples or a new section?"
    )
