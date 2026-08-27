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
# classifier prompt.
#
# `manifest_dispatch` is the case the hook was left here for: a synthetic
# intent fired only by `regex_router.py`, carrying params that name a manifest
# resolved at runtime. The classifier has no way to invent those, so offering
# it the name buys nothing and costs a wrong route.
#
# `shutdown` is deliberately NOT in here, and it was the second name this test
# reported missing. It is classifier-reachable -- `main.py` has a branch for a
# classified `shutdown` -- and leaving it out of the catalogue did not make the
# utterance unroutable, it made it route somewhere else: the routing
# differential recorded the classifier calling "shut down" and "exit"
# `computer_task`. `shutdown` is in `system_commands.ALLOWED_EXECUTABLES` with
# no banning pattern, so that path exists and its object is the machine rather
# than TENKA. The catalogue row and rule 15 now draw that line explicitly.
_ALIAS_INTENTS: set[str] = {"manifest_dispatch"}


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
    """The prompt is paid for on every classified turn, so it has a budget.

    The old figure was 7500 and the prompt is 10.2k. That gap is not the
    verbosity the message warned about -- it is four milestones of intents
    (`manage_monitor`, `manage_backup`, `store_memory`, `forget_memory`) each
    bringing a catalogue row, a disambiguation rule and a few-shot, on a cap
    set before any of them existed. Re-set with headroom for a few more, and
    kept low enough that a pasted-in section still trips it.
    """
    prompt = config.INTENT_SYSTEM_PROMPT
    assert len(prompt) < 11500, (
        f"Prompt too large: {len(prompt)} chars (limit 11500). "
        f"Did someone add verbose examples or a new section?"
    )


def test_the_catalogue_stays_one_line_per_intent():
    """What the size cap was standing in for, asserted directly.

    A total-character budget cannot tell a new intent from a paragraph pasted
    into an existing row, and the compact pipe table is the part that has to
    stay compact -- it is the section the classifier reads as a list.
    """
    prompt = config.INTENT_SYSTEM_PROMPT
    block = prompt[prompt.index("Intent catalog"):prompt.index("Param rules")]
    rows = [ln for ln in block.splitlines() if "|" in ln and not ln.startswith("Intent")]
    assert rows, "walked nothing -- the catalogue section moved or was renamed"

    expected = set(config.INTENTS) - _ALIAS_INTENTS
    named = {ln.split("|")[0].strip() for ln in rows}
    assert named == expected, (
        f"catalogue rows and INTENTS disagree; "
        f"only in catalogue: {sorted(named - expected)}, "
        f"only in INTENTS: {sorted(expected - named)}"
    )
    for ln in rows:
        assert len(ln) <= 220, f"catalogue row is prose, not a row: {ln[:80]}..."
