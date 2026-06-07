"""Tests for manifest-based slot extraction (regex-first, bail-on-fail)."""

from assistant.automation.slot_extraction import (
    extract_slots, SlotExtractionResult,
)


def test_phrase_with_no_slots_succeeds():
    result = extract_slots(utterance="play music", phrase="play music", slot_names=[])
    assert result.ok is True
    assert result.slots == {}


def test_phrase_with_one_slot_extracts():
    result = extract_slots(
        utterance="play blinding lights",
        phrase="play {query}",
        slot_names=["query"],
    )
    assert result.ok is True
    assert result.slots == {"query": "blinding lights"}


def test_bail_when_slot_empty():
    result = extract_slots(
        utterance="play",
        phrase="play {query}",
        slot_names=["query"],
    )
    assert result.ok is False
    assert result.reason


def test_bail_when_phrase_does_not_match():
    result = extract_slots(
        utterance="dance the tango",
        phrase="play {query}",
        slot_names=["query"],
    )
    assert result.ok is False


def test_two_slot_extraction():
    result = extract_slots(
        utterance="copy readme to docs",
        phrase="copy {src} to {dst}",
        slot_names=["src", "dst"],
    )
    assert result.ok is True
    assert result.slots == {"src": "readme", "dst": "docs"}


def test_duplicate_slot_name_bails():
    result = extract_slots(
        utterance="swap a and b",
        phrase="swap {x} and {x}",
        slot_names=["x"],
    )
    assert result.ok is False
    assert "duplicate" in result.reason.lower()
