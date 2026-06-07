"""
test_manifest_contracts.py - Unit tests for manifest-based LLM contracts.

Covers the three promoter-cycle wrappers:
  - ask_for_intent_clustering
  - ask_for_trace_diff_verification
  - ask_for_phrase_synthesis

Each contract degrades gracefully on JSON parse failure so a bad LLM
response never blocks the promoter cycle.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from assistant.llm.contracts import (
    ask_for_intent_clustering,
    ask_for_phrase_synthesis,
    ask_for_trace_diff_verification,
)


# ─── Fake LLM plumbing ──────────────────────────────────────────────────────

_captured_calls: list[dict] = []
_canned_responses: dict[str, str] = {}


async def _fake_get_llm_response(
    prompt, system_prompt=None, json_mode=False,
    max_tokens=256, task_type="default", temperature=None,
    messages=None,
):
    _captured_calls.append({
        "prompt": prompt,
        "system_prompt": system_prompt,
        "task_type": task_type,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    text = _canned_responses.get(task_type, "")
    return SimpleNamespace(text=text)


_patch = patch("assistant.llm.contracts.get_llm_response", _fake_get_llm_response)


def _run(coro):
    return asyncio.run(coro)


def setup_function():
    _captured_calls.clear()
    _canned_responses.clear()
    _patch.start()


def teardown_function():
    _patch.stop()


# ─── ask_for_intent_clustering ──────────────────────────────────────────────

def test_intent_clustering_happy_path():
    """Returns parsed cluster list and routes through task_type='intent'."""
    canned = [
        {
            "intent_id": "play_song",
            "members": ["play_track_a", "play_track_b"],
            "phrases": ["play the song", "start the music", "hit play"],
            "confidence": "high",
        }
    ]
    _canned_responses["intent"] = json.dumps(canned)

    result = _run(ask_for_intent_clustering(
        app="media_player",
        goals=["play_track_a", "play_track_b"],
    ))

    assert result == canned
    assert _captured_calls[-1]["task_type"] == "intent"
    assert "media_player" in _captured_calls[-1]["prompt"]


def test_intent_clustering_parse_failure_returns_empty():
    """Bad JSON degrades to [] without raising."""
    _canned_responses["intent"] = "not even close to JSON {{{"

    result = _run(ask_for_intent_clustering(
        app="test_app",
        goals=["g1", "g2"],
    ))

    assert result == []


# ─── ask_for_trace_diff_verification ────────────────────────────────────────

def test_trace_diff_verification_happy_path():
    """Returns parsed dict and routes through task_type='agent_verify'."""
    canned = {
        "primary_primitive": "click_button",
        "alternatives": ["keyboard_shortcut"],
        "confidence": "high",
        "diff_notes": "both traces hit the same button via different selectors",
    }
    _canned_responses["agent_verify"] = json.dumps(canned)

    traces = [
        [{"action": "click", "target": "play"}],
        [{"action": "click", "target": "play_btn"}],
    ]
    result = _run(ask_for_trace_diff_verification(traces=traces))

    assert result == canned
    assert _captured_calls[-1]["task_type"] == "agent_verify"


def test_trace_diff_verification_parse_failure_returns_low_confidence():
    """Bad JSON degrades to low-confidence sentinel without raising."""
    _canned_responses["agent_verify"] = "totally not JSON"

    result = _run(ask_for_trace_diff_verification(traces=[[{"a": 1}]]))

    assert result["primary_primitive"] is None
    assert result["alternatives"] == []
    assert result["confidence"] == "low"
    assert result["diff_notes"] == "parse failure"


# ─── ask_for_phrase_synthesis ───────────────────────────────────────────────

def test_phrase_synthesis_happy_path():
    """Returns paraphrase list and routes through task_type='default'."""
    canned = [
        "kick off the playlist",
        "fire up the tunes",
        "get the music going",
    ]
    _canned_responses["default"] = json.dumps(canned)

    result = _run(ask_for_phrase_synthesis(
        intent_id="play_music",
        originals=["play music", "start music"],
    ))

    assert result == canned
    assert _captured_calls[-1]["task_type"] == "default"
    assert "play_music" in _captured_calls[-1]["prompt"]


def test_phrase_synthesis_parse_failure_returns_empty():
    """Bad JSON degrades to [] without raising."""
    _canned_responses["default"] = "garbage non-JSON response"

    result = _run(ask_for_phrase_synthesis(
        intent_id="play_music",
        originals=["play music"],
    ))

    assert result == []


def test_phrase_synthesis_filters_duplicates_of_originals():
    """Phrases that duplicate an original (case-insensitive) are dropped."""
    canned = ["Play Music", "kick off the tunes", "hit play"]
    _canned_responses["default"] = json.dumps(canned)

    result = _run(ask_for_phrase_synthesis(
        intent_id="play_music",
        originals=["play music"],
    ))

    assert "Play Music" not in result
    assert "kick off the tunes" in result
    assert "hit play" in result


# ─── ask_for_vision_ground_coords ──────────────────────────────────────────

_vision_canned_text: list[str] = []


async def _fake_get_vision_response(
    image_base64, prompt, system_prompt=None, json_mode=False, max_tokens=4096,
):
    text = _vision_canned_text.pop(0) if _vision_canned_text else ""
    return SimpleNamespace(text=text)


_vision_patch = patch(
    "assistant.llm.contracts.get_vision_response", _fake_get_vision_response,
)


def test_vision_ground_coords_parses():
    """Happy path: well-formed JSON → parsed dict with x/y/confidence."""
    from assistant.llm.contracts import ask_for_vision_ground_coords
    _vision_canned_text.clear()
    _vision_canned_text.append(
        json.dumps({"x": 600, "y": 400, "confidence": 0.95})
    )
    _vision_patch.start()
    try:
        result = _run(ask_for_vision_ground_coords(
            crop_bytes=b"PNG_BYTES", query="play button", crop_origin=(100, 100),
        ))
    finally:
        _vision_patch.stop()
    assert result["x"] == 600
    assert result["y"] == 400
    assert result["confidence"] >= 0.9


def test_vision_ground_coords_degrades_on_bad_json():
    """Malformed response → {"confidence": 0.0} sentinel, no crash."""
    from assistant.llm.contracts import ask_for_vision_ground_coords
    _vision_canned_text.clear()
    _vision_canned_text.append("not json at all")
    _vision_patch.start()
    try:
        result = _run(ask_for_vision_ground_coords(
            crop_bytes=b"x", query="x", crop_origin=(0, 0),
        ))
    finally:
        _vision_patch.stop()
    assert result["confidence"] == 0.0


def test_vision_ground_coords_degrades_on_llm_unavailable():
    """Router returns __LLM_UNAVAILABLE__ sentinel on total provider failure."""
    from assistant.llm.contracts import ask_for_vision_ground_coords
    _vision_canned_text.clear()
    _vision_canned_text.append("__LLM_UNAVAILABLE__")
    _vision_patch.start()
    try:
        result = _run(ask_for_vision_ground_coords(
            crop_bytes=b"x", query="x", crop_origin=(0, 0),
        ))
    finally:
        _vision_patch.stop()
    assert result["confidence"] == 0.0


def test_vision_ground_coords_coerces_string_numbers():
    """Gemini sometimes returns numeric strings in JSON mode — coerce them."""
    from assistant.llm.contracts import ask_for_vision_ground_coords
    _vision_canned_text.clear()
    _vision_canned_text.append(
        json.dumps({"x": "600", "y": "400", "confidence": "0.95"})
    )
    _vision_patch.start()
    try:
        result = _run(ask_for_vision_ground_coords(
            crop_bytes=b"x", query="x", crop_origin=(0, 0),
        ))
    finally:
        _vision_patch.stop()
    assert result["x"] == 600 and isinstance(result["x"], int)
    assert result["y"] == 400 and isinstance(result["y"], int)
    assert result["confidence"] == 0.95 and isinstance(result["confidence"], float)


def test_vision_ground_coords_falls_back_on_uncoercible_values():
    """Garbage in x/y/confidence falls back to {"confidence": 0.0}."""
    from assistant.llm.contracts import ask_for_vision_ground_coords
    _vision_canned_text.clear()
    _vision_canned_text.append(
        json.dumps({"x": "abc", "y": 400, "confidence": 0.9})
    )
    _vision_patch.start()
    try:
        result = _run(ask_for_vision_ground_coords(
            crop_bytes=b"x", query="x", crop_origin=(0, 0),
        ))
    finally:
        _vision_patch.stop()
    assert result["confidence"] == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
