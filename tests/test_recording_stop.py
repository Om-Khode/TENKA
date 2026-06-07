"""
test_recording_stop.py — Tests for voice stop-command detection in recording worker.

Run: python -m pytest tests/test_recording_stop.py -v
"""

import sys
import types
import threading
import numpy as np

# ─── Stub heavy deps before importing recording ────────────────────────────

_memory_mod = types.ModuleType("assistant.memory")
_memory_mod.save_chunk = lambda *a, **kw: None
_memory_mod.get_session_transcript = lambda *a, **kw: []
_memory_mod.list_sessions = lambda *a, **kw: []
sys.modules["assistant.memory"] = _memory_mod

import assistant.recording as rec


# ─── Stop phrase detection ─────────────────────────────────────────────────

def test_stop_phrases_recognized():
    """All stop phrases should be in _STOP_PHRASES."""
    for phrase in ("stop recording", "stop the recording",
                   "end recording", "finish recording"):
        assert phrase in rec._STOP_PHRASES


def test_voice_stop_event_initially_clear():
    rec._voice_stop_event.clear()
    assert rec.voice_stop_requested() is False


def test_voice_stop_event_set_and_clear():
    rec._voice_stop_event.set()
    assert rec.voice_stop_requested() is True
    assert rec.voice_stop_requested() is False


def test_flush_chunk_detects_stop_command():
    """_flush_chunk should set the stop event instead of saving when transcript is a stop command."""
    rec._voice_stop_event.clear()
    rec._chunk_index = 0
    saved_chunks = []

    orig_save = _memory_mod.save_chunk
    _memory_mod.save_chunk = lambda sid, idx, text: saved_chunks.append(text)

    try:
        fake_audio = [np.zeros(1600, dtype=np.int16)]
        rec._flush_chunk("test_session", fake_audio, lambda audio: "Stop recording")
        assert rec._voice_stop_event.is_set(), "Stop event should be set"
        assert len(saved_chunks) == 0, "Stop command should NOT be saved as a chunk"
    finally:
        _memory_mod.save_chunk = orig_save
        rec._voice_stop_event.clear()


def test_flush_chunk_detects_stop_with_punctuation():
    """Trailing punctuation should not prevent stop detection."""
    rec._voice_stop_event.clear()
    rec._chunk_index = 0
    saved_chunks = []

    orig_save = _memory_mod.save_chunk
    _memory_mod.save_chunk = lambda sid, idx, text: saved_chunks.append(text)

    try:
        fake_audio = [np.zeros(1600, dtype=np.int16)]
        rec._flush_chunk("test_session", fake_audio, lambda audio: "Stop recording.")
        assert rec._voice_stop_event.is_set()
        assert len(saved_chunks) == 0
    finally:
        _memory_mod.save_chunk = orig_save
        rec._voice_stop_event.clear()


def test_flush_chunk_saves_normal_speech():
    """Normal speech should be saved as a chunk, not trigger stop."""
    rec._voice_stop_event.clear()
    rec._chunk_index = 0
    saved_chunks = []

    orig_save = _memory_mod.save_chunk
    _memory_mod.save_chunk = lambda sid, idx, text: saved_chunks.append(text)

    try:
        fake_audio = [np.zeros(1600, dtype=np.int16)]
        rec._flush_chunk("test_session", fake_audio, lambda audio: "I need to stop recording this later")
        assert not rec._voice_stop_event.is_set()
        assert len(saved_chunks) == 1
        assert "stop recording" in saved_chunks[0].lower()
    finally:
        _memory_mod.save_chunk = orig_save
        rec._voice_stop_event.clear()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
