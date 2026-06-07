"""Tests for S7b — audio modules relocated to assistant/io/audio/."""
import importlib


def test_stt_importable():
    mod = importlib.import_module("assistant.io.audio.stt")
    assert hasattr(mod, "recorder")
    assert hasattr(mod, "transcribe")


def test_tts_importable():
    mod = importlib.import_module("assistant.io.audio.tts")
    assert hasattr(mod, "speak")


def test_speaker_verify_importable():
    mod = importlib.import_module("assistant.io.audio.speaker_verify")
    assert hasattr(mod, "is_enrolled")


def test_named_imports_from_stt():
    from assistant.io.audio.stt import recorder, transcribe, record_until_silence, calibrate_noise_floor
    assert callable(transcribe)
    assert callable(record_until_silence)
    assert callable(calibrate_noise_floor)


def test_config_reachable_from_audio_modules():
    from assistant.io.audio import stt, tts, speaker_verify
    from assistant import config
    assert stt.config is config
    assert tts.config is config
    assert speaker_verify.config is config
