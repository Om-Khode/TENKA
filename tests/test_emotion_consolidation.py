"""Tests for D3: Emotion map consolidation into config.py.

Verifies:
  - All three canonical maps exist in config
  - VALID_EMOTIONS is derived from EMOTION_VOICE_PROFILES (no drift)
  - Consumer modules use config constants (no local copies)
  - LEGACY_EMOTION_VOICE_MAP entries are valid (voice, speed) tuples
  - UNITY_EXPRESSION_MAP covers all VALID_EMOTIONS
"""

import ast
import inspect


# --- Canonical maps in config ---

def test_valid_emotions_is_frozenset():
    from assistant import config
    assert isinstance(config.VALID_EMOTIONS, frozenset)
    assert len(config.VALID_EMOTIONS) > 0


def test_valid_emotions_matches_voice_profiles():
    from assistant import config
    assert config.VALID_EMOTIONS == frozenset(config.EMOTION_VOICE_PROFILES.keys())


def test_legacy_voice_map_exists():
    from assistant import config
    assert isinstance(config.LEGACY_EMOTION_VOICE_MAP, dict)
    assert len(config.LEGACY_EMOTION_VOICE_MAP) > 0


def test_legacy_voice_map_entries_are_tuples():
    from assistant import config
    for emotion, entry in config.LEGACY_EMOTION_VOICE_MAP.items():
        assert isinstance(entry, tuple), f"{emotion}: expected tuple, got {type(entry)}"
        assert len(entry) == 2, f"{emotion}: expected (voice, speed), got {entry}"
        voice, speed = entry
        assert isinstance(voice, str), f"{emotion}: voice should be str"
        assert isinstance(speed, (int, float)), f"{emotion}: speed should be numeric"


def test_legacy_voice_map_emotions_are_valid():
    from assistant import config
    for emotion in config.LEGACY_EMOTION_VOICE_MAP:
        assert emotion in config.VALID_EMOTIONS, f"legacy map has unknown emotion: {emotion}"


def test_unity_expression_map_exists():
    from assistant import config
    assert isinstance(config.UNITY_EXPRESSION_MAP, dict)
    assert len(config.UNITY_EXPRESSION_MAP) > 0


def test_unity_expression_map_covers_all_emotions():
    from assistant import config
    missing = config.VALID_EMOTIONS - config.UNITY_EXPRESSION_MAP.keys()
    assert not missing, f"UNITY_EXPRESSION_MAP missing emotions: {missing}"


def test_unity_expression_map_values_are_strings():
    from assistant import config
    for emotion, expr in config.UNITY_EXPRESSION_MAP.items():
        assert isinstance(expr, str), f"{emotion}: expression should be str, got {type(expr)}"


# --- No local copies in consumers ---

def test_tts_no_local_emotion_map():
    """tts.py should NOT define its own _LEGACY_EMOTION_VOICE_MAP."""
    from assistant.io.audio import tts
    source = inspect.getsource(tts)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_LEGACY_EMOTION_VOICE_MAP":
                    raise AssertionError("tts.py still defines _LEGACY_EMOTION_VOICE_MAP locally")


def test_llm_no_local_valid_emotions():
    """llm.py should NOT define its own _VALID_EMOTIONS."""
    from assistant import llm
    source = inspect.getsource(llm)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_VALID_EMOTIONS":
                    raise AssertionError("llm.py still defines _VALID_EMOTIONS locally")


def test_main_no_local_expression_map():
    """main.py should use config.UNITY_EXPRESSION_MAP, not an inline dict."""
    from assistant import main
    source = inspect.getsource(main)
    assert "expression_map = {" not in source, \
        "main.py still defines expression_map as an inline dict"
