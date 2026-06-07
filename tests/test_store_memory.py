"""Tests for the store_memory handler — regex key extraction + fact storage."""

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub heavy modules before importing
for mod_name in [
    "faster_whisper", "pyaudio", "sounddevice",
    "sentence_transformers", "faiss",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

import importlib
# Import the module directly to avoid triggering actions/__init__.py
_mod = importlib.import_module("assistant.actions.memory_search")
_IS_PATTERN = _mod._IS_PATTERN


class TestIsPatternExtraction:
    """Test the regex fast-path for 'X is Y' statements."""

    def test_my_birthday_is(self):
        m = _IS_PATTERN.match("my birthday is on 1st Aug")
        assert m is not None
        assert m.group(1).strip().lower() == "birthday"
        assert m.group(2).strip() == "on 1st Aug"

    def test_favorite_pokemon_is(self):
        m = _IS_PATTERN.match("my favorite pokemon is jigglepuff")
        assert m is not None
        assert "pokemon" in m.group(1).strip().lower()
        assert m.group(2).strip() == "jigglepuff"

    def test_favorite_color_is(self):
        m = _IS_PATTERN.match("my favorite color is purple")
        assert m is not None
        assert m.group(2).strip() == "purple"

    def test_name_is(self):
        m = _IS_PATTERN.match("my name is Alex")
        assert m is not None
        assert m.group(1).strip().lower() == "name"
        assert m.group(2).strip() == "Alex"

    def test_dogs_name_is(self):
        m = _IS_PATTERN.match("my dog's name is Bruno")
        assert m is not None
        assert m.group(2).strip() == "Bruno"

    def test_allergic_no_match(self):
        """Statements without 'is/are/was/were' should not match."""
        m = _IS_PATTERN.match("I like biryani")
        assert m is None

    def test_plural_are(self):
        m = _IS_PATTERN.match("my favorite foods are biryani and pizza")
        assert m is not None
        assert m.group(2).strip() == "biryani and pizza"

    def test_past_tense_was(self):
        m = _IS_PATTERN.match("my first car was a Honda")
        assert m is not None
        assert m.group(2).strip() == "a Honda"


class TestKeyGeneration:
    """Verify key sanitization from regex groups."""

    def _make_key(self, raw: str) -> str:
        return re.sub(r"\s+", "_", raw.strip().lower())

    def test_simple_key(self):
        assert self._make_key("birthday") == "birthday"

    def test_multi_word_key(self):
        assert self._make_key("favorite pokemon") == "favorite_pokemon"

    def test_possessive_key(self):
        assert self._make_key("dog's name") == "dog's_name"
