"""Tests for the streaming TTS sentence buffer."""

import pytest
from assistant.io.audio.sentence_buffer import SentenceBuffer


class TestSentenceBufferBasic:
    """Core sentence splitting behavior."""

    def test_single_complete_sentence(self):
        buf = SentenceBuffer()
        result = buf.add("Hello there. ")
        assert result == []
        assert buf.flush() == "Hello there."

    def test_two_sentences_in_one_chunk(self):
        buf = SentenceBuffer(min_length=0)
        result = buf.add("First sentence. Second sentence. ")
        assert result == ["First sentence."]
        assert buf.flush() == "Second sentence."

    def test_token_by_token(self):
        buf = SentenceBuffer()
        tokens = ["The ", "quick ", "brown ", "fox. ", "The ", "lazy ", "dog."]
        all_sentences = []
        for token in tokens:
            all_sentences.extend(buf.add(token))
        assert all_sentences == ["The quick brown fox."]
        assert buf.flush() == "The lazy dog."

    def test_three_sentences(self):
        buf = SentenceBuffer(min_length=0)
        sentences = []
        sentences.extend(buf.add("One. Two. Three."))
        assert sentences == ["One.", "Two."]
        assert buf.flush() == "Three."

    def test_flush_empty_buffer(self):
        buf = SentenceBuffer()
        assert buf.flush() is None

    def test_flush_after_flush(self):
        buf = SentenceBuffer()
        buf.add("Hello.")
        buf.flush()
        assert buf.flush() is None


class TestSentenceBufferEdgeCases:
    """Abbreviations, decimals, ellipsis — must NOT split."""

    def test_abbreviation_dr(self):
        buf = SentenceBuffer()
        result = buf.add("Dr. Smith is here. He said hello.")
        assert result == ["Dr. Smith is here."]
        assert buf.flush() == "He said hello."

    def test_abbreviation_us(self):
        buf = SentenceBuffer()
        result = buf.add("The U.S. is large. Very large.")
        assert result == ["The U.S. is large."]
        assert buf.flush() == "Very large."

    def test_decimal_number(self):
        buf = SentenceBuffer()
        result = buf.add("It costs 3.14 dollars. That is cheap.")
        assert result == ["It costs 3.14 dollars."]
        assert buf.flush() == "That is cheap."

    def test_ellipsis(self):
        buf = SentenceBuffer()
        result = buf.add("Well... I guess so. Maybe.")
        assert result == ["Well... I guess so."]
        assert buf.flush() == "Maybe."

    def test_question_and_exclamation(self):
        buf = SentenceBuffer(min_length=0)
        result = buf.add("Really? Yes! Okay.")
        assert len(result) == 2
        assert result[0] == "Really?"
        assert result[1] == "Yes!"
        assert buf.flush() == "Okay."


class TestSentenceBufferMinLength:
    """Short fragments should accumulate with the next sentence."""

    def test_short_fragment_joins_next(self):
        buf = SentenceBuffer(min_length=20)
        result = buf.add("Oh. That is actually a really great point. Thanks.")
        assert len(result) >= 1
        assert result[0].startswith("Oh.")
        assert len(result[0]) >= 20

    def test_min_length_zero_no_joining(self):
        buf = SentenceBuffer(min_length=0)
        result = buf.add("Oh. Sure. Fine. Let me explain something longer.")
        assert "Oh." in result
        assert "Sure." in result
        assert "Fine." in result


class TestSentenceBufferLazyInit:
    """Segmenter is lazy-loaded on first add() call."""

    def test_no_segmenter_before_add(self):
        buf = SentenceBuffer()
        assert buf._segmenter is None

    def test_segmenter_loaded_after_add(self):
        buf = SentenceBuffer()
        buf.add("Hello.")
        assert buf._segmenter is not None
