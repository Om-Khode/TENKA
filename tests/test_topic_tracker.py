"""
test_topic_tracker.py — Unit tests for topic-tracking topic tracking.

Run: python -m pytest tests/test_topic_tracker.py -v
"""

import pytest


@pytest.fixture
def tracker():
    from assistant.topic_tracker import TopicTracker
    return TopicTracker()


class TestStackPush:
    def test_push_extracts_named_entity(self, tracker):
        tracker.push_turn("Tell me about World War 2", turn_number=1)
        hint = tracker.get_topic_hint()
        assert hint is not None
        assert "World War 2" in hint or "World War" in hint

    def test_push_extracts_noun_chunk(self, tracker):
        tracker.push_turn("I want to learn about machine learning", turn_number=1)
        hint = tracker.get_topic_hint()
        assert hint is not None
        assert "machine learning" in hint.lower()

    def test_push_dedup_bumps_to_top(self, tracker):
        tracker.push_turn("Tell me about Python", turn_number=1)
        tracker.push_turn("What about JavaScript", turn_number=2)
        tracker.push_turn("More about Python", turn_number=3)
        hint = tracker.get_topic_hint()
        assert hint is not None
        assert "Python" in hint

    def test_stack_max_three_items(self, tracker):
        tracker.push_turn("Tell me about Python", turn_number=1)
        tracker.push_turn("Tell me about Java", turn_number=2)
        tracker.push_turn("Tell me about Rust", turn_number=3)
        tracker.push_turn("Tell me about Go", turn_number=4)
        assert len(tracker._stack) <= 3

    def test_decay_evicts_old_entries(self, tracker):
        tracker.push_turn("Tell me about Python", turn_number=1)
        tracker.push_turn("something else", turn_number=7)
        for entity_text, _, _ in tracker._stack:
            assert entity_text.lower() != "python"


class TestResolveQuery:
    def test_pronoun_it_resolves_to_stack_top(self, tracker):
        tracker.push_turn("Tell me about World War 2", turn_number=1)
        resolved = tracker.resolve_query("Who won it?")
        assert "World War" in resolved
        assert "it" not in resolved.lower().split()

    def test_pronoun_they_resolves(self, tracker):
        tracker.push_turn("Tell me about the Beatles", turn_number=1)
        resolved = tracker.resolve_query("When did they break up?")
        assert "Beatles" in resolved

    def test_no_pronoun_unchanged(self, tracker):
        tracker.push_turn("Tell me about Python", turn_number=1)
        original = "What is the weather today?"
        resolved = tracker.resolve_query(original)
        assert resolved == original

    def test_pronoun_but_empty_stack_unchanged(self, tracker):
        resolved = tracker.resolve_query("Who won it?")
        assert resolved == "Who won it?"

    def test_self_contained_entity_unchanged(self, tracker):
        tracker.push_turn("Tell me about Python", turn_number=1)
        original = "What is JavaScript?"
        resolved = tracker.resolve_query(original)
        assert resolved == original

    def test_interrogative_what_not_pushed_to_stack(self, tracker):
        tracker.push_turn("what is Python?", turn_number=1)
        resolved = tracker.resolve_query("who created it?")
        assert "Python" in resolved, f"Expected 'Python', got: {resolved}"

    def test_location_there_resolves_despite_noun_chunk(self, tracker):
        tracker.push_turn("search for Tokyo travel guide", turn_number=1)
        resolved = tracker.resolve_query("what's the weather there?")
        assert "Tokyo" in resolved, f"Expected 'Tokyo', got: {resolved}"

    def test_location_here_resolves(self, tracker):
        tracker.push_turn("Tell me about Mumbai", turn_number=1)
        resolved = tracker.resolve_query("what restaurants are here?")
        assert "Mumbai" in resolved, f"Expected 'Mumbai', got: {resolved}"

    def test_imperative_verb_not_pushed_to_stack(self, tracker):
        tracker.push_turn("search for Tokyo travel guide", turn_number=1)
        top = tracker._stack[0][0] if tracker._stack else ""
        assert top.lower() != "search", f"'search' should not be top of stack, got: {tracker._stack}"

    def test_pronoun_resolves_when_unrelated_entity_co_present(self, tracker):
        """C-Q2: stack=[Aanya], 'she likes Figma' — 'she' should still
        resolve to 'Aanya'. The old logic bailed because 'Figma' is detected
        as an entity, suppressing pronoun resolution."""
        tracker.push_turn("Tell me about Aanya", turn_number=1)
        resolved = tracker.resolve_query("she likes Figma")
        assert "Aanya" in resolved, (
            f"Expected 'Aanya' (resolved from 'she'), got: {resolved!r}"
        )
        # The unrelated entity must NOT be replaced
        assert "Figma" in resolved

    def test_pronoun_resolves_when_unrelated_org_co_present(self, tracker):
        """C-Q2 second case: pronoun + org name co-present."""
        tracker.push_turn("Tell me about Priya", turn_number=1)
        resolved = tracker.resolve_query("does she still work at Stripe?")
        assert "Priya" in resolved, (
            f"Expected 'Priya' (resolved from 'she'), got: {resolved!r}"
        )
        assert "Stripe" in resolved


class TestNamedEntityOutranksNounChunk:
    """Session 4 livetest regression: 'My best friend is Aanya' must
    leave 'Aanya' (PERSON) at the top of the stack, not 'My best friend'
    (NOUN_CHUNK). Without this ordering, the next turn's 'she works at X'
    resolves to the generic chunk and the KG extractor emits subjectless
    facts."""

    def test_person_outranks_noun_chunk(self, tracker):
        tracker.push_turn("My best friend is Aanya", turn_number=1)
        resolved = tracker.resolve_query("she works at Razorpay")
        assert "Aanya" in resolved, (
            f"Expected 'Aanya' at stack top, got resolution: {resolved!r}"
        )
        assert "best friend" not in resolved.lower()

    def test_person_outranks_noun_chunk_in_hint(self, tracker):
        tracker.push_turn("My best friend is Aanya", turn_number=1)
        hint = tracker.get_topic_hint()
        assert hint is not None
        assert "Aanya" in hint


class TestAttrChunkOutranksSubject:
    """live-app bug #2: lowercase names like 'sarvesh' don't get the
    PERSON NER tag from spaCy, so the previous fix (named > chunk) didn't
    help. The copula-attribute chunk ('Y' in 'X is Y') is the actual
    definition pattern and must outrank the subject chunk regardless of
    NER status."""

    def test_lowercase_copula_predicate_wins(self, tracker):
        tracker.push_turn("My best friend is sarvesh", turn_number=1)
        resolved = tracker.resolve_query("he works at Stripe")
        assert "sarvesh" in resolved, (
            f"Expected lowercase 'sarvesh' (attr) to win over 'My best "
            f"friend' (nsubj), got: {resolved!r}"
        )

    def test_lowercase_copula_predicate_in_hint(self, tracker):
        tracker.push_turn("My best friend is sarvesh", turn_number=1)
        hint = tracker.get_topic_hint()
        assert hint is not None
        assert "sarvesh" in hint


class TestDemonstrativeNotOverSubstituted:
    """live-app bug #1: 'this/that' as determiners modifying a noun
    ('this week', 'that bet') must NOT be substituted — doing so corrupted
    real turns like 'what did I commit to this week' → 'Friday week'."""

    def test_this_followed_by_noun_left_alone(self, tracker):
        tracker.push_turn("by Friday", turn_number=1)
        resolved = tracker.resolve_query("what did I commit to this week?")
        assert resolved == "what did I commit to this week?", (
            f"'this week' must stay literal, got: {resolved!r}"
        )

    def test_that_followed_by_noun_left_alone(self, tracker):
        tracker.push_turn("this week", turn_number=1)
        resolved = tracker.resolve_query(
            "Aanya owes me a coffee from that bet"
        )
        assert resolved == "Aanya owes me a coffee from that bet", (
            f"'that bet' must stay literal, got: {resolved!r}"
        )

    def test_standalone_this_still_substitutes(self, tracker):
        tracker.push_turn("Tell me about Mars", turn_number=1)
        # 'this' at end of sentence (no following word) is a real pronoun.
        resolved = tracker.resolve_query("tell me more about this")
        assert "Mars" in resolved, (
            f"Standalone 'this' should still resolve, got: {resolved!r}"
        )

    def test_other_pronouns_still_substitute_after_skipped_demonstrative(
        self, tracker,
    ):
        """If 'this week' is skipped at position 0, a later 'he' must
        still be substituted — first ELIGIBLE match, not first match."""
        tracker.push_turn("My friend Karan", turn_number=1)
        resolved = tracker.resolve_query("this week he is busy")
        assert "Karan" in resolved
        assert "this week" in resolved  # demonstrative left intact


class TestGetTopicHint:
    def test_hint_with_stack(self, tracker):
        tracker.push_turn("Tell me about Mars", turn_number=1)
        hint = tracker.get_topic_hint()
        assert hint is not None
        assert hint.startswith("Active topic:")

    def test_hint_empty_stack(self, tracker):
        assert tracker.get_topic_hint() is None
