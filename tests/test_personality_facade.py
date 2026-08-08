"""Tests for personality.py facade additions."""


def test_list_personalities_returns_the_builtin_bases():
    from assistant import personality
    from assistant.personalities import PersonalityLoader

    assert personality.list_personalities() == list(PersonalityLoader.BUILTIN)
