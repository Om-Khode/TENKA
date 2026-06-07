"""Tests for personality.py merge — verify both facade and events are accessible."""

import importlib
from unittest.mock import patch


class TestPersonalityFacadeAccessible:
    """The old personality_state public API is available on personality."""

    def test_init_personality_db_exists(self):
        from assistant import personality
        assert callable(personality.init_personality_db)

    def test_get_current_traits_exists(self):
        from assistant import personality
        assert callable(personality.get_current_traits)

    def test_update_traits_exists(self):
        from assistant import personality
        assert callable(personality.update_traits)

    def test_get_metadata_exists(self):
        from assistant import personality
        assert callable(personality.get_metadata)

    def test_set_metadata_exists(self):
        from assistant import personality
        assert callable(personality.set_metadata)

    def test_trait_constants_reexported(self):
        from assistant.personality import TRAIT_DEFAULTS, MAX_DELTA_PER_CYCLE, MAX_DELTA_PER_EVENT
        assert isinstance(TRAIT_DEFAULTS, dict)


class TestPersonalityEventsAccessible:
    """The old personality_events public API is available on personality."""

    def test_process_turn_exists(self):
        from assistant import personality
        assert callable(personality.process_turn)

    def test_check_absence_exists(self):
        from assistant import personality
        assert callable(personality.check_absence)


class TestPersonalityEventsInternal:
    """Event bumps call update_traits locally, not via a separate module."""

    @patch("assistant.personality.update_traits", return_value={"warmth": 0.51})
    def test_greeting_bump_calls_local_update(self, mock_update):
        from assistant import personality
        personality._events_fired_this_session.clear()
        personality._bumps_this_session = 0
        personality.process_turn("good morning", "small_talk")
        mock_update.assert_called_once()
        args = mock_update.call_args
        assert "warmth" in args[0][0]

    @patch("assistant.personality.update_traits", return_value={})
    def test_rate_limit_respected(self, mock_update):
        from assistant import personality
        personality._events_fired_this_session.clear()
        personality._bumps_this_session = personality._MAX_BUMPS_PER_SESSION
        personality.process_turn("good morning", "small_talk")
        mock_update.assert_not_called()


class TestOldModulesRemoved:
    """Verify the old modules no longer exist as separate files."""

    def test_personality_state_not_importable(self):
        try:
            importlib.import_module("assistant.personality_state")
            assert False, "personality_state should not exist as a separate module"
        except (ModuleNotFoundError, ImportError):
            pass

    def test_personality_events_not_importable(self):
        try:
            importlib.import_module("assistant.personality_events")
            assert False, "personality_events should not exist as a separate module"
        except (ModuleNotFoundError, ImportError):
            pass
