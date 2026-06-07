"""Tests for preferences.py merge — verify both facade and corrections are accessible."""

import importlib
from unittest.mock import patch, MagicMock


class TestPreferencesFacadeAccessible:
    """The old preference_store public API is available on preferences."""

    def test_init_preference_db_exists(self):
        from assistant import preferences
        assert callable(preferences.init_preference_db)

    def test_get_preference_exists(self):
        from assistant import preferences
        assert callable(preferences.get_preference)

    def test_set_preference_exists(self):
        from assistant import preferences
        assert callable(preferences.set_preference)

    def test_get_active_preferences_exists(self):
        from assistant import preferences
        assert callable(preferences.get_active_preferences)

    def test_confidence_constants_reexported(self):
        from assistant.preferences import (
            CONFIDENCE_SILENT, CONFIDENCE_ASK, CONFIDENCE_IGNORE,
        )
        assert CONFIDENCE_SILENT > CONFIDENCE_ASK > CONFIDENCE_IGNORE


class TestCorrectionsAccessible:
    """The old preference_corrections public API is available on preferences."""

    def test_check_for_corrections_exists(self):
        from assistant import preferences
        assert callable(preferences.check_for_corrections)

    def test_normalize_app_name_exists(self):
        from assistant import preferences
        assert callable(preferences._normalize_app_name)


class TestCorrectionsInternal:
    """Corrections call local preference functions, not a separate module."""

    @patch("assistant.preferences.set_preference")
    @patch("assistant.preferences.get_preference", return_value=None)
    def test_style_correction_calls_local(self, mock_get, mock_set):
        from assistant import preferences
        result = preferences.check_for_corrections("keep it brief")
        assert result is True
        mock_set.assert_called_once()
        args = mock_set.call_args
        assert args[1]["key"] == "verbosity" or args[0][0] == "verbosity"


class TestOldModulesRemoved:
    def test_preference_store_not_importable(self):
        try:
            importlib.import_module("assistant.preference_store")
            assert False, "preference_store should not exist"
        except (ModuleNotFoundError, ImportError):
            pass

    def test_preference_corrections_not_importable(self):
        try:
            importlib.import_module("assistant.preference_corrections")
            assert False, "preference_corrections should not exist"
        except (ModuleNotFoundError, ImportError):
            pass
