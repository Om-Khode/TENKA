"""
test_intent_scopes.py — Unit tests for intent-scoping intent scoping.

Run: python -m pytest tests/test_intent_scopes.py -v
"""

import sys
import types
from unittest.mock import patch

import pytest

from assistant.core.intent_scopes import SCOPES, ALWAYS_AVAILABLE


class TestConstants:
    def test_scopes_keys(self):
        assert "browser_mode" in SCOPES
        assert "recording_mode" in SCOPES
        assert "camera_mode" in SCOPES

    def test_always_available_has_core_intents(self):
        for intent in ("small_talk", "unknown", "memory_query", "store_memory",
                       "planner", "code_executor", "web_search", "shutdown"):
            assert intent in ALWAYS_AVAILABLE

    def test_scope_intents_not_in_always_available(self):
        for scope_intents in SCOPES.values():
            for intent in scope_intents:
                assert intent not in ALWAYS_AVAILABLE, (
                    f"{intent} in both SCOPES and ALWAYS_AVAILABLE"
                )

    def test_browser_scope_contents(self):
        assert SCOPES["browser_mode"] == {
            "browser_cdp_setup", "browse_url", "find_and_click", "read_screen"
        }

    def test_recording_scope_contents(self):
        assert SCOPES["recording_mode"] == {
            "start_recording", "stop_recording", "get_recording", "summarize_recording"
        }

    def test_camera_scope_contents(self):
        assert SCOPES["camera_mode"] == {
            "camera_look", "meet_face", "recognize_face", "forget_face"
        }


class TestDetectScope:
    @pytest.fixture(autouse=True)
    def reset_scope(self):
        import assistant.intent_scopes as mod
        mod._last_scope = ("general", 0)

    def _get_detect_scope(self):
        from assistant.intent_scopes import detect_scope
        return detect_scope

    @patch("assistant.intent_scopes._get_cdp_available", return_value=True)
    @patch("assistant.intent_scopes._get_recording_active", return_value=False)
    @patch("assistant.intent_scopes._get_camera_pending", return_value=False)
    def test_cdp_available_returns_browser_mode(self, _cam, _rec, _cdp):
        detect_scope = self._get_detect_scope()
        scope_name, intents = detect_scope(turn_number=1)
        assert scope_name == "browser_mode"
        assert "browse_url" in intents
        assert "small_talk" in intents

    @patch("assistant.intent_scopes._get_cdp_available", return_value=False)
    @patch("assistant.intent_scopes._get_recording_active", return_value=True)
    @patch("assistant.intent_scopes._get_camera_pending", return_value=False)
    def test_recording_active_returns_recording_mode(self, _cam, _rec, _cdp):
        detect_scope = self._get_detect_scope()
        scope_name, intents = detect_scope(turn_number=1)
        assert scope_name == "recording_mode"
        assert "stop_recording" in intents

    @patch("assistant.intent_scopes._get_cdp_available", return_value=False)
    @patch("assistant.intent_scopes._get_recording_active", return_value=False)
    @patch("assistant.intent_scopes._get_camera_pending", return_value=True)
    def test_camera_pending_returns_camera_mode(self, _cam, _rec, _cdp):
        detect_scope = self._get_detect_scope()
        scope_name, intents = detect_scope(turn_number=1)
        assert scope_name == "camera_mode"
        assert "camera_look" in intents

    @patch("assistant.intent_scopes._get_cdp_available", return_value=False)
    @patch("assistant.intent_scopes._get_recording_active", return_value=False)
    @patch("assistant.intent_scopes._get_camera_pending", return_value=False)
    def test_nothing_active_returns_general(self, _cam, _rec, _cdp):
        detect_scope = self._get_detect_scope()
        scope_name, intents = detect_scope(turn_number=1)
        assert scope_name == "general"
        all_intents = set(ALWAYS_AVAILABLE)
        for s in SCOPES.values():
            all_intents |= s
        assert intents == all_intents


class TestStickyScope:
    def _get_module(self):
        import assistant.intent_scopes as mod
        mod._last_scope = ("general", 0)
        return mod

    @patch("assistant.intent_scopes._get_cdp_available")
    @patch("assistant.intent_scopes._get_recording_active", return_value=False)
    @patch("assistant.intent_scopes._get_camera_pending", return_value=False)
    def test_sticky_persists_two_turns(self, _cam, _rec, mock_cdp):
        mod = self._get_module()
        mock_cdp.return_value = True
        scope1, _ = mod.detect_scope(turn_number=1)
        assert scope1 == "browser_mode"
        mock_cdp.return_value = False
        scope2, _ = mod.detect_scope(turn_number=2)
        assert scope2 == "browser_mode"
        scope3, _ = mod.detect_scope(turn_number=3)
        assert scope3 == "browser_mode"

    @patch("assistant.intent_scopes._get_cdp_available", return_value=False)
    @patch("assistant.intent_scopes._get_recording_active", return_value=False)
    @patch("assistant.intent_scopes._get_camera_pending", return_value=False)
    def test_sticky_decays_after_two_turns(self, _cam, _rec, _cdp):
        mod = self._get_module()
        mod._last_scope = ("browser_mode", 1)
        scope, _ = mod.detect_scope(turn_number=4)
        assert scope == "general"
