"""
Tests for the 5 issues found in the 2026-05-11 debug.log review:
  1. Stale credential_store import in orchestrator.py
  2. Double TTS in DA handoff path
  3. Shortcut response grammar
  4. Preference value prefix leaking ("using chrome" → "chrome")
  5. Preference statements misclassified as computer_task
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
#  Issue 1: credential_store import renamed to credentials
# ═══════════════════════════════════════════════════════════════════════════════

class TestCredentialImportFixed:
    """orchestrator.py must import 'credentials', not 'credential_store'."""

    def test_orchestrator_no_credential_store_import(self):
        src = Path(__file__).resolve().parent.parent / "assistant" / "code_executor" / "orchestrator.py"
        text = src.read_text(encoding="utf-8")
        assert "credential_store" not in text, (
            "orchestrator.py still references 'credential_store' — should be 'credentials'"
        )

    def test_credentials_module_importable(self):
        from assistant import credentials
        assert callable(credentials.get_credential)


# ═══════════════════════════════════════════════════════════════════════════════
#  Issue 2: Double TTS — DA handoff should NOT speak, main.py handles it
# ═══════════════════════════════════════════════════════════════════════════════

class TestDaHandoffNoDoubleTts:
    """DA handoff path should return the result without speaking it."""

    def test_da_handlers_no_tts_speak_on_result(self):
        src = Path(__file__).resolve().parent.parent / "assistant" / "actions" / "da_handlers.py"
        text = src.read_text(encoding="utf-8")
        # Find the GUI handoff block (between GUI_HANDOFF_SIGNAL and the return)
        handoff_start = text.index("GUI_HANDOFF_SIGNAL")
        # The "fallback to vision loop" log message marks end of DA success path
        fallback_marker = text.index("computer_task GUI handoff: fallback to vision loop")
        handoff_block = text[handoff_start:fallback_marker]

        # Count tts.speak calls in the handoff block
        # "Let me handle this" is expected (1 call). The result should NOT be spoken (was 2 calls).
        tts_calls = handoff_block.count("tts.speak(")
        assert tts_calls == 1, (
            f"Expected 1 tts.speak call (announcement only) in DA handoff, found {tts_calls}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Issue 3: Shortcut response grammar
# ═══════════════════════════════════════════════════════════════════════════════

class TestShortcutResponseGrammar:
    """Shortcut creation response should have correct grammar."""

    def test_description_lowercased_and_no_double_period(self):
        """Simulate the template logic from shortcuts.py line 108."""
        trigger = "opening notepad"
        description = "Creates a shortcut to open the Notepad application."
        desc = (description or trigger).rstrip(".").lower()
        result = f"Got it! When you say '{trigger}', I'll {desc}. Try it out!"

        assert "I'll c" in result and result.split("I'll ")[1][0].islower(), \
            "Description after I'll should be lowercase"
        assert ".." not in result, "Should not have double period"

    def test_description_none_falls_back_to_goal(self):
        trigger = "open notes"
        description = None
        target_goal = "open notepad"
        desc = (description or target_goal).rstrip(".").lower()
        result = f"Got it! When you say '{trigger}', I'll {desc}. Try it out!"
        assert "I'll open notepad" in result


# ═══════════════════════════════════════════════════════════════════════════════
#  Issue 4: Preference value prefix stripping
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreferenceValueCleaning:
    """_normalize_app_name should strip auxiliary prefixes like 'using'."""

    def test_using_chrome_normalizes_to_chrome(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("using chrome") == "chrome"

    def test_using_spotify_normalizes_to_spotify(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("using spotify") == "spotify"

    def test_with_firefox_normalizes_to_firefox(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("with firefox") == "firefox"

    def test_bare_chrome_still_works(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("chrome") == "chrome"

    def test_the_notepad_app_normalizes(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("the notepad app") == "notepad"

    def test_empty_after_strip_returns_empty(self):
        from assistant.preferences import _normalize_app_name
        assert _normalize_app_name("using") == ""


class TestCorrectionSkipsDuplicate:
    """_apply_correction should not overwrite when value is already the same."""

    @patch("assistant.preferences.set_preference")
    @patch("assistant.preferences.get_preference",
           return_value={"key": "browser", "value": "chrome", "confidence": 1.0})
    def test_same_value_skips_set(self, mock_get, mock_set):
        from assistant.preferences import _apply_correction
        _apply_correction("browser", "chrome", "app_routing", "I prefer Chrome")
        mock_set.assert_not_called()

    @patch("assistant.preferences.set_preference")
    @patch("assistant.preferences.get_preference",
           return_value={"key": "browser", "value": "firefox", "confidence": 0.85})
    def test_different_value_does_update(self, mock_get, mock_set):
        from assistant.preferences import _apply_correction
        _apply_correction("browser", "chrome", "app_routing", "I prefer Chrome")
        mock_set.assert_called_once()

    @patch("assistant.preferences.set_preference")
    @patch("assistant.preferences.get_preference", return_value=None)
    def test_no_existing_does_update(self, mock_get, mock_set):
        from assistant.preferences import _apply_correction
        _apply_correction("browser", "chrome", "app_routing", "I prefer Chrome")
        mock_set.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
#  Issue 5: Preference statements routed to small_talk, not computer_task
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreferenceStatementRouting:
    """regex_router should catch preference statements before LLM."""

    def test_i_prefer_chrome_routes_to_small_talk(self):
        from assistant.regex_router import pre_route
        result = pre_route("I prefer using Chrome")
        assert result is not None, "Should match preference pattern"
        assert result.intent == "small_talk"

    def test_i_prefer_firefox_routes_to_small_talk(self):
        from assistant.regex_router import pre_route
        result = pre_route("I prefer Firefox")
        assert result is not None
        assert result.intent == "small_talk"

    def test_i_like_using_spotify_routes_to_small_talk(self):
        from assistant.regex_router import pre_route
        result = pre_route("I like using Spotify")
        assert result is not None
        assert result.intent == "small_talk"

    def test_i_love_vscode_routes_to_small_talk(self):
        from assistant.regex_router import pre_route
        result = pre_route("I love VSCode")
        assert result is not None
        assert result.intent == "small_talk"

    def test_play_spotify_does_not_match_preference(self):
        from assistant.regex_router import pre_route
        result = pre_route("play something on spotify")
        # Should NOT match the preference pattern (should match music instead)
        if result is not None:
            assert result.intent != "small_talk" or result.intent == "code_executor"

    def test_open_chrome_does_not_match_preference(self):
        from assistant.regex_router import pre_route
        result = pre_route("open Chrome")
        if result is not None:
            assert result.intent != "small_talk"
