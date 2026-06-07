"""
test_p_items_batch.py — Tests for P-item batch fixes.

Covers: P2, P3, P6, P8, P11, P12, P13, P14 + TTS markdown stripping.

Run: python -m pytest tests/test_p_items_batch.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- P6: No emoji in log-facing strings ---

class TestP6NoEmojiInLogs:

    def test_no_emoji_in_agent_py(self):
        import assistant.automation.vision.agent as mod
        src = open(mod.__file__, encoding="utf-8").read()
        for ch in ("\U0001f3a4", "✅", "❌", "⛔", "\U0001f3a7"):
            assert ch not in src, f"Emoji {ch!r} found in agent.py"

    def test_no_emoji_in_verifier_py(self):
        import assistant.automation.vision.verifier as mod
        src = open(mod.__file__, encoding="utf-8").read()
        for ch in ("✅",):
            assert ch not in src, f"Emoji {ch!r} found in verifier.py"

    def test_no_emoji_in_native_py(self):
        import assistant.automation.native as mod
        src = open(mod.__file__, encoding="utf-8").read()
        for ch in ("⛔",):
            assert ch not in src, f"Emoji {ch!r} found in native.py"

    def test_no_emoji_in_stt_py(self):
        import assistant.io.audio.stt as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "\U0001f3a4" not in src, "Emoji mic found in stt.py"

    def test_no_emoji_in_wake_word_py(self):
        import assistant.io.audio.wake_word as mod
        src = open(mod.__file__, encoding="utf-8").read()
        for ch in ("\U0001f3a4", "\U0001f3a7"):
            assert ch not in src, f"Emoji {ch!r} found in wake_word.py"


# --- TTS markdown stripping ---

class TestTTSMarkdownStripping:

    def _clean(self, text):
        from assistant.io.audio.tts import _preprocess_for_speech
        return _preprocess_for_speech(text)

    def test_bold_stripped(self):
        assert self._clean("This is **bold** text") == "This is bold text"

    def test_italic_stripped(self):
        assert self._clean("This is *italic* text") == "This is italic text"

    def test_bold_and_italic(self):
        result = self._clean("**Bold** and *italic* together")
        assert "**" not in result
        assert "*" not in result
        assert "Bold" in result
        assert "italic" in result

    def test_strikethrough_stripped(self):
        assert self._clean("This is ~~wrong~~ right") == "This is wrong right"

    def test_stray_asterisks_removed(self):
        result = self._clean("What * do * you * want")
        assert "*" not in result

    def test_no_change_for_plain_text(self):
        assert self._clean("Hello world") == "Hello world"

    def test_multiple_bold_sections(self):
        result = self._clean("**First** and **second** things")
        assert "**" not in result
        assert "First" in result
        assert "second" in result

    def test_combined_with_emotion_tag(self):
        result = self._clean("[angry] You're *so* annoying")
        assert "*" not in result
        assert "so" in result
        assert "[angry]" not in result

    def test_real_world_llm_response(self):
        text = "[sarcastic] Ugh, *another* tutorial you want me to find?"
        result = self._clean(text)
        assert "*" not in result
        assert "another" in result


# --- P12: read_file deprecated intent removed ---

class TestP12ReadFileRemoved:

    def test_read_file_not_in_intents(self):
        from assistant.config import INTENTS
        assert "read_file" not in INTENTS

    def test_read_file_not_in_tools(self):
        from assistant.actions import _TOOLS
        assert "read_file" not in _TOOLS

    def test_read_file_remapped_in_execute(self):
        """read_file remap exists so stray LLM emissions still work."""
        import ast
        import assistant.actions as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert '"read_file"' in src or "'read_file'" in src

    def test_file_task_still_in_tools(self):
        from assistant.actions import _TOOLS
        assert "file_task" in _TOOLS


# --- P13: DEBUG_LOG config flag ---

class TestP13DebugLogFlag:

    def test_debug_log_config_exists(self):
        from assistant import config
        assert hasattr(config, "DEBUG_LOG")

    def test_debug_log_is_bool(self):
        from assistant.config import DEBUG_LOG
        assert isinstance(DEBUG_LOG, bool)

    def test_main_uses_config_flag(self):
        import assistant.main as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "config.DEBUG_LOG" in src
        assert "TODO: REMOVE" not in src


# --- P14: no TODO(remove-before-prod) markers ---

class TestP14NoProdTodos:

    def test_no_todo_in_dom(self):
        import assistant.automation.browser.dom as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "TODO(remove-before-prod)" not in src

    def test_no_todo_in_dom_orchestrator(self):
        import assistant.automation.browser.dom_orchestrator as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "TODO(remove-before-prod)" not in src

    def test_diag_helper_uses_debug_level(self):
        """_log_validation_dump should use logger.debug, not logger.info."""
        import assistant.automation.browser.dom_orchestrator as mod
        import inspect
        src = inspect.getsource(mod._log_validation_dump)
        assert "logger.debug(" in src
        assert "logger.info(" not in src


# --- P2: no bare except: in codebase ---

class TestP2NoBareExcept:

    def test_no_bare_except_in_native(self):
        import assistant.automation.native as mod
        src = open(mod.__file__, encoding="utf-8").read()
        import re
        hits = re.findall(r'^\s*except:\s*$', src, re.MULTILINE)
        assert not hits, f"Bare except: found in native.py"

    def test_no_bare_except_in_automation(self):
        import assistant.automation.browser.automation as mod
        src = open(mod.__file__, encoding="utf-8").read()
        import re
        hits = re.findall(r'^\s*except:\s*$', src, re.MULTILINE)
        assert not hits, f"Bare except: found in automation.py"

    def test_no_bare_except_anywhere(self):
        import re
        import assistant
        root = os.path.dirname(assistant.__file__)
        offenders = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as f:
                    src = f.read()
                hits = re.findall(r'^\s*except:\s*$', src, re.MULTILINE)
                if hits:
                    offenders.append(os.path.relpath(path, root))
        assert not offenders, f"Bare except: in: {offenders}"


# --- P3: reminders.py uses storage.db import, not local _get_conn wrapper ---

class TestP3NoLocalGetConn:

    def test_reminders_imports_from_storage_db(self):
        import assistant.reminders as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "from .storage.db import get_db" in src

    def test_no_standalone_get_conn_def(self):
        """No def _get_conn() function defined locally."""
        import assistant.reminders as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "def _get_conn" not in src


# --- P8: personality_say error truncation ---

class TestP8ErrorTruncation:

    def test_error_truncated_to_80_chars(self):
        from assistant.actions.responses import personality_say
        long_error = "x" * 200
        result = personality_say("error", error=long_error)
        assert len(long_error) > 80
        assert ("x" * 81) not in result

    def test_error_first_line_only(self):
        from assistant.actions.responses import personality_say
        multiline_error = "first line\nsecond line\nthird line"
        result = personality_say("error", error=multiline_error)
        assert "second line" not in result
        assert "third line" not in result

    def test_short_error_passes_through(self):
        from assistant.actions.responses import personality_say
        result = personality_say("error", error="brief")
        assert "brief" in result


# --- P11: no phase tags in source ---

class TestP11NoPhaseTags:

    PHASE_TAG_PATTERNS = [
        r'\bPV-\d', r'\bPA-\d', r'\bPE-\d', r'\bAR-\d', r'\bDA-\d',
        r'\bTP-\d', r'\bWA-\d', r'\bVL-\d', r'\bRC-\d', r'\bCE-\d',
        r'\bPhase [12][A-F]',
    ]

    def _scan_file(self, filepath: str) -> list[str]:
        import re
        with open(filepath, encoding="utf-8") as f:
            src = f.read()
        hits = []
        for pattern in self.PHASE_TAG_PATTERNS:
            matches = re.findall(pattern, src)
            hits.extend(matches)
        return hits

    def test_no_phase_tags_in_actions(self):
        import assistant.actions as mod
        src_dir = os.path.dirname(mod.__file__)
        for name in os.listdir(src_dir):
            if name.endswith(".py"):
                hits = self._scan_file(os.path.join(src_dir, name))
                assert not hits, f"Phase tags {hits} in actions/{name}"

    def test_no_phase_tags_in_automation(self):
        import assistant.automation as mod
        root = os.path.dirname(mod.__file__)
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)
                    hits = self._scan_file(path)
                    rel = os.path.relpath(path, root)
                    assert not hits, f"Phase tags {hits} in automation/{rel}"

    def test_no_phase_tags_in_io(self):
        import assistant.io as mod
        root = os.path.dirname(mod.__file__)
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)
                    hits = self._scan_file(path)
                    rel = os.path.relpath(path, root)
                    assert not hits, f"Phase tags {hits} in io/{rel}"

    def test_no_phase_tags_in_code_executor(self):
        import assistant.code_executor as mod
        root = os.path.dirname(mod.__file__)
        for name in os.listdir(root):
            if name.endswith(".py"):
                hits = self._scan_file(os.path.join(root, name))
                assert not hits, f"Phase tags {hits} in code_executor/{name}"

    def test_no_phase_tags_in_top_level_modules(self):
        import assistant
        root = os.path.dirname(assistant.__file__)
        for name in os.listdir(root):
            if name.endswith(".py"):
                hits = self._scan_file(os.path.join(root, name))
                assert not hits, f"Phase tags {hits} in {name}"


# --- P5: LLM prompts include current date context ---

class TestP5DatetimeContext:

    def test_date_context_line_format(self):
        from assistant.core.datetime_utils import date_context_line
        line = date_context_line()
        assert line.startswith("Current date/time:")
        assert "202" in line  # year

    def test_date_context_line_includes_day_name(self):
        from assistant.core.datetime_utils import date_context_line
        import datetime
        today_name = datetime.datetime.now().strftime("%A")
        assert today_name in date_context_line()

    def test_planner_injects_date(self):
        import assistant.actions.planner.planner as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "date_context_line" in src

    def test_code_routing_injects_date(self):
        import assistant.code_executor.routing as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "date_context_line" in src

    def test_orchestrator_injects_date(self):
        import assistant.code_executor.orchestrator as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "date_context_line" in src

    def test_web_search_injects_date(self):
        import assistant.actions.web as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "date_context_line" in src


# --- P1: no deprecated get_event_loop in async contexts ---

class TestP1AsyncioDeprecation:

    def test_main_uses_get_running_loop(self):
        import assistant.main as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "get_running_loop()" in src
        assert "get_event_loop()" not in src

    def test_tts_uses_get_running_loop(self):
        import assistant.io.audio.tts as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "get_running_loop()" in src
        assert "get_event_loop()" not in src

    def test_web_uses_get_running_loop(self):
        import assistant.actions.web as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "get_running_loop()" in src
        assert "get_event_loop()" not in src
