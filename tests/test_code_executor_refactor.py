"""Tests for S11 code_executor package split — verifies import paths and key functions."""

import sys
import types
from pathlib import Path

# Bootstrap: ensure assistant package is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub heavy modules to prevent real side effects
for mod_name in (
    "assistant.io.audio.tts", "assistant.io.audio.stt",
    "assistant.io.audio.speaker_verify", "assistant.io.unity_bridge",
    "assistant.io.audio.wake_word",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)


class TestImportPaths:
    """Verify all public and test-used import paths resolve after the split."""

    def test_public_api_from_package(self):
        from assistant.code_executor import (
            execute_code_task, run_code, GUI_HANDOFF_SIGNAL,
            PLANNER_ESCALATION_SIGNAL,
            _needs_retry, pop_pending_knowledge,
            detect_service_from_packages, get_oauth_env_map,
        )
        assert callable(execute_code_task)
        assert callable(run_code)
        assert GUI_HANDOFF_SIGNAL == "__NEEDS_GUI__"
        assert PLANNER_ESCALATION_SIGNAL == "__ESCALATE_PLANNER__"
        assert callable(_needs_retry)
        assert callable(pop_pending_knowledge)
        assert callable(detect_service_from_packages)
        assert callable(get_oauth_env_map)

    def test_submodule_imports(self):
        from assistant.code_executor.sandbox import run_code, _ast_scan, _run_tier1, _run_tier2
        from assistant.code_executor.prompts import get_router_system_prompt
        from assistant.code_executor.routing import _route_goal, TIER2_ALLOWED_PACKAGES
        from assistant.code_executor.templates import _load_template, _parameterize_code
        from assistant.code_executor.discovery import _run_discovery, _apply_key_fixes
        from assistant.code_executor.retry import _classify_error, _plan_fix
        from assistant.code_executor.orchestrator import execute_code_task
        from assistant.code_executor._utils import _needs_retry, _strip_code_fences
        assert callable(run_code)
        assert callable(_ast_scan)
        assert callable(get_router_system_prompt)
        assert isinstance(TIER2_ALLOWED_PACKAGES, frozenset)

    def test_detect_app_not_running_importable(self):
        """test_4c_app_not_running.py imports this."""
        from assistant.code_executor import _detect_app_not_running
        assert callable(_detect_app_not_running)

    def test_system_commands_importable(self):
        from assistant.automation.system_commands import run_system_command, KNOWN_COMMANDS
        assert callable(run_system_command)
        assert "bluetooth_on" in KNOWN_COMMANDS


class TestAstScan:
    """Verify AST scanning blocks dangerous code at each tier."""

    def test_tier1_blocks_subprocess(self):
        from assistant.code_executor.sandbox import _ast_scan
        result = _ast_scan("import subprocess\nsubprocess.run(['ls'])", tier=1)
        assert result is not None
        assert "BLOCKED" in result

    def test_tier1_blocks_requests(self):
        from assistant.code_executor.sandbox import _ast_scan
        result = _ast_scan("import requests\nrequests.get('http://x')", tier=1)
        assert result is not None
        assert "BLOCKED" in result

    def test_tier1_allows_math(self):
        from assistant.code_executor.sandbox import _ast_scan
        result = _ast_scan("import math\nprint(math.pi)", tier=1)
        assert result is None

    def test_tier2_blocks_eval(self):
        from assistant.code_executor.sandbox import _ast_scan
        result = _ast_scan("eval('1+1')", tier=2)
        assert result is not None
        assert "BLOCKED" in result

    def test_tier2_allows_os_remove(self):
        """Tier 2 allows os.remove (sandbox dir only)."""
        from assistant.code_executor.sandbox import _ast_scan
        result = _ast_scan("import os\nos.remove('file.txt')", tier=2)
        assert result is None

    def test_syntax_error_fallback(self):
        from assistant.code_executor.sandbox import _ast_scan
        result = _ast_scan("import subprocess\nsubprocess.run(", tier=1)
        assert result is not None
        assert "BLOCKED" in result or "unsafe" in result.lower()


class TestClassifyError:
    """Verify deterministic error classification."""

    def test_traceback_unicode(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("Traceback (most recent call last):\n  ...\nUnicodeEncodeError: ...")
        assert r["category"] == "encoding"

    def test_traceback_import(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("Traceback (most recent call last):\n  ...\nModuleNotFoundError: No module named 'foo'")
        assert r["category"] == "import"

    def test_traceback_key_error(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("Traceback (most recent call last):\n  ...\nKeyError: 'track'")
        assert r["category"] == "field_access"

    def test_timeout(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("TIMEOUT")
        assert r["category"] == "timeout"

    def test_blocked(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("BLOCKED: import of 'subprocess' not allowed")
        assert r["category"] == "blocked"

    def test_empty_output(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("")
        assert r["category"] == "unknown"

    def test_none_placeholder_lines(self):
        from assistant.code_executor.retry import _classify_error
        lines = "\n".join(["None"] * 8 + ["Song Title"])
        r = _classify_error(lines)
        assert r["category"] == "field_access"

    def test_scope_error(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("Traceback...\nInsufficient client scope")
        assert r["category"] in ("scope", "logic")

    def test_no_output(self):
        from assistant.code_executor.retry import _classify_error
        r = _classify_error("(no output)")
        assert r["category"] == "no_output"


class TestNeedsRetry:
    """Verify _needs_retry edge cases."""

    def test_empty_string(self):
        from assistant.code_executor._utils import _needs_retry
        assert _needs_retry("") is True

    def test_success(self):
        from assistant.code_executor._utils import _needs_retry
        assert _needs_retry("Playing: Bohemian Rhapsody by Queen") is False

    def test_needs_oauth_no_retry(self):
        from assistant.code_executor._utils import _needs_retry
        assert _needs_retry("NEEDS_OAUTH|spotify|...") is False

    def test_app_not_ready(self):
        from assistant.code_executor._utils import _needs_retry
        assert _needs_retry("APP_NOT_READY|spotify") is True

    def test_timeout(self):
        from assistant.code_executor._utils import _needs_retry
        assert _needs_retry("TIMEOUT") is True

    def test_completed_successfully(self):
        from assistant.code_executor._utils import _needs_retry
        assert _needs_retry("(completed successfully)") is False

    def test_label_only_lines(self):
        from assistant.code_executor._utils import _needs_retry
        assert _needs_retry("Title:\nArtist:") is True


class TestGoalMatchesTemplate:
    """Verify keyword overlap logic for template reuse."""

    def test_exact_match(self):
        from assistant.code_executor.templates import _goal_matches_template
        assert _goal_matches_template("play spotify", "play spotify") is True

    def test_no_overlap(self):
        from assistant.code_executor.templates import _goal_matches_template
        assert _goal_matches_template("send email", "play spotify") is False

    def test_empty_stored_goal(self):
        from assistant.code_executor.templates import _goal_matches_template
        assert _goal_matches_template("anything", "") is True

    def test_partial_overlap(self):
        from assistant.code_executor.templates import _goal_matches_template
        assert _goal_matches_template("play song on spotify", "play spotify music") is True


class TestApplyKeyFixes:
    """Verify deterministic key replacement."""

    def test_single_key_replacement(self):
        from assistant.code_executor.discovery import _apply_key_fixes
        code = "name = item.get('track').get('name')"
        fixed = _apply_key_fixes(code, {"track": "item"})
        assert fixed is not None
        assert ".get('item')" in fixed
        assert ".get('track')" not in fixed

    def test_no_replacements_returns_none(self):
        from assistant.code_executor.discovery import _apply_key_fixes
        result = _apply_key_fixes("x = 1", {})
        assert result is None

    def test_bracket_notation(self):
        from assistant.code_executor.discovery import _apply_key_fixes
        code = "name = item['track']['name']"
        fixed = _apply_key_fixes(code, {"track": "item"})
        assert fixed is not None
        assert "['item']" in fixed


class TestKwargFixes:
    """Verify deterministic kwarg replacement from discovery signatures."""

    def test_extract_finds_force_play(self):
        from assistant.code_executor.discovery import _extract_kwarg_fixes
        result = "An unexpected error occurred: Spotify.transfer_playback() got an unexpected keyword argument 'play'"
        discovery = "DISCOVERY:transfer_playback:signature=(device_id, force_play=True)"
        fixes = _extract_kwarg_fixes(result, discovery)
        assert "transfer_playback" in fixes
        assert fixes["transfer_playback"] == ("play", "force_play")

    def test_extract_no_match_returns_empty(self):
        from assistant.code_executor.discovery import _extract_kwarg_fixes
        result = "Some other error"
        discovery = "DISCOVERY:devices:type=dict"
        assert _extract_kwarg_fixes(result, discovery) == {}

    def test_extract_no_discovery_returns_empty(self):
        from assistant.code_executor.discovery import _extract_kwarg_fixes
        result = "Spotify.transfer_playback() got an unexpected keyword argument 'play'"
        assert _extract_kwarg_fixes(result, "") == {}

    def test_apply_replaces_in_method_call(self):
        from assistant.code_executor.discovery import _apply_kwarg_fixes
        code = "sp.transfer_playback(device_id=did, play=True)"
        fixed = _apply_kwarg_fixes(code, {"transfer_playback": ("play", "force_play")})
        assert fixed is not None
        assert "force_play=True" in fixed
        assert ", play=" not in fixed

    def test_apply_preserves_other_lines(self):
        from assistant.code_executor.discovery import _apply_kwarg_fixes
        code = "x = play\nsp.transfer_playback(device_id=d, play=True)\ny = play"
        fixed = _apply_kwarg_fixes(code, {"transfer_playback": ("play", "force_play")})
        assert fixed is not None
        lines = fixed.split("\n")
        assert lines[0] == "x = play"
        assert "force_play=True" in lines[1]
        assert lines[2] == "y = play"

    def test_apply_no_match_returns_none(self):
        from assistant.code_executor.discovery import _apply_kwarg_fixes
        code = "sp.devices()"
        assert _apply_kwarg_fixes(code, {"transfer_playback": ("play", "force_play")}) is None

    def test_apply_doesnt_touch_other_kwargs(self):
        from assistant.code_executor.discovery import _apply_kwarg_fixes
        code = "sp.transfer_playback(device_id=d, display=True, play=False)"
        fixed = _apply_kwarg_fixes(code, {"transfer_playback": ("play", "force_play")})
        assert fixed is not None
        assert "display=True" in fixed
        assert "force_play=False" in fixed


class TestParameterizeCode:
    """Verify template parameterization."""

    def test_simple_string_replacement(self):
        from assistant.code_executor.templates import _parameterize_code
        code = "import os\nquery = 'hello world'"
        result = _parameterize_code(code, {"query": "hello world"})
        assert "os.environ.get('PARAM_QUERY'" in result
        assert "'hello world'" not in result

    def test_already_parameterized_skipped(self):
        from assistant.code_executor.templates import _parameterize_code
        code = "import os\nquery = os.environ.get('PARAM_QUERY', '')"
        result = _parameterize_code(code, {"query": "hello"})
        assert result == code

    def test_path_variant_matching(self):
        from assistant.code_executor.templates import _parameterize_code
        code = "import os\npath = r'C:\\Users\\test\\file.txt'"
        result = _parameterize_code(code, {"path": "C:/Users/test/file.txt"})
        assert "os.environ.get('PARAM_PATH'" in result
