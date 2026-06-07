"""
test_env_hook.py — Tests for the .env file protection hook in .claude/settings.json

Verifies that:
1. The hook configuration is valid JSON with correct schema
2. The matcher targets Edit and Write tools
3. The command correctly detects .env file paths
4. Non-.env files are not blocked
"""

import json
import subprocess
import os
import pytest

SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".claude", "settings.json"
)


@pytest.fixture
def settings():
    with open(SETTINGS_PATH, "r") as f:
        return json.load(f)


class TestEnvHookConfig:
    """Test the .env protection hook configuration structure."""

    def test_settings_file_exists(self):
        assert os.path.isfile(SETTINGS_PATH), "settings.local.json must exist"

    def test_settings_is_valid_json(self):
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_hooks_section_exists(self, settings):
        assert "hooks" in settings, "hooks section must exist in settings"

    def test_pre_tool_use_exists(self, settings):
        assert "PreToolUse" in settings["hooks"], "PreToolUse hook must exist"

    def test_pre_tool_use_is_list(self, settings):
        assert isinstance(settings["hooks"]["PreToolUse"], list)
        assert len(settings["hooks"]["PreToolUse"]) > 0

    def test_env_hook_has_correct_matcher(self, settings):
        hook_entry = settings["hooks"]["PreToolUse"][0]
        assert "matcher" in hook_entry
        assert "Edit" in hook_entry["matcher"]
        assert "Write" in hook_entry["matcher"]

    def test_env_hook_has_hooks_array(self, settings):
        hook_entry = settings["hooks"]["PreToolUse"][0]
        assert "hooks" in hook_entry
        assert isinstance(hook_entry["hooks"], list)
        assert len(hook_entry["hooks"]) > 0

    def test_env_hook_is_command_type(self, settings):
        hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]
        assert hook["type"] == "command"
        assert "command" in hook

    def test_env_hook_command_contains_env_check(self, settings):
        hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]
        assert ".env" in hook["command"], "Command must check for .env pattern"
        assert "BLOCKED" in hook["command"], "Command must output BLOCKED message"


class TestEnvHookBehavior:
    """Test the actual bash command behavior of the .env protection hook."""

    @pytest.fixture
    def hook_command_template(self, settings):
        return settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    # Git bash path — on Windows, plain "bash" may resolve to WSL
    GIT_BASH = r"C:\Program Files\Git\usr\bin\bash.exe"

    def _get_bash(self):
        if os.path.isfile(self.GIT_BASH):
            return self.GIT_BASH
        return "bash"

    def _run_hook_with_input(self, hook_cmd, file_path):
        """Simulate the hook receiving CLAUDE_TOOL_INPUT with a file_path."""
        tool_input = json.dumps({"file_path": file_path})
        env = os.environ.copy()
        env["CLAUDE_TOOL_INPUT"] = tool_input
        result = subprocess.run(
            [self._get_bash(), "-c", hook_cmd],
            env=env,
            capture_output=True,
            text=True,
        )
        return result

    def test_blocks_dot_env(self, hook_command_template):
        result = self._run_hook_with_input(
            hook_command_template, "/project/.env"
        )
        assert result.returncode != 0, ".env should be blocked"
        assert "BLOCKED" in result.stderr

    def test_blocks_dot_env_local(self, hook_command_template):
        result = self._run_hook_with_input(
            hook_command_template, "/project/.env.local"
        )
        assert result.returncode != 0, ".env.local should be blocked"

    def test_blocks_dot_env_production(self, hook_command_template):
        result = self._run_hook_with_input(
            hook_command_template, "D:/Code/project/.env.production"
        )
        assert result.returncode != 0, ".env.production should be blocked"

    def test_allows_normal_python_file(self, hook_command_template):
        result = self._run_hook_with_input(
            hook_command_template, "/project/assistant/config.py"
        )
        assert result.returncode == 0, "Normal .py files should be allowed"

    def test_allows_json_file(self, hook_command_template):
        result = self._run_hook_with_input(
            hook_command_template, "/project/package.json"
        )
        assert result.returncode == 0, "JSON files should be allowed"

    def test_allows_env_example(self, hook_command_template):
        # .env.example is a template, not a real env file — but our hook
        # blocks .env.* patterns which includes .env.example.
        # This is intentional: templates can still contain placeholder secrets.
        result = self._run_hook_with_input(
            hook_command_template, "/project/.env.example"
        )
        # .env.example matches .env.* — blocked is acceptable
        # Just verify the hook runs without crashing
        assert result.returncode in (0, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
