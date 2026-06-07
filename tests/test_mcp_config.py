"""
test_mcp_config.py — Tests for MCP server configuration (context7).

Verifies that:
1. The context7 MCP server was added to the Claude config
2. The configuration has the correct command structure
"""

import json
import os
import pytest

# context7 is added to the user-level Claude config for this project
CLAUDE_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".claude.json")


@pytest.fixture
def claude_config():
    if not os.path.isfile(CLAUDE_CONFIG_PATH):
        pytest.skip("~/.claude.json not found")
    with open(CLAUDE_CONFIG_PATH, "r") as f:
        return json.load(f)


class TestContext7MCP:
    def test_claude_config_exists(self):
        assert os.path.isfile(CLAUDE_CONFIG_PATH), "~/.claude.json must exist"

    def test_claude_config_is_valid_json(self):
        with open(CLAUDE_CONFIG_PATH, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_mcp_servers_section_exists(self, claude_config):
        # MCP servers can be in projects or top-level mcpServers
        has_mcp = False
        if "mcpServers" in claude_config:
            has_mcp = True
        if "projects" in claude_config:
            for project_key, project_val in claude_config["projects"].items():
                if isinstance(project_val, dict) and "mcpServers" in project_val:
                    has_mcp = True
                    break
        assert has_mcp, "MCP servers section must exist somewhere in config"

    def test_context7_is_configured(self, claude_config):
        """Check that context7 appears in any MCP servers section."""
        found = False

        # Check top-level
        if "mcpServers" in claude_config:
            if "context7" in claude_config["mcpServers"]:
                found = True

        # Check project-level
        if "projects" in claude_config:
            for project_key, project_val in claude_config["projects"].items():
                if isinstance(project_val, dict) and "mcpServers" in project_val:
                    if "context7" in project_val["mcpServers"]:
                        found = True
                        break

        assert found, "context7 MCP server must be configured"

    def test_context7_uses_npx(self, claude_config):
        """Verify context7 is configured with npx command."""
        server_config = None

        if "mcpServers" in claude_config:
            server_config = claude_config["mcpServers"].get("context7")

        if server_config is None and "projects" in claude_config:
            for project_key, project_val in claude_config["projects"].items():
                if isinstance(project_val, dict) and "mcpServers" in project_val:
                    server_config = project_val["mcpServers"].get("context7")
                    if server_config:
                        break

        assert server_config is not None, "context7 config must be found"
        assert "command" in server_config
        assert server_config["command"] == "npx"
        assert "args" in server_config
        assert any("context7" in arg for arg in server_config["args"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
