"""
test_agents.py — Tests for the Claude Code subagent configuration files.

Verifies that:
1. Agent markdown files exist and are well-formed
2. Required YAML frontmatter fields are present
3. Content sections cover the expected review areas
"""

import os
import re
import pytest

AGENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".claude", "agents")


def parse_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return None, content

    frontmatter_raw = match.group(1)
    body = match.group(2)

    # Simple key-value parse (no nested YAML needed)
    frontmatter = {}
    for line in frontmatter_raw.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


class TestSecurityReviewer:
    AGENT_FILE = os.path.join(AGENTS_DIR, "security-reviewer.md")

    def test_file_exists(self):
        assert os.path.isfile(self.AGENT_FILE), "security-reviewer.md must exist"

    def test_has_frontmatter(self):
        fm, _ = parse_frontmatter(self.AGENT_FILE)
        assert fm is not None, "Must have YAML frontmatter"

    def test_frontmatter_has_name(self):
        fm, _ = parse_frontmatter(self.AGENT_FILE)
        assert "name" in fm
        assert fm["name"] == "security-reviewer"

    def test_frontmatter_has_description(self):
        fm, _ = parse_frontmatter(self.AGENT_FILE)
        assert "description" in fm
        assert len(fm["description"]) > 10

    def test_frontmatter_has_model(self):
        fm, _ = parse_frontmatter(self.AGENT_FILE)
        assert "model" in fm

    def test_body_covers_sandbox_escapes(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "sandbox" in body.lower() or "code_executor" in body.lower()

    def test_body_covers_credential_handling(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "credential" in body.lower()

    def test_body_covers_input_validation(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "input" in body.lower() and "validation" in body.lower()

    def test_body_covers_policy_enforcement(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "policy" in body.lower()

    def test_body_has_output_format(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "CRITICAL" in body and "WARNING" in body


class TestCodeReviewer:
    AGENT_FILE = os.path.join(AGENTS_DIR, "code-reviewer.md")

    def test_file_exists(self):
        assert os.path.isfile(self.AGENT_FILE), "code-reviewer.md must exist"

    def test_has_frontmatter(self):
        fm, _ = parse_frontmatter(self.AGENT_FILE)
        assert fm is not None, "Must have YAML frontmatter"

    def test_frontmatter_has_name(self):
        fm, _ = parse_frontmatter(self.AGENT_FILE)
        assert "name" in fm
        assert fm["name"] == "code-reviewer"

    def test_frontmatter_has_description(self):
        fm, _ = parse_frontmatter(self.AGENT_FILE)
        assert "description" in fm
        assert len(fm["description"]) > 10

    def test_body_covers_correctness(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "correctness" in body.lower()

    def test_body_covers_async_patterns(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "async" in body.lower()

    def test_body_covers_bridge_protocol(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "7777" in body or "7778" in body or "bridge" in body.lower()

    def test_body_covers_error_handling(self):
        _, body = parse_frontmatter(self.AGENT_FILE)
        assert "error" in body.lower() and "handling" in body.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
