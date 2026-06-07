"""
test_skill.py — Tests for the test-assistant skill configuration.

Verifies that:
1. The SKILL.md file exists and has valid frontmatter
2. Required fields (name, description, disable-model-invocation) are present
3. The skill content covers all expected smoke-test steps
"""

import os
import re
import pytest

SKILL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".claude", "skills", "test-assistant"
)
SKILL_FILE = os.path.join(SKILL_DIR, "SKILL.md")


def parse_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return None, content

    frontmatter_raw = match.group(1)
    body = match.group(2)

    frontmatter = {}
    for line in frontmatter_raw.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


class TestSkillStructure:
    def test_skill_directory_exists(self):
        assert os.path.isdir(SKILL_DIR), "test-assistant skill directory must exist"

    def test_skill_file_exists(self):
        assert os.path.isfile(SKILL_FILE), "SKILL.md must exist"


class TestSkillFrontmatter:
    def test_has_frontmatter(self):
        fm, _ = parse_frontmatter(SKILL_FILE)
        assert fm is not None, "Must have YAML frontmatter"

    def test_has_name(self):
        fm, _ = parse_frontmatter(SKILL_FILE)
        assert "name" in fm
        assert fm["name"] == "test-assistant"

    def test_has_description(self):
        fm, _ = parse_frontmatter(SKILL_FILE)
        assert "description" in fm
        assert len(fm["description"]) > 10

    def test_is_user_invocable_only(self):
        fm, _ = parse_frontmatter(SKILL_FILE)
        assert "disable-model-invocation" in fm
        assert fm["disable-model-invocation"] == "true"


class TestSkillContent:
    """Verify the skill covers all expected smoke-test areas."""

    @pytest.fixture
    def body(self):
        _, body = parse_frontmatter(SKILL_FILE)
        return body.lower()

    def test_checks_python_environment(self, body):
        assert "python" in body

    def test_checks_config(self, body):
        assert "config" in body

    def test_checks_module_imports(self, body):
        assert "import" in body

    def test_checks_env_file(self, body):
        assert ".env" in body

    def test_checks_bridge_ports(self, body):
        assert "7777" in body or "7778" in body or "port" in body

    def test_has_results_format(self, body):
        assert "result" in body or "report" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
