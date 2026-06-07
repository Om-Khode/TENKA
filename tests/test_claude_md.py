"""
test_claude_md.py — Tests for the CLAUDE.md project instructions file.

Verifies that:
1. CLAUDE.md exists at the repo root
2. It contains essential sections for project context
3. Key architectural details are documented
"""

import os
import pytest

CLAUDE_MD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLAUDE.md")


@pytest.fixture
def claude_md():
    with open(CLAUDE_MD_PATH, "r", encoding="utf-8") as f:
        return f.read()


class TestClaudeMdExists:
    def test_file_exists(self):
        assert os.path.isfile(CLAUDE_MD_PATH), "CLAUDE.md must exist at repo root"

    def test_file_not_empty(self):
        assert os.path.getsize(CLAUDE_MD_PATH) > 100, "CLAUDE.md must have content"


class TestClaudeMdContent:
    def test_has_project_name(self, claude_md):
        assert "TENKA" in claude_md

    def test_documents_architecture(self, claude_md):
        assert "architecture" in claude_md.lower() or "unity" in claude_md.lower()

    def test_documents_tcp_ports(self, claude_md):
        assert "7777" in claude_md
        assert "7778" in claude_md

    def test_documents_python_bridge(self, claude_md):
        assert "bridge.py" in claude_md or "PythonBridge" in claude_md

    def test_documents_running_instructions(self, claude_md):
        assert "python" in claude_md.lower()
        assert "requirements.txt" in claude_md or "pip install" in claude_md

    def test_documents_llm_providers(self, claude_md):
        assert "Groq" in claude_md or "groq" in claude_md
        assert "Ollama" in claude_md or "ollama" in claude_md

    def test_documents_stt(self, claude_md):
        assert "whisper" in claude_md.lower() or "stt" in claude_md.lower()

    def test_documents_tts(self, claude_md):
        assert "Kokoro" in claude_md or "tts" in claude_md.lower()

    def test_documents_key_modules(self, claude_md):
        key_modules = ["actions.py", "code_executor.py", "llm.py", "config.py"]
        for module in key_modules:
            assert module in claude_md, f"{module} should be documented"

    def test_documents_conventions(self, claude_md):
        assert "convention" in claude_md.lower() or "commit" in claude_md.lower()

    def test_documents_license(self, claude_md):
        assert "license" in claude_md.lower()

    def test_documents_unity_version(self, claude_md):
        assert "6000" in claude_md or "Unity" in claude_md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
