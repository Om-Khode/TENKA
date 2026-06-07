"""
test_computer_task_p2_get_text.py — Tests for P2: get_text improvements + list_elements.

Verifies:
1. get_text uses element.text(max_depth=N) instead of old .get_text()
2. get_text uses .within() for window-scoped lookups
3. list_elements for Terminator uses get_window_tree instead of returning a stub
4. _format_ui_tree produces LLM-readable output
"""

import os
import pytest

APP_AUTO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assistant", "automation", "native.py"
)


@pytest.fixture
def source():
    with open(APP_AUTO_PATH, "r", encoding="utf-8") as f:
        return f.read()


class TestGetTextImprovements:
    def test_uses_text_with_max_depth(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def get_text" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and ".text(max_depth=" in line:
                found = True
        assert found, "get_text must use element.text(max_depth=N)"

    def test_uses_within_for_window_scope(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def get_text" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and ".within(" in line:
                found = True
        assert found, "get_text approach 2 must use .within() for window scoping"

    def test_window_title_fallback_preserved(self, source):
        in_func = False
        found_gw = False
        for line in source.splitlines():
            if "async def get_text" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and "pygetwindow" in line:
                found_gw = True
        assert found_gw, "get_text must preserve window title fallback (approach 3)"


class TestListElements:
    def test_list_elements_no_stub_for_terminator(self, source):
        """Verify the old stub string is removed."""
        assert "Element listing not inherently supported" not in source, (
            "list_elements should no longer return a stub for Terminator"
        )

    def test_list_elements_uses_get_window_tree(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def list_elements" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and "get_window_tree" in line:
                found = True
        assert found, "list_elements must use desktop.get_window_tree()"

    def test_list_elements_uses_applications_for_pid(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def list_elements" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and "applications()" in line:
                found = True
        assert found, "list_elements must use desktop.applications() to find PID"


class TestFormatUiTree:
    def test_format_ui_tree_function_exists(self, source):
        assert "def _format_ui_tree" in source, "_format_ui_tree helper must exist"

    def test_format_ui_tree_handles_recursion(self, source):
        assert "max_depth" in source, "_format_ui_tree must respect max_depth"

    def test_format_ui_tree_outputs_selectors(self, source):
        """Verify the formatter outputs role: and name: selectors."""
        # Extract _format_ui_tree function body using indentation-aware parsing
        lines = []
        in_func = False
        base_indent = None
        for line in source.splitlines():
            if "def _format_ui_tree(" in line:
                in_func = True
                base_indent = len(line) - len(line.lstrip())
                continue
            if in_func:
                stripped = line.lstrip()
                current_indent = len(line) - len(line.lstrip())
                if stripped and current_indent <= base_indent and (stripped.startswith("def ") or stripped.startswith("async def ")):
                    break
                lines.append(line)
        body = "\n".join(lines)
        assert "role:" in body, "_format_ui_tree must output role: selectors"
        assert "name:" in body, "_format_ui_tree must output name: selectors"

    def test_pywinauto_list_elements_unchanged(self, source):
        """Verify pywinauto path still uses print_control_identifiers."""
        assert "print_control_identifiers" in source, (
            "pywinauto list_elements must still use print_control_identifiers"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
