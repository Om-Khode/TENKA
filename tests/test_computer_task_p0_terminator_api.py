"""
test_computer_task_p0_terminator_api.py — Tests for P0: Terminator API mismatch fix.

Verifies that app_automation.py calls await locator.first() before element
interaction methods (click, type_text, text) instead of calling them
directly on Locator objects.
"""

import ast
import os
import pytest

APP_AUTO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assistant", "automation", "native.py"
)


@pytest.fixture
def source():
    with open(APP_AUTO_PATH, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def tree(source):
    return ast.parse(source)


def _get_async_func(tree, name):
    """Find an async function definition by name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


class TestClickElementUsesFirst:
    def test_click_element_exists(self, tree):
        func = _get_async_func(tree, "click_element")
        assert func is not None

    def test_click_element_contains_await_first(self, source):
        # Find the click_element function body and check for .first()
        in_func = False
        found_first = False
        for line in source.splitlines():
            if "async def click_element" in line:
                in_func = True
            elif in_func and line and not line[0].isspace():
                break
            if in_func and ".first()" in line:
                found_first = True
        assert found_first, "click_element must call .first() on locator before interaction"

    def test_click_element_no_direct_locator_click(self, source):
        # Ensure we don't have the old pattern: loc.locator(selector)\n  elem.click()
        # without .first() in between
        in_func = False
        for line in source.splitlines():
            if "async def click_element" in line:
                in_func = True
            elif in_func and "async def " in line:
                break
            if in_func and "terminator" in source:
                # Old broken pattern: elem = loc.locator(selector) followed by elem.click()
                # New correct pattern: elem = await loc.locator(selector).first()
                if "loc.locator(selector)" in line and ".first()" not in line:
                    if "elem = " in line:
                        pytest.fail("click_element still assigns locator without .first()")


class TestTypeTextUsesFirst:
    def test_type_text_contains_await_first(self, source):
        in_func = False
        found_first = False
        for line in source.splitlines():
            if "async def type_text" in line:
                in_func = True
            elif in_func and line.strip() and not line[0].isspace() and "def " in line:
                break
            if in_func and ".first()" in line and "locator" in line:
                found_first = True
        assert found_first, "type_text must call .first() on locator before type_text"


class TestGetTextUsesFirst:
    def test_get_text_approach1_uses_first(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def get_text" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and "desktop.locator(selector).first()" in line:
                found = True
        assert found, "get_text approach 1 must use desktop.locator(selector).first()"

    def test_get_text_uses_text_method(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def get_text" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and ".text(max_depth=" in line:
                found = True
        assert found, "get_text must use element.text(max_depth=N) instead of .get_text()"

    def test_get_text_no_old_get_text_call(self, source):
        in_func = False
        for line in source.splitlines():
            if "async def get_text" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and ".get_text()" in line:
                pytest.fail("get_text still uses the old .get_text() Locator method")

    def test_get_text_approach2_uses_within(self, source):
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


class TestFocusWindowUsesFirst:
    def test_focus_window_contains_first(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def focus_window" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and ".first()" in line:
                found = True
        assert found, "focus_window must call .first() before .click()"


class TestCloseAppUsesFirst:
    def test_close_app_contains_first(self, source):
        in_func = False
        found = False
        for line in source.splitlines():
            if "async def close_app" in line:
                in_func = True
            elif in_func and line.strip().startswith("async def "):
                break
            if in_func and ".first()" in line:
                found = True
        assert found, "close_app must call .first() before .click()"


class TestPywinautoUnchanged:
    """Verify pywinauto fallback paths don't use .first() (sync API)."""

    def _extract_function_body(self, source, func_name):
        """Extract lines belonging to a specific function."""
        lines = []
        in_func = False
        base_indent = None
        for line in source.splitlines():
            if f"def {func_name}(" in line:
                in_func = True
                base_indent = len(line) - len(line.lstrip())
                lines.append(line)
                continue
            if in_func:
                stripped = line.lstrip()
                current_indent = len(line) - len(line.lstrip())
                if stripped and current_indent <= base_indent and (stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class ")):
                    break
                lines.append(line)
        return "\n".join(lines)

    def test_sync_click_no_first(self, source):
        body = self._extract_function_body(source, "_sync_click_element")
        assert ".first()" not in body, "pywinauto _sync_click_element should not use .first()"

    def test_sync_type_no_first(self, source):
        body = self._extract_function_body(source, "_sync_type_text")
        assert ".first()" not in body, "pywinauto _sync_type_text should not use .first()"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
