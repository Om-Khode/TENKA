"""
test_computer_task_p1_already_open.py — Tests for P1: "already open" hint injection.

Verifies that _execute_native_task injects an "ALREADY OPEN" hint into the
LLM prompt when a running window is detected, and omits it otherwise.
"""

import os
import re
import pytest

DA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assistant", "automation", "router.py"
)


@pytest.fixture
def source():
    with open(DA_PATH, "r", encoding="utf-8") as f:
        return f.read()


class TestAlreadyOpenHint:
    def test_execute_native_task_has_already_open_check(self, source):
        """Verify the source contains the 'ALREADY OPEN' hint text."""
        assert "ALREADY OPEN" in source, (
            "_execute_native_task must inject 'ALREADY OPEN' hint"
        )

    def test_hint_includes_do_not_open(self, source):
        """Verify the hint tells LLM not to include an 'open' step."""
        assert "Do NOT include an 'open' step" in source, (
            "Hint must instruct LLM to skip the 'open' step"
        )

    def test_hint_conditional_on_running_window(self, source):
        """Verify the hint is only added when running_window is detected."""
        # The pattern should be: if running_window: ... "ALREADY OPEN"
        in_execute = False
        found_conditional = False
        for line in source.splitlines():
            if "async def _execute_native_task" in line:
                in_execute = True
            elif in_execute and line.strip().startswith("async def "):
                break
            if in_execute and "if running_window:" in line:
                found_conditional = True
        assert found_conditional, "Hint must be conditional on running_window detection"

    def test_simple_open_shortcut_bypasses_llm(self, source):
        """Verify simple 'open X' commands still use the direct shortcut path."""
        in_execute = False
        found_shortcut = False
        for line in source.splitlines():
            if "async def _execute_native_task" in line:
                in_execute = True
            elif in_execute and line.strip().startswith("async def "):
                break
            if in_execute and "simple_match" in line and "open_app" not in line:
                found_shortcut = True
        assert found_shortcut, "Simple 'open X' shortcut must still exist"

    def test_available_elements_placeholder_used(self, source):
        """Verify the already-open hint is injected via the {available_elements} placeholder."""
        # The prompt format string should use {available_elements}
        assert "{available_elements}" in source, (
            "Prompt must use {available_elements} placeholder for context injection"
        )
        # And the already_open_hint should be combined with available_elements
        assert "already_open_hint" in source, (
            "The hint should be stored in already_open_hint variable"
        )


class TestWindowNamePinningF6:
    """Regression guard for manifest-based Session-5 Finding F6.

    The already_open_hint must explicitly pin the detected running_window
    so the LLM planner does not hallucinate a different window title
    (e.g., picking 'Spotify - Web Player: Music for everyone' from training
    data when the actual focused window is 'Spotify Premium'). When the LLM
    invents a wrong window string, the native pre_check at native.py
    rejects the step as 'focus drift', which propagates VERIFY_FAILED
    back to router.py:1422 and blocks automation-cache cache writes — breaking the manifest layer's
    promotion pipeline end-to-end.
    """

    def test_hint_interpolates_running_window(self, source):
        """The hint must include the running_window as an f-string substitution."""
        # The literal substitution pattern proving the window name reaches the prompt.
        assert '{running_window}' in source, (
            "already_open_hint must f-string-interpolate {running_window} so the "
            "LLM sees the exact detected window title"
        )

    def test_hint_explicitly_pins_window(self, source):
        """The hint must instruct the LLM that the window string is EXACT and non-negotiable."""
        # Strong language is necessary because Flash-Lite tends to substitute
        # common-in-training-data window names otherwise.
        assert "EXACTLY:" in source, (
            "Hint must mark the window title as EXACT — without this, the LLM "
            "treats the title as a hint and substitutes plausible alternatives"
        )
        assert "MUST use this EXACT string" in source, (
            "Hint must instruct the LLM that the exact string is mandatory in any 'window' param"
        )
        assert "Do NOT invent" in source, (
            "Hint must explicitly forbid invention of alternative window titles"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
