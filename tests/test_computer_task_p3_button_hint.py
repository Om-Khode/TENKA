"""
test_computer_task_p3_button_hint.py — Tests for P3: Button-based UI prompt hint.

Verifies that _APP_PLAN_PROMPT contains a generic rule about clicking
individual buttons instead of typing for apps with discrete button UIs.
"""

import os
import pytest

DA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assistant", "automation", "router.py"
)


@pytest.fixture
def source():
    with open(DA_PATH, "r", encoding="utf-8") as f:
        return f.read()


class TestButtonUiHint:
    def test_prompt_contains_button_click_rule(self, source):
        assert "click each button individually" in source or "click individual" in source, (
            "_APP_PLAN_PROMPT must contain rule about clicking buttons individually"
        )

    def test_prompt_mentions_calculator_pattern(self, source):
        """Verify the hint mentions digit/operator buttons as examples."""
        assert "Seven" in source or "Plus" in source or "Equals" in source, (
            "Hint should mention digit/operator button examples"
        )

    def test_hint_is_generic_not_app_specific(self, source):
        """Verify the hint doesn't hardcode 'Calculator' as a special case."""
        # The hint should say "calculators, number pads, dialogs" — generic categories
        in_prompt = False
        for line in source.splitlines():
            if "_APP_PLAN_PROMPT" in line:
                in_prompt = True
            if in_prompt and '"""' in line and line.strip() != '"""':
                break
        # There should be no "if Calculator" or "if calc" conditionals
        assert "if Calculator" not in source, "Hint must not hardcode Calculator checks"
        assert "if calc" not in source or "calculators" in source, (
            "Any 'calc' reference must be generic (e.g., 'calculators'), not a specific check"
        )

    def test_hint_in_rules_section(self, source):
        """Verify the hint is in the RULES section of _APP_PLAN_PROMPT."""
        in_prompt = False
        in_rules = False
        found_hint = False
        for line in source.splitlines():
            if "_APP_PLAN_PROMPT" in line:
                in_prompt = True
            if in_prompt and "RULES:" in line:
                in_rules = True
            if in_rules and ("click each button" in line or "click individual" in line):
                found_hint = True
            if in_prompt and "Task:" in line:
                break
        assert found_hint, "Button-click hint must be in the RULES section"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
