"""
test_computer_task_p4_intent_config.py — Tests for P4: Intent description clarification.

Verifies:
1. find_and_click description is narrowed to "ALREADY VISIBLE" on screen
2. computer_task mentions "open X" for desktop applications
3. Examples don't pair "Settings" with find_and_click
4. Examples pair "open settings" with computer_task
"""

import os
import pytest

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assistant", "config.py"
)


@pytest.fixture
def source():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return f.read()


class TestFindAndClickNarrowed:
    def test_mentions_already_visible(self, source):
        assert "ALREADY VISIBLE" in source, (
            "find_and_click must mention 'ALREADY VISIBLE' to restrict scope"
        )

    def test_mentions_not_for_launching(self, source):
        assert "NOT for opening" in source or "NOT for launching" in source, (
            "find_and_click must explicitly state it's not for launching apps"
        )

    def test_no_settings_as_find_and_click_example(self, source):
        """Verify 'Settings' is not used as a find_and_click example in the JSON examples."""
        # Look for the specific problematic pattern
        assert '"find_and_click", "params": {"text": "Settings"}' not in source, (
            "find_and_click examples must not use 'Settings' (teaches wrong routing)"
        )


class TestComputerTaskExpanded:
    def test_mentions_open_apps(self, source):
        """Verify computer_task description mentions opening desktop apps."""
        assert "open X" in source.lower() or "open settings" in source.lower(), (
            "computer_task must mention 'open X' for desktop applications"
        )

    def test_open_settings_example_exists(self, source):
        """Verify 'open settings' is an example for computer_task."""
        assert '"computer_task", "params": {"goal": "open settings"}' in source, (
            "Must have an example: computer_task with goal 'open settings'"
        )

    def test_computer_task_still_has_gui_description(self, source):
        """Verify the core computer_task description is preserved."""
        assert "visible application window" in source, (
            "computer_task must still describe GUI interaction"
        )


class TestExamplesConsistency:
    def test_find_and_click_example_is_screen_element(self, source):
        """Verify find_and_click examples use on-screen elements like Submit, Accept."""
        # Find the find_and_click example line
        for line in source.splitlines():
            if '"find_and_click"' in line and '"params"' in line and '"text"' in line:
                # The text should be something clearly on-screen, not an app name
                text_val = line.split('"text":')[1].strip().strip('"').strip("'").rstrip('"}\\n')
                assert text_val.strip('"') not in ("Settings", "Calculator", "Notepad"), (
                    f"find_and_click example text '{text_val}' should be a screen element, "
                    f"not an app name"
                )

    def test_computer_task_has_multiple_examples(self, source):
        """Verify computer_task has at least 2 example JSON objects."""
        count = source.count('"computer_task"')
        # At least 2: one in description examples, one in JSON examples
        assert count >= 2, (
            f"computer_task should appear at least twice (description + examples), found {count}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
