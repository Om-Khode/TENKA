"""Smoke test for active-app detection helper."""

from unittest.mock import patch

from assistant.automation.router import detect_active_app


def test_detect_active_app_returns_keys():
    with patch("assistant.automation.router._get_running_processes",
               return_value=["TestApp.exe"]):
        with patch("assistant.automation.router._get_foreground_window_title",
                   return_value="Test App Window"):
            with patch("assistant.automation.router._get_active_browser_url",
                       return_value=""):
                a = detect_active_app()
                assert a["process_names"] == ["TestApp.exe"]
                assert a["window_title"] == "Test App Window"
                assert a["active_url"] == ""
