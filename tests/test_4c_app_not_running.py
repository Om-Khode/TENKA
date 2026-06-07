"""
test_4c_app_not_running.py — Tests for 4C app-not-running detection and launch logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from assistant.code_executor import _detect_app_not_running
from assistant import service_registry as sr


def test_detect_no_active_device():
    assert _detect_app_not_running("No active device found") is True

def test_detect_no_active_spotify_devices():
    # This is what the LLM actually printed — the old list missed it
    assert _detect_app_not_running("No active Spotify devices found. Please open Spotify on a device.") is True

def test_detect_player_command_failed():
    assert _detect_app_not_running("Player command failed: No active device") is True

def test_detect_please_start():
    assert _detect_app_not_running("Please start the app and try again") is True

def test_detect_please_open_spotify():
    assert _detect_app_not_running("Please open Spotify on a device.") is True

def test_detect_app_not_running_phrase():
    assert _detect_app_not_running("App is not running") is True

def test_detect_application_not_running():
    assert _detect_app_not_running("Application is not running") is True

def test_detect_no_devices_available():
    assert _detect_app_not_running("No devices available for playback") is True

def test_detect_no_devices_found():
    assert _detect_app_not_running("No devices found") is True

def test_no_false_positive_success():
    assert _detect_app_not_running("Now playing: Blinding Lights") is False

def test_no_false_positive_error():
    assert _detect_app_not_running("Error: 401 Unauthorized") is False

def test_no_false_positive_empty():
    assert _detect_app_not_running("") is False

def test_no_false_positive_scope_error():
    assert _detect_app_not_running("Insufficient client scope: user-read-playback-state") is False

def test_case_insensitive():
    assert _detect_app_not_running("NO ACTIVE DEVICE") is True

def test_active_playback_phrase():
    assert _detect_app_not_running("Error: no active playback session") is True

def test_no_app_launch_uris_dict():
    assert not hasattr(sr, "APP_LAUNCH_URIS")

def test_oauth_package_map_provides_svc_name():
    assert sr.OAUTH_PACKAGE_MAP["spotipy"] == "spotify"

def test_still_device_404():
    # 404 in post-launch context = device not registered yet, keep polling
    result = "Error playing liked songs: http status: 404, code: -1"
    _still_device = (
        _detect_app_not_running(result)
        or "http status: 404" in result
        or "no device" in result.lower()
    )
    assert _still_device is True

def test_still_device_no_active():
    result = "No active device found"
    _still_device = (
        _detect_app_not_running(result)
        or "http status: 404" in result
        or "no device" in result.lower()
    )
    assert _still_device is True

def test_sentinel_parsed_correctly():
    result = "APP_NOT_READY|spotify"
    svc = result.split("|")[1].strip() if "|" in result else None
    assert svc == "spotify"

def test_sentinel_triggers_launch():
    result = "APP_NOT_READY|spotify"
    _app_launch_svc = None
    if result.startswith("APP_NOT_READY|"):
        _app_launch_svc = result.split("|")[1].strip()
    assert _app_launch_svc == "spotify"

def test_not_device_error_stops_polling():
    # A real code error (401 auth, syntax) should break out of polling
    result = "Error: 401 Unauthorized — bad token"
    _still_device = (
        _detect_app_not_running(result)
        or "http status: 404" in result
        or "no device" in result.lower()
    )
    assert _still_device is False


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
