"""Tests for S7c — screen and camera modules relocated to assistant/io/."""
import importlib


def test_screen_importable():
    mod = importlib.import_module("assistant.io.screen")
    assert hasattr(mod, "capture_screenshot_base64")


def test_camera_importable():
    mod = importlib.import_module("assistant.io.camera")
    assert hasattr(mod, "capture_camera_frame_base64")


def test_screen_no_assistant_level_import():
    """Old path assistant.screen must not resolve."""
    import importlib
    import sys
    if "assistant.screen" in sys.modules:
        del sys.modules["assistant.screen"]
    try:
        importlib.import_module("assistant.screen")
        assert False, "assistant.screen should not be importable after move"
    except (ModuleNotFoundError, ImportError):
        pass


def test_camera_no_assistant_level_import():
    """Old path assistant.camera must not resolve."""
    import importlib
    import sys
    if "assistant.camera" in sys.modules:
        del sys.modules["assistant.camera"]
    try:
        importlib.import_module("assistant.camera")
        assert False, "assistant.camera should not be importable after move"
    except (ModuleNotFoundError, ImportError):
        pass
