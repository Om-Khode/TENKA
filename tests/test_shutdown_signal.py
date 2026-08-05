"""Tests for core/shutdown_signal.py."""
from assistant.core import shutdown_signal


def teardown_function():
    shutdown_signal._reset_for_testing()


def test_not_requested_by_default():
    assert shutdown_signal.is_requested() is False


def test_request_sets_the_flag():
    shutdown_signal.request()
    assert shutdown_signal.is_requested() is True


def test_reset_clears_the_flag():
    shutdown_signal.request()
    shutdown_signal._reset_for_testing()
    assert shutdown_signal.is_requested() is False
