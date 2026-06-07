"""Confirms vision agent still aborts mid-loop after the lift to core/abort."""
from unittest.mock import patch
from assistant.core.abort import abort
import assistant.automation.vision.agent as agent


def test_check_abort_proxies_to_controller():
    abort.reset()
    abort.register_task("test")
    assert not agent._check_abort()  # back-compat shim returns is_aborted
    abort.request_abort("test")
    assert agent._check_abort()
    abort.reset()
    abort.unregister_task("test")


def test_reset_abort_proxies_to_controller():
    abort.register_task("test")
    abort.request_abort("x")
    agent.reset_abort()
    assert not abort.is_aborted()
    abort.unregister_task("test")


def test_start_stop_esc_monitor_proxies():
    # Should not raise — proxies to esc_monitor singleton
    agent.start_esc_monitor()
    agent.stop_esc_monitor()
