import time
import threading
from unittest.mock import patch, MagicMock
from assistant.io.esc_monitor import EscMonitor, _HOLD_THRESHOLD_SECS


def test_hold_under_threshold_does_not_fire():
    fake_abort = MagicMock()
    with patch("assistant.io.esc_monitor._is_esc_down") as is_down:
        # ESC held for 0.5s then released
        sequence = [True] * 10 + [False] * 5  # 10 * 50ms = 0.5s
        is_down.side_effect = lambda: sequence.pop(0) if sequence else False
        m = EscMonitor(abort_ref=fake_abort, poll_interval=0.05)
        m._tick_until(iterations=15)
    assert not fake_abort.request_abort.called


def test_hold_at_or_above_threshold_fires_once():
    fake_abort = MagicMock()
    with patch("assistant.io.esc_monitor._is_esc_down") as is_down:
        # ESC held for 1.5s
        sequence = [True] * 30
        is_down.side_effect = lambda: sequence.pop(0) if sequence else False
        m = EscMonitor(abort_ref=fake_abort, poll_interval=0.05)
        m._tick_until(iterations=30)
    fake_abort.request_abort.assert_called_once_with("esc_hold")


def test_release_resets_timer():
    fake_abort = MagicMock()
    with patch("assistant.io.esc_monitor._is_esc_down") as is_down:
        # 0.5s down, release, 0.5s down again — should NOT fire
        sequence = [True] * 10 + [False] * 3 + [True] * 10
        is_down.side_effect = lambda: sequence.pop(0) if sequence else False
        m = EscMonitor(abort_ref=fake_abort, poll_interval=0.05)
        m._tick_until(iterations=25)
    assert not fake_abort.request_abort.called


def test_start_returns_daemon_thread():
    fake_abort = MagicMock()
    m = EscMonitor(abort_ref=fake_abort)
    m.start()
    assert m._thread is not None
    assert m._thread.daemon is True
    assert m._thread.is_alive()
    m.stop()


def test_stop_joins_thread():
    fake_abort = MagicMock()
    m = EscMonitor(abort_ref=fake_abort)
    m.start()
    m.stop(timeout=2.0)
    assert not m._thread.is_alive()


def test_hold_threshold_constant():
    assert _HOLD_THRESHOLD_SECS == 1.0


def test_respawn_budget_allows_exactly_N_attempts():
    from assistant.io.esc_monitor import _RESPAWN_BUDGET
    from unittest.mock import MagicMock
    m = EscMonitor(abort_ref=MagicMock(), poll_interval=0.05)
    # Exactly _RESPAWN_BUDGET attempts must pass; (N+1)th must be rejected.
    for _ in range(_RESPAWN_BUDGET):
        assert m._can_respawn() is True
    assert m._can_respawn() is False
