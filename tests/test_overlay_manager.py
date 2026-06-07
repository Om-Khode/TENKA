# tests/test_overlay_manager.py
import time
from unittest.mock import patch, MagicMock
import pytest
from assistant.overlay_manager import OverlayManager


def test_start_spawns_subprocess_and_attaches_ipc():
    with patch("assistant.overlay_manager.subprocess.Popen") as popen, \
         patch("assistant.overlay_manager.status") as status:
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.poll.return_value = None  # alive
        popen.return_value = fake_proc
        m = OverlayManager()
        m.start()
        assert popen.called
        status.attach_ipc.assert_called_with(fake_proc.stdin)
        m._proc = None  # avoid stop side effects in teardown


def test_start_is_idempotent():
    with patch("assistant.overlay_manager.subprocess.Popen") as popen, \
         patch("assistant.overlay_manager.status"):
        popen.return_value = MagicMock(stdin=MagicMock(), poll=lambda: None)
        m = OverlayManager()
        m.start()
        m.start()
        assert popen.call_count == 1
        m._proc = None


def test_stop_sends_quit_then_terminates():
    with patch("assistant.overlay_manager.subprocess.Popen") as popen, \
         patch("assistant.overlay_manager.status"):
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.return_value = 0
        popen.return_value = fake_proc
        m = OverlayManager()
        m.start()
        m.stop()
        assert fake_proc.stdin.write.called  # quit cmd written
        fake_proc.wait.assert_called()


def test_stop_kills_if_wait_times_out():
    with patch("assistant.overlay_manager.subprocess.Popen") as popen, \
         patch("assistant.overlay_manager.status"):
        import subprocess as sp
        fake_proc = MagicMock()
        fake_proc.stdin = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.wait.side_effect = sp.TimeoutExpired(cmd="x", timeout=2)
        popen.return_value = fake_proc
        m = OverlayManager()
        m.start()
        m.stop()
        fake_proc.terminate.assert_called()


def test_respawn_budget_exhausted_enters_silent_mode():
    with patch("assistant.overlay_manager.subprocess.Popen") as popen, \
         patch("assistant.overlay_manager.status") as status:
        popen.return_value = MagicMock(stdin=MagicMock(), poll=lambda: None)
        m = OverlayManager()
        m.start()
        for _ in range(5):
            m.respawn()
        assert m._silent_mode is True
        # After silent mode, further respawns are no-ops
        prior_calls = popen.call_count
        m.respawn()
        assert popen.call_count == prior_calls


def test_pref_disabled_skips_spawn():
    from unittest.mock import patch
    with patch("assistant.overlay_manager.subprocess.Popen") as popen, \
         patch("assistant.overlay_manager._pref_enabled", return_value=False):
        m = OverlayManager()
        m.start()
        assert not popen.called
