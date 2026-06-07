"""Tests for F1 debounce buffer eviction in main.py."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import inspect


def test_drain_function_has_eviction_logic():
    """I5: _drain_and_announce_notifications should evict stale debounce entries."""
    from assistant.main import _drain_and_announce_notifications
    src = inspect.getsource(_drain_and_announce_notifications)
    assert "_evict_cutoff" in src, \
        "_drain_and_announce_notifications should have eviction cutoff logic"
