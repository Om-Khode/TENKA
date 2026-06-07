import time
import pytest
from assistant.core.abort import AbortController, UserAborted


def test_initial_state_is_not_aborted():
    c = AbortController()
    assert c.is_aborted() is False
    assert c.reason is None


def test_request_abort_sets_flag_and_reason():
    c = AbortController()
    c.register_task("t1")
    c.request_abort("esc_hold")
    assert c.is_aborted() is True
    assert c.reason == "esc_hold"


def test_request_abort_is_idempotent():
    c = AbortController()
    c.register_task("t1")
    c.request_abort("first")
    c.request_abort("second")
    assert c.reason == "first"  # first reason wins (flag is idempotent)


def test_subscribers_fire_on_every_request_even_when_already_aborted():
    """Repeat ESC must keep firing subscribers (stop_streaming + STOPPED pill)
    even if the abort flag is stale from a prior task — otherwise TTS during
    a follow-up small-talk reply is uninterruptible."""
    c = AbortController()
    c.register_task("t1")
    received = []
    c.on_abort(lambda r: received.append(r))
    c.request_abort("first")
    c.request_abort("second")  # flag stays True, but subscriber must still fire
    c.request_abort("third")
    assert received == ["first", "second", "third"]
    assert c.reason == "first"  # flag/reason still idempotent


def test_reset_clears_flag_and_reason():
    c = AbortController()
    c.register_task("t1")
    c.request_abort("esc_hold")
    c.reset()
    assert c.is_aborted() is False
    assert c.reason is None


def test_subscribers_fire_on_abort():
    c = AbortController()
    c.register_task("t1")
    received = []
    c.on_abort(lambda reason: received.append(reason))
    c.request_abort("esc_hold")
    assert received == ["esc_hold"]


def test_subscriber_exception_does_not_break_siblings():
    c = AbortController()
    c.register_task("t1")
    called = []
    c.on_abort(lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
    c.on_abort(lambda r: called.append(r))
    c.request_abort("x")
    assert called == ["x"]


def test_off_task_esc_still_fires_subscribers():
    """Off-task suppression removed in livetest pass — ESC must reach
    TTS-stop subscribers even when no overlay-aware handler is active."""
    c = AbortController()
    called = []
    c.on_abort(lambda r: called.append(r))
    # No registered task, last_reset_ts >30s ago — previously suppressed.
    c._last_reset_ts = time.time() - 60
    c.request_abort("esc_hold")
    assert c.is_aborted() is True
    assert called == ["esc_hold"]


def test_off_task_esc_within_30s_also_fires():
    c = AbortController()
    c._last_reset_ts = time.time()
    c.request_abort("esc_hold")
    assert c.is_aborted() is True


def test_register_and_unregister_task():
    c = AbortController()
    c.register_task("t1")
    assert "t1" in c._tasks
    c.unregister_task("t1")
    assert "t1" not in c._tasks


def test_user_aborted_is_exception():
    assert issubclass(UserAborted, Exception)
