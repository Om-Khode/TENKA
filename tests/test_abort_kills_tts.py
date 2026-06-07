import asyncio
from unittest.mock import MagicMock
from assistant.core.abort import AbortController


def test_subscriber_invoked_when_abort_requested():
    """Confirms the on_abort hook fires synchronously when abort fires.
    main.py will register stop_streaming() as a subscriber — this test
    proves the hook surface works without actually running TTS."""
    c = AbortController()
    c.register_task("t")
    fake_stop = MagicMock()
    c.on_abort(lambda _reason: fake_stop())
    c.request_abort("esc_hold")
    fake_stop.assert_called_once()


def test_run_coroutine_threadsafe_pattern():
    """Verifies the cross-thread coroutine scheduling pattern main.py uses."""
    loop = asyncio.new_event_loop()
    called = []

    async def fake_stop_streaming():
        called.append(1)

    c = AbortController()
    c.register_task("t")
    c.on_abort(lambda _r: asyncio.run_coroutine_threadsafe(fake_stop_streaming(), loop))

    async def driver():
        c.request_abort("esc_hold")
        await asyncio.sleep(0.05)

    loop.run_until_complete(driver())
    loop.close()
    assert called == [1]
