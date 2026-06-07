"""tests/test_asyncio_utils.py — Tests for core/asyncio_utils.py"""

import asyncio
import threading
import unittest

from assistant.core.asyncio_utils import set_main_loop, call_async, _main_loop


class TestSetMainLoop(unittest.TestCase):
    def tearDown(self):
        import assistant.core.asyncio_utils as mod
        mod._main_loop = None

    def test_stores_loop_reference(self):
        loop = asyncio.new_event_loop()
        try:
            set_main_loop(loop)
            import assistant.core.asyncio_utils as mod
            self.assertIs(mod._main_loop, loop)
        finally:
            loop.close()


class TestCallAsync(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self._thread.start()
        set_main_loop(self.loop)

    def tearDown(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()
        import assistant.core.asyncio_utils as mod
        mod._main_loop = None

    def test_runs_coroutine_and_returns_result(self):
        async def add(a, b):
            return a + b

        result = call_async(add(3, 4))
        self.assertEqual(result, 7)

    def test_propagates_exception(self):
        async def fail():
            raise ValueError("boom")

        with self.assertRaises(ValueError) as ctx:
            call_async(fail())
        self.assertIn("boom", str(ctx.exception))

    def test_raises_without_loop_set(self):
        import assistant.core.asyncio_utils as mod
        mod._main_loop = None

        async def noop():
            pass

        with self.assertRaises(RuntimeError):
            call_async(noop())

    def test_callable_from_background_thread(self):
        """Verify call_async works when called from a non-main thread."""
        results = []

        async def get_value():
            return 42

        def worker():
            results.append(call_async(get_value()))

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        self.assertEqual(results, [42])
