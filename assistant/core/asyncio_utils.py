"""
core/asyncio_utils.py — Run async coroutines from synchronous background threads.

Background threads (proactive analyzer, reflection engine) need to call async
LLM functions. Instead of spinning up throwaway event loops, they schedule work
on the main loop via run_coroutine_threadsafe and block on the result.
"""

import asyncio
import logging
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store a reference to the main event loop. Call once from main.py."""
    global _main_loop
    _main_loop = loop
    logger.debug("[ASYNC] Main event loop registered")


def call_async(coro) -> T:
    """
    Schedule a coroutine on the main event loop and block until it completes.

    Use from synchronous background threads that need to call async code.
    Raises RuntimeError if set_main_loop() hasn't been called.
    """
    if _main_loop is None or _main_loop.is_closed():
        raise RuntimeError(
            "Main event loop not set. Call set_main_loop() from main.py first."
        )
    future = asyncio.run_coroutine_threadsafe(coro, _main_loop)
    return future.result()
