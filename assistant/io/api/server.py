# assistant/io/api/server.py
"""Run the daemon inside the assistant's event loop.

Loopback only. Milestone 6 adds transports; until then the only way in is from
this machine.
"""
from __future__ import annotations

import asyncio
import logging

import uvicorn

from .app import create_app
from .events import EventHub
from .runtime import StudioRuntime
from .vault import TokenVault

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"


def serve(runtime: StudioRuntime, vault: TokenVault, *, host: str = _HOST,
          port: int = 8787, origins: list[str],
          hub: EventHub | None = None) -> asyncio.Task:
    """Start uvicorn as a task on the running loop. Cancel the task to stop.

    `hub` lets a caller (main.py) subscribe status_broadcaster to the exact
    EventHub instance this app will use, before the socket route ever sees a
    connection. Left unset -- every existing caller (tests, the exporter) --
    `create_app` builds its own, private to that one app, exactly as before.
    """
    if host != _HOST:
        raise ValueError("the Studio daemon binds loopback only in this milestone")

    app = create_app(runtime, vault, origins=origins, hub=hub)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False, lifespan="on")
    server = uvicorn.Server(config)

    async def _run() -> None:
        try:
            await server.serve()
        except asyncio.CancelledError:
            server.should_exit = True
            # uvicorn's own main_loop() awaits asyncio.sleep(0.1) with no
            # try/finally around it, so a bare task.cancel() interrupts that
            # sleep and unwinds straight out of _serve() -- skipping the
            # `await self.shutdown(...)` call that closes self.servers.
            # Left alone, the port stays bound at the OS level even though
            # this task is done: a kill switch that revokes every token but
            # never releases its own socket has only paused, not stopped.
            # Calling shutdown() explicitly here (only if startup ever
            # completed -- a task cancelled before that has nothing to
            # close) is what actually frees the port before this coroutine
            # finishes unwinding.
            if server.started:
                await server.shutdown()
            raise

    logger.info(f"[API] Studio daemon listening on http://{host}:{port}")
    return asyncio.create_task(_run(), name="studio-api")


def shutdown(task: asyncio.Task | None, vault: TokenVault) -> None:
    """Stop serving and invalidate every device.

    Rotating the instance secret is what makes this a kill switch rather than a
    pause: a token handed out before the switch is thrown never works again.
    """
    if task is not None:
        task.cancel()
    vault.reset()
    logger.info("[API] Studio daemon stopped and all devices revoked")
