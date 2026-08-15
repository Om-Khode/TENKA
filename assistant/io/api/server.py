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
from .pairing import PairCodeStore
from .runtime import StudioRuntime
from .vault import TokenVault

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"


def serve(runtime: StudioRuntime, vault: TokenVault, *, host: str = _HOST,
          port: int = 8787, origins: list[str],
          hub: EventHub | None = None,
          pair_store: PairCodeStore | None = None) -> asyncio.Task:
    """Start uvicorn as a task on the running loop. Cancel the task to stop.

    `hub` lets a caller (main.py) subscribe status_broadcaster to the exact
    EventHub instance this app will use, before the socket route ever sees a
    connection. Left unset -- every existing caller (tests, the exporter) --
    `create_app` builds its own, private to that one app, exactly as before.

    `pair_store` is the same shape of escape hatch, for the same reason:
    main.py threads its own module-level `PairCodeStore` through here so
    that `/studio pair` (a slash command, not a route) can mint into the
    exact object `POST /v1/pair` consults. Left unset, `create_app` builds a
    private store nothing outside that one app could ever reach.
    """
    if host != _HOST:
        raise ValueError("the Studio daemon binds loopback only in this milestone")

    # The one listener this milestone binds, declared by the port it binds
    # on. Everything a request is allowed to do is looked up from this map by
    # the port the connection was accepted on; a port that is not in it grants
    # nothing at all. A later transport adds its own entry here rather than
    # relying on any property of the traffic itself.
    app = create_app(runtime, vault, origins=origins, hub=hub,
                     listener_policies={port: "local"}, pair_store=pair_store)
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
    """Invalidate every device immediately; stop serving on the next tick.

    Rotating the instance secret is what makes this a kill switch rather than a
    pause: a token handed out before the switch is thrown never works again.
    That half of the contract is synchronous and holds the instant this
    function returns -- `vault.reset()` runs inline, no event-loop turn
    needed.

    The other half is not synchronous. `task.cancel()` only *schedules*
    cancellation; it does not run `_run()`'s except-CancelledError cleanup
    (the `await server.shutdown()` that actually closes the listening
    socket -- see `serve()` above). This function stays `-> None` rather
    than `async def` because a synchronous kill switch that a signal handler
    or a non-async call site can fire without ceremony is worth more than a
    guarantee this function alone cannot keep anyway: `shutdown()` never
    awaits anything, on any signature, without a caller who can await it in
    turn. Concretely, a caller that needs the port provably free right
    after calling this must give the event loop a turn first -- either
    `await task` (as `main.py`'s `_stop_studio_daemon` already does, before
    it ever calls this function) or an `await asyncio.sleep(0)` at minimum.
    Checking the port with no intervening `await` at all will see it still
    bound. `tests/test_api_server_lifecycle.py::
    test_shutdown_revokes_devices_and_eventually_frees_the_port` pins both
    halves against a real socket, in the correct order.
    """
    if task is not None:
        task.cancel()
    vault.reset()
    logger.info("[API] Studio daemon stopped and all devices revoked")
