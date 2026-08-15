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
from .ui import UiBundle
from .vault import TokenVault

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"


def serve(runtime: StudioRuntime, vault: TokenVault, *, host: str = _HOST,
          port: int = 8787, origins: list[str],
          hub: EventHub | None = None,
          ui_bundle: UiBundle | None = None,
          pair_store: PairCodeStore | None = None) -> asyncio.Task:
    """Start uvicorn as a task on the running loop. Cancel the task to stop.

    `hub` lets a caller (main.py) subscribe status_broadcaster to the exact
    EventHub instance this app will use, before the socket route ever sees a
    connection. Left unset -- every existing caller (tests, the exporter) --
    `create_app` builds its own, private to that one app, exactly as before.

    `ui_bundle` is the third of these, and the one that has to be threaded
    rather than resolved here: deciding *which* bundle means reading
    `studio_ui_path`, and nothing under `io/api` may import `config` (see the
    closing comment in ui.py). So main.py resolves it and hands it in, exactly
    as it already does for `origins`. Left unset -- tests, the exporter -- the
    app mounts no UI route at all and is otherwise unchanged.

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
                     listener_policies={port: "local"}, ui_bundle=ui_bundle,
                     pair_store=pair_store)
    # `proxy_headers=False`, explicitly, and it is a security setting rather
    # than a preference. Uvicorn's default is True with
    # `forwarded_allow_ips="127.0.0.1"`, which installs
    # `ProxyHeadersMiddleware` and lets it rewrite `scope["client"]` from the
    # `X-Forwarded-For` header on any connection whose peer is loopback --
    # which is *every* tunnelled connection, because `cloudflared` and
    # `tailscale funnel` both connect to this daemon from 127.0.0.1.
    #
    # Two things in this codebase are keyed on `request.client.host`: the
    # anonymous rate-limit budget and the exponential auth lockout
    # (`security.py`'s `authenticate()`), and both are documented as safe on
    # the grounds that a tunnel collapses every remote caller onto one shared
    # key. With the default in force that reasoning was simply wrong -- ten
    # wrong-token guesses under one `X-Forwarded-For` value lock that value
    # out, and the next request under a different value is answered as a
    # fresh, unmetered caller. The lockout and the window were both rotatable
    # by a header, and `RateLimiter` grew one permanent entry per distinct
    # value on the way. (Task 10's review missed this because `TestClient`
    # speaks ASGI directly and never installs uvicorn's middleware at all;
    # only a real `server.serve()` shows it.)
    #
    # Turning it off is the right direction rather than trusting the header
    # more carefully: nothing in this daemon wants a client-supplied address.
    # Policy comes from the accepting port, the `Host` gate is a rejection
    # list, and the two budgets above are better off metering the one peer
    # they can actually see. The cost is that a tunnel's traffic all meters
    # as one caller -- which is exactly what the code already claims, and now
    # true. `forwarded_allow_ips` is pinned to an empty list as well so that
    # re-enabling the middleware by accident still trusts nobody.
    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False, lifespan="on",
                            proxy_headers=False, forwarded_allow_ips=[])
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
