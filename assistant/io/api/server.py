# assistant/io/api/server.py
"""Run the daemon inside the assistant's event loop -- on one socket, or many.

**One ASGI app, one `uvicorn.Server` per listener, one shared registry.**
Milestone 6b gives each transport its own socket on its own port, because the
daemon resolves what a request may do from the local port the connection was
accepted on -- the one piece of addressing a client cannot forge (`policy.py`
holds that argument in full). Four separate apps would fork `AuthState`, the
rate limiter, the event hub and the pair store for no benefit, and would give
four different answers to `GET /v1/devices`.

Every listener binds `127.0.0.1`, without exception. Separation is by **port**,
never by host: `cloudflared` and `tailscale funnel` both connect to this daemon
from loopback, so a tunnel is indistinguishable from a local caller by peer
address.

`serve()` binds and serves the `local` listener and returns its task, exactly
as it always has. Transports come later and from elsewhere (`transports/`),
through `bind_listener()` and `serve_socket()` -- the two halves this module
splits binding and serving into so that a caller can bind first, decide, and
serve second.
"""
from __future__ import annotations

import asyncio
import logging
import socket
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI

from .app import create_app
from .listeners import port_for
from .events import EventHub
from .pairing import PairCodeStore
from .raises import RaiseStore
from .runtime import StudioRuntime
from .ui import UiBundle
from .vault import TokenVault

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"

# The listen backlog. Small deliberately: this daemon serves one household's
# devices, and a deep backlog only buys a longer queue of connections nobody
# is going to get to any sooner.
_BACKLOG = 64


# ─── The handle main.py holds ────────────────────────────────────────────

@dataclass
class StudioListeners:
    """Everything a caller needs to reach the running daemon, keyed by policy.

    `serve()`'s return type is pinned by `tests/test_api_server_lifecycle.py`
    -- it returns the local listener's task and nothing else -- so the app and
    the per-listener machinery are reachable through this instead, which
    `serve()` leaves in `current_listeners()`. `TransportManager` (Task 9) is
    the other holder: starting a tunnel means binding a socket, writing one
    entry into `app.state.listener_policies`, and parking the task and the
    socket here so a stop can find them again.

    `tasks` and `sockets` are keyed by **policy name**, not by port. A port is
    derivable from the name (`listeners.port_for`) and the name is what a
    transport, a ceiling and a raise are all scoped by, so keying on it means
    those four things cannot drift apart.
    """

    app: FastAPI
    tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    sockets: dict[str, socket.socket] = field(default_factory=dict)


_listeners: StudioListeners | None = None


def current_listeners() -> StudioListeners | None:
    """The handle for the daemon `serve()` most recently started, or `None`.

    A function rather than a bare module attribute so that a caller reading it
    always gets the current value: `from .server import _listeners` would
    capture whatever was there at import time, which is `None` forever.

    **Lifetime is the caller's, deliberately.** This is set by `serve()` and
    is *not* cleared when the local task ends -- because a stop is exactly
    when a caller still needs the handle, to tear down whatever transports are
    running before the app goes away. `None` here carries **two** meanings,
    not one: "no daemon was ever started in this process", and "the most
    recent `serve()` attempt failed" (see the inline comment inside `serve()`,
    which clears `_listeners` before the first thing that can fail so a bind
    collision cannot leave a previous daemon's handle looking live). Neither
    of those is "no daemon is serving right now" for a daemon that *did*
    start successfully and is only now being stopped: whoever stops the
    daemon (main.py's `_stop_studio_daemon`, via `clear_listeners()` below) is
    the one that must clear it afterwards, the same obligation it already
    carries for `_studio_pair_store`. Do not decide a daemon is live from
    this being non-`None`; ask the task.
    """
    return _listeners


def clear_listeners() -> None:
    """Drop the handle a stopped daemon left behind. Idempotent.

    `serve()` already clears `_listeners` on its own failure paths (see its
    inline comment); this is the other half, for a daemon that started
    *successfully* and has since been stopped. `current_listeners()`'s
    docstring explains why that handle is left in place through the stop
    itself rather than cleared the instant the local task is cancelled --
    `TransportManager.stop_all()` still needs it. Nothing inside this module
    calls this function: tearing the daemon all the way down is main.py's job
    (`_stop_studio_daemon`), the same caller that already owns clearing
    `_studio_pair_store`, and for the same reason -- a handle left behind here
    would describe an app whose sockets are all closed as if it were still
    live.
    """
    global _listeners
    _listeners = None


def _refuse_a_second_primary() -> None:
    """Raise if a primary listener -- or any transport listener -- is already
    serving in this process.

    The failure mode is the one `serve_socket`'s docstring spends a paragraph
    on -- two listeners owning the app's lifespan means the `EventHub` is
    started twice and the first shutdown stops it for both -- so it is made
    unreachable here rather than merely discouraged. Liveness is asked of the
    task, never of the handle: `current_listeners()` deliberately outlives the
    daemon it describes, so a non-`None` handle whose tasks are all done is a
    stopped daemon and no obstacle to starting another.

    Checked across **every** entry in `tasks`, not only `"local"`. A restart
    that stopped the local task but left a transport task running (a
    `TransportManager.stop_all()` that was skipped, or one whose own teardown
    is still in flight) would pass a local-only check, and `serve()` would
    then overwrite `_listeners` with a brand-new `StudioListeners` -- losing
    the only handle to that transport's task and socket, the same
    orphaned-handle failure `current_listeners()`'s docstring already warns
    about. Refusing when *any* task is not done closes it at no cost: a
    caller that genuinely wants to restart still stops every transport first
    (`TransportManager.stop_all()`), same as it always had to.
    """
    existing = _listeners
    if existing is None:
        return
    # `cancelling()` as well as `done()`: a task that has been asked to stop is
    # not one that is still serving, and it may need a loop iteration or two to
    # notice. `serve()` binds the extension listener beside the primary and
    # cancels it from the primary's done-callback, so between "the primary
    # finished" and "the extension task is done" there is a window where the
    # daemon is stopping and nothing is serving. Reading only `done()` there
    # refuses a restart for a listener that is already on its way out.
    if any(not task.done() and not task.cancelling()
           for task in existing.tasks.values()):
        raise RuntimeError(
            "a Studio daemon is already serving in this process; stop it "
            "before starting another (two primary listeners would each run "
            "the app's lifespan, and the first to stop would stop the event "
            "hub for both)")


# ─── Binding, separately from serving ────────────────────────────────────

def bind_listener(port: int, host: str = _HOST) -> socket.socket:
    """A bound, listening socket on loopback. Raises `OSError` on a collision.

    Split from serving because a caller has work to do in between. A transport
    must know its port is genuinely its own *before* it spawns a tunnel at it,
    and it must be able to give the port back -- closing this socket -- if
    anything after the bind fails. Binding inside the serve task instead would
    put both of those on the far side of an `await`, in a task the caller can
    only cancel and hope.

    **`SO_REUSEADDR` is deliberately not set, and this is a Windows machine.**
    On Linux the option merely relaxes TIME_WAIT. On Windows it means
    something else entirely: a second process setting it can bind a port
    another process is *actively listening on*, and takes the connections --
    which is why Microsoft's own guidance is to use `SO_EXCLUSIVEADDRUSE`
    rather than this. A second TENKA answering the first one's requests under
    its own vault, its own devices and its own audit log is far worse than a
    refusal to start. So a bind collision fails loudly, and
    `_start_studio_daemon()` already catches it, logs it and leaves the
    assistant running without a daemon.

    The host argument exists so the refusal below is expressed once. There is
    no configuration that reaches it: every caller in this milestone takes the
    default.
    """
    if host != _HOST:
        raise ValueError("the Studio daemon binds loopback only")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.listen(_BACKLOG)
    except BaseException:
        # Including the collision this function exists to surface: the socket
        # object would otherwise be left holding a file descriptor nobody has
        # a reference to, once the exception has unwound past the caller.
        sock.close()
        raise
    return sock


def serve_socket(app: FastAPI, sock: socket.socket, *, name: str,
                 primary: bool = False) -> asyncio.Task:
    """Serve `app` on an already-bound socket. Cancel the task to stop.

    `primary` marks the one listener that owns this process's lifecycle, and it
    controls two things that must both be true of exactly one socket:

    - **The app's lifespan.** `create_app`'s lifespan starts and stops the
      `EventHub`. Uvicorn runs startup and shutdown per `Server`, so four
      servers would start the hub four times -- and the first one to stop would
      stop it for everybody, taking the event socket's stream away from every
      other listener. Non-primary listeners run `lifespan="off"`.
    - **Uvicorn's signal handlers.** `Server.serve()` installs its own SIGINT
      and SIGTERM handlers for the duration and restores whatever was there
      when it started. With one server that nests correctly. With four,
      starting and stopping at arbitrary times, it does not: a transport
      stopping would restore a handler belonging to a server that has itself
      already stopped, and Ctrl+C would then reach a dead `Server` object and
      do nothing at all. So only the primary listener captures signals; the
      others leave the process's handlers exactly as they found them.

    Defaults to `False`, so a transport listener added later is non-primary
    unless somebody deliberately says otherwise -- the safe direction, since
    the failure mode of a second primary is a hub that stops while three
    sockets are still serving. A second *live* primary is refused outright
    (`_refuse_a_second_primary`), not merely discouraged.

    **Stop the transports before the primary, always.** Cancelling the primary
    runs uvicorn's shutdown, which runs the app's lifespan shutdown, which
    calls `hub.stop()` -- for every listener at once, because there is one app
    and one hub. A transport listener still serving after that is serving a
    stopped hub: its `/v1/events` sockets attach to nothing and its HTTP routes
    answer against an app that has run its own teardown. The ordering is not
    enforceable from inside this function (it does not know what else is
    running); it belongs to whoever holds the `StudioListeners` handle, and it
    is `TransportManager.stop_all()` before the local task's `cancel()`.
    Neutralising `capture_signals` above makes that obligation sharper rather
    than softer: a transport listener will not stop on Ctrl+C on its own, so
    something has to cancel it.

    The uvicorn flags below are pinned identically on **every** listener, and
    two of them are security settings rather than preferences:

    `proxy_headers=False`, explicitly. Uvicorn's default is True with
    `forwarded_allow_ips="127.0.0.1"`, which installs `ProxyHeadersMiddleware`
    and lets it rewrite `scope["client"]` from the `X-Forwarded-For` header on
    any connection whose peer is loopback -- which is *every* tunnelled
    connection, because `cloudflared` and `tailscale funnel` both connect to
    this daemon from 127.0.0.1.

    Two things in this codebase are keyed on the client's identity: the
    anonymous rate-limit budget and the exponential auth lockout
    (`security.py`'s `authenticate()`), and both are documented as safe on the
    grounds that a tunnel collapses every remote caller onto one shared key.
    With the default in force that reasoning was simply wrong -- ten wrong
    token guesses under one `X-Forwarded-For` value lock that value out, and
    the next request under a different value is answered as a fresh, unmetered
    caller. The lockout and the window were both rotatable by a header, and
    `RateLimiter` grew one permanent entry per distinct value on the way.
    (Task 10's review missed this because `TestClient` speaks ASGI directly and
    never installs uvicorn's middleware at all; only a real `server.serve()`
    shows it.)

    Turning it off is the right direction rather than trusting the header more
    carefully: nothing in this daemon wants a client-supplied address. Policy
    comes from the accepting port, the `Host` gate is a rejection list, and the
    two budgets above are better off metering the one peer they can actually
    see. The cost is that a tunnel's traffic all meters as one caller -- which
    is exactly what the code already claims, and now true.
    `forwarded_allow_ips` is pinned to an empty list as well so that
    re-enabling the middleware by accident still trusts nobody.

    `log_config=None` keeps uvicorn out of the host application's logging. Its
    default config is applied through `logging.config.dictConfig`, which calls
    `logging.shutdown()` on every handler in the process -- including main.py's
    `debug.log` FileHandler. `FileHandler.close()` nulls `stream`, and `emit()`
    refuses to reopen a closed `mode="w"` handler (that guard exists so a
    reopen cannot truncate the file), so every record after this line was
    dropped in silence while the handler sat in `root.handlers` looking
    healthy. Every debug log since the daemon shipped ended mid-boot, a few
    lines before this call. Uvicorn's own records still reach root and format
    like the rest.
    """
    if primary:
        _refuse_a_second_primary()
    host, port = sock.getsockname()[:2]
    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False,
                            lifespan="on" if primary else "off",
                            log_config=None,
                            proxy_headers=False, forwarded_allow_ips=[])
    server = uvicorn.Server(config)
    if not primary:
        # See the docstring: `Server.serve()` wraps itself in
        # `capture_signals()`, which is correct for one server and actively
        # harmful for four. Replaced on the instance, which is the whole reach
        # of this -- the class, and therefore the primary listener, is
        # untouched.
        server.capture_signals = nullcontext

    async def _run() -> None:
        try:
            await server.serve(sockets=[sock])
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
                await server.shutdown(sockets=[sock])
            raise

    task = asyncio.create_task(_run(), name=name)

    def _release(_task: asyncio.Task) -> None:
        """Close the socket once this listener's task is over, however it ended.

        A done-callback rather than a `finally` inside `_run()`, and the
        difference is the whole point. The socket is bound by
        `bind_listener()` *before* this task exists, so it is not uvicorn's to
        close on a path where uvicorn never ran -- and there is such a path:
        `task.cancel()` before the loop has given the coroutine its first turn
        cancels it without ever entering the body, so no `try`/`finally`
        written inside `_run()` executes at all. That is not a contrived case;
        it is exactly what a transport does when a step after the bind fails
        and it tears the half-built listener down.

        Left alone, the port stays bound for the life of the process with
        nothing holding a reference to it. A done-callback fires whether the
        coroutine ran, raised, or was never started. Idempotent, because
        `socket.close()` on an already-closed socket is a no-op and uvicorn's
        own shutdown will usually have closed it first.
        """
        with suppress(OSError):
            sock.close()

    task.add_done_callback(_release)

    # No log line here, deliberately. `serve()` already announces the daemon,
    # and `TransportManager` announces a tunnel with the context that actually
    # matters (which provider, which public name); a second line from in here
    # would be the same event said twice, once without a name for it.
    return task


def serve(runtime: StudioRuntime, vault: TokenVault, *, host: str = _HOST,
          extension_digest: str | None = None,
          port: int = 8787, origins: list[str],
          hub: EventHub | None = None,
          ui_bundle: UiBundle | None = None,
          pair_store: PairCodeStore | None = None,
          raises: RaiseStore | None = None) -> asyncio.Task:
    """Start the `local` listener as a task on the running loop.

    Signature and return type are unchanged from Milestone 6a, deliberately:
    `tests/test_api_server_lifecycle.py` asserts on both directly, and
    main.py's whole startup sequence is written around "serve() returns the
    task, or raises". The multi-listener machinery sits *beside* it -- the app
    and every listener's task and socket are reachable through
    `current_listeners()`.

    `hub` lets a caller (main.py) subscribe status_broadcaster to the exact
    EventHub instance this app will use, before the socket route ever sees a
    connection. Left unset -- every existing caller (tests, the exporter) --
    `create_app` builds its own, private to that one app, exactly as before.

    `ui_bundle` is the second of these, and the one that has to be threaded
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

    `raises` is the fourth, added in 6b and defaulting to `None` so every
    existing caller stays valid untouched. It is the live record of ceiling
    raises that `authenticate()` reads and the admin raise route writes; a
    caller that needs to reach the same store the routes do -- to clear it
    with the kill switch, or to drop a stopped transport's raises -- hands one
    in.

    `app.state.transports` is deliberately NOT threaded through here.
    Transports start and stop long after the daemon does, so the manager that
    owns them is attached by main.py to the app this function built, through
    `current_listeners()`.
    """
    global _listeners
    # Refused before anything is torn down: if a daemon really is still
    # serving, its handle below is the live one and must survive this call.
    _refuse_a_second_primary()
    # Cleared before the first thing that can fail, and that ordering is the
    # whole point. Every failure path out of this function -- the host check
    # below, a bind collision, a `serve_socket` that raised -- returns with
    # `_start_studio_daemon()` logging a warning and handing back `None`, and
    # a handle left installed from a *previous* daemon would then be a live
    # answer from `current_listeners()` describing an app whose sockets are
    # closed. A port collision is the likeliest failure here and takes exactly
    # that path. `None` has to mean "nothing this function started is
    # reachable", or the contract in `current_listeners()` is a lie on the
    # only paths where it matters.
    _listeners = None

    if host != _HOST:
        raise ValueError("the Studio daemon binds loopback only in this milestone")

    # The listener this function binds, declared by the port it binds on.
    # Everything a request is allowed to do is looked up from this map by the
    # port the connection was accepted on; a port that is not in it grants
    # nothing at all. A transport adds its own entry to this same dict as it
    # starts -- `HostGate` and `authenticate()` both hold the dict itself, not
    # a copy, so an entry added or dropped is live on the very next request --
    # rather than relying on any property of the traffic itself.
    # The extension listener is bound here beside `local`, not started on demand
    # by `TransportManager` the way the tunnels are. The difference is what
    # dials which way: a tunnel is something TENKA reaches out and starts, while
    # the browser extension dials IN and retries on its own alarm. A socket that
    # only appears once somebody asks for it is a socket the extension has
    # already been failing to reach, quietly, since the browser started.
    extension_port = port_for("extension", port)

    # The digest of the vendored dom_query.js, compared against the copy the
    # extension shipped. Passed IN by the caller: `io/api` may reach `core/` and
    # `config` and nothing else, and the vendored file lives under
    # `automation/`. `main.py` is the one place allowed to see both tiers, so it
    # supplies this; a lazy import here would still be an import, and
    # import-linter counts it.
    #
    # `None` refuses every handshake rather than skipping the comparison, so a
    # caller that forgets fails closed and loudly.
    app = create_app(runtime, vault, origins=origins, hub=hub,
                     listener_policies={port: "local",
                                        extension_port: "extension"},
                     ui_bundle=ui_bundle,
                     pair_store=pair_store, raises=raises,
                     extension_digest=extension_digest)

    # Bound here, synchronously, before any task exists. A port collision is
    # then raised out of this call -- where `_start_studio_daemon()`'s own
    # try/except already turns it into one warning and a running assistant --
    # instead of surfacing later as an exception inside a task somebody has to
    # remember to retrieve.
    sock = bind_listener(port, host)
    try:
        task = serve_socket(app, sock, name="studio-api", primary=True)
    except BaseException:
        # Nothing between the bind and the task takes ownership of the socket,
        # so a raise here (a `uvicorn.Config` this build rejects, a loop that
        # is not running) would leave the port bound with no task, no handle
        # and no reference to close it -- and `_start_studio_daemon()` would
        # swallow the exception into one warning, so the next attempt would
        # then fail on a collision with nothing.
        sock.close()
        raise

    # The extension socket is bound after the primary one and is NOT allowed to
    # take the daemon down with it: a browser driver failing to bind must not
    # stop TENKA from answering, so a collision here is one warning and a tier
    # that falls back to the bundled browser.
    extension_task = None
    extension_sock = None
    try:
        extension_sock = bind_listener(extension_port, host)
        extension_task = serve_socket(app, extension_sock, name="latch", primary=False)
    except BaseException as e:
        if extension_sock is not None:
            extension_sock.close()
            extension_sock = None
        extension_task = None
        logger.warning(
            f"[API] could not bind the browser-extension listener on "
            f"{host}:{extension_port} ({type(e).__name__}: {e}); browser tasks "
            f"will use the bundled browser"
        )

    tasks = {"local": task}
    socks = {"local": sock}
    if extension_task is not None and extension_sock is not None:
        tasks["extension"] = extension_task
        socks["extension"] = extension_sock

        # The extension listener dies with the primary, without anybody having
        # to remember.
        #
        # `serve()` returns the primary task and nothing else, and every caller
        # -- `main._stop_studio_daemon` included -- stops the daemon by
        # cancelling exactly that. A second task started in here and left to
        # outlive it would keep its socket bound after a stop, so the next
        # `serve()` would refuse with "a daemon is already serving" and the port
        # would still be taken. That is the orphaned-handle failure this
        # module's docstrings warn about three separate times; the fix is not to
        # warn a fourth.
        #
        # Transports are the other case and are deliberately NOT handled this
        # way: `TransportManager` owns them, they are started on request, and
        # `_stop_studio_daemon` stops them explicitly before cancelling the
        # primary. This one nobody asked for, so this is where it is buried.
        def _stop_extension(_done_primary, _t=extension_task, _s=extension_sock):
            if not _t.done():
                _t.cancel()
            try:
                _s.close()
            except OSError:
                pass

        task.add_done_callback(_stop_extension)

    _listeners = StudioListeners(app=app, tasks=tasks, sockets=socks)

    logger.info(f"[API] Studio daemon listening on http://{host}:{port}")
    if extension_task is not None:
        logger.info(f"[API] browser extension listener on ws://{host}:{extension_port}/latch")
    return task


def shutdown(task: asyncio.Task | None, vault: TokenVault, *,
            raises: RaiseStore) -> None:
    """Invalidate every device immediately; stop serving on the next tick.

    Rotating the instance secret is what makes this a kill switch rather than a
    pause: a token handed out before the switch is thrown never works again.
    That half of the contract is synchronous and holds the instant this
    function returns -- `vault.reset()` runs inline, no event-loop turn
    needed.

    The other half is not synchronous. `task.cancel()` only *schedules*
    cancellation; it does not run `_run()`'s except-CancelledError cleanup
    (the `await server.shutdown()` that actually closes the listening
    socket -- see `serve_socket()` above). This function stays `-> None` rather
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

    **This covers the local task and nothing else, deliberately, and that
    makes it a partial stop from Milestone 6b on.** It takes one task because
    it is handed one task -- there is no handle here to a transport, and
    inventing one would put tunnel teardown (a `tailscale serve` mapping to
    un-serve, a subprocess to reap, a public hostname to unpublish) inside a
    synchronous function whose whole value is that a signal handler can call
    it without ceremony. So a transport listener's socket stays bound after
    this returns, and with uvicorn's signal capture neutralised on those
    listeners (see `serve_socket`) a Ctrl+C does not reach them either.

    The security half still holds regardless: `vault.reset()` rotates the
    instance secret, so every device is refused on every listener immediately,
    including the ones still bound. What does not hold on its own is this
    module's own standard -- *a kill switch that revokes every token but never
    releases its own socket has only paused, not stopped* -- so the caller owes
    the rest, in this order: **`TransportManager.stop_all()` first, then this.**
    The other way round stops the event hub (the local listener owns the app's
    lifespan) while the transports are still serving against it.

    `raises` drops every ceiling raise alongside the device revocation --
    spec §3.3's fifth drop condition: `vault.reset()` already revokes every
    device, and a raise surviving it would be absurd. `RaiseStore.clear()` is
    itself synchronous (a lock, then a dict clear), so it costs this function
    nothing to call inline, the same way `vault.reset()` already does.

    **Required, not defaulted, and keyword-only.** `hub`, `pair_store` and
    `ui_bundle` on `serve()` default to `None` because their absence
    substitutes something harmless -- a private object nothing outside that
    app can reach. A missing `raises` here is not harmless: it silently skips
    a security action the kill switch is specifically supposed to take,
    every call site would still type-check, and the omission would look
    exactly like "no raises were live" instead of "nobody asked to clear
    them." A parameter that can be forgotten will be forgotten, so this one
    cannot be -- a caller with no `RaiseStore` in hand must go get one
    (`current_listeners().app.state.raises`, if a daemon is running) rather
    than have this function silently do less than a kill switch promises.
    """
    if task is not None:
        task.cancel()
    vault.reset()
    raises.clear()
    logger.info("[API] Studio daemon stopped and all devices revoked")
