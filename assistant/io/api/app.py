# assistant/io/api/app.py
"""The Studio daemon's FastAPI application.

Nothing here knows how to reach the assistant. `runtime` is injected by
main.py; `vault` decides who may ask. `hub` is optional -- main.py passes its
own EventHub when it needs to subscribe status_broadcaster to this exact app's
socket before the daemon starts; every other caller (tests, the OpenAPI
exporter) leaves it unset and gets a private one, scoped to that one app.

Layering: io/api — core + config only.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import datetime, timezone

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ...core.redact import redact_secrets
from .context import request_id_var
from .errors import to_http_exception
from .events import (
    EventHub, build_ack_frame, build_error_frame, build_pong_frame,
    build_status_frame, visible_frame,
)
from .pairing import PairCodeStore
from .policy import effective
from .raises import RaiseStore
from .routes import chat as chat_routes
from .routes import commands as command_routes
from .routes import devices as device_routes
from .routes import files as file_routes
from .routes import listener as listener_routes
from .routes import memory as memory_routes
from .routes import pairing as pairing_routes
from .routes import session as session_routes
from .routes import settings as settings_routes
from .routes import status as status_routes
from .routes import system as system_routes
from .routes import transports as transport_routes
from .runtime import StudioRuntime
from .security import (
    ANONYMOUS_DEVICE_ID,
    CSRF_HEADER,
    AuditEntry,
    AuthState,
    PublishedHosts,
    accepting_port,
    anonymous_key,
    cookie_credential,
    device_key,
    host_is_allowed,
    origin_is_known,
    policy_for_scope,
)
from .ui import UiBundle, mount_ui
from .vault import Capability, Device, TokenVault, VaultUnavailableError

logger = logging.getLogger(__name__)


class HostGate:
    """Refuse a request whose `Host` is not one of ours. 421, no body.

    DNS rebinding: a page served from `evil.example` re-resolves that same
    name to 127.0.0.1 and then talks to this daemon as same-origin -- every
    check the browser performs still passes, because from its point of view
    nothing changed. The one thing that does not change is the `Host` header,
    which still says `evil.example`, and which a page cannot alter. Rejecting
    unknown names is the standard defence.

    Middleware rather than a dependency, for three reasons that all point the
    same way: it must also cover the `/v1/events` WebSocket (which has no
    dependency chain), it must cover the unauthenticated static and pairing
    paths Tasks 7 and 10 add (which by definition never authenticate), and a
    gate that a route can forget to opt into is not a gate. Written as raw
    ASGI rather than `@app.middleware("http")` because Starlette's
    `BaseHTTPMiddleware` skips every non-`http` scope outright, which is
    exactly the scope that matters most here.

    It is a *rejection* gate only. It never selects a policy -- letting
    `Host` do that would be attacker-controlled input choosing its own
    permissions. It *reads* the policy name, which is a different thing: the
    name comes from the accepting port, and is used only to decide which
    names this socket is willing to answer to.

    Milestone 6b makes it port-aware, and that is KI-17's third and
    load-bearing layer -- see `host_is_allowed`, which holds the argument.
    The short version: `local` accepts loopback names only, so a tunnel
    pointed at the local port arrives carrying its public authority and is
    refused here, before anything else runs.
    """

    def __init__(self, app, *, published: PublishedHosts,
                 registry: dict[int, str]) -> None:
        self.app = app
        # The live collection, not a copy: a transport started later publishes
        # its public hostname onto it and this gate must see that immediately.
        self._published = published
        # The live registry, for the same reason and one stronger: 6b starts
        # and stops listeners while the process runs, so a port that stops
        # being a listener must stop being a scope for the names published
        # against it on the very next request. A copy taken here would keep
        # answering for a socket that no longer exists.
        self._registry = registry

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            host = next((value.decode("latin-1")
                         for key, value in scope.get("headers", ())
                         if key == b"host"), "")
            # The same derivation policy resolution uses, deliberately: if
            # these two ever disagreed about which listener a request arrived
            # on, the gate would be scoping names to one socket while the
            # ceiling was resolved from another.
            port = accepting_port(scope)
            policy_name = None if port is None else self._registry.get(port)
            if not host_is_allowed(host, self._published,
                                   port=port, policy_name=policy_name):
                if scope["type"] == "websocket":
                    # A close sent in answer to the handshake, before any
                    # accept: the connection is refused, not established and
                    # then torn down.
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 421,
                                "headers": [(b"content-length", b"0")]})
                    await send({"type": "http.response.body", "body": b""})
                return
        await self.app(scope, receive, send)


# The largest request body this API will read. Every write it serves is a
# small JSON document -- the biggest bounded field in `schemas.py` is
# `ChatRequest.text` at 8,000 characters -- so a megabyte is orders of
# magnitude above any honest call and still small enough that a flood of them
# cannot be a memory story. Deliberately one number for the whole API rather
# than a per-route budget: the check has to run *before* routing and before
# authentication, which is the entire point, and a limit that depends on
# knowing which route was hit is a limit that cannot.
MAX_BODY_BYTES = 1024 * 1024

# How stale the event socket's *inbound* re-verify may be. That check runs on
# every frame a client sends, junk included, and `TokenVault.verify()` re-reads
# devices.json from disk each call -- so unmemoised, the client sets this
# daemon's disk-read rate. One second because the hub's revalidate sweep is
# ~2s: the memo must not be the slowest thing deciding when a revoked socket
# dies, or it becomes the bound instead of the backstop. Only a *positive*
# answer is ever memoised, so a refusal is always freshly read.
_INBOUND_REVERIFY_TTL_SECONDS = 1.0

# How long the streaming refusal below is willing to keep reading a body it has
# already decided to throw away. The drain exists so a client that is genuinely
# mid-upload can finish writing and then read its 413 instead of seeing a
# broken connection -- but a client that stops sending must not be able to hold
# a server task open by simply going quiet, which is slowloris with extra
# steps. Five seconds matches uvicorn's own `timeout_keep_alive` default, and
# is far more than a loopback or LAN client needs to finish pushing the
# byte-bounded remainder.
DRAIN_TIMEOUT_SECONDS = 5.0


class BodyLimit:
    """Refuse an oversized request body. 413, before anything reads it.

    Nothing in this stack bounded a body. Pydantic's field constraints are
    checked *after* the whole body has been received off the socket and
    parsed as JSON, and uvicorn has no application-body limit to set, so a
    20 MiB POST to `/v1/chat` carrying no credential at all was buffered,
    parsed (~40 MiB of Python heap, measured) and only then answered 401.
    Authentication is not a bound on work when the work happens first.

    That matters most where there is no credential to check: `POST /v1/pair`
    is the one unauthenticated write in this API and it becomes publicly
    reachable in 6b. It matters second-most because this daemon shares the
    assistant's event loop -- buffering and parsing a large body is her time,
    not just the API's.

    Two halves, because a client picks which one applies, and they answer
    differently on purpose:

    - **A declared `Content-Length` over the cap is answered 413 immediately,
      without calling `receive()` at all.** Not one byte is read, and nothing
      is awaited. The first draft of this gate drained here too, out of the
      same courtesy the streaming half extends -- and that turned it into a
      slowloris primitive on the middleware added for availability: headers
      declaring two megabytes followed by *nothing* got no response and pinned
      a server task indefinitely, because uvicorn has no body-read timeout and
      the drain sat in `await receive()` for a body that never came. A ~150
      byte request should not be able to do that, least of all on the route
      that goes public in 6b. The client's own header already said the request
      is too big, so reading any of it is work done for a request that is
      already refused, and waiting for it is worse than useless.
    - **A chunked body (no `Content-Length`, or a lying one) is counted as it
      streams**, through a wrapped `receive`. The moment the running total
      crosses the cap the downstream app is cut off and the request is
      answered 413. Here the drain *does* run -- this client is demonstrably
      mid-upload, and letting it finish writing is what lets it read the 413
      rather than a reset connection -- but it is bounded twice over: by bytes
      (8x the cap) and by `DRAIN_TIMEOUT_SECONDS`, so a client that goes quiet
      mid-body is dropped rather than waited on.

    A header cannot be trusted to be the truth, so it is used only to refuse
    early, never to permit: a body arriving under a small (or absent)
    `Content-Length` is still counted on the way in.

    Raw ASGI for the same reason `HostGate` is: it must cover routes that
    never authenticate, it must not be something a route can forget to opt
    into, and `BaseHTTPMiddleware` would consume the body itself.
    """

    def __init__(self, app, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = next((value for key, value in scope.get("headers", ())
                         if key == b"content-length"), None)
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    # No `receive()`, no drain, no await on anything the
                    # client controls -- see the class docstring.
                    await self._refuse(send)
                    return
            except ValueError:
                # A malformed Content-Length is h11's problem, not this
                # gate's -- it will reject the framing itself. Fall through
                # to the streaming count, which does not trust the header
                # anyway.
                pass

        read = 0
        refused = False
        # Whether the body has already been read to its end. `_drain` must not
        # call `receive()` again once it has: an ASGI server is entitled to
        # block that call until the response is complete, and the response is
        # what this gate is on its way to send -- so a drain past the end of
        # the body is a deadlock, not a courtesy.
        body_done = False

        async def counting_receive():
            nonlocal read, refused, body_done
            message = await receive()
            if message["type"] == "http.request":
                read += len(message.get("body", b"") or b"")
                if not message.get("more_body", False):
                    body_done = True
                if read > self.max_bytes:
                    refused = True
                    # Ends the body as far as the app is concerned. The 413
                    # below is what the client actually gets; this only stops
                    # the downstream app from waiting on a body that is not
                    # coming.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            # Suppress whatever the app was about to answer once the body has
            # already blown the cap -- the honest answer is 413, not the 400
            # a truncated body would otherwise produce.
            if refused:
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            # A truncated body reaches the app as `http.disconnect`, which
            # Starlette turns into `ClientDisconnect` -- an exception raised
            # *because* this gate cut the body off, not a failure worth
            # surfacing as a 500. Anything raised while the body was still
            # within the cap is a real error and re-raises untouched.
            if not refused:
                raise
        if refused:
            if not body_done:
                await self._drain(receive)
            await self._refuse(send)

    async def _drain(self, receive) -> None:
        """Read and discard what is left of a body already being refused.

        Not politeness. A client still writing a body nobody is reading fills
        the socket buffer and blocks, so it never reaches the point of reading
        the answer -- it sees a broken connection and cannot tell "too large"
        from "the daemon fell over". Each chunk is discarded as it arrives, so
        the memory this gate exists to protect is never allocated; only
        transfer time is paid, and the sender pays it too.

        Bounded twice, because an unbounded drain is the same denial of
        service by another name. By bytes: past 8x the cap the connection is
        left to break, which is the honest answer to a client that will not
        stop. And by time: `DRAIN_TIMEOUT_SECONDS` caps how long a client can
        keep this task alive by trickling, or by simply going silent
        mid-body -- uvicorn has no body-read timeout of its own, so this is
        the only thing standing between a half-sent body and a pinned task.

        Only ever called when the body is known *not* to be finished. Calling
        `receive()` after the last chunk is a deadlock, not a courtesy: an
        ASGI server is entitled to block that call until the response is
        complete, and the response is what this gate is on its way to send.
        """
        async def _pull() -> None:
            drained = 0
            limit = self.max_bytes * 8
            while drained < limit:
                message = await receive()
                if message["type"] != "http.request":
                    return
                drained += len(message.get("body", b"") or b"")
                if not message.get("more_body", False):
                    return

        try:
            await asyncio.wait_for(_pull(), timeout=DRAIN_TIMEOUT_SECONDS)
        except Exception:
            # Timed out, disconnected, or the transport failed underneath.
            # (`CancelledError` is a `BaseException` and is deliberately not
            # caught -- a shutdown is not a slow client.)
            # The 413 is sent either way; whether it arrives is now the
            # client's problem, and it stopped being ours the moment it went
            # quiet holding a body it said it would send.
            pass

    async def _refuse(self, send) -> None:
        """Answer 413. Nothing is read and nothing is awaited on the client."""
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", b"27")]})
        await send({"type": "http.response.body",
                    "body": b'{"detail":"body too large"}'})


# Route modules are mounted with plain `include_router` -- the call every
# later route module's brief in this milestone instructs. An earlier
# revision of this file bypassed it with a bespoke `_mount()` helper to work
# around FastAPI 0.141.1 wrapping included routers in an internal
# `_IncludedRouter` node that isn't a flat `Route`, which made a naive
# `app.routes` sweep in the auth test see nothing. That was fixing the wrong
# layer: the sweep in tests/test_api_auth.py now walks `app.openapi()`'s
# resolved paths instead, which is correct regardless of how a router was
# registered. Keeping a bespoke mount here would only mean the next nine
# tasks' briefs and the code they edit no longer match.


def create_app(runtime: StudioRuntime, vault: TokenVault, *,
               origins: list[str], hub: EventHub | None = None,
               listener_policies: dict[int, str] | None = None,
               ui_bundle: UiBundle | None = None,
               pair_store: PairCodeStore | None = None,
               raises: RaiseStore | None = None,
               extension_digest: str | None = None) -> FastAPI:
    # Eager, once: instance_secret() is uncached on the environment-override
    # path, and a wrong-length TENKA_SECRET raises ValueError. Resolving it
    # here means a misconfigured override fails when the app is built, not
    # as a 500 on the first authenticated request. The ValueError's own
    # message names TENKA_SECRET, so the operator knows what to fix.
    vault.instance_secret()

    # The SHA-256 of the vendored `dom_query.js`, compared against the copy the
    # browser extension shipped. Passed IN rather than imported: this module is
    # `io/api`, which may reach `core/` and `config` and nothing else, and the
    # vendored file lives under `automation/`. `main.py` supplies it — the one
    # place allowed to see both tiers. `None` means the check has nothing to
    # compare against, and `evaluate_handshake` refuses on a digest mismatch, so
    # an unsupplied digest fails closed rather than skipping the comparison.
    app_extension_digest = extension_digest

    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        await instance.state.hub.start(instance.state.runtime)
        try:
            yield
        finally:
            await instance.state.hub.stop()

    app = FastAPI(
        title="TENKA Studio daemon",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.auth = AuthState(vault=vault)
    app.state.started_at = time.monotonic()
    app.state.hub = hub if hub is not None else EventHub()
    # Scoped to this one app, exactly like `hub` above and for the same
    # reason: a process-wide store would let a code minted by one app be
    # redeemed against another, which is nonsense in production (there is one
    # daemon) and a cross-test leak in the suite. A caller passes one in only
    # when it needs to hold the same store the routes do -- the pairing tests,
    # which mint without going through the loopback-only route.
    app.state.pair_store = pair_store if pair_store is not None else PairCodeStore()
    # Live ceiling raises, keyed on (device, policy). Threaded in for the same
    # reason `hub` and `pair_store` are, and with the same default: a caller
    # that has to hold the *same* store the routes read -- main.py, so that
    # `vault.reset()` can clear every raise the moment the kill switch is
    # thrown, and so `TransportManager` can drop a stopped transport's raises
    # -- passes one in. Left unset (tests, the OpenAPI exporter) the app builds
    # a private one scoped to itself, which is the only safe default: a
    # process-wide store would let a raise minted against one app be spent
    # against another.
    #
    # Never persisted, by construction rather than by convention -- see
    # `raises.py`'s module docstring for why a raise that survives a restart
    # would not be a raise.
    app.state.raises = raises if raises is not None else RaiseStore()
    # Port -> policy name. An empty registry denies everything, which is the
    # correct answer to "nobody said what this socket is": see
    # `policy.py`'s docstring for why the port, and nothing the client sends,
    # is what decides. 6a registers `{studio_api_port: "local"}` only.
    app.state.listener_policies = dict(listener_policies or {})
    # Hostnames a running transport has published, filled in by the transport
    # itself. Mutable and shared with `HostGate` on purpose: a tunnel's public
    # name is not knowable when the app is built.
    #
    # `PublishedHosts`, not a bare `set`, because the lifetime matters as much
    # as the membership: a hostname is trusted for as long as the transport
    # session that published it is running, and `unpublish(owner)` takes it
    # back. A set had no way to express the second half, so a `*.trycloudflare
    # .com` name -- which Cloudflare reassigns to somebody else once the
    # tunnel stops -- stayed an accepted `Host` and a trusted `Origin` for the
    # life of the process. See the class docstring for why that is a session
    # credential handed to a stranger rather than merely a stale entry.
    #
    # Every entry is scoped to the listener that published it (6b), so a name
    # one transport announced is not a trusted `Host` on another listener's
    # port -- and `local` accepts no published name at all, which is KI-17's
    # layer 3.
    published_hosts = PublishedHosts()
    app.state.published_hosts = published_hosts
    app.state.cors_origins = list(origins)

    # Added before HostGate below, which reverses their nesting: Starlette
    # builds the stack so that the *last* middleware added is the outermost.
    # HostGate has to sit outside CORS, or a rebinding page's preflight would
    # be answered by CORSMiddleware before the host was ever looked at.
    #
    # CORS is now a *development* affordance and nothing more. Studio is
    # served by this daemon from Milestone 6 on, so a real client is
    # same-origin and never needs it; the only cross-origin caller left is the
    # Next.js dev server on a developer's own loopback. So the allow-list is
    # dropped entirely unless a `local` listener is registered -- a tunnelled
    # deployment has no business advertising a laptop's dev origin as
    # trusted.
    #
    # `allow_credentials` stays False, and that is now load-bearing rather
    # than incidental: with it off, a browser will not attach the device
    # cookie to a cross-origin request nor let script read the response. The
    # cookie is usable same-origin only, which is the entire cross-site
    # attack surface of moving off `localStorage` closed at the browser.
    # An `Authorization` header the page sets itself is unaffected -- it is
    # not a "credential" in the CORS sense -- so the dev server's bearer flow
    # keeps working.
    serves_local = any(name == "local" for name in app.state.listener_policies.values())

    # Added FIRST, so it ends up INNERMOST of the three. `add_middleware`
    # inserts at index 0, so the last one added is the outermost -- which is
    # the whole reason the ordering above reads backwards, and the reason an
    # earlier revision of this block put `BodyLimit` between CORS and HostGate
    # while its comment claimed it sat inside CORS. It sat outside, and a
    # cross-origin 413 carried no `Access-Control-Allow-Origin` while a 401
    # under the same cap did.
    #
    # Inside is the side worth being on, so the ordering moved rather than the
    # comment: a response that skips CORS is reported by the browser as "could
    # not reach her" rather than as a refusal -- exactly the trap the
    # `BackupProviderError` handler further down was added for -- and a 413 a
    # developer cannot see is a 413 they will debug as a network fault. It
    # costs nothing: `CORSMiddleware` never touches a request body, so putting
    # it outside adds one cheap frame and no buffering. `BodyLimit` still runs
    # before routing, before authentication and before anything reads a byte,
    # which is the property that actually matters.
    app.add_middleware(BodyLimit)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if serves_local else [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", CSRF_HEADER],
    )

    # `app.state.listener_policies` itself, not a copy: 6b's transport manager
    # adds and removes entries as listeners start and stop, and this gate has
    # to see each change on the next request.
    app.add_middleware(HostGate, published=published_hosts,
                       registry=app.state.listener_policies)

    @app.middleware("http")
    async def audit_and_tag(request: Request, call_next):
        # Set before call_next(), not after: schemas.py's Meta reads this
        # contextvar via a default_factory when a route builds its Envelope,
        # deep inside the call this middleware is about to make. Set any
        # later and every route's response would carry the empty default.
        request_id = uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        try:
            try:
                response = await call_next(request)
            except Exception:
                # An exception reaching here means the app-wide handler below
                # (or a more specific one) did *not* convert it into a
                # response -- something truly unhandled. Audited before the
                # re-raise so the failures most worth logging are not also
                # the ones the audit log cannot show; ServerErrorMiddleware
                # (outside this middleware) still turns it into the actual
                # 500 sent on the wire.
                device = getattr(request.state, "device", None)
                request.app.state.auth.audit.record(AuditEntry(
                    at=datetime.now(timezone.utc).isoformat(),
                    device_id=(device.device_id if device
                               else ANONYMOUS_DEVICE_ID),
                    method=request.method,
                    path=redact_secrets(request.url.path),
                    outcome="500",
                ))
                raise
            device = getattr(request.state, "device", None)
            # `request.url.path`, deliberately, and not `scope["path"]` or
            # `raw_path`. `request.url` rebuilds the URL and re-parses it with
            # `urllib.parse.urlsplit`, which strips ASCII tab, CR and LF -- the
            # only reason a caller-chosen path is not CRLF injection into the
            # record an operator greps after an incident. The other two carry
            # the bytes through untouched. A passing test pins this so that
            # swapping it flips to failing.
            #
            # The length bound and the character class live in
            # `AuditLog.record`, at the store, so every call site gets them.
            request.app.state.auth.audit.record(AuditEntry(
                at=datetime.now(timezone.utc).isoformat(),
                device_id=(device.device_id if device
                           else ANONYMOUS_DEVICE_ID),
                method=request.method,
                path=redact_secrets(request.url.path),
                outcome=str(response.status_code),
            ))
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(404)
    async def not_found(_request: Request, _exc) -> JSONResponse:
        # Dispatches on status code alone (Starlette checks a registered
        # status-code handler before it ever consults a class-based one), so
        # this always answers the same fixed body regardless of what a route
        # passed as `detail=` on its `HTTPException(status_code=404, ...)`.
        # Routes under routes/ deliberately omit `detail=` on their 404s for
        # exactly this reason -- writing one there would read as live when it
        # is discarded here, unconditionally, every time.
        return JSONResponse(status_code=404, content={"error": "not found"})

    async def unhandled_error(_request: Request, exc: Exception) -> JSONResponse:
        # Catches anything a route or a runtime call raised that nobody
        # mapped locally -- a bare 500 with a traceback was the prior
        # behaviour for, e.g., int("not-a-number") from a bad memory item id,
        # or a RuntimeError from a backup precondition that was never met.
        http_exc = to_http_exception(exc)
        if http_exc.status_code == 404:
            content: dict = {"error": "not found"}
        else:
            content = {"detail": http_exc.detail}
        return JSONResponse(status_code=http_exc.status_code, content=content)

    # Registered per concrete type below, never on the bare `Exception`
    # class: Starlette's own `build_middleware_stack()` special-cases a
    # handler registered for `Exception` (or the literal status code 500) by
    # routing it to `ServerErrorMiddleware` instead of `ExceptionMiddleware` --
    # the outermost layer, entirely outside `audit_and_tag` above. That
    # middleware still calls the handler and sends its response over the
    # wire, but it unconditionally *re-raises the original exception
    # afterwards* (so a real ASGI server can log it) -- which is also exactly
    # what `starlette.testclient.TestClient`'s default
    # `raise_server_exceptions=True` re-raises into the calling test, instead
    # of ever handing back the 409/400/404 this handler built. Registering
    # each concrete type individually keeps them inside `ExceptionMiddleware`
    # -- the ordinary, non-500 path every other status code in this app
    # already takes -- where the response is simply returned, not
    # sent-then-re-raised. A handful of named types is deliberate: anything
    # NOT in this list is a genuinely *unexpected* failure, and stays a plain
    # 500 rather than being silently absorbed into a false sense of "every
    # exception is handled".
    from ..backup.provider import BackupProviderError

    for _exc_type in (
        ValueError, KeyError, PermissionError, FileNotFoundError,
        NotADirectoryError, RuntimeError, OSError,
        # A failed upload (expired provider token, no network, quota) was NOT
        # in this list, so it was a bare 500: a traceback in her console, and a
        # response that skipped the CORS middleware entirely, which the browser
        # then reported as "could not reach her" while she was running fine.
        BackupProviderError,
    ):
        app.add_exception_handler(_exc_type, unhandled_error)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler forwards Pydantic's exc.errors() verbatim,
        # and each error dict carries an "input" key holding the raw
        # offending value -- for a missing field, the whole body. Left alone,
        # a 422 becomes the one response guaranteed to print back whatever
        # was submitted: a recovery phrase, a chat message, a settings value,
        # a file path, unbounded by anything the field's own validator chose
        # to bound. Rebuilding the body from `loc` and `type` only -- never
        # `input`, never `msg` (which for some error types embeds the value
        # too) -- keeps "where validation failed" without ever repeating
        # "what was sent". This is app-wide: every route with a bounded field
        # inherits it, not just the ones the current tests happen to probe.
        errors = [
            {"loc": list(error.get("loc", [])), "type": error.get("type", "")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    app.include_router(status_routes.router, prefix="/v1")
    app.include_router(session_routes.router, prefix="/v1")
    app.include_router(memory_routes.router, prefix="/v1")
    app.include_router(settings_routes.router, prefix="/v1")
    app.include_router(file_routes.router, prefix="/v1")
    app.include_router(command_routes.router, prefix="/v1")
    app.include_router(chat_routes.router, prefix="/v1")
    app.include_router(system_routes.router, prefix="/v1")
    app.include_router(pairing_routes.router, prefix="/v1")
    app.include_router(listener_routes.router, prefix="/v1")
    app.include_router(device_routes.router, prefix="/v1")
    app.include_router(transport_routes.router, prefix="/v1")

    # ─── the event socket ───────────────────────────────────────────────
    # Not a route module like the ones above: `test_every_registered_route_
    # rejects_an_anonymous_call` in test_api_auth.py walks app.openapi(), and

    # ─── the browser extension's socket ──────────────────────────────────
    # A different door from `/v1/events` and deliberately so: this one carries
    # no capability at all. Its listener's ceiling is empty (`policy.py`), so
    # every HTTP route on this port already refuses; what is left is one socket
    # that speaks a vocabulary with no intents in it.
    #
    # It authenticates in the protocol's own `hello` frame rather than through
    # `authenticate()`. That is not a shortcut around the API's auth: the
    # extension is a target, not a principal — it never asks TENKA to run
    # anything — so it holds no device credential and there is nothing for the
    # capability machinery to decide. Two auth systems that never touch beat one
    # that half-shares a door.
    @app.websocket("/latch")
    async def latch(websocket: WebSocket) -> None:
        import json

        from .extension_ws import (
            LatchConnection, evaluate_handshake, is_occupied, read_token,
            register, unregister,
        )
        from ...core import latch_protocol as latch_proto

        policy = policy_for_scope(websocket.scope, app.state.listener_policies)
        if policy is None or policy.name != "extension":
            # Refused before `accept()`. Serving this socket on any other
            # listener would put a driver for the user's browser on a port whose
            # policy was written for something else — including, on `local`, one
            # that grants EXECUTE.
            await websocket.close(code=1008)
            return

        # Accept first, then decide: the verdict is a `reject` frame carrying a
        # code, and a client that is told PROTOCOL_MISMATCH stops retrying
        # forever. A bare TCP close cannot say which of the four checks failed,
        # so the extension would back off and retry an unfixable state until
        # someone reads a log.
        await websocket.accept()
        try:
            first = json.loads(await websocket.receive_text())
        except Exception:
            await websocket.close(code=1008)
            return

        verdict = evaluate_handshake(
            first,
            origin=websocket.headers.get("origin"),
            expected_token=read_token(),
            expected_digest=app_extension_digest or "",
            occupied=is_occupied(),
        )
        if not verdict.ok:
            logger.info(
                f"[LATCH] refused: {verdict.reason} (code={verdict.code})"
            )
            await websocket.send_text(json.dumps({
                "type": latch_proto.Frame.REJECT,
                "code": verdict.code,
                "reason": verdict.reason,
            }))
            await websocket.close(code=1008)
            return

        connection = LatchConnection(
            send_json=lambda frame: websocket.send_text(json.dumps(frame)),
            browser_name=str(first.get("browser", "other")),
            protocol_version=int(first.get("protocolVersion", 0)),
            extension_version=str(first.get("extensionVersion", "")),
        )
        register(connection)
        await websocket.send_text(json.dumps({"type": latch_proto.Frame.WELCOME}))
        logger.info(
            f"[LATCH] connected: browser={connection.browser_name!r} "
            f"version={connection.extension_version!r}"
        )

        try:
            while True:
                frame = json.loads(await websocket.receive_text())
                connection.handle_frame(frame)
        except WebSocketDisconnect:
            reason = "disconnected"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        else:
            reason = "closed"
        finally:
            # Unregister in `finally` so every exit path frees the slot. A
            # connection left registered after its socket died makes
            # `is_occupied()` refuse the extension's own reconnect, and nothing
            # ever clears it.
            unregister(connection, reason)
            logger.info(f"[LATCH] disconnected: {reason}")

    # FastAPI's schema builder has no representation for a WebSocketRoute --
    # it is skipped entirely, sweep and all. test_api_events.py's own
    # unauthenticated-connect and invalid-token tests are what actually guard
    # this route; they are not decorative, they are the only guard.
    @app.websocket("/v1/events")
    async def events(websocket: WebSocket) -> None:
        # The credential is the same httpOnly cookie every HTTP route reads,
        # which the browser attaches to a handshake on its own -- the reason
        # the browser's WebSocket API being unable to set headers is no
        # longer a problem worth routing around. What it *is* is the reason
        # for the Origin check below.
        #
        # The audit middleware above is HTTP-scope only (Starlette's
        # BaseHTTPMiddleware skips non-"http" ASGI scopes outright), so it
        # never sees this route at all -- the connect outcome (accepted or
        # closed) is recorded explicitly below instead, directly against the
        # same AuditLog, so a socket connection is not the one surface that
        # leaves no trace either way.
        auth: AuthState = app.state.auth
        # `anonymous_key`, not `websocket.client.host`: keyed exactly like
        # `authenticate()`, and for the reason spelled out there -- a client
        # address can be rewritten by a proxy header, and a budget a caller
        # can rotate is not a budget.
        source = anonymous_key(websocket.scope)
        device = None

        def _audit(outcome: str) -> None:
            auth.audit.record(AuditEntry(
                at=datetime.now(timezone.utc).isoformat(),
                device_id=(device.device_id if device
                           else ANONYMOUS_DEVICE_ID),
                method="WS", path="/v1/events", outcome=outcome,
            ))

        policy = policy_for_scope(websocket.scope, app.state.listener_policies)
        if policy is None:
            _audit("1008")
            await websocket.close(code=1008)
            return

        # ─── cross-site WebSocket hijacking ──────────────────────────────
        # Introduced by the cookie, not present before it. A WebSocket
        # handshake is not subject to CORS -- there is no preflight, and the
        # browser will complete it against any host from any page -- and it
        # attaches the cookie regardless of which page asked. So without this
        # check, any site the user happens to visit could open this socket
        # and read her status, her telemetry and (with CHAT_SEND) drive her.
        # The query-string token this replaces prevented it by accident: a
        # random page does not know the token. Validated before accept(),
        # because after accept() the socket already exists.
        #
        # A handshake with no Origin at all is a non-browser client. On
        # loopback that is curl, a script, or a test -- all of which already
        # have this machine, so refusing them buys nothing and costs the one
        # workflow that has to work. Over a tunnel it is anomalous: a cookie
        # is a browser artefact, so a remote client holding one but not
        # behaving like a browser is likelier a replay than a user.
        #
        # `is not None`, not truthiness, and for the reason spelled out in
        # `refuse_unknown_origin`: a present-but-blank `Origin` is malformed
        # input, and truthiness routed it into the *no Origin at all* branch --
        # which on `local` means accept. Absent stays absent; blank is now a
        # value that matches no front door.
        origin = websocket.headers.get("origin")
        if origin is not None:
            allowed = bool(origin.strip()) and origin_is_known(
                origin, app.state, accepting_port(websocket.scope), policy)
        else:
            allowed = policy.name == "local"
        if not allowed:
            _audit("1008")
            await websocket.close(code=1008)
            return

        # `cookie_credential`, not a second `websocket.cookies.get(...)` of
        # its own: two spellings of "the credential is this cookie" is how the
        # HTTP gate and the socket gate drift apart.
        token = cookie_credential(websocket)
        device = auth.vault.verify(token)
        if device is not None:
            # Same intersection `authenticate()` applies, for the same
            # reason: the two capability checks below must read the grants
            # this *listener* carries, not the ones the device was issued.
            # `raised` folded in the same way, or a live raise is invisible
            # over this socket while the identical raise widens the HTTP
            # path for the same device and policy (fix round 5, Important 2).
            raise_store = getattr(app.state, "raises", None)
            raised: frozenset[Capability] = frozenset()
            if raise_store is not None:
                raised = await asyncio.to_thread(
                    raise_store.capabilities_for, device.device_id,
                    policy.name)
            grants = effective(device.grants, policy, raised)
            if not grants:
                _audit("1008")
                await websocket.close(code=1008)
                return
            device = replace(device, grants=grants)

        # This socket has two faces with two different gates. The *stream* --
        # the connect-time status frame plus every later status/telemetry
        # push -- is read-only, the exact analogue of GET /v1/status and
        # GET /v1/telemetry, so it is gated on Capability.OBSERVE here, at the
        # handshake, same as those routes' `require(Capability.OBSERVE)`.
        # Verifying the token proves *a* device is on the other end; it says
        # nothing about what that device is allowed to see or do. Without
        # this, a FILES-only or SCREEN-only token -- issued for a narrower
        # purpose, never meant to hold OBSERVE -- could still stream
        # status/telemetry through this one socket, bypassing the same check
        # its HTTP twins enforce. Checked, audited and closed before
        # `accept()`: a capability failure is a rejection, not a connection
        # that then gets torn down.
        #
        # OBSERVE and not RECALL, deliberately: nothing this socket emits is
        # stored data -- it carries status and telemetry frames -- so gating
        # the handshake on the stored-data grant would deny a live view to a
        # device that holds no stored-data grant and needs none. A wall
        # display watching her work is exactly that device.
        #
        # The socket's one *write* verb -- the client-sent `{"type": "abort"}`
        # frame, the analogue of POST /v1/abort -- is gated separately, per
        # frame, on Capability.CHAT_SEND, further down this handler. It
        # cannot be folded into this same handshake check: refusing the
        # connection outright over a write permission a device may lack
        # would also take away the live status view a watching (read-only)
        # device is entitled to keep. So OBSERVE gets you in and lets you
        # watch; CHAT_SEND is checked again, separately, before any abort
        # actually reaches `runtime.chat.abort()`.
        if device is not None and Capability.OBSERVE not in device.grants:
            _audit("1008")
            await websocket.close(code=1008)
            return

        # An accept-then-close cycle still costs a handshake and a verify()
        # call, so it spends the same shared budget an HTTP request does --
        # a 256-bit token defeats brute force, but nothing upstream of this
        # check ever bounded how many connection attempts one source could
        # trigger per second. Keyed exactly like authenticate(): a verified
        # device spends its own budget (never a NAT-mate's), everything
        # else spends the source's, and only a *presented* wrong token
        # counts as a guess.
        if device is not None:
            budget_key = device_key(device)
            if not auth.limiter.check(budget_key):
                _audit("429")
                await websocket.close(code=1013)
                return
            auth.limiter.record_success(budget_key)
            # Same "when was this device last used?" column `authenticate()`
            # keeps for HTTP. A socket handshake is an authenticated request
            # too, and a phone that only ever holds this socket open would
            # otherwise sit at the top of the revoke list looking abandoned.
            #
            # Best-effort, same reasoning as `authenticate()`'s call to this:
            # the handshake has already verified the device and passed every
            # other gate above, so a transient devices.json lock -- on either
            # the read or the write half, `VaultUnavailableError` covers both
            # -- must not tear down a connection that would otherwise
            # succeed, just because a bookkeeping write couldn't land this
            # time.
            #
            # Off the loop, same as `authenticate()`'s call to this: the
            # write half spawns `icacls` synchronously, and a handshake is
            # not a reason for every other request this daemon is serving --
            # and the assistant sharing its loop -- to stop for tens of
            # milliseconds.
            try:
                await asyncio.to_thread(auth.vault.touch, device.device_id)
            except VaultUnavailableError as exc:
                logger.warning(f"[API] could not record last-seen for "
                               f"{device.device_id}: {exc}")
        else:
            if not auth.limiter.check(source):
                _audit("429")
                await websocket.close(code=1013)
                return
            if token:
                auth.limiter.record_failure(source)
            _audit("1008")
            await websocket.close(code=1008)
            return

        # ─── the socket is not a credential ──────────────────────────────
        # `device` above was resolved once, at the handshake. Everything this
        # daemon says about revocation depends on that answer never being
        # reused: `_COOKIE_MAX_AGE_SECONDS`'s docstring defends a one-year
        # cookie on the grounds that "a revoked device is refused on its very
        # next request no matter how long its cookie claims to live", and that
        # is true because `TokenVault.verify()` re-reads devices.json every
        # time. An accepted socket was the one exception -- it kept receiving
        # every hub frame and kept reaching `runtime.chat.abort()` through its
        # write verb, while HTTP from the same device was already answering
        # 401. A kill switch with an exception is not a kill switch.
        #
        # So the handshake's answer is re-derived, never cached, and on both
        # paths:
        #
        #   - Outbound, per frame, by the hub (`EventHub.attach(viewer=...)`
        #     calls this before every delivery). That makes revocation
        #     immediate for anything the socket would actually have received.
        #   - Outbound, on a timer, by the hub's revalidate sweep. A device
        #     that only *listens* never sends a frame, and a quiet assistant
        #     publishes none, so the per-frame check alone would cut off the
        #     busiest sockets first and a silent one never.
        #   - Inbound, per frame, right here in the receive loop -- unthrottled,
        #     because that is the write verb and any window at all on it is
        #     the hole.
        #
        # The cost is one `verify()` per frame, which is exactly what every
        # HTTP request already pays; the outbound frame rate is set by the
        # assistant's own work, not by the client, so it cannot be driven up
        # from the far end.
        # `[when, answer]` for the memo below, in a one-slot list because this
        # is a closure over a loop-free scope and `nonlocal` on a plain name
        # would be the only other way. One socket, one token, so the token is
        # not part of the key: it cannot change for the life of this handler.
        memo: list = [0.0, None]

        def _reverify() -> Device | None:
            """The device as it stands right now, narrowed by this listener,
            or `None` if this connection may no longer hold the socket.

            Never memoised. The hub calls this before every outbound frame and
            the write verb calls it before honouring an abort; both want the
            exact answer, and both are driven by something other than the
            client's frame rate, so neither is a flood risk.
            """
            fresh = auth.vault.verify(token)
            if fresh is None or fresh.device_id != device.device_id:
                answer = None
            else:
                # Same fold as the connect-time check above, and for the
                # same reason (fix round 5, Important 2): called directly,
                # not through `asyncio.to_thread`, matching `verify()` just
                # above it -- both are synchronous lookups this handler
                # already runs inline, per frame.
                raise_store = getattr(app.state, "raises", None)
                raised: frozenset[Capability] = (
                    raise_store.capabilities_for(fresh.device_id, policy.name)
                    if raise_store is not None else frozenset())
                grants = effective(fresh.grants, policy, raised)
                if not grants or Capability.OBSERVE not in grants:
                    answer = None
                else:
                    answer = replace(fresh, grants=grants)
            memo[0], memo[1] = time.monotonic(), answer
            return answer

        def _reverify_memoised() -> Device | None:
            """The same answer, at most `_INBOUND_REVERIFY_TTL_SECONDS` old.

            For the inbound pre-filter check alone. That check runs on *every*
            frame a client sends, including junk, and `TokenVault.verify()`
            re-reads and re-parses devices.json from disk each call -- so
            without this, a client that can open a socket can set this
            daemon's disk-read rate. The TTL is well under the hub's ~2s
            revalidate sweep, so the sweep stays the outer bound on how long a
            revoked socket can live, and the exact check still stands in front
            of the write verb.
            """
            if memo[1] is not None and (
                    time.monotonic() - memo[0] < _INBOUND_REVERIFY_TTL_SECONDS):
                return memo[1]
            return _reverify()

        def _viewer() -> frozenset[Capability] | None:
            """What the hub asks: what may this socket see now, or `None`."""
            current = _reverify()
            return current.grants if current is not None else None

        async def _safe_send(payload: dict) -> None:
            # Through `visible_frame`, like every frame the hub delivers.
            # This is the socket's *other* send path -- the connect frame and
            # the `error`/`ack` replies app.py builds itself -- and routing it
            # around the classifier would be the one exception to "classify
            # the field once, not per call site", which is exactly the rule
            # the `detail` leak came from breaking. `build_error_frame` even
            # emits a field named `detail`; its four values are all protocol
            # literals ("malformed frame", "unknown frame", "not permitted",
            # "unauthorized") and `_USER_CONTENT_FIELDS` says so by not
            # listing the `error` type -- but that is now a decision recorded
            # in the table, enforceable from one place, rather than a call
            # site that happens to be harmless today.
            #
            # `device.grants` is the live narrowed set: the receive loop below
            # reassigns `device` from `_reverify()` on every inbound frame, so
            # a grant narrowed mid-connection reaches this the same way it
            # reaches the hub.
            #
            # A socket that already broke between the failed receive/abort
            # and this reply must end cleanly through the finally below, not
            # on a second, unhandled exception raised by the reply itself.
            try:
                await websocket.send_json(visible_frame(payload, device.grants))
            except Exception:
                pass

        await websocket.accept()
        # `device_id` is what arms the per-device socket cap; without it only
        # the global one is live, and one device could hold every slot. The
        # hub answers `False` when either cap is already full -- refuse here
        # rather than proceeding, because an unattached socket would sit in
        # the receive loop forever receiving nothing and looking healthy.
        if not await app.state.hub.attach(websocket, _viewer,
                                          device_id=device.device_id,
                                          policy=policy):
            logger.info(f"[API] refused /v1/events, socket cap reached "
                        f"(device={device.device_id})")
            # Audited as a refusal: `_audit("accepted")` used to run above
            # `accept()`, so a refused socket was recorded as accepted and the
            # refusal nowhere. It now runs below, once the hub has the socket.
            #
            # `EventHub.attach` no longer closes on refusal, so this is the
            # only close and the 1013 really arrives. Suppressed anyway --
            # this `return` is above the `try:`, so a raise escapes to uvicorn.
            _audit("1013")
            with suppress(Exception):
                await websocket.close(code=1013)   # try again later
            return
        _audit("accepted")
        try:
            info = await app.state.runtime.system.status()
            # Same builder as every real status frame (`build_status_frame`,
            # events.py) -- so this first frame carries the identical key set
            # as the ones that follow it, with whatever isn't known yet
            # (`v`, `cursorFollows`, `step`, `tier`, `ts`) as `null` rather
            # than simply absent.
            # Logged because nothing else marks a Studio client attaching:
            # the daemon's own startup lines say it is listening, but a user
            # watching the console had no way to tell a connected dashboard
            # from a silent one. Device id only -- never the credential.
            logger.info(f"[API] Studio client attached to /v1/events "
                        f"(device={device.device_id})")
            # `_safe_send` puts this through `visible_frame` like every
            # hub-delivered frame. `detail` on a status frame is withheld from
            # a device without RECALL (see events.py for the argument), and
            # the connect frame is a status frame -- exempting it because
            # *this* producer happens to pass something harmless is how a
            # field ends up classified per call site instead of once. Nothing
            # is lost: the active model rides the OBSERVE-gated telemetry
            # frame and `GET /v1/status` already.
            await _safe_send(build_status_frame(phase="connected",
                                                 detail=info.active_model))
            while True:
                parsed = True
                frame = None
                try:
                    frame = await websocket.receive_json()
                except WebSocketDisconnect:
                    raise
                except Exception:
                    # Unparseable JSON, or anything else receive_json() can
                    # raise short of a disconnect: the frame is unusable, not
                    # the socket. One bad frame from a Studio build that
                    # doesn't match this daemon must not take down the one
                    # channel carrying status and telemetry (see events.py
                    # for what actually flows here today).
                    parsed = False

                # A frame arrived, so this connection is alive whether or not
                # the frame parsed. The hub otherwise judges liveness only by
                # what it manages to *send*, which would reap a device that
                # talks to her constantly while she happens to have nothing to
                # say. Recorded before the authorisation check below: this is
                # a statement about the socket, not about the device's rights.
                app.state.hub.note_activity(websocket)

                # ─── every inbound frame re-proves the device ─────────────
                # This check used to sit below the `type != "abort"` filter,
                # which made how promptly a revoked device lost its socket
                # depend on it choosing to send the one verb that mattered: a
                # client that only ever sent `ping`, or malformed JSON, held
                # the stream until the hub's next sweep noticed. Revocation is
                # not a thing a caller gets to schedule.
                #
                # Memoised, and only here. `TokenVault.verify()` re-reads and
                # re-parses devices.json from disk every call, and inbound
                # frame rate is set by the client -- so above the filter, an
                # unmemoised read turns a junk-frame flood into a disk-read
                # flood. The TTL is the window this check may be stale by, and
                # it is bounded on both sides: the hub's outbound per-frame
                # check and its ~2s revalidate sweep are the backstops, and the
                # write verb below takes the exact, unmemoised answer.
                current = _reverify_memoised()
                if current is None:
                    _audit("1008")
                    await _safe_send(build_error_frame("unauthorized"))
                    break
                device = current

                if not parsed:
                    await _safe_send(build_error_frame("malformed frame"))
                    continue
                if not isinstance(frame, dict):
                    await _safe_send(build_error_frame("unknown frame"))
                    continue
                # The keepalive's inbound half. `note_activity()` above has
                # already recorded that this connection is alive -- that is
                # true of any frame, junk included -- so all this branch adds
                # is an answer that is not an error. Without it, a client
                # keeping itself alive receives "unknown frame" per heartbeat,
                # which reads as a protocol mismatch and teaches client authors
                # to send junk instead. `pong` is accepted and answered with
                # nothing, so a client replying to the hub's own ping (see
                # `build_ping_frame` in events.py) is not answered back into a
                # loop. Neither is a write verb, so neither is capability
                # gated -- the re-verify above has already run.
                kind = frame.get("type")
                if kind == "ping":
                    await _safe_send(build_pong_frame())
                    continue
                if kind == "pong":
                    continue
                if kind != "abort":
                    await _safe_send(build_error_frame("unknown frame"))
                    continue
                # Re-verified per frame, against the vault, before the write
                # verb is honoured. The comment that used to sit here argued
                # the opposite -- that re-verification was wasted work
                # "since capability grants can't change mid-connection" --
                # and revocation is precisely the thing that falsifies that
                # premise: a device cut off a second ago kept driving
                # `runtime.chat.abort()` through this branch for as long as it
                # held the socket open. A closed door that only checks who you
                # are on the way in is a door that was never closed.
                #
                # Refused and *closed*, not merely refused: a device that no
                # longer verifies has no business holding the stream either,
                # and leaving it attached would mean the read half outlived
                # the write half's own check. The error frame goes first so
                # the client learns why rather than seeing an unexplained
                # drop.
                current = _reverify()
                if current is None:
                    _audit("1008")
                    await _safe_send(build_error_frame("unauthorized"))
                    break
                device = current
                # OBSERVE alone must not be enough here: see the handshake
                # comment above for why the write verb is gated separately
                # from the stream.
                if Capability.CHAT_SEND not in device.grants:
                    _audit("403")
                    await _safe_send(build_error_frame("not permitted"))
                    continue
                await app.state.runtime.chat.abort()
                await _safe_send(build_ack_frame("abort"))
        except WebSocketDisconnect:
            pass
        finally:
            await app.state.hub.detach(websocket)

    # ─── the Studio front-end ───────────────────────────────────────────
    # Last, deliberately: it registers a catch-all, and Starlette matches
    # routes in registration order, so every real API path above must already
    # be claimed by the time it is added. It also sits *inside* `HostGate` --
    # a public route is not an ungated one, and DNS rebinding does not stop
    # mattering because a page needs no credential. `mount_ui` does nothing
    # when there is no bundle, so an app built without one has no `/` at all
    # and answers 404 there, exactly as it did before this existed.
    mount_ui(app, ui_bundle)

    return app
