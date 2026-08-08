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

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ...core.redact import redact_secrets
from .context import request_id_var
from .errors import to_http_exception
from .events import EventHub, build_status_frame
from .routes import chat as chat_routes
from .routes import commands as command_routes
from .routes import files as file_routes
from .routes import memory as memory_routes
from .routes import settings as settings_routes
from .routes import status as status_routes
from .routes import system as system_routes
from .runtime import StudioRuntime
from .security import AuditEntry, AuthState, device_key
from .vault import Capability, TokenVault

logger = logging.getLogger(__name__)

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
               origins: list[str], hub: EventHub | None = None) -> FastAPI:
    # Eager, once: instance_secret() is uncached on the environment-override
    # path, and a wrong-length TENKA_SECRET raises ValueError. Resolving it
    # here means a misconfigured override fails when the app is built, not
    # as a 500 on the first authenticated request. The ValueError's own
    # message names TENKA_SECRET, so the operator knows what to fix.
    vault.instance_secret()

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

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
                    device_id=device.device_id if device else "-",
                    method=request.method,
                    path=redact_secrets(request.url.path),
                    outcome="500",
                ))
                raise
            device = getattr(request.state, "device", None)
            request.app.state.auth.audit.record(AuditEntry(
                at=datetime.now(timezone.utc).isoformat(),
                device_id=device.device_id if device else "-",
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
    for _exc_type in (
        ValueError, KeyError, PermissionError, FileNotFoundError,
        NotADirectoryError, RuntimeError, OSError,
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
    app.include_router(memory_routes.router, prefix="/v1")
    app.include_router(settings_routes.router, prefix="/v1")
    app.include_router(file_routes.router, prefix="/v1")
    app.include_router(command_routes.router, prefix="/v1")
    app.include_router(chat_routes.router, prefix="/v1")
    app.include_router(system_routes.router, prefix="/v1")

    # ─── the event socket ───────────────────────────────────────────────
    # Not a route module like the ones above: `test_every_registered_route_
    # rejects_an_anonymous_call` in test_api_auth.py walks app.openapi(), and
    # FastAPI's schema builder has no representation for a WebSocketRoute --
    # it is skipped entirely, sweep and all. test_api_events.py's own
    # unauthenticated-connect and invalid-token tests are what actually guard
    # this route; they are not decorative, they are the only guard.
    @app.websocket("/v1/events")
    async def events(websocket: WebSocket) -> None:
        # The browser's WebSocket API cannot set headers, so the token rides a
        # query parameter here and nowhere else -- test_api_auth.py's
        # test_the_query_string_exception_is_only_the_socket pins that this
        # stays the only occurrence. It is loopback-only in this milestone;
        # Milestone 6 replaces this with a subprotocol handshake before any
        # tunnel exists. The audit middleware above is HTTP-scope only
        # (Starlette's BaseHTTPMiddleware skips non-"http" ASGI scopes
        # outright), so it never sees this route at all -- the connect
        # outcome (accepted or closed) is recorded explicitly below instead,
        # directly against the same AuditLog, so a socket connection is no
        # longer the one surface that leaves no trace either way.
        auth: AuthState = app.state.auth
        source = websocket.client.host if websocket.client else "unknown"
        token = websocket.query_params.get("access_token", "")
        device = auth.vault.verify(token)

        def _audit(outcome: str) -> None:
            auth.audit.record(AuditEntry(
                at=datetime.now(timezone.utc).isoformat(),
                device_id=device.device_id if device else "-",
                method="WS", path="/v1/events", outcome=outcome,
            ))

        # Every HTTP analogue of what this socket serves -- GET /v1/status,
        # GET /v1/telemetry, POST /v1/abort -- requires Capability.CHAT via
        # `require(Capability.CHAT)`. Verifying the token proves *a* device
        # is on the other end; it says nothing about what that device is
        # allowed to see or do. Without this, a FILES-only or SCREEN-only
        # token -- issued for a narrower purpose, never meant to hold CHAT --
        # could stream status/telemetry and call `runtime.chat.abort()`
        # through this one socket, bypassing every one of those routes'
        # capability checks. Checked, audited and closed before `accept()`:
        # a capability failure is a rejection, not a connection that then
        # gets torn down.
        if device is not None and Capability.CHAT not in device.grants:
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

        _audit("accepted")

        async def _safe_send(payload: dict) -> None:
            # A socket that already broke between the failed receive/abort
            # and this reply must end cleanly through the finally below, not
            # on a second, unhandled exception raised by the reply itself.
            try:
                await websocket.send_json(payload)
            except Exception:
                pass

        await websocket.accept()
        await app.state.hub.attach(websocket)
        try:
            info = await app.state.runtime.system.status()
            # Same builder as every real status frame (`build_status_frame`,
            # events.py) -- so this first frame carries the identical key set
            # as the ones that follow it, with whatever isn't known yet
            # (`v`, `cursorFollows`, `step`, `tier`, `ts`) as `null` rather
            # than simply absent.
            await _safe_send(build_status_frame(phase="connected",
                                                 detail=info.active_model))
            while True:
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
                    await _safe_send({"type": "error", "detail": "malformed frame"})
                    continue
                if not isinstance(frame, dict) or frame.get("type") != "abort":
                    await _safe_send({"type": "error", "detail": "unknown frame"})
                    continue
                await app.state.runtime.chat.abort()
                await _safe_send({"type": "ack", "of": "abort"})
        except WebSocketDisconnect:
            pass
        finally:
            await app.state.hub.detach(websocket)

    return app
