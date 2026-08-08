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
from .events import EventHub
from .routes import chat as chat_routes
from .routes import commands as command_routes
from .routes import files as file_routes
from .routes import memory as memory_routes
from .routes import settings as settings_routes
from .routes import status as status_routes
from .routes import system as system_routes
from .runtime import StudioRuntime
from .security import AuditEntry, AuthState
from .vault import TokenVault

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
        request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
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

    @app.exception_handler(404)
    async def not_found(_request: Request, _exc) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": "not found"})

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
        # tunnel exists. The audit middleware below is HTTP-scope only
        # (Starlette's BaseHTTPMiddleware skips non-"http" ASGI scopes
        # outright), so this token never reaches request.url.path or the
        # audit log -- there is simply no audit entry for this route at all
        # yet, success or failure.
        token = websocket.query_params.get("access_token", "")
        device = app.state.auth.vault.verify(token)
        if device is None:
            await websocket.close(code=1008)
            return

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
            await websocket.send_json({"type": "status", "phase": "connected",
                                       "detail": info.active_model})
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
                    # channel carrying status, steps, telemetry and toasts.
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
