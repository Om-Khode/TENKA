# assistant/io/api/app.py
"""The Studio daemon's FastAPI application.

Nothing here knows how to reach the assistant. `runtime` is injected by
main.py; `vault` decides who may ask.

Layering: io/api — core + config only.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ...core.redact import redact_secrets
from .routes import chat as chat_routes
from .routes import commands as command_routes
from .routes import files as file_routes
from .routes import memory as memory_routes
from .routes import settings as settings_routes
from .routes import status as status_routes
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
               origins: list[str]) -> FastAPI:
    # Eager, once: instance_secret() is uncached on the environment-override
    # path, and a wrong-length TENKA_SECRET raises ValueError. Resolving it
    # here means a misconfigured override fails when the app is built, not
    # as a 500 on the first authenticated request. The ValueError's own
    # message names TENKA_SECRET, so the operator knows what to fix.
    vault.instance_secret()

    app = FastAPI(
        title="TENKA Studio daemon",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.runtime = runtime
    app.state.auth = AuthState(vault=vault)
    app.state.started_at = time.monotonic()

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

    app.include_router(status_routes.router, prefix="/v1")
    app.include_router(memory_routes.router, prefix="/v1")
    app.include_router(settings_routes.router, prefix="/v1")
    app.include_router(file_routes.router, prefix="/v1")
    app.include_router(command_routes.router, prefix="/v1")
    app.include_router(chat_routes.router, prefix="/v1")
    return app
