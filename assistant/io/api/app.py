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

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from ...core.redact import redact_secrets
from .routes import status as status_routes
from .runtime import StudioRuntime
from .security import AuditEntry, AuthState
from .vault import TokenVault

logger = logging.getLogger(__name__)


def _mount(app: FastAPI, router: APIRouter, *, prefix: str) -> None:
    """Register a route module's routes directly on `app`, flat.

    `FastAPI.include_router` no longer appends plain routes to `app.routes`;
    it wraps the whole router in an internal node so nested routers can share
    one prefix/dependency context. That is fine for dispatch, but it means
    `app.routes` is no longer a flat, introspectable list of real routes --
    which breaks anything that walks it looking for every registered path,
    including this daemon's own route-sweep test (`test_api_auth.py`,
    `test_every_registered_route_rejects_an_anonymous_call`), and that sweep
    is exactly what keeps a future route from shipping unauthenticated.
    Registering each route directly with the prefix baked into its path
    keeps `app.routes` flat while dispatch behaves identically.
    """
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        app.add_api_route(
            prefix + route.path,
            route.endpoint,
            methods=list(route.methods),
            name=route.name,
            response_model=route.response_model,
            status_code=route.status_code,
            tags=route.tags,
            dependencies=route.dependencies,
            summary=route.summary,
            description=route.description,
            deprecated=route.deprecated,
            include_in_schema=route.include_in_schema,
        )


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

    _mount(app, status_routes.router, prefix="/v1")
    return app
