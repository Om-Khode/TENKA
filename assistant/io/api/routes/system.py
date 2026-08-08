# assistant/io/api/routes/system.py
"""Telemetry, backup and enrollment.

The recovery phrase is verified here, on the assistant's side. It never ships
in a client bundle and never appears in a response body. A wrong or malformed
phrase is a 400 built from a static `detail` string, never the submitted
value. `RestoreRequest` bounds the field's length rather than matching it
against a pattern only so a *pattern* violation doesn't read differently from
a length violation to a caller -- the length bound alone does not, by itself,
keep the phrase out of a 422: Pydantic's `ValidationError.errors()` carries an
`"input"` key with the raw value regardless of which check failed, and
FastAPI's default `RequestValidationError` handler forwards it verbatim. What
actually closes that channel is the handler registered in `app.py`, which
rebuilds every 422 body from `loc`/`type` only and drops `input` app-wide --
this route does not, and could not on its own, guarantee the phrase stays out
of a validation error.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas import Envelope, RestoreRequest
from ..security import require, throttle
from ..vault import Capability

router = APIRouter()

# A Literal path parameter is validated by FastAPI/Pydantic before the route
# body ever runs, so an unknown kind ("fingerprint") is a 422 raised by the
# framework, not a 404 this handler would have to construct by hand.
EnrollmentKind = Literal["voice", "face"]

# run_backup writes to real cloud storage on every call. The shared limiter
# (120/60s, the same budget a status poll spends) would let a CHAT-only
# device -- a phone that has never held SYSTEM_CONTROL -- trigger dozens of
# real uploads a minute. A short, route-scoped budget on top of the shared
# one bounds that without touching what any other CHAT route may do.
_BACKUP_RUN_MAX_PER_WINDOW = 5
_BACKUP_RUN_WINDOW_SECONDS = 60.0


def telemetry_body(snapshot) -> dict:
    """The one serialisation of `TelemetrySnapshot` -- reused verbatim by the
    `"telemetry"` WebSocket frame in `../events.py` so the same numbers carry
    the same names on both transports instead of growing a second vocabulary.
    """
    return {
        "cpuPercent": snapshot.cpu_percent,
        "ramPercent": snapshot.ram_percent,
        "batteryPercent": snapshot.battery_percent,
        "activeModel": snapshot.active_model,
        "uptimeSeconds": snapshot.uptime_seconds,
    }


@router.get("/telemetry")
async def telemetry(request: Request,
                    _=Depends(require(Capability.CHAT))) -> Envelope:
    snapshot = await request.app.state.runtime.system.telemetry()
    return Envelope(data=telemetry_body(snapshot))


def _backup_body(state) -> dict:
    return {
        "enabled": state.enabled,
        "provider": state.provider,
        "lastBackupAt": state.last_backup_at,
        "lastResult": state.last_result,
        "sizeBytes": state.size_bytes,
    }


@router.get("/backup")
async def backup_state(request: Request,
                       _=Depends(require(Capability.CHAT))) -> Envelope:
    return Envelope(data=_backup_body(
        await request.app.state.runtime.system.backup_state()))


@router.post("/backup/run")
async def run_backup(request: Request,
                     _=Depends(throttle(Capability.CHAT, "backup_run",
                                        max_per_window=_BACKUP_RUN_MAX_PER_WINDOW,
                                        window_seconds=_BACKUP_RUN_WINDOW_SECONDS))
                     ) -> Envelope:
    return Envelope(data=_backup_body(
        await request.app.state.runtime.system.run_backup()))


@router.post("/backup/restore")
async def restore_backup(body: RestoreRequest, request: Request,
                         _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope:
    ok = await request.app.state.runtime.system.restore_backup(body.recovery_phrase)
    if not ok:
        # Deliberately generic: "restore failed" tells a caller the phrase was
        # wrong (or the backup unreadable) without repeating any part of what
        # was submitted -- HTTPException's default JSON body only ever holds
        # this literal string, never `body.recovery_phrase`.
        raise HTTPException(status_code=400, detail="restore failed")
    return Envelope(data={"restored": True})


def _enrolled_item(item) -> dict:
    return {
        "itemId": item.item_id,
        "name": item.name,
        "enrolledAt": item.enrolled_at,
        "count": item.count,
        "lastSeenAt": item.last_seen_at,
    }


@router.get("/enrollment")
async def enrollment(request: Request,
                     _=Depends(require(Capability.CHAT))) -> Envelope:
    state = await request.app.state.runtime.system.enrollment()
    return Envelope(data={
        "voices": [_enrolled_item(v) for v in state.voices],
        "faces": [_enrolled_item(f) for f in state.faces],
    })


@router.delete("/enrollment/{kind}/{item_id}")
async def forget_enrolled(kind: EnrollmentKind, item_id: str, request: Request,
                          # SYSTEM_CONTROL, not CHAT: destroying a biometric
                          # enrollment (a voiceprint, a face) is not ordinary
                          # conversational use -- the same reasoning that put
                          # forget-all and settings writes behind this grant.
                          # Reading the enrollment list stays on chat.
                          _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope:
    removed = await request.app.state.runtime.system.forget_enrolled(kind, item_id)
    if not removed:
        # `detail=` is dead on a 404: app.py's `@app.exception_handler(404)`
        # dispatches on status code alone and always answers a fixed
        # `{"error": "not found"}` body, discarding whatever is passed here.
        raise HTTPException(status_code=404)
    return Envelope(data={"forgotten": item_id, "kind": kind})


@router.get("/audit")
async def audit(request: Request,
                _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope:
    entries = request.app.state.auth.audit.entries()
    return Envelope(data={"entries": [
        {"at": e.at, "deviceId": e.device_id, "method": e.method,
         "path": e.path, "outcome": e.outcome}
        for e in reversed(entries)
    ]})
