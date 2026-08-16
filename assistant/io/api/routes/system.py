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

from ..payloads import (
    AuditEntryPayload, AuditPayload, BackupStatePayload, EnrolledItemPayload,
    EnrollmentPayload, ForgetEnrolledPayload, RestorePayload, TelemetryPayload,
    UnlockPayload,
)
from ..schemas import Envelope, RestoreRequest, UnlockRequest
from ..security import require, require_admin, throttle
from ..vault import Capability

router = APIRouter()

# A Literal path parameter is validated by FastAPI/Pydantic before the route
# body ever runs, so an unknown kind ("fingerprint") is a 422 raised by the
# framework, not a 404 this handler would have to construct by hand.
EnrollmentKind = Literal["voice", "face"]

# run_backup writes to real cloud storage on every call. The budget below is
# a second bound on top of the SYSTEM_CONTROL grant it now demands: the
# shared limiter (120/60s, the same budget a status poll spends) would let one
# authorised device trigger dozens of real uploads a minute, which is a cost
# and a quota question rather than a permission one. Both bounds are wanted.
_BACKUP_RUN_MAX_PER_WINDOW = 5
_BACKUP_RUN_WINDOW_SECONDS = 60.0

# Tighter than the run budget above. /backup/unlock takes a recovery phrase and
# derives a key from it, which is the one route where an attacker who already
# holds a SYSTEM_CONTROL token could grind candidate phrases -- and unlike
# restore, a wrong-but-well-formed guess has no destructive side effect to make
# attempts self-limiting. Five a minute is far above any honest use (a person
# types this once per restart) and far below anything useful for guessing.
_UNLOCK_MAX_PER_WINDOW = 5
_UNLOCK_WINDOW_SECONDS = 60.0


def telemetry_body(snapshot) -> TelemetryPayload:
    """The one serialisation of `TelemetrySnapshot` -- reused verbatim by the
    `"telemetry"` WebSocket frame in `../events.py` so the same numbers carry
    the same names on both transports instead of growing a second vocabulary.
    """
    return TelemetryPayload(
        cpu_percent=snapshot.cpu_percent,
        ram_percent=snapshot.ram_percent,
        battery_percent=snapshot.battery_percent,
        active_model=snapshot.active_model,
        uptime_seconds=snapshot.uptime_seconds,
    )


@router.get("/telemetry")
async def telemetry(request: Request,
                    _=Depends(require(Capability.OBSERVE))) -> Envelope[TelemetryPayload]:
    snapshot = await request.app.state.runtime.system.telemetry()
    return Envelope(data=telemetry_body(snapshot))


def _backup_body(state) -> BackupStatePayload:
    return BackupStatePayload(
        enabled=state.enabled,
        provider=state.provider,
        last_backup_at=state.last_backup_at,
        last_result=state.last_result,
        size_bytes=state.size_bytes,
        unlocked=state.unlocked,
    )


# OBSERVE: this reports *whether* backups are running -- enabled, provider,
# last result, size, whether the key is armed -- and nothing about what is in
# one. That is her operational state, the same class of fact as telemetry.
@router.get("/backup")
async def backup_state(request: Request,
                       _=Depends(require(Capability.OBSERVE))) -> Envelope[BackupStatePayload]:
    return Envelope(data=_backup_body(
        await request.app.state.runtime.system.backup_state()))


@router.post("/backup/run")
async def run_backup(request: Request,
                     # SYSTEM_CONTROL, not a read grant. This was the worst of the
                     # four routes a read-only listener could still reach:
                     # it encrypts the whole memory database and pushes it to
                     # a third-party cloud provider, spending the user's
                     # storage quota and putting every fact she owns onto
                     # someone else's disk. Reading /backup's *state* is a
                     # read and stays on OBSERVE; causing an upload is the
                     # single most consequential act in this module, and its
                     # two siblings (/backup/restore, /backup/unlock) already
                     # demanded SYSTEM_CONTROL -- it was the odd one out, not
                     # the precedent.
                     _=Depends(throttle(Capability.SYSTEM_CONTROL, "backup_run",
                                        max_per_window=_BACKUP_RUN_MAX_PER_WINDOW,
                                        window_seconds=_BACKUP_RUN_WINDOW_SECONDS))
                     ) -> Envelope[BackupStatePayload]:
    return Envelope(data=_backup_body(
        await request.app.state.runtime.system.run_backup()))


@router.post("/backup/restore")
async def restore_backup(body: RestoreRequest, request: Request,
                         _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope[RestorePayload]:
    ok = await request.app.state.runtime.system.restore_backup(body.recovery_phrase)
    if not ok:
        # Deliberately generic: "restore failed" tells a caller the phrase was
        # wrong (or the backup unreadable) without repeating any part of what
        # was submitted -- HTTPException's default JSON body only ever holds
        # this literal string, never `body.recovery_phrase`.
        raise HTTPException(status_code=400, detail="restore failed")
    return Envelope(data=RestorePayload(restored=True))


@router.post("/backup/unlock")
async def unlock_backup(body: UnlockRequest, request: Request,
                        _=Depends(throttle(Capability.SYSTEM_CONTROL, "backup_unlock",
                                           max_per_window=_UNLOCK_MAX_PER_WINDOW,
                                           window_seconds=_UNLOCK_WINDOW_SECONDS))
                        ) -> Envelope[UnlockPayload]:
    """Arm this process's backup encryption key from the recovery phrase.

    SYSTEM_CONTROL, not a read grant: the phrase this accepts is the same
    secret that can overwrite every memory she has through /backup/restore, so
    it must not be reachable by a device paired only to watch or to read.

    Throttled harder than the other write routes. Deriving a key from a
    submitted phrase is the one place an attacker with a foothold could grind
    candidate phrases, and unlike restore there is no destructive side effect
    to make attempts self-limiting -- a wrong-but-well-formed phrase simply
    arms a useless key and can be retried. The rate limit is the only thing
    bounding that.
    """
    ok = await request.app.state.runtime.system.unlock_backup(body.recovery_phrase)
    if not ok:
        # Generic, and never echoes any part of what was submitted -- the same
        # reasoning as /backup/restore's 400 above.
        raise HTTPException(status_code=400, detail="unlock failed")
    return Envelope(data=UnlockPayload(unlocked=True))


def _enrolled_item(item) -> EnrolledItemPayload:
    return EnrolledItemPayload(
        item_id=item.item_id,
        name=item.name,
        enrolled_at=item.enrolled_at,
        count=item.count,
        last_seen_at=item.last_seen_at,
    )


# RECALL, not OBSERVE, and this is the one assignment in this module that is
# not obvious. It looks like configuration -- which sensors are set up -- but
# every row names a *person*: a voiceprint label, a face's name, and a
# `last_seen_at` saying when that named person was last in front of this
# machine. That is stored data about the people around her, closer to a
# preference record than to a telemetry meter, and it must not ride the one
# transport a third party reads.
@router.get("/enrollment")
async def enrollment(request: Request,
                     _=Depends(require(Capability.RECALL))) -> Envelope[EnrollmentPayload]:
    state = await request.app.state.runtime.system.enrollment()
    return Envelope(data=EnrollmentPayload(
        voices=[_enrolled_item(v) for v in state.voices],
        faces=[_enrolled_item(f) for f in state.faces],
    ))


@router.delete("/enrollment/{kind}/{item_id}")
async def forget_enrolled(kind: EnrollmentKind, item_id: str, request: Request,
                          # SYSTEM_CONTROL, not a read grant: destroying a
                          # biometric enrollment (a voiceprint, a face) is not
                          # ordinary conversational use -- the same reasoning
                          # that put forget-all and settings writes behind
                          # this grant. Reading the list stays on RECALL.
                          _=Depends(require(Capability.SYSTEM_CONTROL))) -> Envelope[ForgetEnrolledPayload]:
    removed = await request.app.state.runtime.system.forget_enrolled(kind, item_id)
    if not removed:
        # `detail=` is dead on a 404: app.py's `@app.exception_handler(404)`
        # dispatches on status code alone and always answers a fixed
        # `{"error": "not found"}` body, discarding whatever is passed here.
        raise HTTPException(status_code=404)
    return Envelope(data=ForgetEnrolledPayload(forgotten=item_id, kind=kind))


@router.get("/audit")
async def audit(request: Request,
                _=Depends(require_admin(Capability.SYSTEM_CONTROL))) -> Envelope[AuditPayload]:
    # `require_admin`, not `require`: the audit log is a strictly richer
    # answer than `GET /v1/devices`, which is admin-only. It carries every
    # device id that has made a request since process start, plus the method,
    # path and outcome for each -- so it says not only how many credentials
    # exist but what each one does, which the device list itself does not
    # even carry. `routes/devices.py`'s own docstring already names this
    # route as the same class of thing; the gate had not caught up.
    #
    # This is the second, independent control. The capability ceilings keep a
    # non-loopback listener away from SYSTEM_CONTROL at all; the admin gate is
    # what holds if a ceiling is ever widened by mistake.
    #
    # Nothing here may move into the route's docstring: docstrings are
    # published as the OpenAPI `description` and `ui.contract_hash()`
    # fingerprints the schema, so editing one takes the vendored Studio
    # bundle dark with a stale-contract 503.
    entries = request.app.state.auth.audit.entries()
    return Envelope(data=AuditPayload(entries=[
        AuditEntryPayload(at=e.at, device_id=e.device_id, method=e.method,
                          path=e.path, outcome=e.outcome)
        for e in reversed(entries)
    ]))
