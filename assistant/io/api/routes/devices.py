# assistant/io/api/routes/devices.py
"""The revoke list: which credentials reach this machine, killing one, and
lifting one for a while.

Loopback only, like minting a pair code and for the mirror-image reason. A
remote session that could enumerate the device list would learn what else to
attack; one that could revoke would be able to lock the owner out of her own
machine from the outside. Both gates live in `require_admin`, which also
demands `SYSTEM_CONTROL` -- this is the security configuration, the same class
of thing as `/v1/audit`, not something a device paired to watch her work
should read.

Revocation is immediate, and that is a property of the vault rather than of
this route: `TokenVault.revoke` deletes the record, and `verify()` re-reads
`devices.json` on every call, so a token stops working on the next request
with no cache to wait out. It is the one control that has to work under
duress, so it must not depend on anything expiring.

The two raise routes (Milestone 6b, spec §3.4) sit behind the same gate, and
that is the whole answer to "who may authorise a raise": `policy.admin` is the
loopback listener alone, so **a raise is minted at the keyboard, always**. It
is unreachable from any tunnel, which means there is no remote surface to
attack and no confirmation flow for a remote device to hijack -- the latter
would have collided head-on with KI-13, where pending state has no owner.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..payloads import DevicePayload, DevicesPayload, RaisePayload, RevokedPayload
from ..policy import POLICIES
from ..raises import RaiseGrant, RaiseStore
from ..schemas import Envelope, RaiseRequest
from ..security import require_admin
from ..vault import Capability, Device, VaultUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()


def _raise_store(request: Request) -> RaiseStore | None:
    """The live raise store, or `None` when nothing attached one.

    `app.state.raises` is populated by `create_app` in a sibling task. Read
    defensively, the same shape `routes/session.py` already uses: an app built
    before that wiring lands has no attribute at all, and "no raises" is the
    correct fail-closed answer either way.
    """
    return getattr(request.app.state, "raises", None)


def _raise_body(device_id: str, policy_name: str, grant: RaiseGrant) -> RaisePayload:
    # Converted from the store's own monotonic reading, exactly as
    # `mint_pair_code` converts a `PairCode`'s -- never recomputed as
    # `now + duration`, which would drift the moment the store's cap changed
    # and would put that cap in a second place.
    return RaisePayload(
        device_id=device_id,
        transport=policy_name,
        capabilities=sorted(c.value for c in grant.capabilities),
        expires_in_seconds=round(max(0.0, grant.expires_at - time.monotonic())),
        reason=grant.reason,
    )


def _device_body(device: Device, raises: list[RaisePayload]) -> DevicePayload:
    return DevicePayload(
        device_id=device.device_id,
        label=device.label,
        grants=sorted(c.value for c in device.grants),
        created_at=device.created_at,
        last_seen_at=device.last_seen_at,
        raises=raises,
    )


@router.get("/devices")
async def list_devices(
    request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[DevicesPayload]:
    """Every device that holds a credential, as `vault.devices()` reports them,
    each with whatever ceiling raises are live for it right now.

    Nothing else is added on top -- not which listener a device was last seen
    on, not a source address, not whether it is the caller's own row. A caller
    that needs to know which device it is asks `GET /v1/session`; inventing a
    second answer here would be a second thing that could disagree.
    """
    devices = request.app.state.auth.vault.devices()
    # One snapshot for the whole listing, not one lookup per row: `active()`
    # takes the store's lock, and it also *drops* every expired record it walks
    # past, so calling it once per device would take the lock N times to do the
    # same sweep. Grouped by device id here rather than by the store, which is
    # keyed on the pair and has no reason to learn this route's shape.
    store = _raise_store(request)
    by_device: dict[str, list[RaisePayload]] = {}
    if store is not None:
        for (device_id, policy_name), grant in store.active().items():
            by_device.setdefault(device_id, []).append(
                _raise_body(device_id, policy_name, grant))
    return Envelope(data=DevicesPayload(
        devices=[_device_body(d, sorted(by_device.get(d.device_id, []),
                                        key=lambda r: r.transport))
                 for d in devices]))


@router.delete("/devices/{device_id}")
async def revoke_device(
    device_id: str, request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[RevokedPayload]:
    """Kill one credential. A device that is not there is a 404.

    Not a silent 200: this is the control somebody reaches for when they think
    a device is compromised, and "nothing was revoked" has to be
    distinguishable from "it is gone now". The distinction discloses nothing a
    caller holding SYSTEM_CONTROL on the loopback listener could not read
    straight off `GET /v1/devices` a line earlier.

    A third outcome, 503, is what `TokenVault.revoke()` raising
    `VaultUnavailableError` becomes -- either half of it. If the read fails,
    devices.json could not be read at all, so whether `device_id` exists is
    genuinely unknown; answering 404 there would tell the one person trying
    to cut a device off that it is already gone when the truth is "ask again
    once the file is readable" -- exactly the control that must not lie
    under duress. If the *write* fails instead (the same Windows lock
    contention, on the save half), the underlying `PermissionError` would
    otherwise reach `errors.py`'s app-wide mapping and answer 403 "protected
    path" -- which here would read as "you are not allowed to revoke this
    device", the same lie the 404 case argues against, just from the other
    direction. Both halves become 503: unavailable, not unauthorised and not
    already gone.
    """
    # Off the event loop, like every other vault *write* reachable from a
    # request: `revoke()` rewrites `devices.json` and then spawns `icacls` to
    # re-apply the ACL, synchronously, for tens of milliseconds. This daemon
    # shares its loop with the assistant, so a write left inline stalls her
    # too, not just this route. `TokenVault`'s own `threading.Lock` covers the
    # whole load-filter-save sequence, which is what makes a worker thread the
    # right place for it.
    #
    # A comment rather than another docstring paragraph, deliberately: FastAPI
    # publishes a route handler's docstring as the OpenAPI `description`, and
    # `ui.contract_hash()` fingerprints the whole schema -- so a sentence added
    # above this line takes the vendored Studio bundle dark with a stale-
    # contract 503. Prose about *how* a route is implemented has no business in
    # the API's published description anyway.
    try:
        revoked = await asyncio.to_thread(
            request.app.state.auth.vault.revoke, device_id)
    except VaultUnavailableError as exc:
        logger.warning(f"[API] revoke failed: vault unavailable ({exc})")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="try again")
    if not revoked:
        # `detail=` is dead on a 404: app.py's own 404 handler dispatches on
        # the status code alone and always answers a fixed body.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    # Spec §3.3: a raise outliving the device it was granted to would be
    # absurd. After the vault write, never before -- a revoke that failed
    # must leave the device exactly as it was, raises included.
    store = _raise_store(request)
    if store is not None:
        store.drop_device(device_id)
    # The device id, never a label and never a token. An id is what the caller
    # already sent; it is the one field here that reveals nothing new.
    logger.info(f"[API] device revoked (device={device_id})")
    return Envelope(data=RevokedPayload(revoked=device_id))


# ─── raising a ceiling, for a while ──────────────────────────────────────
def _known_device(request: Request, device_id: str) -> Device:
    """The device this raise is *about*, or a 404.

    Never the authenticated caller: the person at the keyboard raises some
    *other* device's ceiling (a phone on the tailnet), and the two coinciding
    is the special case, not the rule. 404 for the same reason `revoke_device`
    answers one -- "nothing happened" has to be distinguishable from "it is
    there, and this is what it now holds" -- and it discloses nothing a caller
    holding SYSTEM_CONTROL on the loopback listener could not read straight off
    `GET /v1/devices` a line earlier.
    """
    for device in request.app.state.auth.vault.devices():
        if device.device_id == device_id:
            return device
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@router.post("/devices/{device_id}/raise")
async def raise_device_ceiling(
    device_id: str, body: RaiseRequest, request: Request,
    granter: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[RaisePayload]:
    """Let one device do more, on one transport, for a bounded while.

    Three bounds stack and only the innermost one moves. What the device was
    issued is never widened -- a raise lifts a transport's refusal, it cannot
    manufacture a grant. `ceiling | raisable` is static module data vetted once
    by a human, and `raisable` is empty on every listener but `tailnet`, which
    is what makes the public tunnels unraisable by construction rather than by
    a check somebody could forget. Only the live record this route mints is
    new, and it expires on a clock the caller does not control.

    Refused with distinct statuses, because each one is a different thing for
    the operator to fix: 404 when the device does not exist, 422 when a
    capability name is not one, 403 when the transport was never vetted to
    carry it or the device was never issued it, and 409 when the transport is
    not currently running -- a raise scoped to a listener that is down can
    never be exercised, and must not be left behind for a future listener that
    reuses the name.

    An over-long request is clamped to the seven-day cap rather than refused.
    The cap is the safety property, not a promise the caller kept its word.
    """
    # `_known_device` reads the vault, which re-reads and re-parses
    # devices.json -- off the event loop, like every other vault read reachable
    # from a request, because this daemon shares its loop with the assistant.
    #
    # A comment rather than another docstring paragraph, for the reason spelled
    # out at length in `revoke_device` above: FastAPI publishes a handler's
    # docstring as the OpenAPI `description`, and `ui.contract_hash()`
    # fingerprints the whole schema, so implementation prose added there takes
    # the vendored Studio bundle dark with a stale-contract 503.
    target = await asyncio.to_thread(_known_device, request, device_id)

    try:
        requested = frozenset(Capability(name) for name in body.capabilities)
    except ValueError:
        # Never echoes the submitted name -- `mint_pair_code`'s shape and its
        # reasoning: app.py's validation handler strips Pydantic's
        # `input`/`msg` from every 422 body app-wide, and a 422 built by hand
        # has to keep that promise itself. The literal, not
        # `status.HTTP_422_*`: Starlette renamed that constant and deprecated
        # the old spelling, and this is not a number that moves.
        raise HTTPException(status_code=422, detail="unknown capability")

    policy = POLICIES.get(body.transport)
    if policy is None:
        # Same shape, same silence about the submitted string. A transport
        # nobody declared is a validation problem, not a 409: 409 says "not
        # running just now, try again", which would be a lie about a name that
        # can never run.
        raise HTTPException(status_code=422, detail="unknown transport")

    if requested - policy.raisable:
        # `raisable`, never `ceiling | raisable`: asking to raise something the
        # transport already carries is not a raise, and answering 200 to it
        # would mint a record that widens nothing while looking like it did.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="this transport may not carry that")
    if requested - target.grants:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="this device was never issued that")

    # `getattr`, because a sibling task wires the transport manager and an app
    # built without one must fail closed here. Membership rather than a
    # coerced set, so a manager whose `running()` hands back a mapping, a set
    # or a list of names all read the same way.
    manager = getattr(request.app.state, "transports", None)
    running = manager.running() if manager is not None else ()
    if body.transport not in running:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="that transport is not running")

    store = _raise_store(request)
    if store is None:
        # Nothing to write the record into, so nothing was raised. 503 rather
        # than a 200 describing a raise that does not exist: the caller has to
        # be able to tell that retrying might work.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="try again")

    # Keyword arguments throughout, deliberately. `granted_by` and `reason` are
    # adjacent plain strings on `RaiseStore.grant()`, so a positional
    # transposition here would silently write the wrong audit trail and nothing
    # would ever fail. This is the only call site.
    grant = store.grant(
        device_id=device_id,
        policy_name=body.transport,
        capabilities=requested,
        seconds=body.minutes * 60,
        granted_by=granter.device_id,
        reason=body.reason,
    )

    payload = _raise_body(device_id, body.transport, grant)
    # Spec §3.6: a log line on mint. Ids, the transport, the capabilities and
    # how long -- never a token, and never the reason, which is free text the
    # operator typed and has no business in a log file that outlives the raise.
    logger.info(
        f"[API] ceiling raised (device={device_id} transport={body.transport} "
        f"capabilities={'+'.join(payload.capabilities)} "
        f"seconds={payload.expires_in_seconds} by={granter.device_id})")
    return Envelope(data=payload)


@router.delete("/devices/{device_id}/raise")
async def revoke_device_raise(
    device_id: str, request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[RevokedPayload]:
    """Drop every live raise a device holds. A device with none is a 404.

    Every raise, on every transport, rather than one named in the request.
    This is the undo for a control that deliberately widens what a device may
    do, so it must not be possible to half-revoke it by naming the wrong
    transport -- the same reasoning that keeps `revoke_device` a single
    unconditional kill rather than a per-listener one.

    404, not a silent 200, and the argument is `revoke_device`'s: somebody
    reaching for this has decided a raise should stop, and "there was nothing
    to stop" has to be distinguishable from "it is gone now".
    """
    store = _raise_store(request)
    # The device check first, so an unknown id answers 404 for the reason it
    # is unknown rather than for having no raises -- the two are the same
    # status here, but the vault read is what makes the answer honest, and it
    # is the same ordering `raise_device_ceiling` uses.
    await asyncio.to_thread(_known_device, request, device_id)
    held = [key for key in (store.active() if store is not None else {})
            if key[0] == device_id]
    if not held:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    store.drop_device(device_id)
    logger.info(f"[API] ceiling raise revoked (device={device_id} "
                f"transports={'+'.join(sorted(key[1] for key in held))})")
    return Envelope(data=RevokedPayload(revoked=device_id))
