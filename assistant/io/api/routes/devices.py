# assistant/io/api/routes/devices.py
"""The revoke list: which credentials reach this machine, and killing one.

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

Layering: io/api -- core + config only.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..payloads import DevicePayload, DevicesPayload, RevokedPayload
from ..schemas import Envelope
from ..security import require_admin
from ..vault import Capability, Device, VaultUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()


def _device_body(device: Device) -> DevicePayload:
    return DevicePayload(
        device_id=device.device_id,
        label=device.label,
        grants=sorted(c.value for c in device.grants),
        created_at=device.created_at,
        last_seen_at=device.last_seen_at,
    )


@router.get("/devices")
async def list_devices(
    request: Request,
    _: Device = Depends(require_admin(Capability.SYSTEM_CONTROL)),
) -> Envelope[DevicesPayload]:
    """Every device that holds a credential, as `vault.devices()` reports them.

    Nothing is added on top -- not which listener a device was last seen on,
    not a source address, not whether it is the caller's own row. A caller
    that needs to know which device it is asks `GET /v1/session`; inventing a
    second answer here would be a second thing that could disagree.
    """
    devices = request.app.state.auth.vault.devices()
    return Envelope(data=DevicesPayload(
        devices=[_device_body(d) for d in devices]))


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
    try:
        revoked = request.app.state.auth.vault.revoke(device_id)
    except VaultUnavailableError as exc:
        logger.warning(f"[API] revoke failed: vault unavailable ({exc})")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="try again")
    if not revoked:
        # `detail=` is dead on a 404: app.py's own 404 handler dispatches on
        # the status code alone and always answers a fixed body.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    # The device id, never a label and never a token. An id is what the caller
    # already sent; it is the one field here that reveals nothing new.
    logger.info(f"[API] device revoked (device={device_id})")
    return Envelope(data=RevokedPayload(revoked=device_id))
