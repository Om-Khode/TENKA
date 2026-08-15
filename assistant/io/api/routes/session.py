# assistant/io/api/routes/session.py
"""Who is calling, and what this connection lets them carry.

Task 5 moved the device credential into an httpOnly cookie, so JavaScript can
no longer read a token out of `localStorage` to decide whether Studio's `/app`
tree should render its gate. This route is the replacement: `authenticate`
alone, no `require(...)` -- a device issued a single narrow grant still needs
to ask who it is, or it can never render its own gate at all.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..payloads import SessionPayload
from ..schemas import Envelope
from ..security import authenticate
from ..vault import Device

router = APIRouter()


@router.get("/session")
async def get_session(request: Request,
                      device: Device = Depends(authenticate)) -> Envelope[SessionPayload]:
    # `device.grants` here is already the effective set -- `authenticate()`
    # narrows before returning. The issued (pre-ceiling) set was stashed
    # separately on request.state by that same call, precisely so this route
    # can report both without a second vault read.
    issued = request.state.issued_grants
    policy = request.state.policy
    return Envelope(data=SessionPayload(
        device_id=device.device_id,
        label=device.label,
        grants=sorted(c.value for c in issued),
        effective=sorted(c.value for c in device.grants),
        policy=policy.name,
    ))
