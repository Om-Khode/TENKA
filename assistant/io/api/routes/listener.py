# assistant/io/api/routes/listener.py
"""Which listener answered, before anybody holds a credential.

Studio's `/connect` screen offers a bearer-token exchange that only `local`
can accept (`security.py`'s `credential_from`, gated on `policy.allow_bearer`),
and it has no way to know which listener served the page it is running on --
a phone connected over `tailnet` and a laptop connected over `local` both just
see "a page", with nothing in the DOM saying which port answered. `GET
/v1/listener` is that missing fact, and only that fact: no auth dependency at
all, because it has to be answerable *before* a credential exists, and no
capability, transport ceiling, device, or hostname -- the caller already knows
which port it reached (it chose it), so none of the three fields below is a
secret, and nothing else here would be safe to hand to a stranger before any
credential has been presented.

Layering: io/api -- core + config only.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from .pairing import pairing_denied_by_transport
from ..payloads import ListenerPayload
from ..schemas import Envelope
from ..security import UNAUTHORIZED, AuthState, anonymous_key, policy_for_scope

router = APIRouter()


@router.get("/listener")
async def get_listener(request: Request) -> Envelope[ListenerPayload]:
    """`{"policy": ..., "allowBearer": ..., "canPair": ...}`, nothing else.

    Rate-limited on the anonymous key, exactly like every other unauthenticated
    call this daemon answers (`authenticate()`'s own no-token branch) -- a
    caller with no credential yet still spends the same shared per-listener
    budget a wrong or missing token would, so this route cannot be turned into
    an unmetered probe.

    A port that resolves to no policy fails closed the same way the rest of
    the API does for that case (`authenticate()`, `pair_device`): 401, not a
    default listener guessed at. There is nothing to tell a caller on a socket
    nobody declared.
    """
    state: AuthState = request.app.state.auth
    source = anonymous_key(request.scope)
    if not state.limiter.check(source):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="too many requests")

    policy = policy_for_scope(request.scope, request.app.state.listener_policies)
    if policy is None:
        raise UNAUTHORIZED

    return Envelope(data=ListenerPayload(
        policy=policy.name,
        allow_bearer=policy.allow_bearer,
        can_pair=not pairing_denied_by_transport(policy),
    ))
