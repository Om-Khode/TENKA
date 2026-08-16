# assistant/io/api/routes/session.py
"""Who is calling, what this connection lets them carry, and which channel
the credential rides on.

Task 5 moved the device credential into an httpOnly cookie, so JavaScript can
no longer read a token out of `localStorage` to decide whether Studio's `/app`
tree should render its gate. `GET /session` is the replacement: `authenticate`
alone, no `require(...)` -- a device issued a single narrow grant still needs
to ask who it is, or it can never render its own gate at all.

`POST /session/cookie` is the other half of that migration, and it exists
because the daemon has two credential channels that do not reach equally far.
HTTP accepts a cookie *or* `Authorization: Bearer` (the latter only where
`policy.allow_bearer`, which is loopback alone); the event socket accepts the
cookie and nothing else, because a browser's `WebSocket` constructor cannot
set a header and the query-string exception was deliberately removed in Task
5. So a browser session that authenticated with a bearer had working HTTP and
a socket that could never authenticate -- close 1008, reconnect, close again,
forever, with every chat reply stranded on the socket that never opened. This
route is the bridge: it takes the credential that just authenticated the
request and hands the same string back as the cookie, so the socket can use
it too.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response, status

from ..payloads import SessionPayload
from ..schemas import Envelope
from ..security import (
    UNAUTHORIZED,
    authenticate,
    cookie_kwargs,
    cookie_name_for,
    credential_from,
)
from ..vault import Device

logger = logging.getLogger(__name__)

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


@router.post("/session/cookie", status_code=status.HTTP_204_NO_CONTENT)
async def adopt_session_cookie(request: Request,
                               device: Device = Depends(authenticate)) -> Response:
    """Put the credential that authenticated *this* request into the cookie.

    **It moves a credential between channels. It does not mint one.** The
    value written to `Set-Cookie` is the exact string `credential_from()` just
    read off this request and `TokenVault.verify()` just accepted -- not a
    reissue, not a re-grant, not a second `Device` record. The vault is never
    written; `devices.json` has the same number of rows afterwards as before,
    the device id is the same id, and the grants are the same grants. There is
    no argument to widen with, because there is no argument at all: the body
    is empty and the route's entire input is the credential it already
    verified. A caller that could reach this route could already do everything
    the resulting cookie can do -- over HTTP, this same second.

    **A POST, never a GET.** It sets a cookie, which is session state, and a
    GET that changes state is reachable by a plain navigation, an `<img>`, a
    link prefetch and a browser's own speculative fetch -- none of which the
    page asked for. It would also sit outside `_AMBIENT_METHODS`, the set that
    decides when the CSRF header is demanded, so the one gate that covers a
    request no script built would not apply to it. There is no body and no
    payload: the reply is a 204, exactly like `POST /v1/pair`, and for the
    same reason -- returning the token in JSON as well would put a working
    credential into a response body a client could log or cache, which is the
    thing this whole migration exists to stop.

    **Only where `policy.allow_bearer`.** That is loopback today, and the
    coupling is not incidental: `allow_bearer` is precisely "this listener may
    accept a credential on a channel other than the cookie", which is the only
    situation in which the two channels can disagree and therefore the only
    situation this route has anything to fix. A remote listener reads the
    cookie and nothing else (`credential_from`), so a caller there already
    holds the cookie and the exchange is meaningless -- and offering it anyway
    would mean a route that writes `Set-Cookie` reachable from a tunnel, which
    is a strictly larger surface for no gain. Refused with the same
    `UNAUTHORIZED` every other auth failure in this API raises: constant
    shape, and it must not be `_NOT_ADMIN`'s "capability not granted", which
    would be a lie -- this route is gated on no capability whatsoever.

    **CSRF applies exactly as it does everywhere else, and nothing special is
    needed here.** `authenticate()` runs `_refuse_cross_site()` first, which
    demands the CSRF header on a write *when the credential was the cookie*.
    The interesting case is the other one: a bearer-authenticated call carries
    no cookie, so the header is not demanded -- and that is correct rather
    than a gap. CSRF exists because a browser attaches an *ambient* credential
    to whatever request any page makes; an `Authorization` header is never
    ambient, so a cross-site page cannot produce one at all and there is no
    forgeable request to defend against. Bearer is loopback-only besides. The
    cookie-authenticated case is the one that could be forged, and it is
    covered by the ordinary gate -- where all it could achieve is re-setting
    the cookie the victim's browser already had, byte for byte, which is why
    that case is left as a harmless idempotent no-op rather than refused.

    **No capability requirement**, for the same reason `GET /session` has
    none. A device issued a single narrow grant still has to be able to reach
    its own event socket; gating this on, say, SYSTEM_CONTROL would mean a
    CHAT_SEND-only session could authenticate over HTTP forever and never
    open the socket its replies arrive on.
    """
    policy = request.state.policy
    if not policy.allow_bearer:
        raise UNAUTHORIZED

    # Re-read rather than reconstructed: `authenticate()` returns the `Device`
    # and deliberately never the token, and inventing a second way to get the
    # credential string is how the two spellings drift. This is the same
    # function `authenticate()` itself used moments ago on the same request,
    # so it returns the same string -- the one the vault already accepted.
    token = credential_from(request, policy)
    if not token:
        # Unreachable: `authenticate()` above verified a credential off this
        # very request, so there is one to read. Guarded rather than trusted,
        # because the alternative to a refusal here is writing `None` into a
        # `Set-Cookie` header and handing back a session that is authenticated
        # by an empty string.
        raise UNAUTHORIZED

    # The device id, never the token and never the label. One line saying a
    # credential changed channels is worth having in the log; its value is not.
    logger.info(f"[API] moved a credential onto the cookie for {device.device_id}")

    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    # `cookie_name_for(policy)`, not a fixed name: a listener that can set
    # `Secure` gets the `__Host-` prefixed name, which a browser refuses to
    # store unless it is host-only. See `HOST_COOKIE_NAME` in security.py --
    # without it, a sibling host under `*.ts.net` can plant this cookie inward.
    response.set_cookie(cookie_name_for(policy), token, **cookie_kwargs(policy))
    return response
